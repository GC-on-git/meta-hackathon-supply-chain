import argparse
import json
import os
import sys
import textwrap
import time
import traceback
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from service.grading import grade_episode
from service.hackathon_environment import SupplyChainEnv
from service.models import AgentAction
from service.tasks import TaskSpec, task_specs

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — all sourced from environment variables per hackathon spec
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

MAX_STEPS: int = 30
DEFAULT_TEMPERATURE: float = 0.2
MAX_TOKENS: int = 1024
LLM_TIMEOUT: float = 60.0
LLM_MAX_RETRIES: int = 2

TASK_SEEDS: dict[str, int] = {
    "easy":   int(os.getenv("TASK_SEED_EASY",   "1001")),
    "medium": int(os.getenv("TASK_SEED_MEDIUM", "2002")),
    "hard":   int(os.getenv("TASK_SEED_HARD",   "3003")),
}

# ---------------------------------------------------------------------------
# Node / topology constants (mirrors hackathon_environment.py for prompt)
# ---------------------------------------------------------------------------
ECHELON_NAMES = [
    "factory",
    "warehouse_a", "warehouse_b",
    "retailer_1", "retailer_2", "retailer_3", "retailer_4",
]
FALLBACK_LEAD_TIMES = [4, 3, 3, 2, 2, 2, 2]

# ---------------------------------------------------------------------------
# System prompt — accurate to env mechanics
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert supply chain planner for a 7-node, 3-product network.

TOPOLOGY (1 Factory -> 2 Warehouses -> 4 Retailers):
  Slot layout: index i = node*3 + product  (node 0-6, product 0-2 = 21 slots)
  Nodes:  0=factory  1=warehouse_a  2=warehouse_b
          3=retailer_1  4=retailer_2  5=retailer_3  6=retailer_4
  Upstream: retailers 3,4 <- warehouse_a(1); retailers 5,6 <- warehouse_b(2);
            warehouses 1,2 <- factory(0)

DAILY ORDER OF OPERATIONS (each step):
  1. News events tick / spawn (Canal Blockage, Labor Strike, Social Media Trend).
  2. In-transit shipments whose countdown reached 0 arrive into node inventory.
  3. Existing customer backlogs at retailers are served from on-hand stock.
  4. YOUR ACTION is applied: orders placed into method-specific backlogs.
  5. A link disruption may be sampled (hard difficulty only).
  6. Dispatch: express orders processed first, then standard; upstream inventory
     is consumed, shipments enter the pipeline with stochastic lead times.
  7. Customer demand is sampled for each (product x retailer) stream.
  8. Retailers serve new demand from remaining inventory; unmet -> backlog.
  9. Fuel price multiplier does a bounded random walk [0.8, 1.5].
  10. Step reward is computed and clipped to [-1, 1].

SHIPPING METHODS:
  0 = standard: full base lead time, 1x transport cost, 1x carbon per unit.
  1 = express:  half base lead time (min 1 day), 2x variable transport cost,
                5x carbon per unit.  Use sparingly — it raises both cost and CO2.

EVENTS (when active):
  - Canal Blockage (14 days): effective lead times double.
  - Labor Strike (7 days): warehouse_a (node 1) cannot dispatch; orders to
    retailers 3,4 are blocked.  Route demand via warehouse_b if possible.
  - Social Media Trend (5 days): demand for product 0 is 3x normal.

PARTIAL VISIBILITY (hard): factory inventory shown as -1 (unknown).
  Infer factory stock indirectly from your prior orders and lead times.

SCORING:
  Episode score (0-1) combines weighted fill-rate, cost efficiency, and
  carbon footprint (hard only).  Step reward is a shaped proxy — not
  identical to the final score.  Maximise long-horizon fill rate while
  controlling cost and carbon.

STRATEGY GUIDELINES:
  1. Order enough to cover forecast demand x lead_time at each node, plus a
     small safety buffer.
  2. If inventory + in_transit already exceeds ~2x daily forecast x lead_time,
     order 0 to avoid holding cost.
  3. Factory (node 0) draws from infinite external supply — always keep it
     stocked to feed warehouses.
  4. Prefer standard shipping by default.  Use express (1) only for retailers
     (nodes 3-6) with positive customer backlog AND critically low inventory.
  5. React to events: order more during Canal Blockage, reroute during Strike,
     boost product 0 orders during Social Media Trend.

OUTPUT: Reply with ONLY a valid JSON object — no markdown, no explanation:
{"order_quantities": [<21 floats>], "shipping_methods": [<21 ints, 0 or 1>]}\
""")


def build_system_prompt(spec: TaskSpec) -> str:
    return SYSTEM_PROMPT + f"\n\nTASK OBJECTIVE:\n{spec.objective}"


# ---------------------------------------------------------------------------
# Protocol helpers (stdout = grader, stderr = diagnostics)
# ---------------------------------------------------------------------------
def _protocol_line(line: str) -> None:
    print(line, flush=True)


def _fmt_reward(x: float) -> str:
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _fmt_score(x: float) -> str:
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _diag(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Heuristic fallback action (used when LLM fails after retries)
# ---------------------------------------------------------------------------
def heuristic_action(obs: Any) -> AgentAction:
    """Base-stock heuristic using observation-backed lead times."""
    inv = list(obs.inventory_levels)
    in_tr = list(obs.in_transit_qty)
    forecast = list(obs.demand_forecast)
    bounds = list(obs.action_bounds)
    backlog = list(obs.customer_backlog)
    active_events = list(getattr(obs, "active_events", []))

    lead_times = list(getattr(obs, "node_base_lead_times", FALLBACK_LEAD_TIMES))
    if len(lead_times) < 7:
        lead_times = FALLBACK_LEAD_TIMES

    canal_active = "Canal Blockage" in active_events
    trend_active = "Social Media Trend" in active_events
    strike_active = "Labor Strike" in active_events

    quantities: list[float] = []
    methods: list[int] = []

    for node in range(7):
        lead_multiplier = 2.0 if canal_active else 1.0
        lt = lead_times[node] * lead_multiplier

        for prod in range(3):
            idx = node * 3 + prod
            current_inv = inv[idx] if idx < len(inv) else 0.0
            current_it = in_tr[idx] if idx < len(in_tr) else 0.0
            fc = forecast[idx] if idx < len(forecast) else 10.0
            max_q = bounds[idx] if idx < len(bounds) else 40.0

            fc_effective = fc * (3.0 if trend_active and prod == 0 else 1.0)

            # During strike, warehouse_a (node 1) is blocked — boost warehouse_b
            if strike_active and node == 2:
                fc_effective *= 1.5

            safety = 2.0 if lt > 0 else 0.5
            target = fc_effective * (lt + safety)
            net_position = current_inv + current_it

            order = max(0.0, target - net_position)
            order = min(order, max_q)

            if net_position > fc_effective * (lt + 6.0):
                order = 0.0

            quantities.append(round(order, 2))

            use_express = 0
            if node >= 3:
                demand_idx = prod * 4 + (node - 3)
                bl = backlog[demand_idx] if demand_idx < len(backlog) else 0.0
                if bl > 0 and current_inv < 3.0:
                    use_express = 1
            methods.append(use_express)

    return AgentAction(order_quantities=quantities, shipping_methods=methods)


# ---------------------------------------------------------------------------
# Prompt builder (full context per step)
# ---------------------------------------------------------------------------
def build_user_prompt(step: int, obs: Any, prev_reward_terms: dict | None = None) -> str:
    inv_lines = []
    for node in range(7):
        parts = []
        for prod in range(3):
            idx = node * 3 + prod
            inv_val = obs.inventory_levels[idx] if idx < len(obs.inventory_levels) else 0.0
            it_val = obs.in_transit_qty[idx] if idx < len(obs.in_transit_qty) else 0.0
            fc_val = obs.demand_forecast[idx] if idx < len(obs.demand_forecast) else 0.0
            bl_val = obs.order_backlogs[idx] if idx < len(obs.order_backlogs) else 0.0
            parts.append(f"P{prod}:inv={inv_val:.1f},transit={it_val:.1f},fc={fc_val:.1f},bl={bl_val:.1f}")
        inv_lines.append(f"  {ECHELON_NAMES[node]}: " + " | ".join(parts))

    backlog = obs.customer_backlog if hasattr(obs, "customer_backlog") else []
    bl_str = ", ".join(f"{v:.1f}" for v in backlog) if backlog else "none"

    active_events = ", ".join(obs.active_events) if obs.active_events else "none"
    horizon_remaining = int(obs.horizon) - step

    lead_times = getattr(obs, "node_base_lead_times", FALLBACK_LEAD_TIMES)
    lt_str = ", ".join(f"{ECHELON_NAMES[i]}={int(lead_times[i])}" for i in range(7))

    fuel = getattr(obs, "fuel_price_multiplier", 1.0)
    regime = getattr(obs, "regime", "baseline")
    disruption = getattr(obs, "disruption_link", None) or "none"

    prev_block = ""
    if prev_reward_terms:
        total = prev_reward_terms.get("total", 0)
        fill_b = prev_reward_terms.get("fill_rate_bonus", 0)
        hold = prev_reward_terms.get("holding_cost", 0)
        stock = prev_reward_terms.get("stockout_penalty", 0)
        trans = prev_reward_terms.get("transport_cost", 0)
        prev_block = (
            f"\nPrevious step reward: {total:+.3f} "
            f"(fill_bonus={fill_b:.3f}, hold={hold:.1f}, stockout={stock:.1f}, transport={trans:.2f})"
        )

    prompt = textwrap.dedent(f"""\
Day {step} of {obs.horizon} | Remaining: {horizon_remaining} | Regime: {regime}
Fill rate: {obs.fill_rate:.3f} | Carbon: {obs.carbon_footprint:.1f} | Fuel multiplier: {fuel:.3f}
Active events: {active_events} | Disrupted link: {disruption}
Base lead times: {lt_str}{prev_block}

INVENTORY / IN-TRANSIT / FORECAST / ORDER-BACKLOG per node-product:
{chr(10).join(inv_lines)}

Customer backlog (P0_R1..R4, P1_R1..R4, P2_R1..R4):
[{bl_str}]

Order bounds (max_order_qty per slot):
{[round(b, 1) for b in obs.action_bounds]}

Respond with the optimal JSON action now.""")
    return prompt


# ---------------------------------------------------------------------------
# LLM call with retry + fallback
# ---------------------------------------------------------------------------
def _call_llm(
    client: OpenAI,
    messages: list[dict],
    verbose: bool,
    *,
    temperature: float,
) -> tuple[Optional[AgentAction], Optional[str]]:
    """Returns (AgentAction, source) where source is 'llm' or None on failure."""
    last_error: str = ""

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            t0 = time.perf_counter()
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                stream=False,
                timeout=LLM_TIMEOUT,
            )
            elapsed = time.perf_counter() - t0

            if not completion.choices:
                raise ValueError("API returned 0 choices — empty response")

            response_text = (completion.choices[0].message.content or "").strip()
            finish_reason = getattr(completion.choices[0], "finish_reason", None)

            _diag(
                f"  API {elapsed:.2f}s model={MODEL_NAME!r} "
                f"finish={finish_reason!r} chars={len(response_text)}",
                verbose=verbose,
            )

            if not response_text:
                raise ValueError("Model returned empty content string")

            action = _parse_action_safe(response_text)
            return action, "llm"

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            _diag(
                f"  LLM attempt {attempt + 1}/{LLM_MAX_RETRIES + 1} failed: {last_error}",
                verbose=verbose,
            )
            if attempt < LLM_MAX_RETRIES:
                time.sleep(1.5)

    _diag(f"  All LLM attempts failed. Last error: {last_error}", verbose=verbose)
    return None, "failed"


def _parse_action_safe(response_text: str) -> AgentAction:
    """Parse JSON from LLM response. Handles markdown code fences and auto-pads lists."""
    text = response_text.strip()

    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    data = json.loads(text)

    order_qs: list = data.get("order_quantities", [0.0] * 21)
    ship_ms: list = data.get("shipping_methods", [0] * 21)

    order_qs = [float(v) for v in order_qs]
    ship_ms = [int(v) for v in ship_ms]
    if len(order_qs) < 21:
        order_qs += [0.0] * (21 - len(order_qs))
    if len(ship_ms) < 21:
        ship_ms += [0] * (21 - len(ship_ms))

    return AgentAction(order_quantities=order_qs[:21], shipping_methods=ship_ms[:21])


# ---------------------------------------------------------------------------
# Main task runner
# ---------------------------------------------------------------------------
def run_task(
    client: Optional[OpenAI],
    spec: TaskSpec,
    *,
    verbose: bool = True,
    policy: str = "llm",
    temperature: float = DEFAULT_TEMPERATURE,
) -> float:
    tid = spec.task_id
    _protocol_line(f"[START] task={tid}")

    try:
        env = SupplyChainEnv()
        obs = env.reset(
            difficulty=spec.difficulty,
            seed=spec.seed,
            horizon=spec.horizon,
        )
    except Exception as exc:
        _diag(f"ENV RESET FAILED for task={tid}: {exc}", verbose=True)
        _protocol_line(f"[END] task={tid} score=0 steps=0")
        return 0.0

    _diag(
        f"--- Task {tid}: seed={spec.seed} horizon={spec.horizon} "
        f"fill_rate={obs.fill_rate:.4f} day={obs.day} policy={policy} ---",
        verbose=verbose,
    )

    system_prompt = build_system_prompt(spec)
    prev_reward_terms: dict | None = None
    steps_taken = 0
    llm_calls = 0
    heuristic_calls = 0
    zeros_calls = 0

    for step in range(1, spec.horizon + 1):
        try:
            _diag(
                f"Step {step}/{spec.horizon} | inv_sum={sum(obs.inventory_levels):.1f} "
                f"backlog={sum(obs.customer_backlog):.1f} events={obs.active_events}",
                verbose=verbose,
            )

            if policy == "zeros":
                action = AgentAction(
                    order_quantities=[0.0] * 21,
                    shipping_methods=[0] * 21,
                )
                source = "zeros"
                zeros_calls += 1
            elif policy == "heuristic":
                action = heuristic_action(obs)
                source = "heuristic"
                heuristic_calls += 1
            else:
                if client is None:
                    action = heuristic_action(obs)
                    source = "heuristic"
                    heuristic_calls += 1
                else:
                    user_prompt = build_user_prompt(step, obs, prev_reward_terms)
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    action, source = _call_llm(
                        client,
                        messages,
                        verbose=verbose,
                        temperature=temperature,
                    )
                    if action is None:
                        action = heuristic_action(obs)
                        source = "heuristic"
                    if source == "llm":
                        llm_calls += 1
                    else:
                        heuristic_calls += 1

            _diag(
                f"  [{source}] orders={[round(q, 1) for q in action.order_quantities[:7]]}... "
                f"express_count={sum(action.shipping_methods)}",
                verbose=verbose,
            )

            obs = env.step(action)
            steps_taken = step
            reward = float(obs.reward) if obs.reward is not None else 0.0
            prev_reward_terms = dict(obs.reward_terms) if obs.reward_terms else None
            _protocol_line(f"[STEP] step={step} reward={_fmt_reward(reward)}")

            _diag(
                f"  fill_rate={obs.fill_rate:.4f} carbon={obs.carbon_footprint:.2f} done={obs.done}",
                verbose=verbose,
            )

            if obs.done:
                _diag("  Episode done (horizon reached).", verbose=verbose)
                break

        except Exception as exc:
            _diag(
                f"  UNHANDLED exception at step {step}: {type(exc).__name__}: {exc}\n"
                f"  {traceback.format_exc()}",
                verbose=True,
            )
            _protocol_line(f"[STEP] step={step} reward=0")
            steps_taken = step
            try:
                obs = env.step(heuristic_action(obs))
            except Exception:
                break

    try:
        state = env.state
        score = float(grade_episode(state, tid))
    except Exception as exc:
        _diag(f"  GRADING FAILED: {exc}", verbose=True)
        score = 0.0

    _protocol_line(f"[END] task={tid} score={_fmt_score(score)} steps={steps_taken}")

    _diag(
        f"Task {tid} complete: score={score:.4f} llm_calls={llm_calls} "
        f"heuristic_calls={heuristic_calls} zeros_calls={zeros_calls} "
        f"fill_rate={getattr(env.state, 'fill_rate', 0):.4f}",
        verbose=verbose,
    )
    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM + heuristic fallback agent on SupplyChainEnv.")
    p.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Protocol lines on stdout only; diagnostics on stderr suppressed.",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Use temperature=0 for LLM calls (also set via INFERENCE_DETERMINISTIC=1).",
    )
    p.add_argument(
        "--policy",
        choices=("llm", "zeros", "heuristic"),
        default="llm",
        help="llm: OpenAI client + heuristic fallback; zeros/heuristic: fully deterministic actions.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    det_env = os.getenv("INFERENCE_DETERMINISTIC", "").strip().lower() in ("1", "true", "yes")
    deterministic = bool(args.deterministic or det_env)
    temperature = 0.0 if deterministic else DEFAULT_TEMPERATURE
    policy = args.policy

    if policy == "llm" and not API_KEY:
        print(
            "WARNING: OPENAI_API_KEY, HF_TOKEN, and API_KEY are unset. "
            "LLM calls will fail; heuristic fallback will be used each step.",
            file=sys.stderr,
            flush=True,
        )

    specs = task_specs()
    if verbose:
        print("=" * 60, file=sys.stderr, flush=True)
        print("Inference run  (diagnostics->stderr | protocol->stdout)", file=sys.stderr, flush=True)
        print(f"  API base URL : {API_BASE_URL}", file=sys.stderr, flush=True)
        print(f"  Model        : {MODEL_NAME}", file=sys.stderr, flush=True)
        print(
            f"  API key set  : {bool(API_KEY)} (OPENAI_API_KEY | HF_TOKEN | API_KEY)",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"  Policy       : {policy}  temperature={temperature}  deterministic={deterministic}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"  Generation   : max_tokens={MAX_TOKENS} timeout={LLM_TIMEOUT}s",
            file=sys.stderr,
            flush=True,
        )
        print(
            "  Tasks        : "
            + ", ".join(f"{s.task_id}(seed={s.seed},h={s.horizon})" for s in specs),
            file=sys.stderr,
            flush=True,
        )
        print("=" * 60, file=sys.stderr, flush=True)

    client: Optional[OpenAI] = None
    if policy == "llm":
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY or "placeholder-no-key",
        )

    scores: dict[str, float] = {}

    for spec in specs:
        try:
            scores[spec.task_id] = run_task(
                client,
                spec,
                verbose=verbose,
                policy=policy,
                temperature=temperature,
            )
        except Exception as exc:
            _diag(
                f"FATAL: run_task({spec.task_id}) raised {type(exc).__name__}: {exc}",
                verbose=True,
            )
            _protocol_line(f"[END] task={spec.task_id} score=0 steps=0")
            scores[spec.task_id] = 0.0

    if verbose:
        print("", file=sys.stderr, flush=True)
        print("Final Scores:", file=sys.stderr, flush=True)
        for task_id, score in scores.items():
            print(f"  {task_id.capitalize():8s}: {score:.4f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
