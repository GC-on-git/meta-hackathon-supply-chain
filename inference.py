import argparse
import json
import os
import sys
import textwrap
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from service.grading import grade_episode
from service.hackathon_environment import SupplyChainEnv
from service.models import AgentAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
MAX_STEPS = 30
TEMPERATURE = 0.2
MAX_TOKENS = 1000

TASK_SEEDS = {
    "easy": int(os.getenv("TASK_SEED_EASY", "1001")),
    "medium": int(os.getenv("TASK_SEED_MEDIUM", "2002")),
    "hard": int(os.getenv("TASK_SEED_HARD", "3003")),
}

SYSTEM_PROMPT = textwrap.dedent("""
    You are a supply chain manager controlling 7 nodes and 3 products (21 total slots).
    You need to output order quantities and shipping methods for each slot to maximize fill rate and minimize costs.
    Output MUST be a valid JSON object matching this schema:
    {
      "order_quantities": [float, float, ...], // exactly 21 floats
      "shipping_methods": [int, int, ...]      // exactly 21 ints (0 for standard, 1 for express)
    }
    No explanations, no markdown blocks, just the JSON.
""").strip()


def _protocol_line(line: str) -> None:
    """Validator-facing structured lines: always stdout, always flushed."""
    print(line, flush=True)


def _fmt_reward(x: float) -> str:
    """Stable decimal string for protocol lines (avoids float noise)."""
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _fmt_score(x: float) -> str:
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _diag(message: str, *, verbose: bool) -> None:
    """Human-readable progress; stderr so stdout stays easy to parse."""
    if verbose:
        print(message, file=sys.stderr, flush=True)


def build_user_prompt(step: int, observation: Any) -> str:
    state_vector = observation.state_vector
    fill_rate = observation.fill_rate
    prompt = textwrap.dedent(f"""
        Day: {step}
        Current Fill Rate: {fill_rate:.2f}
        State Vector (summarized context): {state_vector[:10]}... (truncated)
        Demand Forecast: {observation.demand_forecast}
        Inventory Levels: {observation.inventory_levels}

        Provide the next action as JSON. Keep orders reasonable based on the forecast.
    """).strip()
    return prompt


def parse_model_action(response_text: str) -> AgentAction:
    text = response_text.strip()
    if not text:
        raise ValueError(
            "Model returned empty content. Expected a JSON object with "
            "order_quantities and shipping_methods."
        )
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        preview = text[:500] + ("…" if len(text) > 500 else "")
        raise ValueError(
            f"Model output is not valid JSON ({e}). Preview: {preview!r}"
        ) from e

    order_qs = data.get("order_quantities", [0.0] * 21)
    ship_ms = data.get("shipping_methods", [0] * 21)

    if len(order_qs) < 21:
        order_qs += [0.0] * (21 - len(order_qs))
    if len(ship_ms) < 21:
        ship_ms += [0] * (21 - len(ship_ms))

    return AgentAction(order_quantities=order_qs[:21], shipping_methods=ship_ms[:21])


def _usage_str(completion: Any) -> str:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return "usage=n/a"
    pt = getattr(usage, "prompt_tokens", None)
    ct = getattr(usage, "completion_tokens", None)
    tt = getattr(usage, "total_tokens", None)
    parts = []
    if pt is not None:
        parts.append(f"prompt={pt}")
    if ct is not None:
        parts.append(f"completion={ct}")
    if tt is not None:
        parts.append(f"total={tt}")
    return " ".join(parts) if parts else repr(usage)


def _preview_text(s: str, max_len: int = 320) -> str:
    s = s.replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _step_reward(obs: Any) -> float:
    r = getattr(obs, "reward", None)
    return float(r) if r is not None else 0.0


def run_task(client: OpenAI, task_name: str, *, verbose: bool = True) -> float:
    _protocol_line(f"[START] task={task_name}")

    env = SupplyChainEnv()
    seed = TASK_SEEDS.get(task_name, 0)
    obs = env.reset(difficulty=task_name, seed=seed, horizon=MAX_STEPS)

    _diag(
        f"--- Task {task_name}: seed={seed} horizon={MAX_STEPS} "
        f"initial_fill_rate={obs.fill_rate:.4f} day={obs.day} ---",
        verbose=verbose,
    )

    steps_taken = 0
    for step in range(1, MAX_STEPS + 1):
        user_prompt = build_user_prompt(step, obs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        if verbose:
            _diag(
                f"Step {step}/{MAX_STEPS} prompt_chars system={len(SYSTEM_PROMPT)} user={len(user_prompt)}",
                verbose=verbose,
            )

        t0 = time.perf_counter()
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        elapsed = time.perf_counter() - t0

        response_text = completion.choices[0].message.content or ""
        choice = completion.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)

        if verbose:
            _diag(
                f"  API {elapsed:.2f}s model={MODEL_NAME!r} finish_reason={finish_reason!r} {_usage_str(completion)}",
                verbose=verbose,
            )
            _diag(
                f"  Raw response ({len(response_text)} chars): {_preview_text(response_text)!r}",
                verbose=verbose,
            )

        action = parse_model_action(response_text)
        if verbose:
            oq = action.order_quantities
            total_q = sum(oq)
            express = sum(action.shipping_methods)
            _diag(
                f"  Parsed action: sum(orders)={total_q:.2f} min/max={min(oq):.2f}/{max(oq):.2f} express={express}/21",
                verbose=verbose,
            )

        obs = env.step(action)
        steps_taken = step
        reward = _step_reward(obs)
        _protocol_line(f"[STEP] step={step} reward={_fmt_reward(reward)}")

        if verbose:
            _diag(
                f"  After step: fill_rate={obs.fill_rate:.4f} carbon={obs.carbon_footprint:.2f} "
                f"done={obs.done} day={obs.day}",
                verbose=verbose,
            )

        if obs.done:
            _diag("  Episode terminated early (done=True).", verbose=verbose)
            break

    state = env.state
    score = float(grade_episode(state, task_name))
    _protocol_line(
        f"[END] task={task_name} score={_fmt_score(score)} steps={steps_taken}"
    )

    if verbose:
        _diag(
            f"Task {task_name} done: score={score:.4f} fill_rate={state.fill_rate:.2f} "
            f"cost={state.total_cost:.2f} carbon={state.carbon_footprint:.2f} "
            f"termination_reason={state.termination_reason!r}",
            verbose=verbose,
        )
    return score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM baseline rollouts on SupplyChainEnv.")
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Structured protocol on stdout only; diagnostics on stderr suppressed.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    if not API_KEY:
        print(
            "Error: HF_TOKEN or API_KEY is not set. "
            "Add it to your environment or a .env file in the project root.",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        print("=" * 60, file=sys.stderr, flush=True)
        print("Inference run (diagnostics on stderr; protocol on stdout)", file=sys.stderr, flush=True)
        print(f"  API base URL: {API_BASE_URL}", file=sys.stderr, flush=True)
        print(f"  Model: {MODEL_NAME}", file=sys.stderr, flush=True)
        print(
            f"  Generation: temperature={TEMPERATURE} max_tokens={MAX_TOKENS} max_env_steps={MAX_STEPS}",
            file=sys.stderr,
            flush=True,
        )
        print(f"  Task seeds: {TASK_SEEDS}", file=sys.stderr, flush=True)
        print("=" * 60, file=sys.stderr, flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = ["easy", "medium", "hard"]
    scores: dict[str, float] = {}

    for task in tasks:
        scores[task] = run_task(client, task, verbose=verbose)

    if verbose:
        print("", file=sys.stderr, flush=True)
        print("Baseline Inference Scores:", file=sys.stderr, flush=True)
        for task, score in scores.items():
            print(f"  {task.capitalize()}: {score:.4f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
