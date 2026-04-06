---
title: Supply Chain Environment Server
emoji: 🚀
colorFrom: green
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Multi-Echelon Supply Chain Environment (OpenEnv)

This package is an **OpenEnv**-compatible supply-chain simulation: a **multi-node, multi-product** inventory network with **stochastic demand**, **two shipping modes** (standard vs express), **fuel-price noise**, **carbon accounting**, **random news-style disruptions**, and a **scalar reward** that trades off service level against cost. It is exposed over **HTTP/WebSocket** so agents (including RL policies) can interact without being embedded in the simulator.

The sections below explain what the product does, how reinforcement learning maps onto it, what the agent and environment are in concrete terms, how rewards are built, a real-world analogy, and how to run and extend the codebase.

### OpenEnv pattern and tutorial

This package mirrors the layout described in the official **OpenEnv** walkthrough ([`OpenEnv_Tutorial.ipynb`](https://github.com/meta-pytorch/OpenEnv/blob/c719decf2b19175d5ca35301d58a14c83e985480/tutorial/examples/OpenEnv_Tutorial.ipynb)):

| Tutorial idea | In this repo |
|---------------|----------------|
| Type-safe **Action** / **Observation** / **State** (Pydantic models) | `hackathon/models.py` — `AgentAction`, `AgentObservation`, `SupplyChainState` |
| **Environment** — `reset`, `step`, `state` | `hackathon/server/hackathon_environment.py` — `SupplyChainEnv` |
| **EnvClient** — HTTP `reset` / `step` / `state` | `hackathon/client.py` — `SupplyChainClient` |
| FastAPI server wiring | `hackathon/server/app.py` — `create_app(SupplyChainEnv, …)` |

Run the tutorial notebook top-to-bottom for the client/server mental model (REST-style envs, isolation, typing); then use this repo for the supply-chain **domain** on the same APIs.

---

## 1. What the product does (end-to-end)

Each **episode** is a sequence of **days** (steps), up to a **horizon** (default 365). On every day:

1. **News events** may start, tick down, or expire (e.g. canal delay, strike, viral demand spike).
2. **Inbound shipments** whose countdown reached zero **arrive** into node inventory.
3. **Existing customer backlogs** at retailers are filled from on-hand stock where possible.
4. The **agent submits an action**: how much to order for each **(node × product)** slot, and whether each order ships **standard** or **express**.
5. Those orders are placed into **per-method backlogs** at each node; the environment then **dispatches** what upstream inventory allows, creates **pipeline shipments** with **stochastic lead times**, and accrues **transport-related carbon**.
6. **Customer demand** is sampled for each **(product × retailer)** stream (12 streams), possibly modified by difficulty, seasonality, weekly patterns, and active events.
7. Retailers **serve** demand from inventory; **unmet demand** adds to **per-stream customer backlog**.
8. A **fuel price multiplier** does a bounded random walk (affects transport cost in the reward).
9. **Reward terms** are computed from holding, stockouts/backlogs, transport, carbon, and same-day fill rate; the step **reward** is clipped to **[-1, 1]**.
10. The server returns an **observation** (inventories, in-transit, forecasts, masks, events, etc.) plus **reward** and **done**.

So the “product” is both the **simulator** (`SupplyChainEnv`) and the **remote API** (`hackathon.server.app`) that lets any client train or evaluate policies against it.

---

## 2. How reinforcement learning fits in

Formally this is a **Markov decision process (MDP)** (or **POMDP** on `hard`, where factory inventory can be hidden):

| RL concept | In this codebase |
|------------|------------------|
| **State** | Internal inventories, pipelines, backlogs, fuel multiplier, active events, RNG state, step count, etc. The agent sees a **partial summary** in `AgentObservation` (e.g. `state_vector`, flattened tensors, regime). |
| **Action** | `AgentAction`: 21 order quantities + 21 shipping method flags (see §4). |
| **Transition** | One call to `step()` runs the full daily physics (shipments, demand, costs). |
| **Reward** | Scalar `observation.reward` (also in `reward_terms` for shaping breakdown). |
| **Episode** | `reset()` then repeated `step()` until `done` (horizon) or early stop if you add one. |

**How RL gets used:** you pick a policy **π(a | s)** (see `train/` for PyTorch examples on the in-process OpenEnv API, or use Ray RLlib / other stacks with your own adapter). Each step you encode `AgentObservation` into a numeric vector (often `state_vector` plus a few scalars), sample or argmax an action, map it to `AgentAction`, call `env.step(action)`, and store transitions **(s, a, r, s′, done)**. The OpenEnv HTTP client (`SupplyChainClient`; alias `SupplyChainInventoryEnv`) is one way to get **(s′, r, done)** from a remote process—useful for distributed training or HF Spaces deployments.

This repo does **not** ship a trained neural network; it ships the **environment contract** so you can plug in any RL or optimization method.

---

## 3. What is the “agent”?

The **agent** is whatever code chooses the next `AgentAction`. Examples:

- **Rule-based:** base-stock or min–max per SKU, fixed express fraction, etc.
- **MPC / optimization:** solve a rolling horizon with a demand forecast from `demand_forecast` / `recent_customer_demand`.
- **RL policy:** neural network that maps observation features to 21 continuous order values and 21 binary or discrete shipping choices (multi-discrete or factorized heads).

The environment is **agnostic**: it only requires valid non-negative orders and shipping methods in `{0, 1}` after clipping.

---

## 4. What is the “environment”?

### 4.1 Topology (7 nodes, 3 products)

**Nodes** (`echelon_names`):

| Index | Name | Role |
|------:|------|------|
| 0 | `factory` | Infinite external supply; no upstream |
| 1 | `warehouse_a` | Feeds retailers 1–2 |
| 2 | `warehouse_b` | Feeds retailers 3–4 |
| 3–6 | `retailer_1` … `retailer_4` | Face stochastic end-customer demand |

**Upstream links** (who ships to whom):

- Retailers **3, 4** ← warehouse **A** (node 1)  
- Retailers **5, 6** ← warehouse **B** (node 2)  
- Warehouses **1, 2** ← factory **0**

**Products:** three SKUs with different demand scales (see `_sample_customer_demand` in `server/hackathon_environment.py`).

### 4.2 Action layout (21 + 21)

Actions are **flattened** in **node-major, then product** order:

- Index `i = node * 3 + product` for `node ∈ [0..6]`, `product ∈ [0..2]`.
- `order_quantities[i]`: units ordered at that node for that product (clipped to `[0, max_order_qty]`).
- `shipping_methods[i]`: `0` = standard, `1` = express (anything non-1 is treated as 0).

So **21** order entries and **21** shipping entries total.

### 4.3 Customer demand streams (12)

Demand is one time series per **(product, retailer)**:

- Ordering in the sampled list is: product 0 for retailers 1–4, then product 1 for retailers 1–4, then product 2 for retailers 1–4.
- `customer_backlog` has length **12** in the same order.

### 4.4 Shipping, lead time, and inventory dynamics

- Orders accumulate in **backlogs split by shipping method**; dispatch processes **express before standard** at each node/product.
- **Factory** (no upstream): shipments can land in the pipeline or immediately in inventory if base lead time is 0.
- **Base lead times** per node: `[4, 3, 3, 2, 2, 2, 2]` days (before modifiers).
- **Express** uses a shorter effective lead time (`max(1, base_lt // 2)` before stochastic jitter).
- **Stochasticity:** integer lead time is jittered in `[effective_lt - 1, effective_lt + 1]` (clamped to ≥ 1).
- **Carbon:** per unit shipped, standard adds **1.0**, express adds **5.0** to cumulative `carbon_footprint` (see `_dispatch_replenishment_orders`).

### 4.5 Difficulty presets (`easy` | `medium` | `mvp` | `hard`)

`DIFFICULTY_PRESETS` controls **base demand**, **demand noise**, **season shift** (for `hard` and product 2 behavior), **partial visibility** (hide factory inventory as `-1` in observations), **link disruption probability**, **`max_order_qty`**, and **`reward_scale`**.

**Note:** The network is always the **full 7×3** layout in the current implementation; `active_echelons` from presets is overridden to all nodes active on `reset()`. Difficulty still strongly affects demand process, visibility, disruptions, and reward scaling.

### 4.6 Random “news” events

With a small probability each day (higher on `hard`), if no event is active, one may start:

| Event | Duration (days) | Effect (high level) |
|-------|-----------------|---------------------|
| **Canal Blockage** | 14 | Multiplies effective lead times |
| **Labor Strike** | 7 | Blocks dispatch from warehouse A (node 1) and from sources that depend on it |
| **Social Media Trend** | 5 | Strongly increases mean demand for **product 0** |

Active event names appear in `observation.active_events`.

### 4.7 Link disruptions (`hard`)

With probability `disruption_probability`, a random **node name** may be “disrupted”; shipped quantity on that link can be scaled down (see `_dispatch_replenishment_orders`).

### 4.8 Observation highlights

Besides raw lists, the code builds a **`state_vector`**: for each of 21 slots, six features: inventory, in-transit qty, forecast, days since last order, holding cost rate, minimum pipeline ETA. That vector is a convenient input for ML models.

`recent_customer_demand` is a **flattened** history (each of 12 streams keeps up to 7 recent days).

---

## 5. Rewards and penalties (exact logic)

Each step, `_compute_reward_terms` returns a dictionary; **`reward_terms["total"]`** is the clipped step reward. Components:

### 5.1 Holding cost

For each active node and product:

- Linear term: `HOLDING_COSTS[node] * inventory`
- Quadratic “excess over target” term: `HOLDING_COSTS[node] * max(inventory - target[node], 0)² / target[node]`

Node targets (hard-coded): `[60, 30, 30, 15, 15, 15, 15]` per node, same across products at that node.

### 5.2 Stockout / backlog penalty

- `0.11 * sum(customer_backlog)` (all 12 retailer-product backlogs)
- Plus `0.02 * sum(order_backlogs)` over nodes (flattened internal backlogs across methods)

### 5.3 Transport cost

For each node/product, for **shipped** quantity this step:

- `TRANSPORT_COSTS[node] * shipped + FIXED_ORDER_COSTS[node]` per positive shipment line  
- Entire transport sum multiplied by **`_fuel_price_multiplier`** (random walk, bounded [0.8, 1.5]).

Constants `HOLDING_COSTS`, `TRANSPORT_COSTS`, `FIXED_ORDER_COSTS` are per-echelon arrays at the top of `hackathon_environment.py`.

### 5.4 Carbon penalty

- `carbon_penalty = 0.05 * total_carbon / (step_count + 1)`  
  so early steps penalize accumulated carbon slightly less aggressively than a raw linear term.

### 5.5 Fill-rate bonus (same step)

- `fill_rate_bonus = 0.65 * (served_demand / current_demand)` with safe handling when demand is 0.

Here `served_demand` / `current_demand` are **sums over all 12 streams** for that day.

### 5.6 Total and clipping

```text
total_cost = holding_cost + stockout_penalty + transport_cost + carbon_penalty
reward_scale = config["reward_scale"] * (num_nodes / 3)   # i.e. scaled up for 7 nodes
raw_reward = fill_rate_bonus - (total_cost / reward_scale)
reward = clip(raw_reward, -1, 1)
```

**Cumulative** `SupplyChainState.cumulative_reward` sums the same **clipped** per-step `reward_terms["total"]` that is returned on each `step()` (see `step()` in `hackathon_environment.py`).

Episode-level **`fill_rate` in state** is cumulative: `total_served / total_demand`.

---

## 6. Real-life scenario (analogy)

Imagine a **consumer electronics** company:

- A **factory** in Asia produces three related SKUs (e.g. phone, earbuds, case).
- Two **regional warehouses** serve different retail clusters.
- **Four large retail chains** each sell all three products with different popularity.

Each morning, planners decide **how much to reorder** for every SKU at every location and whether to use **ocean + truck (standard)** vs **air freight (express)**. Demand **jumps** during a **social trend**; a **port strike** freezes one warehouse lane; a **canal blockage** lengthens transit. The **board** cares about **service level**, **inventory carrying cost**, **freight spend**, and **emissions**.

This environment abstracts that into daily decisions, stochastic demand, lead times, and a single reward that forces tradeoffs—exactly the kind of problem **RL or hybrid optimizers** are researched for.

---

## 7. How to run

### 7.1 Python path and installs

Imports use the package name **`hackathon`**. The `hackathon/` folder is the package root (see `pyproject.toml`). Typical setups:

**Option A — Repository root on `PYTHONPATH` (no install)**

```bash
cd /path/to/parent-of-hackathon    # e.g. your Meta repo root that contains hackathon/
export PYTHONPATH="$PWD"
uvicorn hackathon.server.app:app --reload --host 0.0.0.0 --port 8000
```

**Option B — Editable install from `hackathon/`**

```bash
cd /path/to/hackathon
pip install -e ".[dev]"    # or: uv pip install -e ".[dev]"
uvicorn hackathon.server.app:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health: [http://localhost:8000/health](http://localhost:8000/health)

### 7.2 Run module entrypoint

```bash
cd /path/to/hackathon
PYTHONPATH=.. python -m hackathon.server.app
```

(Here `..` is the parent directory that contains the `hackathon` package folder.)

### 7.3 Docker

From `hackathon/`:

```bash
docker build -t hackathon-env:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 hackathon-env:latest
```

From repo root:

```bash
docker build -t hackathon-env:latest -f hackathon/server/Dockerfile hackathon
```

### 7.4 Client example (HTTP)

Start the server, then:

```python
from hackathon.client import SupplyChainClient
from hackathon.models import AgentAction

env = SupplyChainClient(base_url="http://localhost:8000")

result = env.reset(difficulty="mvp", seed=0, horizon=365)
obs = result.observation if hasattr(result, "observation") else result

action = AgentAction(
    order_quantities=[5.0] * 21,
    shipping_methods=[0] * 21,  # all standard
)
step = env.step(action)
print(step.reward, step.observation.fill_rate, step.observation.active_events)
```

`SupplyChainClient` sends `order_quantities` and `shipping_methods` in the JSON body; the client unwraps nested `observation` payloads if the server double-wraps them. (`SupplyChainInventoryEnv` is an alias for `SupplyChainClient`.)

### 7.5 Async smoke test against a live server

With the server running:

```bash
cd /path/to/parent-of-hackathon
python hackathon/test_run.py
```

(`test_run.py` adds the repo root to `sys.path` and uses `asyncio` if the client returns coroutines.)

### 7.6 Hugging Face Spaces (`openenv push`)

From the directory that contains `openenv.yaml`:

```bash
openenv push
openenv push --repo-id my-org/my-env --private
```

See the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) repo for login and CLI details. For a guided introduction (architecture, `Environment` vs `EnvClient`, HTTP flow), use the [**OpenEnv tutorial notebook**](https://github.com/meta-pytorch/OpenEnv/blob/c719decf2b19175d5ca35301d58a14c83e985480/tutorial/examples/OpenEnv_Tutorial.ipynb).

---

## 8. How to test

### 8.1 Direct environment (no HTTP)

Instantiate `SupplyChainEnv` and call `reset` / `step` (fast, good for unit logic):

```bash
cd /path/to/parent-of-hackathon
PYTHONPATH=. python -c "
from hackathon.server.hackathon_environment import SupplyChainEnv
from hackathon.models import AgentAction

env = SupplyChainEnv()
o = env.reset(difficulty='mvp', seed=1)
a = AgentAction(order_quantities=[1.0]*21, shipping_methods=[0]*21)
o2 = env.step(a)
print(o2.reward, o2.day, len(o2.inventory_levels))
"
```

### 8.2 Diagnostic scripts in `hackathon/`

For a **single combined notebook** with verbose `DEBUG` output, open `hackathon/test/supply_chain_tests.ipynb` (run setup, then definitions, then the “Run all in-process” cell).

Scripts such as `test_news_events.py`, `test_stochastic_lt.py`, `test_sustainability.py`, `test_variable_costs.py`, and `test_network_topology.py` mutate or inspect the env to print **PASS/FAIL** style checks. Run from repo root:

```bash
PYTHONPATH=. python hackathon/test_news_events.py
```

Some scripts were authored during iterative development; if a check prints **FAIL**, compare the message to the current 7×3 layout and update the expectation (e.g. lengths **21** and **12**).

### 8.3 Pytest

Optional dev dependency: `pip install -e ".[dev]"` then:

```bash
cd /path/to/hackathon
pytest -q
```

Not all `test_*.py` files may be strict pytest modules; prefer **named scripts** above for exploratory verification.

---

## 9. How to improve (engineering and research)

**Environment fidelity**

- Replace Gaussian demand with **empirical** or **copula** models; add **promotions** and **cross-SKU correlation**.
- Model **capacity constraints** at warehouses, **MOQs**, and **supplier reliability** separate from “news” events.
- Split **transport vs carbon** pricing so express cost reflects fuel more consistently than a fixed factor.

**Reward design**

- **Multi-objective** Pareto training (service vs cost vs carbon) instead of one weighted scalar.
- **Constraint-aware RL** (e.g. minimum fill rate as a hard constraint via Lagrangian methods).
- If you need **dense unclipped** rewards for learning, expose `raw_reward` before clipping or log `reward_terms` components separately.

**Observation and POMDP**

- On `hard`, factory inventory is masked; train **recurrent** or **belief** policies; add explicit **uncertainty** features.

**Algorithms**

- **Discrete** shipping + **continuous** order sizes: use **parameterized action** spaces or hierarchical policies.
- **Offline RL** from logged historical `(state, action, cost)` if you can export trajectories from this env.

**Ops**

- Use **WebSocket** sessions from `openenv-core` for high-throughput rollouts.
- Pin versions via `uv.lock` / `pyproject.toml` for reproducible benchmarks.

---

## 10. Project structure

```text
hackathon/
├── openenv.yaml              # OpenEnv manifest (HF / CLI)
├── pyproject.toml            # Package metadata, openenv-core dep
├── uv.lock                   # Locked deps (optional but recommended)
├── README.md                 # This file
├── __init__.py
├── _compat.py                # OpenEnv types / app factory
├── client.py                 # HTTP client (SupplyChainClient)
├── models.py                 # Pydantic action/observation/state
├── test_*.py                 # Diagnostics / informal tests
├── test/                     # Combined Jupyter notebook diagnostics
├── train/                    # OpenEnv-native PyTorch RL entrypoints + obs/action helpers
└── server/
    ├── app.py                # FastAPI app (create_app)
    ├── hackathon_environment.py  # Core simulation + reward
    ├── Dockerfile
    └── requirements.txt
```

---

## 11. Evaluation and Graded Tasks

The environment supports three distinct, graded tasks, each scored programmatically between 0.0 and 1.0 based on fill rate and costs.

- **Task 1 (Easy):** Target > 70% fill rate with basic demand.
- **Task 2 (Medium):** Target > 80% fill rate with volatile demand.
- **Task 3 (Hard):** Target > 85% fill rate, considering link disruptions and carbon footprint penalties.

### Baseline Inference

We provide a baseline script `inference.py` at the repository root that evaluates the three tasks sequentially using an OpenAI-compatible client. Ensure the following environment variables are set:

```bash
export API_BASE_URL="https://api.openai.com/v1" # Or Hugging Face Router
export MODEL_NAME="gpt-4o"                      # Or another capable model
export HF_TOKEN="<your_api_key>"                # Can also use API_KEY
```

Run the inference:
```bash
python inference.py
```
**Baseline Scores:**
- Easy Task: `~0.85`
- Medium Task: `~0.60`
- Hard Task: `~0.45`

### Pre-submission Validation

Run the pre-submission validator before submitting:
```bash
python validate_submission.py https://your-space-url.hf.space --repo-dir .
```

---

## 12. Training RL agents (`train/`)

This package does **not** ship a pretrained network. Training scripts use the **OpenEnv** surface area only: in-process `SupplyChainEnv.reset` / `step(AgentAction)` (the same `Environment` implementation the HTTP server wraps). There is **no Gym / Gymnasium** dependency.

### 11.1 Install training extras

From the `hackathon/` directory (or install the package editable from repo root):

```bash
pip install -e ".[train]"
```

That pulls in **PyTorch** and **NumPy** for the reference trainers. The core environment still installs without these.

### 11.2 Observation and action helpers (`train/__init__.py`)

| Symbol | Role |
|--------|------|
| `STATE_VECTOR_DIM` / `ACTION_DIM` | **126** / **42** |
| `observation_to_vector(obs)` | Pads/truncates `AgentObservation.state_vector` to a fixed NumPy vector. |
| `vector_to_agent_action(vec, max_order_qty)` | Maps **42** floats in **[0, 1]** to `AgentAction`: first **21** × `max_order_qty` → orders; last **21** threshold **0.5** → shipping (0 standard / 1 express). |
| `new_supply_chain_env(...)` | `SupplyChainEnv` + first `reset` (optional helper). |

Run with repo root on `PYTHONPATH` (parent of the `hackathon` folder).

### 11.3 Algorithm entrypoints (`agent_{algorithm}.py`)

Each script is **PyTorch** only and saves a **`.pt`** checkpoint (not Stable-Baselines3 / Gym).

| File | Method | Output |
|------|--------|--------|
| `train/agent_ppo.py` | PPO (GAE, clipped surrogate, Beta policy) | `models/ppo_supply_chain.pt` |
| `train/agent_sac.py` | SAC (twin Q, replay buffer, Beta policy) | `models/sac_supply_chain.pt` |
| `train/agent_reinforce.py` | REINFORCE (Monte Carlo returns, Beta policy) | `models/reinforce_supply_chain.pt` |

**Run length:** `agent_reinforce.py` uses **`--episodes`** (one policy update per full rollout). `agent_ppo.py` and `agent_sac.py` stop after **`--total-steps`** calls to **`env.step(AgentAction)`** (default **100_000** / **80_000**)—each call is one simulated **day** in the supply-chain env, not a separate OpenEnv “timestep” field. Checkpoints store the parsed value under **`total_steps`** in the saved **`config`** dict.

Examples:

```bash
cd /path/to/parent-of-hackathon
PYTHONPATH=. python -m hackathon.train.agent_ppo --total-steps 100_000 --difficulty easy --horizon 120
PYTHONPATH=. python -m hackathon.train.agent_sac --total-steps 80_000 --difficulty easy
PYTHONPATH=. python -m hackathon.train.agent_reinforce --episodes 500 --difficulty easy
```

Training is **stochastic**; pass `--seed` where supported. For harder presets (`mvp`, `hard`), expect longer runs and possible reward saturation (see `debug.md` §3.9).

### 11.4 Evaluation and integration

- Load checkpoints with `torch.load` and restore `state_dict` into the same architecture as in the script (or refactor shared `nn.Module` definitions into a small module).
- At inference, map policy outputs in **[0, 1]^{42}** with `vector_to_agent_action` using `env._max_order_qty` from the live `SupplyChainEnv`.
- For **HTTP** deployment, keep one env per session (see `debug.md` §3.1); do not train concurrently against a single shared server process.

---

## 12. API summary (models)

| Model | Fields (main) |
|-------|----------------|
| **AgentAction** | `order_quantities: list[float]` length **21**, `shipping_methods: list[int]` length **21** |
| **AgentObservation** | `state_vector`, `inventory_levels` (21), `in_transit_qty`, `demand_forecast`, `order_backlogs`, `customer_backlog` (12), `recent_customer_demand`, `carbon_footprint`, `fill_rate`, `reward_terms`, `action_bounds`, `active_events`, `done`, `reward`, … |
| **SupplyChainState** | Episode aggregates: `cumulative_reward`, `total_cost`, `total_demand`, `total_served`, `fill_rate`, etc. |

Aliases: `SupplyChainInventoryEnv` → `SupplyChainClient`; `SupplyChainInventoryEnvironment` → `SupplyChainEnv`.

---

## License

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.  
This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
