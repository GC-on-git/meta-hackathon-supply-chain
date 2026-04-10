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

An OpenEnv environment that simulates a **7-node, 3-product** inventory network with stochastic demand, two shipping modes (standard vs express with monetary and carbon surcharges), fuel-price noise, news-style disruptions, and a reward trading off service level against cost. POMDP on hard difficulty (factory inventory hidden).

## Quick Start

```python
from service.hackathon_environment import SupplyChainEnv
from service.models import AgentAction

env = SupplyChainEnv()
obs = env.reset(difficulty="medium", seed=42, horizon=30)
print(f"Question: inventory={obs.inventory_levels[:3]}, forecast={obs.demand_forecast[:3]}")

obs = env.step(AgentAction(order_quantities=[10.0]*21, shipping_methods=[0]*21))
print(f"Reward: {obs.reward}, Fill: {obs.fill_rate}, Done: {obs.done}")
```

### Baseline Inference

```bash
export HF_TOKEN="<your_key>"
export MODEL_NAME="gpt-4o"
python inference.py                    # LLM + heuristic fallback
python inference.py --policy heuristic # deterministic baseline, no LLM
python inference.py --policy zeros     # zero-order baseline
```

Baseline scores (`--policy heuristic`, default seeds, `TASK_HORIZON=30`): Easy ~0.53, Medium ~0.41, Hard ~0.30.

### Docker

```bash
docker build -t service-env:latest .
docker run --rm -p 8000:8000 service-env:latest
```

### Deploying to Hugging Face Spaces

```bash
openenv push
openenv push --repo-id my-org/my-env --private
```

## Environment Details

### Topology

```
Factory (node 0) ─── infinite external supply
  ├── Warehouse A (node 1) ──┬── Retailer 1 (node 3)
  │                          └── Retailer 2 (node 4)
  └── Warehouse B (node 2) ──┬── Retailer 3 (node 5)
                             └── Retailer 4 (node 6)
```

Three products per node. Slot index: `i = node * 3 + product`.

### Action

**AgentAction**: 21 order quantities + 21 shipping method flags.

| Field | Type | Description |
|-------|------|-------------|
| `order_quantities` | `list[float]` (21) | Units to order per slot, clipped to `[0, max_order_qty]` |
| `shipping_methods` | `list[int]` (21) | `0` = standard, `1` = express |

### Observation

**AgentObservation**: full environment state visible to the agent.

| Field | Shape | Description |
|-------|-------|-------------|
| `inventory_levels` | 21 | On-hand stock per slot. Factory masked as -1 on hard. |
| `in_transit_qty` | 21 | Units in pipeline per slot. |
| `demand_forecast` | 21 | Exponentially-smoothed forecast per slot. |
| `order_backlogs` | 21 | Pending unfilled replenishment orders per slot. |
| `customer_backlog` | 12 | Unmet end-customer demand per (product, retailer). |
| `node_base_lead_times` | 7 | Base lead times per node (from difficulty preset). |
| `fuel_price_multiplier` | scalar | Current fuel multiplier [0.8, 1.5] scaling transport cost. |
| `active_events` | list | Currently active news events. |
| `action_bounds` | 21 | `max_order_qty` per slot. |
| `state_vector` | 126 | Compact feature vector (21 slots x 6 features) for ML. |
| `recent_customer_demand` | variable | Flattened history (12 streams x up to 7 days). |
| `reward_terms` | dict | Breakdown: holding, stockout, transport, carbon, fill bonus, total. |

### Episode Structure

Each step simulates one day:

1. News events tick/spawn (Canal Blockage, Labor Strike, Social Media Trend).
2. In-transit shipments arrive.
3. Existing customer backlogs served from inventory.
4. **Agent action applied**: orders placed into method-specific backlogs.
5. Dispatch: express first, then standard; upstream inventory consumed, shipments enter pipeline.
6. Customer demand sampled (12 streams); retailers serve from inventory; unmet demand backlogs.
7. Fuel price random walk. Step reward computed and clipped to [-1, 1].

Episode ends at `horizon` steps. Final score from `grade_episode()` in [0, 1].

### Shipping

| | Standard (0) | Express (1) |
|--|-------------|-------------|
| Lead time | Full base | `max(1, base // 2)` |
| Variable transport cost | 1x | 2x (`EXPRESS_VARIABLE_MULTIPLIER`) |
| Carbon per unit | 1.0 | 5.0 |

### Lead Times by Difficulty

Presets define 3-tier `[factory, warehouse, retailer]`, expanded to 7 nodes:

| Difficulty | Tier | Expanded |
|------------|:----:|:--------:|
| easy | `[0, 0, 0]` | `[0, 0, 0, 0, 0, 0, 0]` |
| medium | `[0, 3, 3]` | `[0, 3, 3, 3, 3, 3, 3]` |
| mvp | `[4, 3, 2]` | `[4, 3, 3, 2, 2, 2, 2]` |
| hard | `[5, 4, 2]` | `[5, 4, 4, 2, 2, 2, 2]` |

### Events

| Event | Duration | Effect |
|-------|----------|--------|
| Canal Blockage | 14 days | Doubles effective lead times |
| Labor Strike | 7 days | Blocks warehouse A (node 1) dispatch |
| Social Media Trend | 5 days | 3x demand for product 0 |

### Reward

Step reward components (normalised, soft-capped, then averaged):

- **Holding cost**: linear + quadratic excess over target inventory.
- **Stockout penalty**: 0.11 x customer backlog + 0.02 x order backlogs.
- **Transport cost**: per-shipment variable (2x for express) + fixed, scaled by fuel multiplier.
- **Carbon penalty**: `0.05 * total_carbon / (step + 1)`.
- **Fill-rate bonus**: `0.65 * (served / demand)`.

`total_cost` in state accumulates holding + stockout + transport. Carbon is tracked separately as `carbon_footprint` and graded independently on hard.

## Evaluation and Graded Tasks

Three tasks scored in [0, 1] by `grade_episode()` (`service/grading.py`):

| Task | Fill target | Weights (fill / cost / co2) |
|------|:-----------:|:---------------------------:|
| Easy | > 70% | 0.70 / 0.30 / -- |
| Medium | > 80% | 0.65 / 0.35 / -- |
| Hard | > 85% | 0.55 / 0.25 / 0.20 |

Task specs (ids, seeds, objectives) defined in `service/tasks.py`. Objectives are injected into the LLM system prompt by `inference.py`.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` / `API_KEY` | -- | API key for LLM provider |
| `API_BASE_URL` | HF Router | OpenAI-compatible endpoint |
| `MODEL_NAME` | `gpt-3.5-turbo` | Chat model id |
| `TASK_HORIZON` | `30` | Steps per episode |
| `TASK_SEED_EASY/MEDIUM/HARD` | 1001/2002/3003 | Reproducibility seeds |
| `INFERENCE_DETERMINISTIC` | `0` | Set `1` for temperature=0 |

### Pre-submission Validation

```bash
python validate_submission.py https://your-space-url.hf.space --repo-dir .
```

## Training RL Agents

```bash
pip install -e ".[train]"  # from service/ directory

PYTHONPATH=. python -m service.train.agent_ppo --total-steps 100_000 --difficulty easy
PYTHONPATH=. python -m service.train.agent_sac --total-steps 80_000 --difficulty easy
PYTHONPATH=. python -m service.train.agent_reinforce --episodes 500 --difficulty easy
```

Helpers in `service/train/__init__.py`: `STATE_VECTOR_DIM=126`, `ACTION_DIM=42`, `observation_to_vector()`, `vector_to_agent_action()`.

## Project Structure

```
./
├── Dockerfile                    # Container image
├── openenv.yaml                  # OpenEnv manifest (app: server.app:app)
├── pyproject.toml / uv.lock      # Dependencies
├── app.py                        # ASGI shim: uvicorn app:app
├── inference.py                  # Baseline LLM/heuristic agent
├── server/
│   └── app.py                    # FastAPI app (create_app from openenv-core)
└── service/
    ├── hackathon_environment.py  # Core simulation + reward
    ├── models.py                 # Pydantic action/observation/state
    ├── grading.py                # Episode grader (grade_episode)
    ├── tasks.py                  # Task registry (ids, seeds, objectives)
    ├── client.py                 # HTTP client (SupplyChainClient)
    ├── test/                     # Tests + notebook
    └── train/                    # PyTorch RL entrypoints (PPO, SAC, REINFORCE)
```

## License

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
Licensed under the BSD-style license found in the LICENSE file.
