---
title: Hackathon Environment Server
emoji: 📣
colorFrom: green
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Hackathon Environment

A supply-chain inventory environment exposed via an OpenEnv-compatible HTTP/WebSocket API.

You control **order quantities** across a 3‑echelon chain (**manufacturer → warehouse → retailer**). Each step simulates demand, lead times, shipments in transit, backlogs, and (at higher difficulties) partial observability and random disruptions. The server returns an observation vector plus a bounded reward (roughly trading off fill-rate vs costs).

## Quick Start

The simplest way to use the environment is through the client in `hackathon/client.py`.

```python
from hackathon.client import SupplyChainInventoryEnv
from hackathon.models import SupplyChainAction

# Connect to a running server (local or remote)
env = SupplyChainInventoryEnv(base_url="http://localhost:8000")

# Reset (you can pass difficulty/seed/horizon/max_order_qty)
obs = env.reset(difficulty="mvp", seed=0, horizon=365)
print("Day:", obs.day)
print("Echelons:", obs.echelon_names)
print("Inventory:", obs.inventory_levels)

# Take a few steps (order quantities are non-negative, length 3)
for _ in range(5):
    action = SupplyChainAction(order_quantities=[10.0, 8.0, 6.0])
    result = env.step(action)
    print("Reward:", result.reward, "Done:", result.done)
    print("Day:", result.observation.day, "Fill rate:", result.observation.fill_rate)
    print("Customer backlog:", result.observation.customer_backlog)
```

## Running the Server Locally

From the `hackathon/` directory:

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

Then open:
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From the hackathon/ directory
docker build -t hackathon-env:latest -f server/Dockerfile .

# Or from the repository root
# docker build -t hackathon-env:latest -f hackathon/server/Dockerfile hackathon
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**`SupplyChainAction`**
- `order_quantities` (`list[float]`, length 3): non-negative order quantities for each echelon.

### Observation
**`SupplyChainObservation`** (selected fields)
- `day`: current timestep
- `horizon`: episode length
- `difficulty`: one of `easy|medium|mvp|hard`
- `echelon_names`: `["manufacturer", "warehouse", "retailer"]`
- `active_echelons`: which nodes are enabled for the difficulty
- `visibility_mask`: which nodes are visible (partial observability in `hard`)
- `inventory_levels`, `in_transit_qty`, `order_backlogs`, `customer_backlog`
- `demand_forecast`, `recent_customer_demand`
- `fill_rate`
- `disruption_link`: which link is disrupted (if any)
- `reward_terms`: breakdown of the reward components
- `action_bounds`: per‑echelon max order quantity
- `reward`, `done`, `metadata`

### Reward
The environment computes a reward that encourages **high service levels (fill rate)** while penalizing **holding**, **stockouts/backlogs**, and **transport/fixed order costs**. The returned reward is clipped to \([-1, 1]\).

## Advanced Usage

### Connecting to an Existing Server

If you already have the server running somewhere, just point the client at it:

```python
from hackathon.client import SupplyChainInventoryEnv

env = SupplyChainInventoryEnv(base_url="<ENV_HTTP_URL_HERE>")

obs = env.reset(difficulty="medium", seed=123)
result = env.step({"order_quantities": [5.0, 0.0, 12.0]})
```

### Using the Context Manager

The lightweight client in this repo is stateless HTTP by default; for best performance across many sequential steps, use the OpenEnv WebSocket client (when running with `openenv-core`’s full runtime).

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/hackathon_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
hackathon/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # Supply-chain env client
├── models.py              # Action, observation, and state models
└── server/
    ├── __init__.py        # Server module exports
    ├── hackathon_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
