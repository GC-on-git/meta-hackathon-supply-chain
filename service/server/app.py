# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the supply chain environment.

This module creates an HTTP server that exposes SupplyChainEnv
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn service.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m service.server.app
"""

from service._compat import create_app
from service.models import AgentAction, AgentObservation
from service.server.hackathon_environment import SupplyChainEnv

app = create_app(
    SupplyChainEnv,
    AgentAction,
    AgentObservation,
    env_name="service",
)

@app.get("/")
def landing_page():
    return {"api_name": "Supply Chain Environment Server"}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
