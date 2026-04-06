# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the supply chain environment.

Run from repo root (PYTHONPATH includes `.`):

    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

Or use the top-level shim: ``uvicorn app:app`` (see repo-root ``app.py``).
"""

from openenv.core.env_server.http_server import create_app

from service.hackathon_environment import SupplyChainEnv
from service.models import AgentAction, AgentObservation

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
