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
    uvicorn hackathon.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m hackathon.server.app
"""

from hackathon._compat import create_app
from hackathon.models import AgentAction, AgentObservation
from hackathon.server.hackathon_environment import SupplyChainEnv
from fastapi import Request
from fastapi.responses import HTMLResponse
import time

app = create_app(
    SupplyChainEnv,
    AgentAction,
    AgentObservation,
    env_name="hackathon",
)

stats = {
    "start_time": time.time(),
    "total_requests": 0,
    "endpoints": {}
}

@app.middleware("http")
async def track_stats(request: Request, call_next):
    stats["total_requests"] += 1
    path = request.url.path
    stats["endpoints"][path] = stats["endpoints"].get(path, 0) + 1
    response = await call_next(request)
    return response

@app.get("/", response_class=HTMLResponse)
async def landing_page():
    uptime = time.time() - stats["start_time"]
    
    endpoints_html = "".join(
        f'<div class="stat-row"><span class="label">{k}</span><span class="value">{v}</span></div>' 
        for k, v in sorted(stats['endpoints'].items())
    )
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Supply Chain API Server</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; background-color: #f4f4f9; }}
                h1 {{ color: #2c3e50; text-align: center; }}
                .stats-container {{ background: #fff; padding: 30px; border-radius: 12px; max-width: 600px; margin: 0 auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .stat-row {{ display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid #dee2e6; }}
                .stat-row:last-child {{ border-bottom: none; }}
                .label {{ font-weight: bold; color: #495057; }}
                .value {{ color: #007bff; font-weight: bold; }}
                .footer {{ text-align: center; margin-top: 30px; }}
                .btn {{ display: inline-block; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 6px; font-weight: bold; transition: background 0.3s; }}
                .btn:hover {{ background-color: #0056b3; }}
            </style>
        </head>
        <body>
            <h1>🚀 Supply Chain Environment</h1>
            <div class="stats-container">
                <div class="stat-row">
                    <span class="label">Server Status</span>
                    <span class="value" style="color: #28a745;">Online</span>
                </div>
                <div class="stat-row">
                    <span class="label">Uptime</span>
                    <span class="value">{uptime:.1f} seconds</span>
                </div>
                <div class="stat-row">
                    <span class="label">Total API Hits</span>
                    <span class="value">{stats['total_requests']}</span>
                </div>
                <h3 style="margin-top: 25px; color: #2c3e50; border-bottom: 2px solid #007bff; padding-bottom: 5px; display: inline-block;">Endpoints Usage</h3>
                <div style="margin-top: 10px;">
                    {endpoints_html}
                </div>
            </div>
            <div class="footer">
                <a href="/docs" class="btn">📚 View API Documentation</a>
            </div>
        </body>
    </html>
    """
    return html_content


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
