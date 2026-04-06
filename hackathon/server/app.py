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

from fastapi.responses import HTMLResponse

from hackathon._compat import create_app
from hackathon.models import AgentAction, AgentObservation
from hackathon.server.hackathon_environment import SupplyChainEnv

app = create_app(
    SupplyChainEnv,
    AgentAction,
    AgentObservation,
    env_name="hackathon",
)

@app.get("/", response_class=HTMLResponse)
def landing_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Supply Chain Env - Live API Stats</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            async function fetchStats() {
                try {
                    const response = await fetch('/state');
                    if (response.ok) {
                        const data = await response.json();
                        document.getElementById('step-count').innerText = data.step_count || 0;
                        document.getElementById('fill-rate').innerText = ((data.fill_rate || 0) * 100).toFixed(2) + '%';
                        document.getElementById('total-cost').innerText = '$' + (data.total_cost || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                        document.getElementById('carbon').innerText = (data.carbon_footprint || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' kg';
                        document.getElementById('reward').innerText = (data.cumulative_reward || 0).toFixed(4);
                        document.getElementById('difficulty').innerText = data.difficulty || 'N/A';
                        
                        const eventsList = document.getElementById('active-events');
                        eventsList.innerHTML = '';
                        if (data.active_events && data.active_events.length > 0) {
                            data.active_events.forEach(event => {
                                const li = document.createElement('li');
                                li.className = "px-3 py-1.5 bg-red-100 text-red-800 rounded-md text-sm font-bold shadow-sm";
                                li.innerText = event;
                                eventsList.appendChild(li);
                            });
                        } else {
                            eventsList.innerHTML = '<span class="text-gray-500 italic">None (Normal conditions)</span>';
                        }
                    }
                } catch (error) {
                    console.error('Error fetching state:', error);
                }
            }
            setInterval(fetchStats, 2000);
            window.onload = fetchStats;
        </script>
    </head>
    <body class="bg-gray-50 min-h-screen text-gray-800 font-sans">
        <div class="max-w-5xl mx-auto px-4 py-12">
            <div class="text-center mb-12">
                <h1 class="text-5xl font-extrabold text-blue-600 mb-4 tracking-tight">Supply Chain Simulator</h1>
                <p class="text-xl text-gray-600 font-medium">Real-time simulation statistics from the latest active episode.</p>
                <div class="mt-8 flex justify-center gap-4">
                    <a href="/docs" class="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition shadow hover:shadow-md">API Documentation</a>
                    <a href="/health" class="px-6 py-2.5 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 rounded-lg font-semibold transition shadow-sm hover:shadow">Health Check</a>
                </div>
            </div>
            
            <div class="bg-white rounded-2xl shadow-md border border-gray-200 p-8">
                <h2 class="text-2xl font-bold text-gray-800 mb-8 flex items-center justify-between">
                    Live Dashboard
                    <span class="flex h-3.5 w-3.5 relative">
                      <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                      <span class="relative inline-flex rounded-full h-3.5 w-3.5 bg-green-500"></span>
                    </span>
                </h2>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <div class="bg-gray-50 rounded-xl p-6 border border-gray-100 shadow-inner">
                        <div class="text-sm font-semibold text-gray-500 mb-1 uppercase tracking-wider">Current Step (Day)</div>
                        <div id="step-count" class="text-4xl font-black text-gray-900">0</div>
                    </div>
                    <div class="bg-indigo-50 rounded-xl p-6 border border-indigo-100 shadow-inner">
                        <div class="text-sm font-semibold text-indigo-500 mb-1 uppercase tracking-wider">Difficulty</div>
                        <div id="difficulty" class="text-4xl font-black text-indigo-700 capitalize">N/A</div>
                    </div>
                    <div class="bg-green-50 rounded-xl p-6 border border-green-100 shadow-inner">
                        <div class="text-sm font-semibold text-green-600 mb-1 uppercase tracking-wider">Fill Rate</div>
                        <div id="fill-rate" class="text-4xl font-black text-green-700">0.00%</div>
                    </div>
                    <div class="bg-red-50 rounded-xl p-6 border border-red-100 shadow-inner">
                        <div class="text-sm font-semibold text-red-500 mb-1 uppercase tracking-wider">Total Cost</div>
                        <div id="total-cost" class="text-4xl font-black text-red-700">$0.00</div>
                    </div>
                    <div class="bg-orange-50 rounded-xl p-6 border border-orange-100 shadow-inner">
                        <div class="text-sm font-semibold text-orange-600 mb-1 uppercase tracking-wider">Carbon Footprint</div>
                        <div id="carbon" class="text-4xl font-black text-orange-700">0.00 kg</div>
                    </div>
                    <div class="bg-purple-50 rounded-xl p-6 border border-purple-100 shadow-inner">
                        <div class="text-sm font-semibold text-purple-500 mb-1 uppercase tracking-wider">Cumulative Reward</div>
                        <div id="reward" class="text-4xl font-black text-purple-700">0.00</div>
                    </div>
                </div>
                
                <div class="border-t border-gray-100 pt-6 mt-4">
                    <h3 class="text-lg font-bold text-gray-800 mb-4">Active Events / Disruptions</h3>
                    <ul id="active-events" class="flex flex-wrap gap-3">
                        <span class="text-gray-500 italic">Loading...</span>
                    </ul>
                </div>
            </div>
            
            <div class="text-center mt-12 text-gray-400 text-sm">
                Powered by FastAPI and OpenEnv &bull; Hosted on Hugging Face Spaces
            </div>
        </div>
    </body>
    </html>
    """


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
