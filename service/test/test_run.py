import sys
import os
import asyncio

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from service.client import SupplyChainClient
from service.models import AgentAction

async def main():
    # Connect to your running server
    env = SupplyChainClient(base_url="http://localhost:8000")

    # 1. Reset the game
    print("--- Resetting Environment ---")
    obs_raw = env.reset(difficulty="mvp")
    if asyncio.iscoroutine(obs_raw):
        obs_raw = await obs_raw
        
    # Handle both StepResult (object with .observation) and direct Observation
    obs = obs_raw.observation if hasattr(obs_raw, 'observation') else obs_raw
    print(f"Day: {obs.day}, Inventory Nodes: {len(obs.inventory_levels)}")

    # 2. Take a step (Order 10 units of all 3 products at all 7 nodes)
    print("\n--- Taking a Step with Express Shipping ---")
    action = AgentAction(
        order_quantities=[10.0] * 21,
        shipping_methods=[1] * 21  # All Express
    )
    result_raw = env.step(action)
    if asyncio.iscoroutine(result_raw):
        result_raw = await result_raw

    # Support both StepResult and direct result
    result = result_raw.observation if hasattr(result_raw, 'observation') else result_raw
    reward = result_raw.reward if hasattr(result_raw, 'reward') else getattr(result_raw, 'reward', 0.0)

    print(f"New Day: {result.day}")
    print(f"Reward: {reward}")
    print(f"Carbon Footprint: {result.carbon_footprint}")
    print(f"Active Events: {result.active_events}")
    print(f"Fill Rate: {result.fill_rate}")

if __name__ == "__main__":
    asyncio.run(main())
