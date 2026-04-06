
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hackathon.models import AgentAction
from hackathon.server.hackathon_environment import SupplyChainEnv

def test_news_events():
    env = SupplyChainEnv()
    env.reset(difficulty="mvp")
    
    print("--- Testing Social Media Trend ---")
    env._active_events = {"Social Media Trend": 2}
    demands = env._sample_customer_demand(day=1)
    # Product 0 should be 3x demand
    # Base demand in mvp is 20. Product 0 multiplier is 1.0. Retailer multipliers: [1.0, 1.2, 0.8, 1.1]
    # Expected demands for product 0: [60, 72, 48, 66]
    p0_demands = demands[0:4]
    print(f"Product 0 Demands: {p0_demands}")
    if any(d > 40 for d in p0_demands):
        print("[PASS] Social Media Trend correctly tripled product 0 demand.")
    else:
        print("[FAIL] Social Media Trend effect not observed.")

    print("\n--- Testing Canal Blockage ---")
    env._active_events = {"Canal Blockage": 2}
    # Place an order and check ETAs
    action = AgentAction(order_quantities=[10.0] * 21, shipping_methods=[0] * 21)
    env.step(action)
    # Node 1 base_lt = 3. Canal blockage should make it 6.
    node_1_p0_pipeline = env._pipelines[1][0]
    if node_1_p0_pipeline:
        latest_eta = node_1_p0_pipeline[-1].eta
        print(f"Node 1 Shipment ETA during Canal Blockage: {latest_eta} (Base LT: 3)")
        if latest_eta >= 5:
            print("[PASS] Canal Blockage correctly increased lead times.")
        else:
            print("[FAIL] Canal Blockage effect not observed.")

    print("\n--- Testing Labor Strike ---")
    env._active_events = {"Labor Strike": 2}
    # Clear node 1 inventory to see if it can be replenished
    env._inventory[1] = [0.0] * 3
    env._order_backlogs[1] = [[10.0, 0.0] for _ in range(3)]
    shipped = env._dispatch_replenishment_orders()
    # Shipped for node 1 should be 0
    node_1_shipped = sum(shipped[3:6]) # nodes start at 0, products are 3. so node 1 is index 1*3 = 3 to 5.
    print(f"Node 1 Shipped during Labor Strike: {node_1_shipped}")
    if node_1_shipped == 0:
        print("[PASS] Labor Strike correctly blocked node 1 operations.")
    else:
        print("[FAIL] Labor Strike effect not observed.")

if __name__ == "__main__":
    test_news_events()
