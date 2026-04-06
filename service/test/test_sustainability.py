
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from service.models import AgentAction
from service.server.hackathon_environment import SupplyChainEnv

def test_sustainability():
    env = SupplyChainEnv()
    obs = env.reset(difficulty="mvp")
    
    print(f"Initial Carbon Footprint: {obs.carbon_footprint}")
    
    # Step 1: Order using Standard Method (0)
    orders_std = [10.0] * 21
    methods_std = [0] * 21
    action_std = AgentAction(order_quantities=orders_std, shipping_methods=methods_std)
    obs = env.step(action_std)
    
    carbon_after_std = obs.carbon_footprint
    print(f"Carbon after Standard ordering: {carbon_after_std}")
    # Note: Carbon is only added when shipped, which happens in the same step as registration in this env.
    
    # Step 2: Order using Express Method (1)
    orders_exp = [10.0] * 21
    methods_exp = [1] * 21
    action_exp = AgentAction(order_quantities=orders_exp, shipping_methods=methods_exp)
    obs = env.step(action_exp)
    
    carbon_after_exp = obs.carbon_footprint
    increase_exp = carbon_after_exp - carbon_after_std
    print(f"Carbon after Express ordering: {carbon_after_exp} (Increase: {increase_exp})")
    
    if increase_exp > (carbon_after_std * 4): # Express is 5.0 vs Standard 1.0
        print("[PASS] Carbon footprint correctly shows higher impact for Express shipping.")
    else:
        print(f"[FAIL] Carbon increase {increase_exp} is not significantly higher than Standard.")

    # Check if Express shipments have shorter ETA in pipelines
    # Pipelines are updated in _dispatch which is called in step()
    # Let's check a pipeline for a node with base_lt > 2
    # Node 1 has base_lt = 3. Express LT should be 1 or 2.
    node_1_p0_pipeline = env._pipelines[1][0]
    if node_1_p0_pipeline:
        latest_shipment = node_1_p0_pipeline[-1]
        print(f"Latest Shipment to Node 1 ETA: {latest_shipment.eta} (Method: {latest_shipment.method})")
        if latest_shipment.method == 1 and latest_shipment.eta < 3:
            print("[PASS] Express shipment has shorter ETA.")
        elif latest_shipment.method == 0:
            print("Wait, checking the right shipment...")

if __name__ == "__main__":
    test_sustainability()
