
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from service.models import AgentAction
from service.server.hackathon_environment import SupplyChainEnv

def test_stochastic_lead_times():
    env = SupplyChainEnv()
    obs = env.reset(difficulty="mvp")
    
    print(f"Initial Lead Times: {env._lead_times}")
    
    all_lead_times = [[], [], []]
    
    for i in range(20):
        action = AgentAction(order_quantities=[10.0, 10.0, 10.0])
        obs = env.step(action)
        
        for node_index in range(3):
            # Per-node pipelines are indexed by product; use product 0 for sampling ETAs
            per_product = env._pipelines[node_index][0]
            if per_product:
                newest_shipment = per_product[-1]
                all_lead_times[node_index].append(newest_shipment.eta)
    
    for i, lts in enumerate(all_lead_times):
        unique_lts = set(lts)
        print(f"Node {i} (Base LT {env._lead_times[i]}) unique ETAs: {unique_lts}")
        if len(unique_lts) > 1:
            print(f"[PASS] Node {i} shows stochastic lead times.")
        else:
            print(f"[FAIL] Node {i} shows FIXED lead times.")

if __name__ == "__main__":
    test_stochastic_lead_times()
