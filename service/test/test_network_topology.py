
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from service.models import AgentAction
from service.hackathon_environment import SupplyChainEnv

def test_network_topology():
    env = SupplyChainEnv()
    obs = env.reset(difficulty="mvp")
    
    print(f"Num Nodes: {env._num_nodes}")
    print(f"Inventory Levels (flattened): {len(obs.inventory_levels)}")
    
    if len(obs.inventory_levels) == 21: # 7 nodes * 3 products
        print("[PASS] Inventory levels have length 21.")
    else:
        print(f"[FAIL] Inventory levels have length {len(obs.inventory_levels)}, expected 21.")

    if len(obs.customer_backlog) == 12: # 4 retailers * 3 products
        print("[PASS] Customer backlog has length 12.")
    else:
        print(f"[FAIL] Customer backlog has length {len(obs.customer_backlog)}, expected 12.")

    # Take a step with 21 order quantities
    orders = [10.0] * 21
    action = AgentAction(order_quantities=orders)
    obs = env.step(action)
    
    print(f"New Day: {obs.day}")
    print(f"Reward: {obs.reward}")
    print(f"Current Demands (from metadata): {len(obs.metadata.get('current_demands'))}")
    
    if len(obs.metadata.get("current_demands")) == 12:
        print("[PASS] Current demands in metadata has length 12.")
    else:
        print(f"[FAIL] Current demands in metadata has length {len(obs.metadata.get('current_demands'))}, expected 12.")

    if len(obs.state_vector) == 126: # 7 nodes * 3 products * 6 features
        print(f"[PASS] State vector has length {len(obs.state_vector)}.")
    else:
        print(f"[FAIL] State vector has length {len(obs.state_vector)}, expected 126.")

    # Check upstream source mapping
    tests = [
        (0, None),
        (1, 0),
        (2, 0),
        (3, 1),
        (4, 1),
        (5, 2),
        (6, 2)
    ]
    for node, expected in tests:
        src = env._upstream_source(node)
        if src == expected:
            print(f"[PASS] Node {node} upstream source is {src} (Expected: {expected})")
        else:
            print(f"[FAIL] Node {node} upstream source is {src} (Expected: {expected})")

if __name__ == "__main__":
    test_network_topology()
