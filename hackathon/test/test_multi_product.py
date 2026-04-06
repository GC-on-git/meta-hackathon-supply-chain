
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hackathon.models import AgentAction
from hackathon.server.hackathon_environment import SupplyChainEnv

def test_multi_product():
    env = SupplyChainEnv()
    obs = env.reset(difficulty="mvp")
    
    print(f"Num Products: {env._num_products}")
    print(f"Inventory Levels (flattened): {obs.inventory_levels}")
    print(f"Initial Backlog (flattened): {obs.customer_backlog}")
    
    if len(obs.inventory_levels) == 9:
        print("[PASS] Inventory levels have length 9 (3 echelons * 3 products).")
    else:
        print(f"[FAIL] Inventory levels have length {len(obs.inventory_levels)}, expected 9.")

    if len(obs.customer_backlog) == 3:
        print("[PASS] Customer backlog has length 3 (1 per product).")
    else:
        print(f"[FAIL] Customer backlog has length {len(obs.customer_backlog)}, expected 3.")

    # Take a step with 9 order quantities
    orders = [5.0] * 9
    action = AgentAction(order_quantities=orders)
    obs = env.step(action)
    
    print(f"New Day: {obs.day}")
    print(f"Reward: {obs.reward}")
    print(f"Current Demands (from metadata): {obs.metadata.get('current_demands')}")
    
    if obs.metadata.get("current_demands") and len(obs.metadata.get("current_demands")) == 3:
        print("[PASS] Current demands in metadata has length 3.")
    else:
        print("[FAIL] Current demands in metadata is missing or has wrong length.")

    if len(obs.state_vector) == 54: # 3 echelons * 3 products * 6 features
        print("[PASS] State vector has length 54.")
    else:
        print(f"[FAIL] State vector has length {len(obs.state_vector)}, expected 54.")

if __name__ == "__main__":
    test_multi_product()
