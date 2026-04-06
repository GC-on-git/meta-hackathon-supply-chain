
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from service.models import AgentAction
from service.server.hackathon_environment import SupplyChainEnv

def test_variable_costs():
    env = SupplyChainEnv()
    obs = env.reset(difficulty="mvp")
    
    multipliers = []
    transport_costs = []
    
    for i in range(20):
        # Always order the same amount to see cost variation
        action = AgentAction(order_quantities=[10.0, 10.0, 10.0])
        obs = env.step(action)
        
        multipliers.append(env._fuel_price_multiplier)
        transport_costs.append(obs.reward_terms["transport_cost"])
        
        print(f"Day {i}: Multiplier={env._fuel_price_multiplier:.4f}, Transport Cost={obs.reward_terms['transport_cost']}")

    unique_multipliers = set(multipliers)
    print(f"\nUnique Multipliers: {len(unique_multipliers)} / 20")
    
    if len(unique_multipliers) > 1:
        print("[PASS] Fuel price multiplier is fluctuating.")
    else:
        print("[FAIL] Fuel price multiplier is FIXED.")
        
    unique_costs = set(transport_costs)
    if len(unique_costs) > 1:
        print("[PASS] Transport costs are varying.")
    else:
        print("[FAIL] Transport costs are CONSTANT.")

if __name__ == "__main__":
    test_variable_costs()
