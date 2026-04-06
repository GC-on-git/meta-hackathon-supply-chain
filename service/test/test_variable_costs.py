
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from service.models import AgentAction
from service.hackathon_environment import SupplyChainEnv

def test_variable_costs():
    env = SupplyChainEnv()
    obs = env.reset(difficulty="mvp")

    num_slots = env._num_nodes * env._num_products
    multipliers = []
    transport_costs = []

    for _ in range(20):
        action = AgentAction(order_quantities=[10.0] * num_slots, shipping_methods=[0] * num_slots)
        obs = env.step(action)
        multipliers.append(env._fuel_price_multiplier)
        transport_costs.append(obs.reward_terms["transport_cost"])

    assert len(set(multipliers)) > 1, "fuel price multiplier should fluctuate"
    assert len(set(transport_costs)) > 1, "transport costs should vary with fuel price"

if __name__ == "__main__":
    test_variable_costs()
