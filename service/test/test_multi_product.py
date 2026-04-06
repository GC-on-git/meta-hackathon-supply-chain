
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from service.models import AgentAction
from service.hackathon_environment import SupplyChainEnv

def test_multi_product():
    env = SupplyChainEnv()
    obs = env.reset(difficulty="mvp")

    num_nodes = env._num_nodes       # 7
    num_products = env._num_products  # 3
    num_slots = num_nodes * num_products  # 21

    assert len(obs.inventory_levels) == num_slots
    assert len(obs.customer_backlog) == 4 * num_products  # 4 retailers * 3 products

    action = AgentAction(order_quantities=[5.0] * num_slots, shipping_methods=[0] * num_slots)
    obs = env.step(action)

    assert obs.day == 1
    assert obs.reward is not None
    demands = obs.metadata.get("current_demands")
    assert demands is not None and len(demands) == 4 * num_products
    assert len(obs.state_vector) == num_slots * 6

if __name__ == "__main__":
    test_multi_product()
