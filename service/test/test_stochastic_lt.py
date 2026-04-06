
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from service.models import AgentAction
from service.hackathon_environment import SupplyChainEnv

def test_stochastic_lead_times():
    env = SupplyChainEnv()
    obs = env.reset(difficulty="mvp")

    num_slots = env._num_nodes * env._num_products  # 21
    all_lead_times: dict[int, list[int]] = {n: [] for n in range(env._num_nodes)}

    for _ in range(20):
        action = AgentAction(order_quantities=[10.0] * num_slots, shipping_methods=[0] * num_slots)
        obs = env.step(action)
        for node_idx in range(env._num_nodes):
            pipeline = env._pipelines[node_idx][0]
            if pipeline:
                all_lead_times[node_idx].append(pipeline[-1].eta)

    nodes_with_lt = [n for n in range(env._num_nodes) if env._lead_times[n] > 0]
    assert nodes_with_lt, "expected at least one node with positive base lead time"
    for n in nodes_with_lt:
        assert len(set(all_lead_times[n])) > 1, f"node {n} shows fixed lead times"

if __name__ == "__main__":
    test_stochastic_lead_times()
