# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HTTP client for the supply chain environment (OpenEnv EnvClient)."""
from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import AgentAction, AgentObservation, SupplyChainState


class SupplyChainClient(EnvClient[AgentAction, AgentObservation, SupplyChainState]):
    def _step_payload(self, action: AgentAction) -> Dict[str, Any]:
        return {
            "order_quantities": action.order_quantities,
            "shipping_methods": action.shipping_methods,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AgentObservation]:
        obs_data = payload.get("observation", {})
        if "observation" in obs_data:
            obs_data = obs_data["observation"]
        observation = AgentObservation(
            day=obs_data.get("day", 0),
            horizon=obs_data.get("horizon", 365),
            difficulty=obs_data.get("difficulty", "mvp"),
            echelon_names=obs_data.get("echelon_names", []),
            active_echelons=obs_data.get("active_echelons", []),
            visibility_mask=obs_data.get("visibility_mask", []),
            state_vector=obs_data.get("state_vector", []),
            inventory_levels=obs_data.get("inventory_levels", []),
            in_transit_qty=obs_data.get("in_transit_qty", []),
            demand_forecast=obs_data.get("demand_forecast", []),
            days_since_last_order=obs_data.get("days_since_last_order", []),
            holding_cost_rate=obs_data.get("holding_cost_rate", []),
            lead_time_remaining=obs_data.get("lead_time_remaining", []),
            order_backlogs=obs_data.get("order_backlogs", []),
            customer_backlog=obs_data.get("customer_backlog", []),
            recent_customer_demand=obs_data.get("recent_customer_demand", []),
            carbon_footprint=obs_data.get("carbon_footprint", 0.0),
            fill_rate=obs_data.get("fill_rate", 0.0),
            disruption_link=obs_data.get("disruption_link"),
            regime=obs_data.get("regime", "baseline"),
            reward_terms=obs_data.get("reward_terms", {}),
            action_bounds=obs_data.get("action_bounds", []),
            active_events=obs_data.get("active_events", []),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs_data.get("reward")),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SupplyChainState:
        return SupplyChainState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            seed=payload.get("seed", 0),
            difficulty=payload.get("difficulty", "mvp"),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            total_cost=payload.get("total_cost", 0.0),
            total_demand=payload.get("total_demand", 0.0),
            total_served=payload.get("total_served", 0.0),
            fill_rate=payload.get("fill_rate", 0.0),
            customer_backlog=payload.get("customer_backlog", []),
            carbon_footprint=payload.get("carbon_footprint", 0.0),
            regime=payload.get("regime", "baseline"),
            last_disruption_link=payload.get("last_disruption_link"),
            termination_reason=payload.get("termination_reason", "not_started"),
            active_events=payload.get("active_events", []),
        )


# Backwards-compatible alias (older client name from the original env).
SupplyChainInventoryEnv = SupplyChainClient

__all__ = ["SupplyChainClient", "SupplyChainInventoryEnv"]
