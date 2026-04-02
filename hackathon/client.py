# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hackathon Environment Client."""
from __future__ import annotations

from typing import Any, Dict

from ._compat import EnvClient, StepResult
from .models import HackathonAction, HackathonObservation, HackathonState


class HackathonEnv(EnvClient[HackathonAction, HackathonObservation, HackathonState]):
    def _step_payload(self, action: HackathonAction) -> Dict[str, Any]:
        return {"order_quantities": action.order_quantities}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[HackathonObservation]:
        obs_data = payload.get("observation", {})
        observation = HackathonObservation(
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
            customer_backlog=obs_data.get("customer_backlog", 0.0),
            recent_customer_demand=obs_data.get("recent_customer_demand", []),
            fill_rate=obs_data.get("fill_rate", 0.0),
            disruption_link=obs_data.get("disruption_link"),
            regime=obs_data.get("regime", "baseline"),
            reward_terms=obs_data.get("reward_terms", {}),
            action_bounds=obs_data.get("action_bounds", []),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs_data.get("reward")),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> HackathonState:
        return HackathonState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            seed=payload.get("seed", 0),
            difficulty=payload.get("difficulty", "mvp"),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            total_cost=payload.get("total_cost", 0.0),
            total_demand=payload.get("total_demand", 0.0),
            total_served=payload.get("total_served", 0.0),
            fill_rate=payload.get("fill_rate", 0.0),
            customer_backlog=payload.get("customer_backlog", 0.0),
            regime=payload.get("regime", "baseline"),
            last_disruption_link=payload.get("last_disruption_link"),
            termination_reason=payload.get("termination_reason", "not_started"),
        )


# Backwards-compatible alias (older client name from the original env).
SupplyChainInventoryEnv = HackathonEnv

__all__ = ["HackathonEnv", "SupplyChainInventoryEnv"]
