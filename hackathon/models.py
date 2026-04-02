# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Hackathon Environment.

The hackathon environment is a simple test environment that echoes back messages.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import Field, field_validator

from ._compat import Action, Observation, State


DifficultyName = Literal["easy", "medium", "mvp", "hard"]


class HackathonAction(Action):
    order_quantities: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)

    @field_validator("order_quantities")
    @classmethod
    def validate_quantities(cls, value: List[float]) -> List[float]:
        if any(quantity < 0 for quantity in value):
            raise ValueError("order quantities must be non-negative")
        return value


class HackathonObservation(Observation):
    day: int = 0
    horizon: int = 365
    difficulty: DifficultyName = "mvp"
    echelon_names: List[str] = Field(default_factory=lambda: ["manufacturer", "warehouse", "retailer"])
    active_echelons: List[bool] = Field(default_factory=lambda: [True, True, True], min_length=3, max_length=3)
    visibility_mask: List[bool] = Field(default_factory=lambda: [True, True, True], min_length=3, max_length=3)
    state_vector: List[float] = Field(default_factory=list, min_length=18, max_length=18)
    inventory_levels: List[float] = Field(default_factory=list, min_length=3, max_length=3)
    in_transit_qty: List[float] = Field(default_factory=list, min_length=3, max_length=3)
    demand_forecast: List[float] = Field(default_factory=list, min_length=3, max_length=3)
    days_since_last_order: List[float] = Field(default_factory=list, min_length=3, max_length=3)
    holding_cost_rate: List[float] = Field(default_factory=list, min_length=3, max_length=3)
    lead_time_remaining: List[float] = Field(default_factory=list, min_length=3, max_length=3)
    order_backlogs: List[float] = Field(default_factory=list, min_length=3, max_length=3)
    customer_backlog: float = 0.0
    recent_customer_demand: List[float] = Field(default_factory=list)
    fill_rate: float = 0.0
    disruption_link: Optional[str] = None
    regime: str = "baseline"
    reward_terms: Dict[str, float] = Field(default_factory=dict)
    action_bounds: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)


class HackathonState(State):
    seed: int = 0
    difficulty: DifficultyName = "mvp"
    cumulative_reward: float = 0.0
    total_cost: float = 0.0
    total_demand: float = 0.0
    total_served: float = 0.0
    fill_rate: float = 0.0
    customer_backlog: float = 0.0
    regime: str = "baseline"
    last_disruption_link: Optional[str] = None
    termination_reason: str = "not_started"


# Backwards-compatible aliases (older name from the original env).
SupplyChainAction = HackathonAction
SupplyChainObservation = HackathonObservation
SupplyChainState = HackathonState

__all__ = [
    "DifficultyName",
    "HackathonAction",
    "HackathonObservation",
    "HackathonState",
    "SupplyChainAction",
    "SupplyChainObservation",
    "SupplyChainState",
]
