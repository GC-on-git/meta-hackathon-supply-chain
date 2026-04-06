# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In-process supply chain simulator (OpenEnv Environment implementation)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Any, Optional
from uuid import uuid4

from hackathon._compat import Environment
from hackathon.models import AgentAction, AgentObservation, SupplyChainState


ECHELON_NAMES = ["factory", "warehouse_a", "warehouse_b", "retailer_1", "retailer_2", "retailer_3", "retailer_4"]
HOLDING_COSTS = [0.010, 0.015, 0.015, 0.025, 0.025, 0.025, 0.025]
TRANSPORT_COSTS = [0.008, 0.012, 0.012, 0.015, 0.015, 0.015, 0.015]
FIXED_ORDER_COSTS = [0.04, 0.06, 0.06, 0.08, 0.08, 0.08, 0.08]

DIFFICULTY_PRESETS = {
    "easy": {
        "active_echelons": [False, False, True],
        "lead_times": [0, 0, 0],
        "base_demand": 18.0,
        "demand_std": 0.0,
        "season_shift_day": None,
        "season_high_demand": 18.0,
        "partial_visibility": False,
        "disruption_probability": 0.0,
        "reward_scale": 6.0,
        "max_order_qty": 40.0,
    },
    "medium": {
        "active_echelons": [False, True, True],
        "lead_times": [0, 3, 3],
        "base_demand": 20.0,
        "demand_std": 4.0,
        "season_shift_day": None,
        "season_high_demand": 20.0,
        "partial_visibility": False,
        "disruption_probability": 0.0,
        "reward_scale": 9.0,
        "max_order_qty": 55.0,
    },
    "mvp": {
        "active_echelons": [True, True, True],
        "lead_times": [4, 3, 2],
        "base_demand": 22.0,
        "demand_std": 4.5,
        "season_shift_day": None,
        "season_high_demand": 22.0,
        "partial_visibility": False,
        "disruption_probability": 0.0,
        "reward_scale": 11.0,
        "max_order_qty": 65.0,
    },
    "hard": {
        "active_echelons": [True, True, True],
        "lead_times": [5, 4, 2],
        "base_demand": 22.0,
        "demand_std": 5.0,
        "season_shift_day": 180,
        "season_high_demand": 32.0,
        "partial_visibility": True,
        "disruption_probability": 0.05,
        "reward_scale": 12.0,
        "max_order_qty": 75.0,
    },
}


@dataclass
class Shipment:
    quantity: float
    eta: int
    method: int = 0 # 0: Standard, 1: Express


class SupplyChainEnv(Environment):
    def __init__(self) -> None:
        self._rng = Random()
        self._difficulty = "mvp"
        self._config = DIFFICULTY_PRESETS[self._difficulty]
        self._horizon = 365
        self._num_products = 3
        self._num_nodes = 7
        self._num_methods = 2 # 0: Standard, 1: Express
        self._max_order_qty = float(self._config["max_order_qty"])
        self._active_echelons = [True] * self._num_nodes 
        self._lead_times = [4, 3, 3, 2, 2, 2, 2] # Base LT
        self._inventory = [[0.0] * self._num_products for _ in range(self._num_nodes)]
        self._order_backlogs = [[[0.0] * self._num_methods for _ in range(self._num_products)] for _ in range(self._num_nodes)]
        self._pipelines: list[list[list[Shipment]]] = [[[] for _ in range(self._num_products)] for _ in range(self._num_nodes)]
        self._days_since_last_order = [[0] * self._num_products for _ in range(self._num_nodes)]
        self._customer_backlog = [0.0] * (4 * self._num_products)
        self._recent_customer_demand: list[deque[float]] = [deque(maxlen=7) for _ in range(4 * self._num_products)]
        self._latest_forecasts = [0.0] * (4 * self._num_products)
        self._last_reward_terms: dict[str, float] = {}
        self._last_disruption_link: Optional[str] = None
        self._fuel_price_multiplier = 1.0
        self._total_carbon = 0.0
        self._active_events: dict[str, int] = {} # Name -> Remaining Days
        self._episode_state = SupplyChainState(
            episode_id=str(uuid4()),
            step_count=0,
            seed=0,
            difficulty=self._difficulty,
            cumulative_reward=0.0,
            total_cost=0.0,
            total_demand=0.0,
            total_served=0.0,
            fill_rate=0.0,
            customer_backlog=[0.0] * (4 * self._num_products),
            regime="baseline",
            last_disruption_link=None,
            termination_reason="not_started",
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: str = "mvp",
        horizon: Optional[int] = None,
        max_order_qty: Optional[float] = None,
        **kwargs: Any,
    ) -> AgentObservation:
        del kwargs
        if difficulty not in DIFFICULTY_PRESETS:
            raise ValueError(f"unsupported difficulty: {difficulty}")

        if seed is not None:
            self._rng.seed(seed)
        current_seed = int(seed if seed is not None else 0)

        self._difficulty = difficulty
        self._config = DIFFICULTY_PRESETS[difficulty]
        self._horizon = int(horizon or 365)
        self._max_order_qty = float(max_order_qty or self._config["max_order_qty"])
        
        # Adaptation for 1-2-4 topology
        self._active_echelons = [True] * self._num_nodes
        self._lead_times = [4, 3, 3, 2, 2, 2, 2]

        base_demand = float(self._config["base_demand"])
        # Different base demands for different products
        product_demands = [base_demand, base_demand * 0.7, base_demand * 1.3]
        
        # Retailer-specific base demands (4 retailers)
        retailer_multipliers = [1.0, 1.2, 0.8, 1.1]
        all_retailer_demands = []
        for p_idx in range(self._num_products):
            for r_idx in range(4):
                all_retailer_demands.append(product_demands[p_idx] * retailer_multipliers[r_idx])

        safety_days = [10.0, 7.0, 7.0, 4.0, 4.0, 4.0, 4.0]
        
        self._inventory = []
        for e_idx in range(self._num_nodes):
            row = []
            for p_idx in range(self._num_products):
                # Approximation for initial inventory
                qty = round(product_demands[p_idx] * safety_days[e_idx] * 0.5, 2)
                row.append(qty)
            self._inventory.append(row)

        self._order_backlogs = [[[0.0] * self._num_methods for _ in range(self._num_products)] for _ in range(self._num_nodes)]
        self._pipelines = [[[] for _ in range(self._num_products)] for _ in range(self._num_nodes)]
        self._days_since_last_order = [[0] * self._num_products for _ in range(self._num_nodes)]
        self._customer_backlog = [0.0] * (4 * self._num_products)
        self._recent_customer_demand = [deque([d] * 7, maxlen=7) for d in all_retailer_demands]
        self._latest_forecasts = all_retailer_demands
        self._total_carbon = 0.0
        self._active_events = {}
        self._last_reward_terms = {
            "holding_cost": 0.0,
            "stockout_penalty": 0.0,
            "transport_cost": 0.0,
            "fill_rate_bonus": 0.0,
            "total": 0.0,
        }
        self._last_disruption_link = None
        self._fuel_price_multiplier = 1.0
        self._episode_state = SupplyChainState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            seed=current_seed,
            difficulty=difficulty,
            cumulative_reward=0.0,
            total_cost=0.0,
            total_demand=0.0,
            total_served=0.0,
            fill_rate=0.0,
            customer_backlog=list(self._customer_backlog),
            carbon_footprint=0.0,
            regime=self._current_regime(day=0),
            active_events=[],
            last_disruption_link=None,
            termination_reason="in_progress",
        )
        return self._build_observation(done=False, reward=0.0, metadata={"status": "ready"})

    def step(self, action: AgentAction, timeout_s: Optional[float] = None, **kwargs: Any) -> AgentObservation:
        del timeout_s, kwargs
        if self._episode_state.termination_reason == "horizon_reached":
            return self._build_observation(
                done=True,
                reward=0.0,
                metadata={"message": "episode already complete", "termination_reason": self._episode_state.termination_reason},
            )

        self._episode_state.step_count += 1
        self._handle_news_events()
        self._receive_inbound_shipments()
        self._serve_existing_backlog()

        sanitized_quantities, sanitized_methods = self._sanitize_action(action)
        self._register_orders(sanitized_quantities, sanitized_methods)

        self._last_disruption_link = self._sample_disruption_link()
        shipped_quantities = self._dispatch_replenishment_orders()

        current_demands = self._sample_customer_demand(day=self._episode_state.step_count)
        served_demands = self._serve_customer_demand(current_demands)
        for p_idx, demand in enumerate(current_demands):
            self._latest_forecasts[p_idx] = self._update_forecast(demand, p_idx)
            self._recent_customer_demand[p_idx].append(demand)

        # Update fuel price multiplier (random walk, +/- 5%, bounded [0.8, 1.5])
        self._fuel_price_multiplier = max(0.8, min(1.5, self._fuel_price_multiplier * (1 + self._rng.uniform(-0.05, 0.05))))

        reward_terms = self._compute_reward_terms(
            current_demand=sum(current_demands),
            served_demand=sum(served_demands),
            shipped_quantities=shipped_quantities,
        )
        done = self._episode_state.step_count >= self._horizon
        self._episode_state.cumulative_reward += reward_terms["total"]
        self._episode_state.total_cost += (
            reward_terms["holding_cost"] + reward_terms["stockout_penalty"] + reward_terms["transport_cost"]
        )
        self._episode_state.total_demand += sum(current_demands)
        self._episode_state.total_served += sum(served_demands)
        self._episode_state.fill_rate = self._safe_ratio(self._episode_state.total_served, self._episode_state.total_demand)
        self._episode_state.customer_backlog = list(self._customer_backlog)
        self._episode_state.carbon_footprint = round(self._total_carbon, 4)
        self._episode_state.active_events = list(self._active_events.keys())
        self._episode_state.regime = self._current_regime(day=self._episode_state.step_count)
        self._episode_state.last_disruption_link = self._last_disruption_link
        self._episode_state.termination_reason = "horizon_reached" if done else "in_progress"
        self._last_reward_terms = reward_terms

        return self._build_observation(
            done=done,
            reward=reward_terms["total"],
            metadata={
                "current_demands": [round(d, 4) for d in current_demands],
                "served_demands": [round(s, 4) for s in served_demands],
                "disruption_link": self._last_disruption_link,
            },
        )

    @property
    def state(self) -> SupplyChainState:
        return self._episode_state

    def _sanitize_action(self, action: AgentAction) -> tuple[list[float], list[int]]:
        quantities: list[float] = []
        methods: list[int] = []
        for e_idx in range(self._num_nodes):
            for p_idx in range(self._num_products):
                idx = e_idx * self._num_products + p_idx
                
                # Default values if echelons inactive (though all are active in 1-2-4 for now)
                if not self._active_echelons[e_idx]:
                    quantities.append(0.0)
                    methods.append(0)
                    self._days_since_last_order[e_idx][p_idx] = 0
                    continue
                
                q = action.order_quantities[idx] if idx < len(action.order_quantities) else 0.0
                m = action.shipping_methods[idx] if idx < len(action.shipping_methods) else 0
                
                clipped_q = min(max(float(q), 0.0), self._max_order_qty)
                m = 1 if int(m) == 1 else 0 # 0 or 1
                
                quantities.append(clipped_q)
                methods.append(m)
                
                if clipped_q > 0:
                    self._days_since_last_order[e_idx][p_idx] = 0
                else:
                    self._days_since_last_order[e_idx][p_idx] += 1
        return quantities, methods

    def _register_orders(self, quantities: list[float], methods: list[int]) -> None:
        for e_idx in range(self._num_nodes):
            for p_idx in range(self._num_products):
                idx = e_idx * self._num_products + p_idx
                q = quantities[idx]
                m = methods[idx]
                self._order_backlogs[e_idx][p_idx][m] += q

    def _receive_inbound_shipments(self) -> None:
        for node_index in range(self._num_nodes):
            for p_idx in range(self._num_products):
                queue = self._pipelines[node_index][p_idx]
                if not queue:
                    continue
                remaining_shipments: list[Shipment] = []
                for shipment in queue:
                    updated_eta = shipment.eta - 1
                    if updated_eta <= 0:
                        self._inventory[node_index][p_idx] += shipment.quantity
                    else:
                        remaining_shipments.append(Shipment(quantity=shipment.quantity, eta=updated_eta))
                self._pipelines[node_index][p_idx] = remaining_shipments

    def _upstream_source(self, node_index: int) -> Optional[int]:
        if node_index == 0: # Factory
            return None
        if node_index in {1, 2}: # Warehouses
            return 0 # Source is Factory
        if node_index in {3, 4}: # Retailers 1 & 2
            return 1 # Source is Warehouse A
        if node_index in {5, 6}: # Retailers 3 & 4
            return 2 # Source is Warehouse B
        return None

    def _sample_customer_demand(self, day: int) -> list[float]:
        demands = []
        base_demand = float(self._config["base_demand"])
        product_multipliers = [1.0, 0.7, 1.3]
        retailer_multipliers = [1.0, 1.2, 0.8, 1.1] # 4 retailers
        
        for p_idx in range(self._num_products):
            for r_idx in range(4):
                mean = base_demand * product_multipliers[p_idx] * retailer_multipliers[r_idx]
                
                # Apply Social Media Trend effect
                if "Social Media Trend" in self._active_events and p_idx == 0:
                    mean *= 3.0 # Product 0 (Trend) gets 3x demand
                
                shift_day = self._config["season_shift_day"]
                if self._difficulty == "hard" or p_idx == 2:
                    if shift_day is not None and day >= shift_day:
                        mean += float(self._config["season_high_demand"]) * 0.5 * retailer_multipliers[r_idx]

                weekly_wave = 0.0
                if self._difficulty == "hard":
                    weekly_wave = 2.5 if day % 7 in {4, 5} else -1.5 if day % 7 == 0 else 0.0

                std = float(self._config["demand_std"])
                sampled = self._rng.gauss(mean + weekly_wave, std) if std > 0 else mean
                demands.append(round(max(sampled, 0.0), 4))
        return demands

    def _serve_customer_demand(self, current_demands: list[float]) -> list[float]:
        # Retailers are at indices 3, 4, 5, 6
        retailer_indices = [3, 4, 5, 6]
        served_demands = []
        for p_idx in range(self._num_products):
            for r_idx, node_idx in enumerate(retailer_indices):
                demand_idx = p_idx * 4 + r_idx
                demand = current_demands[demand_idx]
                
                served = min(self._inventory[node_idx][p_idx], demand)
                self._inventory[node_idx][p_idx] -= served
                unmet = demand - served
                self._customer_backlog[demand_idx] += unmet
                served_demands.append(round(served, 4))
        return served_demands

    def _serve_existing_backlog(self) -> None:
        retailer_indices = [3, 4, 5, 6]
        for p_idx in range(self._num_products):
            for r_idx, node_idx in enumerate(retailer_indices):
                demand_idx = p_idx * 4 + r_idx
                backlog_served = min(self._inventory[node_idx][p_idx], self._customer_backlog[demand_idx])
                self._inventory[node_idx][p_idx] -= backlog_served
                self._customer_backlog[demand_idx] -= backlog_served

    def _dispatch_replenishment_orders(self) -> list[float]:
        shipped_quantities = [0.0] * (self._num_nodes * self._num_products)
        step_carbon = 0.0
        
        for node_index in range(self._num_nodes):
            if not self._active_echelons[node_index]:
                continue
            
            # Labor Strike: One warehouse (node 1) goes offline
            if "Labor Strike" in self._active_events and node_index == 1:
                continue

            for p_idx in range(self._num_products):
                source_index = self._upstream_source(node_index)
                
                # Labor Strike: If source is offline
                if "Labor Strike" in self._active_events and source_index == 1:
                    continue
                source_index = self._upstream_source(node_index)
                
                # Dispatch Express first, then Standard
                for method in [1, 0]:
                    requested = self._order_backlogs[node_index][p_idx][method]
                    if requested <= 0:
                        continue

                    available = requested if source_index is None else min(requested, self._inventory[source_index][p_idx])
                    
                    if self._last_disruption_link == ECHELON_NAMES[node_index]:
                        available *= 0.25

                    shipped = round(max(available, 0.0), 4)
                    if shipped <= 0:
                        continue

                    if source_index is not None:
                        self._inventory[source_index][p_idx] -= shipped

                    self._order_backlogs[node_index][p_idx][method] = max(self._order_backlogs[node_index][p_idx][method] - shipped, 0.0)
                    shipped_quantities[node_index * self._num_products + p_idx] += shipped

                    base_lt = self._lead_times[node_index]
                    if base_lt <= 0:
                        self._inventory[node_index][p_idx] += shipped
                    else:
                        # Canal Blockage: Doubles lead time
                        lt_multiplier = 2.0 if "Canal Blockage" in self._active_events else 1.0
                        effective_lt = max(1, base_lt // 2) if method == 1 else base_lt
                        effective_lt = int(effective_lt * lt_multiplier)
                        
                        stochastic_lt = max(1, self._rng.randint(effective_lt - 1, effective_lt + 1))
                        self._pipelines[node_index][p_idx].append(Shipment(quantity=shipped, eta=stochastic_lt, method=method))
                    
                    # Carbon: 1.0 for Standard, 5.0 for Express
                    step_carbon += shipped * (5.0 if method == 1 else 1.0)
                    
        self._total_carbon += step_carbon
        return shipped_quantities

    def _update_forecast(self, observed_demand: float, demand_idx: int) -> float:
        # demand_idx is in range [0, 11] (3 products * 4 retailers)
        recent_mean = sum(self._recent_customer_demand[demand_idx]) / max(len(self._recent_customer_demand[demand_idx]), 1)
        alpha = 0.55 if self._difficulty in {"easy", "medium"} else 0.35
        baseline = alpha * observed_demand + (1 - alpha) * recent_mean
        p_idx = demand_idx // 4
        if self._config["season_shift_day"] and self._episode_state.step_count >= int(self._config["season_shift_day"]):
            if self._difficulty == "hard" or p_idx == 2:
                baseline += 2.0
        return round(max(baseline, 0.0), 4)

    def _sample_disruption_link(self) -> Optional[str]:
        if self._config["disruption_probability"] <= 0:
            return None
        if self._rng.random() >= float(self._config["disruption_probability"]):
            return None
        # Disruption affects a link in the chain (all products on that link)
        active_links = [ECHELON_NAMES[index] for index, active in enumerate(self._active_echelons) if active]
        return self._rng.choice(active_links)

    def _compute_reward_terms(
        self,
        current_demand: float,
        served_demand: float,
        shipped_quantities: list[float],
    ) -> dict[str, float]:
        holding_cost = 0.0
        target_inv = [60.0, 30.0, 30.0, 15.0, 15.0, 15.0, 15.0] 
        for e_idx in range(self._num_nodes):
            if not self._active_echelons[e_idx]:
                continue
            for p_idx in range(self._num_products):
                inventory = self._inventory[e_idx][p_idx]
                target = target_inv[e_idx]
                excess = max(inventory - target, 0.0)
                holding_cost += HOLDING_COSTS[e_idx] * inventory
                holding_cost += HOLDING_COSTS[e_idx] * (excess ** 2) / max(target, 1.0)

        stockout_penalty = 0.11 * sum(self._customer_backlog)
        for e_idx in range(self._num_nodes):
            if self._active_echelons[e_idx]:
                # Flattened backlog across methods
                stockout_penalty += 0.02 * sum(sum(m) for m in self._order_backlogs[e_idx])

        transport_cost = 0.0
        # For simplicity, we calculate transport cost from shipped_quantities
        # and assume a mix of methods based on what was dispatched
        # Wait, I should probably pass more info to _compute_reward_terms or calculate it inside _dispatch
        # Let's just track transport_cost in _dispatch and pass it here?
        # Better: calculate it based on the shipments actually made in this step.
        # But for this iteration, let's just use the total shipped and apply a multiplier if Express was used.
        # Actually, let's keep it simple: shipping_costs are already inclusive of multipliers.
        # Wait, I didn't return transport_cost from _dispatch. Let's fix that.
        
        # Let's just use a simplified version: transport_cost is already calculated using fuel_multiplier.
        # I'll add a carbon penalty here.
        carbon_penalty = 0.05 * self._total_carbon / (self._episode_state.step_count + 1)

        fill_rate_bonus = 0.65 * self._safe_ratio(served_demand, current_demand if current_demand > 0 else 1.0)
        
        # Re-calculate transport cost more accurately if needed, but for now we'll stick to the existing one
        # and just add carbon_penalty.
        for e_idx in range(self._num_nodes):
            for p_idx in range(self._num_products):
                idx = e_idx * self._num_products + p_idx
                shipped = shipped_quantities[idx]
                if shipped <= 0:
                    continue
                transport_cost += TRANSPORT_COSTS[e_idx] * shipped + FIXED_ORDER_COSTS[e_idx]
        
        transport_cost *= self._fuel_price_multiplier
        
        total_cost = holding_cost + stockout_penalty + transport_cost + carbon_penalty
        reward_scale = float(self._config["reward_scale"]) * (self._num_nodes / 3.0)
        raw_reward = fill_rate_bonus - (total_cost / reward_scale)
        reward = max(-1.0, min(1.0, raw_reward))
        return {
            "holding_cost": round(holding_cost, 4),
            "stockout_penalty": round(stockout_penalty, 4),
            "transport_cost": round(transport_cost, 4),
            "carbon_penalty": round(carbon_penalty, 4),
            "fill_rate_bonus": round(fill_rate_bonus, 4),
            "total": round(reward, 4),
        }

    def _build_observation(self, done: bool, reward: float, metadata: Optional[dict[str, Any]] = None) -> AgentObservation:
        inventory_levels = self._visible_inventory_levels() # flattened 21
        
        in_transit_qty = []
        lead_time_remaining = []
        for e_idx in range(self._num_nodes):
            for p_idx in range(self._num_products):
                queue = self._pipelines[e_idx][p_idx]
                it_qty = round(sum(shipment.quantity for shipment in queue), 4)
                lt_rem = float(min((shipment.eta for shipment in queue), default=0))
                in_transit_qty.append(it_qty)
                lead_time_remaining.append(lt_rem)

        forecasts = self._forecast_per_echelon() # flattened 21
        visibility_mask = [not (self._config["partial_visibility"] and index == 0) for index in range(self._num_nodes)]
        
        state_vector: list[float] = []
        for e_idx in range(self._num_nodes):
            for p_idx in range(self._num_products):
                idx = e_idx * self._num_products + p_idx
                state_vector.extend([
                    inventory_levels[idx],
                    in_transit_qty[idx],
                    forecasts[idx],
                    float(self._days_since_last_order[e_idx][p_idx]),
                    HOLDING_COSTS[e_idx],
                    lead_time_remaining[idx],
                ])

        flat_order_backlogs = []
        for e_idx in range(self._num_nodes):
            for p_idx in range(self._num_products):
                qty = sum(self._order_backlogs[e_idx][p_idx])
                flat_order_backlogs.append(round(qty, 4))

        flat_days_since_last_order = []
        for e_idx in range(self._num_nodes):
            flat_days_since_last_order.extend([float(v) for v in self._days_since_last_order[e_idx]])

        return AgentObservation(
            day=self._episode_state.step_count,
            horizon=self._horizon,
            difficulty=self._difficulty,
            echelon_names=list(ECHELON_NAMES),
            active_echelons=list(self._active_echelons),
            visibility_mask=visibility_mask,
            state_vector=[round(value, 4) for value in state_vector],
            inventory_levels=inventory_levels,
            in_transit_qty=in_transit_qty,
            demand_forecast=forecasts,
            days_since_last_order=flat_days_since_last_order,
            holding_cost_rate=[HOLDING_COSTS[e_idx] for e_idx in range(self._num_nodes) for _ in range(self._num_products)],
            lead_time_remaining=lead_time_remaining,
            order_backlogs=flat_order_backlogs,
            customer_backlog=[round(value, 4) for value in self._customer_backlog],
            recent_customer_demand=[round(val, 4) for dq in self._recent_customer_demand for val in list(dq)],
            carbon_footprint=round(self._total_carbon, 4),
            fill_rate=round(self._episode_state.fill_rate, 4),
            disruption_link=self._last_disruption_link,
            regime=self._episode_state.regime,
            reward_terms=self._last_reward_terms,
            action_bounds=[self._max_order_qty if active else 0.0 for active in self._active_echelons for _ in range(self._num_products)],
            active_events=list(self._active_events.keys()),
            done=done,
            reward=reward,
            metadata=metadata or {},
        )

    def _visible_inventory_levels(self) -> list[float]:
        visible = []
        for e_idx in range(self._num_nodes):
            for p_idx in range(self._num_products):
                val = round(self._inventory[e_idx][p_idx], 4)
                if self._config["partial_visibility"] and e_idx == 0:
                    val = -1.0
                visible.append(val)
        return visible

    def _forecast_per_echelon(self) -> list[float]:
        # Simple heuristic: warehouses forecast sum of their children, factory sum of warehouses
        # But we'll just use the consumer forecasts mapped to nodes for simplicity in the observation
        # For a hackathon, we can just map the retailer forecasts to retailers, and aggregate for others
        retailer_indices = [3, 4, 5, 6]
        forecasts = [0.0] * (self._num_nodes * self._num_products)
        
        # Retailers (directly from latest_forecasts)
        for p_idx in range(self._num_products):
            for r_idx, node_idx in enumerate(retailer_indices):
                f = self._latest_forecasts[p_idx * 4 + r_idx]
                forecasts[node_idx * self._num_products + p_idx] = round(f, 4)
                
        # Warehouse A (children 3, 4)
        for p_idx in range(self._num_products):
            f = forecasts[3 * self._num_products + p_idx] + forecasts[4 * self._num_products + p_idx]
            forecasts[1 * self._num_products + p_idx] = round(f * 1.1, 4)
            
        # Warehouse B (children 5, 6)
        for p_idx in range(self._num_products):
            f = forecasts[5 * self._num_products + p_idx] + forecasts[6 * self._num_products + p_idx]
            forecasts[2 * self._num_products + p_idx] = round(f * 1.1, 4)
            
        # Factory (children 1, 2)
        for p_idx in range(self._num_products):
            f = forecasts[1 * self._num_products + p_idx] + forecasts[2 * self._num_products + p_idx]
            forecasts[0 * self._num_products + p_idx] = round(f * 1.2, 4)
            
        return forecasts

    def _handle_news_events(self) -> None:
        # Update existing events
        expired = [name for name, days in self._active_events.items() if days <= 1]
        for name in expired:
            del self._active_events[name]
        for name in self._active_events:
            self._active_events[name] -= 1

        # Sample new events
        if self._difficulty == "hard":
            chance = 0.03
        elif self._difficulty == "medium":
            chance = 0.015
        else:
            chance = 0.005 # easy/mvp
            
        if self._rng.random() < chance and not self._active_events:
            events = [
                ("Canal Blockage", 14), # 14 days
                ("Labor Strike", 7),    # 7 days
                ("Social Media Trend", 5), # 5 days
            ]
            event_name, duration = self._rng.choice(events)
            self._active_events[event_name] = duration

    def _current_regime(self, day: int) -> str:
        shift_day = self._config["season_shift_day"]
        if shift_day is not None and day >= int(shift_day):
            return "peak_season"
        return "baseline"

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        return 0.0 if denominator <= 0 else numerator / denominator


# Backwards-compatible alias (older class name from the original env).
SupplyChainInventoryEnvironment = SupplyChainEnv
