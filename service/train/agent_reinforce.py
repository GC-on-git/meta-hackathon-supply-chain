# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
**REINFORCE** (Monte Carlo policy gradient) with PyTorch on **OpenEnv** ``SupplyChainEnv``.

Rollouts use ``reset`` / ``step(AgentAction)`` and ``observation_to_vector`` — no Gymnasium.

Example::

    PYTHONPATH=. python -m service.train.agent_reinforce --episodes 500 --difficulty easy
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np

from service.train import (
    ACTION_DIM,
    STATE_VECTOR_DIM,
    new_supply_chain_env,
    observation_to_vector,
    vector_to_agent_action,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="REINFORCE (PyTorch) on SupplyChainEnv (OpenEnv)")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--difficulty", type=str, default="easy")
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="models/reinforce_supply_chain.pt")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.distributions import Beta
    except ImportError as e:
        raise SystemExit(
            'PyTorch is required. Install with: pip install -e ".[train]" or pip install torch numpy'
        ) from e

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    class Policy(nn.Module):
        def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            self.alpha_head = nn.Linear(hidden, act_dim)
            self.beta_head = nn.Linear(hidden, act_dim)

        def forward(self, x: torch.Tensor) -> Beta:
            h = self.trunk(x)
            alpha = torch.nn.functional.softplus(self.alpha_head(h)) + 1.0
            beta = torch.nn.functional.softplus(self.beta_head(h)) + 1.0
            return Beta(alpha, beta)

        def act(self, obs: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            dist = self.forward(x)
            a = dist.sample()
            lp = dist.log_prob(a).sum(dim=-1)
            return a.squeeze(0).detach().numpy(), lp.squeeze(0)

    policy = Policy(STATE_VECTOR_DIM, ACTION_DIM)
    opt = optim.Adam(policy.parameters(), lr=args.lr)

    print(
        f"DEBUG REINFORCE: episodes={args.episodes} gamma={args.gamma} lr={args.lr} "
        f"difficulty={args.difficulty!r} horizon={args.horizon} (OpenEnv in-process)"
    )

    episode_returns: List[float] = []
    for ep in range(args.episodes):
        env, obs_vec = new_supply_chain_env(
            difficulty=args.difficulty, horizon=args.horizon, seed=args.seed + ep
        )
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        terminated = False
        steps = 0
        while not terminated:
            action, lp = policy.act(obs_vec)
            aa = vector_to_agent_action(action, env._max_order_qty)
            next_obs = env.step(aa)
            obs_vec = observation_to_vector(next_obs)
            log_probs.append(lp)
            rewards.append(float(next_obs.reward) if next_obs.reward is not None else 0.0)
            terminated = bool(next_obs.done)
            steps += 1

        G = 0.0
        returns: List[float] = []
        for r in reversed(rewards):
            G = r + args.gamma * G
            returns.append(G)
        returns.reverse()
        R = torch.tensor(returns, dtype=torch.float32)
        R = (R - R.mean()) / (R.std(unbiased=False) + 1e-8)

        opt.zero_grad()
        if not log_probs:
            continue
        loss = torch.stack([-lp * R[t] for t, lp in enumerate(log_probs)]).mean()
        loss.backward()
        opt.step()

        ep_ret = sum(rewards)
        episode_returns.append(ep_ret)
        if ep % 20 == 0 or ep == args.episodes - 1:
            tail = episode_returns[-20:]
            print(
                f"DEBUG REINFORCE: episode {ep:4d} steps={steps:3d} return={ep_ret:8.4f} "
                f"last20_mean={float(np.mean(tail)):8.4f} loss={float(loss.detach()):.4f}"
            )

    import os

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save({"policy_state_dict": policy.state_dict(), "config": vars(args)}, args.save_path)
    print(f"DEBUG REINFORCE: saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
