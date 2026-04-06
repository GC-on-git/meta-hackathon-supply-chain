# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
**SAC** (Soft Actor-Critic) in PyTorch on **OpenEnv** ``SupplyChainEnv``.

Uses a **Beta** policy on ``[0, 1]^{42}`` (same encoding as ``vector_to_agent_action``).
Replay buffer + twin Q-networks — no Gymnasium / Stable-Baselines3.

Example::

    PYTHONPATH=. python -m service.train.agent_sac --difficulty easy
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from typing import Deque

import numpy as np

from service.hackathon_environment import SupplyChainEnv
from service.train import (
    ACTION_DIM,
    STATE_VECTOR_DIM,
    observation_to_vector,
    vector_to_agent_action,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SAC (PyTorch) on SupplyChainEnv (OpenEnv)")
    parser.add_argument(
        "--total-steps",
        type=int,
        default=80_000,
        help="stop after this many env.step(AgentAction) calls (simulated days of interaction)",
    )
    parser.add_argument("--buffer", type=int, default=50_000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005, help="soft target update")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.2, help="entropy coefficient (fixed)")
    parser.add_argument("--warmup", type=int, default=2000, help="random actions before SAC updates")
    parser.add_argument("--difficulty", type=str, default="easy")
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="models/sac_supply_chain.pt")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torch.distributions import Beta
    except ImportError as e:
        raise SystemExit(
            'PyTorch is required. Install with: pip install -e ".[train]" or pip install torch numpy'
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    def mlp(sizes: list, act: type = nn.ReLU) -> nn.Sequential:
        layers: list = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act())
        return nn.Sequential(*layers)

    class Actor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            h = 256
            self.trunk = mlp([STATE_VECTOR_DIM, h, h])
            self.alpha = nn.Linear(h, ACTION_DIM)
            self.beta = nn.Linear(h, ACTION_DIM)

        def forward(self, s: torch.Tensor) -> Beta:
            h = self.trunk(s)
            al = F.softplus(self.alpha(h)) + 1.0
            be = F.softplus(self.beta(h)) + 1.0
            return Beta(al, be)

    class QNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = mlp([STATE_VECTOR_DIM + ACTION_DIM, 256, 256, 1])

        def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)

    env = SupplyChainEnv()
    actor = Actor().to(device)
    q1 = QNetwork().to(device)
    q2 = QNetwork().to(device)
    q1_t = QNetwork().to(device)
    q2_t = QNetwork().to(device)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())
    for p in q1_t.parameters():
        p.requires_grad = False
    for p in q2_t.parameters():
        p.requires_grad = False

    opt_a = optim.Adam(actor.parameters(), lr=args.lr)
    opt_q = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)

    buf: Deque[tuple] = deque(maxlen=args.buffer)

    def soft_update() -> None:
        with torch.no_grad():
            for tp, p in zip(q1_t.parameters(), q1.parameters()):
                tp.data.mul_(1 - args.tau).add_(args.tau * p.data)
            for tp, p in zip(q2_t.parameters(), q2.parameters()):
                tp.data.mul_(1 - args.tau).add_(args.tau * p.data)

    obs_p = env.reset(difficulty=args.difficulty, seed=args.seed, horizon=args.horizon)
    s = observation_to_vector(obs_p)

    print(
        f"DEBUG SAC: total_steps={args.total_steps} device={device} alpha={args.alpha} "
        f"difficulty={args.difficulty!r} (OpenEnv in-process)"
    )

    ep_reward = 0.0
    for t in range(args.total_steps):
        if t < args.warmup:
            a = np.random.uniform(0.0, 1.0, size=(ACTION_DIM,)).astype(np.float32)
        else:
            st = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist = actor(st)
                a = dist.sample().cpu().numpy().reshape(-1)
        aa = vector_to_agent_action(a, env._max_order_qty)
        next_p = env.step(aa)
        s2 = observation_to_vector(next_p)
        r = float(next_p.reward) if next_p.reward is not None else 0.0
        d = float(next_p.done)
        buf.append((s.copy(), a.copy(), r, s2.copy(), d))
        ep_reward += r
        s = s2
        if next_p.done:
            if t % 1000 == 0:
                print(f"DEBUG SAC: step {t:6d} last_ep_return={ep_reward:.4f} buffer={len(buf)}")
            ep_reward = 0.0
            obs_p = env.reset(difficulty=args.difficulty, seed=None, horizon=args.horizon)
            s = observation_to_vector(obs_p)

        if len(buf) < args.batch or t < args.warmup:
            continue

        batch = random.sample(buf, args.batch)
        sb = torch.as_tensor(np.stack([x[0] for x in batch]), dtype=torch.float32, device=device)
        ab = torch.as_tensor(np.stack([x[1] for x in batch]), dtype=torch.float32, device=device)
        rb = torch.as_tensor([x[2] for x in batch], dtype=torch.float32, device=device)
        s2b = torch.as_tensor(np.stack([x[3] for x in batch]), dtype=torch.float32, device=device)
        db = torch.as_tensor([x[4] for x in batch], dtype=torch.float32, device=device)

        with torch.no_grad():
            n_dist = actor(s2b)
            a2 = n_dist.rsample()
            logp2 = n_dist.log_prob(a2).sum(-1)
            q_tgt = torch.min(q1_t(s2b, a2), q2_t(s2b, a2)) - args.alpha * logp2
            target = rb + (1.0 - db) * args.gamma * q_tgt

        q1v = q1(sb, ab)
        q2v = q2(sb, ab)
        loss_q = F.mse_loss(q1v, target) + F.mse_loss(q2v, target)
        opt_q.zero_grad()
        loss_q.backward()
        opt_q.step()

        dist_pi = actor(sb)
        api = dist_pi.rsample()
        logp_pi = dist_pi.log_prob(api).sum(-1)
        qpi = torch.min(q1(sb, api), q2(sb, api))
        loss_pi = (args.alpha * logp_pi - qpi).mean()
        opt_a.zero_grad()
        loss_pi.backward()
        opt_a.step()

        soft_update()

    import os

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(
        {
            "actor": actor.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "config": vars(args),
        },
        args.save_path,
    )
    print(f"DEBUG SAC: saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
