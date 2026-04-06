# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch training device selection (CUDA → MPS → CPU for ``auto``)."""

from __future__ import annotations

import argparse
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def _mps_available(torch_module: object) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is None:
        return False
    is_built = getattr(mps, "is_built", lambda: False)
    is_available = getattr(mps, "is_available", lambda: False)
    try:
        return bool(is_built() and is_available())
    except Exception:
        return False


def resolve_training_device(request: str = "auto"):
    """Pick a ``torch.device`` from ``auto`` / ``cuda`` / ``gpu`` / ``mps`` / ``cpu``.

    ``auto``: CUDA if available, else Apple MPS if available, else CPU.
    """
    import torch

    r = request.strip().lower()
    if r == "gpu":
        r = "cuda"

    if r == "cpu":
        return torch.device("cpu")

    if r == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested (--device cuda) but CUDA is not available.")

    if r == "mps":
        if _mps_available(torch):
            return torch.device("mps")
        raise RuntimeError("MPS was requested (--device mps) but MPS is not available.")

    if r == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available(torch):
            return torch.device("mps")
        return torch.device("cpu")

    raise ValueError(f"Unknown device {request!r}; use auto, cuda, mps, or cpu.")


def add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "gpu", "mps", "cpu"),
        help="Training backend: auto (CUDA if available, else MPS, else CPU); gpu aliases cuda.",
    )
