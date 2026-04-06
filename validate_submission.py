#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Optional

import urllib.request


@dataclass
class CheckResult:
    ok: bool
    message: str
    details: str = ""


def _log(msg: str) -> None:
    now = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{now}] {msg}")


def _run(cmd: list[str], cwd: Optional[str] = None, timeout_s: Optional[int] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )


def _http_post_json(url: str, payload: bytes, timeout_s: int = 30) -> int:
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(getattr(resp, "status", 200))
    except Exception as exc:
        if hasattr(exc, "code"):
            return int(getattr(exc, "code"))
        return 0


def check_hf_space(ping_url: str) -> CheckResult:
    ping_url = ping_url.rstrip("/")
    candidates = [f"{ping_url}/reset", f"{ping_url}/web/reset"]
    payload = b"{}"
    for url in candidates:
        code = _http_post_json(url, payload)
        if code == 200:
            return CheckResult(True, f"HF Space responds to POST {url} (200)")
    return CheckResult(
        False,
        "HF Space did not return 200 for reset",
        details="Tried: " + ", ".join(candidates),
    )


def check_docker_build(repo_dir: str, timeout_s: int = 600) -> CheckResult:
    if shutil.which("docker") is None:
        return CheckResult(False, "docker not found", details="Install Docker: https://docs.docker.com/get-docker/")

    dockerfile = os.path.join(repo_dir, "Dockerfile")
    if not os.path.isfile(dockerfile):
        return CheckResult(False, "Dockerfile not found at repo root", details=f"Expected: {dockerfile}")

    proc = _run(["docker", "build", "-f", dockerfile, repo_dir], timeout_s=timeout_s)
    if proc.returncode == 0:
        return CheckResult(True, "Docker build succeeded")
    tail = "\n".join(proc.stdout.splitlines()[-40:])
    return CheckResult(False, "Docker build failed", details=tail)


def check_openenv_validate(repo_dir: str) -> CheckResult:
    if shutil.which("openenv") is None:
        return CheckResult(
            False,
            "openenv not found",
            details="Install: uv sync (repo root) or pip install 'openenv-core[core]>=0.2.2'",
        )

    proc = _run(["openenv", "validate"], cwd=repo_dir)
    if proc.returncode == 0:
        out = proc.stdout.strip()
        return CheckResult(True, "openenv validate passed", details=out)
    return CheckResult(False, "openenv validate failed", details=proc.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="validate_submission.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            OpenEnv submission pre-validation:
            - Ping HF Space /reset (also tries /web/reset)
            - docker build (repo root Dockerfile)
            - openenv validate
            """
        ).strip(),
    )
    parser.add_argument("ping_url", help="HuggingFace Space URL, e.g. https://your-space.hf.space")
    parser.add_argument("--repo-dir", default=".", help="Path to repo root (default: .)")
    parser.add_argument("--docker-timeout", type=int, default=600, help="Docker build timeout seconds (default: 600)")
    args = parser.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    if not os.path.isdir(repo_dir):
        print(f"Error: repo-dir not found: {repo_dir}", file=sys.stderr)
        return 2

    _log("========================================")
    _log("OpenEnv Submission Validator")
    _log("========================================")
    _log(f"Repo:     {repo_dir}")
    _log(f"Ping URL: {args.ping_url.rstrip('/')}")

    checks: list[tuple[str, CheckResult]] = []

    _log("Step 1/3: Pinging HF Space reset ...")
    r1 = check_hf_space(args.ping_url)
    checks.append(("Step 1", r1))
    _log(("PASSED" if r1.ok else "FAILED") + f" -- {r1.message}")
    if r1.details:
        print(r1.details)
    if not r1.ok:
        return 1

    _log("Step 2/3: Running docker build ...")
    r2 = check_docker_build(repo_dir, timeout_s=args.docker_timeout)
    checks.append(("Step 2", r2))
    _log(("PASSED" if r2.ok else "FAILED") + f" -- {r2.message}")
    if r2.details:
        print(r2.details)
    if not r2.ok:
        return 1

    _log("Step 3/3: Running openenv validate ...")
    r3 = check_openenv_validate(repo_dir)
    checks.append(("Step 3", r3))
    _log(("PASSED" if r3.ok else "FAILED") + f" -- {r3.message}")
    if r3.details:
        print(r3.details)
    if not r3.ok:
        return 1

    _log("========================================")
    _log("All 3/3 checks passed. Ready to submit.")
    _log("========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

