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


def check_grader_smoke(repo_dir: str) -> CheckResult:
    """Import graders and env; one zero-action step per task; scores must be in [0, 1]."""
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from service.grading import grade_episode
        from service.hackathon_environment import SupplyChainEnv
        from service.models import AgentAction
        from service.tasks import task_specs
    except ImportError as exc:
        return CheckResult(False, "grader smoke import failed", str(exc))

    for spec in task_specs():
        env = SupplyChainEnv()
        env.reset(difficulty=spec.difficulty, seed=spec.seed, horizon=spec.horizon)
        env.step(
            AgentAction(
                order_quantities=[0.0] * 21,
                shipping_methods=[0] * 21,
            )
        )
        score = float(grade_episode(env.state, spec.task_id))
        if score < 0.0 or score > 1.0:
            return CheckResult(
                False,
                f"grader score out of range for task={spec.task_id}",
                f"score={score!r}",
            )
    return CheckResult(True, "grader smoke passed (all tasks in [0.0, 1.0])")


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
            Optional: --smoke-graders (after the above), or --smoke-graders-only.
            """
        ).strip(),
    )
    parser.add_argument("ping_url", nargs="?", default="", help="HuggingFace Space URL (optional if --smoke-graders-only)")
    parser.add_argument("--repo-dir", default=".", help="Path to repo root (default: .)")
    parser.add_argument("--docker-timeout", type=int, default=600, help="Docker build timeout seconds (default: 600)")
    parser.add_argument(
        "--smoke-graders",
        action="store_true",
        help="After other checks, run a short grader smoke test (scores in [0,1]).",
    )
    parser.add_argument(
        "--smoke-graders-only",
        action="store_true",
        help="Only run grader smoke test (no HF ping, docker, or openenv validate).",
    )
    args = parser.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    if not os.path.isdir(repo_dir):
        print(f"Error: repo-dir not found: {repo_dir}", file=sys.stderr)
        return 2

    if args.smoke_graders_only:
        _log("========================================")
        _log("OpenEnv Grader Smoke Test")
        _log("========================================")
        _log(f"Repo: {repo_dir}")
        r = check_grader_smoke(repo_dir)
        _log(("PASSED" if r.ok else "FAILED") + f" -- {r.message}")
        if r.details:
            print(r.details)
        return 0 if r.ok else 1

    ping_url = (args.ping_url or "").strip()
    if not ping_url:
        print("Error: ping_url is required unless --smoke-graders-only is set.", file=sys.stderr)
        return 2

    _log("========================================")
    _log("OpenEnv Submission Validator")
    _log("========================================")
    _log(f"Repo:     {repo_dir}")
    _log(f"Ping URL: {ping_url.rstrip('/')}")

    _log("Step 1/3: Pinging HF Space reset ...")
    r1 = check_hf_space(ping_url)
    _log(("PASSED" if r1.ok else "FAILED") + f" -- {r1.message}")
    if r1.details:
        print(r1.details)
    if not r1.ok:
        return 1

    _log("Step 2/3: Running docker build ...")
    r2 = check_docker_build(repo_dir, timeout_s=args.docker_timeout)
    _log(("PASSED" if r2.ok else "FAILED") + f" -- {r2.message}")
    if r2.details:
        print(r2.details)
    if not r2.ok:
        return 1

    _log("Step 3/3: Running openenv validate ...")
    r3 = check_openenv_validate(repo_dir)
    _log(("PASSED" if r3.ok else "FAILED") + f" -- {r3.message}")
    if r3.details:
        print(r3.details)
    if not r3.ok:
        return 1

    if args.smoke_graders:
        _log("Step 4: Grader smoke test ...")
        r4 = check_grader_smoke(repo_dir)
        _log(("PASSED" if r4.ok else "FAILED") + f" -- {r4.message}")
        if r4.details:
            print(r4.details)
        if not r4.ok:
            return 1

    _log("========================================")
    _log("All checks passed. Ready to submit.")
    _log("========================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

