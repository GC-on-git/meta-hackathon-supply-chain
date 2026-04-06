# OpenEnv Hackathon Submission Checklist & Audit

This document reflects the **current** `Meta/` repo layout: **repo root** holds the Docker image, `openenv.yaml`, root `pyproject.toml` / `uv.lock`, ASGI entrypoints (`app.py`, `server/app.py`), **`inference.py`**, and **`validate_submission.py`**. The **`service/`** directory is the Python package for the simulator, Pydantic models, **graders**, optional RL training, and tests.

---

## 1. Concrete Confirmation Steps for Pre-submission Checklist

### 1. HF Space deploys (200 + `reset()`)

- Open the Space; confirm build/runtime logs are clean.
- The validator tries **`POST …/reset`** and **`POST …/web/reset`** (see `validate_submission.py`).
- Manual check:

  ```bash
  curl -sS -X POST "https://YOUR_SPACE.hf.space/reset" \
    -H "Content-Type: application/json" \
    -d '{}' -w "\n%{http_code}\n"
  ```

  **Success:** HTTP **200** and JSON (observation/state), not an HTML error page.

### 2. OpenEnv spec compliance

- Compare **`openenv.yaml`** (repo root) to the organizer’s spec.
- **Important:** `openenv.yaml` currently sets `app: service.server.app:app`, but the implemented FastAPI app lives at **`server.app:app`** (see `server/app.py`) with a top-level shim **`app:app`** (`app.py`). If validation or deployment expects `service.server.app`, either **update `openenv.yaml`** to `server.app:app` (or `app:app` if allowed) or **reintroduce** a `service.server` module consistent with the spec.
- Exercise **`POST /reset`**, **`POST /step`**, **`GET /state`** with real payloads (see `service/models.py`).

### 3. Dockerfile builds

From the **repo root** (same as `validate_submission.py`):

```bash
docker build -f Dockerfile -t supply-chain-env:test .
```

The Dockerfile copies the **entire repo** to `/app/env`, runs **`uv sync`** with **`--no-install-project`** using the **root** `pyproject.toml` / `uv.lock`, and starts **`uvicorn app:app`** on port **8000**.

**Success:** build completes without error.

### 4. Baseline reproduces (`inference.py`)

- At **repo root**, set API credentials. The script uses **`OPENROUTER_BASE_URL`** (default Open Router), **`OPENROUTER_API_KEY`** or **`API_KEY`**, and **`MODEL_NAME`** (see `inference.py` — there are commented Hugging Face Router examples).
- Optional: **`TASK_SEED_EASY`**, **`TASK_SEED_MEDIUM`**, **`TASK_SEED_HARD`** for reproducibility.
- Run:

  ```bash
  python inference.py
  ```

  **Success:** three tasks (**easy**, **medium**, **hard**) run; final block prints **Baseline Inference Scores** in **[0.0, 1.0]** via **`service.grading.grade_episode`**.

### 5. Three tasks with graders (scores in [0, 1])

- **`service/grading.py`** defines **`grade_episode(state, task_name)`** with per-task weights and fill/cost/(hard) carbon components; output is clamped to **[0.0, 1.0]**.
- **`inference.py`** runs those three difficulties as separate tasks and records scores.

### 6. Environment variables & OpenAI-compatible client

- Confirm vars used by **`inference.py`** for your chosen provider (e.g. Open Router vs HF Router).
- The client is **`openai.OpenAI`** with **`base_url`** and **`api_key`** from the environment — not a hard-coded third-party endpoint in code (model name via **`MODEL_NAME`**).

### 7. Runtime & hardware (organizer limits)

- Time: `time python inference.py` (adjust **`MAX_STEPS`** in `inference.py` if the rubric expects a different horizon).
- Optionally run the container with **`--cpus=2 --memory=8g`** to mimic Space limits.

### 8. Pre-submission validator

```bash
python validate_submission.py https://YOUR_USERNAME-YOUR_SPACE.hf.space --repo-dir .
```

Steps: HF **`/reset`** (or **`/web/reset`**), **`docker build`** from **repo-root Dockerfile**, **`openenv validate`** with **`cwd`** = repo root.

If **`openenv validate`** reports **“Missing pyproject.toml”** or **multi-mode deployment** issues, confirm the **repo root** contains **`pyproject.toml`** and that **`openenv.yaml`**’s **`app`** path matches an importable module on `PYTHONPATH` (see caveat in §1.2).

---

## 2. Requirements Audit (current repo)

### Real-world task

**Met.** Multi-echelon, multi-product supply chain with stochastic demand, lead times, disruptions, carbon pressure, service vs cost.

### OpenEnv spec (typed models, reset/step/state, openenv.yaml)

**Mostly met (verify with `openenv validate`).**

- **Models:** `service/models.py` — `AgentAction`, `AgentObservation`, `SupplyChainState`.
- **Server:** `openenv.core.env_server.http_server.create_app` in `server/app.py` with `SupplyChainEnv`, `AgentAction`, `AgentObservation`.
- **`openenv.yaml`:** Repo root; **`app`** entry should match the actual module path (see §1.2).

### Minimum 3 tasks with graders [0, 1]

**Met** for the baseline path: **`inference.py`** + **`service/grading.py`**, tasks **easy / medium / hard**.

### Meaningful reward

**Met.** Dense step reward in `service/hackathon_environment.py` with **`reward_terms`**; separate **episode grader** for baseline scoring.

### Baseline inference script

**Met at repo root:** `inference.py` uses an OpenAI-compatible client and prints per-task scores.

### HF Space / Docker

**Repo-side:** Root **`Dockerfile`**, **`openenv.yaml`**, health check on **`/health`**. Live Space must still be verified on the Hub.

### Documentation

**Partially met** until README paths are fully aligned with **`server.app:app`** vs **`service.server.app`** (grep the repo and fix any stale commands).

---

## 3. BrowserGym sample script

**Still not applicable.** This environment is numeric (`AgentAction` with 21+21 dimensions), not BrowserGym/browser UI. Reuse only the **high-level loop** (client → chat → parse JSON → `step`).

---

## 4. `validate_submission.py` vs this repo

| Step | Behavior |
|------|----------|
| 1 | Tries **`$URL/reset`** and **`$URL/web/reset`**. |
| 2 | Requires **`$REPO_DIR/Dockerfile`** at **repo root**; build context is **`repo_dir`**. Matches current layout. |
| 3 | Runs **`openenv validate`** with **`cwd=repo_dir`**; root must include **`openenv.yaml`** and satisfy the CLI’s **pyproject** / multi-mode rules. |

---

## 5. Summary status (high level)

- **Strong:** Environment design, typed API, Docker from root, **inference + grading** for three tasks.
- **Verify:** **`openenv validate`** passes; **`openenv.yaml` `app`** matches real entrypoint; HF Space URL and path prefix; README commands updated.

---

## 6. Estimated scoring (illustrative only)

| Category | Notes |
|----------|--------|
| Real-world utility | Strong supply-chain framing. |
| Task & grader quality | **`grading.py`** + three named tasks in **`inference.py`**. |
| Environment design | Strong. |
| Code quality / spec | Depends on **`openenv validate`**, HF build, and **`openenv.yaml`/`app` consistency**. |
| Creativity | Multi-product, events, partial visibility, carbon in reward/grader. |

*Numeric totals omitted on purpose — use the official validator and rubric for authoritative scoring.*
