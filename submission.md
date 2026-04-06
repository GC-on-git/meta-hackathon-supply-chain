# OpenEnv Hackathon Submission Checklist & Audit

This document consolidates the detailed audits of the `service/` package (within the parent `Meta/` repo) against the OpenEnv Hackathon requirements, judging criteria, and validation scripts.

---

## 1. Concrete Confirmation Steps for Pre-submission Checklist

Here is a practical way to **confirm each item** from the checklist based on the current layout. The repo is an OpenEnv Space under `service/` (with `openenv.yaml` at the repo root, and routes documented in `service/server/app.py`).

### 1. HF Space deploys (200 + `reset()`)
- Open the Space in the browser; confirm it loads without build/runtime errors.
- From a terminal, hit the **public Space URL** (and any **base path** your app uses — the README frontmatter mentions `base_path: /web`, so paths may be under that prefix):
  ```bash
  curl -sS -o /dev/null -w "%{http_code}\n" "https://YOUR_USERNAME-YOUR_SPACE.hf.space/"
  ```
- Call **`reset`** the same way the OpenEnv client does. Your server documents:
  ```python
      - POST /reset: Reset the environment
      - POST /step: Execute an action
      - GET /state: Get current environment state
  ```
  Confirm with something like (adjust URL and path for `/web` if needed):
  ```bash
  curl -sS -X POST "https://YOUR_SPACE.hf.space/reset" \
    -H "Content-Type: application/json" \
    -d '{}' -w "\n%{http_code}\n"
  ```
  **Success:** HTTP 200 and a valid JSON observation/state payload (not an error page).

### 2. OpenEnv spec compliance
- **Manual:** Open `openenv.yaml` (yours is `openenv.yaml` at the repo root) against the spec the organizers publish; confirm **typed models** (`service/models.py`) match what the server returns; exercise **`reset` / `step` / `state`** with real payloads.
- **Automated:** Use the **organizer’s validator**. That’s the authoritative check; local manual checks catch obvious issues first.

### 3. Dockerfile builds
- Run a local build from the directory that contains the Dockerfile submitted (yours includes `Dockerfile` at repo root and `service/server/Dockerfile`):
  ```bash
  docker build -f Dockerfile -t supply-chain-env:test .
  ```
  **Success:** image builds **without error**. Fix any missing files or wrong `WORKDIR`/`COPY` paths until it does.

### 4. Baseline reproduces (`inference.py`)
- Set **`API_BASE_URL`**, **`MODEL_NAME`**, **`HF_TOKEN`** in the Space **Settings → Variables** or your shell.
- Run **`python inference.py`** from the **repo root**.
  **Success:** exits 0 and prints or writes **scores** in the expected format.

### 5. Three+ tasks with graders (scores in 0.0–1.0)
- Run whatever script or CI job enumerates tasks (organizer docs or validator).
- For each grader output, assert each score is **`0.0 <= score <= 1.0`** (parse JSON/logs accordingly).

### 6. Environment variables & OpenAI client
- In the Space (or local): confirm the three vars are **defined** and non-empty where required.
- Grep your `inference.py` for **`OpenAI`** client usage and that the base URL / model name come from **`os.environ`** (or equivalent), not hard-coded endpoints.

### 7. Runtime & hardware (≤20 min, 2 vCPU / 8 GB)
- Time a full run: `time python inference.py` (or `/usr/bin/time -l` on macOS for memory).
- Optionally simulate limits: Docker `--cpus=2 --memory=8g` when running the same workload inside a container.

### 8. Pre-submission validator
- Run the **exact validator command** from the hackathon materials. Treat its **exit code and printed report** as the final gate; fix anything it flags before submitting.

**Gap vs your tree right now:** there is **no `inference.py` at the repo root** in this workspace, and `openenv.yaml` lives under **`hackathon/`**. Confirm whether the submission must be the **whole repo** with those paths or a **flattened layout** the grader expects.

---

## 2. Detailed Requirements Audit (Are key requirements being met?)

### Real-world task (not a toy)
**Met.** The domain is a multi-echelon, multi-product supply chain with demand, lead times, disruptions, and carbon-style pressure — clearly operational work people do (planning, ordering, service vs cost). It is not framed as a game or toy.

### Full OpenEnv spec (typed models, `reset`/`step`/`state`, `openenv.yaml`)
**Partially met.** 
- **Typed models:** You have Pydantic `AgentAction`, `AgentObservation`, and `SupplyChainState` in `service/models.py`. The scalar **reward** is a field on the observation (and HTTP `/step` returns `reward` alongside `observation` and `done`), not a separate **`Reward` Pydantic model** as the text literally asks. If the grader insists on a distinct `Reward` model type, this may fail their checklist.
- **`reset()` → initial observation:** Implemented (`SupplyChainEnv.reset` → `AgentObservation`).
- **`step(action)` → observation, reward, done:** Implemented; extra context is carried in **`metadata`** on the observation rather than a field named **`info`**. HTTP `/step` returns `observation`, `reward`, `done` (via `openenv.core.env_server.http_server.create_app`).
- **`state()`:** Implemented (`SupplyChainEnv.state` → `SupplyChainState`; `GET /state`).
- **`openenv.yaml`:** Present at `openenv.yaml` (repo root).
- **`openenv validate`:** Cannot be confirmed from the repo alone; you have to run it in your environment. Structure and `openenv-core` dependency suggest you are aiming for compliance, but **"tested via openenv validate" is not evidenced** in-repo.

### Minimum 3 tasks with agent graders (easy → hard, scores in [0, 1])
**Not met.** The repo defines **difficulty presets** (`easy`, `medium`, `mvp`, `hard`) for the **same** environment (`DIFFICULTY_PRESETS` in `service/server/hackathon_environment.py`), not three **separate tasks** each with an **agent grader** that outputs a **normalized score in [0, 1]** and stated success/failure criteria. No grader modules or task specs appear in the tree.

### Meaningful reward function
**Mostly met.** Step rewards are **dense** (every day), clipped to **[-1, 1]**, with **decomposed `reward_terms`** (holding, stockouts, transport, carbon-related penalty, fill-rate bonus) so the agent gets signal along the trajectory, not only at termination. Penalties for bad behavior are implicit (costs, backlogs) rather than an explicit "infinite loop" detector; that is typical for this domain.

### Baseline inference script (OpenAI client, `OPENAI_API_KEY`, reproducible scores on 3 tasks)
**Not met.** Training/inference in-repo is **PyTorch RL** (`service/train/agent_*.py`), not an **OpenAI API** baseline. There is **no** script at the repo root that uses the OpenAI client, reads **`OPENAI_API_KEY`**, or reports **fixed baseline scores on three graded tasks** (and there are no three tasks to score).

### Hugging Face Space (containerized, tagged `openenv`)
**Partially met in repo; deployment unverified.** `service/README.md` frontmatter uses **`sdk: docker`** and lists **`openenv`** under `tags`. Whether the Space **actually builds, runs, and is tagged** on HF must be checked on the Hub; the repo alone does not prove it.

### Containerized execution (Dockerfile, clean `build` + `run`)
**Partially met.** `service/server/Dockerfile` exists (and there is also a repo-root `Dockerfile`). Whether **`docker build` / `docker run`** succeed with the exact paths and entrypoint HF expects depends on your Space config and context directory; that is not proven here without running Docker.

### Documentation (description, motivation, action/obs, tasks, setup, baseline scores)
**Partially met.** The README is strong on **environment description**, **motivation**, **action/observation**, **rewards**, **difficulty presets**, and **setup/training**. It does **not** document **three distinct tasks** with grader definitions, and it does **not** include **reported baseline scores** for graded tasks (it even states the package does not ship a pretrained network).

---

## 3. Review of the Sample BrowserGym Inference Script

**Will it work with our environment? No.**
That sample is built for **BrowserGym** (browser UI, screenshots, `BrowserGymAction` / `BrowserGymEnv`, Docker image `browsergym-env:latest`). Your project is a **supply-chain OpenEnv** with **numeric control**: `AgentAction` (`order_quantities`, `shipping_methods`) and **no** `goal`, `url`, `screenshot`, or BrowserGym metadata.

**What would break if dropped in:**
1. **Imports:** `from browsergym_env import BrowserGymAction, BrowserGymEnv` will fail; you’d use `SupplyChainEnv` (in-process) or your HTTP client, not BrowserGym.
2. **Env API:** `env.reset()` / `env.step(...)` must use **`AgentAction`**, not `BrowserGymAction(action_str=...)`.
3. **Observations:** Your model sees **vectors / structured lists** (inventory, forecasts, etc.), not pixels + DOM. An LLM loop would need a **text (or JSON) description** of `AgentObservation`, not `extract_screenshot_uri` / `extract_clickable_elements`.
4. **Docker:** The sample spins up `browsergym-env:latest`. Your stack uses `Dockerfile` (repo root) / `service/server/Dockerfile` for the simulator server.
5. **Bug in sample:** `API_BASE_URL = os.getenv("API_BASE_URL") // "https://..."` uses **floor division** (`//`), which throws a TypeError or gives wrong semantics. You'd use `or`.

**What you can reuse conceptually:**
The pattern `OpenAI(base_url=..., api_key=...)`, `chat.completions.create`, reading `API_BASE_URL` / `MODEL_NAME` / `HF_TOKEN`, and a loop over the env. The prompts, action parsing, and env class must be rewritten for `AgentAction` and your observation fields.

---

## 4. Review of the Pre-submission Validation Script (`validate_submission.py`)

**Will this pre-validation script work in our code?** It will, if it uses the correct repo root context (and the Space URL matches `/reset` or `/web/reset`).

### Step 1 — `POST $PING_URL/reset`
- Your app exposes **`POST /reset`**, so this matches what the script calls.
- **Caveat:** If the live Space serves the API under the `/web` prefix (as hinted in README), the script must ping `$PING_URL/web/reset`, not `$PING_URL/reset`. 

### Step 2 — `docker build`
**Likely wrong for this repo.**
The script checks for `$REPO_DIR/Dockerfile` or `$REPO_DIR/server/Dockerfile` and runs `docker build "$DOCKER_CONTEXT"`.
In your tree, the Dockerfile is at `Dockerfile` (repo root) and `service/server/Dockerfile`.
- If `repo_dir=Meta`, it fails with "No Dockerfile found".
- If `repo_dir=Meta/service`, it finds `service/server/Dockerfile` and builds using **`service/server`** as the context.
Your repo-root Dockerfile does `COPY . /app/env` and `uv sync`, expecting the **full repo** as the build context.

### Step 3 — `openenv validate`
- `openenv.yaml` is at the repo root.
- You can run validation with `repo_dir=.../Meta` so `openenv validate` sees the repo-root yaml file.

**Takeaway:** The script is not a drop-in pre-validation for your current structure without changing `repo_dir`, adjusting the ping URL (for `/web`), and fixing the Docker build context.

---

## 5. Summary of Overall Status Against Judging Criteria

**Do we meet the judging criteria? No — not as a complete submission against the rubric.**

**What aligns:**
- **OpenEnv-shaped server:** `openenv.yaml`, Pydantic Action/Observation/State, and `POST /reset`, `POST /step`, `GET /state` exist.
- **Dockerfile:** Present, but CI build context might mismatch.

**What does NOT meet the criteria:**
- **`inference.py` at project root:** Not present.
- **OpenAI client + env vars (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`):** Not implemented (you only have PyTorch training scripts).
- **Baseline runs cleanly and prints scores:** Nothing submitted that does this.
- **3+ tasks with separate graders and scores [0.0, 1.0]:** Not implemented; you only have difficulty presets on one env.
- **HF Space 200 + `reset()`:** Only verifiable on a live Space URL.
- **< 20 min, 2 vCPU / 8 GB:** Not demonstrated without the missing baseline.
- **Pre-submission validator:** You still have to run it, and the script needs adjustment for your layout.

---

## 6. Estimated Scoring Breakdown

*Note: This is an estimate. Anything depending on a live Space, `openenv validate`, and Docker can only be guessed from the repo.*

| Category | Weight | Honest read of *this* repo | Approx. points |
|----------|--------|----------------------------|----------------|
| **1. Real-world utility** | 30 | Multi-echelon supply chain with demand noise, lead times, disruptions, carbon, service vs cost — clearly **not** a toy; strong operational framing. | **~22–28** |
| **2. Task & grader quality** | 25 | Rubric wants 3+ separate tasks, 0.0–1.0 graders, deterministic/reproducible, hard challenging frontier models. You have one env with difficulty presets, not enumerated tasks/graders. This bucket is **mostly missing**. | **~0–8** |
| **3. Environment design** | 20 | `reset()` re-inits episode; action/obs are typed and documented; dense step reward + `reward_terms`; fixed horizon episodes. Strong fit. | **~16–19** |
| **4. Code quality & spec compliance** | 15 | Implementation looks serious, but baseline script + reproducible scores are **not** in tree; `openenv validate` / HF / docker are **not** proven here; Docker context must match. Partial credit at best until those pass. | **~5–11** |
| **5. Creativity & novelty** | 10 | Supply chain sims exist elsewhere; your mix (multi-product, events, partial visibility, carbon in reward) is **reasonably** distinctive for OpenEnv. | **~5–8** |

### Rough Total Score
- **Low end** (harsh on missing tasks/graders + spec): **~48–55 / 100**
- **Mid** (strong env + real-world, weak task/grader + unproven compliance): **~55–65 / 100**
- **High end** (generous on real-world/design, penalize missing graders/baseline): **~65–72 / 100**

**Main ceiling:** **Task & grader quality (25%)** and **Code/spec (15%)**. Without **3 graded tasks**, **deterministic [0,1] scores**, and a **running baseline + validate/docker/HF proof**, you likely sit **below ~70** no matter how strong the simulator is.
