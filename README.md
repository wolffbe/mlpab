<p align="center">
  <img src="testbed/docs/logo.png" alt="MLPlatformAgentBench" width="420">
</p>

# MLPlatformAgentBench (MLPAB)

MLPlatformAgentBench is a benchmark for evaluating large language model coding
agents on machine learning platform tasks. It measures how the interface given
to the agent, a command-line interface (CLI) or a Python SDK,
affects how well the agent completes those tasks.

## What it measures

Each task is run across a grid of conditions so the conditions can be compared
directly.

- **Platform.** Hopsworks, Databricks, AWS SageMaker, Azure ML, GCP Vertex, and a `local` baseline (`none`).
- **Interface.** The platform CLI or the Python SDK (or `none`).
- **Skills.** With or without a bundle of platform skills (Claude Code slash-commands and docs).
- **Agent engine and model.** `claude-*` (Claude Code), `gpt-*` (Codex), `mistral-*` (Mistral Vibe).

The agent's work is graded by reading the result back through the platform's own
data paths, not by inspecting the agent transcript. Tasks are generated as fresh,
seeded instances of the Feature, Training, and Inference (FTI) lifecycle, plus
Ops and Capstone tasks. Each run uses a new seed, so runs are reproducible but
not identical. The agent runs in a sandbox confined to its run directory and is
restricted to the one interface under test. Per-platform setup and teardown
scripts provision and remove cloud resources around each run.

MLPAB is the testbed for a KTH master's thesis on how interface design and
supporting skills affect agent performance on ML platform operations. The thesis
is in `thesis/`.

```
banter/
  testbed/   MLPlatformAgentBench: framework, configs, evals, results (cd here)
  thesis/    KTH LaTeX thesis (thesis/thesis.tex)
```

## How a run works

1. A treatment config (`configs/treatments/<platform>/<platform>-<engine>.yaml`, where `<engine>` is `claude`, `gpt`, or `mistral`) expands into a grid of runs over models, interfaces, skills, and tasks.
2. A session-start check confirms that every platform credential, skill bundle, and model is available and responds to a live probe before any work begins.
3. Each interface is built once and installed into a per-interface prepared virtual environment. Every run then clones that environment read-only, so runs do not reinstall anything or mutate shared state, and parallel sessions do not interfere.
4. For each run: a fresh seeded task instance is generated, `setup.py` provisions the platform and a `verify` step confirms it is ready, the sandboxed agent attempts the task using only its interface, then `teardown.py` removes what the agent created and a `verify` step confirms nothing was left behind.
5. A platform adapter (`evals/adapters/<platform>.py`) reads the deliverable back through the platform and runs the task's assertion suite.
6. One row per run is appended to `results/results.csv`.

### Eval families (`evals/`)

| Family | Example tasks |
|---|---|
| feature | `ingest`, `backfill`, `mit`, `validate`, `incremental_load`, `full_reload`, `pit` (point-in-time correct), `leakage` |
| training | `train`, `mdt`, `register`, `llm_finetuning` |
| inference | `batch`, `online`, `odt`, `skew`, `llm_serving`, `recsys`, `vector_search` |
| ops | `drift`, `prediction_monitoring`, `scheduled_jobs`, `alerting`, `lineage` |
| capstone | `ccfraud`, `airquality` |

## Quickstart

```bash
cd testbed
make install        # create .venv, install mlpab + evals + dev tools, link `mlpab` onto PATH
make setup          # interactive: authenticate agent engine(s) and set up platform(s)

# Verify a config is runnable: platform reachable, and each model answers a live probe
make check CONFIG=configs/treatments/aws/aws-claude.yaml

# Run a treatment session in tmux, detached from any terminal
mlpab start configs/treatments/databricks/databricks-claude.yaml
mlpab status                # list running sessions
mlpab attach <config.yaml>  # watch live (detach with Ctrl-b d)

# Or run inline
mlpab run configs/treatments/gcp/gcp-mistral.yaml

# Resume: re-running a config skips combos already completed — a valid row whose
# run folder still exists on disk. --retry also re-runs FAILED combos (no valid
# row, or a valid row whose folder was deleted), purging their rows + folders
# and running clean; --no-skip re-runs every combo, accumulating attempts.
mlpab run --retry configs/treatments/databricks/databricks-claude.yaml
```

Credentials are read from `testbed/.env`. macOS is the primary target. It uses
APFS copy-on-write clones for the per-run virtual environments and the macOS
Keychain for the Claude OAuth token. Other systems fall back to plain copies.

## Development

```bash
make test        # unit tests
make lint        # ruff check plus isort and format checks (no changes)
make fmt         # auto-fix: isort imports, then ruff format, then ruff --fix
```

## Results

```
testbed/results/results.csv     one row per run (the single results table)
testbed/results/<config>/...    per-run artifacts: agent.log, submission/, grading, commands
testbed/results/*.ipynb         hand-curated analysis notebooks
```

Each row records the identity of the cell (`model`, `platform`, `interface`,
`version`, `skills`, `category`, `task`, `n`) and its outcome (`valid`,
`success`, `asserts_passed`, `asserts_total`), along with cost, latency, and the
agent's tool-call counts. This is enough to reproduce and compare any value in
the table.

## Adding a platform interface

A platform lives in `configs/platforms/<platform>/`. It has one flat manifest per
interface (`cli.yaml`, `sdk.yaml`) describing how to build, install,
authenticate, and test that interface. It also has `setup.py` and `teardown.py`
(each supporting a `verify` mode) and a `skills/` bundle. A matching grader
adapter goes in `evals/adapters/<platform>.py` and implements the read-back
contract (`get_feature_table`, `read_rows`, `read_training_dataset`, and the
state reads).
