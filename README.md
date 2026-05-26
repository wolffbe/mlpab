# banter

Driver for Claude Code against vendor CLIs, MCP servers, and Python SDKs on top of MLE-bench.

```
banter/
  testbed/   ← the framework + configs + run results (everything in this README)
  thesis/    ← KTH LaTeX thesis this work supports (thesis/thesis.tex)
```

The thesis sources live at `thesis/thesis.tex` (KTH `kththesis` class). RQ statements below are taken from there; the configs in `testbed/configs/` are the experimental harness that answers them. Bibliography in `thesis/references.bib`, acronyms in `thesis/lib/acronyms.tex`.

Every shell command below assumes you are inside `testbed/`:

```bash
cd testbed
```

---

## Research questions

Numbered to match the thesis. Each maps 1:1 to one autoresearch config.

### RQ 1 — Framework efficacy (Main)
> To what extent can an autonomous research framework iteratively improve Model Context Protocol (MCP) tools, command-line interface (CLI) tools, and skills, without modifying the underlying platform API, such that coding agents achieve higher task success, lower token consumption, lower latency, and greater use of the remote platform over local Python?

Answered by: `configs/autoresearch_rq1_hopsworks.yaml`

### RQ 2 — Generalisation vs. specialisation
> Does a single interface produced by the framework generalize across ML task categories, or do interfaces tailored to specific categories yield better performance on task success, token consumption, latency, and use of the remote platform over local Python?

Answered by: `configs/autoresearch_rq2_hopsworks.yaml` (specialised per task type) + `configs/benchmark_hopsworks_{cli,mcp}.yaml` (cross-evaluation)

### RQ 3 — Attribution: skills vs. interface changes
> Within the framework's outputs, how much of the improvement in each of the four metrics is attributable to skills versus changes to MCP and CLI tools, and does this attribution differ across interface types and ML task categories?

Answered by: `configs/autoresearch_rq3_hopsworks_skills.yaml` (skills on top of RQ1/RQ2 winners) + downstream subtraction of metrics (see RQ 3 section below).

### Four metrics tracked across all RQs
| # | results.csv column | Direction | Maps to thesis term |
|---|---|---|---|
| 1 | `score` | maximize | task success |
| 2 | `total_tokens` | minimize | token consumption |
| 3 | `wall_time_s` | minimize | latency |
| 4 | `cli_calls` / `mcp_calls` ↑, `python_calls` ↓ | as shown | remote-platform delegation |

---

## One-time setup

```bash
make install          # creates testbed/.venv, installs banter + mle-bench, then runs `banter setup`
```

`banter setup` prompts for Kaggle credentials (legacy 32-char API key) and Claude Code auth (`api-key` or `login`).

Then install each Hopsworks interface manifest. `banter install` clones source into `cache/interfaces/<name>/<mode>/src/`, runs the manifest's `install:` steps to build the binary into `interfaces/<name>/<mode>/0/`, then runs `auth_command` interactively:

```bash
banter install configs/interfaces/hopsworks/cli.yaml    # Go binary → interfaces/hopsworks/cli/0/hops
banter install configs/interfaces/hopsworks/mcp.yaml    # once the MCP server manifest is filled in
banter install configs/interfaces/hopsworks/sdk.yaml    # pip install runs per-run; creates the v0 marker
```

Smoke-test the full pipeline (no interface, one tiny challenge):

```bash
banter run configs/benchmark_smoke_test.yaml
```

---

## Answering each RQ

Every autoresearch session writes:

- `results/autoresearch/<session_id>/results.csv` — per-session, one row per challenge run
- `results/autoresearch/<session_id>/cycles.jsonl` — one line per improvement cycle (hypothesis, change, before/after metrics, verdict)
- `results/autoresearch/<session_id>/transcript.log` — human-readable researcher transcript
- `results/results.csv` — global cross-session aggregate (every run from every session)

`banter reset configs/<rq_config>.yaml` clears improved interface versions (anything > v0) from the manifests referenced and (with confirmation) deletes all `results/autoresearch/*` sessions. v0 is always preserved.

### RQ 1 — `autoresearch_rq1_hopsworks.yaml`

Optimise one *general-purpose* version per interface mode for all 8 MLE-bench challenges side-by-side over 10 cycles. Produces the winning CLI / MCP / SDK versions used as inputs by RQ 3.

```bash
banter run configs/autoresearch_rq1_hopsworks.yaml
```

Budget: 10 cycles × 3 interfaces × 8 challenges ≈ 240 solver runs.

After the session ends, read `cycles.jsonl` to identify the best-verdict version per interface — those numbers feed `autoresearch_rq3_hopsworks_skills.yaml`:

```bash
cat results/autoresearch/<session_id>/cycles.jsonl
```

Reset:

```bash
banter reset configs/autoresearch_rq1_hopsworks.yaml
```

### RQ 2 — `autoresearch_rq2_hopsworks.yaml`

Specialise the interface per ML task type via `challenge_groups:` (image_classification, image_to_image, text_classification, image_regression, audio_classification, tabular, sequence_to_sequence). The researcher processes groups sequentially — 10 cycles per group, each cycle using only that group's challenges, producing one specialised version per (interface, group). CLI + MCP only (per the RQ's "MCP and CLI" wording — no SDK).

```bash
banter run configs/autoresearch_rq2_hopsworks.yaml
```

Budget: 7 groups × 10 cycles × 2 interfaces × challenges-per-group.

Every `cycles.jsonl` entry carries a `"group": "<task_type>"` tag so versions can be traced back to a task category.

Then cross-evaluate each specialised version against the OTHER challenges in its task type. Edit `configs/benchmark_hopsworks_cli.yaml` (and `..._mcp.yaml`): set `version:` to the specialised one, list the task-type challenges, run:

```bash
banter run configs/benchmark_hopsworks_cli.yaml
banter run configs/benchmark_hopsworks_mcp.yaml
```

Four-metric comparison:
- **Specialised wins on its category** if the per-task-type version outscores the RQ1 general-purpose version on the same challenges.
- **Specialised generalises** if the per-task-type version remains close to (or beats) the general-purpose version on *other* task-type challenges.

Reset:

```bash
banter reset configs/autoresearch_rq2_hopsworks.yaml
```

### RQ 3 — `autoresearch_rq3_hopsworks_skills.yaml`

Pin the RQ 1 (or RQ 2) winning interface versions and let autoresearch improve only a **skill bundle** on top. Edit the config to set `version:` per interface to the relevant winner:

```yaml
starting_interfaces:
  - {name: hopsworks, mode: cli, version: <winning_cli_v>}
  - {name: hopsworks, mode: mcp, version: <winning_mcp_v>}
  - {name: hopsworks, mode: sdk, version: <winning_sdk_v>}
```

Verify the pinned folders exist:

```bash
ls interfaces/hopsworks/cli/<winning_cli_v>/
ls interfaces/hopsworks/mcp/<winning_mcp_v>/
ls interfaces/hopsworks/sdk/<winning_sdk_v>/
```

Run:

```bash
banter run configs/autoresearch_rq3_hopsworks_skills.yaml
```

#### Attribution analysis (downstream of RQ 3)

To answer *how much of each metric's improvement is attributable to skills vs. MCP/CLI changes, and does this attribution differ across interface types and ML task categories?* — take three points per (interface_type, task_category) combination:

| Point | Where it comes from |
|---|---|
| **Baseline** (v0 interface, no skills) | Cycle 0 entries in RQ 1's `cycles.jsonl`, OR run `banter run configs/benchmark_hopsworks_<mode>.yaml` with `version: 0` across all 8 challenges. |
| **Interface-improved** (RQ 1 winner, no skills) | Final RQ 1 cycle entry per interface; or pin the winner in `benchmark_hopsworks_<mode>.yaml`. |
| **Interface + skills** (RQ 1 winner + autoresearched skills) | Final RQ 3 cycle entry per (interface, skills) pair. |

Per (interface_type, task_category) and per metric:

```
total_improvement   = improved_with_skills − baseline
interface_share     = improved_no_skills    − baseline
skills_share        = improved_with_skills  − improved_no_skills
```

Group rows from `results/results.csv` by `interface`+`mode` (interface type) and by the task category each `challenge_id` belongs to (use `challenge_groups:` in `configs/autoresearch_rq2_hopsworks.yaml` as the canonical category → challenge map).

Reset RQ 3 (preserves the pinned interface winners; clears only skill bundles + sessions):

```bash
banter reset configs/autoresearch_rq3_hopsworks_skills.yaml
```

---

## Ad-hoc commands

```bash
banter interfaces                                        # list installed interface manifests
banter skills                                            # list skill bundles under configs/skills/
banter run --challenge titanic --interface none          # one-off, no autoresearch
banter run --challenge titanic --interface hopsworks --mode cli --interface-version 1
banter run --challenge titanic --interface hopsworks --mode cli --skills hopsworks-essentials --skills-version 2
```

`banter run` accepts either a positional `CONFIG.yaml` (dispatches to autoresearch or benchmark based on the keys present) or the explicit `--challenge`/`--interface`/`--mode` flags for a one-off run.

---

## Repository layout

```
banter/
  README.md                                  # this file
  thesis/
    thesis.tex                               # KTH LaTeX thesis (main entry point)
    references.bib                           # bibliography
    lib/acronyms.tex                         # glossary / acronyms
    img/                                     # figures
  testbed/
    configs/
      autoresearch_rq1_hopsworks.yaml        # RQ 1 — framework efficacy
      autoresearch_rq2_hopsworks.yaml        # RQ 2 — generalisation vs specialisation
      autoresearch_rq3_hopsworks_skills.yaml # RQ 3 — attribution
      benchmark_smoke_test.yaml              # pipeline sanity check
      benchmark_hopsworks_{cli,mcp,sdk}.yaml # pinned-version benchmarks
      interfaces/<name>/<mode>.yaml          # install manifests with versions:
      skills/<bundle>/<v>/<skill>/SKILL.md   # skill bundles
    interfaces/<name>/<mode>/<v>/            # built binary artifacts (banter install / autoresearch)
    results/
      results.csv                            # cross-session aggregate
      autoresearch/<session_id>/             # per-session results + cycles.jsonl + transcript
      benchmark/<session_id>/                # per-session benchmark results
    cache/                                   # mle-bench data + pip + interface source clones
```
