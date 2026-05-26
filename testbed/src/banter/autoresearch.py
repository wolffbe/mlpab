"""Manager Claude that iteratively optimizes interface and skill configurations.

Spawns a `claude -p` researcher session. The researcher:
  - Runs ALL target challenges with the current config (one full round = one cycle)
  - Reads all run dirs and results.csv to understand solver behaviour across challenges
  - Creates new interface versions by appending to the manifest YAML's `versions:`
    dict (and copying the binary into interfaces/<name>/<mode>/<v+1>/ if any), or
    new skill bundle versions (configs/skills/<bundle>/<v+1>/<skill>/SKILL.md)
  - Evaluates the change by running ALL challenges again
  - Makes a conclusive decision by comparing cross-challenge aggregate metrics
  - Iterates until budget (max_cycles) is exhausted

Session layout:
    results/autoresearch/<session_id>/
        results.csv        # one row per individual challenge run
        cycles.jsonl       # one entry per improvement cycle
        prompt.txt         # researcher prompt (for debugging)
        transcript.jsonl   # raw researcher stream-json
        transcript.log     # human-readable researcher output
        <interface>/<mode>/<skills>/v<version>/<challenge>/
            prompt.txt     # solver prompt
            venv/
            submission/
            transcript.log
            grading.json
            ...

Entry points:
  run_autoresearch(config, testbed_root, runs_root)   — called by CLI
  build_researcher_prompt(...)                          — exposed for testing
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Goal:
    metric: str       # column name in results.csv, e.g. 'score', 'total_tokens'
    direction: str    # 'maximize' or 'minimize'


@dataclass
class Budget:
    max_cycles: int = 10                       # each cycle = one full round across all (interface, challenge) pairs
    max_cost_usd: float = float("inf")         # no cap by default


@dataclass
class InterfaceRef:
    name: str
    mode: str
    version: int | None = None      # None → use latest; integer → pin to that version

    def __str__(self) -> str:
        v = f" v{self.version}" if self.version is not None else ""
        return f"{self.name}/{self.mode}{v}"


_VALID_IMPROVE_SCOPES = {"interface", "skills"}


@dataclass
class AutoresearchConfig:
    challenges: list[str]
    starting_interfaces: list[InterfaceRef]   # one or more interfaces run side-by-side per cycle
    starting_skills: str
    goals: list[Goal]
    budget: Budget
    improve: list[str]              # subset of {"interface", "skills"}
    # Optional grouping: ML task type → list of challenges in that group.
    # When set, the researcher runs `max_cycles` cycles PER GROUP (each cycle
    # uses only that group's challenges), producing a specialised version
    # per group rather than one general-purpose version. `challenges:` is
    # auto-populated as the flattened union of all groups if left empty.
    challenge_groups: dict[str, list[str]] | None = None
    researcher_model: str = "claude-sonnet-4-6"
    auth: str = "api-key"
    model: str = "claude-sonnet-4-6"    # model for the solver agent


# Metrics where lower is better; everything else defaults to maximize.
_MINIMIZE_METRICS = {
    "total_tokens", "input_tokens", "output_tokens",
    "cost_usd", "wall_time_s",
    "llm_calls", "cli_calls", "mcp_calls", "sdk_calls",
    "python_calls", "bash_calls", "other_tool_calls",
}


def _parse_goal(entry: Any) -> Goal:
    """Accept either a plain metric name (str) or a {metric, direction} dict."""
    if isinstance(entry, str):
        direction = "minimize" if entry in _MINIMIZE_METRICS else "maximize"
        return Goal(metric=entry, direction=direction)
    metric = entry["metric"]
    direction = entry.get(
        "direction", "minimize" if metric in _MINIMIZE_METRICS else "maximize"
    )
    return Goal(metric=metric, direction=direction)


def _parse_interfaces(data: dict) -> list[InterfaceRef]:
    """Accept either `starting_interfaces: [{name, mode}, ...]` (new, plural)
    or `starting_interface: {name, mode}` (legacy, singular)."""
    raw = data.get("starting_interfaces")
    if raw is None and "starting_interface" in data:
        single = data["starting_interface"]
        raw = [single] if isinstance(single, dict) else []
    if not raw:
        return [InterfaceRef(name="none", mode="none")]
    out: list[InterfaceRef] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(
                f"`starting_interfaces` entries must be mappings with `name`+`mode`; "
                f"got {entry!r}"
            )
        version = entry.get("version")
        if version is not None:
            version = int(version)
        out.append(
            InterfaceRef(
                name=entry.get("name", "none"),
                mode=entry.get("mode", "none"),
                version=version,
            )
        )
    return out


def load_config(path: Path) -> AutoresearchConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    goals = [_parse_goal(g) for g in (data.get("goals") or [])]
    bud = data.get("budget") or {}
    raw_improve = data.get("improve") or ["interface", "skills"]
    if isinstance(raw_improve, str):
        raw_improve = [raw_improve]
    improve = [s.strip() for s in raw_improve]
    invalid = set(improve) - _VALID_IMPROVE_SCOPES
    if invalid:
        raise ValueError(
            f"Invalid `improve` values: {invalid}. "
            f"Must be a subset of {sorted(_VALID_IMPROVE_SCOPES)}."
        )
    # Parse optional challenge_groups (ML task type → challenges in that group).
    raw_groups = data.get("challenge_groups")
    groups: dict[str, list[str]] | None = None
    if raw_groups:
        if not isinstance(raw_groups, dict):
            raise ValueError(
                f"`challenge_groups` must be a mapping of group_name → [challenges]; "
                f"got {type(raw_groups).__name__}."
            )
        groups = {}
        for name, items in raw_groups.items():
            if not isinstance(items, list) or not items:
                raise ValueError(
                    f"challenge_groups[{name!r}] must be a non-empty list of challenge ids."
                )
            groups[str(name)] = [str(c) for c in items]

    challenges = data.get("challenges") or []
    if not challenges and groups:
        # Flatten groups into challenges (deduplicated, preserves order of first occurrence).
        seen: set[str] = set()
        challenges = [c for items in groups.values() for c in items if not (c in seen or seen.add(c))]

    return AutoresearchConfig(
        challenges=challenges,
        starting_interfaces=_parse_interfaces(data),
        starting_skills=data.get("starting_skills", "none"),
        goals=goals,
        budget=Budget(
            max_cycles=int(bud.get("max_cycles", bud.get("max_runs", 10))),
            max_cost_usd=float(bud.get("max_cost_usd", float("inf"))),
        ),
        improve=improve,
        challenge_groups=groups,
        researcher_model=data.get("researcher_model", data.get("model", "claude-sonnet-4-6")),
        auth=data.get("auth", "api-key"),
        model=data.get("model", "claude-sonnet-4-6"),
    )


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _recent_results_table(runs_root: Path, n: int = 40) -> str:
    results_path = runs_root / "results.csv"
    if not results_path.exists():
        return "(no results yet)"
    with open(results_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "(no results yet)"
    header = (
        f"{'challenge':<35} {'iface/mode':<18} {'skills':<18} {'ver':<4} "
        f"{'score':<8} {'tokens':<8} {'wall_s':<7} {'py_calls':<5} run_dir"
    )
    sep = "-" * 150
    lines = [header, sep]
    for row in rows[-n:]:
        lines.append(
            f"{row.get('challenge_id', '?'):<35} "
            f"{row.get('interface', '?')}/{row.get('mode', '?'):<16} "
            f"{row.get('skills', '?'):<18} "
            f"{row.get('version', ''):<4} "
            f"{row.get('score', ''):<8} "
            f"{row.get('total_tokens', ''):<8} "
            f"{row.get('wall_time_s', ''):<7} "
            f"{row.get('python_calls', ''):<5} "
            f"{row.get('run_dir', '')}"
        )
    return "\n".join(lines)


def _scan_interfaces(testbed_root: Path) -> str:
    idir = testbed_root / "configs" / "interfaces"
    if not idir.exists():
        return "  (none)"
    lines = []
    for name_dir in sorted(idir.iterdir()):
        if not name_dir.is_dir():
            continue
        for yaml_file in sorted(name_dir.iterdir()):
            if not yaml_file.is_file() or yaml_file.suffix != ".yaml":
                continue
            type_ = yaml_file.stem
            manifest = yaml.safe_load(yaml_file.read_text()) or {}
            raw = manifest.get("versions") or {}
            if isinstance(raw, dict):
                versions = sorted(int(k) for k in raw.keys() if str(k).isdigit() or isinstance(k, int))
            elif isinstance(raw, list):
                versions = sorted(int(e.get("version", 0)) for e in raw if isinstance(e, dict))
            else:
                versions = []
            if versions:
                lines.append(
                    f"  {name_dir.name}/{type_}  versions={versions}  latest={max(versions)}"
                )
    return "\n".join(lines) or "  (none)"


def _scan_skills(testbed_root: Path) -> str:
    sdir = testbed_root / "configs" / "skills"
    lines = ["  none  (control — no skills injected into the solver)"]
    if not sdir.exists():
        return "\n".join(lines)
    for bundle_dir in sorted(sdir.iterdir()):
        if not bundle_dir.is_dir() or bundle_dir.name.startswith("."):
            continue
        versions = sorted(
            int(p.name) for p in bundle_dir.iterdir() if p.is_dir() and p.name.isdigit()
        )
        if not versions:
            continue
        latest_ver_dir = bundle_dir / str(max(versions))
        skill_names = sorted(
            d.name
            for d in latest_ver_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        )
        lines.append(
            f"  {bundle_dir.name}  versions={versions}  skills={skill_names}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Researcher prompt
# ---------------------------------------------------------------------------


def _current_version(testbed_root: Path, iface: InterfaceRef) -> int:
    """Latest version listed in the manifest for an interface; 0 if absent."""
    if iface.name == "none" and iface.mode == "none":
        return 0
    mpath = testbed_root / "configs" / "interfaces" / iface.name / f"{iface.mode}.yaml"
    if not mpath.exists():
        return 0
    try:
        data = yaml.safe_load(mpath.read_text()) or {}
    except Exception:
        return 0
    versions = data.get("versions") or {}
    if isinstance(versions, dict):
        ints = [int(k) for k in versions.keys() if str(k).lstrip("-").isdigit() or isinstance(k, int)]
    elif isinstance(versions, list):
        ints = [int(e.get("version", 0)) for e in versions if isinstance(e, dict)]
    else:
        ints = []
    return max(ints) if ints else 0


def build_researcher_prompt(
    config: AutoresearchConfig,
    testbed_root: Path,
    runs_root: Path,
    session_id: str,
    banter_bin: Path,
) -> str:
    goals_lines = "\n".join(
        f"  {i + 1}. {g.direction.upper()} `{g.metric}`"
        for i, g in enumerate(config.goals)
    )
    if config.challenge_groups:
        challenges_lines = "\n".join(
            f"  [{group}]\n" + "\n".join(f"    - {c}" for c in items)
            for group, items in config.challenge_groups.items()
        )
    else:
        challenges_lines = "\n".join(f"  - {c}" for c in config.challenges)
    avail_interfaces = _scan_interfaces(testbed_root)
    avail_skills = _scan_skills(testbed_root)
    recent_results = _recent_results_table(runs_root)
    n_challenges = len(config.challenges)
    n_interfaces = len(config.starting_interfaces)
    runs_per_cycle = n_challenges * n_interfaces

    # Per-interface current/next versions
    iface_versions = {
        str(iface): (_current_version(testbed_root, iface), _current_version(testbed_root, iface) + 1)
        for iface in config.starting_interfaces
    }
    interfaces_table = "\n".join(
        f"  - {iface}  (current v{cur}, next would be v{nxt})"
        for iface, (cur, nxt) in iface_versions.items()
    )

    # Improvement scope description
    can_interface = "interface" in config.improve
    can_skills = "skills" in config.improve
    if can_interface and can_skills:
        scope_desc = (
            "any of the **interface manifests** listed above (append entries to "
            "`versions:`) AND **skill bundles** (configs/skills/...)"
        )
        scope_deny = ""
    elif can_interface:
        scope_desc = (
            "any of the **interface manifests** listed above (append entries to "
            "`versions:` in configs/interfaces/<name>/<mode>.yaml)"
        )
        scope_deny = "❌ Do NOT create or modify skill bundles."
    else:
        scope_desc = "**skill bundles only**"
        scope_deny = "❌ Do NOT create or modify interface manifests."

    cycles_path = runs_root / "cycles.jsonl"
    first_iface = config.starting_interfaces[0]
    cur_iface_version, next_iface_version = iface_versions[str(first_iface)]

    # When challenge_groups: is set, the session is run PER GROUP. Each group
    # gets its own pass of max_cycles cycles (using only that group's
    # challenges), producing a specialised version per group. Sketch a
    # per-group section the researcher can copy into the cycle log.
    if config.challenge_groups:
        group_section_lines = [
            "## Challenge groups (RQ2-style: per-group specialisation)",
            "",
            f"This session has {len(config.challenge_groups)} ML task type(s). Process them",
            f"sequentially: for each group, run {config.budget.max_cycles} cycles using ONLY that",
            "group's challenges, producing a specialised interface version per group.",
            "",
            "Group order + challenges:",
        ]
        for gname, gitems in config.challenge_groups.items():
            group_section_lines.append(f"  [{gname}]")
            for c in gitems:
                group_section_lines.append(f"    - {c}")
        group_section_lines.append("")
        group_section_lines.append(
            "Tag every cycle.jsonl entry with `\"group\": \"<group_name>\"` so the resulting"
        )
        group_section_lines.append(
            "interface versions can be cross-referenced to ML task type during analysis."
        )
        group_section_lines.append("")
        group_section_lines.append(
            f"Total runs in this session: {len(config.challenge_groups)} groups × "
            f"{config.budget.max_cycles} cycles × n_interfaces × challenges_per_group."
        )
        groups_block = "\n".join(group_section_lines) + "\n\n---\n"
    else:
        groups_block = ""
    skills_label = config.starting_skills if config.starting_skills != "none" else "no_skills"
    run_dir_example = (
        f"{runs_root}/{first_iface.name}/{first_iface.mode}/"
        f"{skills_label}/v{cur_iface_version}/"
        f"{config.challenges[0] if config.challenges else 'challenge-id'}/"
    )

    # Pre-render the "run every (interface, challenge) pair" block once
    # (used for baseline + per-cycle evaluation snippets below). When an
    # interface pins a `version:`, pass it as --interface-version so the
    # solver runs against the pinned variant (used heavily by RQ3-style
    # skills-only autoresearch on top of frozen RQ1/RQ2 interface winners).
    eval_block_lines = []
    for iface in config.starting_interfaces:
        version_flag = f" --interface-version {iface.version}" if iface.version is not None else ""
        for c in (config.challenges or ["<challenge>"]):
            eval_block_lines.append(
                f"{banter_bin} run \\\n"
                f"  --challenge {c} \\\n"
                f"  --interface {iface.name} \\\n"
                f"  --mode {iface.mode}{version_flag} \\\n"
                f"  --skills {config.starting_skills} \\\n"
                f"  --runs-root {runs_root}"
            )
    eval_block = "\n".join(eval_block_lines)

    return f"""You are a research agent managing a Claude Code MLE-bench testbed. Your job is to iteratively improve solver performance by modifying interfaces and/or skills, guided by the goals below.

## Session
- ID: {session_id}
- Testbed root: {testbed_root}
- Session directory: {runs_root}
- Per-session results CSV: {runs_root}/results.csv       ← this session only
- Global results CSV: {testbed_root}/results/results.csv ← every run ever (cross-session)
- Cycle log: {cycles_path}

## Goals
{goals_lines}

## Budget
- Max improvement cycles: {config.budget.max_cycles}
- Each cycle runs ALL {n_challenges} challenge(s) across ALL {n_interfaces} interface(s) = {runs_per_cycle} runs/cycle
- Total individual banter runs across the session: up to {config.budget.max_cycles * runs_per_cycle}
- Max total solver cost: {('unlimited' if config.budget.max_cost_usd == float('inf') else f'${config.budget.max_cost_usd:.2f} USD')}

## Target challenges
{challenges_lines}

## Starting configuration — {n_interfaces} interface(s) run side-by-side per cycle
{interfaces_table}
- Skills: {config.starting_skills}

---

## Improvement scope

You are allowed to improve: {scope_desc}
{scope_deny}

---

## How the testbed works

Each evaluation run:
1. Creates a fresh Python venv
2. Prepares competition data from the MLE-bench cache
3. Installs the interface if configured (CLI binary / MCP server / SDK)
4. Injects skill SKILL.md files into the solver's `.claude/skills/` if a bundle is chosen
5. Runs `claude -p <task_prompt>` — the **solver** Claude Code instance
6. Grades the solver's `submission.csv` with MLE-bench

You control what the solver sees through:
- **Interface prompt** — added to the task prompt (e.g. "use the `hops` CLI")
- **Skill bundles** — named Claude Code skills the solver can invoke
- **Interface config** — install steps, binary name, MCP servers

---

## File structure

### Interface manifests

A single YAML file is the source of truth for each interface:

```
{testbed_root}/configs/interfaces/<name>/<type>.yaml
```

**Manifest schema:**
```yaml
# Install-time (banter install):
repo: https://github.com/...    # optional
ref: main
install:                         # one-time build steps
  - go build -o $INTERFACE_DIR/my-cli .
auth_command: my-cli login       # optional

# Runtime defaults (per-run; versions can override):
binary: my-cli                   # CLI binary name
runtime_install:                 # steps run per evaluation
  - cp $INTERFACE_DIR/my-cli $VIRTUAL_ENV/bin/my-cli
# mcp_servers:                   # for MCP interfaces
#   my_service:
#     command: my-mcp-server

# Versions (0 = base; you append improved versions):
versions:
  0:
    prompt: |
      ...base prompt seen by the solver...
  # Improved versions you create:
  # 1:
  #   prompt: |
  #     ...refined prompt...
  #   # Optional per-version overrides:
  #   # runtime_install:
  #   #   - ...
```

`$INTERFACE_DIR` → `{testbed_root}/interfaces/<name>/<type>/<version>/` (binary artifacts)
`$VIRTUAL_ENV`   → per-run venv (copy/install the tool here so the solver can invoke it)

**To create a new interface version** (almost always a prompt tweak):

1. Pick which interface to evolve this cycle (e.g. `{first_iface.name}/{first_iface.mode}`)
2. Open `{testbed_root}/configs/interfaces/<name>/<mode>.yaml`
3. Append a new entry under `versions:` keyed by `<cur+1>` with a refined `prompt:`
4. If the binary should be reused (typical), copy it:
   ```bash
   mkdir -p {testbed_root}/interfaces/<name>/<mode>/<cur+1>
   cp -r {testbed_root}/interfaces/<name>/<mode>/<cur>/. \\
         {testbed_root}/interfaces/<name>/<mode>/<cur+1>/
   ```
   (skip this step if the interface has no binary — SDK/MCP)

The testbed always picks the **highest-numbered version** automatically.

### Skill bundles

```
{testbed_root}/configs/skills/<bundle>/<version>/<skill_name>/
    SKILL.md        # plain Markdown — what the solver sees as a named skill
```

**To create a new bundle:**
```bash
mkdir -p {testbed_root}/configs/skills/my_bundle/0/skill_name
cat > {testbed_root}/configs/skills/my_bundle/0/skill_name/SKILL.md << 'SKILL'
<skill instructions here>
SKILL
```

**To add a new version of an existing bundle:**
```bash
cp -r {testbed_root}/configs/skills/<bundle>/0 {testbed_root}/configs/skills/<bundle>/1
# Edit files inside version 1
```

### Run directory layout

Each challenge run produces:
```
{run_dir_example}
    prompt.txt       # solver task prompt
    venv/
    submission/
    transcript.log   # human-readable solver transcript
    grading.json
```

---

## Running evaluations

```bash
# Run ONE challenge with a specific config
{banter_bin} run \\
  --challenge <challenge_id> \\
  --interface <name> \\
  --mode <type_or_none> \\
  --skills <bundle_or_none> \\
  --runs-root {runs_root}
```

Always pass `--runs-root {runs_root}`.

Key columns in `{runs_root}/results.csv`:
| Column | Meaning |
|--------|---------|
| `score` | MLE-bench accuracy 0.0–1.0 |
| `medal` | gold/silver/bronze/None |
| `total_tokens` | input + output tokens |
| `wall_time_s` | elapsed seconds |
| `python_calls` | Bash calls invoking a Python interpreter |
| `cli_calls` | calls to the interface CLI binary |
| `cost_usd` | estimated solver cost |
| `run_dir` | path to run folder (contains prompt.txt, transcript.log, grading.json) |

---

## Cycle log

After **every** complete cycle (all interfaces × all challenges run), append
one JSON line to `{cycles_path}`. Group metrics PER INTERFACE so improvement
attribution is unambiguous:

```json
{{
  "cycle": 1,
  "scope": "interface:{first_iface.name}/{first_iface.mode}",
  "hypothesis": "Adding a hops feature-group example to the CLI prompt reduces python_calls",
  "change": "Appended versions.{next_iface_version} to configs/interfaces/{first_iface.name}/{first_iface.mode}.yaml",
  "before": {{
    "per_interface": {{
      "{first_iface.name}/{first_iface.mode}": {{
         "avg_score": 0.62, "avg_total_tokens": 8400, "avg_wall_time_s": 1240,
         "avg_python_calls": 9.5, "avg_cli_calls": 0.3
      }}
      // ... one block per interface in scope ...
    }}
  }},
  "after": {{
    "per_interface": {{
      "{first_iface.name}/{first_iface.mode}": {{
         "avg_score": 0.68, "avg_total_tokens": 6900, "avg_wall_time_s": 1100,
         "avg_python_calls": 5.2, "avg_cli_calls": 3.1
      }}
    }}
  }},
  "verdict": "positive",
  "verdict_reason": "{first_iface.name}/{first_iface.mode}: +9.7% score, -18% tokens, +cli_calls. Other interfaces unchanged.",
  "keep": true
}}
```

`verdict` must be one of: `positive` | `negative` | `neutral`
`scope` should name the specific interface modified, e.g. `interface:hopsworks/cli`,
`interface:hopsworks/mcp`, or `skills:<bundle>`.

Write with:
```bash
echo '<json line>' >> {cycles_path}
```

---

## Current state

### Available interfaces
{avail_interfaces}

### Available skill bundles
{avail_skills}

### Session results so far
```
{recent_results}
```

---

{groups_block}## Installation responsibility (YOU own this)

The solver Claude assumes every interface it is asked to use is already
installed and ready. You — the researcher — are responsible for ensuring
this is true before EVERY `banter run` invocation. Concretely:

### Pre-flight: install all base (v0) interfaces

Before Cycle 0, make sure every interface listed under "Starting
configuration" has a working v0:

```bash
{chr(10).join(
    f"{banter_bin} install {testbed_root}/configs/interfaces/{iface.name}/{iface.mode}.yaml"
    for iface in config.starting_interfaces
    if not (iface.name == 'none' and iface.mode == 'none')
)}
```

`banter install` is idempotent — it clones/updates the source repo, runs
the manifest's one-time `install:` steps (writing the binary into
`{testbed_root}/interfaces/<name>/<mode>/0/`), and runs the `auth_command`
if any. Run it once per interface before any solver runs.

For SDK-type interfaces the `install` step is a `pip install` that happens
per-run in the solver's venv — but you must still create v0's metadata.
Running `banter install` on an SDK manifest writes the `interfaces/.../0/`
folder (it'll be empty for SDKs — that's expected).

### When you ADD a new version `<v+1>` for an interface

The solver expects `interfaces/<name>/<mode>/<v+1>/` to exist and (for CLI
interfaces with a `binary:` key) to contain the binary. After you append
the new entry to the manifest's `versions:` map, provision the artifact
folder yourself:

```bash
# For CLI interfaces — copy the v_cur binary across (the binary itself is
# typically unchanged; only the prompt/config differs).
mkdir -p {testbed_root}/interfaces/<name>/<mode>/<v+1>
cp -r {testbed_root}/interfaces/<name>/<mode>/<v>/. \\
      {testbed_root}/interfaces/<name>/<mode>/<v+1>/

# For SDK / MCP — just ensure the version folder exists (may be empty).
mkdir -p {testbed_root}/interfaces/<name>/<mode>/<v+1>
```

Verify with `ls`:
```bash
ls {testbed_root}/interfaces/<name>/<mode>/<v+1>/
```

### If a `banter run` fails with "binary not found" / "Run: banter install ..."

You forgot to provision either v0 (run `banter install`) or v+1 (run the
mkdir/cp above). Fix the install state, then retry — do NOT log the failed
run as a result; results.csv should reflect successful evaluations only.

### A clean cycle checklist

1. Verify install state of every interface in scope (one `ls` per interface).
2. Run the (interface × challenge) evaluation block.
3. Inspect results.csv + transcripts.
4. Propose ONE change → append new `versions:` entry → provision its folder.
5. Re-run the evaluation block.
6. Log cycle entry with verdict.

---

## Research process

### Cycle 0 — Establish baseline (run once, first thing)

Run every (interface × challenge) pair once with the starting config. That's
{runs_per_cycle} runs total ({n_interfaces} interface(s) × {n_challenges} challenge(s)):

```bash
{eval_block}
```

Read every run's `transcript.log` and `grading.json` to understand solver behaviour
PER INTERFACE. Then log cycle 0 in `{cycles_path}` with `"change": "baseline"`
and `"before": null`, recording averaged metrics grouped by interface (see
the cycle log schema above).

### Cycles 1–{config.budget.max_cycles} — Improvement cycles

For each cycle:

1. **Analyse all runs** — read `{runs_root}/results.csv` and group rows by
   `interface`+`mode`. Inspect `<run_dir>/transcript.log` for representative
   challenges. Look for patterns PER INTERFACE:
   - Does one interface consistently score worse across challenges?
   - Are token counts or python_calls particularly high for one interface?
   - Does an interface prompt miss context that hurts every challenge?

2. **Hypothesize** — ONE specific, testable change to ONE of the
   {n_interfaces} interfaces in scope. Examples:
   - "CLI prompt doesn't tell the solver to use `hops fg create` — every
     challenge fell back to local Python for feature group creation"
   - "MCP tool list isn't documented in the prompt — solver ignored it
     and used Python instead"

3. **Implement** — append a new entry under `versions:` in the relevant
   interface's manifest AND provision its `interfaces/<name>/<mode>/<v+1>/`
   artifact folder per the "Installation responsibility" section above.
   The solver will fail fast if v+1's binary is missing, so do this BEFORE
   the next `banter run`. You may modify ONE interface per cycle for clean
   attribution.

4. **Evaluate** — re-run ALL {runs_per_cycle} (interface × challenge) pairs:
```bash
{eval_block}
```

5. **Decide conclusively** — compare AVERAGED metrics across challenges
   PER INTERFACE between before and after:
   - If the modified interface's avg_score improved AND no challenge regressed
     significantly → `positive`, keep
   - If avg_score unchanged but token/call counts improved consistently
     for that interface → `positive`, keep
   - If results are mixed → `neutral`, investigate further
   - If avg_score dropped on any challenge for that interface → `negative`,
     drop the new version (or pin to its previous version) and try again

6. **Log** — append a cycle entry to `{cycles_path}` with per-interface
   per-challenge breakdown AND per-interface averaged metrics.

**Budget tracking**: count every `banter run` call. Stop when total runs ≥
{config.budget.max_cycles * runs_per_cycle}.

### Final summary

When budget is exhausted or goals are met, output exactly one JSON object:
```json
{{
  "session_id": "{session_id}",
  "cycles_completed": 0,
  "best_versions": {{
    // best version per interface, e.g.:
    // "hopsworks/cli": 2,
    // "hopsworks/mcp": 1,
    // "hopsworks/sdk": 0
  }},
  "best_avg_score_per_interface": {{}},
  "positive_changes": [],
  "negative_changes": [],
  "recommendations": []
}}
```

Begin by running the pre-flight `banter install` for every interface in scope,
THEN start Cycle 0: check the results CSV, run the baseline across all
(interface × challenge) pairs if needed, inspect every run, then start Cycle 1.
Whenever you create a new version, provision its interfaces/.../<v+1>/ folder
BEFORE the next `banter run` — the solver will refuse to start without it.
"""


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------


def _display_event(event: dict[str, Any]) -> None:
    """Print a human-readable line for a stream-json event from the researcher."""
    etype = event.get("type")
    if etype == "assistant":
        content = (event.get("message") or {}).get("content") or []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = (block.get("text") or "").rstrip()
                if text:
                    for line in text.splitlines():
                        print(f"[researcher] {line}", flush=True)
            elif btype == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input") or {}
                if name == "Bash":
                    snippet = (inp.get("command") or "").strip().splitlines()[0][:120]
                    print(f"[researcher:bash] {snippet}", flush=True)
                elif name in ("Read", "Write", "Edit"):
                    fpath = inp.get("file_path", "?")
                    print(f"[researcher:{name.lower()}] {fpath}", flush=True)
                else:
                    print(f"[researcher:tool] {name}", flush=True)
    elif etype == "result":
        cost = event.get("total_cost_usd", 0)
        turns = event.get("num_turns", "?")
        stop = event.get("stop_reason") or event.get("subtype", "?")
        print(
            f"\n[autoresearch] researcher finished: {turns} turns, "
            f"${cost:.4f} researcher cost, stop={stop}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_autoresearch(
    config: AutoresearchConfig,
    testbed_root: Path,
    runs_root: Path,
) -> None:
    if shutil.which("claude") is None:
        raise RuntimeError("`claude` CLI not found on PATH. Install Claude Code first.")

    banter_bin = testbed_root / ".venv" / "bin" / "banter"
    if not banter_bin.exists():
        raise RuntimeError(
            f"`banter` not found at {banter_bin}. Run `make install` in the testbed first."
        )

    session_id = uuid.uuid4().hex[:8]
    session_dir = runs_root / "autoresearch" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_researcher_prompt(config, testbed_root, session_dir, session_id, banter_bin)
    (session_dir / "prompt.txt").write_text(prompt)

    transcript_path = session_dir / "transcript.jsonl"
    stderr_path = session_dir / "researcher.stderr.log"

    env = os.environ.copy()
    env.pop("ANTHROPIC_BASE_URL", None)
    if config.auth == "login":
        env.pop("ANTHROPIC_API_KEY", None)
    venv_bin = str(testbed_root / ".venv" / "bin")
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("KAGGLE_CONFIG_DIR", str(testbed_root / ".kaggle"))

    cmd = [
        "claude",
        "-p", prompt,
        "--permission-mode", "bypassPermissions",
        "--model", config.researcher_model,
        "--max-turns", "500",
        "--output-format", "stream-json",
        "--verbose",
    ]

    n_ifaces = len(config.starting_interfaces)
    n_chals = len(config.challenges)
    iface_list = ", ".join(str(i) for i in config.starting_interfaces)
    print(
        f"[autoresearch] session={session_id}\n"
        f"  interfaces={iface_list}\n"
        f"  goals={[f'{g.direction}({g.metric})' for g in config.goals]}\n"
        f"  budget={config.budget.max_cycles} cycles × {n_ifaces} interfaces × {n_chals} challenges "
        f"= up to {config.budget.max_cycles * n_ifaces * n_chals} runs\n"
        f"  cost_cap={'∞' if config.budget.max_cost_usd == float('inf') else f'${config.budget.max_cost_usd:.2f}'}",
        flush=True,
    )
    print(f"[autoresearch] session dir: {session_dir}", flush=True)
    print()

    def _on_line(raw_line: str) -> None:
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            return
        _display_event(event)

    # Researcher Claude shares the solver's rate-limit retry helper: 12h budget,
    # exponential back-off, automatic resume on 429/529/overloaded.
    from banter import claude_runner
    try:
        claude_runner.run_with_retry(
            cmd=cmd,
            cwd=testbed_root,
            env=env,
            transcript_path=transcript_path,
            stderr_path=stderr_path,
            on_line=_on_line,
            log_prefix="autoresearch",
        )
    except KeyboardInterrupt:
        print("\n[autoresearch] interrupted by user.", flush=True)

    try:
        from banter import results as results_mod
        results_mod.write_readable_transcript(transcript_path, session_dir / "transcript.log")
    except Exception:
        pass

    print(f"\n[autoresearch] done. session dir: {session_dir}", flush=True)
