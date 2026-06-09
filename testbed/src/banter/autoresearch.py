"""Manager Claude (the researcher) that iteratively improves the engineer's interface.

Hierarchy: a RUN has many INCREMENTS; each spans many TASKS (ML task types);
each task spans its CHALLENGES.

The researcher MODIFIES THE INTERFACE handed to the engineer (never the engineer
prompt). Each increment has its own interface copy at `v<N>/interface/` (v0
auto-seeded from the committed base — or a `prev_run`+`prev_version`
continuation — and pre-built; v>0 prepared via `banter prepare-version`, then
source-edited). `banter run --interface-dir v<N>/interface` force-rebuilds the
edits and the engineer uses them.

Layout under `results/autoresearch/`:
    experiments.csv        # GLOBAL table: one row per (config, version, task,
                           #   challenge) across ALL runs (written live by the runner)
    analysis.ipynb         # GLOBAL analysis (composite, baseline-vs-best, charts)
    <run_id>/<platform>/<interface>/<skills>/
        prompt.txt         # researcher prompt (for debugging)
        researcher.log     # human-readable live researcher view
        CHANGELOG.md       # per-version narrative (researcher appends after each version)
        report.md          # final report (researcher writes at the end)
        v0/, v1/, …
            interface/     # the per-version interface copy (built)
            <task>/<challenge>/   # one engineer run + its artifacts (incl. engineer.log)
    (No per-run results.csv or raw transcript — results go to the global
     experiments.csv; versions scored via the `normalized_composite` MCP tool.)

Budget is graceful: `budget.max_seconds` caps COMPUTE time (wall clock minus
rate-limit waiting). The researcher runs `banter budget-check` BEFORE each new
version and stops at the version boundary when exhausted (no hard kill).

Entry points:
  run_autoresearch(config, testbed_root, runs_root)   — called by CLI
  build_researcher_prompt(...)                          — exposed for testing
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from banter import claude_runner, interfaces, streaming


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Goal:
    metric: str       # column name in results.csv, e.g. 'score', 'total_tokens'
    direction: str    # 'maximize' or 'minimize'


@dataclass
class Budget:
    # `iterations` = TOTAL versions INCLUDING the v0 baseline. v0 is always
    # created (fresh from the committed base, or seeded from `prev_run`/
    # `prev_version`); v1..v<iterations-1> are improvements. iterations=4 → v0..v3.
    iterations: int = 11
    # `max_cost_usd` caps COMBINED engineer + researcher spend (total_cost).
    # `max_seconds` caps COMPUTE time — wall clock minus rate-limit waiting
    # (from the ledger). Both default UNLIMITED (no silent cap; run bounded only
    # by `iterations`). Graceful: researcher checks `banter budget-check` before
    # each new version and stops when exceeded.
    max_cost_usd: float = float("inf")
    max_seconds: float = float("inf")

    # `max_increments` = improvements on top of v0 = iterations - 1. Derived
    # property kept so existing code/prompts keep working.
    @property
    def max_increments(self) -> int:
        return max(0, self.iterations - 1)

    # Back-compat: older configs/code referenced `max_cycles`.
    @property
    def max_cycles(self) -> int:
        return self.max_increments


@dataclass
class InterfaceRef:
    platform: str
    interface: str
    version: int | None = None      # None/0 → base manifest; >0 → a session version
    # Session that produced a pinned `version` > 0 (e.g. RQ3 pinning an RQ1
    # winner). None means "this session" / base.
    session: str | None = None
    # Platform config path this ref came from (when referenced by `config:`).
    config: str | None = None

    def __str__(self) -> str:
        v = f" v{self.version}" if self.version is not None else ""
        s = f" @{self.session}" if self.session else ""
        return f"{self.platform}/{self.interface}{v}{s}"


_VALID_IMPROVE_SCOPES = {"interface", "skills"}


@dataclass
class AutoresearchConfig:
    # tasks: ML task type → its challenges. Unifying structure across all RQ
    # configs (RQ1 = one challenge/task; RQ2 = several; RQ3 = RQ1 with skills).
    # Each version spans ALL tasks; `task` is a grouping dimension on
    # results.csv, not a separate per-task version chain.
    tasks: dict[str, list[str]]
    challenges: list[str]                # flattened union of all tasks (derived)
    interfaces: list[InterfaceRef]       # platforms run side-by-side per increment
    skills: str                          # starting skill bundle name, or "none"
    docs: str                            # docs bundle name, or "none"
    goals: list[Goal]
    budget: Budget
    improve: list[str]              # subset of {"interface", "skills"}
    # Engineer is the controlled instance; the researcher controls it. Defaults:
    # engineer (cheap, fast iterations) runs Sonnet; researcher (planner / harder
    # reasoning over many runs) runs Opus.
    engineer_model: str = "claude-sonnet-4-6"
    engineer_auth: str = "api-key"
    researcher_model: str = "claude-opus-4-5"
    researcher_auth: str = "api-key"
    # Pin this run's id (`run` column + run-dir prefix). Unset (None) →
    # auto-increment via `next_session_id(results/autoresearch)`. Useful for
    # stable, named runs (e.g. `run: rq1-cli-skills-v2`).
    run: int | str | None = None
    # Take a PAST version as the base for further improvement. v0 is seeded from
    # `results/autoresearch/<prev_run's folder>/v<prev_version>/interface`
    # instead of the committed base. Both set together, or neither.
    prev_run: int | None = None
    prev_version: int | None = None
    # Experiment-design metadata (set by generated treatment configs). All
    # optional, default None so existing configs are unaffected. When
    # `experiment` is set, the end-of-run hook propagates this treatment's
    # outcome into the master `results/experiments.csv` table.
    experiment: int | None = None
    research_question: int | None = None
    treatment: int | None = None
    optimization_variable: str | None = None   # univariate | bivariate | multivariate
    language: str | None = None
    # Absolute path of the config file this was loaded from. The experiment
    # table keys off this (the config IS the treatment's identity).
    config_path: str | None = None
    # REST-endpoint coverage policy: `{"whitelist": [pat, …], "blacklist": [pat, …]}`,
    # each pattern "<METHOD> <path-regex>". Drives `whitelist_hits` (maximize) /
    # `blacklist_hits` (minimize). Empty = no scoring.
    endpoints: dict = field(default_factory=lambda: {"whitelist": [], "blacklist": []})


# Metrics where lower is better; everything else defaults to maximize.
_MINIMIZE_METRICS = {
    "eng_input_tokens", "eng_output_tokens", "eng_total_tokens", "eng_cost_usd", "eng_wall_time_s",
    "res_input_tokens", "res_output_tokens", "res_total_tokens", "res_cost_usd", "res_wall_time_s",
    "total_tokens", "total_cost", "total_wall_time_s",
    "llm_calls", "cli_calls", "mcp_calls", "sdk_calls",
    "python_calls", "bash_calls", "other_tool_calls",
    "blacklist_hits",
}


def _as_pattern_list(v: Any) -> list[str]:
    """Coerce an `endpoints.whitelist`/`blacklist` value to a list of pattern
    strings. Bare string → ONE pattern (not split); None → []; list →
    stringified pass-through."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


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
    """Parse `interfaces`. Each entry references a platform by config file
    (`config: platforms/<platform>/<interface>/config.yaml`, preferred) or by
    `platform` + `interface`. Optional `version` + `session` say where to start
    (base v0, or a version pinned from a session). Accepts legacy
    `starting_interfaces` / `starting_interface`."""
    raw = data.get("interfaces") or data.get("starting_interfaces")
    if raw is None and "starting_interface" in data:
        single = data["starting_interface"]
        raw = [single] if isinstance(single, dict) else []
    if not raw:
        return [InterfaceRef(platform="none", interface="none")]
    out: list[InterfaceRef] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(
                f"`interfaces` entries must be mappings with `config:` "
                f"(or `platform`+`interface`); got {entry!r}"
            )
        config_ref = entry.get("config")
        if config_ref:
            platform, interface = interfaces.platform_interface_from_config(config_ref)
        else:
            platform, interface = entry.get("platform", "none"), entry.get("interface", "none")
        version = entry.get("version")
        if version is not None:
            version = int(version)
        out.append(
            InterfaceRef(
                platform=platform, interface=interface, version=version,
                session=entry.get("session"), config=config_ref,
            )
        )
    return out


def load_config(path: Path) -> AutoresearchConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    goals = [_parse_goal(g) for g in (data.get("goals") or [])]
    endpoints = {
        "whitelist": _as_pattern_list((data.get("endpoints") or {}).get("whitelist")),
        "blacklist": _as_pattern_list((data.get("endpoints") or {}).get("blacklist")),
    }
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
    # tasks: ML task type → challenges. Canonical key `tasks`; legacy
    # `challenge_groups` and a flat `challenges:` list (wrapped as one "all"
    # task) accepted for back-compat.
    raw_tasks = data.get("tasks") or data.get("challenge_groups")
    tasks: dict[str, list[str]] = {}
    if raw_tasks:
        if not isinstance(raw_tasks, dict):
            raise ValueError(
                f"`tasks` must be a mapping of task_name → [challenges]; "
                f"got {type(raw_tasks).__name__}."
            )
        for name, items in raw_tasks.items():
            if not isinstance(items, list) or not items:
                raise ValueError(
                    f"tasks[{name!r}] must be a non-empty list of challenge ids."
                )
            tasks[str(name)] = [str(c) for c in items]
    elif data.get("challenges"):
        tasks = {"all": [str(c) for c in data["challenges"]]}

    # Flattened union of all task challenges (deduped, order-preserving).
    seen: set[str] = set()
    challenges = [c for items in tasks.values() for c in items if not (c in seen or seen.add(c))]

    return AutoresearchConfig(
        tasks=tasks,
        challenges=challenges,
        interfaces=_parse_interfaces(data),
        skills=data.get("skills", data.get("starting_skills", "none")),
        docs=data.get("docs", "none"),
        goals=goals,
        endpoints=endpoints,
        budget=Budget(
            # `iterations` = total versions incl. v0 (preferred). Legacy
            # `max_increments`/`max_cycles`/`max_runs` counted improvements over
            # v0, so iterations = that + 1. Default 11 (= old default 10).
            iterations=(
                int(bud["iterations"]) if "iterations" in bud
                else int(bud.get("max_increments", bud.get("max_cycles", bud.get("max_runs", 10)))) + 1
            ),
            max_cost_usd=float(bud.get("max_cost_usd", float("inf"))),
            # Compute-time cap. `max_seconds` preferred; `max_min` accepted as
            # legacy (× 60); UNLIMITED when neither is set (no silent cap).
            max_seconds=(
                float(bud["max_seconds"]) if "max_seconds" in bud
                else float(bud["max_min"]) * 60.0 if "max_min" in bud
                else float("inf")
            ),
        ),
        improve=improve,
        # `*_model` are research knobs in the config. Auth is a machine/setup
        # concern: defaults to BANTER_AUTH (what `make setup` chose) unless the
        # config overrides it. (`model`/`auth` accepted as legacy fallbacks.)
        engineer_model=data.get("engineer_model", data.get("model", "claude-sonnet-4-6")),
        engineer_auth=data.get(
            "engineer_auth", data.get("auth", os.environ.get("BANTER_AUTH", "api-key"))
        ),
        # Researcher defaults to Opus (planner); engineer defaults to Sonnet.
        researcher_model=data.get("researcher_model", "claude-opus-4-5"),
        researcher_auth=data.get(
            "researcher_auth",
            data.get("engineer_auth", data.get("auth", os.environ.get("BANTER_AUTH", "api-key"))),
        ),
        run=data.get("run"),
        prev_run=(int(data["prev_run"]) if data.get("prev_run") is not None else None),
        prev_version=(int(data["prev_version"]) if data.get("prev_version") is not None else None),
        experiment=(int(data["experiment"]) if data.get("experiment") is not None else None),
        research_question=(
            int(data["research_question"]) if data.get("research_question") is not None else None
        ),
        treatment=(int(data["treatment"]) if data.get("treatment") is not None else None),
        optimization_variable=data.get("optimization_variable"),
        language=data.get("language"),
        config_path=str(path),
    )


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------




def _scan_interfaces(testbed_root: Path) -> str:
    """List configured platforms (platforms/<project>/<interface>/config.yaml).
    Base IS version 0; higher versions are created session-locally."""
    idir = testbed_root / "platforms"
    if not idir.exists():
        return "  (none)"
    lines = []
    for project in sorted(p for p in idir.iterdir() if p.is_dir()):
        for interface in ("cli", "mcp", "sdk", "none"):
            cfg = project / interface / "config.yaml"
            if not cfg.is_file():
                continue
            manifest = yaml.safe_load(cfg.read_text()) or {}
            has_bin = " (binary)" if manifest.get("binary") else ""
            lines.append(f"  {project.name}/{interface}  base=v0{has_bin}")
    return "\n".join(lines) or "  (none)"


def _scan_skills(testbed_root: Path, project: str) -> str:
    """List a project's base skill bundles (platforms/<project>/skills/)."""
    sdir = testbed_root / "platforms" / project / "skills"
    lines = ["  none  (control — no skills injected into the engineer)"]
    if not sdir.exists():
        return "\n".join(lines)
    for bundle_dir in sorted(sdir.iterdir()):
        if not bundle_dir.is_dir() or bundle_dir.name.startswith("."):
            continue
        skill_names = sorted(
            d.name for d in bundle_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        )
        if skill_names:
            lines.append(f"  {bundle_dir.name}  skills={skill_names}  (base v0)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Researcher prompt
# ---------------------------------------------------------------------------


def _current_version(testbed_root: Path, iface: InterfaceRef) -> int:  # noqa: ARG001
    """Base version is always 0 — improved versions live run-local, not in the
    committed config. Args kept for future per-interface lookup."""
    return 0


def _fragment(testbed_root: Path, name: str, **kw: Any) -> str:
    """Load a prompt fragment from prompts/fragments/<name>, formatting if needed.

    All researcher prompt wording lives under prompts/; the code only selects
    and fills fragments, holding no prose.
    """
    p = testbed_root / "prompts" / "fragments" / name
    text = p.read_text().rstrip("\n") if p.exists() else ""
    return text.format(**kw) if kw else text


def build_researcher_prompt(
    config: AutoresearchConfig,
    testbed_root: Path,
    runs_root: Path,
    run_id: str,
    banter_bin: Path,
) -> str:
    # Bullets, NOT numbered — order isn't a priority ranking; goals are
    # optimized JOINTLY (see the prompt's joint-optimization note). A treatment
    # with NO declared goals is a CONTROL: the researcher freely chooses what to
    # optimize, so we swap in a free-choice directive and drop the joint-goals
    # note (which assumes a fixed goal set + the composite).
    if config.goals:
        goals_lines = "\n".join(
            f"  - {g.direction.upper()} `{g.metric}`" for g in config.goals
        )
        joint_goals_note = _fragment(testbed_root, "goals_joint.md")
    else:
        goals_lines = (
            "_This treatment declares NO fixed optimization target._ You MAY CHOOSE "
            "what to optimize — e.g. the interface's remote capability / endpoint "
            "coverage, competition `score`, or fewer engineer turns (`llm_calls`) / "
            "tokens. Pick objectives you can justify from the metric columns in "
            "`experiments.csv`, DECLARE them in your CHANGELOG / report, and pursue "
            "them consistently across versions (the `normalized_composite` tool needs "
            "a goal set, so reason over the raw metric columns instead)."
        )
        joint_goals_note = ""
    # Endpoint-coverage targets: tell the researcher WHAT `whitelist_hits`
    # measures (target REST endpoints) and that a miss is a capability the
    # interface must expose. Empty when no `endpoints:` configured.
    _wl = (config.endpoints or {}).get("whitelist") or []
    _bl = (config.endpoints or {}).get("blacklist") or []

    def _ep_short(p: str) -> str:
        return str(p).replace("/hopsworks-api/api/project/[0-9]+/", ".../")

    if _wl or _bl:
        _lines = "\n".join(f"  - `{_ep_short(p)}`" for p in _wl)
        endpoints_block = (
            f"\n## Endpoint coverage targets (goal `whitelist_hits`, max {len(_wl)})\n"
            "`whitelist_hits` counts how many of these target REST endpoints the engineer\n"
            "actually hit (distinct, >=1 call each). After EACH run, read\n"
            "`<run_dir>/endpoint_coverage.json` (`covered` / `missed`) to see which lifecycle\n"
            "steps it reached. A **missed** endpoint is a capability the interface does NOT\n"
            "yet expose — that's exactly what to add so the next engineer can reach it.\n\n"
            f"Targets ({len(_wl)}):\n{_lines}\n"
        )
        if _bl:
            endpoints_block += (
                "\nForbidden (`blacklist_hits`, minimize — must NOT be hit):\n"
                + "\n".join(f"  - `{_ep_short(p)}`" for p in _bl) + "\n"
            )
    else:
        endpoints_block = ""
    if config.tasks:
        challenges_lines = "\n".join(
            f"  [{group}]\n" + "\n".join(f"    - {c}" for c in items)
            for group, items in config.tasks.items()
        )
    else:
        challenges_lines = "\n".join(f"  - {c}" for c in config.challenges)
    avail_interfaces = _scan_interfaces(testbed_root)
    avail_skills = _scan_skills(testbed_root, config.interfaces[0].platform if config.interfaces else "none")
    n_challenges = len(config.challenges)
    n_interfaces = len(config.interfaces)
    runs_per_increment = n_challenges * n_interfaces

    # Per-interface current/next versions
    iface_versions = {
        str(iface): (_current_version(testbed_root, iface), _current_version(testbed_root, iface) + 1)
        for iface in config.interfaces
    }
    interfaces_table = "\n".join(
        f"  - {iface}  (current v{cur}, next would be v{nxt})"
        for iface, (cur, nxt) in iface_versions.items()
    )

    # Improvement scope + conditional prose (loaded from prompts/fragments/).
    can_interface = "interface" in config.improve
    can_skills = "skills" in config.improve
    if can_interface and can_skills:
        scope_desc = _fragment(testbed_root, "scope_both.md")
        scope_deny = ""
    elif can_interface:
        scope_desc = _fragment(testbed_root, "scope_interface.md")
        scope_deny = _fragment(testbed_root, "scope_deny_skills.md")
    else:
        # The from-scratch wording (when skills == none) is carried by
        # `skills_note` below, which fires in every skills+none case — no need
        # to repeat it here.
        scope_desc = _fragment(testbed_root, "scope_skills.md")
        scope_deny = _fragment(testbed_root, "scope_deny_interface.md")

    hierarchy_note = _fragment(
        testbed_root,
        "hierarchy_tasks.md" if config.tasks else "hierarchy_single.md",
    )
    skills_note = (
        _fragment(testbed_root, "skills_note.md")
        if (config.skills == "none" and "skills" in config.improve) else ""
    )
    cost_cap = (
        "unlimited" if config.budget.max_cost_usd == float("inf")
        else f"${config.budget.max_cost_usd:.2f} USD"
    )
    time_cap = (
        "unlimited" if config.budget.max_seconds == float("inf")
        else (
            f"{config.budget.max_seconds:.0f} s "
            f"(≈ {config.budget.max_seconds / 3600:.1f} h) of compute — the TOTAL "
            f"for this whole session, SHARED across every version and challenge "
            f"(not per-version, not per-challenge)"
        )
    )
    # COMPUTE-time budget is GRACEFUL: `banter budget-check` compares
    # (now - start - rate_limit_waits) against max_seconds. Rate-limit waits
    # (logged in the ledger by run_with_retry) are EXCLUDED, so only computation
    # counts. Researcher runs it before each new version, stopping at the version
    # boundary when exceeded (no hard subprocess kill).
    session_start_epoch = int(time.time())
    # `--max-seconds` takes a float; pass literal `inf` for unlimited (avoids
    # `int(inf)`, reads as always-CONTINUE).
    max_seconds_arg = (
        "inf" if config.budget.max_seconds == float("inf")
        else str(int(config.budget.max_seconds))
    )
    ledger_path = runs_root / ".rate_limit_wait_s"

    # Per-increment annotations live in `results.csv` columns (filled by
    # `banter annotate-increment`); CHANGELOG.md is the human-readable companion.
    changelog_path = runs_root / "CHANGELOG.md"
    first_iface = config.interfaces[0]
    _, next_iface_version = iface_versions[str(first_iface)]

    _ex_task = next(iter(config.tasks), "no_task") if config.tasks else "no_task"
    _ex_challenge = config.challenges[0] if config.challenges else "challenge-id"
    # Run-dir hierarchy: <run>/v<N>/<task>/<challenge>/. v0 = copy of the
    # committed original (or a prev session/increment when continuing);
    # v>0 = the researcher's edited copy of v<N-1>.
    run_dir_example = f"{runs_root}/v<N>/{_ex_task}/{_ex_challenge}/"

    # Eval command: each increment runs the engineer against its OWN interface
    # copy at <run>/v<N>/interface (set up via `banter prepare-version`, except
    # v0 auto-prepared at run start). --interface-dir points there; the runner
    # force-rebuilds and the engineer uses it.
    eval_block_lines = []
    task_items = list(config.tasks.items()) or [("no_task", config.challenges or ["<challenge>"])]
    # Records results straight into the global results/autoresearch/
    # experiments.csv via this flag (no per-run results.csv).
    exp_flag = f" \\\n  --experiment-config {config.config_path}" if config.config_path else ""
    for iface in config.interfaces:
        for task, task_challenges in task_items:
            for c in task_challenges:
                prev_flags = ""
                if config.prev_run is not None:
                    prev_flags += f" --prev-run {config.prev_run}"
                if config.prev_version is not None:
                    prev_flags += f" --prev-version v{config.prev_version}"
                eval_block_lines.append(
                    f"{banter_bin} run \\\n"
                    f"  --task {task} \\\n"
                    f"  --challenge {c} \\\n"
                    f"  --platform {iface.platform} --interface {iface.interface} \\\n"
                    f"  --skills {config.skills} \\\n"
                    f"  --docs {config.docs} \\\n"
                    f"  --interface-dir {runs_root}/v<N>/interface \\\n"
                    f"  --runs-root {runs_root}/v<N> \\\n"
                    f"  --run {run_id} --version v<N>{prev_flags}{exp_flag}"
                )
    eval_block = "\n".join(eval_block_lines)

    # v0 = baseline (always present — fresh or seeded from prev_run/
    # prev_version). v1..v<max_increments> are improvements ON TOP of v0.
    # max_increments=3 → v0..v3 (1 baseline + 3 improvements).
    start_version = 1                              # first improvement
    last_version_idx = config.budget.max_increments
    new_versions_count = config.budget.max_increments + 1  # incl. v0

    ctx = dict(
        run_id=run_id,
        testbed_root=testbed_root,
        runs_root=runs_root,
        changelog_path=changelog_path,
        hierarchy_note=hierarchy_note,
        goals_lines=goals_lines,
        joint_goals_note=joint_goals_note,
        endpoints_block=endpoints_block,
        # `max_versions` in the prompt = chain total. `start_version` /
        # `last_version` are the range to run THIS session.
        max_versions=config.budget.max_increments,
        start_version=start_version,
        last_version=last_version_idx,
        n_challenges=n_challenges,
        n_interfaces=n_interfaces,
        runs_per_version=runs_per_increment,
        total_runs=new_versions_count * runs_per_increment,
        cost_cap=cost_cap,
        time_cap=time_cap,
        session_start_epoch=session_start_epoch,
        max_seconds=max_seconds_arg,
        ledger_path=ledger_path,
        challenges_lines=challenges_lines,
        interfaces_table=interfaces_table,
        starting_skills=config.skills,
        skills_note=skills_note,
        docs_block=(
            f"\n## Reference docs (yours to study — the engineer never sees them)\n\n"
            f"A docs bundle (`{config.docs}`) is materialized at `{runs_root}/docs/` "
            f"FOR YOU. Study them to learn the platform's core concepts and how its "
            f"capabilities map to your goals (and to the coverage targets, if any). "
            f"They are static — not edited across versions, not measured. **The "
            f"engineer does NOT receive these docs**: its only guide is the interface "
            f"itself, so fold what you learn here into the interface's own "
            f"self-description (names, `--help`, docstrings, errors).\n"
            if config.docs != "none"
            else "\n## Reference docs\n\nNo docs bundle configured for this "
                 "session (`docs: none`). Work from the interface source "
                 "alone.\n"
        ),
        scope_desc=scope_desc,
        scope_deny=scope_deny,
        first_iface_name=first_iface.platform,
        first_iface_mode=first_iface.interface,
        next_iface_version=next_iface_version,
        run_dir_example=run_dir_example,
        banter_bin=banter_bin,
        experiment_config=config.config_path or "",
        eval_block=eval_block,
        avail_interfaces=avail_interfaces,
        avail_skills=avail_skills,
    )
    template = (testbed_root / "prompts" / "researcher.md").read_text()
    return template.format(**ctx)


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------


def _display_event(event: dict[str, Any], log_path: Path | None = None) -> None:
    """Print a human-readable line for a researcher stream-json event, mirroring
    to `log_path` (the session's researcher.log) when given.

    Assistant lines share the engineer's formatter (`streaming.assistant_lines`);
    `result` events keep the researcher's richer finish summary.
    """
    etype = event.get("type")
    if etype == "assistant":
        for line in streaming.assistant_lines(event, "researcher"):
            streaming.emit(line, log_path)
    elif etype == "user":
        for line in streaming.tool_result_lines(event, "researcher"):
            streaming.emit(line, log_path)
    elif etype == "result":
        cost = event.get("total_cost_usd", 0)
        turns = event.get("num_turns", "?")
        stop = event.get("stop_reason") or event.get("subtype", "?")
        streaming.emit(
            f"\n[banter] researcher finished: {turns} turns, "
            f"${cost:.4f} researcher cost, stop={stop}",
            log_path,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _version_root_for(testbed_root: Path, session: str | None) -> Path | None:
    """Session dir holding pinned interface versions, or None for the base."""
    if not session:
        return None
    return testbed_root / "results" / "autoresearch" / session


def run_autoresearch(
    config: AutoresearchConfig,
    testbed_root: Path,
    runs_root: Path,
    config_name: str | None = None,
    assume_yes: bool = False,
) -> None:
    if shutil.which("claude") is None:
        raise RuntimeError("`claude` CLI not found on PATH. Install Claude Code first.")

    # `banter` runs outside the venv — prefer PATH (host install), fall back to
    # a venv copy if present.
    found = shutil.which("banter")
    venv_banter = testbed_root / ".venv" / "bin" / "banter"
    banter_bin = Path(found) if found else venv_banter
    if not banter_bin.exists():
        raise RuntimeError(
            "`banter` not found on PATH or in .venv. Install it (e.g. `make install`)."
        )

    # The researcher only starts once every required platform is built, set up,
    # authenticated, and tested (deterministically, no AI) and any skill is
    # accessible. Fail fast before spawning it otherwise.
    from banter import preflight as preflight_mod
    reqs = [
        preflight_mod.Requirement(
            platform=i.platform,
            interface=i.interface,
            interface_version=i.version,
            version_root=_version_root_for(testbed_root, i.session),
            skills=config.skills,
        )
        for i in config.interfaces
    ]
    try:
        # Build + test the platforms once at session start (login checked per
        # challenge by the runner, in each run's own venv).
        # cleanup_build: this upfront check builds each interface only to verify
        # it, deleting the artifacts after so the committed source folder stays
        # source-only. Real build/install happens per version (v0/vN).
        preflight_mod.preflight(
            reqs, auth=config.engineer_auth, model=config.engineer_model, check_login=False,
            cleanup_build=True,
        )
    except preflight_mod.PreflightError as e:
        raise RuntimeError(
            f"[banter] preflight failed — fix before the researcher can start:\n{e}"
        )

    # Pre-download every (task, challenge) dataset NOW, before the researcher
    # spawns: its `deny` patterns block writes outside <run>/, so it can't
    # populate <testbed>/cache itself. Idempotent: skips already-prepared comps.
    from banter import mlebench_wrapper, runner as runner_mod
    all_challenges: list[str] = []
    if config.tasks:
        for ch_list in config.tasks.values():
            all_challenges.extend(ch_list)
    else:
        all_challenges.extend(config.challenges)
    for n, comp in enumerate(sorted(set(all_challenges)), 1):
        print(f"[banter] preparing data {n}/{len(set(all_challenges))}: {comp}", flush=True)
        mlebench_wrapper.download_competition(comp, runner_mod.DEFAULT_DATA_ROOT)

    from banter import results as results_mod
    parent = runs_root / "autoresearch"
    parent.mkdir(parents=True, exist_ok=True)

    # The config FILENAME stem names the results folder (e.g. `rq1`,
    # `test-skills`). Explicit `run:` overrides it; else fall back to the legacy
    # auto-increment id when no name was passed in.
    if config_name:
        run_id = config_name
    elif config.run is not None:
        run_id = str(config.run)
    else:
        run_id = results_mod.next_session_id(parent)

    # One folder per config: results/autoresearch/<run_id>/. Inside, one sub-tree
    # per platform (<platform>/<interface>/<skills>/); EACH leaf is its own
    # researcher (run sequentially) with v<N>/<task>/<challenge>/ underneath.
    # Overwrite a pre-existing config folder, but confirm first.
    config_root = parent / run_id
    if not results_mod.confirm_overwrite(config_root, assume_yes):
        print(f"[banter] {config_root} exists — overwrite declined. Aborting.",
              flush=True)
        return
    config_root.mkdir(parents=True, exist_ok=True)

    # Clear this config's prior rows from the global experiments table so a
    # re-run replaces (not duplicates) them; the leaves re-append live.
    try:
        from banter import experiments as experiments_mod
        experiments_mod.clear_treatment(parent.parent, config)
    except Exception as e:
        print(f"[banter] experiments clear skipped: {e}", flush=True)

    iface_list = config.interfaces or [InterfaceRef(platform="none", interface="none")]
    skills_seg = config.skills or "none"
    print(f"[banter] config={run_id}  "
          f"interfaces={[str(i) for i in iface_list]}\n  dir={config_root}", flush=True)

    # Each platform is an independent experiment with its own researcher,
    # results.csv, and report under its leaf — no combined root rollup.
    from banter import experiments as _exp
    for n, iface in enumerate(iface_list, 1):
        iface_root = config_root / iface.platform / iface.interface / skills_seg
        print(f"\n[banter] === platform {n}/{len(iface_list)}: {iface} "
              f"→ {iface_root.relative_to(config_root)} ===", flush=True)
        # Per-leaf goals: keep only THIS interface's delegation (`*_calls`)
        # goal — can't maximize cli/mcp/sdk calls simultaneously on one interface.
        leaf_goal_pairs = _exp.goals_for_interface(
            [(g.metric, g.direction) for g in config.goals], iface.interface)
        leaf_goals = [Goal(metric=m, direction=d) for m, d in leaf_goal_pairs]
        _run_one_interface(
            replace(config, interfaces=[iface], goals=leaf_goals),
            testbed_root, parent, iface_root, run_id, banter_bin,
        )

    print(f"\n[banter] done. config dir: {config_root}", flush=True)


def _run_one_interface(
    config: AutoresearchConfig,
    testbed_root: Path,
    parent: Path,
    run_path: Path,
    run_id: str,
    banter_bin: Path,
) -> None:
    """Run ONE platform's researcher, rooted at its leaf dir `run_path`
    (`<config>/<platform>/<interface>/<skills>/`). `config` is narrowed to that
    platform; `parent` is `results/autoresearch/` (for prev-run lookup).
    """
    i0 = config.interfaces[0] if config.interfaces else None
    run_path.mkdir(parents=True, exist_ok=True)

    # Clone the docs bundle (git URL in `config.docs`) ONCE at the run root.
    # Researcher browses these at `<run>/docs/`; each engineer gets its own
    # APFS-cloned copy at `<challenge>/docs/` (cheap COW, no re-fetch). Docs are
    # static across versions.
    if config.docs and config.docs != "none":
        from banter import docs as docs_mod
        docs_platform = config.interfaces[0].platform if config.interfaces else None
        try:
            docs_url = docs_mod.resolve(config.docs, docs_platform)
            docs_ref = docs_mod.resolve_ref(config.docs, docs_platform)
        except ValueError as e:
            raise RuntimeError(f"[banter] docs {config.docs!r}: {e}")
        docs_setup = docs_mod.apply(docs_url, run_path, ref=docs_ref)
        print(f"[banter] docs {config.docs!r}: {len(docs_setup.files)} "
              f"files at {run_path / 'docs'}", flush=True)

    # v0 = this session's original interface, auto-created here (no AI), copied
    # from EITHER a previous session/increment (when `prev_*` is set) OR the
    # committed base, then built in place so it's immediately runnable. The
    # researcher prepares v1+ via `banter prepare-version`; between runs it only
    # edits source — the system handles copy / build / install / uninstall.
    if i0 is not None and i0.platform != "none":
        incr0 = run_path / "v0" / "interface"
        src: Path | None = None
        if config.prev_run is not None and config.prev_version is not None:
            # Resolve the prev config's matching leaf in the nested tree:
            #   <parent>/<prev_run>/<platform>/<interface>/<skills>/v<prev_version>/interface
            prev_config = parent / str(config.prev_run)
            if not prev_config.is_dir():
                # Back-compat: match an old flat dir by leading id segment.
                flat = [
                    d for d in parent.iterdir()
                    if d.is_dir() and d.name.split("__", 1)[0] == str(config.prev_run)
                ]
                prev_config = flat[0] if flat else None
            if prev_config is None:
                raise RuntimeError(
                    f"[banter] prev_run={config.prev_run} not found under {parent}"
                )
            skills_seg = config.skills or "none"
            prev_iface = (
                prev_config / i0.platform / i0.interface / skills_seg
                / f"v{config.prev_version}" / "interface"
            )
            if not prev_iface.is_dir():
                raise RuntimeError(
                    f"[banter] prev_version dir not found: {prev_iface}"
                )
            src = prev_iface
        else:
            base = testbed_root / "platforms" / i0.platform / i0.interface
            if base.is_dir():
                src = base
        if src is not None and not incr0.exists():
            incr0.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, incr0)
            for w in incr0.glob("*.whl"):
                w.unlink()
            interfaces.set_interface_home(i0.platform, i0.interface, incr0)
            try:
                interfaces.build(i0.platform, i0.interface)
            except Exception as e:
                raise RuntimeError(f"[banter] v0 build failed in {incr0}: {e}")
            print(
                f"[banter] v0 ready + built (← {src}): {incr0}",
                flush=True,
            )

    prompt = build_researcher_prompt(config, testbed_root, run_path, run_id, banter_bin)
    (run_path / "prompt.txt").write_text(prompt)

    # Results go straight to the GLOBAL results/autoresearch/experiments.csv
    # (one row per version/task/challenge, appended live by each `banter run`).
    # No per-run results.csv — the single global analysis.ipynb does the math.
    # Stale rows for this leaf are cleared at session start in `run_autoresearch`.

    # Pre-create CHANGELOG.md so it sits next to report.md from the start. The
    # researcher appends one section per increment (what changed + what
    # happened); it's also long-term memory read BEFORE planning the next
    # increment, so ideas aren't repeated even after context compaction.
    changelog_md = run_path / "CHANGELOG.md"
    if not changelog_md.exists():
        goals_md_lines = "\n".join(
            f"- **{g.direction}**(`{g.metric}`)" for g in config.goals
        ) or "_(none declared)_"
        changelog_md.write_text(
            f"# Autoresearch run `{run_id}` — changelog\n\n"
            "Per-increment record of changes the researcher made and what they "
            "produced. The researcher reads this before each new increment to "
            "remember what was already tried.\n\n"
            f"## Goals\n{goals_md_lines}\n\n"
            "---\n"
        )

    # Source-study notes live IN project memory (CLAUDE.md below), not a separate
    # STUDY.md — CLAUDE.md is auto-loaded as system context every turn, so source
    # knowledge survives context compaction without a manual re-read.

    # (Per-run analysis.ipynb retired — the single global notebook does the math.)

    # The raw stream-json transcript isn't kept in the run dir (nothing reads it,
    # and we don't want the researcher loading it as context). Streamed to a
    # throwaway temp file (human-readable view stays in researcher.log) and
    # removed at session end. The researcher's memory is CHANGELOG.md.
    import tempfile
    _tf = tempfile.NamedTemporaryFile(prefix="banter-researcher-", suffix=".jsonl", delete=False)
    _tf.close()
    transcript_path = Path(_tf.name)
    stderr_path = run_path / "researcher.stderr.log"

    # Project-scoped `.claude/` INSIDE the run dir so the researcher's settings,
    # hooks, and memory live alongside its results (same place the engineer's
    # per-challenge `.claude/` lives). With `cwd=run_path` below, claude-code
    # discovers `<run>/.claude/` as the project root: anything Claude writes
    # (settings, project memory) lands here, not in `<testbed>/.claude/`.
    claude_dir = run_path / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    # No researcher command log (commands.jsonl): the researcher's own tool calls
    # aren't measured (only the engineer's are, via its per-challenge transcript),
    # and we keep the run dir free of context it might load. The PreToolUse hook
    # still enforces confinement; it just skips logging when TESTBED_COMMAND_LOG
    # is unset.
    # Confinement is tool-layer only (permissions.deny). No kernel sandbox —
    # Claude Code's `allowWrite` has a depth cutoff that breaks the deep
    # mkdir/file ops the spawned `banter run` subprocesses need inside
    # `<run>/v<N>/<task>/<challenge>/`. Deny patterns catch the common escape
    # vectors (`../`, `$HOME/.*` dotfiles, `cd ..` in bash).
    (claude_dir / "settings.json").write_text(json.dumps({
        "hooks": {
            "PreToolUse": [{
                "matcher": ".*",
                "hooks": [{
                    "type": "command",
                    "command": f"python3 {claude_runner.HOOK_SCRIPT}",
                }],
            }],
        },
        "permissions": {
            "deny": list(claude_runner.COMMON_DENY)
                    + claude_runner.deny_patterns_for(run_path.resolve()),
        },
    }, indent=2))
    # Project-scoped memory: loaded as system context by claude-code on every
    # tool invocation. Persistent notes that survive context compaction and stay
    # visible if you re-open the run dir later.
    goals_md = "\n".join(f"- **{g.direction}**(`{g.metric}`)" for g in config.goals)
    iface_md = ", ".join(str(i) for i in config.interfaces)
    (claude_dir / "CLAUDE.md").write_text(
        f"# Autoresearch run `{run_id}` — project memory\n\n"
        f"This dir is the working tree for one autoresearch run. The "
        f"researcher (Claude Opus) iterates the interface across increments "
        f"to improve the goal metrics; the engineer (Claude Sonnet) is "
        f"invoked per `(increment, task, challenge)` via `banter run`.\n\n"
        f"## Interfaces\n{iface_md}\n\n"
        f"## Optimization goals\n{goals_md}\n\n"
        "## Structure\n"
        "- `v<N>/interface/` — the interface for increment N; the REAL source is\n"
        "  under `v<N>/interface/src/` — study + edit THAT to improve the interface.\n"
        "- `v<N>/<task>/<challenge>/` — per-challenge engineer artifacts.\n"
        "- `CHANGELOG.md` — your long-term memory; auto-written per version by\n"
        "  `banter annotate-version` (read it; don't hand-edit it).\n"
        "- `researcher.log` — human-readable view of your own run.\n"
        "\n## The prompt is FROZEN\n"
        "The engineer's task prompt AND the interface `prompt:` field are fixed.\n"
        "Editing `prompt:` in a version copy has no effect (it's read from the\n"
        "committed base), and a version whose only change is the prompt/config is\n"
        "refused at run time. Improvements MUST be source changes.\n"
        "- Results go to the GLOBAL `results/autoresearch/experiments.csv` (one row per\n"
        "  version/task/challenge); use the `normalized_composite` MCP tool to score versions.\n"
        "\n## Source study (your memory — keep this section current)\n"
        "Maintain your understanding of the INTERFACE SOURCE under `v<N>/interface/src/`\n"
        "RIGHT HERE in this file (it's auto-loaded every turn, so it survives context\n"
        "compaction — no separate STUDY.md). Append as you read: file path → what it\n"
        "does → weaknesses spotted / ideas. Update it before hypothesizing each change.\n"
        "\n### Source map\n"
        "_(file path → what it does → weaknesses spotted / ideas)_\n"
    )

    env = os.environ.copy()
    env.pop("ANTHROPIC_BASE_URL", None)
    # HOME → run dir so `.claude/` state lands inside (not in user's real ~).
    env["HOME"] = str(run_path.resolve())
    # OAuth token side-channel. Claude Code's Bash tool strips
    # CLAUDE_CODE_OAUTH_TOKEN before spawning children, so the engineer's
    # `banter run` can't inherit it. Write the JWT to `<run>/.claude-oauth`
    # (mode 0600) and point BANTER_TOKEN_CACHE at it — BANTER_* vars DO survive
    # the Bash hop. Cache lives in the run dir, so it vanishes when the run is removed.
    token = claude_runner.oauth_token_from_keychain()
    if token:
        cache_path = claude_runner.write_token_cache(token, run_path.resolve())
        env[claude_runner.TOKEN_CACHE_ENV] = str(cache_path)
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    if config.researcher_auth == "login":
        env.pop("ANTHROPIC_API_KEY", None)
    venv_bin = str(testbed_root / ".venv" / "bin")
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env.setdefault("KAGGLE_CONFIG_DIR", str(testbed_root / ".kaggle"))
    # (No TESTBED_COMMAND_LOG for the researcher — see above.)
    # Rate-limit-wait ledger for the compute-time budget. `run_with_retry` (this
    # researcher AND every engineer subprocess it spawns — BANTER_* vars survive
    # the Bash hop) appends each rate-limit sleep here; `banter budget-check`
    # subtracts the total from wall clock so only compute counts.
    env[claude_runner.RATE_LIMIT_LEDGER_ENV] = str(run_path / ".rate_limit_wait_s")
    # Mark engineer subprocesses as nested: they write engineer.log but don't
    # print to stdout (researcher captures it). The FileTailer below surfaces
    # those logs live in this terminal instead.
    env["BANTER_NESTED"] = "1"
    # Keep each `banter run` the researcher launches SYNCHRONOUS. A single
    # `banter run` (preflight + engineer + grade, engineer may wait on a
    # multi-minute remote job) runs far past the Bash tool's default timeout.
    # Two SEPARATE Claude Code mechanisms to handle:
    #   1) Auto-backgrounding — Claude Code moves the long command to a
    #      background task. In `-p` mode the completion notification NEVER
    #      re-invokes the model, so the researcher falls into a blind `wc -l`
    #      poll loop on a task-output file that froze when the engineer phase
    #      redirected output into engineer.log. `BASH_MAX_TIMEOUT_MS` does NOT
    #      prevent this — only CLAUDE_CODE_DISABLE_BACKGROUND_TASKS does.
    #   2) Kill timeout — size both Bash timeouts so a long `banter run`
    #      completes instead of being killed (see rate-limit note below).
    env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] = "1"
    _bash_cap_s = (
        config.budget.max_seconds
        if config.budget.max_seconds != float("inf")
        else 8 * 3600
    )
    # Rate-limit waiting must NOT count against this hard kill timeout. Each
    # `banter run` spawns an engineer whose `run_with_retry` SLEEPS on rate-limit
    # backoff INSIDE that Bash command, so those sleeps accrue to its wall clock.
    # Capping the Bash timeout at exactly the compute budget would let sustained
    # rate-limiting kill a healthy run. Add the rate-limit retry window so
    # waiting never trips the kill — the REAL compute budget is enforced
    # gracefully by `banter budget-check`, which already subtracts those waits.
    _bash_cap_s += claude_runner.RATE_LIMIT_RETRY_WINDOW_S
    env["BASH_DEFAULT_TIMEOUT_MS"] = str(int(_bash_cap_s) * 1000)
    env["BASH_MAX_TIMEOUT_MS"] = str(int(_bash_cap_s) * 1000)

    # Give the researcher a `normalized_composite` MCP tool over the global
    # experiments table, scoped to THIS leaf (config + platform/interface/skills)
    # and its goals. Dependency-free stdio server (`banter.research_mcp`).
    from banter import experiments as experiments_mod
    results_root = parent.parent
    leaf_iface = config.interfaces[0] if config.interfaces else None
    cfg_rel = experiments_mod._config_rel(config, results_root)
    goals_str = experiments_mod._goals_str([(g.metric, g.direction) for g in config.goals])
    venv_py = str(testbed_root / ".venv" / "bin" / "python")
    mcp_config_path = run_path / ".mcp.json"
    mcp_config_path.write_text(json.dumps({
        "mcpServers": {
            "banter-research": {
                "command": venv_py,
                "args": ["-m", "banter.research_mcp"],
                "env": {
                    "BANTER_EXPERIMENTS_CSV": str(experiments_mod.table_path(results_root)),
                    "BANTER_CONFIG": cfg_rel,
                    "BANTER_PLATFORM": leaf_iface.platform if leaf_iface else "",
                    "BANTER_INTERFACE": leaf_iface.interface if leaf_iface else "",
                    "BANTER_GOALS": goals_str,
                },
            }
        }
    }, indent=2))

    # `--settings` + `--setting-sources` mirror the engineer's setup
    # (claude_runner.run): `-p` mode silently skips project settings unless asked
    # explicitly, so the hook config in <run>/.claude/settings.json would
    # otherwise be ignored.
    settings_file = (claude_dir / "settings.json").resolve()
    cmd = [
        "claude",
        "-p", prompt,
        "--permission-mode", "bypassPermissions",
        "--model", config.researcher_model,
        "--max-turns", "500",
        "--output-format", "stream-json",
        "--verbose",
        "--settings", str(settings_file),
        "--setting-sources", "project,local,user",
    ]
    if mcp_config_path is not None:
        cmd.extend(["--mcp-config", str(mcp_config_path)])
    # Researcher's deny patterns + HOME redirect (settings written above +
    # env set above) keep it inside <run>/ at the tool layer.

    n_ifaces = len(config.interfaces)
    n_chals = len(config.challenges)
    iface_list = ", ".join(str(i) for i in config.interfaces)
    print(
        f"[banter] run={run_id}\n"
        f"  interfaces={iface_list}\n"
        f"  goals={[f'{g.direction}({g.metric})' for g in config.goals]}\n"
        f"  budget={config.budget.max_increments} increments × {n_ifaces} interfaces × {n_chals} challenges "
        f"= up to {config.budget.max_increments * n_ifaces * n_chals} runs\n"
        f"  cost_cap={'∞' if config.budget.max_cost_usd == float('inf') else f'${config.budget.max_cost_usd:.2f}'}\n"
        f"  compute_cap={'∞' if config.budget.max_seconds == float('inf') else f'{config.budget.max_seconds:.0f}s (rate-limit waits excluded)'}",
        flush=True,
    )
    print(f"[banter] run dir: {run_path}", flush=True)
    print()

    # The researcher's own stream view is saved to researcher.log and printed
    # live (emit() honors BANTER_QUIET for stdout, always writes the file).
    # (Engineer per-challenge logs are `engineer.log`; this one is distinct.)
    researcher_log = run_path / "researcher.log"

    def _on_line(raw_line: str) -> None:
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            return
        _display_event(event, researcher_log)

    # Surface nested engineer runs live: tail every engineer.log that
    # appears under the session dir (excluding the researcher's own).
    tailer = None
    if not streaming.quiet():
        tailer = streaming.FileTailer(
            run_path, "**/engineer.log", exclude=(researcher_log,)
        )
        tailer.start()

    # Shares the engineer's rate-limit retry helper: 6h budget, exponential
    # back-off on 429/529/5xx/overloaded.
    researcher_wall_s = 0.0
    researcher_usage: dict[str, float] = {}
    try:
        _, researcher_wall_s = claude_runner.run_with_retry(
            cmd=cmd,
            # cwd = the run dir so claude-code project-scopes to <run>/.claude/.
            # The prompt already uses absolute paths everywhere ({runs_root},
            # {testbed_root}/.venv/bin/banter, …) so nothing breaks from the
            # shift away from testbed_root.
            cwd=run_path,
            env=env,
            transcript_path=transcript_path,
            stderr_path=stderr_path,
            on_line=_on_line,
            log_prefix="banter",
        )
    except KeyboardInterrupt:
        print("\n[banter] interrupted by user.", flush=True)
    finally:
        if tailer is not None:
            tailer.stop()
            tailer.join(timeout=2)
        # Read token/cost usage from the transcript BEFORE discarding it, then
        # remove the throwaway raw transcript (keep researcher.log).
        try:
            from banter import results as results_mod
            u = results_mod.parse_transcript_usage(transcript_path)
            researcher_usage = {
                "input_tokens": float(u.get("input_tokens", 0) or 0),
                "output_tokens": float(u.get("output_tokens", 0) or 0),
                "total_tokens": float(u.get("total_tokens", 0) or 0),
                "cost_usd": float(u.get("cost_usd", 0.0) or 0.0),
                "wall_s": float(researcher_wall_s),
            }
        except Exception:
            researcher_usage = {"wall_s": float(researcher_wall_s)}
        try:
            transcript_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Distribute the researcher's tokens + wall time + cost across this leaf's
    # rows in the global table (shared overhead → total_* = eng_* + res_*).
    try:
        from banter import experiments as experiments_mod
        i0 = config.interfaces[0] if config.interfaces else None
        experiments_mod.attribute_researcher(
            parent.parent, config,
            i0.platform if i0 else "", i0.interface if i0 else "",
            researcher_usage,
        )
    except Exception as e:
        print(f"[banter] researcher attribution skipped: {e}", flush=True)

    # Rows were appended live to experiments.csv by the runner (engineer-side
    # metrics); `attribute_researcher` filled the researcher's share. (Re)generate
    # AND EXECUTE the global analysis notebook so its diagrams reflect the final,
    # attributed numbers.
    try:
        from banter import experiments
        nb = experiments.build_global_notebook(testbed_root, execute=True)
        print(f"[banter] refreshed global analysis notebook: {nb}", flush=True)
    except Exception as e:
        print(f"[banter] global notebook refresh skipped: {e}", flush=True)

    print(f"\n[banter] done. run dir: {run_path}", flush=True)
