"""Manager Claude that iteratively optimizes interface and skill configurations.

Hierarchy: a SESSION contains multiple INCREMENTS; each increment spans multiple
TASKS (ML task types); each task spans its CHALLENGES. Every RQ config uses the
same `tasks:` mapping (RQ1 = one challenge per task; RQ2 = several per task; RQ3
= same shape with skills layered on frozen interfaces).

Spawns a `claude -p` researcher session. The researcher:
  - Runs every (interface × task × challenge) with the current config
  - Reads all run dirs and results.csv to understand engineer behaviour
  - Creates session-local interface versions
    (results/autoresearch/<id>/interfaces/<name>/<mode>/v<n>/version.yaml) or
    skill bundle versions (skills/<bundle>/<v+1>/<skill>/SKILL.md)
  - Evaluates the change by re-running, then decides conclusively
  - Ends each increment with a SUMMARY OF OBSERVATIONS and a clear STATEMENT OF
    PROPOSED CHANGES; ends the session with a FINAL REPORT
  - Iterates until budget (max_increments) is exhausted

Session layout:
    results/autoresearch/<session_id>/
        results.csv        # one row per individual challenge run
        increments.jsonl   # one entry per improvement increment
        report.md          # final session report (written at the end)
        prompt.txt         # researcher prompt (for debugging)
        transcript.jsonl   # raw researcher stream-json
        transcript.log     # human-readable researcher output
        interfaces/<name>/<mode>/v<n>/version.yaml   # session-local versions
        <interface>/<mode>/<skills>/v<version>/<challenge>/
            prompt.txt     # engineer prompt
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from banter import interfaces


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Goal:
    metric: str       # column name in results.csv, e.g. 'score', 'total_tokens'
    direction: str    # 'maximize' or 'minimize'


@dataclass
class Budget:
    # One increment = one full round across all (interface × task × challenge).
    max_increments: int = 10
    max_cost_usd: float = float("inf")         # no cap by default

    # Back-compat: older configs/code referenced `max_cycles`.
    @property
    def max_cycles(self) -> int:
        return self.max_increments


@dataclass
class InterfaceRef:
    name: str
    mode: str
    version: int | None = None      # None/0 → base manifest; >0 → a session version
    # Autoresearch session that produced a pinned `version` > 0 (e.g. RQ3 pinning
    # an RQ1 winner). None means "this session" / base.
    session: str | None = None
    # The interface config path this ref came from (when referenced by `config:`).
    config: str | None = None

    def __str__(self) -> str:
        v = f" v{self.version}" if self.version is not None else ""
        s = f" @{self.session}" if self.session else ""
        return f"{self.name}/{self.mode}{v}{s}"


_VALID_IMPROVE_SCOPES = {"interface", "skills"}


@dataclass
class AutoresearchConfig:
    # tasks: ML task type → its challenges. The unifying structure across all
    # RQ configs (RQ1 = one challenge per task; RQ2 = several; RQ3 = same as
    # RQ1 with skills). The researcher runs `max_increments` increments PER TASK.
    tasks: dict[str, list[str]]
    challenges: list[str]                # flattened union of all tasks (derived)
    interfaces: list[InterfaceRef]       # interfaces run side-by-side per increment
    skills: str                          # starting skill bundle name, or "none"
    goals: list[Goal]
    budget: Budget
    improve: list[str]              # subset of {"interface", "skills"}
    # The engineer is the controlled instance; the researcher controls it.
    engineer_model: str = "claude-sonnet-4-6"
    engineer_auth: str = "api-key"
    researcher_model: str = "claude-sonnet-4-6"
    researcher_auth: str = "api-key"


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
    """Parse `interfaces`. Each entry references an interface either by its
    config file (`config: configs/interfaces/<name>/<type>.yaml`) — the
    preferred, explicit form — or by `name` + `mode`. Optional `version` +
    `session` say where to start (base v0, or a version pinned from a session).
    Accepts legacy `starting_interfaces` / `starting_interface`."""
    raw = data.get("interfaces") or data.get("starting_interfaces")
    if raw is None and "starting_interface" in data:
        single = data["starting_interface"]
        raw = [single] if isinstance(single, dict) else []
    if not raw:
        return [InterfaceRef(name="none", mode="none")]
    out: list[InterfaceRef] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(
                f"`interfaces` entries must be mappings with `config:` "
                f"(or `name`+`mode`); got {entry!r}"
            )
        config_ref = entry.get("config")
        if config_ref:
            name, mode = interfaces.name_type_from_config(config_ref)
        else:
            name, mode = entry.get("name", "none"), entry.get("mode", "none")
        version = entry.get("version")
        if version is not None:
            version = int(version)
        out.append(
            InterfaceRef(
                name=name, mode=mode, version=version,
                session=entry.get("session"), config=config_ref,
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
    # tasks: ML task type → its challenges. Canonical key is `tasks`; we accept
    # the legacy `challenge_groups`, and a flat `challenges:` list (wrapped as a
    # single "all" task) for back-compat.
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
        goals=goals,
        budget=Budget(
            max_increments=int(
                bud.get("max_increments", bud.get("max_cycles", bud.get("max_runs", 10)))
            ),
            max_cost_usd=float(bud.get("max_cost_usd", float("inf"))),
        ),
        improve=improve,
        # `engineer_*` are canonical; `model`/`auth` are accepted as legacy
        # fallbacks. The researcher defaults to the engineer's auth/model when
        # not given its own.
        engineer_model=data.get("engineer_model", data.get("model", "claude-sonnet-4-6")),
        engineer_auth=data.get("engineer_auth", data.get("auth", "api-key")),
        researcher_model=data.get(
            "researcher_model", data.get("engineer_model", data.get("model", "claude-sonnet-4-6"))
        ),
        researcher_auth=data.get(
            "researcher_auth", data.get("engineer_auth", data.get("auth", "api-key"))
        ),
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
    """List configured interfaces. Version 0 is the base config; higher versions
    are created session-locally during the run (not listed here)."""
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
            if type_ not in ("cli", "mcp", "sdk", "none"):
                continue
            manifest = yaml.safe_load(yaml_file.read_text()) or {}
            if isinstance(manifest, dict) and manifest:
                has_bin = " (binary)" if manifest.get("binary") else ""
                lines.append(f"  {name_dir.name}/{type_}  base=v0{has_bin}")
    return "\n".join(lines) or "  (none)"


def _scan_skills(testbed_root: Path) -> str:
    sdir = testbed_root / "skills"
    lines = ["  none  (control — no skills injected into the engineer)"]
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


def _fragment(testbed_root: Path, name: str, **kw: Any) -> str:
    """Load a prompt fragment from prompts/fragments/<name>, formatting if needed.

    All researcher prompt wording lives under prompts/ — the code only selects
    and fills fragments, it does not hold prose.
    """
    p = testbed_root / "prompts" / "fragments" / name
    text = p.read_text().rstrip("\n") if p.exists() else ""
    return text.format(**kw) if kw else text


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
    if config.tasks:
        challenges_lines = "\n".join(
            f"  [{group}]\n" + "\n".join(f"    - {c}" for c in items)
            for group, items in config.tasks.items()
        )
    else:
        challenges_lines = "\n".join(f"  - {c}" for c in config.challenges)
    avail_interfaces = _scan_interfaces(testbed_root)
    avail_skills = _scan_skills(testbed_root)
    recent_results = _recent_results_table(runs_root)
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
        from_scratch = (
            _fragment(testbed_root, "scope_skills_from_scratch.md")
            if config.skills == "none" else ""
        )
        scope_desc = _fragment(testbed_root, "scope_skills.md", from_scratch=from_scratch)
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

    increments_path = runs_root / "increments.jsonl"
    first_iface = config.interfaces[0]
    cur_iface_version, next_iface_version = iface_versions[str(first_iface)]

    # Tasks section (RQ2-style per-task specialisation) — loaded from prompts/.
    if config.tasks:
        task_lines = []
        for gname, gitems in config.tasks.items():
            task_lines.append(f"  [{gname}]")
            for c in gitems:
                task_lines.append(f"    - {c}")
        groups_block = _fragment(
            testbed_root, "tasks_section.md",
            n_tasks=len(config.tasks),
            max_increments=config.budget.max_increments,
            tasks_list="\n".join(task_lines),
        ) + "\n"
    else:
        groups_block = ""
    skills_label = config.skills if config.skills != "none" else "no_skills"
    run_dir_example = (
        f"{runs_root}/inc<N>/{first_iface.name}/{first_iface.mode}/"
        f"{skills_label}/v{cur_iface_version}/"
        f"{config.challenges[0] if config.challenges else 'challenge-id'}/"
    )

    # Pre-render the "run every (interface, challenge) pair" block once (used for
    # baseline + per-cycle evaluation snippets below). A pinned starting version
    # (e.g. RQ3 on top of an RQ1 winner) passes --interface-version AND the
    # --version-root of the session that produced it. New versions you create
    # this session live under THIS session's --version-root ({runs_root}).
    eval_block_lines = []
    for iface in config.interfaces:
        flags = ""
        if iface.version:
            flags += f" --interface-version {iface.version}"
            vr = _version_root_for(testbed_root, iface.session)
            if vr is not None:
                flags += f" --version-root {vr}"
        for c in (config.challenges or ["<challenge>"]):
            eval_block_lines.append(
                f"{banter_bin} run \\\n"
                f"  --challenge {c} \\\n"
                f"  --interface {iface.name} \\\n"
                f"  --mode {iface.mode}{flags} \\\n"
                f"  --skills {config.skills} \\\n"
                f"  --runs-root {runs_root}/inc<N>"
            )
    eval_block = "\n".join(eval_block_lines)

    ctx = dict(
        session_id=session_id,
        testbed_root=testbed_root,
        runs_root=runs_root,
        increments_path=increments_path,
        hierarchy_note=hierarchy_note,
        goals_lines=goals_lines,
        max_increments=config.budget.max_increments,
        n_challenges=n_challenges,
        n_interfaces=n_interfaces,
        runs_per_increment=runs_per_increment,
        total_runs=config.budget.max_increments * runs_per_increment,
        cost_cap=cost_cap,
        challenges_lines=challenges_lines,
        interfaces_table=interfaces_table,
        starting_skills=config.skills,
        skills_note=skills_note,
        scope_desc=scope_desc,
        scope_deny=scope_deny,
        first_iface_name=first_iface.name,
        first_iface_mode=first_iface.mode,
        next_iface_version=next_iface_version,
        run_dir_example=run_dir_example,
        banter_bin=banter_bin,
        eval_block=eval_block,
        avail_interfaces=avail_interfaces,
        avail_skills=avail_skills,
        recent_results=recent_results,
        groups_block=groups_block,
    )
    template = (testbed_root / "prompts" / "researcher.md").read_text()
    return template.format(**ctx)


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


def _version_root_for(testbed_root: Path, session: str | None) -> Path | None:
    """Session dir holding pinned interface versions, or None for the base."""
    if not session:
        return None
    return testbed_root / "results" / "autoresearch" / session


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

    # The researcher only kicks in once every required interface is built, set
    # up, authenticated, and tested — deterministically, no AI — and any skill
    # is accessible. Fail fast before spawning the researcher otherwise.
    from banter import preflight as preflight_mod
    reqs = [
        preflight_mod.Requirement(
            interface=i.name,
            mode=i.mode,
            interface_version=i.version,
            version_root=_version_root_for(testbed_root, i.session),
            skills=config.skills,
        )
        for i in config.interfaces
    ]
    try:
        preflight_mod.preflight(reqs, auth=config.engineer_auth, model=config.engineer_model)
    except preflight_mod.PreflightError as e:
        raise RuntimeError(
            f"[autoresearch] preflight failed — fix before the researcher can start:\n{e}"
        )

    from banter import results as results_mod
    parent = runs_root / "autoresearch"
    session_id = results_mod.next_session_id(parent)
    session_dir = parent / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_researcher_prompt(config, testbed_root, session_dir, session_id, banter_bin)
    (session_dir / "prompt.txt").write_text(prompt)

    transcript_path = session_dir / "transcript.jsonl"
    stderr_path = session_dir / "researcher.stderr.log"

    env = os.environ.copy()
    env.pop("ANTHROPIC_BASE_URL", None)
    if config.researcher_auth == "login":
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

    n_ifaces = len(config.interfaces)
    n_chals = len(config.challenges)
    iface_list = ", ".join(str(i) for i in config.interfaces)
    print(
        f"[autoresearch] session={session_id}\n"
        f"  interfaces={iface_list}\n"
        f"  goals={[f'{g.direction}({g.metric})' for g in config.goals]}\n"
        f"  budget={config.budget.max_increments} increments × {n_ifaces} interfaces × {n_chals} challenges "
        f"= up to {config.budget.max_increments * n_ifaces * n_chals} runs\n"
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

    # Researcher Claude shares the engineer's rate-limit retry helper: 12h budget,
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

    # Deterministically aggregate the per-increment run CSVs up to the session
    # results.csv (one row per increment) and the top-level autoresearch
    # results.csv (one before/after row per session). No AI involved.
    try:
        from banter import results as results_mod
        results_mod.rollup_autoresearch(session_dir)
    except Exception as e:
        print(f"[autoresearch] rollup skipped: {e}", flush=True)

    print(f"\n[autoresearch] done. session dir: {session_dir}", flush=True)
