"""results.csv writer.

One row per (challenge, platform, interface, skills, auth) run. Token/cost columns
come from the stream-json transcript's `result` event — Claude Code reports
`total_cost_usd` and aggregated `usage` there. Command counts are derived
from the same transcript by walking each `tool_use` block. Medal/score come
from the MLE-bench grader.
"""
from __future__ import annotations

import csv
import json
import re
import shlex
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_PYTHON_PREFIXES = ("python", "python3", "uv run python", "uv run", "pip", "pip3")
_PY_INTERPRETER_BASENAMES = ("python", "python3", "pip", "pip3")


# High-level (one row per session) summary schemas. These live at the top of
# each runner's results tree: results/<benchmark|autoresearch>/results.csv.
# Metrics averaged at every results.csv level — one per numeric per-run column
# in FIELDS, so the rollups reflect ALL per-run metrics. `platform_invocations`
# is the derived remote-platform delegation total (cli + mcp + sdk calls).
AGG_METRICS = [
    "score",
    "eng_wall_time_s", "eng_input_tokens", "eng_output_tokens", "eng_total_tokens", "eng_cost_usd",
    "res_wall_time_s", "res_input_tokens", "res_output_tokens", "res_total_tokens", "res_cost_usd",
    "total_wall_time_s", "total_tokens", "total_cost",
    "llm_calls", "cli_calls", "mcp_calls", "sdk_calls", "python_calls",
    "bash_calls", "other_tool_calls", "skill_calls",
    "platform_invocations",
    # `valid_submission` is the only grading bool we keep in FIELDS; averaging
    # it gives the valid-submission rate across a combo's runs.
    "valid_submission",
]

# Canonical metrics that are ALWAYS tracked + charted in every autoresearch
# run, with their natural improvement direction. The autoresearch config's
# `goals` list only decides WHICH of these (or others) the researcher
# optimizes for; charts are produced for the full TRACKED_METRICS set
# regardless of what the config picks. Order = chart order in the notebook.
# Per-category tool-call counts tracked on every (engineer) run. Each gets a
# cumulative running ("rolling") average column `<cat>_avg` — the mean of that
# count across results.csv rows up to and including the row, recomputed on every
# append/replace (see `recompute_rolling_averages`). Directions are cosmetic
# (chart-title labels); more interface use (cli/mcp/sdk/skill) is "good",
# self-written code (python/bash) and raw turn count are "bad".
CALL_COUNT_DIRECTIONS: dict[str, str] = {
    "llm_calls": "minimize",
    "cli_calls": "maximize",
    "mcp_calls": "maximize",
    "sdk_calls": "maximize",
    "python_calls": "minimize",
    "bash_calls": "minimize",
    "skill_calls": "maximize",
    "other_tool_calls": "minimize",
}
CALL_COUNT_COLS = tuple(CALL_COUNT_DIRECTIONS)

# One chart per tracked metric. Wall time + every call category additionally
# carry a rolling cumulative average, drawn as a dotted overlay on the same
# chart (see `rolling_avg_col`) rather than as a separate diagram.
TRACKED_METRICS: list[tuple[str, str]] = [
    ("score", "maximize"),
    ("total_tokens", "minimize"),
    ("eng_wall_time_s", "minimize"),
    # Charted for visibility only (not an optimization target) — eng + researcher
    # wall time is partly controller overhead, so it gets a neutral `observe`
    # label rather than minimize/maximize.
    ("total_wall_time_s", "observe"),
    ("total_cost", "minimize"),
    *((c, d) for c, d in CALL_COUNT_DIRECTIONS.items()),
]

# Benchmark CSV uses plain metric names (no `eng_`/`total_` prefix since
# there's no researcher). Same canonical list, different wall-time name.
BENCHMARK_TRACKED_METRICS: list[tuple[str, str]] = [
    ("score", "maximize"),
    ("total_tokens", "minimize"),
    ("wall_time_s", "minimize"),
    ("cost_usd", "minimize"),
    *((c, d) for c, d in CALL_COUNT_DIRECTIONS.items()),
]


# Metric column → its rolling-average column. Rolling averages are an
# AUTORESEARCH-only concept (a cumulative average across runs accumulates across
# versions); benchmark is a single session of independent (task, challenge)
# rows with nothing to accumulate, so it has no `*_avg` columns. This map plus
# `CALL_COUNT_COLS` is the single source of raw→avg pairs, used both for chart
# overlays (`rolling_avg_col`) and for the recompute (`_rolling_pairs`).
_ROLLING_AVG_OF = {
    "score": "score_avg",
    "total_tokens": "total_tokens_avg",
    "total_cost": "total_cost_avg",
    "eng_wall_time_s": "eng_wall_time_avg_s",
    "total_wall_time_s": "total_wall_time_avg_s",
}


def rolling_avg_col(metric: str) -> str | None:
    """The rolling-average column to overlay (dotted) on `metric`'s chart, or
    None when the metric has no rolling average."""
    if metric in _ROLLING_AVG_OF:
        return _ROLLING_AVG_OF[metric]
    if metric in CALL_COUNT_COLS:
        return f"{metric}_avg"
    return None

# Per-run rollup (benchmark): one row per <run> folder, averaging its
# per-challenge rows. Written to results/benchmark/results.csv.
COMBO_SUMMARY_FIELDS = (
    ["combo", "platform", "interface", "skills", "n_runs"]
    + [f"avg_{m}" for m in AGG_METRICS]
    + ["dir"]   # the run folder
)
# Autoresearch has no global rollup CSV: each session writes its own
# <run>/results.csv (one row per (run, version, task, challenge), columns =
# FIELDS below) and the researcher reports the best increment in report.md.


def _avg(values: list[float]) -> str:
    nums = [v for v in values if v is not None]
    return f"{sum(nums) / len(nums):.4f}" if nums else ""


def _read_runs(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _num_col(runs: list[dict[str, str]], key: str) -> list[float]:
    out: list[float] = []
    for r in runs:
        try:
            out.append(float(r.get(key, "")))
        except (TypeError, ValueError):
            pass
    return out


def _platform_col(runs: list[dict[str, str]]) -> list[float]:
    """Per-run remote-platform invocations = cli + mcp + sdk calls."""
    out: list[float] = []
    for r in runs:
        total, seen = 0.0, False
        for k in ("cli_calls", "mcp_calls", "sdk_calls"):
            try:
                total += float(r.get(k, ""))
                seen = True
            except (TypeError, ValueError):
                pass
        if seen:
            out.append(total)
    return out


def _agg_runs(runs: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_runs": len(runs)}
    for m in AGG_METRICS:
        col = _platform_col(runs) if m == "platform_invocations" else _num_col(runs, m)
        out[f"avg_{m}"] = _avg(col)
    return out


def roll_up_combos(parent: Path) -> list[dict[str, Any]]:
    """Average each `<combo>/results.csv` run folder under `parent` into one row
    and (over)write `parent/results.csv` (COMBO_SUMMARY_FIELDS). Returns the rows.

    A "combo" folder is any immediate subdir holding a detailed per-challenge
    results.csv (e.g. `0_no_interface_no_type_no_skills_no_session_v0/`).
    """
    rows: list[dict[str, Any]] = []
    if not parent.exists():
        return rows
    for d in sorted(p for p in parent.iterdir() if p.is_dir()):
        runs = _read_runs(d / "results.csv")
        if not runs:
            continue
        first = runs[0]
        rows.append({
            "combo": d.name,
            "platform": first.get("platform", ""),
            "interface": first.get("interface", ""),
            "skills": first.get("skills", ""),
            **_agg_runs(runs),
            "dir": str(d),
        })
    with open(parent / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMBO_SUMMARY_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in COMBO_SUMMARY_FIELDS})
    return rows


# The runner appends each challenge row directly to `<run>/results.csv`
# (one CSV per session, tagged with `session` + `increment`); the notebook
# computes its own per-increment means from those raw rows.


# Benchmark CSV has no researcher (no eng/res split needed) and no increments,
# so it uses plain column names (`wall_time_s`, `total_tokens`, …) rather than
# the autoresearch `eng_*` / `res_*` / `total_*` triple. Mapping from the
# autoresearch Row's fields → benchmark column names. Listed in the order they
# appear in the benchmark CSV; `BENCHMARK_FIELDS` is the .keys() view.
_BENCHMARK_VIEW = {
    "started_at": "started_at",
    "platform": "platform",
    "interface": "interface",
    "skills": "skills",
    "prev_run": "prev_run",
    "prev_version": "prev_version",
    "task": "task",
    "challenge": "challenge",
    "valid_submission": "valid_submission",
    "score": "score",
    "medal": "medal",
    # Engineer metrics: drop the `eng_` prefix in the benchmark CSV. Benchmark
    # has NO `*_avg` rolling columns — a benchmark session is one run of
    # independent (task, challenge) rows, with nothing to accumulate across.
    "wall_time_s": "eng_wall_time_s",
    "input_tokens": "eng_input_tokens",
    "output_tokens": "eng_output_tokens",
    "total_tokens": "eng_total_tokens",
    "cost_usd": "eng_cost_usd",
    "llm_calls": "llm_calls",
    "cli_calls": "cli_calls",
    "mcp_calls": "mcp_calls",
    "sdk_calls": "sdk_calls",
    "python_calls": "python_calls",
    "bash_calls": "bash_calls",
    "skill_calls": "skill_calls",
    "other_tool_calls": "other_tool_calls",
    "run_dir": "run_dir",
}


def next_session_id(parent: Path) -> str:
    """Next incrementing integer session id (as a string) under `parent`.

    Counts the leading integer of each child dir name — both pure-integer session
    dirs (autoresearch: `0`, `1`) and `<id>_<combo>` run folders (benchmark:
    `0_no_interface_..._v0`). Returns max+1, or "0" when there are none.
    """
    nums: list[int] = []
    if parent.exists():
        for p in parent.iterdir():
            if not p.is_dir():
                continue
            head = p.name.split("_", 1)[0]
            if head.isdigit():
                nums.append(int(head))
    return str(max(nums) + 1 if nums else 0)


def confirm_overwrite(path: Path, assume_yes: bool = False) -> bool:
    """If `path` exists, confirm before removing it; return True to proceed.

    Returns True when the dir is absent (nothing to do) or the user agreed
    (the dir is then removed so the caller can recreate it fresh). Returns
    False only when the user declined at the prompt. `assume_yes` (the CLI
    `-y/--yes` flag) skips the prompt and overwrites. A non-interactive stdin
    without `assume_yes` is treated as a decline — never silently clobber.
    """
    if not path.exists():
        return True
    if not assume_yes:
        if not sys.stdin or not sys.stdin.isatty():
            print(
                f"[banter] results dir already exists and stdin is not a TTY; "
                f"refusing to overwrite without --yes:\n  {path}",
                flush=True,
            )
            return False
        resp = input(
            f"Results dir already exists:\n  {path}\nOverwrite it? [y/N] "
        ).strip().lower()
        if resp not in ("y", "yes"):
            return False
    shutil.rmtree(path)
    return True


FIELDS = [
    # When the engineer's `claude -p` started (UTC ISO-8601). One per row.
    "started_at",
    # Run / version identity (autoresearch). `version` is `v<N>` (e.g. `v2`)
    # — the same string that names the increment's filesystem dir.
    "run",
    "version",
    "platform",
    "interface",       # interface: cli/mcp/sdk/none
    "skills",
    "prev_run",        # carried from autoresearch config; "" if not a continuation
    "prev_version",    # e.g. "v2" — last version of `prev_run` we continue from
    # The work this row covers.
    "task",
    "challenge",
    # Grading (slim set — full breakdown is in the per-challenge `grading.json`).
    "valid_submission",
    "score",
    "medal",
    # Engineer-side wall/tokens/cost — per-challenge values from the
    # engineer's `claude -p`.
    "eng_wall_time_s",
    "eng_input_tokens",
    "eng_output_tokens",
    "eng_total_tokens",
    "eng_cost_usd",
    # Researcher-side contribution attributed to this row. The researcher's
    # `claude -p` runs ONCE per autoresearch session; at end-of-session
    # `autoresearch.run_autoresearch` parses its transcript and distributes
    # the totals equally across all rows of the session. Empty/0 for
    # benchmark rows (no researcher).
    "res_wall_time_s",
    "res_input_tokens",
    "res_output_tokens",
    "res_total_tokens",
    "res_cost_usd",
    # Combined totals = eng_* + res_*. Pre-computed so the CSV is
    # self-contained (no need to derive them downstream). Config budgets
    # (`max_cost_usd`, `max_seconds`) check against these, not the engineer alone.
    "total_wall_time_s",
    "total_tokens",
    "total_cost",
    # Rolling cumulative averages across rows (filled by `append`; the dotted
    # "average across all runs" line on each chart). `total_*` are recomputed
    # after the researcher backfill in autoresearch.
    "score_avg",
    "total_tokens_avg",
    "total_cost_avg",
    "eng_wall_time_avg_s",
    "total_wall_time_avg_s",
    # Tool-call accounting parsed from the engineer's transcript.
    "llm_calls",
    "cli_calls",
    "mcp_calls",
    "sdk_calls",
    "python_calls",
    "bash_calls",
    "skill_calls",
    "other_tool_calls",
    # Per-category rolling cumulative averages of the counts above (the mean of
    # each count across rows up to and including the row). Recomputed by
    # `append` on every write so they stay correct as rows are added/replaced.
    "llm_calls_avg",
    "cli_calls_avg",
    "mcp_calls_avg",
    "sdk_calls_avg",
    "python_calls_avg",
    "bash_calls_avg",
    "skill_calls_avg",
    "other_tool_calls_avg",
    # Per-increment annotations the researcher fills in after evaluating the
    # increment (via `banter annotate-increment`). Every challenge row in the
    # same (session, increment) carries the same annotation — denormalised
    # so a single CSV is the full record.
    "hypothesis",         # one-line rationale for the change
    "change",             # what was modified vs. the previous increment
    "verdict",            # positive | negative | neutral
    "verdict_reason",
    "keep",               # 0/1 — did the researcher keep this change?
    "observations",       # qualitative findings from the engineer transcripts
    "proposed_changes",   # what to try next
    # Pointer to the engineer's per-challenge artifacts dir (transcript, stream.log, submission, grading.json, …).
    "run_dir",
]

# Benchmark CSV columns, in order. Sourced from `_BENCHMARK_VIEW.keys()`.
BENCHMARK_FIELDS = list(_BENCHMARK_VIEW.keys())


def _benchmark_view(row_dict: dict[str, Any]) -> dict[str, Any]:
    """Project an autoresearch Row dict into the benchmark column set.

    Renames `eng_*` → plain names, drops everything that isn't a benchmark
    column (session/increment/res_*/total_*/annotations).
    """
    return {dest: row_dict.get(src, "") for dest, src in _BENCHMARK_VIEW.items()}


@dataclass
class Row:
    # UTC ISO-8601 timestamp recorded when the engineer's `claude -p` started.
    started_at: str
    # Identity. Field order MATCHES `FIELDS` so csv.DictWriter emits columns
    # in the documented order.
    run: str                # autoresearch run id (empty for benchmark rows)
    version: str            # `v<N>` (e.g. "v2"); "" for benchmark rows
    platform: str           # platform NAME, e.g. "mlkit" or "none"
    interface: str          # interface: cli/mcp/sdk/none
    skills: str             # skill bundle name or "none"
    prev_run: str           # autoresearch config continuation hint; "" if none
    prev_version: str       # e.g. "v2"; "" if none
    task: str               # ML task / challenge group this challenge belongs to
    challenge: str          # MLE-bench competition id
    # Slim grading (full breakdown in the per-challenge `grading.json`).
    valid_submission: int = 0
    score: float | None = None
    medal: str | None = None
    # Engineer-side wall + tokens + cost.
    eng_wall_time_s: float = 0.0
    eng_input_tokens: int = 0
    eng_output_tokens: int = 0
    eng_total_tokens: int = 0
    eng_cost_usd: float = 0.0
    # Researcher attribution (set at end-of-session by autoresearch).
    res_wall_time_s: float = 0.0
    res_input_tokens: int = 0
    res_output_tokens: int = 0
    res_total_tokens: int = 0
    res_cost_usd: float = 0.0
    # Combined totals (eng + res); also written at end-of-session.
    total_wall_time_s: float = 0.0
    total_tokens: int = 0
    total_cost: float = 0.0
    llm_calls: int = 0
    cli_calls: int = 0
    mcp_calls: int = 0
    sdk_calls: int = 0
    python_calls: int = 0
    bash_calls: int = 0
    skill_calls: int = 0
    other_tool_calls: int = 0
    # Rolling cumulative averages across results.csv rows (the mean of the
    # corresponding raw column up to and including each row). All filled by
    # `append` at write time, so they're left as None on a fresh Row.
    score_avg: float | None = None
    total_tokens_avg: float | None = None
    total_cost_avg: float | None = None
    eng_wall_time_avg_s: float | None = None
    total_wall_time_avg_s: float | None = None
    llm_calls_avg: float | None = None
    cli_calls_avg: float | None = None
    mcp_calls_avg: float | None = None
    sdk_calls_avg: float | None = None
    python_calls_avg: float | None = None
    bash_calls_avg: float | None = None
    skill_calls_avg: float | None = None
    other_tool_calls_avg: float | None = None
    # Per-increment annotations (researcher fills in via `banter annotate-increment`).
    hypothesis: str = ""
    change: str = ""
    verdict: str = ""          # "positive" | "negative" | "neutral"
    verdict_reason: str = ""
    keep: int = 0              # 0 / 1
    observations: str = ""
    proposed_changes: str = ""
    run_dir: str = ""


def parse_transcript_usage(transcript_path: Path) -> dict[str, Any]:
    """Token + cost totals from Claude Code's stream-json transcript.

    Claude Code emits assistant messages carrying a `message.usage` block per
    turn, and a final `result` event with `total_cost_usd` and aggregated
    `usage`. We prefer the `result` event when present and fall back to
    summing per-turn usages otherwise.
    """
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "llm_calls": 0}
    if not transcript_path.exists():
        return totals

    final_result: dict[str, Any] | None = None
    per_turn_input = 0
    per_turn_output = 0
    turn_count = 0

    for line in transcript_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "result":
            final_result = event
            continue

        if event.get("type") == "assistant":
            usage = (event.get("message") or {}).get("usage") or {}
            if usage:
                per_turn_input += int(usage.get("input_tokens") or 0)
                per_turn_output += int(usage.get("output_tokens") or 0)
                turn_count += 1

    if final_result:
        usage = final_result.get("usage") or {}
        totals["input_tokens"] = int(usage.get("input_tokens") or per_turn_input)
        totals["output_tokens"] = int(usage.get("output_tokens") or per_turn_output)
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
        totals["cost_usd"] = float(final_result.get("total_cost_usd") or 0.0)
        totals["llm_calls"] = int(final_result.get("num_turns") or turn_count)
    else:
        totals["input_tokens"] = per_turn_input
        totals["output_tokens"] = per_turn_output
        totals["total_tokens"] = per_turn_input + per_turn_output
        totals["llm_calls"] = turn_count
    return totals


def _is_python_first(first: str) -> bool:
    return (
        first in _PYTHON_PREFIXES
        or first.endswith(("/python", "/python3", "/pip", "/pip3"))
        or first.endswith(".py")
    )


def _sdk_import_pattern(module: str) -> re.Pattern[str]:
    m = re.escape(module)
    # Matches `import <m>`, `from <m> ...`, or `python -m <m>...`. `\b` (not
    # `\s`) before the keyword so quoting chars like `"`/`'` don't block the
    # match — `python -c "import <m>"` is the common case.
    return re.compile(rf"\b(?:import|from)\s+{m}\b|-m\s+{m}\b")


def _command_uses_sdk(tokens: list[str], command: str, sdk_module: str, run_dir: Path | None) -> bool:
    pattern = _sdk_import_pattern(sdk_module)
    if pattern.search(command):
        return True
    # `python3 some_script.py [args]` — peek the file if it lives in run_dir.
    if run_dir is None:
        return False
    for tok in tokens[1:]:
        if not tok.endswith(".py"):
            continue
        candidate = (run_dir / tok) if not tok.startswith("/") else Path(tok)
        try:
            if candidate.is_file() and pattern.search(candidate.read_text(errors="ignore")):
                return True
        except OSError:
            continue
    return False


_SEGMENT_SEPARATORS = {";", "&&", "||", "|", "&", "(", ")", "{", "}", "\n"}


def _is_env_var_assignment(tok: str) -> bool:
    """True for shell var-assignment tokens like `FOO=bar`, `BASE=/x`."""
    if "=" not in tok or tok.startswith("="):
        return False
    head = tok.split("=", 1)[0]
    if not head or not (head[0].isalpha() or head[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in head)


def _executable_tokens(tokens: list[str], depth: int = 0) -> list[str]:
    """Yield the executable token of each command SEGMENT in `tokens`.

    Walks the token list tracking command-segment boundaries (`;`, `&&`,
    `||`, `|`, `&`, etc.), skipping environment-variable-assignment prefixes
    (`FOO=bar python script.py` → python is the executable). When a segment
    starts with `bash`/`sh`/`zsh -c "<script>"`, recursively scans the
    quoted inner script so e.g. `bash -c "python train.py"` is correctly
    classified as a python call.
    """
    out: list[str] = []
    expect_exec = True
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _SEGMENT_SEPARATORS:
            expect_exec = True
            i += 1
            continue
        if not expect_exec:
            i += 1
            continue
        if _is_env_var_assignment(tok):
            i += 1
            continue  # still expecting the executable
        # bash/sh -c "..." → recurse into the quoted script body.
        if depth < 2 and tok in ("bash", "sh", "zsh") and i + 2 < len(tokens) and tokens[i + 1] == "-c":
            inner = tokens[i + 2]
            try:
                inner_tokens = shlex.split(inner)
            except ValueError:
                inner_tokens = inner.split()
            out.extend(_executable_tokens(inner_tokens, depth + 1))
            i += 3
            expect_exec = False
            continue
        out.append(tok)
        expect_exec = False
        i += 1
    return out


def _classify_tool_use(
    tool_name: str,
    tool_input: dict[str, Any],
    cli_binary: str | None = None,
    sdk_module: str | None = None,
    run_dir: Path | None = None,
) -> str:
    if tool_name == "Skill":
        return "skill"
    if tool_name.startswith("mcp__"):
        return "mcp"
    if tool_name != "Bash":
        return "other"
    command = (tool_input.get("command") or "").strip()
    if not command:
        return "bash"
    # Engineers commonly write multi-line bash scripts where the python
    # invocation isn't the first token (env-var assignments first, or
    # cd/setup lines before the actual call). `shlex.split` strips newlines,
    # so split by line first; within a line, segment by `;`/`&&`/`||`/`|`.
    exec_tokens: list[str] = []
    tokens: list[str] = []
    for line in command.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            line_tokens = shlex.split(line)
        except ValueError:
            line_tokens = line.split()
        tokens.extend(line_tokens)
        exec_tokens.extend(_executable_tokens(line_tokens))
    if not tokens:
        return "bash"
    if not exec_tokens:
        return "bash"
    # Priority: cli (interface usage) > python (counts even when nested in
    # a bash script) > bash (pure shell utilities).
    def _is_cli(tok: str) -> bool:
        return bool(cli_binary) and (tok == cli_binary or tok.endswith(f"/{cli_binary}"))
    if any(_is_cli(t) for t in exec_tokens):
        return "cli"
    if any(_is_python_first(t) for t in exec_tokens):
        if sdk_module and _command_uses_sdk(tokens, command, sdk_module, run_dir):
            return "sdk"
        return "python"
    return "bash"


def aggregate_commands(
    transcript_path: Path,
    cli_binary: str | None = None,
    sdk_module: str | None = None,
    run_dir: Path | None = None,
) -> dict[str, int]:
    """Count tool calls by category from the stream-json transcript.

    Each `assistant` event carries a `message.content` list; tool calls are
    blocks with `type=="tool_use"`. `cli_binary` promotes matching Bash calls
    to cli_calls; `sdk_module` (typically the platform name when interface=sdk)
    promotes python invocations that import or `-m`-run that module to
    sdk_calls. When a script file is run we also peek it (relative to
    `run_dir`) for SDK use.
    """
    counts = {
        "cli_calls": 0,
        "mcp_calls": 0,
        "sdk_calls": 0,
        "python_calls": 0,
        "bash_calls": 0,
        "skill_calls": 0,
        "other_tool_calls": 0,
    }
    bucket = {
        "cli": "cli_calls",
        "mcp": "mcp_calls",
        "sdk": "sdk_calls",
        "python": "python_calls",
        "bash": "bash_calls",
        "skill": "skill_calls",
        "other": "other_tool_calls",
    }
    if not transcript_path.exists():
        return counts
    if run_dir is None:
        run_dir = transcript_path.parent
    for line in transcript_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "assistant":
            continue
        for block in (event.get("message") or {}).get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            category = _classify_tool_use(
                block.get("name", ""),
                block.get("input") or {},
                cli_binary=cli_binary,
                sdk_module=sdk_module,
                run_dir=run_dir,
            )
            counts[bucket.get(category, "other_tool_calls")] += 1
    return counts


def write_commands_log(
    transcript_path: Path,
    commands_log: Path,
    cli_binary: str | None = None,
    sdk_module: str | None = None,
    run_dir: Path | None = None,
) -> None:
    """Render one JSONL line per tool call from the stream-json transcript.

    Mirrors the (currently flaky) PreToolUse hook so commands.jsonl is always
    populated — the hook is best-effort. Each line carries the tool name,
    category bucket, raw tool_input, and the assistant event's timestamp.
    """
    if not transcript_path.exists():
        commands_log.write_text("")
        return
    if run_dir is None:
        run_dir = transcript_path.parent
    lines: list[str] = []
    for line in transcript_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "assistant":
            continue
        timestamp = event.get("timestamp")
        session_id = event.get("session_id")
        for block in (event.get("message") or {}).get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_name = block.get("name", "")
            tool_input = block.get("input") or {}
            category = _classify_tool_use(
                tool_name,
                tool_input,
                cli_binary=cli_binary,
                sdk_module=sdk_module,
                run_dir=run_dir,
            )
            record = {
                "timestamp": timestamp,
                "session_id": session_id,
                "tool_name": tool_name,
                "category": category,
                "tool_input": tool_input,
            }
            lines.append(json.dumps(record, default=str))
    commands_log.write_text("\n".join(lines) + ("\n" if lines else ""))


def append(results_csv: Path, row: Row, fields: list[str] | None = None) -> None:
    """Write `row` to results.csv. If a previous row has the same `run_dir`
    (i.e. the same combo was re-run), it's replaced rather than appended, so
    each combo has at most one row at any time.

    `fields` narrows the column set. `None` (default) writes the full
    autoresearch schema. Pass `BENCHMARK_FIELDS` for benchmark: columns are
    renamed and trimmed via `_benchmark_view`.
    """
    cols = fields if fields is not None else FIELDS
    # Detect benchmark by column-list equality (callers import BENCHMARK_FIELDS
    # so either `is` or `==` works; `==` is robust across module re-imports).
    use_benchmark_view = cols == BENCHMARK_FIELDS
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    raw = asdict(row)
    new_row = _benchmark_view(raw) if use_benchmark_view else raw
    kept: list[dict[str, Any]] = []
    if results_csv.exists():
        with results_csv.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("run_dir") == new_row.get("run_dir"):
                    continue  # replaced by new_row below
                kept.append(r)
    # Rolling averages: recompute the cumulative running mean of each tracked
    # raw column over the rows in file order (kept first, the new/replacement
    # row last) so the `*_avg` columns stay correct after an append OR a
    # replace. Blank raw values don't contribute.
    ordered = kept + [new_row]
    recompute_rolling_averages(ordered, cols)
    with results_csv.open("w", newline="") as fw:
        writer = csv.DictWriter(fw, fieldnames=cols)
        writer.writeheader()
        for r in ordered:
            writer.writerow({k: r.get(k, "") for k in cols})


def _rolling_pairs(cols: list[str]) -> list[tuple[str, str]]:
    """(raw_col, avg_col) pairs to recompute, filtered to those present in
    `cols`. Derived from the single `_ROLLING_AVG_OF` map plus the per-category
    call counts — so benchmark (whose schema has no `*_avg` columns) yields an
    empty list and the recompute is a no-op there."""
    pairs = list(_ROLLING_AVG_OF.items()) + [(c, f"{c}_avg") for c in CALL_COUNT_COLS]
    return [(r, a) for r, a in pairs if r in cols and a in cols]


def recompute_rolling_averages(ordered: list[dict[str, Any]], cols: list[str]) -> None:
    """In-place fill each rolling `*_avg` column with the cumulative running
    mean of its raw column across `ordered` (file order). Skips pairs whose
    columns aren't in the active schema; blank/None raw values are excluded
    from the running mean (a blank row carries the mean-so-far forward)."""
    for raw_col, avg_col in _rolling_pairs(cols):
        running_sum = 0.0
        running_n = 0
        for r in ordered:
            try:
                x = float(r.get(raw_col))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                x = None  # blank/None → no sample this row
            if x is not None:
                running_sum += x
                running_n += 1
            r[avg_col] = f"{running_sum / running_n:.4f}" if running_n else ""


