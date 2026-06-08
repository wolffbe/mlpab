"""results.csv writer.

One row per (challenge, platform, interface, skills, auth) run. Token/cost
columns come from the stream-json transcript's `result` event (Claude Code's
`total_cost_usd` + aggregated `usage`); command counts from walking each
`tool_use` block in the same transcript; medal/score from the MLE-bench grader.
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


# One-row-per-session summary schemas, at the top of each runner's results tree:
# results/<benchmark|autoresearch>/results.csv. One metric per numeric per-run
# FIELDS column, so rollups reflect ALL per-run metrics. `platform_invocations`
# is the derived remote-platform delegation total (cli + mcp + sdk calls).
AGG_METRICS = [
    "score",
    "eng_wall_time_s", "eng_input_tokens", "eng_output_tokens", "eng_total_tokens", "eng_cost_usd",
    "res_wall_time_s", "res_input_tokens", "res_output_tokens", "res_total_tokens", "res_cost_usd",
    "total_wall_time_s", "total_tokens", "total_cost",
    "llm_calls", "cli_calls", "mcp_calls", "sdk_calls", "python_calls",
    "bash_calls", "other_tool_calls", "skill_calls",
    "whitelist_hits", "blacklist_hits",
    "platform_invocations",
    # Only grading bool kept in FIELDS; its average is the valid-submission rate.
    "valid_submission",
]

# Per-category tool-call counts tracked on every (engineer) run, with their
# improvement direction. Each gets a cumulative rolling-average column `<cat>_avg`
# (mean of the count across results.csv rows up to and including the row),
# recomputed on every append/replace (see `recompute_rolling_averages`).
# Directions are cosmetic chart-title labels: more interface use
# (cli/mcp/sdk/skill) is "good", self-written code (python/bash) is "bad".
CALL_COUNT_DIRECTIONS: dict[str, str] = {
    "llm_calls": "minimize",
    "cli_calls": "maximize",
    "mcp_calls": "maximize",
    "sdk_calls": "maximize",
    "python_calls": "minimize",
    "bash_calls": "minimize",
    "skill_calls": "maximize",
    "other_tool_calls": "minimize",
    # REST-endpoint coverage (from `endpoint_hits`, NOT aggregate_commands):
    # distinct whitelisted endpoints hit (more = better), forbidden calls (fewer).
    "whitelist_hits": "maximize",
    "blacklist_hits": "minimize",
}
CALL_COUNT_COLS = tuple(CALL_COUNT_DIRECTIONS)

# Metrics ALWAYS tracked + charted in every autoresearch run, with improvement
# direction. Config `goals` only picks WHICH the researcher optimizes for; charts
# cover the full set regardless. Order = chart order. One chart per metric; wall
# time + every call category also carry a rolling cumulative average drawn as a
# dotted overlay on the same chart (see `rolling_avg_col`), not a separate diagram.
TRACKED_METRICS: list[tuple[str, str]] = [
    ("score", "maximize"),
    ("total_tokens", "minimize"),
    ("eng_wall_time_s", "minimize"),
    # `observe` (neutral, not an optimization target): eng + researcher wall time
    # is partly controller overhead, so charted for visibility only.
    ("total_wall_time_s", "observe"),
    ("total_cost", "minimize"),
    *((c, d) for c, d in CALL_COUNT_DIRECTIONS.items()),
]

# Benchmark CSV: same canonical list, plain metric names (no `eng_`/`total_`
# prefix since there's no researcher) and a different wall-time name.
BENCHMARK_TRACKED_METRICS: list[tuple[str, str]] = [
    ("score", "maximize"),
    ("total_tokens", "minimize"),
    ("wall_time_s", "minimize"),
    ("cost_usd", "minimize"),
    *((c, d) for c, d in CALL_COUNT_DIRECTIONS.items()),
]


# Metric column → its rolling-average column. Rolling averages are
# AUTORESEARCH-only (a cumulative average accumulating across versions); benchmark
# is independent (task, challenge) rows with nothing to accumulate, so no `*_avg`
# columns. This map plus `CALL_COUNT_COLS` is the single source of raw→avg pairs,
# used for chart overlays (`rolling_avg_col`) and the recompute (`_rolling_pairs`).
_ROLLING_AVG_OF = {
    "score": "score_avg",
    "total_tokens": "total_tokens_avg",
    "total_cost": "total_cost_avg",
    "eng_wall_time_s": "eng_wall_time_avg_s",
    "total_wall_time_s": "total_wall_time_avg_s",
}


def rolling_avg_col(metric: str) -> str | None:
    """The dotted rolling-average column to overlay on `metric`'s chart, or None
    when the metric has no rolling average."""
    if metric in _ROLLING_AVG_OF:
        return _ROLLING_AVG_OF[metric]
    if metric in CALL_COUNT_COLS:
        return f"{metric}_avg"
    return None

# Per-run rollup (benchmark): one row per <run> folder averaging its per-challenge
# rows. Written to results/benchmark/results.csv.
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


# The runner appends each challenge row directly to `<run>/results.csv` (one CSV
# per session, tagged with `session` + `increment`); the notebook computes its own
# per-increment means from those raw rows.


# Benchmark CSV has no researcher (no eng/res split) and no increments, so it uses
# plain column names (`wall_time_s`, `total_tokens`, …) instead of the
# autoresearch `eng_*`/`res_*`/`total_*` triple. Maps autoresearch Row fields →
# benchmark column names, in benchmark-CSV order; `BENCHMARK_FIELDS` is the
# .keys() view.
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
    # Engineer metrics: drop the `eng_` prefix. Benchmark has NO `*_avg` rolling
    # columns — one run of independent (task, challenge) rows, nothing to accumulate.
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
    "whitelist_hits": "whitelist_hits",
    "blacklist_hits": "blacklist_hits",
    "error": "error",
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

    True when the dir is absent (nothing to do) or the user agreed (the dir is
    then removed so the caller recreates it fresh); False only on decline.
    `assume_yes` (CLI `-y/--yes`) skips the prompt and overwrites. Non-interactive
    stdin without `assume_yes` is treated as a decline — never silently clobber.
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
    # Engineer's `claude -p` start (UTC ISO-8601). One per row.
    "started_at",
    # Run / version identity (autoresearch). `version` is `v<N>` (e.g. `v2`) —
    # same string that names the increment's filesystem dir.
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
    # Grading (slim — full breakdown in the per-challenge `grading.json`).
    "valid_submission",
    "score",
    "medal",
    # Engineer-side wall/tokens/cost: per-challenge values from the engineer's
    # `claude -p`.
    "eng_wall_time_s",
    "eng_input_tokens",
    "eng_output_tokens",
    "eng_total_tokens",
    "eng_cost_usd",
    # Researcher-side contribution attributed to this row. The researcher's
    # `claude -p` runs ONCE per autoresearch session; at end-of-session
    # `autoresearch.run_autoresearch` parses its transcript and distributes the
    # totals equally across all rows. Empty/0 for benchmark rows (no researcher).
    "res_wall_time_s",
    "res_input_tokens",
    "res_output_tokens",
    "res_total_tokens",
    "res_cost_usd",
    # Combined totals = eng_* + res_*. Pre-computed so the CSV is self-contained.
    # Config budgets (`max_cost_usd`, `max_seconds`) check against these, not the
    # engineer alone.
    "total_wall_time_s",
    "total_tokens",
    "total_cost",
    # Rolling cumulative averages across rows (filled by `append`; the dotted
    # "average across all runs" line per chart). `total_*` are recomputed after
    # the researcher backfill in autoresearch.
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
    # REST-endpoint coverage (from the venv API-log shim; see `endpoint_hits`).
    "whitelist_hits",
    "blacklist_hits",
    # Per-category rolling cumulative averages of the counts above (mean across
    # rows up to and including the row). Recomputed by `append` on every write so
    # they stay correct as rows are added/replaced.
    "llm_calls_avg",
    "cli_calls_avg",
    "mcp_calls_avg",
    "sdk_calls_avg",
    "python_calls_avg",
    "bash_calls_avg",
    "skill_calls_avg",
    "other_tool_calls_avg",
    "whitelist_hits_avg",
    "blacklist_hits_avg",
    # Per-increment annotations the researcher fills in after evaluating the
    # increment (via `banter annotate-increment`). Every challenge row in the same
    # (session, increment) carries the same annotation — denormalised so a single
    # CSV is the full record.
    "hypothesis",         # one-line rationale for the change
    "change",             # what was modified vs. the previous increment
    "verdict",            # positive | negative | neutral
    "verdict_reason",
    "keep",               # 0/1 — did the researcher keep this change?
    "observations",       # qualitative findings from the engineer transcripts
    "proposed_changes",   # what to try next
    # Failure reason for a DEAD run (no valid submission): recorded with all
    # numeric metrics zeroed and this string set, so the researcher sees a
    # comparable score-0 row + why it crashed (vs a graceful give-up, which floors
    # a real submission with a low but non-zero score).
    "error",
    # Pointer to the engineer's per-challenge artifacts dir (engineer.log, submission, grading.json, …).
    "run_dir",
]

# Benchmark CSV columns, in order. Sourced from `_BENCHMARK_VIEW.keys()`.
BENCHMARK_FIELDS = list(_BENCHMARK_VIEW.keys())


def _benchmark_view(row_dict: dict[str, Any]) -> dict[str, Any]:
    """Project an autoresearch Row dict into the benchmark column set.

    Renames `eng_*` → plain names, drops non-benchmark columns
    (session/increment/res_*/total_*/annotations).
    """
    return {dest: row_dict.get(src, "") for dest, src in _BENCHMARK_VIEW.items()}


@dataclass
class Row:
    # UTC ISO-8601 recorded when the engineer's `claude -p` started.
    started_at: str
    # Identity. Field order MATCHES `FIELDS` so csv.DictWriter emits columns in
    # the documented order.
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
    # REST-endpoint coverage (from the venv API-log shim; see `endpoint_hits`).
    whitelist_hits: int = 0
    blacklist_hits: int = 0
    # Rolling cumulative averages across results.csv rows (mean of the raw column
    # up to and including each row). All filled by `append` at write time, so
    # left None on a fresh Row.
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
    whitelist_hits_avg: float | None = None
    blacklist_hits_avg: float | None = None
    # Per-increment annotations (researcher fills in via `banter annotate-increment`).
    hypothesis: str = ""
    change: str = ""
    verdict: str = ""          # "positive" | "negative" | "neutral"
    verdict_reason: str = ""
    keep: int = 0              # 0 / 1
    observations: str = ""
    proposed_changes: str = ""
    # Failure reason for a dead run (no valid submission); "" for normal runs.
    error: str = ""
    run_dir: str = ""


def parse_transcript_usage(transcript_path: Path) -> dict[str, Any]:
    """Token + cost totals from Claude Code's stream-json transcript.

    Claude Code emits per-turn assistant messages carrying `message.usage`, plus a
    final `result` event with `total_cost_usd` and aggregated `usage`. Prefer the
    `result` event; fall back to summing per-turn usages.
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
    # Matches `import <m>`, `from <m> ...`, or `python -m <m>...`. `\b` (not `\s`)
    # before the keyword so quoting chars (`"`/`'`) don't block the match — the
    # common case is `python -c "import <m>"`.
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

    Tracks segment boundaries (`;`, `&&`, `||`, `|`, `&`, etc.), skips env-var
    assignment prefixes (`FOO=bar python script.py` → python is the executable),
    and for a `bash`/`sh`/`zsh -c "<script>"` segment recursively scans the quoted
    inner script (so `bash -c "python train.py"` is classified as a python call).
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
    # Engineers often write multi-line bash where the python call isn't the first
    # token (env-var assignments, or cd/setup lines before it). `shlex.split`
    # strips newlines, so split by line first; within a line, segment by
    # `;`/`&&`/`||`/`|`.
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
    # Priority: cli (interface usage) > python (counts even when nested in bash) >
    # bash (pure shell utilities).
    def _is_cli(tok: str) -> bool:
        return bool(cli_binary) and (tok == cli_binary or tok.endswith(f"/{cli_binary}"))
    if any(_is_cli(t) for t in exec_tokens):
        return "cli"
    if any(_is_python_first(t) for t in exec_tokens):
        if sdk_module and _command_uses_sdk(tokens, command, sdk_module, run_dir):
            return "sdk"
        return "python"
    return "bash"


# Patterns meaning an interface tool runs python/compute LOCALLY instead of
# delegating to the remote platform. A researcher could "win" an interface by
# adding such a tool — laundering local compute as mcp/cli usage. Flagged in the
# engineer-facing entry-point files (MCP tools, CLI commands).
_LOCAL_EXEC_PATTERNS = [
    ("subprocess", re.compile(r"\bsubprocess\b")),
    ("os.system", re.compile(r"\bos\.system\s*\(")),
    ("os.popen", re.compile(r"\bos\.popen\s*\(")),
    ("Popen", re.compile(r"\bPopen\s*\(")),
    ("exec(", re.compile(r"(?<![A-Za-z_.])exec\s*\(")),
    ("runpy", re.compile(r"\brunpy\b")),
    ("sys.executable", re.compile(r"\bsys\.executable\b")),
]


def audit_interface_local_exec(
    interface_src: Path | str | None,
    baseline_src: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Flag engineer-facing interface tools that execute python/compute LOCALLY.

    Scans MCP-tool (`**/mcp/tools/*.py`) and CLI-command (`**/cli/commands/*.py`)
    source under `interface_src` for local-execution patterns (subprocess, exec,
    runpy, …). When `baseline_src` (the v0 interface source) is given, flag only
    files the researcher CHANGED or added — an unchanged upstream file using
    subprocess isn't laundering and is skipped.

    Returns `[{"file": <relpath>, "patterns": [...]}, ...]` (empty = clean).
    Enforces the remote-only contract: interface tools must delegate to the
    cluster, never run the work locally.
    """
    flagged: list[dict[str, Any]] = []
    if not interface_src:
        return flagged
    src = Path(interface_src)
    if not src.exists():
        return flagged
    base = Path(baseline_src) if baseline_src else None
    for path in sorted(src.rglob("*.py")):
        rel = path.relative_to(src).as_posix()
        tagged = f"/{rel}"
        if "/build/" in tagged or "/tests/" in tagged or "/test/" in tagged:
            continue  # built copies + test fixtures aren't engineer-facing
        is_entrypoint = "/mcp/tools/" in tagged or "/cli/commands/" in tagged
        # WITH a v0 baseline, scan ALL researcher-changed source (covers
        # SDK-interface laundering too) and flag only ADDED patterns — low false
        # positives. WITHOUT a baseline, scan only the narrow MCP/CLI entry points
        # (scanning all source would flag legit upstream subprocess).
        if base is None and not is_entrypoint:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        hits = [name for name, rx in _LOCAL_EXEC_PATTERNS if rx.search(text)]
        if not hits:
            continue
        if base is not None:
            counterpart = base / rel
            base_text = ""
            try:
                if counterpart.exists():
                    base_text = counterpart.read_text(errors="ignore")
            except OSError:
                base_text = ""
            if base_text == text:
                continue  # unchanged file — not researcher-introduced
            # Flag only patterns the researcher ADDED (absent from baseline).
            hits = [name for name, rx in _LOCAL_EXEC_PATTERNS
                    if rx.search(text) and not rx.search(base_text)]
            if not hits:
                continue
        flagged.append({"file": rel, "patterns": hits})
    return flagged


def _parse_endpoint_patterns(patterns: list[str] | None) -> list[tuple[str, str, "re.Pattern[str]"]]:
    """Parse `"<METHOD> <path-regex>"` strings into (source, method, compiled
    regex). `source` is the original string (kept for the covered/missed report);
    missing method = "any method"; unparseable entries skipped.

    The path regex is END-anchored (`(?:rx)/?$`, optional trailing slash) and
    matched with `.search`, so a prefix pattern can't over-match a deeper path —
    e.g. feature-view `POST .../featureview` must NOT also match the
    training-dataset `.../featureview/{name}/version/{v}/trainingdatasets` POST —
    without forcing a brittle start-anchor on the exact base prefix.
    """
    out: list[tuple[str, str, Any]] = []
    for p in patterns or []:
        s = str(p).strip()
        if not s:
            continue
        parts = s.split(None, 1)
        if len(parts) == 2 and parts[0].isalpha():
            method, rx = parts[0].upper(), parts[1].strip()
        else:
            method, rx = "", s
        try:
            out.append((s, method, re.compile(rf"(?:{rx})/?$")))
        except re.error:
            continue
    return out


def _read_api_log(api_log: Path | str) -> list[tuple[str, str, str]]:
    """Read the per-run API log into (METHOD, path, src) rows. Missing → [].

    `src` ∈ {mcp, cli, sdk, other} is the interface the call came through (set by
    the venv shim); "" for legacy logs without attribution."""
    p = Path(api_log)
    if not p.exists():
        return []
    rows: list[tuple[str, str, str]] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows.append((
            str(e.get("method", "")).upper(),
            str(e.get("path", "")),
            str(e.get("src", "")),
        ))
    return rows


def endpoint_coverage(
    api_log: Path | str,
    whitelist: list[str] | None,
    blacklist: list[str] | None,
    interface: str | None = None,
) -> dict[str, Any]:
    """Full REST endpoint-coverage breakdown from the per-run API log (written by
    the venv shim). Drives BOTH the `whitelist_hits`/`blacklist_hits` metrics AND
    the per-run `endpoint_coverage.json` the researcher reads to see WHICH
    lifecycle steps the engineer reached.

    `interface` ATTRIBUTES whitelist coverage: when set (cli/mcp/sdk), a target
    endpoint counts as covered only if hit THROUGH that interface (the shim's
    `src` tag) — hand-rolled `requests` ("other") and server-side Job calls (never
    logged) don't count, so `whitelist_hits` reflects real interface usage.
    `interface=None` counts any logged call (back-compat / unattributed logs).
    Blacklist is NOT attribution-filtered: a forbidden call is a violation however
    made.

    Returns: `whitelist_hits` (# distinct whitelist patterns hit), `blacklist_hits`
    (# log rows matching any blacklist pattern), `total_whitelist`, the
    `covered`/`missed` pattern-source lists (a `missed` endpoint = a capability the
    interface must expose), and the original `whitelist`/`blacklist` patterns
    scored against — so the persisted `endpoint_coverage.json` stays self-contained
    after the raw `api_calls.jsonl` is discarded.
    """
    rows = _read_api_log(api_log)
    # Whitelist coverage scores only calls through the interface under test (the
    # shim's `src` tag); blacklist sees every call.
    wl_rows = [
        (m, p) for m, p, so in rows if interface is None or so == interface
    ]

    def hit(method: str, rx: "re.Pattern[str]", m: str, path: str) -> bool:
        return (not method or method == m) and bool(rx.search(path))

    wl = _parse_endpoint_patterns(whitelist)
    covered = [src for src, method, rx in wl if any(hit(method, rx, m, p) for m, p in wl_rows)]
    covered_set = set(covered)
    missed = [src for src, _m, _rx in wl if src not in covered_set]
    bl = _parse_endpoint_patterns(blacklist)
    blacklist_hits = sum(
        1 for m, p, _so in rows if any(hit(method, rx, m, p) for _s, method, rx in bl)
    )
    return {
        "whitelist_hits": len(covered),
        "blacklist_hits": blacklist_hits,
        "total_whitelist": len(wl),
        "covered": covered,
        "missed": missed,
        "whitelist": list(whitelist or []),
        "blacklist": list(blacklist or []),
    }


def endpoint_hits(
    api_log: Path | str,
    whitelist: list[str] | None,
    blacklist: list[str] | None,
    interface: str | None = None,
) -> dict[str, int]:
    """The two scored metrics (see `endpoint_coverage` for the full breakdown):
    `whitelist_hits` = # distinct whitelist patterns hit ≥once THROUGH `interface`;
    `blacklist_hits` = # log rows matching any blacklist pattern."""
    c = endpoint_coverage(api_log, whitelist, blacklist, interface=interface)
    return {"whitelist_hits": c["whitelist_hits"], "blacklist_hits": c["blacklist_hits"]}


def aggregate_commands(
    transcript_path: Path,
    cli_binary: str | None = None,
    sdk_module: str | None = None,
    run_dir: Path | None = None,
) -> dict[str, int]:
    """Count tool calls by category from the stream-json transcript.

    Tool calls are `type=="tool_use"` blocks in each `assistant` event's
    `message.content`. `cli_binary` promotes matching Bash calls to cli_calls;
    `sdk_module` (typically the platform name when interface=sdk) promotes python
    invocations that import or `-m`-run that module to sdk_calls; a run script file
    is also peeked (relative to `run_dir`) for SDK use.
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

    Mirrors the (flaky, best-effort) PreToolUse hook so commands.jsonl is always
    populated. Each line carries the tool name, category bucket, raw tool_input,
    and the assistant event's timestamp.
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


# Substrings marking a client-side failure worth flagging at the top of the
# collected log (the engineer otherwise only sees "No such tool available").
_CRASH_MARKERS = (
    "Traceback (most recent call last)",
    "Connection failed",
    "Connection closed",
    "Server disconnected",
)


def _slug_for(path: Path) -> str:
    """Claude Code keys its per-project cache dir on the cwd, with every path
    separator turned into a dash (e.g. /a/b → -a-b)."""
    return str(Path(path).resolve()).replace("/", "-")


def _flatten_text(content: Any) -> str:
    """tool_result `content` is either a string or a list of {type,text} blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return ""


def _mcp_server_log(caches_dir: Path, run_dir: Path, server: str) -> list[str]:
    """Lines Claude buried for one stdio MCP server during THIS run, oldest first.

    Each `mcp-logs-<server>/*.jsonl` line is either a dict (debug/error events with
    a timestamp) or a bare `"Server stderr: ..."` string carrying the server's own
    stdout/stderr — including a startup traceback that kills the process before any
    tool registers. Both are surfaced.
    """
    out: list[str] = []
    # Cache dir is keyed on the engineer's cwd (the challenge dir); fall back to a
    # glob so a layout change doesn't silently drop the logs.
    candidates = [caches_dir / _slug_for(run_dir) / f"mcp-logs-{server}"]
    candidates += [
        p for p in caches_dir.glob(f"*/mcp-logs-{server}") if p not in candidates
    ]
    for logdir in candidates:
        if not logdir.is_dir():
            continue
        for jf in sorted(logdir.glob("*.jsonl")):
            for line in jf.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    out.append(line)
                    continue
                if isinstance(rec, str):
                    out.append(rec)
                    continue
                if isinstance(rec, dict):
                    ts = rec.get("timestamp", "")
                    body = (
                        rec.get("message")
                        or rec.get("error")
                        or rec.get("debug")
                        or json.dumps(rec, default=str)
                    )
                    out.append(f"{ts} {body}".strip() if ts else str(body))
    return out


def _transcript_crashes(transcript_path: Path) -> list[str]:
    """Errored / traceback-bearing tool results from the engineer transcript.

    Covers the cli (`hops …` subprocess) and sdk (`import hopsworks`) paths, where
    a client crash surfaces as a Python traceback or an errored tool result rather
    than in a separate server log."""
    out: list[str] = []
    if not transcript_path.exists():
        return out
    for line in transcript_path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "user":
            continue
        for block in (event.get("message") or {}).get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            text = _flatten_text(block.get("content"))
            is_err = bool(block.get("is_error"))
            if not text:
                continue
            if is_err or any(m in text for m in _CRASH_MARKERS):
                out.append(text.strip()[:4000])
    return out


def collect_client_logs(
    run_dir: Path,
    boundary: Path,
    interface: str,
    platform: str,
    mcp_servers: dict[str, Any] | None = None,
    transcript_path: Path | None = None,
) -> dict[str, Any]:
    """Write `<run_dir>/<platform>_client.logs` for a cli/mcp/sdk run.

    The interface client's runtime output is otherwise scattered and, for mcp,
    effectively invisible: Claude launches the stdio server and buries its stderr +
    connection status under
    `<HOME>/Library/Caches/claude-cli-nodejs/<cwd-slug>/mcp-logs-<server>/*.jsonl`.
    A server that crashes at startup leaves the engineer with only "No such tool
    available", reading like an empty interface rather than a crash. This
    aggregates, per run:

    - **mcp** — every `mcp-logs-<server>` line for the run's servers.
    - **cli / sdk** — errored / traceback-bearing tool results from the
      transcript (the client runs in-process / as a subprocess there).

    Always written for these interfaces (so each version has it). Returns
    ``{"path", "crashed", "markers"}``; ``crashed`` is True when any crash marker
    was seen, so the caller can warn.
    """
    out_path = run_dir / f"{platform}_client.logs"
    if interface not in ("cli", "mcp", "sdk"):
        return {"path": str(out_path), "crashed": False, "markers": []}

    caches_dir = Path(boundary) / "Library" / "Caches" / "claude-cli-nodejs"
    sections: list[str] = []
    body_for_scan: list[str] = []

    if interface == "mcp":
        for server in sorted((mcp_servers or {}).keys()):
            lines = _mcp_server_log(caches_dir, run_dir, server)
            header = f"===== MCP server: {server} ====="
            if lines:
                sections.append(header + "\n" + "\n".join(lines))
                body_for_scan.extend(lines)
            else:
                sections.append(header + "\n(no server log captured)")

    crashes = _transcript_crashes(transcript_path) if transcript_path else []
    if crashes:
        sections.append(
            "===== client errors (engineer transcript) =====\n"
            + "\n---\n".join(crashes)
        )
        body_for_scan.extend(crashes)

    blob = "\n".join(body_for_scan)
    markers = [m for m in _CRASH_MARKERS if m in blob]
    crashed = bool(markers)

    head = (
        f"# {platform} client log ({interface} interface)\n"
        f"# crashed={crashed}"
        + (f"  markers={markers}" if markers else "")
        + "\n\n"
    )
    out_path.write_text(head + ("\n\n".join(sections) if sections else "(no client output captured)\n"))
    return {"path": str(out_path), "crashed": crashed, "markers": markers}


def append(results_csv: Path, row: Row, fields: list[str] | None = None) -> None:
    """Write `row` to results.csv. A previous row with the same `run_dir` (same
    combo re-run) is replaced rather than appended, so each combo has at most one
    row at any time.

    `fields` narrows the column set. `None` (default) writes the full autoresearch
    schema. Pass `BENCHMARK_FIELDS` for benchmark: columns are renamed and trimmed
    via `_benchmark_view`.
    """
    cols = fields if fields is not None else FIELDS
    # Detect benchmark by column-list equality (callers import BENCHMARK_FIELDS, so
    # either `is` or `==` works; `==` is robust across module re-imports).
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
    # Rolling averages: recompute the cumulative running mean of each tracked raw
    # column in file order (kept first, new/replacement row last) so `*_avg` stays
    # correct after append OR replace. Blank raw values don't contribute.
    ordered = kept + [new_row]
    recompute_rolling_averages(ordered, cols)
    with results_csv.open("w", newline="") as fw:
        writer = csv.DictWriter(fw, fieldnames=cols)
        writer.writeheader()
        for r in ordered:
            writer.writerow({k: r.get(k, "") for k in cols})


def _rolling_pairs(cols: list[str]) -> list[tuple[str, str]]:
    """(raw_col, avg_col) pairs to recompute, filtered to those present in `cols`.
    Derived from the single `_ROLLING_AVG_OF` map plus the per-category call counts
    — so benchmark (no `*_avg` columns) yields an empty list and the recompute is a
    no-op there."""
    pairs = list(_ROLLING_AVG_OF.items()) + [(c, f"{c}_avg") for c in CALL_COUNT_COLS]
    return [(r, a) for r, a in pairs if r in cols and a in cols]


def recompute_rolling_averages(ordered: list[dict[str, Any]], cols: list[str]) -> None:
    """In-place fill each rolling `*_avg` column with the cumulative running mean
    of its raw column across `ordered` (file order). Skips pairs whose columns
    aren't in the active schema; blank/None raw values are excluded (a blank row
    carries the mean-so-far forward)."""
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
