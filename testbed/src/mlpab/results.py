"""results.csv writer.

One row per (task, platform, interface, skills, auth) run. Token/cost
columns come from the stream-json transcript's `result` event (Claude Code's
`total_cost_usd` + aggregated `usage`); command counts from walking each
`tool_use` block in the same transcript; the assert tally
(asserts_passed/asserts_failed/asserts_skipped/total_asserts) from the
assertion-suite grading report.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# One parser for "which service follows the CLI binary": the same rule the
# enforcement hook applies, so counted interface_calls match what the hook
# actually allowed.
from mlpab.hooks.log_tool_call import _cli_arg_after as _hook_cli_arg_after


@contextlib.contextmanager
def _locked_csv(csv_path: Path):
    """Serialize cross-process read-modify-writes of a shared CSV.

    `append` and `roll_up_results` both READ the whole table and REWRITE
    it (run_dir dedup + `n` renumbering), so two treatment processes running in
    parallel (e.g. the per-platform rq1 configs) would silently drop each
    other's rows without this. flock on a sidecar `<name>.lock` (the CSV itself
    is replaced atomically, so its fd can't serve as the lock anchor).
    """
    lock_path = csv_path.with_name(csv_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _write_csv_atomic(csv_path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Replace `csv_path` in one rename, so lock-less readers (the analysis
    notebook, a parallel process's pre-lock peek) never see a torn file."""
    fd, tmp_name = tempfile.mkstemp(
        dir=str(csv_path.parent), prefix=f".{csv_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
        os.replace(tmp_name, csv_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


_PYTHON_PREFIXES = ("python", "python3", "uv run python", "uv run", "pip", "pip3")
_PY_INTERPRETER_BASENAMES = ("python", "python3", "pip", "pip3")


# Per-category tool-call count columns tracked on every (agent) run.
CALL_COUNT_COLUMNS: tuple[str, ...] = (
    "llm_calls",
    "cli_calls",
    "mcp_calls",
    "sdk_calls",
    "python_calls",
    "bash_calls",
    "skill_calls",
    # Explicit workspace-tool buckets (no catch-all "other").
    "read_calls",
    "write_calls",
    "edit_calls",
    "glob_calls",
    "grep_calls",
    "todo_calls",
    # Outcome friction: tool results that ERRORED. `denied_calls` = hook
    # rejections (the `DENIED:` enforcement messages — attempts to leave the
    # interface); `failed_commands` = every OTHER errored tool result (real
    # execution failures: bad CLI args, SDK exceptions, non-zero exits).
    "failed_commands",
    "denied_calls",
    # REST-endpoint coverage (from `endpoint_hits`, NOT aggregate_commands):
    # distinct whitelisted endpoints hit, and forbidden calls.
    "whitelist_hits",
    "blacklist_hits",
)

# Metrics ALWAYS tracked + charted on every run.
# Order = chart order. One chart per metric.
TRACKED_METRICS: list[str] = [
    "asserts_passed",
    "asserts_failed",
    "asserts_skipped",
    "total_asserts",
    "total_tokens",
    "wall_time_s",
    "cost_usd",
    *CALL_COUNT_COLUMNS,
]

# Results CSV: the cli/mcp/sdk triple collapsed into `interface_calls`, and no
# endpoint columns (no per-config endpoint policy is configured).
RESULTS_TRACKED_METRICS: list[str] = [
    "asserts_passed",
    "asserts_failed",
    "asserts_skipped",
    "total_asserts",
    "total_tokens",
    "wall_time_s",
    "cost_usd",
    "llm_calls",
    "interface_calls",
    "python_calls",
    "bash_calls",
    "skill_calls",
    "read_calls",
    "write_calls",
    "edit_calls",
    "glob_calls",
    "grep_calls",
    "todo_calls",
    "failed_commands",
    "denied_calls",
]


def _avg(values: list[float]) -> str:
    nums = [v for v in values if v is not None]
    return f"{sum(nums) / len(nums):.4f}" if nums else ""


def _read_runs(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


# The identity columns that name one combo (everything except the repeat `n`).
# A combo's results dir mirrors these, so this is the join key between a CSV row
# and an on-disk attempt folder.
COMBO_KEY_COLUMNS: tuple[str, ...] = (
    "model",
    "platform",
    "interface",
    "version",
    "skills",
    "category",
    "task",
)


def combo_key(row: dict[str, str]) -> tuple[str, ...]:
    """The combo identity of a CSV row: the COMBO_KEY_COLUMNS values, with an
    empty `skills` normalized to "none" (the config's literal `skills: none`)."""
    return tuple(
        (row.get(c, "") or "none") if c == "skills" else row.get(c, "")
        for c in COMBO_KEY_COLUMNS
    )


def prune_runs(csv_path: Path, config: str, combo_keys: set[tuple[str, ...]]) -> int:
    """Drop rows of `config` whose combo identity is in `combo_keys`; return the
    count removed. Used by --skip to purge DEAD combos (no valid=True attempt) so
    a restart re-runs them clean instead of leaving a stale row beside the retry.
    Cross-process safe (same lock + atomic replace as `append`/`roll_up`)."""
    if not combo_keys or not csv_path.exists():
        return 0
    with _locked_csv(csv_path):
        rows = _read_runs(csv_path)
        if not rows:
            return 0
        fieldnames = list(rows[0].keys())
        kept = [
            r
            for r in rows
            if not (r.get("config") == config and combo_key(r) in combo_keys)
        ]
        removed = len(rows) - len(kept)
        if removed:
            _write_csv_atomic(csv_path, fieldnames, kept)
        return removed


def _num_col(runs: list[dict[str, str]], key: str) -> list[float]:
    out: list[float] = []
    for r in runs:
        try:
            out.append(float(r.get(key, "")))
        except (TypeError, ValueError):
            pass
    return out


# The results CSV uses plain column names (`wall_time_s`, `total_tokens`, …).
# Maps Row fields → results column names, in results-CSV order;
# `RESULTS_FIELDS` is the .keys() view.
# Column order mirrors the run-folder hierarchy:
#   <config>/<model>/<platform>/<interface>/<version>/<skills>/<category>/<task>/<n>
# (run-dir paths use BOTH Row.category — the FTI stage — and Row.task — the
# sub-task; only the latter surfaces in the CSV, as the `task` column.)
_RESULTS_VIEW = {
    # The treatment config name (results/<config>/ — the yaml stem).
    "config": "run",
    "model": "model",
    "platform": "platform",
    "interface": "interface",
    # The interface build under test: pinned ref/version (git SHA / ==X.Y.Z pin /
    # package version).
    "version": "interface_ref",
    "skills": "skills",
    # The meaningful task id — the FTI sub-task (Row.task, e.g.
    # training_data/skew/drift). Row.category (the parent FTI stage) stays an
    # internal field only.
    "category": "category",  # FTI stage: feature / training / inference / ops
    "task": "task",
    # Repeat counter (derived; src None): 1 for the first execution of a given
    # (platform, interface, skills, category, task) config, 2 for its second
    # run, … Numbered against the global results.csv at append time;
    # roll_up_results re-numbers only when merging legacy leaf CSVs.
    "n": None,
    "started_at": "started_at",
    # Grading outcome (full breakdown in the per-run grading.json):
    #   valid   — the agent produced a gradeable deliverable (the deliverable-
    #             exists/columns assert, the FIRST assert, passed). False on a
    #             crash / no-deliverable run.
    #   success — the task was solved CORRECTLY (every assert green).
    # success ⊆ valid. asserts_passed/total give the partial breakdown.
    "valid": "valid",
    "success": "success",
    # Assertion-suite grading (full breakdown in the per-run grading.json).
    # `total_asserts` is the family's full suite size on every row; passed +
    # failed + skipped == total. `skipped` = checks not reached (a prerequisite
    # failed) or not applicable on this platform.
    "asserts_passed": "asserts_passed",
    "asserts_failed": "asserts_failed",
    "asserts_skipped": "asserts_skipped",
    "total_asserts": "total_asserts",
    # Agent metrics. `wall_time_s` is COMPUTE time; rate-limit back-off
    # sleeps land in `rate_limit_wait_s` instead.
    "wall_time_s": "wall_time_s",
    "rate_limit_wait_s": "rate_limit_wait_s",
    # Exact two-way split of `wall_time_s` (wall = platform + local):
    #   platform_time_s — seconds inside tool calls THROUGH the interface
    #     under test (cli/mcp/sdk): remote execution against the platform.
    #   local_time_s — everything else client-side (wall − platform): local
    #     tool execution + LLM generation. See `platform_tool_time`.
    "platform_time_s": "platform_time_s",
    "local_time_s": "local_time_s",
    # In-task foreground `sleep` (polling waits): count + total seconds. Part of
    # wall_time_s, broken out so idle waiting is distinguishable from compute.
    "sleep_calls": "sleep_calls",
    "sleep_time_s": "sleep_time_s",
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "total_tokens": "total_tokens",
    "cost_usd": "cost_usd",
    "llm_calls": "llm_calls",
    # The cli/mcp/sdk triple collapses into ONE column (src None = derived in
    # `_results_view`):
    #   interface_calls — calls through the interface UNDER TEST (cli row →
    #     cli_calls, mcp row → mcp_calls, sdk row → sdk_calls).
    #   bash_calls — plain shell PLUS any off-interface cli/mcp attempts (e.g.
    #     mcp tool calls during a cli run): not the interface, so not
    #     interface_calls. python stays separate (python_calls = local python
    #     that isn't the SDK under test).
    # No whitelist/blacklist either — no endpoint policy is configured for
    # treatment runs.
    "interface_calls": None,
    "python_calls": "python_calls",
    "bash_calls": None,
    "read_calls": "read_calls",
    "write_calls": "write_calls",
    "edit_calls": "edit_calls",
    "glob_calls": "glob_calls",
    "grep_calls": "grep_calls",
    "todo_calls": "todo_calls",
    "skill_calls": "skill_calls",
    "failed_commands": "failed_commands",
    "denied_calls": "denied_calls",
    "error": "error",
    "run_dir": "run_dir",
}

# Row field holding the interface's own call count, per interface kind.
_INTERFACE_CALL_SRC = {"cli": "cli_calls", "mcp": "mcp_calls", "sdk": "sdk_calls"}


def next_session_id(parent: Path) -> str:
    """Next incrementing integer session id (as a string) under `parent`.

    Counts the leading integer of each child dir name — both pure-integer
    session dirs (`0`, `1`) and `<id>_<combo>` run folders. Returns max+1,
    or "0" when there are none.
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
                f"[mlpab] results dir already exists and stdin is not a TTY; "
                f"refusing to overwrite without --yes:\n  {path}",
                flush=True,
            )
            return False
        resp = input(f"Results dir already exists:\n  {path}\nOverwrite it? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            return False
    shutil.rmtree(path)
    return True


# Results CSV columns, in order. Sourced from `_RESULTS_VIEW.keys()`.
RESULTS_FIELDS = list(_RESULTS_VIEW.keys())

# Results summary. One row PER EXECUTION (no
# averaging), appended by the runner after every run (already flat
# `_RESULTS_VIEW` names, incl. the per-row `interface_calls`). The single
# global table at results/results.csv is the ONLY results CSV
# (per-leaf results.csv files are deprecated); which run a row came from is
# recoverable from `run_dir`.
RESULTS_SUMMARY_FIELDS = list(RESULTS_FIELDS)


def _bench_identity(row: dict[str, Any]) -> tuple:
    """The config identity that `n` counts repeats of."""
    return tuple(
        str(row.get(k, ""))
        for k in (
            "config",
            "model",
            "platform",
            "interface",
            "version",
            "skills",
            "task",
            "category",
        )
    )


def roll_up_results(parent: Path, out_csv: Path) -> list[dict[str, Any]]:
    """Merge LEGACY per-leaf results.csv rows into the single global table at
    `out_csv` (results/results.csv) — one row per execution, NO
    averaging, no eng_/res_ split.

    Per-leaf CSVs are deprecated: the runner appends every new row straight
    into the global CSV. This merge exists for migration only — it folds in
    leaf rows under `parent`'s config dirs
    (results/<config>/**/results.csv, flat RESULTS_FIELDS columns)
    that never made it into the global table (sessions run before the
    deprecation, or a crashed session's leftovers). Rows already present in
    `out_csv` are kept as-is and deduped against leaf rows by `run_dir`.
    `n` is RE-numbered across the merged whole: the i-th execution of the same
    config (platform/interface/skills/category/task) gets n=i. Writes
    `out_csv` (header only when there is nothing) and returns the rows."""
    with _locked_csv(out_csv):
        # A pre-migration global CSV in the old combo-summary schema (avg_*
        # columns, no run_dir COLUMN) is not execution rows — drop those
        # rather than renumbering garbage into the table. Rows whose run_dir
        # column exists but is empty are kept (legacy executions without the
        # pointer).
        merged: list[dict[str, Any]] = [r for r in _read_runs(out_csv) if "run_dir" in r]
        have = {r["run_dir"] for r in merged if r.get("run_dir")}
        if parent.exists():
            for d in sorted(p for p in parent.iterdir() if p.is_dir()):
                for leaf_csv in sorted(d.rglob("results.csv")):
                    if leaf_csv.resolve() == out_csv.resolve():
                        continue  # the global table itself, already read above
                    for r in _read_runs(leaf_csv):
                        # The old code wrote each execution to BOTH a per-leaf
                        # CSV and a per-config rollup; rglob sees both copies,
                        # so dedup must track rows added THIS merge too. An
                        # empty run_dir can't be deduped — kept verbatim.
                        rd = r.get("run_dir")
                        if rd:
                            if rd in have:
                                continue  # already in the global table
                            have.add(rd)
                        merged.append(r)
        seen: dict[tuple, int] = {}
        for r in merged:
            ident = _bench_identity(r)
            seen[ident] = seen.get(ident, 0) + 1
            r["n"] = seen[ident]
        _write_csv_atomic(out_csv, RESULTS_SUMMARY_FIELDS, merged)
    return merged


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _results_view(row_dict: dict[str, Any]) -> dict[str, Any]:
    """Project a Row dict into the results column set.

    Drops columns outside the results table (the raw cli/mcp/sdk triple,
    endpoint hits). Derived columns:
      interface_calls — the call count of the row's OWN interface
        (cli→cli_calls, mcp→mcp_calls, sdk→sdk_calls; none → 0).
      bash_calls — plain bash PLUS the OFF-interface remainder of the
        cli/mcp/sdk triple (an mcp attempt during a cli run is not the
        interface — it's noise, bucketed with bash).
    """
    out = {dest: row_dict.get(src, "") for dest, src in _RESULTS_VIEW.items() if src}
    own_src = _INTERFACE_CALL_SRC.get(str(row_dict.get("interface", "")))
    triple = {k: _num(row_dict.get(k)) for k in ("cli_calls", "mcp_calls", "sdk_calls")}
    own = triple.pop(own_src) if own_src else 0.0
    out["interface_calls"] = f"{own:g}"
    out["bash_calls"] = f"{_num(row_dict.get('bash_calls')) + sum(triple.values()):g}"
    return {dest: out.get(dest, "") for dest in RESULTS_FIELDS}  # restore order


@dataclass
class Row:
    # UTC ISO-8601 recorded when the agent's `claude -p` started.
    started_at: str
    # Identity.
    run: str  # the treatment config name (the `config` column
    # of the results CSV)
    version: str  # raw Row field, always ""; the CSV `version`
    # column is sourced from `interface_ref`
    platform: str  # platform NAME, e.g. "hopsworks" or "none"
    interface: str  # interface: cli/mcp/sdk/none
    skills: str  # skill bundle name or "none"
    category: str  # FTI stage (feature/inference/ops) this task belongs to
    task: str  # FTI sub-task (an evals_provider family)
    # Interface build identity (pinned ref / version pin) — surfaced as the
    # results `version` column.
    interface_ref: str = ""
    # Agent model — the results `model` column.
    model: str = ""
    # Grading outcome. `valid` = a gradeable deliverable was produced (first
    # assert passed); `success` = the task was solved correctly (all asserts).
    valid: bool = False
    success: bool = False
    # Slim grading: assertion-suite tallies (full breakdown in the
    # per-run `grading.json`). passed + failed + skipped == total_asserts, the
    # family's full suite size (always reported, even on failed/no-deliverable
    # runs). `skipped` = checks not reached (prerequisite failed) or not
    # applicable on this platform.
    asserts_passed: int = 0
    asserts_failed: int = 0
    asserts_skipped: int = 0
    total_asserts: int = 0
    # Wall + tokens + cost from the agent's `claude -p`. Wall time is COMPUTE
    # time — rate-limit back-off sleeps are excluded and recorded in
    # `rate_limit_wait_s`.
    wall_time_s: float = 0.0
    rate_limit_wait_s: float = 0.0
    # Exact split of wall_time_s (wall = platform + local): seconds inside
    # interface (cli/mcp/sdk) tool calls = remote/platform execution; the
    # rest of wall = local (local tool execution + LLM generation). Spans are
    # stamped live at stream arrival (`claude_runner.ToolTimer`) — the
    # transcript events carry no timestamps.
    platform_time_s: float = 0.0
    local_time_s: float = 0.0
    # In-task `sleep` (foreground waits the agent issued, e.g. polling a job to
    # finish). Counted from issued Bash commands across ALL buckets — a sleep
    # chained onto an interface command (`cmd && sleep N`, or a sleep line in a
    # multi-line cli/sdk command) classifies as a cli/sdk call, so the sleep
    # would be missed if only bash_calls were scanned. This time is INCLUDED in
    # wall_time_s (foreground sleeps inflate compute), surfaced separately so it
    # can be told apart from real work.
    sleep_calls: int = 0
    sleep_time_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    cli_calls: int = 0
    mcp_calls: int = 0
    sdk_calls: int = 0
    python_calls: int = 0
    bash_calls: int = 0
    skill_calls: int = 0
    read_calls: int = 0
    write_calls: int = 0
    edit_calls: int = 0
    glob_calls: int = 0
    grep_calls: int = 0
    todo_calls: int = 0
    failed_commands: int = 0
    denied_calls: int = 0
    # REST-endpoint coverage (from the venv API-log shim; see `endpoint_hits`).
    whitelist_hits: int = 0
    blacklist_hits: int = 0
    # Failure reason for a dead run (no valid submission); "" for normal runs.
    error: str = ""
    run_dir: str = ""


# Testbed model alias → litellm-canonical id where litellm prices the model under
# a provider prefix. Mistral models register in litellm as `mistral/<api-id>`.
_LITELLM_ALIASES = {
    "mistral-medium-3.5": "mistral/mistral-medium-latest",
    "mistral-small-4": "mistral/mistral-small-latest",
    "mistral-large-3": "mistral/mistral-large-latest",
}


def usd_cost(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    price: dict | None = None,
) -> float | None:
    """Token→USD. A manual per-million-token `price` (from the treatment yaml's
    `prices:` block) wins when given — use it for models litellm can't price
    (e.g. `mistral-medium-3-5`). Otherwise via litellm — uniform across every
    agent engine. Tries the model id as-is, its litellm-canonical alias, and a
    generic `mistral/<id>` form (the claude-*/gpt-* ids litellm ships directly;
    mistral-* are priced under the `mistral/` provider prefix). None if there is
    no manual price and litellm can't price it either — the caller then keeps any
    engine-reported cost (e.g. Claude Code's `total_cost_usd`), which is 0 for
    codex/mistral.

    `price` is `{"input": <usd/M tokens>, "output": <usd/M tokens>}`; the manual
    path bills every input token at the input rate (no cache discount — the
    simple config defines none).

    `cache_read_input_tokens` is the cache-hit subset of `input_tokens` (NOT a
    separate addend); litellm rebills that portion at the model's discounted
    cache-read rate. Only pass it when `input_tokens` is cache-INCLUSIVE — codex
    reports it that way, so without this the cached bulk (≈90% on long runs) is
    billed at the full input rate. Claude's `input_tokens` already EXCLUDES cache,
    so its caller passes 0 here."""
    if int(input_tokens) == 0 and int(output_tokens) == 0:
        return 0.0
    # Manual price override: read from the yaml `prices:` block. If unset we fall
    # through to litellm; if litellm also can't price it the caller keeps 0.
    if price:
        try:
            pin = float(price.get("input", 0) or 0)
            pout = float(price.get("output", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            pin = pout = 0.0
        if pin or pout:
            return round((int(input_tokens) * pin + int(output_tokens) * pout) / 1_000_000, 6)
    if not model:
        return None
    cands = [model, _LITELLM_ALIASES.get(model)]
    if model.lower().startswith("mistral-"):
        cands.append("mistral/" + model)
    try:
        import litellm

        litellm.suppress_debug_info = True
    except Exception:
        return None
    for cand in cands:
        if not cand:
            continue
        try:
            pin, pout = litellm.cost_per_token(
                model=cand,
                prompt_tokens=int(input_tokens),
                completion_tokens=int(output_tokens),
                cache_read_input_tokens=int(cache_read_input_tokens) or None,
            )
            cost = round(float(pin) + float(pout), 6)
        except Exception:
            continue
        # A recognized-but-zero price (preview/free tiers, or a partial litellm
        # entry) must NOT override a real engine-reported cost — treat 0 as
        # "litellm can't price this" and keep looking / fall back to None.
        if cost > 0:
            return cost
    return None


def parse_transcript_usage(
    transcript_path: Path, model: str | None = None, price: dict | None = None
) -> dict[str, Any]:
    """Token + cost totals from the agent's stream-json transcript.

    Tokens come from the final `result` event's aggregated `usage` (fallback:
    summed per-turn `message.usage`). COST is computed from those tokens via a
    manual per-million-token `price` (from the treatment yaml) when given, else
    via litellm (`usd_cost`) so it is uniform across claude/codex/mistral
    engines; only if neither can price the model do we fall back to the
    transcript's own `total_cost_usd` (Claude Code reports it; codex/mistral
    report 0, i.e. cost stays 0).
    """
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "llm_calls": 0,
    }
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

    # Cost via litellm (uniform across engines); keep the transcript's own
    # total_cost_usd only when litellm can't price the model.
    # Discount cached input ONLY when input_tokens is cache-inclusive (codex
    # reports it that way). Claude's input_tokens excludes cache, so its
    # cache_read (≫ input_tokens) fails the guard and is dropped — leaving
    # Claude's pricing unchanged.
    cache_read = 0
    if final_result:
        cr = int((final_result.get("usage") or {}).get("cache_read_input_tokens") or 0)
        if cr <= totals["input_tokens"]:
            cache_read = cr
    cost = usd_cost(model, totals["input_tokens"], totals["output_tokens"], cache_read, price)
    if cost is not None:
        totals["cost_usd"] = cost
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


def _command_uses_sdk(
    tokens: list[str], command: str, sdk_module: str, run_dir: Path | None
) -> bool:
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


def _exec_segments(tokens: list[str], depth: int = 0) -> list[list[str]]:
    """Token lists of each command SEGMENT in `tokens`, starting at the
    segment's executable token (so `seg[0]` is what runs).

    Tracks segment boundaries (`;`, `&&`, `||`, `|`, `&`, etc.), skips env-var
    assignment prefixes (`FOO=bar python script.py` → python is the executable),
    and for a `bash`/`sh`/`zsh -c "<script>"` segment recursively scans the quoted
    inner script (so `bash -c "python train.py"` is classified as a python call).
    """
    out: list[list[str]] = []
    cur: list[str] | None = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _SEGMENT_SEPARATORS:
            cur = None
            i += 1
            continue
        if cur is not None:
            cur.append(tok)
            i += 1
            continue
        if _is_env_var_assignment(tok):
            i += 1
            continue  # still expecting the executable
        # bash/sh -c "..." → recurse into the quoted script body.
        if (
            depth < 2
            and tok in ("bash", "sh", "zsh")
            and i + 2 < len(tokens)
            and tokens[i + 1] == "-c"
        ):
            inner = tokens[i + 2]
            try:
                inner_tokens = shlex.split(inner)
            except ValueError:
                inner_tokens = inner.split()
            out.extend(_exec_segments(inner_tokens, depth + 1))
            i += 3
            cur = []  # consume the rest of this segment as already handled
            continue
        cur = [tok]
        out.append(cur)
        i += 1
    return [seg for seg in out if seg]


def _executable_tokens(tokens: list[str], depth: int = 0) -> list[str]:
    """The executable token of each command segment (see `_exec_segments`)."""
    return [seg[0] for seg in _exec_segments(tokens, depth)]


_TOOL_BUCKETS = {
    "Read": "read",
    "Write": "write",
    "Edit": "edit",
    "MultiEdit": "edit",
    "NotebookEdit": "edit",
    "Glob": "glob",
    "Grep": "grep",
    "TodoWrite": "todo",
}


def _classify_tool_use(
    tool_name: str,
    tool_input: dict[str, Any],
    cli_binary: str | None = None,
    sdk_module: str | None = None,
    run_dir: Path | None = None,
    cli_subcommand: str | None = None,
    cli_aux: list[str] | None = None,
) -> str:
    if tool_name == "Skill":
        return "skill"
    if tool_name.startswith("mcp__"):
        return "mcp"
    if tool_name != "Bash":
        # Explicit workspace-tool buckets — no catch-all "other". Tools outside
        # this map (denied ones like WebFetch/Task) are not bucketed.
        return _TOOL_BUCKETS.get(tool_name, "")
    command = (tool_input.get("command") or "").strip()
    if not command:
        return "bash"
    # Agents often write multi-line bash where the python call isn't the first
    # token (env-var assignments, or cd/setup lines before it). `shlex.split`
    # strips newlines, so split by line first; within a line, segment by
    # `;`/`&&`/`||`/`|`.
    exec_segs: list[list[str]] = []
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
        exec_segs.extend(_exec_segments(line_tokens))
    exec_tokens = [seg[0] for seg in exec_segs]
    if not tokens:
        return "bash"
    if not exec_tokens:
        return "bash"

    # Priority: cli (interface usage) > python (counts even when nested in bash) >
    # bash (pure shell utilities).
    def _is_cli(tok: str) -> bool:
        return bool(cli_binary) and (tok == cli_binary or tok.endswith(f"/{cli_binary}"))

    def _cli_present() -> bool:
        if not cli_binary:
            return False
        # `cli_subcommand` is a comma-joined entrypoint allowlist (e.g.
        # `sagemaker,sagemaker-runtime,s3`); empty → any use of the binary counts.
        subs = [s for s in (p.strip() for p in (cli_subcommand or "").split(",")) if s]
        if not subs:
            return any(_is_cli(t) for t in exec_tokens)
        # Subcommand entrypoint: the allowed service must follow the binary in
        # an EXEC-position segment (`echo aws sagemaker …` doesn't count),
        # skipping global options (`aws --region us-east-1 sagemaker …`) via
        # the hook's shared parser — one rule for enforcement and counting.
        return any(
            _is_cli(seg[0]) and _hook_cli_arg_after(seg, cli_binary) in subs for seg in exec_segs
        )

    def _aux_present() -> bool:
        # Extra on-interface binaries (e.g. `bq` alongside `gcloud`) count as cli.
        aux = [b for b in (cli_aux or []) if b]
        return any(t == b or t.endswith(f"/{b}") for t in exec_tokens for b in aux)

    if _cli_present() or _aux_present():
        return "cli"
    if any(_is_python_first(t) for t in exec_tokens):
        if sdk_module and _command_uses_sdk(tokens, command, sdk_module, run_dir):
            return "sdk"
        return "python"
    return "bash"


def _parse_endpoint_patterns(
    patterns: list[str] | None,
) -> list[tuple[str, str, "re.Pattern[str]"]]:
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
        rows.append(
            (
                str(e.get("method", "")).upper(),
                str(e.get("path", "")),
                str(e.get("src", "")),
            )
        )
    return rows


def endpoint_coverage(
    api_log: Path | str,
    whitelist: list[str] | None,
    blacklist: list[str] | None,
    interface: str | None = None,
) -> dict[str, Any]:
    """Full REST endpoint-coverage breakdown from the per-run API log (written by
    the venv shim). Drives BOTH the `whitelist_hits`/`blacklist_hits` metrics AND
    the per-run `endpoint_coverage.json` report showing WHICH lifecycle steps
    the agent reached.

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
    wl_rows = [(m, p) for m, p, so in rows if interface is None or so == interface]

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


# Hook denial marker in errored tool results — FALLBACK only (used when no
# hook-written commands.jsonl is available). Line-anchored (optional "- "
# bullet from Claude Code's blocked-by-hook formatting) so echoed remote
# errors like `PERMISSION_DENIED: …` mid-line don't count as hook denials.
_DENIED_MARKER_RE = re.compile(r"(?m)^(?:- )?DENIED:")


def _denied_from_commands_log(commands_log: Path | None) -> int | None:
    """Count of structured denial records in the hook-written commands.jsonl,
    or None when the log is absent/empty (fall back to the transcript scan)."""
    if commands_log is None or not commands_log.exists():
        return None
    denied = 0
    seen_any = False
    for line in commands_log.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        seen_any = True
        if rec.get("denied"):
            denied += 1
    return denied if seen_any else None


# `sleep N[smhd]` in a shell command. Bare number = seconds; GNU suffixes scale.
# Word boundary + required whitespace avoids matching `sleepy`/`asleep`; finditer
# catches each occurrence so chained `sleep 5; cmd; sleep 10` counts twice.
_SLEEP_RE = re.compile(r"\bsleep\s+(\d+(?:\.\d+)?)\s*([smhd])?\b", re.IGNORECASE)
_SLEEP_UNIT_S = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _sleep_stats(command: str) -> tuple[int, float]:
    """(count, total seconds) of `sleep` invocations in a shell command."""
    n = 0
    secs = 0.0
    for m in _SLEEP_RE.finditer(command or ""):
        n += 1
        secs += float(m.group(1)) * _SLEEP_UNIT_S.get((m.group(2) or "").lower(), 1)
    return n, secs


def aggregate_commands(
    transcript_path: Path,
    cli_binary: str | None = None,
    sdk_module: str | None = None,
    run_dir: Path | None = None,
    cli_subcommand: str | None = None,
    commands_log: Path | None = None,
    cli_aux: list[str] | None = None,
) -> dict[str, float]:
    """Count tool calls by category from the stream-json transcript.

    Values are ints except `sleep_time_s` (seconds, float).

    Tool calls are `type=="tool_use"` blocks in each `assistant` event's
    `message.content`. `cli_binary` promotes matching Bash calls to cli_calls;
    `sdk_module` (typically the platform name when interface=sdk) promotes python
    invocations that import or `-m`-run that module to sdk_calls; a run script file
    is also peeked (relative to `run_dir`) for SDK use.

    `commands_log` (the hook-written commands.jsonl) is the preferred source for
    `denied_calls`: the hook logs every denial structurally, so the count can't
    be skewed by remote error text echoed into tool results. Without it, errored
    tool results are split by the line-anchored `DENIED:` marker.
    """
    counts = {
        "cli_calls": 0,
        "mcp_calls": 0,
        "sdk_calls": 0,
        "python_calls": 0,
        "bash_calls": 0,
        "skill_calls": 0,
        "read_calls": 0,
        "write_calls": 0,
        "edit_calls": 0,
        "glob_calls": 0,
        "grep_calls": 0,
        "todo_calls": 0,
        "failed_commands": 0,
        "denied_calls": 0,
        "sleep_calls": 0,
        "sleep_time_s": 0.0,
    }
    bucket = {
        "cli": "cli_calls",
        "mcp": "mcp_calls",
        "sdk": "sdk_calls",
        "python": "python_calls",
        "bash": "bash_calls",
        "skill": "skill_calls",
        "read": "read_calls",
        "write": "write_calls",
        "edit": "edit_calls",
        "glob": "glob_calls",
        "grep": "grep_calls",
        "todo": "todo_calls",
    }
    if not transcript_path.exists():
        return counts
    if run_dir is None:
        run_dir = transcript_path.parent
    denied_from_log = _denied_from_commands_log(commands_log)
    total_errors = 0
    # Sleep is paired tool_use→result by id: a sleep in a DENIED command never
    # ran (the hook blocks it before execution → 0 wall seconds), so it must not
    # inflate sleep_time_s. A merely FAILED command (bad args, non-zero exit)
    # DID sleep first, so it still counts. Bash tool_uses carry an `id`; results
    # reference it via `tool_use_id`. Blocks without an id (e.g. codex-normalized
    # events) can't be paired and are counted directly as a safe fallback.
    sleep_by_id: dict[str, tuple[int, float]] = {}
    denied_ids: set[str] = set()
    for line in transcript_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        if etype == "user":
            # Outcome friction: errored tool results. A hook rejection carries
            # the enforcement `DENIED:` marker → denied_calls; every other
            # error (non-zero exit, exception, bad args) → failed_commands.
            for block in (event.get("message") or {}).get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                if not block.get("is_error"):
                    continue
                total_errors += 1
                text = _flatten_text(block.get("content"))
                if _DENIED_MARKER_RE.search(text):
                    counts["denied_calls"] += 1
                    tid = block.get("tool_use_id")
                    if tid:
                        denied_ids.add(tid)
                else:
                    counts["failed_commands"] += 1
            continue
        if etype != "assistant":
            continue
        for block in (event.get("message") or {}).get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            # Scan every Bash command for `sleep`, independent of its bucket: a
            # sleep chained onto an interface command classifies as cli/sdk, not
            # bash. Defer accounting by tool_use id so a later DENIED result can
            # exclude it; unpaired (no-id) blocks count immediately.
            if block.get("name") == "Bash":
                sc, st = _sleep_stats((block.get("input") or {}).get("command", ""))
                if sc:
                    tid = block.get("id")
                    if tid:
                        prev = sleep_by_id.get(tid, (0, 0.0))
                        sleep_by_id[tid] = (prev[0] + sc, prev[1] + st)
                    else:
                        counts["sleep_calls"] += sc
                        counts["sleep_time_s"] += st
            category = _classify_tool_use(
                block.get("name", ""),
                block.get("input") or {},
                cli_binary=cli_binary,
                sdk_module=sdk_module,
                run_dir=run_dir,
                cli_subcommand=cli_subcommand,
                cli_aux=cli_aux,
            )
            col = bucket.get(category)
            if col:
                counts[col] += 1
    if denied_from_log is not None:
        # Structured hook records win: every denial (exit 2) produces exactly
        # one errored tool_result, so the remaining errors are real failures.
        counts["denied_calls"] = denied_from_log
        counts["failed_commands"] = max(0, total_errors - denied_from_log)
    # Fold in paired sleeps, skipping commands whose result was DENIED (blocked
    # before running, so no wall time elapsed).
    for tid, (sc, st) in sleep_by_id.items():
        if tid not in denied_ids:
            counts["sleep_calls"] += sc
            counts["sleep_time_s"] += st
    counts["sleep_time_s"] = round(counts["sleep_time_s"], 2)
    return counts


# Tool-call categories that execute against the PLATFORM (remote): the
# interface under test. Everything else a tool call does is local execution.
_REMOTE_CATEGORIES = ("cli", "mcp", "sdk")


def platform_tool_time(
    spans: list[dict[str, Any]],
    cli_binary: str | None = None,
    sdk_module: str | None = None,
    run_dir: Path | None = None,
    cli_subcommand: str | None = None,
    cli_aux: list[str] | None = None,
) -> float:
    """Seconds spent inside tool calls THROUGH the interface under test.

    `spans` are `claude_runner.ToolTimer` records ({tool_name, tool_input,
    seconds}). Each span is classified with the SAME `_classify_tool_use` that
    drives the call counters, under the same active-interface kwargs — so a
    `hops …` Bash call is platform time only when cli is the interface under
    test, exactly as it is `cli_calls` only then. The caller derives
    `local_time_s = wall_time_s − platform_time_s` (local tool execution +
    LLM generation), so wall always splits exactly into platform + local.
    """
    platform = 0.0
    for span in spans:
        category = _classify_tool_use(
            span.get("tool_name", ""),
            span.get("tool_input") or {},
            cli_binary=cli_binary,
            sdk_module=sdk_module,
            run_dir=run_dir,
            cli_subcommand=cli_subcommand,
            cli_aux=cli_aux,
        )
        if category in _REMOTE_CATEGORIES:
            platform += span.get("seconds") or 0.0
    return round(platform, 2)


def write_commands_log(
    transcript_path: Path,
    commands_log: Path,
    cli_binary: str | None = None,
    sdk_module: str | None = None,
    run_dir: Path | None = None,
    cli_subcommand: str | None = None,
    cli_aux: list[str] | None = None,
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
                cli_subcommand=cli_subcommand,
                cli_aux=cli_aux,
            )
            record = {
                "timestamp": timestamp,
                "session_id": session_id,
                "tool_name": tool_name,
                "category": category,
                "tool_input": tool_input,
            }
            lines.append(json.dumps(record, default=str))
    # The transcript only shows tool calls; denials live solely in the hook's
    # own records (`denied: true`). Carry them over so the rebuild doesn't
    # erase the structured denial count `aggregate_commands` relies on.
    if commands_log.exists():
        for line in commands_log.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("denied"):
                lines.append(json.dumps(rec, default=str))
    commands_log.write_text("\n".join(lines) + ("\n" if lines else ""))


# Substrings marking a client-side failure worth flagging at the top of the
# collected log (the agent otherwise only sees "No such tool available").
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
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def _mcp_server_log(caches_dir: Path, run_dir: Path, server: str) -> list[str]:
    """Lines Claude buried for one stdio MCP server during THIS run, oldest first.

    Each `mcp-logs-<server>/*.jsonl` line is either a dict (debug/error events with
    a timestamp) or a bare `"Server stderr: ..."` string carrying the server's own
    stdout/stderr — including a startup traceback that kills the process before any
    tool registers. Both are surfaced.
    """
    out: list[str] = []
    # Cache dir is keyed on the agent's cwd (the run dir); fall back to a
    # glob so a layout change doesn't silently drop the logs.
    candidates = [caches_dir / _slug_for(run_dir) / f"mcp-logs-{server}"]
    candidates += [p for p in caches_dir.glob(f"*/mcp-logs-{server}") if p not in candidates]
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
    """Errored / traceback-bearing tool results from the agent transcript.

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
    A server that crashes at startup leaves the agent with only "No such tool
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
        sections.append("===== client errors (agent transcript) =====\n" + "\n---\n".join(crashes))
        body_for_scan.extend(crashes)

    blob = "\n".join(body_for_scan)
    markers = [m for m in _CRASH_MARKERS if m in blob]
    crashed = bool(markers)

    head = (
        f"# {platform} client log ({interface} interface)\n"
        f"# crashed={crashed}" + (f"  markers={markers}" if markers else "") + "\n\n"
    )
    out_path.write_text(
        head + ("\n\n".join(sections) if sections else "(no client output captured)\n")
    )
    return {"path": str(out_path), "crashed": crashed, "markers": markers}


def append(results_csv: Path, row: Row, fields: list[str] | None = None) -> None:
    """Write `row` to results.csv. A previous row with the same `run_dir` (same
    combo re-run) is replaced rather than appended, so each combo has at most one
    row at any time.

    `fields` narrows the column set; `None` (default) writes `RESULTS_FIELDS`.
    For `RESULTS_FIELDS` the Row is renamed and trimmed via `_results_view`.
    """
    cols = fields if fields is not None else RESULTS_FIELDS
    # Detect the results view by column-list equality (callers import
    # RESULTS_FIELDS, so either `is` or `==` works; `==` is robust across
    # module re-imports).
    use_results_view = cols == RESULTS_FIELDS
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    raw = asdict(row)
    new_row = _results_view(raw) if use_results_view else raw
    with _locked_csv(results_csv):
        kept: list[dict[str, Any]] = []
        replaced: dict[str, Any] | None = None
        if results_csv.exists():
            with results_csv.open() as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("run_dir") == new_row.get("run_dir"):
                        replaced = r  # replaced by new_row below
                        continue
                    kept.append(r)
        if use_results_view:
            # `n` = repeat counter: 1 + how many KEPT rows already ran this exact
            # config (same platform/interface/skills/category/task). A re-run of
            # the same run_dir replaces its row and KEEPS its number — counting
            # kept rows instead would collide with a later repeat's n whenever
            # the replaced row wasn't the latest one.
            ident = _bench_identity(new_row)
            old_n = (
                (replaced or {}).get("n")
                if replaced is not None and _bench_identity(replaced) == ident
                else None
            )
            if old_n and str(old_n).isdigit():
                new_row["n"] = int(old_n)
            else:
                new_row["n"] = 1 + sum(1 for r in kept if _bench_identity(r) == ident)
        _write_csv_atomic(results_csv, cols, kept + [new_row])
