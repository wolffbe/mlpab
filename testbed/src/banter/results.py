"""results.csv writer.

One row per (challenge, interface, mode, skills, auth) run. Token/cost columns
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_PYTHON_PREFIXES = ("python", "python3", "uv run python", "uv run", "pip", "pip3")
_PY_INTERPRETER_BASENAMES = ("python", "python3", "pip", "pip3")


FIELDS = [
    "started_at",
    "challenge_id",
    "interface",
    "mode",
    "skills",
    "skills_version",
    "skills_hash",
    "skills_dir",
    "version",
    "hash",
    "interface_dir",
    "prompt_version",
    "prompt_hash",
    "prompt_file",
    "auth",
    "model",
    "wall_time_s",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cost_usd",
    "llm_calls",
    "cli_calls",
    "mcp_calls",
    "sdk_calls",
    "python_calls",
    "bash_calls",
    "other_tool_calls",
    "medal",
    "score",
    "run_dir",
]


@dataclass
class Row:
    started_at: str
    challenge_id: str
    interface: str          # interface NAME, e.g. "hopsworks" or "none"
    mode: str               # interface TYPE: cli/mcp/sdk/none
    skills: str             # skill bundle name or "none"
    skills_version: int     # 0 when skills=none; else version of the snapshot used
    skills_hash: str        # "" when skills=none; else 8-hex hash of the snapshot
    skills_dir: str         # configs/skills/<name>/<version>/, "" when skills=none
    version: int            # interface variant index (0=base, autoresearch creates 1+)
    hash: str               # 8-hex hash of manifest + binary folder + version
    interface_dir: str      # configs/interfaces/<name>/<mode>.yaml (manifest path)
    prompt_version: int     # same integer as `version`; prompts are per-version
    prompt_hash: str        # 8-hex hash of the resolved prompt text
    prompt_file: str        # configs/interfaces/<name>/<mode>.yaml#versions.<v>.prompt
    auth: str
    model: str
    wall_time_s: float
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
    other_tool_calls: int = 0
    medal: str | None = None
    score: float | None = None
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


def _classify_tool_use(
    tool_name: str,
    tool_input: dict[str, Any],
    cli_binary: str | None = None,
    sdk_module: str | None = None,
    run_dir: Path | None = None,
) -> str:
    if tool_name.startswith("mcp__"):
        return "mcp"
    if tool_name != "Bash":
        return "other"
    command = (tool_input.get("command") or "").strip()
    if not command:
        return "bash"
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    if not tokens:
        return "bash"
    first = tokens[0]
    if cli_binary and (first == cli_binary or first.endswith(f"/{cli_binary}")):
        return "cli"
    if _is_python_first(first):
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
    to cli_calls; `sdk_module` (typically the interface name when mode=sdk)
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
        "other_tool_calls": 0,
    }
    bucket = {
        "cli": "cli_calls",
        "mcp": "mcp_calls",
        "sdk": "sdk_calls",
        "python": "python_calls",
        "bash": "bash_calls",
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


def _truncate(text: str, limit: int = 2000) -> str:
    """Shorten long tool I/O blobs so the readable log stays scannable."""
    if len(text) <= limit:
        return text
    head = text[: limit - 200]
    return f"{head}\n... <truncated {len(text) - (limit - 200)} chars>"


def _block_text(block: dict[str, Any]) -> str:
    """Flatten a content block to a readable line or short paragraph."""
    btype = block.get("type")
    if btype == "text":
        return block.get("text") or ""
    if btype == "thinking":
        return f"(thinking) {block.get('thinking') or ''}"
    if btype == "tool_use":
        name = block.get("name", "?")
        try:
            args = json.dumps(block.get("input") or {}, ensure_ascii=False)
        except (TypeError, ValueError):
            args = str(block.get("input"))
        return f"TOOL_USE {name}({_truncate(args)})"
    if btype == "tool_result":
        content = block.get("content")
        if isinstance(content, list):
            parts = []
            for sub in content:
                if isinstance(sub, dict) and sub.get("type") == "text":
                    parts.append(sub.get("text") or "")
                else:
                    parts.append(json.dumps(sub, ensure_ascii=False))
            content_text = "\n".join(parts)
        elif isinstance(content, str):
            content_text = content
        else:
            content_text = json.dumps(content, ensure_ascii=False)
        prefix = "TOOL_ERROR" if block.get("is_error") else "TOOL_RESULT"
        return f"{prefix}\n{_truncate(content_text)}"
    return f"{btype}: {json.dumps(block, ensure_ascii=False)}"


def write_readable_transcript(transcript_path: Path, out_path: Path) -> Path:
    """Render the stream-json transcript as plain text alongside it.

    One section per event, headed by `[timestamp] ROLE`, body is the
    flattened content (text, tool calls, tool results). Long tool blobs
    are truncated so the file remains scannable.
    """
    if not transcript_path.exists():
        out_path.write_text("(no transcript)\n")
        return out_path

    sections: list[str] = []
    for line in transcript_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")
        timestamp = event.get("timestamp") or ""
        prefix = f"[{timestamp}] " if timestamp else ""
        if etype == "system":
            sections.append(f"{prefix}SYSTEM {event.get('subtype', '')}")
            continue
        if etype == "result":
            cost = event.get("total_cost_usd")
            turns = event.get("num_turns")
            stop = event.get("stop_reason") or event.get("subtype") or ""
            text = event.get("result") or ""
            sections.append(
                f"{prefix}RESULT stop={stop} turns={turns} cost=${cost}"
                + (f"\n{text}" if text else "")
            )
            continue

        role = (event.get("message") or {}).get("role") or etype or "?"
        blocks = (event.get("message") or {}).get("content") or []
        if isinstance(blocks, str):
            body = blocks
        else:
            body = "\n\n".join(_block_text(b) for b in blocks if isinstance(b, dict))
        sections.append(f"{prefix}{role.upper()}\n{body}")

    out_path.write_text("\n\n" + ("\n\n---\n\n".join(sections)) + "\n")
    return out_path


def append(results_csv: Path, row: Row) -> None:
    """Write `row` to results.csv. If a previous row has the same `run_dir`
    (i.e. the same combo was re-run), it's replaced rather than appended, so
    each combo has at most one row at any time. Also migrates the header in
    place when FIELDS changes between releases."""
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    new_row = asdict(row)
    kept: list[dict[str, Any]] = []
    if results_csv.exists():
        with results_csv.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("run_dir") == new_row["run_dir"]:
                    continue  # replaced by new_row below
                kept.append(r)
    with results_csv.open("w", newline="") as fw:
        writer = csv.DictWriter(fw, fieldnames=FIELDS)
        writer.writeheader()
        for r in kept:
            writer.writerow({k: r.get(k, "") for k in FIELDS})
        writer.writerow(new_row)
