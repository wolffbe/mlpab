"""Live terminal display + saved logs of a `claude -p` stream-json transcript.

The agent (`claude_runner.run`) emits stream-json on stdout; `run_with_retry`'s
`on_line` callback forwards each line here so activity shows live instead of
the run looking frozen. Per run:
  * `transcript.jsonl` — raw stream-json; transient (mined for usage + commands,
                         then discarded at teardown).
  * `agent.log`        — one-line-per-event human-readable view mirrored to file;
                         the KEPT artifact.

Terminal streaming is on by default; `MLPAB_QUIET=1`/`--quiet` silences the
terminal but still writes `agent.log`.

Nesting: an outer controller may capture each `mlpab run` subprocess's
stdout; printing agent lines there would bloat its context, so under
`MLPAB_NESTED=1` the agent writes only `agent.log` and the parent tails
those files (`FileTailer`) to show them live.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Iterator


def quiet() -> bool:
    """True when terminal streaming is suppressed (MLPAB_QUIET truthy)."""
    return os.environ.get("MLPAB_QUIET", "").strip().lower() not in ("", "0", "false", "no")


def nested() -> bool:
    """True when running nested (agent stdout captured by an outer controller)."""
    return os.environ.get("MLPAB_NESTED", "").strip().lower() not in ("", "0", "false", "no")


# Bound a tool RESULT body before LIVE echo so a chatty command (a big codex
# `aggregated_output` / `cat bigfile` / job-log tail) can't flood the pane and
# agent.log. The full body still exists in the engine's own event log
# (vibe_events.jsonl / codex_events.jsonl). The Claude path is unaffected — it
# already surfaces only what the model chose to return.
LIVE_RESULT_MAX_LINES = 200
LIVE_RESULT_MAX_CHARS = 20000


def cap_for_live(text: str) -> str:
    """Truncate a result body to LIVE_RESULT_MAX_CHARS / _LINES with a marker."""
    text = text or ""
    if len(text) > LIVE_RESULT_MAX_CHARS:
        text = text[:LIVE_RESULT_MAX_CHARS] + "\n… (truncated; see the engine event log)"
    lines = text.splitlines()
    if len(lines) > LIVE_RESULT_MAX_LINES:
        extra = len(lines) - LIVE_RESULT_MAX_LINES
        lines = lines[:LIVE_RESULT_MAX_LINES] + [f"… ({extra} more lines, see the event log)"]
    return "\n".join(lines)


def _wrap_block(text: str, prefix: str) -> list[str]:
    """Tag every line of `text` with `prefix`. No truncation — the live log is
    the durable record (raw transcript.jsonl is discarded at teardown)."""
    raw = text.splitlines() or [""]
    return [f"{prefix} {line}" for line in raw]


def assistant_lines(event: dict[str, Any], label: str) -> list[str]:
    """Readable lines for an `assistant` stream-json event.

    Text printed verbatim (one line per source line); tool calls expand to their
    input (bash body, thinking). Each line carries a `[label:kind]` prefix for grep.
    """
    lines: list[str] = []
    content = (event.get("message") or {}).get("content") or []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = (block.get("text") or "").rstrip()
            for line in text.splitlines():
                if line.strip():
                    lines.append(f"[{label}] {line}")
        elif btype == "thinking":
            # Surface reasoning so the live view keeps moving between tool calls.
            thought = (block.get("thinking") or "").strip()
            if thought:
                lines.extend(_wrap_block(thought, f"[{label}:thinking]"))
        elif btype == "tool_use":
            name = block.get("name", "?")
            inp = block.get("input") or {}
            if name == "Bash":
                cmd = (inp.get("command") or "").rstrip()
                lines.extend(_wrap_block(cmd, f"[{label}:bash]"))
            elif name in ("Read", "Write", "Edit"):
                lines.append(f"[{label}:{name.lower()}] {inp.get('file_path', '?')}")
            elif name.startswith("mcp__"):
                lines.append(f"[{label}:mcp] {name}")
            else:
                lines.append(f"[{label}:tool] {name}")
    return lines


def tool_result_lines(event: dict[str, Any], label: str) -> list[str]:
    """Readable lines for tool_result blocks in a `user` stream-json event.

    The outputs coming BACK from each tool call (file reads, bash/command stdout),
    surfaced in full so they're inspectable in the durable live log.
    """
    lines: list[str] = []
    content = (event.get("message") or {}).get("content") or []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        c = block.get("content")
        if isinstance(c, list):
            text = "\n".join(
                b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = c if isinstance(c, str) else ""
        tag = "result-err" if block.get("is_error") else "result"
        lines.extend(_wrap_block(text.rstrip(), f"[{label}:{tag}]"))
    return lines


def result_line(event: dict[str, Any], label: str) -> str:
    """A brief one-line summary for the final `result` event."""
    stop = event.get("stop_reason") or event.get("subtype") or "?"
    turns = event.get("num_turns", "?")
    cost = event.get("total_cost_usd")
    cost_str = f", ${cost:.4f}" if isinstance(cost, (int, float)) else ""
    return f"[{label}] done: {turns} turns{cost_str}, stop={stop}"


def emit(line: str, log_path: Path | None = None) -> None:
    """Print `line` (unless quiet) and append it to `log_path` (always)."""
    if not quiet():
        print(line, flush=True)
    if log_path is not None:
        with open(log_path, "a") as f:
            f.write(line + "\n")


def make_printer(label: str) -> Callable[[str], None]:
    """An `on_line` callback that renders each event to stdout (fd 1).

    Does NOT decide visibility or write a file: during an agent run `tee_to`
    captures fd 1/2 into `agent.log` and handles terminal echo (passthrough),
    so this just emits the formatted line.
    """

    def _on_line(raw_line: str) -> None:
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            return
        etype = event.get("type")
        if etype == "assistant":
            for line in assistant_lines(event, label):
                print(line, flush=True)
        elif etype == "user":
            for line in tool_result_lines(event, label):
                print(line, flush=True)
        elif etype == "result":
            print(result_line(event, label), flush=True)

    return _on_line


@contextlib.contextmanager
def tee_to(log_path: Path, passthrough: bool = True) -> Iterator[None]:
    """Mirror everything written to fd 1 and fd 2 into `log_path` for the block.

    FD-level (not just `sys.stdout`), so it captures ALL terminal output — Python
    prints AND subprocess output (pip, streamed agent lines, tracebacks).
    `passthrough` also echoes to the real terminal; set False under nesting
    (an outer controller captures the agent's stdout) to avoid bloating its
    context. fds are always restored, even on exception.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    log = open(log_path, "ab")
    saved_out, saved_err = os.dup(1), os.dup(2)
    read_fd, write_fd = os.pipe()

    def _pump() -> None:
        with os.fdopen(read_fd, "rb", buffering=0) as r:
            for chunk in iter(lambda: r.read(65536), b""):
                try:
                    log.write(chunk)
                    log.flush()
                except (OSError, ValueError):
                    pass
                if passthrough:
                    try:
                        os.write(saved_out, chunk)
                    except OSError:
                        pass

    pump = threading.Thread(target=_pump, daemon=True)
    pump.start()
    try:
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(write_fd)  # EOF → pump drains and exits
        pump.join(timeout=2)
        for fd in (saved_out, saved_err):
            try:
                os.close(fd)
            except OSError:
                pass
        log.close()


class FileTailer(threading.Thread):
    """Background thread that prints lines appended to `root/<glob>` files.

    Used by an outer controller to surface nested `agent.log` files live.
    Daemon; call `stop()` then `join()` to drain. Truncated/recreated files (a
    combo re-run) reset to offset 0.
    """

    def __init__(self, root: Path, glob: str, exclude: tuple[Path, ...] = (), poll_s: float = 0.5):
        super().__init__(daemon=True)
        self.root = root
        self.glob = glob
        self.exclude = {p.resolve() for p in exclude}
        self.poll_s = poll_s
        self._stop_event = threading.Event()
        self._offsets: dict[Path, int] = {}

    def stop(self) -> None:
        self._stop_event.set()

    def poll_once(self) -> None:
        for p in sorted(self.root.glob(self.glob)):
            if p.resolve() in self.exclude:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            off = self._offsets.get(p, 0)
            if size < off:  # file was truncated/recreated → re-read
                off = 0
            if size <= off:
                self._offsets[p] = size
                continue
            try:
                with p.open() as f:
                    f.seek(off)
                    chunk = f.read()
                    self._offsets[p] = f.tell()
            except OSError:
                continue
            for line in chunk.splitlines():
                print(line, flush=True)

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.poll_once()
            self._stop_event.wait(self.poll_s)
        self.poll_once()  # final drain after stop
