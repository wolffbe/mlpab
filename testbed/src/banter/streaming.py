"""Live terminal display + saved logs of a `claude -p` stream-json transcript.

Both the engineer (`claude_runner.run`) and the researcher (`autoresearch`)
emit stream-json on stdout. `run_with_retry`'s `on_line` callback forwards each
raw line here so activity shows up as it happens instead of the run looking
frozen while everything lands silently in `transcript.jsonl`.

Two persistent artifacts per run, both written regardless of terminal output:
  * `transcript.jsonl` — raw stream-json (written by `run_with_retry`).
  * `stream.log`       — live one-line-per-event human-readable view that this
                         module mirrors to a file in the same folder.

Terminal streaming is on by default; `BANTER_QUIET=1` (or `--quiet`) silences
the terminal but still writes `stream.log`.

Nesting (autoresearch): the researcher launches each engineer run as a
`banter run --challenge` subprocess whose stdout it captures. Printing engineer
lines to that stdout would bloat the researcher's context, so under
`BANTER_NESTED=1` the engineer writes only its `stream.log`; the parent
autoresearch process tails those files (`FileTailer`) to show them live.
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
    """True when terminal streaming is suppressed (BANTER_QUIET truthy)."""
    return os.environ.get("BANTER_QUIET", "").strip().lower() not in ("", "0", "false", "no")


def nested() -> bool:
    """True when running under autoresearch (engineer stdout is captured)."""
    return os.environ.get("BANTER_NESTED", "").strip().lower() not in ("", "0", "false", "no")


def _wrap_block(text: str, prefix: str) -> list[str]:
    """Tag every line of `text` with `prefix`. No truncation — the live log
    shows the full payload verbatim. (transcript.jsonl still has the raw
    stream-json events for tooling that wants the structured form.)"""
    raw = text.splitlines() or [""]
    return [f"{prefix} {line}" for line in raw]


def assistant_lines(event: dict[str, Any], label: str) -> list[str]:
    """Readable lines for an `assistant` stream-json event.

    Assistant text is printed verbatim (one terminal line per source line);
    tool calls expand to the full input (bash command body, thinking text)
    with per-event line/width caps so we surface what's happening without
    drowning the terminal in large file reads. Each emitted line carries the
    same `[label:kind]` prefix for easy grep.
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
            # Surface reasoning so the log updates between tool calls (keeps the
            # live view moving instead of going silent while the model thinks).
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

    These are the outputs coming BACK from each tool call; we now surface
    multiple lines (capped) instead of only the first, so file reads, bash
    output, and command stdout are actually inspectable in the live log.
    The verbatim payload always remains in `transcript.jsonl`.
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
    """An `on_line` callback that renders each event to stdout.

    Always prints to stdout (fd 1); it does NOT decide visibility or write a
    file. During an engineer run, `tee_to` captures fd 1/2 into the run's
    `stream.log` and decides whether to echo to the real terminal (passthrough),
    so this printer just needs to emit the formatted line.
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

    FD-level (not just `sys.stdout`), so it captures the COMPLETE terminal output
    of the run — Python prints AND subprocess output (mle-bench data prep, pip,
    the engineer's streamed lines, tracebacks). `passthrough` also echoes it to
    the real terminal; set it False under autoresearch nesting (where the
    engineer subprocess's stdout is captured by the researcher) so the researcher
    context isn't bloated. fds are always restored, even on exception.
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
        os.close(write_fd)        # EOF → pump drains and exits
        pump.join(timeout=2)
        for fd in (saved_out, saved_err):
            try:
                os.close(fd)
            except OSError:
                pass
        log.close()


class FileTailer(threading.Thread):
    """Background thread that prints lines appended to `root/<glob>` files.

    Used by autoresearch to surface nested engineer `stream.log` files live in
    the user's terminal. Daemon thread; call `stop()` then `join()` to drain the
    final lines. Truncated/recreated files (a combo re-run) reset to offset 0.
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
            if size < off:           # file was truncated/recreated → re-read
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
