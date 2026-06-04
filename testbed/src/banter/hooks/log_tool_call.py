#!/usr/bin/env python3
"""Claude Code PreToolUse hook. Reads JSON from stdin, classifies the tool
call as cli / mcp / sdk / python / bash / other, appends one JSONL line to
TESTBED_COMMAND_LOG, and blocks calls that try to escape the boundary or
run unobservable background work.

Exit 0 → allow. Non-zero → block (the tool call is rejected by claude-code
and an error is shown to the model).

Blocked:
  - Tool-input flags `dangerouslyDisableSandbox: true`, `run_in_background: true`.
  - `Read`/`Write`/`Edit` paths outside the boundary (env `TESTBED_BOUNDARY`),
    or pointing at known-sensitive $HOME subdirs. This is the only way to
    enforce Read/Write/Edit path denies under `bypassPermissions` — Claude
    Code's `permissions.deny` patterns for those tools are silently skipped
    in bypass mode, but PreToolUse hooks fire regardless of mode.
"""
from __future__ import annotations

import json
import os
import pwd
import re
import shlex
import sys
import time

PYTHON_PREFIXES = ("python", "python3", "uv run python", "uv run", "pip", "pip3")


def classify(tool_name: str, tool_input: dict) -> str:
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
    cli_binary = os.environ.get("TESTBED_CLI_BINARY") or None
    if cli_binary and (first == cli_binary or first.endswith(f"/{cli_binary}")):
        return "cli"
    is_python = (
        first in PYTHON_PREFIXES
        or first.endswith(("/python", "/python3", "/pip", "/pip3"))
        or first.endswith(".py")
    )
    if is_python:
        sdk_module = os.environ.get("TESTBED_SDK_MODULE") or None
        if sdk_module:
            m = re.escape(sdk_module)
            if re.search(rf"\b(?:import|from)\s+{m}\b|-m\s+{m}\b", command):
                return "sdk"
        return "python"
    return "bash"


# --- Remote-only / single-interface enforcement -------------------------------
# Everything must run on the platform; local model training is forbidden and the
# engineer must stay on the interface under test. These helpers parse a Bash
# command into segments, find each segment's executable, and extract the python
# payload (inline `-c`, referenced `.py` files, `-m module`) so we can tell
# LOCAL execution of a compute library apart from a script merely written to be
# shipped to the cluster.

def _is_env_assign(tok: str) -> bool:
    if "=" not in tok or tok.startswith("="):
        return False
    head = tok.split("=", 1)[0]
    return bool(head) and (head[0].isalpha() or head[0] == "_") and all(
        c.isalnum() or c == "_" for c in head
    )


def _split(text: str) -> list:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _segments(command: str) -> list:
    """Split a command into command segments at shell separators (`;`, `&&`,
    `||`, `|`, `&`, newline), returning each as a token list.

    Splits the RAW string char-by-char respecting quotes — `shlex.split` does
    NOT treat `;`/`&&`/`|` as separators (it keeps `foo;` and `true;python` as
    single tokens), so a token-membership test misses every separator that isn't
    space-delimited (`cd x; python y`, `true&&python y`). Quote-awareness keeps
    an inline `python -c "import os; os.system(...)"` body as ONE segment (the
    `;` is inside quotes), so payload extraction stays intact.
    """
    raw_segs: list[str] = []
    cur: list[str] = []
    quote: str | None = None
    i, n = 0, len(command)
    while i < n:
        c = command[i]
        if quote:
            cur.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            cur.append(c)
            i += 1
            continue
        if c in "&|" and i + 1 < n and command[i + 1] == c:   # && or ||
            raw_segs.append("".join(cur)); cur = []; i += 2; continue
        if c in ";\n&|":                                       # ; & | newline
            raw_segs.append("".join(cur)); cur = []; i += 1; continue
        cur.append(c)
        i += 1
    raw_segs.append("".join(cur))
    out: list = []
    for seg in raw_segs:
        toks = _split(seg)
        if toks:
            out.append(toks)
    return out


def _seg_exec(tokens: list, depth: int = 0) -> str | None:
    """The executable token of a segment, skipping env-assignment prefixes,
    `env [-i] [VAR=VAL]...` and `uv run` wrappers, and recursing into
    `bash -c "<script>"`."""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if _is_env_assign(tok):
            i += 1
            continue
        if depth < 2 and tok in ("bash", "sh", "zsh") and i + 2 < len(tokens) and tokens[i + 1] == "-c":
            return _seg_exec(_split(tokens[i + 2]), depth + 1)
        # `env [-i|--unset=..] [VAR=VAL]... cmd ...` → the real command follows.
        if tok in ("env", "/usr/bin/env") and depth < 3:
            j = i + 1
            while j < len(tokens) and (_is_env_assign(tokens[j]) or tokens[j].startswith("-")):
                j += 1
            i = j
            continue
        # `uv run [-flags] python ...` → skip the `uv run` wrapper.
        if tok == "uv" and i + 1 < len(tokens) and tokens[i + 1] == "run" and depth < 3:
            j = i + 2
            while j < len(tokens) and tokens[j].startswith("-"):
                j += 1
            i = j
            continue
        return tok
    return None


def _interp_basename(tok: str) -> str:
    return tok.rsplit("/", 1)[-1]


# A python interpreter basename: python, python3, python3.11, python2.7, …
_PYTHON_INTERP_RE = re.compile(r"^python(\d+(\.\d+)*)?$")


def _is_python_tok(tok: str) -> bool:
    base = _interp_basename(tok)
    return (
        tok in PYTHON_PREFIXES
        or bool(_PYTHON_INTERP_RE.match(base))
        or base in ("pip", "pip3")
        or tok.endswith(".py")
    )


def _executes_local_python(command: str) -> bool:
    """True if any segment runs local python — interpreter, `.py` script, OR
    `pip` (versioned interpreters and absolute paths included). All of it is
    off-interface in CLI/MCP mode: the engineer never needs to install, because
    banter (autoresearch) builds + installs the interface AND its dependencies
    into the run venv before the engineer starts."""
    for seg in _segments(command):
        ex = _seg_exec(seg)
        if ex and _is_python_tok(ex):
            return True
    return False


def _is_cli_tok(tok: str, cli_binary: str) -> bool:
    return tok == cli_binary or tok.endswith(f"/{cli_binary}")


def _python_payloads(command: str) -> tuple[str, list]:
    """What python actually EXECUTES locally in this command.

    Returns `(payload_text, unreadable_scripts)`:
      - `payload_text` — concatenated inline `-c` bodies, `-m module` (synthetic
        import), and the contents of the executed `.py` SCRIPT (the first `.py`
        token after the interpreter; trailing `.py` ARGS are not treated as
        executed code). Empty when no python runs locally — a script merely
        created with `Write` is never flagged. `cd` is tracked so
        `cd work && python train.py` resolves train.py.
      - `unreadable_scripts` — executed `.py` scripts we could NOT read (so their
        imports can't be verified). The caller fails CLOSED on these in SDK mode.
    """
    texts: list = []
    unreadable: list = []
    curdir = os.getcwd()
    for seg in _segments(command):
        ex = _seg_exec(seg)
        if ex == "cd" and len(seg) > 1:
            target = seg[1]
            curdir = target if os.path.isabs(target) else os.path.normpath(os.path.join(curdir, target))
            continue
        if not (ex and _is_python_tok(ex)):
            continue
        if "-c" in seg:
            j = seg.index("-c")
            if j + 1 < len(seg):
                texts.append(seg[j + 1])
            continue
        if "-m" in seg:
            j = seg.index("-m")
            if j + 1 < len(seg):
                texts.append("import " + seg[j + 1])
            continue
        # `python script.py [args]` — the script is the FIRST .py token; later
        # .py tokens are arguments (e.g. `--config c.py`), not executed code.
        script = next((t for t in seg if t.endswith(".py")), None)
        if script is None:
            continue
        path = script if os.path.isabs(script) else os.path.join(curdir, script)
        try:
            with open(path, errors="ignore") as f:
                texts.append(f.read())
        except OSError:
            unreadable.append(script)
    return "\n".join(texts), unreadable


def _imported_modules(text: str) -> set:
    """Root module names imported by `text`. Handles `import a, b.c as d`,
    `from x.y import z`, and statements separated by `;` or newlines — so
    comma-lists like `import hopsworks, sklearn` are fully covered."""
    mods: set = set()
    for stmt in re.split(r"[;\n]", text):
        s = stmt.strip()
        if s.startswith("import "):
            for part in s[len("import "):].split(","):
                head = part.strip().split()
                if head:
                    root = head[0].split(".")[0]
                    if root:
                        mods.add(root)
        elif s.startswith("from "):
            after = s[len("from "):].split(" import ")[0].strip()
            root = after.split(".")[0]
            if root:
                mods.add(root)
    # Dynamic imports with a string-literal module name:
    # `__import__("torch")`, `importlib.import_module('sklearn')`.
    for m in re.finditer(r"""(?:__import__|import_module)\(\s*['"]([\w.]+)""", text):
        mods.add(m.group(1).split(".")[0])
    return mods


def _imports_any(text: str, modules: list) -> str | None:
    """Return the first module in `modules` that `text` imports, else None."""
    imported = _imported_modules(text)
    for mod in modules:
        if mod in imported:
            return mod
    return None


# What is the ONLY sanctioned way to do work in each mode (for messages).
_ONLY = {
    "cli": "the `hops` CLI",
    "mcp": "the MCP tools (`mcp__hopsworks__*`)",
    "sdk": "the `hopsworks` Python SDK (no ML libraries)",
}


def enforce(tool_name: str, tool_input: dict) -> str | None:
    """Return a denial reason if this call escapes the remote-only / single-
    interface contract, else None. Reads the active interface + markers from env
    (set by claude_runner): TESTBED_INTERFACE, TESTBED_SDK_MODULE,
    TESTBED_CLI_BINARY, TESTBED_COMPUTE_DENY.

    Contract (everything runs on Hopsworks; nothing runs locally):
      - CLI mode → only `hops`; NO local python at all.
      - MCP mode → only MCP tools; NO local python, NO `hops`.
      - SDK mode → only `hopsworks` python; NO ML libraries, NO `hops`, NO MCP.
      - ML libraries are blocked locally in EVERY mode.
    Basic bash (cp/mkdir/ls/cat …) stays allowed so the floor submission can be
    written and data inspected.
    """
    interface = os.environ.get("TESTBED_INTERFACE") or None
    # Enforce ONLY for the real delegation interfaces. A none/none baseline
    # (interface "none" or unset) trains locally by design — never blocked.
    if interface not in ("cli", "mcp", "sdk"):
        return None
    cli_binary = os.environ.get("TESTBED_CLI_BINARY") or None
    compute_deny = [m for m in (os.environ.get("TESTBED_COMPUTE_DENY") or "").split(",") if m]
    use_only = _ONLY.get(interface, f"the {interface} interface")

    # MCP tools are only legitimate in MCP mode.
    if tool_name.startswith("mcp__"):
        if interface != "mcp":
            return (
                f"DENIED: MCP tools are off-interface in {interface!r} mode. "
                f"Use only {use_only} for Hopsworks operations."
            )
        return None
    if tool_name != "Bash":
        return None

    command = (tool_input.get("command") or "").strip()
    if not command:
        return None

    # 1) ML libraries may never be executed locally — training stays remote.
    payload, unreadable = _python_payloads(command)
    if payload and compute_deny:
        lib = _imports_any(payload, compute_deny)
        if lib:
            return (
                f"DENIED: local execution of ML library {lib!r} is forbidden. "
                f"Training must run on Hopsworks — push it to the cluster as a "
                f"remote Job via {use_only}. If that's not possible, STOP and "
                f"report the missing capability; do NOT train locally."
            )

    # 1b) SDK mode allows python, so we must READ an executed script to confirm
    #     it imports no ML library. If we can't read it, fail CLOSED — otherwise
    #     `python train.py` with an unresolvable path would skip the check. (In
    #     CLI/MCP mode rule 2 blocks all local python regardless, so this only
    #     matters for SDK.)
    if interface == "sdk" and unreadable:
        return (
            f"DENIED: cannot read local script(s) {', '.join(unreadable)} to verify "
            f"they don't train locally. Run python from your working directory so "
            f"the script is readable, drive the SDK inline (`python -c ...`), or "
            f"push the work to Hopsworks as a remote Job."
        )

    # 2) In CLI / MCP modes, ANY local python is off-interface (only the CLI /
    #    the MCP tools may do work). SDK mode IS python, so it's allowed there
    #    (restricted to no-ML by rule 1).
    if interface in ("cli", "mcp") and _executes_local_python(command):
        return (
            f"DENIED: local python is off-interface in {interface!r} mode. "
            f"Use only {use_only} to run work on Hopsworks. Basic shell "
            f"(cp/mkdir) is fine for writing the floor submission."
        )

    # 3) The `hops` CLI is off-interface unless the CLI is under test.
    if cli_binary and interface != "cli":
        for seg in _segments(command):
            ex = _seg_exec(seg)
            if ex and _is_cli_tok(ex, cli_binary):
                return (
                    f"DENIED: the {cli_binary!r} CLI is off-interface in "
                    f"{interface!r} mode. Use only {use_only}."
                )
    return None


# Tool-input flags that escape our intended boundaries. Blocking via hook
# (the only mechanism available — settings.json can't deny by parameter).
FORBIDDEN_FLAGS = {
    "dangerouslyDisableSandbox": (
        "DENIED: dangerouslyDisableSandbox is forbidden. The sandbox stays "
        "ON. If a write to your cwd is failing with EPERM, check `pwd`, "
        "create parents with `os.makedirs(..., exist_ok=True)`, and keep "
        "paths relative — DO NOT try to bypass the sandbox."
    ),
    "run_in_background": (
        "DENIED: run_in_background is forbidden. Run commands synchronously; "
        "chain with `&&` instead of detaching."
    ),
}


# Tools we intercept for path-based deny. Their tool_input contains
# `file_path` (absolute, resolved by Claude before submitting).
_PATH_TOOLS = ("Read", "Write", "Edit", "NotebookEdit", "MultiEdit")


def _real_home() -> str:
    """User's REAL home from /etc/passwd, ignoring any $HOME redirect."""
    return pwd.getpwuid(os.getuid()).pw_dir


def _path_violates_boundary(file_path: str) -> str | None:
    """Return a rejection reason if `file_path` is outside the boundary or
    points at a sensitive $HOME subdir; None if allowed."""
    if not file_path:
        return None
    p = os.path.realpath(file_path) if os.path.isabs(file_path) else os.path.realpath(
        os.path.join(os.getcwd(), file_path)
    )
    # Boundary check: TESTBED_BOUNDARY is the engineer's challenge dir.
    boundary = os.environ.get("TESTBED_BOUNDARY")
    if boundary:
        b = os.path.realpath(boundary)
        if not (p == b or p.startswith(b + "/")):
            home = _real_home()
            # Allowed escapes — both are Claude Code's OWN infrastructure, not
            # user data:
            #   1. ~/.claude (Claude's config + tokens we already trust).
            #   2. Background-task output files. When Claude auto-moves a long
            #      foreground command to a background task, it writes the output
            #      to <tmp>/claude-*/<cwd-slug>/<session>/tasks/<id>.output —
            #      outside the boundary. Denying the read leaves the agent
            #      blind to its own command. The path embeds the cwd slug, so
            #      this allowance stays scoped to THIS run's own tasks.
            cwd_slug = os.getcwd().replace(os.sep, "-")
            is_task_output = (
                "/tasks/" in p and p.endswith(".output") and cwd_slug in p
            )
            if not p.startswith(home + "/.claude/") and not is_task_output:
                return f"path is outside the engineer boundary ({b})"
    # Always block dotfiles + dot-dirs at $HOME root (secrets), regardless of
    # whether a boundary is set. Catches ~/.ssh, ~/.aws, ~/.gnupg, .kaggle,
    # .gitconfig, .netrc, .zshrc — every secret-bearing path on macOS.
    home = _real_home()
    rel = p[len(home) + 1:] if p.startswith(home + "/") else None
    if rel and rel.startswith("."):
        # ~/.claude is allowed (Claude's config + tokens we already trust).
        if rel == ".claude" or rel.startswith(".claude/"):
            return None
        return f"path targets a sensitive $HOME location ({home}/{rel.split('/')[0]})"
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}

    # Block forbidden tool-input flags. Stderr is shown to the model so it
    # learns why the call was rejected and (hopefully) stops trying.
    for flag, msg in FORBIDDEN_FLAGS.items():
        if tool_input.get(flag) is True:
            print(msg, file=sys.stderr, flush=True)
            return 2  # non-zero → block the tool call

    # Remote-only / single-interface enforcement: block local compute and any
    # escape to an interface other than the one under test.
    reason = enforce(tool_name, tool_input)
    if reason:
        print(reason, file=sys.stderr, flush=True)
        return 2

    # Path-based deny for Read/Write/Edit. Enforces what `permissions.deny`
    # silently skips under bypassPermissions.
    if tool_name in _PATH_TOOLS:
        # NotebookEdit uses `notebook_path`; everything else uses `file_path`.
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        reason = _path_violates_boundary(path)
        if reason:
            print(f"DENIED: {tool_name}({path!r}) — {reason}. Stay inside your "
                  f"working directory (or ~/.claude); rewrite the call to a "
                  f"path under cwd.", file=sys.stderr, flush=True)
            return 2

    # Log the (allowed) call.
    log_path = os.environ.get("TESTBED_COMMAND_LOG")
    if log_path:
        record = {
            "timestamp": time.time(),
            "session_id": payload.get("session_id"),
            "tool_name": tool_name,
            "category": classify(tool_name, tool_input),
            "tool_input": tool_input,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
