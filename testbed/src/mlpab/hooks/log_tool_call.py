#!/usr/bin/env python3
"""Claude Code PreToolUse hook. Reads stdin JSON, classifies the call as
cli / mcp / sdk / python / bash / other, appends one JSONL line to
TESTBED_COMMAND_LOG, and blocks calls that escape the boundary or run
unobservable background work.

Exit 0 → allow. Non-zero → block (claude-code rejects the call, shows the
error to the model).

Blocked:
  - Tool-input flags `dangerouslyDisableSandbox: true`, `run_in_background: true`.
  - `Read`/`Write`/`Edit` paths outside the boundary (`TESTBED_BOUNDARY`) or at
    sensitive $HOME subdirs. Only way to enforce these denies under
    `bypassPermissions`: `permissions.deny` patterns are silently skipped in
    bypass mode, but PreToolUse hooks fire in every mode.
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


def _env_subcommands() -> list:
    """TESTBED_CLI_SUBCOMMAND is a comma-joined allowlist of subcommand
    entrypoints (e.g. `sagemaker,sagemaker-runtime,s3`); empty → no scoping."""
    raw = os.environ.get("TESTBED_CLI_SUBCOMMAND") or ""
    return [s for s in (p.strip() for p in raw.split(",")) if s]


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
    cli_subcommands = _env_subcommands()
    if cli_binary and (first == cli_binary or first.endswith(f"/{cli_binary}")):
        # Subcommand entrypoint (e.g. `aws {sagemaker,s3}`): only the configured
        # services count as the CLI interface; `aws ec2 …` falls through to bash.
        # `_cli_arg_after` skips global options (`aws --region x sagemaker …`).
        if not cli_subcommands or _cli_arg_after(tokens, cli_binary) in cli_subcommands:
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
# Everything runs on the platform; local training is forbidden and the agent
# must stay on the interface under test. These helpers split a Bash command into
# segments, find each segment's executable, and extract the python payload
# (inline `-c`, `.py` files, `-m module`) to distinguish LOCAL execution of a
# compute library from a script merely written to be shipped to the cluster.


def _is_env_assign(tok: str) -> bool:
    if "=" not in tok or tok.startswith("="):
        return False
    head = tok.split("=", 1)[0]
    return (
        bool(head)
        and (head[0].isalpha() or head[0] == "_")
        and all(c.isalnum() or c == "_" for c in head)
    )


def _split(text: str) -> list:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _segments(command: str) -> list:
    """Token lists for each shell segment (separators dropped).

    Thin wrapper over `_pipeline_segments` — same quote/redirect-aware split,
    minus the trailing operator. Keeping one parser means quote/redirect bugs
    are fixed in a single place, and `_segments` inherits the redirect-`&`
    awareness (`2>&1` stays glued instead of splitting into a bogus segment).
    """
    return [toks for toks, _sep in _pipeline_segments(command)]


def _seg_exec(tokens: list, depth: int = 0) -> str | None:
    """Segment's executable token: skips env-assignment prefixes,
    `env [-i] [VAR=VAL]...` and `uv run` wrappers; recurses into
    `bash -c "<script>"`."""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if _is_env_assign(tok):
            i += 1
            continue
        if (
            depth < 2
            and tok in ("bash", "sh", "zsh")
            and i + 2 < len(tokens)
            and tokens[i + 1] == "-c"
        ):
            return _seg_exec(_split(tokens[i + 2]), depth + 1)
        # `env [-i|--unset=..] [VAR=VAL]... cmd ...` → real command follows.
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


# Shell keywords / builtins / control-flow — not external commands, so never
# gated by the allowlist (skipping them keeps compound commands like
# `for f in *; do cp "$f" out/; done` from tripping it).
_SHELL_KEYWORDS = {
    "if",
    "then",
    "elif",
    "else",
    "fi",
    "for",
    "while",
    "until",
    "do",
    "done",
    "case",
    "esac",
    "in",
    "select",
    "function",
    "time",
    "{",
    "}",
    "[[",
    "]]",
    "!",
    ":",
    ".",
    "true",
    "false",
    "test",
    "[",
    "]",
    "cd",
    "pwd",
    "echo",
    "printf",
    "export",
    "set",
    "unset",
    "local",
    "read",
    "source",
    "command",
    "which",
    "type",
    "exit",
    "return",
    "pushd",
    "popd",
    "wait",
    "shift",
    "getopts",
    "let",
    "declare",
    "alias",
    "umask",
}

# Basic POSIX file/text utilities permitted in EVERY interface mode: they move
# or inspect bytes (cp/mkdir to write the floor submission, cat/head/grep to
# inspect data) and do NO off-interface platform work. Anything NOT here, NOT a
# shell keyword, and NOT the mode's own interface (python in SDK / CLI binary in
# CLI) is denied by `enforce`. Deliberately EXCLUDES general-purpose interpreters
# (node/ruby/perl/php/…) and network tools (curl/wget) — the escape hatches.
_BASIC_SHELL = {
    "ls",
    "cat",
    "head",
    "tail",
    "tac",
    "nl",
    "wc",
    "find",
    "stat",
    "file",
    "du",
    "df",
    "tree",
    "realpath",
    "readlink",
    "basename",
    "dirname",
    "cp",
    "mv",
    "mkdir",
    "rmdir",
    "rm",
    "ln",
    "touch",
    "chmod",
    "grep",
    "egrep",
    "fgrep",
    "sed",
    "awk",
    "cut",
    "sort",
    "uniq",
    "tr",
    "diff",
    "cmp",
    "comm",
    "tee",
    "fold",
    "column",
    "rev",
    "paste",
    "join",
    "xxd",
    "od",
    "hexdump",
    "split",
    "expand",
    "unexpand",
    "date",
    "gzip",
    "gunzip",
    "zcat",
    "bzip2",
    "bunzip2",
    "xz",
    "unxz",
    "tar",
    "unzip",
    "zip",
    "shasum",
    "md5sum",
    "sha256sum",
    "cksum",
}


def _is_cli_tok(tok: str, cli_binary: str) -> bool:
    return tok == cli_binary or tok.endswith(f"/{cli_binary}")


# Global options that take NO value, so the token after them is the service,
# not the option's argument (`aws --debug sagemaker …`). Options outside this
# set (and not `--opt=value`) are assumed to consume the next token
# (`aws --region us-east-1 sagemaker …`).
_NO_VALUE_OPTS = {
    "--debug",
    "--version",
    "--help",
    "--no-verify-ssl",
    "--no-paginate",
    "--no-sign-request",
    "--no-cli-pager",
    "--no-cli-auto-prompt",
}


def _cli_arg_after(tokens: list, cli_binary: str) -> str | None:
    """The CLI service/subcommand token following `cli_binary` in `tokens`
    (e.g. `sagemaker` in `aws sagemaker list-models`), skipping global options
    that may precede it (`aws --region us-east-1 sagemaker …`). None if the
    binary isn't present or no service token follows.

    Option-value ambiguity resolves AGAINST the caller (fail closed): an
    unknown valueless option swallows the following token, so the service is
    missed and the command denied — never the reverse (`aws --profile
    sagemaker ec2 …` must not read the option's value as the service).
    """
    for i, tok in enumerate(tokens):
        if not _is_cli_tok(tok, cli_binary):
            continue
        j = i + 1
        while j < len(tokens):
            t = tokens[j]
            if not t.startswith("-"):
                return t
            if "=" in t or t in _NO_VALUE_OPTS or t.startswith("--no-"):
                j += 1  # option carries no separate value token
            else:
                j += 2  # `--opt value` — skip both
        return None
    return None


def _python_payloads(command: str) -> tuple[str, list]:
    """What python actually EXECUTES locally in this command.

    Returns `(payload_text, unreadable_scripts)`:
      - `payload_text` — concatenated inline `-c` bodies, `-m module` (synthetic
        import), and contents of the executed `.py` SCRIPT (first `.py` token
        after the interpreter; trailing `.py` ARGS are not executed code). Empty
        when no python runs locally — a script merely created with `Write` is
        never flagged. `cd` is tracked so `cd work && python train.py` resolves
        train.py.
      - `unreadable_scripts` — executed `.py` scripts we could NOT read (imports
        unverifiable). Caller fails CLOSED on these in SDK mode.
    """
    texts: list = []
    unreadable: list = []
    curdir = os.getcwd()
    for seg in _segments(command):
        ex = _seg_exec(seg)
        if ex == "cd" and len(seg) > 1:
            target = seg[1]
            curdir = (
                target if os.path.isabs(target) else os.path.normpath(os.path.join(curdir, target))
            )
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
        # `python script.py [args]` — script is the FIRST .py token; later .py
        # tokens are args (e.g. `--config c.py`), not executed code.
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
    `from x.y import z`, and `;`/newline-separated statements — so comma-lists
    like `import hopsworks, sklearn` are fully covered."""
    mods: set = set()
    for stmt in re.split(r"[;\n]", text):
        s = stmt.strip()
        if s.startswith("import "):
            for part in s[len("import ") :].split(","):
                head = part.strip().split()
                if head:
                    root = head[0].split(".")[0]
                    if root:
                        mods.add(root)
        elif s.startswith("from "):
            after = s[len("from ") :].split(" import ")[0].strip()
            root = after.split(".")[0]
            if root:
                mods.add(root)
    # Dynamic imports with string-literal module name:
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


def _use_only(interface: str) -> str:
    """The ONLY sanctioned way to do work in this mode (for messages). Derived
    from the active platform's env (set by claude_runner) — the hook serves
    every platform, so nothing platform-specific is hardcoded here."""
    if interface == "cli":
        binary = os.environ.get("TESTBED_CLI_BINARY") or None
        subs = _env_subcommands()
        if not binary:
            return "the platform CLI"
        if len(subs) == 1:
            return f"the `{binary} {subs[0]}` CLI"
        if subs:
            return f"the `{binary}` CLI ({'/'.join(subs)} services only)"
        return f"the `{binary}` CLI"
    if interface == "mcp":
        platform = os.environ.get("TESTBED_PLATFORM") or None
        return f"the MCP tools (`mcp__{platform}__*`)" if platform else "the platform's MCP tools"
    if interface == "sdk":
        module = os.environ.get("TESTBED_SDK_MODULE") or None
        return (
            f"the `{module}` Python SDK (no ML libraries)"
            if module
            else "the platform Python SDK (no ML libraries)"
        )
    return f"the {interface} interface"


def enforce(tool_name: str, tool_input: dict, segments: list | None = None) -> str | None:
    """Return a denial reason if this call escapes the remote-only / single-
    interface contract, else None. Reads active interface + markers from env (set
    by claude_runner): TESTBED_INTERFACE, TESTBED_SDK_MODULE, TESTBED_CLI_BINARY,
    TESTBED_COMPUTE_DENY, TESTBED_PLATFORM.

    Contract: everything runs on the platform, nothing locally. Bash execs are a
    FAIL-CLOSED allowlist — only the mode's own interface plus basic shell;
    node/ruby/perl/curl/any other binary denied by default:
      - CLI mode → only the platform CLI; NO local python, NO other interpreters.
      - MCP mode → only MCP tools; NO local python/interpreters, NO platform CLI.
      - SDK mode → only the platform SDK's python; NO ML libraries, NO other
        interpreters, NO platform CLI, NO MCP.
      - ML libraries blocked locally in EVERY mode.
    Basic shell (cp/mkdir/ls/cat/grep …) stays allowed to write the floor
    submission and inspect data.

    `segments` may carry the pre-parsed `_pipeline_segments(command)` so the
    caller can reuse one parse across checks; None → parse here.
    """
    interface = os.environ.get("TESTBED_INTERFACE") or None
    # Enforce ONLY for real delegation interfaces. A none/none baseline
    # ("none" or unset) trains locally by design — never blocked.
    if interface not in ("cli", "mcp", "sdk"):
        return None
    cli_binary = os.environ.get("TESTBED_CLI_BINARY") or None
    # Optional subcommand entrypoint allowlist: when set (e.g.
    # `sagemaker,sagemaker-runtime,s3`), only `<cli_binary> <one of these> …` is
    # on-interface — other services of the same binary (`aws ec2 …`) are
    # off-interface escapes.
    cli_subcommands = _env_subcommands()
    compute_deny = [m for m in (os.environ.get("TESTBED_COMPUTE_DENY") or "").split(",") if m]
    platform = os.environ.get("TESTBED_PLATFORM") or "the remote platform"
    use_only = _use_only(interface)

    # MCP tools legitimate only in MCP mode.
    if tool_name.startswith("mcp__"):
        if interface != "mcp":
            return (
                f"DENIED: MCP tools are off-interface in {interface!r} mode. "
                f"Use only {use_only} for {platform} operations."
            )
        return None
    if tool_name != "Bash":
        return None

    command = (tool_input.get("command") or "").strip()
    if not command:
        return None

    # 1) ML libraries never run locally — training stays remote.
    payload, unreadable = _python_payloads(command)
    if payload and compute_deny:
        lib = _imports_any(payload, compute_deny)
        if lib:
            return (
                f"DENIED: local execution of ML library {lib!r} is forbidden. "
                f"Training must run on {platform} — push it there as a "
                f"remote job via {use_only}. If that's not possible, STOP and "
                f"report the missing capability; do NOT train locally."
            )

    # 1b) SDK mode allows python, so we must READ an executed script to confirm
    #     it imports no ML library. Unreadable → fail CLOSED, else
    #     `python train.py` with an unresolvable path skips the check. (CLI/MCP
    #     rule 2 blocks all local python anyway, so this only matters for SDK.)
    if interface == "sdk" and unreadable:
        return (
            f"DENIED: cannot read local script(s) {', '.join(unreadable)} to verify "
            f"they don't train locally. Run python from your working directory so "
            f"the script is readable, drive the SDK inline (`python -c ...`), or "
            f"push the work to {platform} as a remote job."
        )

    # 2) Fail-closed allowlist for Bash execs. The ONLY sanctioned compute is the
    #    interface itself, so rather than chase a denylist of escape hatches
    #    (node/ruby/perl/deno/curl/…) we ALLOW a fixed set and deny the rest. Per
    #    segment, resolve the real executable (through env / `uv run` / `bash -c`
    #    wrappers) and permit it only if it is:
    #      - a shell keyword/builtin or basic file/text utility (cp/mkdir/cat/
    #        head/grep/… — floor submission + data inspection),
    #      - python, in SDK mode (rule 1 already barred ML libraries), or
    #      - the CLI binary, in CLI mode.
    #    Else — node/ruby/curl/stray binary, python in cli/mcp, or CLI outside cli
    #    — is off-interface and denied. Redirect-aware segmentation
    #    (`_pipeline_segments`) so `… 2>&1` / `> out` don't misparse.
    if segments is None:
        segments = _pipeline_segments(command)
    for seg, _sep in segments:
        ex = _seg_exec(seg)
        if not ex:
            continue
        base = _interp_basename(ex)
        if base in _SHELL_KEYWORDS or base in _BASIC_SHELL:
            continue
        if _is_python_tok(ex):
            if interface == "sdk":
                continue
            return (
                f"DENIED: local python is off-interface in {interface!r} mode. "
                f"Use only {use_only} to run work on {platform}. Basic shell "
                f"(cp/mkdir/cat) is fine for writing the floor submission."
            )
        if cli_binary and _is_cli_tok(ex, cli_binary):
            if interface == "cli":
                # Subcommand entrypoint: only `<cli_binary> <allowed service> …`
                # is on-interface; another service of the same binary is an escape.
                if cli_subcommands:
                    sub = _cli_arg_after(seg, cli_binary)
                    if sub not in cli_subcommands:
                        seen = f"{cli_binary} {sub}" if sub else cli_binary
                        allowed = "/".join(cli_subcommands)
                        return (
                            f"DENIED: only `{cli_binary} {{{allowed}}}` is the "
                            f"interface under test; `{seen}` is off-interface. Use "
                            f"only {use_only} — other {cli_binary!r} services are "
                            f"blocked."
                        )
                continue
            return (
                f"DENIED: the {cli_binary!r} CLI is off-interface in "
                f"{interface!r} mode. Use only {use_only}."
            )
        return (
            f"DENIED: {base!r} is off-interface in {interface!r} mode — only "
            f"{use_only} may do work on {platform}; local interpreters, network "
            f"tools, and stray binaries are blocked. Basic shell "
            f"(cp/mkdir/cat/head/grep) is allowed for the floor submission and "
            f"data inspection. If you can't do the work through {use_only}, STOP "
            f"and report the missing capability — do NOT work around it locally."
        )
    return None


# --- Free-tier instance-type allowlist -----------------------------------------
# SageMaker-style instance types (`ml.<family>.<size>`) appear LITERALLY in every
# job-creating call, whatever the interface: CLI args (`--resource-config
# InstanceType=…` / `--cli-input-json`), MCP tool arguments, SDK python code, and
# job-spec files written via Write/Edit before being referenced. When
# TESTBED_INSTANCE_ALLOW is set (comma-separated, from the interface manifest's
# `instance_allowlist`), any OTHER `ml.*.*` token in a tool call is denied — the
# hard guarantee that agent runs only request AWS Free Tier instance types.

_INSTANCE_TYPE_RE = re.compile(
    r"\bml\.[a-z][a-z0-9-]*\.(?:nano|micro|small|medium|large|\d*xlarge)\b"
)


def enforce_instance_types(tool_name: str, tool_input: dict) -> str | None:
    """Return a denial reason if the call names an instance type outside
    TESTBED_INSTANCE_ALLOW, else None. Unset/empty allowlist → no-op (platforms
    without instance types never set it)."""
    allow = {
        t.strip() for t in (os.environ.get("TESTBED_INSTANCE_ALLOW") or "").split(",") if t.strip()
    }
    if not allow:
        return None
    if tool_name == "Bash":
        command = tool_input.get("command") or ""
        # Include what python EXECUTES (inline -c bodies + executed .py scripts)
        # so `python job.py` is checked against the script's contents too.
        payload, _ = _python_payloads(command)
        text = f"{command}\n{payload}"
    else:
        # MCP tool args + Write/Edit/NotebookEdit content: a job spec written to
        # a .json/.py file and only referenced later is caught at write time.
        text = json.dumps(tool_input, default=str)
    denied = sorted({m for m in _INSTANCE_TYPE_RE.findall(text) if m not in allow})
    if denied:
        return (
            f"DENIED: instance type(s) {', '.join(denied)} are outside the AWS "
            f"Free Tier. Use ONLY {', '.join(sorted(allow))} — single instance, "
            f"short jobs. Serverless Inference (no instance type) is also "
            f"Free Tier. Do not retry with a bigger or different instance type."
        )
    return None


# --- `mlpab run` must be FOREGROUND, never piped or backgrounded ---------------
# A nested `mlpab run` (spawned from inside another agent session) must stay
# synchronous. Two fatal ways a caller tries to "background" it:
#   - Piping (`mlpab run … | head`/`| tail`): downstream exits after N lines and
#     SIGPIPE-kills `mlpab run` mid-build, but the Bash tool returns 0 (head's
#     exit), so the caller reads truncated build output, thinks the run
#     started, and polls a dead run forever via `sleep`/`tail`.
#   - Backgrounding (trailing `&`): detaches the same way.
# Neither is caught by `run_in_background` (a tool flag, not shell syntax), so the
# hook parses the command and blocks both. The caller is told to run it
# synchronously and redirect to a file to cap output.


def _pipeline_segments(command: str) -> list:
    """Split `command` into segments, pairing each segment's tokens with the
    shell operator that FOLLOWS it: '|', '&&', '||', ';', '&', or '' (last).

    Quote-aware (mirrors `_segments`), but — unlike `_segments` — does NOT treat
    a redirect `&` (`2>&1`, `&>file`, `>&2`) as a separator: an `&` glued to a
    `>` on either side is part of a redirect, not an operator.
    """
    raw: list = []  # (text, sep_after)
    cur: list = []
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
        if c == "\\":  # backslash escape
            nxt = command[i + 1] if i + 1 < n else ""
            if nxt == "\n":  # line continuation → fold to whitespace
                cur.append(" ")
                i += 2
                continue
            if nxt:  # escaped char: literal, never a separator
                cur.append(c)
                cur.append(nxt)
                i += 2
                continue
        if c in "&|" and i + 1 < n and command[i + 1] == c:  # && or ||
            raw.append(("".join(cur), c + c))
            cur = []
            i += 2
            continue
        if c == "|":
            raw.append(("".join(cur), "|"))
            cur = []
            i += 1
            continue
        if c in ";\n":
            raw.append(("".join(cur), ";"))
            cur = []
            i += 1
            continue
        if c == "&":
            prev = command[i - 1] if i > 0 else ""
            nxt = command[i + 1] if i + 1 < n else ""
            if prev == ">" or nxt == ">":  # redirect (2>&1, &>file) — not sep
                cur.append(c)
                i += 1
                continue
            raw.append(("".join(cur), "&"))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    raw.append(("".join(cur), ""))
    out: list = []
    for text, sep in raw:
        toks = _split(text)
        if toks:
            out.append((toks, sep))
    return out


def _is_mlpab_run(tokens: list) -> bool:
    """True if tokens are a `mlpab run …` invocation (the `mlpab` executable —
    bare or path — immediately followed by the `run` subcommand), skipping
    leading `VAR=val` env-assignment prefixes."""
    i = 0
    while i < len(tokens) and _is_env_assign(tokens[i]):
        i += 1
    if i >= len(tokens):
        return False
    if tokens[i].rsplit("/", 1)[-1] != "mlpab":
        return False
    return i + 1 < len(tokens) and tokens[i + 1] == "run"


def _mlpab_run_misuse(command: str, segments: list | None = None) -> str | None:
    """Return a denial reason if a `mlpab run` invocation is piped or
    backgrounded, else None. See the module note above for why. `segments` may
    carry a pre-parsed `_pipeline_segments(command)` to reuse one parse; None →
    parse here."""
    for toks, sep in segments if segments is not None else _pipeline_segments(command):
        if _is_mlpab_run(toks) and sep in ("|", "&"):
            how = "piped into another command" if sep == "|" else "backgrounded with `&`"
            return (
                f"DENIED: `mlpab run` must run in the FOREGROUND; it may not be "
                f"{how}. Piping it (e.g. `| head` / `| tail`) makes the downstream "
                f"process exit early and SIGPIPE-kill the run mid-build, and "
                f"backgrounding detaches it — either way the call returns while the "
                f"run is dead and you'll poll a frozen file forever. Run it "
                f"synchronously; to cap output redirect to a file "
                f"(`mlpab run … > run.log 2>&1`) and read the run's agent.log "
                f"afterward."
            )
    return None


# Tool-input flags that escape our boundaries. Blocked via hook — the only
# mechanism, since settings.json can't deny by parameter.
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


# Tools intercepted for path-based deny. tool_input carries `file_path`
# (absolute, resolved by Claude before submitting).
_PATH_TOOLS = ("Read", "Write", "Edit", "NotebookEdit", "MultiEdit")


def _real_home() -> str:
    """User's REAL home from /etc/passwd, ignoring $HOME redirect."""
    return pwd.getpwuid(os.getuid()).pw_dir


def _path_violates_boundary(file_path: str) -> str | None:
    """Return a rejection reason if `file_path` is outside the boundary or
    points at a sensitive $HOME subdir; None if allowed."""
    if not file_path:
        return None
    p = (
        os.path.realpath(file_path)
        if os.path.isabs(file_path)
        else os.path.realpath(os.path.join(os.getcwd(), file_path))
    )
    # Boundary check: TESTBED_BOUNDARY is the agent's run dir.
    boundary = os.environ.get("TESTBED_BOUNDARY")
    if boundary:
        b = os.path.realpath(boundary)
        # The run's solution/ folder holds the eval's ANSWER KEY: inside the
        # boundary (so the grader and humans can use it) but DENIED to the
        # agent — reading it would invalidate the measurement.
        sol = b + "/solution"
        if p == sol or p.startswith(sol + "/"):
            return "the solution/ folder is the eval's answer key — off limits to the agent"
        if not (p == b or p.startswith(b + "/")):
            home = _real_home()
            # Allowed escapes — both Claude Code's OWN infrastructure, not user
            # data:
            #   1. ~/.claude (Claude's config + tokens we already trust).
            #   2. Background-task output. When Claude auto-moves a long
            #      foreground command to a background task, output goes to
            #      <tmp>/claude-*/<cwd-slug>/<session>/tasks/<id>.output —
            #      outside the boundary; denying it blinds the agent to its own
            #      command. The path embeds the cwd slug, scoping this to THIS
            #      run's own tasks.
            cwd_slug = os.getcwd().replace(os.sep, "-")
            is_task_output = "/tasks/" in p and p.endswith(".output") and cwd_slug in p
            if not p.startswith(home + "/.claude/") and not is_task_output:
                return f"path is outside the agent boundary ({b})"
    # Always block dotfiles + dot-dirs at $HOME root (secrets), boundary set or
    # not. Catches ~/.ssh, ~/.aws, ~/.gnupg, .gitconfig, .netrc, .zshrc —
    # every secret-bearing path on macOS.
    home = _real_home()
    rel = p[len(home) + 1 :] if p.startswith(home + "/") else None
    if rel and rel.startswith("."):
        # ~/.claude allowed (Claude's config + tokens we already trust).
        if rel == ".claude" or rel.startswith(".claude/"):
            return None
        return f"path targets a sensitive $HOME location ({home}/{rel.split('/')[0]})"
    return None


def _log_call(
    payload: dict, tool_name: str, tool_input: dict, denied_reason: str | None = None
) -> None:
    """Append one JSONL record to TESTBED_COMMAND_LOG. Denials carry
    `denied: true` + the reason, so results.py can count them structurally
    instead of substring-scanning transcripts for the `DENIED:` marker."""
    log_path = os.environ.get("TESTBED_COMMAND_LOG")
    if not log_path:
        return
    record = {
        "timestamp": time.time(),
        "session_id": payload.get("session_id"),
        "tool_name": tool_name,
        "category": classify(tool_name, tool_input),
        "tool_input": tool_input,
    }
    if denied_reason is not None:
        record["denied"] = True
        record["reason"] = denied_reason
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}

    def deny(reason: str) -> int:
        # Stderr is shown to the model so it learns why the call was rejected
        # and stops trying; the log record feeds results.denied_calls.
        _log_call(payload, tool_name, tool_input, denied_reason=reason)
        print(reason, file=sys.stderr, flush=True)
        return 2  # non-zero → block the tool call

    # Block forbidden tool-input flags.
    for flag, msg in FORBIDDEN_FLAGS.items():
        if tool_input.get(flag) is True:
            return deny(msg)

    # Parse a Bash command into shell segments ONCE; both the `mlpab run`
    # foreground check and the remote-only allowlist reuse the same parse.
    segments: list | None = None
    if tool_name == "Bash":
        segments = _pipeline_segments(tool_input.get("command") or "")
        # A nested `mlpab run` must be foreground: block piping into
        # head/tail (SIGPIPE-kills it mid-build) or backgrounding with `&`
        # (both make the caller poll a dead run).
        misuse = _mlpab_run_misuse("", segments)
        if misuse:
            return deny(misuse)

    # Remote-only / single-interface enforcement: block local compute and any
    # escape to an interface other than the one under test.
    reason = enforce(tool_name, tool_input, segments)
    if reason:
        return deny(reason)

    # Free-tier guard: deny any instance type outside TESTBED_INSTANCE_ALLOW
    # (no-op when the active platform sets no allowlist).
    reason = enforce_instance_types(tool_name, tool_input)
    if reason:
        return deny(reason)

    # Path-based deny for Read/Write/Edit. Enforces what `permissions.deny`
    # silently skips under bypassPermissions.
    if tool_name in _PATH_TOOLS:
        # NotebookEdit uses `notebook_path`; everything else uses `file_path`.
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        reason = _path_violates_boundary(path)
        if reason:
            return deny(
                f"DENIED: {tool_name}({path!r}) — {reason}. Stay inside your "
                f"working directory (or ~/.claude); rewrite the call to a "
                f"path under cwd."
            )

    # The run's solution/ folder is the eval's ANSWER KEY: in the boundary
    # (grader + humans read it) but off limits to the agent in EVERY mode —
    # including the none/none local baseline, which skips enforce(). Covers
    # the surfaces the file-tool check above doesn't: bash (allowed shell like
    # `cat solution/truth.json`) and the search tools.
    if tool_name == "Bash":
        if re.search(r'(?:^|[\s"\'=/(`;|&])solution/', tool_input.get("command") or ""):
            return deny(
                "DENIED: the solution/ folder is the eval's answer key — "
                "off limits to the agent. Solve the task from data/ only."
            )
    elif tool_name in ("Glob", "Grep"):
        blob = f"{tool_input.get('path') or ''} {tool_input.get('pattern') or ''}"
        if "solution" in blob:
            return deny(
                "DENIED: the solution/ folder is the eval's answer key — "
                "off limits to the agent. Search data/ instead."
            )

    # Log the (allowed) call.
    _log_call(payload, tool_name, tool_input)
    return 0


if __name__ == "__main__":
    sys.exit(main())
