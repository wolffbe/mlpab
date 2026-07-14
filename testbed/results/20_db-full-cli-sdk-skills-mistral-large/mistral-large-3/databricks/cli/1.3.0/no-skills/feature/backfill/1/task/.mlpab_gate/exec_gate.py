#!/usr/bin/env python3
"""Engine-agnostic in-flight exec gate — the SHELL that the hookless agents
(vibe / codex) run their commands through, so the single-interface allowlist is
enforced AT EXECUTION TIME for them too, not just post-hoc.

Why this exists: the Claude runner enforces the contract via a PreToolUse hook,
but vibe and codex have no such hook. Both, however, execute shell commands
through a configurable shell — vibe via `create_subprocess_shell(cmd,
executable=$SHELL)`, codex via its `user_shell` (derived from $SHELL). So the
runners point `SHELL` (and a `bash`/`sh` on PATH) at THIS script. Invoked as
`exec_gate.py -c "<command>"` (the standard non-interactive shell form), it:

  1. extracts the command string,
  2. runs the SAME `enforce()` / `enforce_instance_types()` as the Claude hook
     (this module's sibling `log_tool_call.py`, copied alongside; both read the
     TESTBED_* env the runner sets),
  3. on a denial: prints the `DENIED: …` reason to stderr and exits non-zero —
     the agent sees a failed command with the identical message a Claude run
     would have shown, so it reacts the same way,
  4. otherwise: hands off UNCHANGED to the real shell (`MLPAB_REAL_SHELL`, an
     absolute path, so there is no recursion back into the gate).

Any invocation WITHOUT `-c` (an interactive/login shell with no command) passes
straight through. Stdlib-only, like the hook, so it runs under any interpreter.
"""

from __future__ import annotations

import os
import sys

# Resolved by install_exec_gate to a REAL shell (absolute path) so exec'ing it
# never re-enters this gate. Fall back to /bin/bash, then /bin/sh.
_REAL_SHELL = os.environ.get("MLPAB_REAL_SHELL") or (
    "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
)


# Short-flag clusters that may bundle `-c` (login/interactive/restricted/etc.).
# Restricting to known shell flag letters avoids treating an arbitrary `-Xc`
# token as `-c` and mis-reading the following arg as the command.
_SHELL_FLAG_LETTERS = set("cilsxeruvfmnptbBCEHPT")


def _extract_command(argv: list[str]) -> str | None:
    """The command string from a non-interactive shell invocation. Handles
    `-c CMD`, combined short clusters containing c (`-lc CMD`, `-ic CMD`),
    `-l -c CMD`, and a `--` between the flag and the command (`-c -- CMD`).
    Returns None when there is no `-c` (interactive/login shell)."""
    for i, tok in enumerate(argv):
        is_c = tok == "-c" or (
            tok.startswith("-")
            and not tok.startswith("--")
            and "c" in tok[1:]
            and all(ch in _SHELL_FLAG_LETTERS for ch in tok[1:])
        )
        if not is_c:
            continue
        # The command is the next non-`--` argument (a `--` end-of-options
        # marker between `-c` and the command must not be taken as the command).
        j = i + 1
        while j < len(argv) and argv[j] == "--":
            j += 1
        return argv[j] if j < len(argv) else None
    return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = _extract_command(argv)
    if command is not None:
        # Import the shared enforcement logic copied next to this gate.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import log_tool_call as gate  # noqa: E402  (sibling, stdlib-only)

        # gate_check = the Bash subset of the Claude hook's main(): single-interface
        # allowlist + instance guard + solution/ answer-key deny (same source).
        reason = gate.gate_check(command)
        if reason:
            gate.gate_log_denial(command, reason)  # structured denied_calls record
            sys.stderr.write(reason.rstrip() + "\n")
            return 2
    # Allowed (or no command): run the real shell with the original args.
    os.execv(_REAL_SHELL, [_REAL_SHELL, *argv])
    return 127  # unreachable unless execv fails


if __name__ == "__main__":
    sys.exit(main())
