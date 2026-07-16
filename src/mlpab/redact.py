"""Redact secret values from text bound for logs, exceptions, or the results CSV.

The testbed injects platform credentials (API keys, tokens, client secrets) into
subprocess environments. Captured output — a failing auth command, an agent's
stderr tail, a setup/teardown verify probe — can echo those values back. Scrub
them before the text is printed, stored in `results.csv` (`error` column), or
surfaced in an exception, so a credential never lands on disk or a terminal.

Conservative by design: only env values whose NAME looks credential-bearing
(KEY/TOKEN/SECRET/PASSWORD/…) and that are long enough to be a real secret are
redacted, so short, non-secret values (a region, `true`, a model id) are left
intact. Callers may pass `extra` values (e.g. manifest-embedded key values not in
the environment).
"""

from __future__ import annotations

import os
from typing import Iterable

PLACEHOLDER = "***REDACTED***"

# Env var NAME substrings that mark its value as a secret.
_SECRET_NAME_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")
# Short values are not treated as secrets (avoids scrubbing region names, "true",
# a model id that happens to sit in a *_KEY var, etc.).
_MIN_LEN = 8


def secret_values(env: dict[str, str] | None = None) -> set[str]:
    """The set of credential values to scrub, taken from env vars whose name
    looks secret-bearing and whose value is long enough to be a real secret."""
    src = os.environ if env is None else env
    out: set[str] = set()
    for name, val in src.items():
        if not val or len(val) < _MIN_LEN:
            continue
        if any(hint in name.upper() for hint in _SECRET_NAME_HINTS):
            out.add(val)
    return out


def redact(text: str | None, extra: Iterable[str] = (), env: dict[str, str] | None = None) -> str:
    """Replace every known secret value in `text` with a placeholder. `extra`
    supplies secret values from outside the environment (e.g. manifest key
    values). Longest values are replaced first so a secret that contains another
    secret as a substring is fully masked."""
    if not text:
        return text or ""
    secrets = secret_values(env) | {s for s in extra if s and len(s) >= _MIN_LEN}
    for s in sorted(secrets, key=len, reverse=True):
        if s in text:
            text = text.replace(s, PLACEHOLDER)
    return text
