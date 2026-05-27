"""Upfront, fail-fast preflight for engineer runs.

Before any engineer run — and once, over the union of requirements, before a
benchmark or autoresearch session — we verify that everything the engineer
(the controlled Claude Code instance) will depend on is actually ready:

  * Interfaces — built (from the config's install steps), logged in, and tested
    (delegated to interfaces.preflight; on failure it points at the config or
    `make setup`).
  * Skills — the bundle exists, installs, AND the engineer can actually access
    it in a run. The skill check is a real probe: we stand up the bundle exactly
    as a run would, spawn a short engineer `claude -p`, and confirm every skill
    is visible to it. If a skill can't be accessed, the run fails immediately.

The first failed check raises PreflightError with an actionable fix.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from banter import interfaces, skills


# Prompts live at the repo top level (prompts/). The skill-access probe prompt
# is loaded from there so it can be edited without touching code.
PROMPTS_DIR = interfaces._TESTBED_ROOT / "prompts"
_DEFAULT_SKILL_PROBE = (
    "List the exact names of every skill currently available to you, one per "
    "line, with no other text. If you have no skills, output NONE."
)


def _skill_probe_prompt() -> str:
    p = PROMPTS_DIR / "skill_probe.md"
    if p.exists():
        return p.read_text().strip()
    return _DEFAULT_SKILL_PROBE


class PreflightError(RuntimeError):
    """Raised when a required interface or skill isn't ready for a run."""


@dataclass
class Requirement:
    """One (interface, skills) combination a run will use."""
    interface: str
    mode: str
    interface_version: int | None = None
    version_root: Path | None = None
    skills: str = "none"
    skills_version: int | None = None


@dataclass
class _ProbeResult:
    ok: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def preflight(
    requirements: list[Requirement],
    *,
    auth: str,
    model: str,
    check_login: bool = True,
    probe_skills: bool = True,
    timeout_s: int = 120,
) -> None:
    """Verify every requirement upfront. Raises PreflightError on first failure.

    Runs in two distinct phases:

      Phase 1 — INTERFACES (deterministic, no AI). Every interface is built,
      logged in, and tested via plain shell commands. This phase is fully
      independent of the researcher and the engineer; nothing here invokes a
      Claude instance, and it completes before any AI is involved.

      Phase 2 — SKILLS (uses the engineer). The only check that needs an AI:
      a short engineer probe confirms each skill is actually accessible.

    Interfaces and skills are de-duplicated so each unique one is checked once.
    """
    # Phase 1 — interfaces only (no AI).
    seen_ifaces: set[tuple] = set()
    for req in requirements:
        ikey = (req.interface, req.mode, req.interface_version, str(req.version_root))
        if ikey in seen_ifaces:
            continue
        seen_ifaces.add(ikey)
        status = interfaces.preflight(
            req.interface, req.mode, req.interface_version, req.version_root,
            check_login=check_login, timeout_s=timeout_s,
        )
        if not status.ok:
            raise PreflightError(status.message)

    # Phase 2 — skills (engineer probe).
    seen_skills: set[tuple] = set()
    for req in requirements:
        if req.skills and req.skills != "none":
            skey = (req.skills, req.skills_version)
            if skey not in seen_skills:
                seen_skills.add(skey)
                _check_skill(
                    req.skills, req.skills_version,
                    auth=auth, model=model, probe=probe_skills, timeout_s=timeout_s,
                )


def check_run(
    *,
    interface: str,
    mode: str,
    interface_version: int | None,
    version_root: Path | None,
    skills: str,
    skills_version: int | None,
    auth: str,
    model: str,
    probe_skills: bool = True,
) -> None:
    """Preflight a single run's requirement. Raises PreflightError on failure."""
    preflight(
        [Requirement(
            interface=interface, mode=mode, interface_version=interface_version,
            version_root=version_root, skills=skills, skills_version=skills_version,
        )],
        auth=auth, model=model, probe_skills=probe_skills,
    )


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def _check_skill(
    name: str,
    version: int | None,
    *,
    auth: str,
    model: str,
    probe: bool,
    timeout_s: int,
) -> None:
    # 1) exists + well-formed (raises ValueError on problems)
    try:
        version, _hash, variant = skills.verify_installed(name, version)
    except ValueError as e:
        raise PreflightError(
            f"skill bundle {name!r}: {e}\n"
            f"  → Fix the bundle under skills/{name}/ "
            f"(<version>/<skill>/SKILL.md layout)."
        )

    if not probe:
        return

    # 2) real probe — can the engineer actually access the skill in a run?
    expected = _skill_display_names(variant)
    result = _probe_skill_access(name, version, auth=auth, model=model, timeout_s=timeout_s)
    if not result.ok:
        raise PreflightError(
            f"skill bundle {name!r} v{version}: engineer could not access "
            f"skill(s) {expected} in a probe run.\n  {result.detail}"
        )


_FRONTMATTER_NAME = re.compile(r"(?m)^name:\s*(.+?)\s*$")


def _skill_display_names(variant: Path) -> list[str]:
    """Names the engineer would see — SKILL.md `name:` frontmatter, else folder."""
    names: list[str] = []
    for d in sorted(p for p in variant.iterdir() if p.is_dir()):
        name = d.name
        md = d / "SKILL.md"
        if md.is_file():
            head = md.read_text()[:2000]
            m = _FRONTMATTER_NAME.search(head)
            if m:
                name = m.group(1).strip().strip("\"'")
        names.append(name)
    return names


def _probe_skill_access(
    name: str, version: int, *, auth: str, model: str, timeout_s: int
) -> _ProbeResult:
    """Stand the bundle up like a real run and confirm the engineer sees it."""
    if shutil.which("claude") is None:
        raise PreflightError("`claude` CLI not found on PATH. Install Claude Code first.")

    expected = []
    tmp = Path(tempfile.mkdtemp(prefix="banter-skillprobe-"))
    try:
        setup = skills.apply(name, tmp, version)
        variant = skills.SKILLS_ROOT / name / str(setup.version)
        expected = _skill_display_names(variant)
        # Minimal project settings so the project setting-source has a file to load.
        (tmp / ".claude").mkdir(parents=True, exist_ok=True)
        settings_file = tmp / ".claude" / "settings.json"
        settings_file.write_text("{}")

        env = os.environ.copy()
        env.pop("ANTHROPIC_BASE_URL", None)
        if auth == "login":
            env.pop("ANTHROPIC_API_KEY", None)
        env["ANTHROPIC_MODEL"] = model

        prompt = _skill_probe_prompt()
        cmd = [
            "claude", "-p", prompt,
            "--model", model,
            "--permission-mode", "bypassPermissions",
            "--settings", str(settings_file.resolve()),
            "--setting-sources", "project,local,user",
            "--max-turns", "3",
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=str(tmp), env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return _ProbeResult(False, "probe timed out before the engineer responded.")
        out = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            return _ProbeResult(
                False, f"probe `claude -p` exited {proc.returncode}: {' / '.join(tail)}"
            )
        low = out.lower()
        missing = [s for s in expected if s.lower() not in low]
        if missing:
            return _ProbeResult(
                False, f"engineer's skill listing did not include: {missing}. "
                f"Got: {out.strip()[:300]!r}"
            )
        return _ProbeResult(True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
