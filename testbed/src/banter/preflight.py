"""Upfront, fail-fast preflight for engineer runs.

Before any engineer run — and once, over the union of requirements, before a
benchmark or autoresearch session — we verify that everything the engineer
(the controlled Claude Code instance) will depend on is actually ready:

  * Platforms — built (from the config's install steps), logged in, and tested
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
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from banter import interfaces, skills


class PreflightError(RuntimeError):
    """Raised when a required platform or skill isn't ready for a run."""


@dataclass
class Requirement:
    """One (platform, interface, skills) combination a run will use."""
    platform: str
    interface: str
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
    cleanup_build: bool = False,
) -> None:
    """Verify every requirement upfront. Raises PreflightError on first failure.

    Runs in two distinct phases:

      Phase 1 — PLATFORMS (deterministic, no AI). Every platform is built,
      logged in, and tested via plain shell commands. This phase is fully
      independent of the researcher and the engineer; nothing here invokes a
      Claude instance, and it completes before any AI is involved.

      Phase 2 — SKILLS (uses the engineer). The only check that needs an AI:
      a short engineer probe confirms each skill is actually accessible.

    Platforms and skills are de-duplicated so each unique one is checked once.
    """
    # Phase 1 — platforms only (no AI).
    seen_platforms: set[tuple] = set()
    for req in requirements:
        pkey = (req.platform, req.interface, req.interface_version, str(req.version_root))
        if pkey in seen_platforms:
            continue
        seen_platforms.add(pkey)
        status = interfaces.preflight(
            req.platform, req.interface, req.interface_version, req.version_root,
            check_login=check_login, timeout_s=timeout_s, cleanup_build=cleanup_build,
        )
        if not status.ok:
            raise PreflightError(status.message)

    # Phase 2 — skills (engineer invokes the skill directly).
    seen_skills: set[tuple] = set()
    for req in requirements:
        if req.skills and req.skills != "none":
            skey = (req.platform, req.skills, req.skills_version, str(req.version_root))
            if skey not in seen_skills:
                seen_skills.add(skey)
                _check_skill(
                    req.platform, req.skills, req.skills_version, req.version_root,
                    auth=auth, model=model, probe=probe_skills, timeout_s=timeout_s,
                )


def check_run(
    *,
    platform: str,
    interface: str,
    interface_version: int | None,
    version_root: Path | None,
    skills: str,
    skills_version: int | None,
    auth: str,
    model: str,
    probe_skills: bool = True,
) -> None:
    """Preflight a single run's requirement (build + test + skill probe). Login is
    checked per challenge by the runner (interfaces.login_status), not here.
    Raises PreflightError on failure."""
    preflight(
        [Requirement(
            platform=platform, interface=interface, interface_version=interface_version,
            version_root=version_root, skills=skills, skills_version=skills_version,
        )],
        auth=auth, model=model, probe_skills=probe_skills, check_login=False,
    )


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def _check_skill(
    project: str,
    name: str,
    version: int | None,
    version_root: Path | None,
    *,
    auth: str,
    model: str,
    probe: bool,
    timeout_s: int,
) -> None:
    # 1) exists + well-formed (raises ValueError on problems)
    try:
        version, _hash, _variant = skills.verify_installed(project, name, version, version_root)
    except ValueError as e:
        raise PreflightError(
            f"skill bundle {name!r}: {e}\n"
            f"  → Fix the bundle under platforms/{project}/skills/{name}/."
        )

    if not probe:
        return

    # 2) Confirm the engineer can ACCESS each skill by invoking it directly with
    # `/skill-name` (per https://code.claude.com/docs/en/skills) — no LLM judgment.
    for skill_name in skills.skill_names(project, name, version, version_root):
        result = _probe_skill_invocation(
            project, name, version, version_root, skill_name,
            auth=auth, model=model, timeout_s=timeout_s,
        )
        if not result.ok:
            raise PreflightError(
                f"skill {skill_name!r} (bundle {name!r} v{version}): the engineer "
                f"could not invoke it.\n  {result.detail}"
            )


def _probe_skill_invocation(
    project: str, name: str, version: int, version_root: Path | None, skill_name: str,
    *, auth: str, model: str, timeout_s: int,
) -> _ProbeResult:
    """Install the bundle like a real run and invoke `/skill-name` directly."""
    if shutil.which("claude") is None:
        raise PreflightError("`claude` CLI not found on PATH. Install Claude Code first.")

    tmp = Path(tempfile.mkdtemp(prefix="banter-skillprobe-"))
    try:
        skills.apply(project, name, tmp, version, version_root)
        (tmp / ".claude").mkdir(parents=True, exist_ok=True)
        settings_file = tmp / ".claude" / "settings.json"
        settings_file.write_text("{}")

        env = os.environ.copy()
        env.pop("ANTHROPIC_BASE_URL", None)
        if auth == "login":
            env.pop("ANTHROPIC_API_KEY", None)
        env["ANTHROPIC_MODEL"] = model

        cmd = [
            "claude", "-p", f"/{skill_name}",
            "--model", model,
            "--permission-mode", "bypassPermissions",
            "--settings", str(settings_file.resolve()),
            "--setting-sources", "project,local,user",
            "--max-turns", "3",
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=str(tmp), env=env, stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return _ProbeResult(False, f"invoking /{skill_name} timed out.")
        out = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            return _ProbeResult(False, f"`claude -p /{skill_name}` exited {proc.returncode}: {' / '.join(tail)}")
        low = out.lower()
        if any(p in low for p in ("unknown command", "no such", "not found", "isn't a", "unrecognized")):
            return _ProbeResult(False, f"/{skill_name} was not recognized as a skill. Got: {out.strip()[:300]!r}")
        return _ProbeResult(True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
