"""Upfront, fail-fast preflight for agent runs.

Before any agent run — and once, over the union of requirements, before a
treatment session — we verify that everything the agent
(the controlled Claude Code instance) will depend on is actually ready:

  * Platforms — built (from the config's install steps), logged in, and tested
    (delegated to interfaces.preflight; on failure it points at the config or
    `make setup`).
  * Skills — the bundle exists, installs, AND the agent can actually access
    it in a run. The skill check is a real probe: we stand up the bundle exactly
    as a run would, spawn a short agent `claude -p`, and confirm every skill
    is visible to it. If a skill can't be accessed, the run fails immediately.

The first failed check raises PreflightError with an actionable fix.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mlpab import interfaces, skills


# How many skills to spot-check per bundle in the access probe. Skill access is a
# property of the bundle install, not the individual skill, so a small random
# sample is enough to confirm the bundle is reachable without probing all ~23.
_SKILL_PROBE_SAMPLE = 3


class PreflightError(RuntimeError):
    """Raised when a required platform or skill isn't ready for a run."""


class PlatformNotReadyError(PreflightError):
    """Raised when a run's setup verification fails — the platform itself is not
    usable (no connection / setup did not establish what the agent needs). This
    aborts the WHOLE config (every subsequent run would hit the same broken
    platform), unlike a per-run failure which is recorded and skipped."""


# NOTE: parallel `mlpab` sessions used to coordinate through a cross-session
# build barrier here (a fast session would otherwise open its timed agent while
# a sibling was still building). That barrier is gone: interfaces are now built
# AND materialized into a per-interface PREPARED venv once, up front, during the
# (serial) setup phase (interfaces.prepare). Each run only CLONES its prepared
# venv read-only, so runs do no build work and mutate no shared state — there is
# nothing left to serialize, and parallel runs never wait on one another.


@dataclass
class Requirement:
    """One (platform, interface, skills) combination a run will use."""

    platform: str
    interface: str
    skills: str = "none"


@dataclass
class _ProbeResult:
    ok: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def check_availability(
    requirements: list[Requirement],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Fast, no-build, no-network gate run at the START of a session — before
    building/preparing interfaces or launching a single run. Confirms that every
    platform/interface/skill a config needs is even usable:

      * the interface config manifest exists;
      * every declared credential key has a value (so the platform CAN be
        reached — the live connection itself is verified in the build+login
        preflight that follows);
      * each skills bundle is present and well-formed.

    Collects ALL problems across the union and raises ONE PreflightError listing
    them, so an operator fixes everything at once instead of one build-cycle at a
    time. De-duplicates so each unique platform/skill is checked once.
    """
    problems: list[str] = []

    seen_platforms: set[tuple] = set()
    for req in requirements:
        if req.platform == "none" and req.interface == "none":
            continue
        pkey = (req.platform, req.interface)
        if pkey in seen_platforms:
            continue
        seen_platforms.add(pkey)
        if not interfaces.load_manifest(req.platform, req.interface):
            problems.append(
                f"{req.platform}/{req.interface}: no config manifest at "
                f"{interfaces.manifest_path(req.platform, req.interface)}"
            )
            continue
        missing = interfaces.missing_keys(req.platform, req.interface, env)
        if missing:
            problems.append(
                f"{req.platform}/{req.interface}: missing credential(s): "
                f"{', '.join(missing)} — set them in .env (or `mlpab setup`)"
            )

    seen_skills: set[tuple] = set()
    for req in requirements:
        if not req.skills or req.skills == "none":
            continue
        skey = (req.platform, req.skills)
        if skey in seen_skills:
            continue
        seen_skills.add(skey)
        try:
            skills.verify_installed(req.platform, req.skills)
        except ValueError as e:
            problems.append(f"skill bundle {req.skills!r} ({req.platform}): {e}")

    if problems:
        raise PreflightError(
            "availability check failed — fix these before the session can run:\n  - "
            + "\n  - ".join(problems)
        )


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
      logged in, and tested via plain shell commands. Nothing here invokes a
      Claude instance; it completes before any AI is involved.

      Phase 2 — SKILLS (uses the agent). The only check that needs an AI:
      a short agent probe confirms each skill is actually accessible.

    Platforms and skills are de-duplicated so each unique one is checked once.
    """
    # Phase 1 — platforms only (no AI).
    seen_platforms: set[tuple] = set()
    for req in requirements:
        pkey = (req.platform, req.interface)
        if pkey in seen_platforms:
            continue
        seen_platforms.add(pkey)
        status = interfaces.preflight(
            req.platform,
            req.interface,
            check_login=check_login,
            timeout_s=timeout_s,
            cleanup_build=cleanup_build,
        )
        if not status.ok:
            raise PreflightError(status.message)

    # Per-platform PLUMBING venv (setup/verify/teardown run under it — it holds
    # the platform's python client on EVERY interface, where a CLI run venv may
    # not). Built once here in the serial setup phase so no timed run pays for
    # it; idempotent, and a no-op for platforms that need none. Best-effort —
    # the runner rebuilds it (and falls back to the run venv python) if missing.
    for plat in sorted({r.platform for r in requirements if r.platform != "none"}):
        try:
            interfaces.prepare_plumbing(plat)
        except Exception as e:
            print(f"[mlpab] plumbing venv prep for {plat!r} skipped: {e}", file=sys.stderr)

    # Phase 2 — skills (agent invokes the skill directly).
    seen_skills: set[tuple] = set()
    for req in requirements:
        if req.skills and req.skills != "none":
            skey = (req.platform, req.skills)
            if skey not in seen_skills:
                seen_skills.add(skey)
                _check_skill(
                    req.platform,
                    req.skills,
                    auth=auth,
                    model=model,
                    probe=probe_skills,
                    timeout_s=timeout_s,
                )


def check_run(
    *,
    platform: str,
    interface: str,
    skills: str,
    auth: str,
    model: str,
    probe_skills: bool = True,
) -> None:
    """Preflight a single run's requirement (build + test + skill probe). Login is
    checked per run by the runner (interfaces.login_status), not here.
    Raises PreflightError on failure."""
    preflight(
        [Requirement(platform=platform, interface=interface, skills=skills)],
        auth=auth,
        model=model,
        probe_skills=probe_skills,
        check_login=False,
    )


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


def _check_skill(
    project: str,
    name: str,
    *,
    auth: str,
    model: str,
    probe: bool,
    timeout_s: int,
) -> None:
    # 1) exists + well-formed (raises ValueError on problems)
    try:
        _hash, _variant = skills.verify_installed(project, name)
    except ValueError as e:
        raise PreflightError(
            f"skill bundle {name!r}: {e}\n"
            f"  → Fix the bundle under configs/platforms/{project}/skills/{name}/ "
            f"(or point at its content via configs/platforms/{project}/skills.yaml)."
        )

    if not probe:
        return

    # The invocation probe shells the `claude` CLI (`claude -p /skill`), so it is
    # only meaningful for claude models. For a multi-model config whose first
    # model is gpt-*/mistral-*, probing `claude --model gpt-…` would fail and
    # abort the whole config — so skip the agent probe for non-claude models and
    # rely on the static bundle verification above.
    if not model.lower().startswith("claude"):
        return

    # 2) Confirm the agent can ACCESS skills by invoking them directly with
    # `/skill-name` (per https://code.claude.com/docs/en/skills) — no LLM judgment.
    # Each probe spawns a blocking `claude -p /skill` whose output is captured (not
    # streamed), so without the log lines below the run looks frozen during preflight.
    # Probing every skill in a large bundle (e.g. ~23) is slow and redundant — skill
    # access is a property of the bundle install, not the individual skill, so a
    # random spot-check of a few is enough to confirm the bundle is reachable.
    all_skills = list(skills.skill_names(project, name))
    sample_n = min(_SKILL_PROBE_SAMPLE, len(all_skills))
    skill_list = random.sample(all_skills, sample_n) if all_skills else []
    print(
        f"[mlpab] preflight: probing {len(skill_list)} of {len(all_skills)} skill(s) "
        f"in bundle {name!r} with `claude -p` (model {model})...",
        file=sys.stderr,
        flush=True,
    )
    for i, skill_name in enumerate(skill_list, 1):
        print(
            f"[mlpab] preflight: probing skill {i}/{len(skill_list)} /{skill_name} ...",
            file=sys.stderr,
            flush=True,
        )
        result = _probe_skill_invocation(
            project,
            name,
            skill_name,
            auth=auth,
            model=model,
            timeout_s=timeout_s,
        )
        if not result.ok:
            raise PreflightError(
                f"skill {skill_name!r} (bundle {name!r}): the agent "
                f"could not invoke it.\n  {result.detail}"
            )
        print(
            f"[mlpab] preflight: skill {i}/{len(skill_list)} /{skill_name} OK",
            file=sys.stderr,
            flush=True,
        )


def _probe_skill_invocation(
    project: str,
    name: str,
    skill_name: str,
    *,
    auth: str,
    model: str,
    timeout_s: int,
) -> _ProbeResult:
    """Install the bundle like a real run and invoke `/skill-name` directly."""
    if shutil.which("claude") is None:
        raise PreflightError("`claude` CLI not found on PATH. Install Claude Code first.")

    tmp = Path(tempfile.mkdtemp(prefix="mlpab-skillprobe-"))
    try:
        skills.apply(project, name, tmp)
        (tmp / ".claude").mkdir(parents=True, exist_ok=True)
        settings_file = tmp / ".claude" / "settings.json"
        settings_file.write_text("{}")

        env = os.environ.copy()
        env.pop("ANTHROPIC_BASE_URL", None)
        if auth == "login":
            env.pop("ANTHROPIC_API_KEY", None)
        env["ANTHROPIC_MODEL"] = model

        cmd = [
            "claude",
            "-p",
            f"/{skill_name}",
            "--model",
            model,
            "--permission-mode",
            "bypassPermissions",
            "--settings",
            str(settings_file.resolve()),
            "--setting-sources",
            "project,local,user",
            "--max-turns",
            "3",
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(tmp),
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return _ProbeResult(False, f"invoking /{skill_name} timed out.")
        out = f"{proc.stdout or ''}\n{proc.stderr or ''}"
        low = out.lower()
        # An unrecognized slash command short-circuits with this exact banner
        # ("Unknown command: /name", exit 0) and never consumes turns — the
        # definitive "not a skill" signal, independent of return code.
        if "unknown command" in low:
            return _ProbeResult(
                False, f"/{skill_name} was not recognized as a skill. Got: {out.strip()[:300]!r}"
            )
        # A substantial skill (interactive / multi-step — e.g. one that calls
        # AskUserQuestion or hands off to other skills) loads, starts working,
        # then hits the probe's 3-turn ceiling and exits non-zero with
        # "Reached max turns". That PROVES the skill was recognized and invoked,
        # which is all this accessibility probe needs to confirm.
        if "reached max turns" in low:
            return _ProbeResult(True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            return _ProbeResult(
                False, f"`claude -p /{skill_name}` exited {proc.returncode}: {' / '.join(tail)}"
            )
        return _ProbeResult(True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
