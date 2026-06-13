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

import contextlib
import fcntl
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from mlpab import interfaces, skills


class PreflightError(RuntimeError):
    """Raised when a required platform or skill isn't ready for a run."""


# ---------------------------------------------------------------------------
# Cross-session build barrier
#
# The per-platform rq1 configs run as PARALLEL `mlpab` sessions. Their setup
# work differs wildly: hopsworks clones + builds a wheel for minutes, while
# sagemaker has nothing to build — so without coordination the fast session
# opens its agent (claude) while a parallel session is still setting up.
# The barrier serializes that boundary IN BOTH DIRECTIONS, via two flock files:
#
#   * Setup side — every session holds the SHARED side of the BUILD lock for
#     its WHOLE setup phase: platform builds and data preparation
#     (concurrent setups stay concurrent). The window must span the full
#     pre-agent phase: guarding the build alone leaves a sibling's
#     data-prep minutes unguarded and the fast session opens claude anyway.
#   * Agent side — every agent run holds the SHARED side of the RUN
#     lock for its duration (concurrent agents stay concurrent). An
#     agent only STARTS when no session is setting up (momentary EXCLUSIVE
#     on the BUILD lock), and — the reverse direction — a session that begins
#     its setup while an agent is already open waits in `building()` until
#     every open run finishes (momentary EXCLUSIVE on the RUN lock). Without
#     the reverse check, sessions started STAGGERED slip past the barrier:
#     the first session sees no sibling, opens claude, and the late sibling
#     then builds right under the open (timed) agent run.
#
# Crash-safe: the OS drops flocks with the process.
# ---------------------------------------------------------------------------

BUILD_BARRIER_LOCK = interfaces.BUILD_DIR / ".build-barrier.lock"
AGENT_RUN_LOCK = interfaces.BUILD_DIR / ".agent-run.lock"


def _run_lock_for(build_lock: Path, run_lock: Path | None) -> Path:
    """The RUN lock that pairs with `build_lock` — explicit, or the sibling
    file next to it (keeps tests with a tmpdir build lock self-contained)."""
    if run_lock is not None:
        return run_lock
    if build_lock == BUILD_BARRIER_LOCK:
        return AGENT_RUN_LOCK
    return build_lock.with_name(AGENT_RUN_LOCK.name)


@contextlib.contextmanager
def building(lock_path: Path | None = None, run_lock_path: Path | None = None):
    """Mark this process as SETTING UP (building, preparing data) for the
    duration of the block (shared flock on the BUILD lock).
    Parallel sessions may set up concurrently; agent runs gate on
    `agent_slot()`/`await_builds()` until every holder releases.

    Entry blocks until no agent run is open (exclusive side of the RUN
    lock, taken momentarily): a session started AFTER a sibling's agent
    opened must not build under that run. The BUILD lock is already held
    (shared) while waiting, so no NEW agent starts in the gap.
    """
    path = lock_path or BUILD_BARRIER_LOCK
    rpath = _run_lock_for(path, run_lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        # Reverse direction — wait for open agent runs to drain before any
        # setup work. Best-effort: an OS error here must not fail the session.
        try:
            rfd = os.open(rpath, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                try:
                    fcntl.flock(rfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    print("[mlpab] a parallel session has an agent run "
                          "open — waiting for it to finish before setting up "
                          "(building / preparing data)…",
                          file=sys.stderr, flush=True)
                    fcntl.flock(rfd, fcntl.LOCK_EX)
                with contextlib.suppress(OSError):
                    fcntl.flock(rfd, fcntl.LOCK_UN)
            finally:
                os.close(rfd)
        except OSError:
            pass
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextlib.contextmanager
def agent_slot(lock_path: Path | None = None, run_lock_path: Path | None = None):
    """Hold the AGENT side of the barrier for the duration of one run.

    Agents run concurrently (shared flock on the RUN lock), but a run only
    STARTS when no parallel session is setting up — and once held, the slot
    makes a late-starting sibling's `building()` wait instead of building
    under this run. Yields the seconds spent waiting for setups to clear.

    Deadlock-free dance: take the RUN slot first, then PROBE the BUILD lock
    non-blocking; if a setup is in progress, drop the slot (so the sibling's
    drain in `building()` can proceed), wait for the setup to finish, retry.
    Best-effort: any OS error yields a zero wait rather than failing the run.
    """
    bpath = lock_path or BUILD_BARRIER_LOCK
    rpath = _run_lock_for(bpath, run_lock_path)
    start = time.monotonic()
    run_fd = None
    warned = False
    # Acquisition is best-effort: any OS error proceeds with whatever was
    # acquired (possibly nothing) rather than failing the run it gates.
    try:
        rpath.parent.mkdir(parents=True, exist_ok=True)
        run_fd = os.open(rpath, os.O_CREAT | os.O_RDWR, 0o644)
        while True:
            fcntl.flock(run_fd, fcntl.LOCK_SH)
            bfd = os.open(bpath, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                try:
                    fcntl.flock(bfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with contextlib.suppress(OSError):
                        fcntl.flock(bfd, fcntl.LOCK_UN)
                    break  # no setup in progress — slot held, start the run
                except OSError:
                    # A sibling is setting up: release our slot so its drain
                    # completes, wait for the setup to clear, then retry.
                    fcntl.flock(run_fd, fcntl.LOCK_UN)
                    if not warned:
                        print("[mlpab] a parallel session is still building "
                              "/ preparing data — waiting at the build "
                              "barrier before starting the agent…",
                              file=sys.stderr, flush=True)
                        warned = True
                    fcntl.flock(bfd, fcntl.LOCK_EX)
                    with contextlib.suppress(OSError):
                        fcntl.flock(bfd, fcntl.LOCK_UN)
            finally:
                os.close(bfd)
    except OSError:
        pass
    try:
        yield time.monotonic() - start
    finally:
        if run_fd is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(run_fd, fcntl.LOCK_UN)
            os.close(run_fd)


def await_builds(lock_path: Path | None = None) -> float:
    """Block until NO parallel session is inside `building()`, by taking (and
    immediately releasing) the exclusive side of the barrier flock. Returns the
    seconds waited. Best-effort: any OS error waits zero rather than failing
    the run it gates. NOTE: agent runs should prefer `agent_slot()`,
    which also HOLDS the run's side so late-starting siblings wait in
    `building()` instead of building under the open run."""
    path = lock_path or BUILD_BARRIER_LOCK
    if not path.exists():
        return 0.0
    start = time.monotonic()
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        return 0.0
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("[mlpab] a parallel session is still building / preparing "
                  "data — waiting at the build barrier before starting the "
                  "agent…", file=sys.stderr, flush=True)
            fcntl.flock(fd, fcntl.LOCK_EX)
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        return time.monotonic() - start
    except OSError:
        return time.monotonic() - start
    finally:
        os.close(fd)


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
            req.platform, req.interface,
            check_login=check_login, timeout_s=timeout_s, cleanup_build=cleanup_build,
        )
        if not status.ok:
            raise PreflightError(status.message)

    # Phase 2 — skills (agent invokes the skill directly).
    seen_skills: set[tuple] = set()
    for req in requirements:
        if req.skills and req.skills != "none":
            skey = (req.platform, req.skills)
            if skey not in seen_skills:
                seen_skills.add(skey)
                _check_skill(
                    req.platform, req.skills,
                    auth=auth, model=model, probe=probe_skills, timeout_s=timeout_s,
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
        auth=auth, model=model, probe_skills=probe_skills, check_login=False,
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

    # 2) Confirm the agent can ACCESS each skill by invoking it directly with
    # `/skill-name` (per https://code.claude.com/docs/en/skills) — no LLM judgment.
    for skill_name in skills.skill_names(project, name):
        result = _probe_skill_invocation(
            project, name, skill_name,
            auth=auth, model=model, timeout_s=timeout_s,
        )
        if not result.ok:
            raise PreflightError(
                f"skill {skill_name!r} (bundle {name!r}): the agent "
                f"could not invoke it.\n  {result.detail}"
            )


def _probe_skill_invocation(
    project: str, name: str, skill_name: str,
    *, auth: str, model: str, timeout_s: int,
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
        low = out.lower()
        # An unrecognized slash command short-circuits with this exact banner
        # ("Unknown command: /name", exit 0) and never consumes turns — the
        # definitive "not a skill" signal, independent of return code.
        if "unknown command" in low:
            return _ProbeResult(False, f"/{skill_name} was not recognized as a skill. Got: {out.strip()[:300]!r}")
        # A substantial skill (interactive / multi-step — e.g. one that calls
        # AskUserQuestion or hands off to other skills) loads, starts working,
        # then hits the probe's 3-turn ceiling and exits non-zero with
        # "Reached max turns". That PROVES the skill was recognized and invoked,
        # which is all this accessibility probe needs to confirm.
        if "reached max turns" in low:
            return _ProbeResult(True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            return _ProbeResult(False, f"`claude -p /{skill_name}` exited {proc.returncode}: {' / '.join(tail)}")
        return _ProbeResult(True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
