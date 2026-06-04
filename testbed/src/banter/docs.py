"""Reference docs cloned from a git URL into the run.

Configured in autoresearch / benchmark / `banter run` via the `docs:` field
(or `--docs` flag). The value is one of:

  - any name (preferred) → selects the platform's SINGLE docs config at
    `platforms/<platform>/docs/config.yaml`, whose `repo:` is the GitHub URL to
    download, e.g. `docs: hopsworks-docs`;
  - a raw git URL (back-compat), e.g.
    `docs: https://github.com/logicalclocks/logicalclocks.github.io`;
  - a local path (test fixtures / pre-downloaded mirror);
  - `none` (the without-docs control).

`resolve(spec, platform)` turns the name into the URL/path that `apply` then
materializes.

Materialization:
  - autoresearch:    cloned ONCE into `<run>/docs/` at session start.
                     Each engineer challenge gets its own copy at
                     `<challenge>/docs/` (APFS-cloned from the run-level
                     copy when available, otherwise re-cloned from git).
  - benchmark / standalone `banter run`:
                     cloned per-challenge directly into `<challenge>/docs/`.

`docs: none` is the without-docs control — no clone, nothing in the run.

Docs are static across versions (they describe the platform, not the
attempt). They influence neither the call-accounting metrics nor any
permission/deny pattern — they're just files the model can `Read`.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_TESTBED_ROOT = Path(__file__).resolve().parents[2]
PLATFORMS_DIR = _TESTBED_ROOT / "platforms"


@dataclass
class DocsSetup:
    spec: str               # original config value: URL, path, or "none"
    files: list[str] = field(default_factory=list)


def _looks_like_url(spec: str) -> bool:
    return spec.startswith(("http://", "https://", "git@", "ssh://", "git://"))


def _platform_docs_config(platform: str | None) -> Path | None:
    """The platform's single docs config, or None. There is exactly one config
    per platform at `platforms/<platform>/docs/config.yaml` — no per-name subdir."""
    if not platform:
        return None
    cfg = PLATFORMS_DIR / platform / "docs" / "config.yaml"
    return cfg if cfg.is_file() else None


def resolve(spec: str, platform: str | None = None) -> str:
    """Resolve a `--docs` value to something `apply` can materialize.

    - ``"none"`` / empty       → ``"none"``
    - a git URL / local path    → returned unchanged (back-compat)
    - any other name            → selects the platform's SINGLE docs config at
                                  ``platforms/<platform>/docs/config.yaml`` and
                                  returns its ``repo:`` (a GitHub URL to clone)
                                  or ``path:`` (a local mirror, resolved relative
                                  to the config).

    Raises ``ValueError`` for a name that matches neither the platform's docs
    config, a URL, nor an existing directory.
    """
    if not spec or spec == "none":
        return "none"
    if _looks_like_url(spec):
        return spec
    cfg = _platform_docs_config(platform)
    if cfg is not None:
        manifest = yaml.safe_load(cfg.read_text()) or {}
        url = manifest.get("repo") or manifest.get("url")
        if url:
            return str(url)
        local = manifest.get("path")
        if local:
            return str((cfg.parent / local).expanduser().resolve())
        raise ValueError(
            f"docs bundle {spec!r} at {cfg} declares no `repo:`/`url:`/`path:`."
        )
    p = Path(spec).expanduser()
    if p.is_dir():
        return str(p)
    raise ValueError(
        f"docs {spec!r}: not a git URL, not the platform's docs config at "
        f"platforms/{platform or '<platform>'}/docs/config.yaml, and not an "
        f"existing directory."
    )


def resolve_ref(spec: str, platform: str | None = None) -> str | None:
    """The pinned commit/branch (`ref:`/`commit:`) for a named platform docs
    config, or None. None for `none`, a raw URL, or a local path — those are not
    pinned here. Lets `apply` check out a fixed commit for reproducible docs."""
    if not spec or spec == "none" or _looks_like_url(spec):
        return None
    cfg = _platform_docs_config(platform)
    if cfg is None:
        return None
    m = yaml.safe_load(cfg.read_text()) or {}
    ref = m.get("ref") or m.get("commit")
    return str(ref) if ref else None


def _clone(url: str, dst: Path, ref: str | None = None) -> None:
    """Shallow git clone (depth 1) into dst. Removes dst if it exists.

    When `ref` is given (a fixed commit SHA, branch, or tag) the exact ref is
    fetched and checked out detached — so the docs are pinned/reproducible.
    """
    if dst.exists():
        shutil.rmtree(dst)
    if ref:
        dst.mkdir(parents=True, exist_ok=True)
        run = lambda *a: subprocess.run(  # noqa: E731
            ["git", "-C", str(dst), *a], check=True, capture_output=True, text=True
        )
        run("init", "-q")
        run("remote", "add", "origin", url)
        run("fetch", "--depth", "1", "origin", ref)
        run("checkout", "-q", "FETCH_HEAD")
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dst)],
            check=True, capture_output=True, text=True,
        )


def _clone_tree(src: Path, dst: Path) -> None:
    """APFS-clone src → dst (near-zero cost on macOS), fallback to copy."""
    if dst.exists():
        shutil.rmtree(dst)
    try:
        subprocess.run(["cp", "-Rc", str(src), str(dst)], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        shutil.copytree(src, dst)


# Agent instructions a cloned repo may ship that Claude Code could auto-load and
# ACT ON — `.claude/` (settings, hooks, project memory), `CLAUDE.md`, `.mcp.json`.
# ALWAYS stripped from ANY repo we materialize (reference docs AND cloned
# interface source like hopsworks-api), so a vendored repo's own directives never
# leak into the engineer/researcher. `.git`/`.github` are handled separately
# (docs-only — kept on interface source, where rebuilds may `git pull`).
_AGENT_PLUMBING_DIRS = (".claude",)
_AGENT_PLUMBING_FILES = ("CLAUDE.md", ".mcp.json")
_DOCS_ONLY_STRIP_DIRS = (".git", ".github")


def strip_agent_plumbing(dst: Path) -> None:
    """Recursively remove `.claude/`, `CLAUDE.md`, and `.mcp.json` from `dst`.

    Use after cloning/copying ANY repo so its own agent instructions are never
    auto-loaded as directives by Claude Code. (Functional VCS metadata is left
    alone — see `_strip_git_dir` for the docs-only VCS strip.)
    """
    dst = Path(dst)
    for d in _AGENT_PLUMBING_DIRS:
        for p in dst.rglob(d):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
    for f in _AGENT_PLUMBING_FILES:
        for p in dst.rglob(f):
            if p.is_file():
                p.unlink(missing_ok=True)


def _strip_git_dir(dst: Path) -> None:
    """Docs-only: drop VCS metadata + CI config so the model reads documentation,
    not git internals. (Agent plumbing is stripped by `strip_agent_plumbing`.)"""
    for d in _DOCS_ONLY_STRIP_DIRS:
        for p in dst.rglob(d):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)


def apply(
    spec: str,
    dst_dir: Path,
    share_from: Path | None = None,
    ref: str | None = None,
) -> DocsSetup:
    """Materialize the docs bundle inside ``dst_dir/docs/``.

    Args:
      spec: ``"none"``, a git URL, or a local path (mostly for tests).
      dst_dir: target dir; docs land at ``dst_dir/docs/``.
      share_from: when given and ``share_from/docs/`` exists, the docs are
        APFS-cloned from there instead of fetched anew — used by per-challenge
        engineers in autoresearch (the run-level docs were fetched once).
      ref: a fixed commit/branch/tag to check out when cloning from a URL, so
        the docs are pinned/reproducible (ignored for share/local-path).
    """
    if not spec or spec == "none":
        return DocsSetup(spec="none")

    dst = dst_dir / "docs"

    if share_from is not None and (share_from / "docs").is_dir():
        _clone_tree(share_from / "docs", dst)
    elif _looks_like_url(spec):
        _clone(spec, dst, ref=ref)
    else:
        # Local path (test fixture / pre-downloaded mirror).
        src = Path(spec).expanduser()
        if not src.is_absolute():
            src = (Path.cwd() / src).resolve()
        if not src.is_dir():
            raise ValueError(
                f"docs spec {spec!r} is not a URL and not an existing dir."
            )
        _clone_tree(src, dst)

    strip_agent_plumbing(dst)
    _strip_git_dir(dst)
    files = sorted(
        str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file()
    )
    return DocsSetup(spec=spec, files=files)
