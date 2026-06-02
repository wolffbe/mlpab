"""Reference docs cloned from a git URL into the run.

Configured in autoresearch / benchmark / `banter run` via the `docs:` field
(or `--docs` flag) — value is a git URL, e.g.:

    docs: https://github.com/logicalclocks/hops-docs.git

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


@dataclass
class DocsSetup:
    spec: str               # original config value: URL, path, or "none"
    files: list[str] = field(default_factory=list)


def _looks_like_url(spec: str) -> bool:
    return spec.startswith(("http://", "https://", "git@", "ssh://", "git://"))


def _clone(url: str, dst: Path) -> None:
    """Shallow git clone (depth 1) into dst. Removes dst if it exists."""
    if dst.exists():
        shutil.rmtree(dst)
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


def _strip_git_dir(dst: Path) -> None:
    """Remove `.git/` so the model isn't reading git metadata as "docs"."""
    g = dst / ".git"
    if g.is_dir():
        shutil.rmtree(g)


def apply(
    spec: str,
    dst_dir: Path,
    share_from: Path | None = None,
) -> DocsSetup:
    """Materialize the docs bundle inside ``dst_dir/docs/``.

    Args:
      spec: ``"none"``, a git URL, or a local path (mostly for tests).
      dst_dir: target dir; docs land at ``dst_dir/docs/``.
      share_from: when given and ``share_from/docs/`` exists, the docs are
        APFS-cloned from there instead of fetched anew — used by per-challenge
        engineers in autoresearch (the run-level docs were fetched once).
    """
    if not spec or spec == "none":
        return DocsSetup(spec="none")

    dst = dst_dir / "docs"

    if share_from is not None and (share_from / "docs").is_dir():
        _clone_tree(share_from / "docs", dst)
    elif _looks_like_url(spec):
        _clone(spec, dst)
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

    _strip_git_dir(dst)
    files = sorted(
        str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file()
    )
    return DocsSetup(spec=spec, files=files)
