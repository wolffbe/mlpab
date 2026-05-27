"""Skill bundles for Claude Code runs.

A *bundle* is a folder under the top-level `skills/` directory containing
versioned subfolders, each of which holds one or more skill subfolders (each
with a SKILL.md):

    skills/<name>/<version>/<skill-subfolder>/SKILL.md

`<version>` is an integer (0, 1, 2, ...). `banter run --skills <name>`
selects the highest existing version and copies every skill subfolder
into the run's `.claude/skills/` so claude-code discovers them as
project-scoped skills.

The bundle's hash is computed on the fly each run by recursively
sha256'ing the chosen version folder (paths + bytes + version). It isn't
stored anywhere — change a file, the hash changes automatically.

Bundle name "none" is a no-op — the without-skills control.
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path


SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills"


@dataclass
class SkillsSetup:
    name: str
    version: int = 0
    hash: str = ""
    installed: list[str] = field(default_factory=list)


def _bundle_dir(name: str, version: int | None = None) -> Path:
    """Return `skills/<name>/<version>/`. When version is None, picks the
    highest existing version folder. Raises if the bundle or version is absent."""
    bundle = SKILLS_ROOT / name
    if not bundle.is_dir():
        raise ValueError(
            f"Skill bundle {name!r} not found under {SKILLS_ROOT}. "
            f"Available: {available()}."
        )
    versions = [int(p.name) for p in bundle.iterdir() if p.is_dir() and p.name.isdigit()]
    if not versions:
        raise ValueError(
            f"Bundle {name!r} has no version subfolders. Expected "
            f"{bundle}/<version>/<skill>/SKILL.md layout."
        )
    chosen = version if version is not None else max(versions)
    if chosen not in versions:
        raise ValueError(
            f"Skill bundle {name!r} has no version {chosen}. Available: {sorted(versions)}."
        )
    return bundle / str(chosen)


def _compute_hash(variant: Path, version: int) -> str:
    """Recursive sha256 of the version folder's files + version int."""
    h = hashlib.sha256()
    for path in sorted(p for p in variant.rglob("*") if p.is_file()):
        rel = path.relative_to(variant).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    h.update(f"|v={version}".encode())
    return h.hexdigest()[:8]


def available() -> list[str]:
    """Bundles available under testbed/skills/ (at least one version folder)."""
    if not SKILLS_ROOT.exists():
        return ["none"]
    out = ["none"]
    for p in SKILLS_ROOT.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        if any(c.is_dir() and c.name.isdigit() for c in p.iterdir()):
            out.append(p.name)
    return sorted(set(out))


def verify_installed(name: str, version: int | None = None) -> tuple[int, str, Path]:
    """Fail fast if the chosen bundle version isn't usable. Checks:

    - A version subfolder exists.
    - At least one skill subfolder with a SKILL.md file is present.

    Returns (version, hash, version_path). Caller is expected to
    short-circuit when `name == "none"`. When version is None, picks latest.
    """
    variant = _bundle_dir(name, version)
    version = int(variant.name)
    skill_dirs = [p for p in variant.iterdir() if p.is_dir()]
    if not skill_dirs:
        raise ValueError(
            f"Skill bundle {name!r} v{version} has no skill subfolders. "
            f"Expected {variant}/<skill-name>/SKILL.md layout."
        )
    missing_md = [p.name for p in skill_dirs if not (p / "SKILL.md").is_file()]
    if missing_md:
        raise ValueError(
            f"Skill bundle {name!r} v{version}: missing SKILL.md in "
            f"subfolder(s): {', '.join(missing_md)}."
        )
    return version, _compute_hash(variant, version), variant


def apply(name: str, run_dir: Path, version: int | None = None) -> SkillsSetup:
    """Used by `banter run`: copy a skill bundle version's subfolders into
    `run_dir/.claude/skills/`. `name == "none"` is a no-op. When version is
    None, picks the latest version."""
    if name == "none":
        return SkillsSetup(name="none")

    version, hash_, src = verify_installed(name, version)
    dst = run_dir / ".claude" / "skills"
    dst.mkdir(parents=True, exist_ok=True)

    installed: list[str] = []
    for entry in sorted(src.iterdir()):
        if not entry.is_dir():
            continue
        shutil.copytree(entry, dst / entry.name, dirs_exist_ok=True)
        installed.append(entry.name)

    return SkillsSetup(name=name, version=version, hash=hash_, installed=installed)
