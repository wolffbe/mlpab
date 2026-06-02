"""Per-project skill bundles for Claude Code runs.

Skills mirror interfaces: the base (version 0) bundle is committed in the
project's interface tree, and improved versions are created session-locally
under an autoresearch session (results/autoresearch/<id>/...):

    platforms/<project>/skills/<bundle>/<skill>/SKILL.md          # base (v0)
    <version_root>/skills/<bundle>/v<n>/<skill>/SKILL.md           # session v>0

`banter run --skills <bundle>` copies every skill subfolder of the chosen
version into the run's `.claude/skills/` so Claude Code discovers them as
project skills. Bundle name "none" is the without-skills control.

The bundle's hash is the recursive sha256 of the chosen version folder.
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path


_TESTBED_ROOT = Path(__file__).resolve().parents[2]
PLATFORMS_DIR = _TESTBED_ROOT / "platforms"


@dataclass
class SkillsSetup:
    name: str
    version: int = 0
    hash: str = ""
    installed: list[str] = field(default_factory=list)


def _skills_root(project: str) -> Path:
    """The project's base skills dir: platforms/<project>/skills/."""
    return PLATFORMS_DIR / project / "skills"


def _bundle_dir(project: str, name: str, version: int, version_root: Path | None) -> Path:
    """Resolve a bundle version folder. Version 0 → base (in platforms);
    version > 0 → session-local under version_root."""
    if version and version_root is not None:
        return Path(version_root) / "skills" / name / f"v{version}"
    return _skills_root(project) / name


def _compute_hash(variant: Path, version: int) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in variant.rglob("*") if p.is_file()):
        rel = path.relative_to(variant).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    h.update(f"|v={version}".encode())
    return h.hexdigest()[:8]


def verify_installed(
    project: str, name: str, version: int | None = None, version_root: Path | None = None
) -> tuple[int, str, Path]:
    """Fail fast if the chosen bundle version isn't usable. Returns
    (version, hash, version_path). Caller short-circuits when name == 'none'."""
    chosen = version or 0
    variant = _bundle_dir(project, name, chosen, version_root)
    if not variant.is_dir():
        raise ValueError(
            f"Skill bundle {name!r} v{chosen} not found at {variant} "
            f"(project {project!r})."
        )
    skill_dirs = [p for p in variant.iterdir() if p.is_dir()]
    missing_md = [p.name for p in skill_dirs if not (p / "SKILL.md").is_file()]
    if not skill_dirs:
        raise ValueError(
            f"Skill bundle {name!r} v{chosen} has no skill subfolders "
            f"({variant}/<skill-name>/SKILL.md)."
        )
    if missing_md:
        raise ValueError(
            f"Skill bundle {name!r} v{chosen}: missing SKILL.md in: {', '.join(missing_md)}."
        )
    return chosen, _compute_hash(variant, chosen), variant


def apply(
    project: str,
    name: str,
    run_dir: Path,
    version: int | None = None,
    version_root: Path | None = None,
) -> SkillsSetup:
    """Copy a bundle version's skill subfolders into run_dir/.claude/skills/.
    `name == 'none'` is a no-op."""
    if name == "none":
        return SkillsSetup(name="none")

    chosen, hash_, src = verify_installed(project, name, version, version_root)
    dst = run_dir / ".claude" / "skills"
    dst.mkdir(parents=True, exist_ok=True)

    installed: list[str] = []
    for entry in sorted(src.iterdir()):
        if not entry.is_dir():
            continue
        shutil.copytree(entry, dst / entry.name, dirs_exist_ok=True)
        installed.append(entry.name)
    return SkillsSetup(name=name, version=chosen, hash=hash_, installed=installed)


def skill_names(project: str, name: str, version: int | None = None,
                version_root: Path | None = None) -> list[str]:
    """The names Claude sees (SKILL.md `name:` frontmatter, else folder)."""
    import re
    _, _, variant = verify_installed(project, name, version, version_root)
    out: list[str] = []
    for d in sorted(p for p in variant.iterdir() if p.is_dir()):
        nm = d.name
        md = d / "SKILL.md"
        if md.is_file():
            m = re.search(r"(?m)^name:\s*(.+?)\s*$", md.read_text()[:2000])
            if m:
                nm = m.group(1).strip().strip("\"'")
        out.append(nm)
    return out
