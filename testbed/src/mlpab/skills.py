"""Per-platform skill bundles for Claude Code runs.

Skills mirror interfaces: the manifest carries a REFERENCE, the content is
fetched on demand. Each platform's skills.yaml defines ONE bundle — the vendor's
official skills, always used when skills are on — as a FLAT pointer (no named-
bundle key):

    configs/platforms/<project>/skills.yaml
        repo: <git url>          # repo-backed: fetched at runtime into the
        ref: <FIXED commit sha>  # gitignored build home (pinned — never
        path: <subdir in repo>   # re-pulled; bump the sha to advance)
        # — OR — local committed content:
        path: <dir relative to testbed root>

This single bundle is labelled `official`. A platform with no skills.yaml falls
back to default-dir bundles at configs/platforms/<project>/skills/<name>.

Repo-backed bundles materialize into build/<project>/skills/<bundle>/ on first
use (verify/apply trigger the fetch — preflight auto-setup, like interface
builds): a shallow clone at the PINNED ref, then every `<skill>/SKILL.md` dir
under `path:` is collected — one category level is flattened automatically
(e.g. hopsworks-api's skills/{data,hops,ml,platform}/<skill>). To re-fetch
(after bumping `ref:`), delete the bundle's build dir.

A run with `skills: <bundle>` copies every skill subfolder of the resolved
bundle into the run's `.claude/skills/` so Claude Code discovers them as
project skills. Bundle name "none" is the without-skills control.

The bundle's hash is the recursive sha256 of the resolved bundle folder.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_TESTBED_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = _TESTBED_ROOT / "configs" / "platforms"
BUILD_DIR = _TESTBED_ROOT / "build"


@dataclass
class SkillsSetup:
    name: str
    hash: str = ""
    installed: list[str] = field(default_factory=list)


def _skills_manifest(project: str) -> dict:
    """The platform's skills.yaml pointer manifest ({} when absent)."""
    p = CONFIGS_DIR / project / "skills.yaml"
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def _skills_root(project: str) -> Path:
    """The project's default base skills dir, inside its config folder:
    configs/platforms/<project>/skills/. (A bundle may point elsewhere via
    skills.yaml — e.g. a repo-backed `official` bundle.)"""
    return CONFIGS_DIR / project / "skills"


_DEFAULT_BUNDLE = "official"


def _manifest_entry(project: str) -> dict:
    """The single (official) skill bundle declared by skills.yaml as a flat
    {repo, ref, path} or {path}; {} when the platform has no skills.yaml."""
    m = _skills_manifest(project)
    return m if (m.get("repo") or m.get("path")) else {}


def bundle_names(project: str) -> list[str]:
    """Bundle names: the implicit `official` bundle when skills.yaml defines one,
    plus any default-dir subfolders (configs/platforms/<p>/skills/<name>)."""
    names = set()
    if _manifest_entry(project):
        names.add(_DEFAULT_BUNDLE)
    root = _skills_root(project)
    if root.is_dir():
        names.update(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    return sorted(names)


def _is_skill_dir(p: Path) -> bool:
    return p.is_dir() and (p / "SKILL.md").is_file()


def _skill_dirs_under(sub: Path) -> list[Path]:
    """Skill dirs to vendor from a bundle's `path:` root `sub`: `sub` itself when
    it IS a single skill dir (e.g. Microsoft's skills/azure-machine-learning),
    else its child skill dirs, flattening one category level (subdirs without
    their own SKILL.md, e.g. hopsworks data/hops/ml)."""
    if _is_skill_dir(sub):
        return [sub]
    out: list[Path] = []
    for child in sorted(sub.iterdir()):
        if _is_skill_dir(child):
            out.append(child)
        elif child.is_dir():
            out += [g for g in sorted(child.iterdir()) if _is_skill_dir(g)]
    return out


def _fetch_repo_bundle(project: str, name: str, entry: dict) -> Path:
    """Materialize a repo-backed bundle into build/<project>/skills/<name>/.

    Idempotent: an already-materialized bundle is reused as-is (the ref is
    FIXED, so the content cannot drift; delete the dir to force a re-fetch).
    The clone is shallow at exactly the pinned ref — same pattern as
    `interfaces.build` — and only the `<skill>/SKILL.md` dirs under `path:`
    are kept; one category level (subdirs without their own SKILL.md) is
    flattened automatically.
    """
    home = BUILD_DIR / project / "skills" / name
    if home.is_dir() and any(_is_skill_dir(p) for p in home.iterdir()):
        return home

    repo, ref = entry["repo"], str(entry.get("ref", "main"))
    if home.exists():
        shutil.rmtree(home)
    src = home / ".src"
    src.mkdir(parents=True)
    try:
        subprocess.run(["git", "-C", str(src), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(src), "remote", "add", "origin", repo], check=True)
        subprocess.run(
            ["git", "-C", str(src), "fetch", "-q", "--depth", "1", "origin", ref], check=True
        )
        subprocess.run(["git", "-C", str(src), "checkout", "-q", "FETCH_HEAD"], check=True)
    except subprocess.CalledProcessError as e:
        shutil.rmtree(home, ignore_errors=True)
        raise RuntimeError(
            f"fetch of skill bundle {name!r} for {project!r} failed ({repo} @ {ref}): {e}"
        ) from e

    sub = src / entry["path"] if entry.get("path") else src
    if not sub.is_dir():
        shutil.rmtree(home, ignore_errors=True)
        raise RuntimeError(
            f"skill bundle {name!r}: path {entry.get('path')!r} not found in {repo} @ {ref}"
        )
    copied = 0
    for child in _skill_dirs_under(sub):
        shutil.copytree(child, home / child.name)
        copied += 1
    # Drop the checkout: the build home keeps ONLY the flattened skill dirs.
    # (rmtree must handle read-only .git objects.)
    for root, dirs, files in os.walk(src):
        for n in files + dirs:
            try:
                os.chmod(os.path.join(root, n), 0o700)
            except OSError:
                pass
    shutil.rmtree(src, ignore_errors=True)
    if not copied:
        shutil.rmtree(home, ignore_errors=True)
        raise RuntimeError(
            f"skill bundle {name!r}: no <skill>/SKILL.md dirs under "
            f"{entry.get('path')!r} in {repo} @ {ref}"
        )
    return home


def _bundle_dir(project: str, name: str) -> Path:
    """Resolve a bundle folder. The implicit `official` bundle comes from
    skills.yaml (repo-backed → fetched into the build home on first use; `path:`
    → relative to the testbed root); any other name resolves to the default
    configs/platforms/<project>/skills/<name>."""
    entry = _manifest_entry(project) if name == _DEFAULT_BUNDLE else {}
    if entry.get("repo"):
        return _fetch_repo_bundle(project, name, entry)
    if entry.get("path"):
        return _TESTBED_ROOT / entry["path"]
    return _skills_root(project) / name


def _compute_hash(variant: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in variant.rglob("*") if p.is_file()):
        rel = path.relative_to(variant).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:8]


def verify_installed(project: str, name: str) -> tuple[str, Path]:
    """Fail fast if the chosen bundle isn't usable (fetching a repo-backed
    bundle on first use). Returns (hash, bundle_path). Caller short-circuits
    when name == 'none'."""
    variant = _bundle_dir(project, name)
    if not variant.is_dir():
        raise ValueError(f"Skill bundle {name!r} not found at {variant} (project {project!r}).")
    skill_dirs = [p for p in variant.iterdir() if p.is_dir()]
    missing_md = [p.name for p in skill_dirs if not (p / "SKILL.md").is_file()]
    if not skill_dirs:
        raise ValueError(
            f"Skill bundle {name!r} has no skill subfolders ({variant}/<skill-name>/SKILL.md)."
        )
    if missing_md:
        raise ValueError(f"Skill bundle {name!r}: missing SKILL.md in: {', '.join(missing_md)}.")
    return _compute_hash(variant), variant


def apply(project: str, name: str, run_dir: Path) -> SkillsSetup:
    """Copy a bundle's skill subfolders into run_dir/.claude/skills/.
    `name == 'none'` is a no-op."""
    if name == "none":
        return SkillsSetup(name="none")

    hash_, src = verify_installed(project, name)
    dst = run_dir / ".claude" / "skills"
    dst.mkdir(parents=True, exist_ok=True)

    installed: list[str] = []
    for entry in sorted(src.iterdir()):
        if not entry.is_dir():
            continue
        shutil.copytree(entry, dst / entry.name, dirs_exist_ok=True)
        installed.append(entry.name)
    return SkillsSetup(name=name, hash=hash_, installed=installed)


def skill_names(project: str, name: str) -> list[str]:
    """The names Claude sees (SKILL.md `name:` frontmatter, else folder)."""
    import re

    _, variant = verify_installed(project, name)
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
