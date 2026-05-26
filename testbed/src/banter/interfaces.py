"""Install a chosen (interface, type) into a challenge run.

An *interface* is a vendor or product (e.g. `hopsworks`, `none`). A *type*
is how Claude talks to it (`cli`, `mcp`, `sdk`, or `none`). Variants live
under a nested layout:

    interfaces/<name>/<type>/<version>/
        config.yaml      # repo, ref, install, binary, mcp_servers, ...
        <other files>    # binaries, scripts, etc. — all included in the hash

`<version>` is an integer (0, 1, 2, ...) — banter picks the highest
existing version for a (name, type). The hash of the variant is computed
on the fly each run by recursively sha256'ing the contents combined with
the version int, so it stays off disk.

Prompts are kept separately and constant across variants of the same
(name, type):

    prompts/interfaces/<name>_<type>.md

`setup()` returns the artifacts the runner needs: a prompt fragment to
splice into the task, the CLI binary name (used to count CLI tool calls),
and the MCP server config to write to .mcp.json (if any).
"""
from __future__ import annotations

import hashlib
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


TYPES = ("cli", "mcp", "sdk", "none")
_TESTBED_ROOT = Path(__file__).resolve().parents[2]
INTERFACES_DIR = _TESTBED_ROOT / "interfaces"
PROMPTS_DIR = _TESTBED_ROOT / "prompts" / "interfaces"


@dataclass
class InterfaceSetup:
    name: str
    type: str
    prompt_fragment: str
    version: int            # 0-based variant index from the folder name
    hash: str               # runtime sha256 of the variant folder contents
    cli_binary: str | None = None
    sdk_module: str | None = None
    mcp_servers: dict[str, Any] = field(default_factory=dict)


def _variant_dir(name: str, type_: str) -> Path:
    """Return `interfaces/<name>/<type>/<highest-version>/` or raise."""
    type_dir = INTERFACES_DIR / name / type_
    if not type_dir.is_dir():
        raise ValueError(
            f"No variants for interface {name!r}, type {type_!r}. "
            f"Expected a folder under {type_dir} named with an integer version."
        )
    versions = [int(p.name) for p in type_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not versions:
        raise ValueError(
            f"No version folder for interface {name!r}, type {type_!r} under {type_dir}."
        )
    return type_dir / str(max(versions))


def _compute_hash(variant: Path, version: int) -> str:
    """Recursive sha256 of the variant folder's files + version int."""
    h = hashlib.sha256()
    for path in sorted(p for p in variant.rglob("*") if p.is_file()):
        rel = path.relative_to(variant).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    h.update(f"|v={version}".encode())
    return h.hexdigest()[:8]


def _load_config(variant: Path) -> dict[str, Any]:
    cfg_path = variant / "config.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text()) or {}


def _load_prompt(name: str, type_: str) -> str:
    p = PROMPTS_DIR / name / f"{type_}.md"
    return p.read_text().strip() if p.exists() else ""


def prompt_hash_for(name: str, type_: str) -> str:
    """8-hex sha256 of the prompt file content (empty string -> empty hash)."""
    text = _load_prompt(name, type_)
    if not text:
        return ""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def available() -> dict[str, list[str]]:
    """Return {interface_name: [types...]} by scanning the interfaces folder."""
    out: dict[str, set[str]] = defaultdict(set)
    if not INTERFACES_DIR.exists():
        return {}
    for name_dir in INTERFACES_DIR.iterdir():
        if not name_dir.is_dir():
            continue
        for type_dir in name_dir.iterdir():
            if not type_dir.is_dir() or type_dir.name not in TYPES:
                continue
            # Must contain at least one version folder.
            if any(p.is_dir() and p.name.isdigit() for p in type_dir.iterdir()):
                out[name_dir.name].add(type_dir.name)
    return {k: sorted(v) for k, v in sorted(out.items())}


def variant_for(name: str, type_: str) -> tuple[int, str]:
    """Return (version, hash) of the latest variant for (name, type) without
    performing any setup side effects. Hash is computed on the fly."""
    if type_ not in TYPES:
        raise ValueError(f"Unknown interface type {type_!r}; expected one of {TYPES}")
    variant = _variant_dir(name, type_)
    version = int(variant.name)
    return version, _compute_hash(variant, version)


def setup(name: str, type_: str, run_dir: Path, venv_python: Path) -> InterfaceSetup:
    if type_ not in TYPES:
        raise ValueError(f"Unknown interface type {type_!r}; expected one of {TYPES}")

    variant = _variant_dir(name, type_)
    version = int(variant.name)
    cfg = _load_config(variant)
    prompt_fragment = _load_prompt(name, type_)
    hash_ = _compute_hash(variant, version)

    if name == "none" and type_ == "none":
        return InterfaceSetup(
            name=name,
            type=type_,
            prompt_fragment=prompt_fragment,
            version=version,
            hash=hash_,
        )

    repo = cfg.get("repo")
    install_steps = cfg.get("install") or []
    if repo:
        _git_clone(repo, cfg.get("ref", "main"), run_dir / "interface")
    if install_steps:
        _run_install(install_steps, cwd=run_dir, venv_python=venv_python)

    return InterfaceSetup(
        name=name,
        type=type_,
        prompt_fragment=prompt_fragment,
        version=version,
        hash=hash_,
        cli_binary=cfg.get("binary"),
        # SDK module name is always the interface name when mode is `sdk`.
        sdk_module=name if type_ == "sdk" else None,
        mcp_servers=cfg.get("mcp_servers") or {},
    )


def _git_clone(repo: str, ref: str, target: Path) -> None:
    if target.exists():
        return
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", ref, repo, str(target)],
        check=True,
    )


def _run_install(steps: list[str], cwd: Path, venv_python: Path) -> None:
    import os

    env = os.environ.copy()
    env["PATH"] = f"{venv_python.parent}:{env['PATH']}"
    env["VIRTUAL_ENV"] = str(venv_python.parent.parent)
    for step in steps:
        subprocess.run(step, shell=True, cwd=cwd, env=env, check=True)
