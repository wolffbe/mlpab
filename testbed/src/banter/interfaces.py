"""Resolve a chosen (interface, type, version) for a challenge run.

The single source of truth for an interface is its install manifest:

    configs/interfaces/<name>/<type>.yaml

The manifest carries everything the testbed needs:

    repo, ref, install, auth_command         # one-time install (banter install)
    binary, runtime_install, mcp_servers     # runtime defaults (banter run)
    versions:                                # autoresearch-managed
      0: { prompt: "..." }                   # base
      1: { prompt: "...", install: [...] }   # per-version overrides

Binary artifacts live separately and only exist when needed:

    interfaces/<name>/<type>/<version>/      # e.g. .../hopsworks/cli/0/hops

For SDK/MCP interfaces no binary is needed — the version folder may be absent.

Version 0 is the base, set up by `banter install`. Versions 1+ are created by
autoresearch (appended to `versions:` in the manifest, with a copy of the
binary if applicable).
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

# configs/interfaces/<name>/<type>.yaml — install manifests
IFACE_CONFIGS_DIR = _TESTBED_ROOT / "configs" / "interfaces"
# interfaces/<name>/<type>/<version>/ — built binary artifacts only
IFACE_BINS_DIR = _TESTBED_ROOT / "interfaces"


@dataclass
class InterfaceSetup:
    name: str
    type: str
    prompt_fragment: str
    version: int
    hash: str
    cli_binary: str | None = None
    sdk_module: str | None = None
    mcp_servers: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def manifest_path(name: str, type_: str) -> Path:
    return IFACE_CONFIGS_DIR / name / f"{type_}.yaml"


def bin_dir(name: str, type_: str, version: int) -> Path:
    return IFACE_BINS_DIR / name / type_ / str(version)


def load_manifest(name: str, type_: str) -> dict[str, Any]:
    p = manifest_path(name, type_)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def _version_entries(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Normalise manifest['versions'] to {int: dict}."""
    raw = manifest.get("versions") or {}
    out: dict[int, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[int(k)] = v or {}
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            try:
                out[int(entry.get("version", 0))] = entry
            except (TypeError, ValueError):
                continue
    return out


def available_versions(name: str, type_: str) -> list[int]:
    return sorted(_version_entries(load_manifest(name, type_)).keys())


def _resolved_version_config(
    manifest: dict[str, Any], version: int
) -> dict[str, Any]:
    """Merge base manifest defaults with per-version overrides."""
    ver_entries = _version_entries(manifest)
    ver = ver_entries.get(version) or {}
    merged: dict[str, Any] = {
        "binary": manifest.get("binary"),
        "runtime_install": manifest.get("runtime_install") or [],
        "mcp_servers": manifest.get("mcp_servers") or {},
        "prompt": ver.get("prompt"),
    }
    # Per-version overrides (apply on top of base defaults)
    for key in ("binary", "runtime_install", "mcp_servers"):
        if key in ver:
            merged[key] = ver[key]
    return merged


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _auto_prompt(name: str, type_: str, binary: str | None) -> str:
    """Generate a sensible default prompt when the manifest doesn't supply one."""
    cap = name.capitalize()
    if type_ == "cli" and binary:
        return (
            f"The {cap} `{binary}` CLI is installed and authenticated. "
            f"Use `{binary} <subcommand>` for all {cap} operations."
        )
    if type_ == "sdk":
        return f"The {cap} Python SDK is installed. Import and use it for all {cap} operations."
    if type_ == "mcp":
        return f"You have access to the {cap} MCP server. Use the provided MCP tools for all {cap} operations."
    return ""


def _prompt_for(name: str, type_: str, manifest: dict[str, Any], version: int) -> str:
    cfg = _resolved_version_config(manifest, version)
    text = cfg.get("prompt")
    if text:
        return text.strip()
    return _auto_prompt(name, type_, cfg.get("binary"))


def prompt_hash_for(name: str, type_: str, version: int | None = None) -> str:
    if name == "none" and type_ == "none":
        return ""
    manifest = load_manifest(name, type_)
    versions = sorted(_version_entries(manifest).keys())
    if not versions:
        return ""
    chosen = version if version is not None else max(versions)
    text = _prompt_for(name, type_, manifest, chosen)
    if not text:
        return ""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _compute_hash(name: str, type_: str, version: int) -> str:
    """sha256 over manifest bytes + binary folder bytes + version int."""
    h = hashlib.sha256()
    mp = manifest_path(name, type_)
    if mp.exists():
        h.update(mp.read_bytes())
        h.update(b"\0")
    bd = bin_dir(name, type_, version)
    if bd.is_dir():
        for path in sorted(p for p in bd.rglob("*") if p.is_file()):
            rel = path.relative_to(bd).as_posix()
            h.update(rel.encode())
            h.update(b"\0")
            h.update(path.read_bytes())
            h.update(b"\0")
    h.update(f"|v={version}".encode())
    return h.hexdigest()[:8]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def available() -> dict[str, list[str]]:
    """Return {interface_name: [types...]} by scanning manifest YAML files."""
    out: dict[str, set[str]] = defaultdict(set)
    if not IFACE_CONFIGS_DIR.exists():
        return {}
    for name_dir in IFACE_CONFIGS_DIR.iterdir():
        if not name_dir.is_dir():
            continue
        for yaml_file in name_dir.iterdir():
            if not yaml_file.is_file() or yaml_file.suffix != ".yaml":
                continue
            type_ = yaml_file.stem
            if type_ not in TYPES:
                continue
            manifest = yaml.safe_load(yaml_file.read_text()) or {}
            if _version_entries(manifest):
                out[name_dir.name].add(type_)
    return {k: sorted(v) for k, v in sorted(out.items())}


def variant_for(name: str, type_: str, version: int | None = None) -> tuple[int, str]:
    """Return (version, hash). Picks latest version when version is None."""
    if name == "none" and type_ == "none":
        return 0, ""
    if type_ not in TYPES:
        raise ValueError(f"Unknown interface type {type_!r}; expected one of {TYPES}")
    manifest = load_manifest(name, type_)
    versions = sorted(_version_entries(manifest).keys())
    if not versions:
        raise ValueError(
            f"No versions for {name!r}/{type_!r} in {manifest_path(name, type_)}. "
            f"Run: banter install configs/interfaces/{name}/{type_}.yaml"
        )
    chosen = version if version is not None else max(versions)
    if chosen not in versions:
        raise ValueError(
            f"Interface {name!r}/{type_!r} has no version {chosen}. Available: {versions}."
        )
    return chosen, _compute_hash(name, type_, chosen)


def setup(
    name: str,
    type_: str,
    run_dir: Path,
    venv_python: Path,
    version: int | None = None,
) -> InterfaceSetup:
    if type_ not in TYPES:
        raise ValueError(f"Unknown interface type {type_!r}; expected one of {TYPES}")

    if name == "none" and type_ == "none":
        return InterfaceSetup(name="none", type="none", prompt_fragment="", version=0, hash="")

    manifest = load_manifest(name, type_)
    versions = sorted(_version_entries(manifest).keys())
    if not versions:
        raise ValueError(
            f"Interface {name!r}/{type_!r} has no versions. "
            f"Run: banter install configs/interfaces/{name}/{type_}.yaml"
        )
    chosen = version if version is not None else max(versions)
    if chosen not in versions:
        raise ValueError(
            f"Interface {name!r}/{type_!r} has no version {chosen}. Available: {versions}."
        )

    cfg = _resolved_version_config(manifest, chosen)
    binary = cfg.get("binary")
    runtime_install = cfg.get("runtime_install") or []
    mcp_servers = cfg.get("mcp_servers") or {}
    prompt_fragment = _prompt_for(name, type_, manifest, chosen)
    hash_ = _compute_hash(name, type_, chosen)
    bins = bin_dir(name, type_, chosen)

    # Guard: if runtime steps reference $INTERFACE_DIR (pre-built binary), it must exist.
    if binary and runtime_install and any("$INTERFACE_DIR" in s for s in runtime_install):
        if not (bins / binary).exists():
            raise RuntimeError(
                f"Interface {name!r}/{type_!r} v{chosen} binary '{binary}' not found at "
                f"{bins / binary}. Run: banter install configs/interfaces/{name}/{type_}.yaml"
            )

    if runtime_install:
        _run_install(runtime_install, cwd=run_dir, venv_python=venv_python, interface_dir=bins)

    return InterfaceSetup(
        name=name,
        type=type_,
        prompt_fragment=prompt_fragment,
        version=chosen,
        hash=hash_,
        cli_binary=binary,
        sdk_module=name if type_ == "sdk" else None,
        mcp_servers=mcp_servers,
    )


def _run_install(
    steps: list[str],
    cwd: Path,
    venv_python: Path,
    interface_dir: Path | None = None,
) -> None:
    import os

    env = os.environ.copy()
    env["PATH"] = f"{venv_python.parent}:{env['PATH']}"
    env["VIRTUAL_ENV"] = str(venv_python.parent.parent)
    if interface_dir is not None:
        env["INTERFACE_DIR"] = str(interface_dir)
    pip_cache = _TESTBED_ROOT / "cache" / "pip"
    pip_cache.mkdir(parents=True, exist_ok=True)
    env["PIP_CACHE_DIR"] = str(pip_cache)
    for step in steps:
        subprocess.run(step, shell=True, cwd=cwd, env=env, check=True)
