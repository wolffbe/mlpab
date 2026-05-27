"""Resolve a chosen (interface, type, version) for an engineer run.

Two locations, with one job each:

    configs/interfaces/<name>/<type>.yaml   — CONFIG: where the interface lives,
                                              how to build it, how to run it,
                                              which credential keys it needs, and
                                              the base (version 0) prompt.

    interfaces/<name>/<type>/               — BINARY: the built artifact only
                                              (e.g. interfaces/hopsworks/cli/hops).
                                              SDK interfaces build nothing (they
                                              pip-install per run) and so have no
                                              folder here. `none` has no folder.

The config manifest carries NO versions. Its `prompt:` is version 0 — the base.
Improved versions are created and live INSIDE an autoresearch session:

    <version_root>/interfaces/<name>/<type>/v<n>/version.yaml

where `<version_root>` is the autoresearch session directory. A version.yaml
overrides the base manifest (a refined `prompt:`, and optionally `binary`,
`runtime_install`, or `mcp_servers`); it may ship its own binary copy, else the
base binary is reused. Version 0 always refers to the base manifest and needs no
folder anywhere.

`banter run` selects a version with `--interface-version <n>` plus
`--version-root <dir>` (the session dir). Without a version-root only v0 (base)
is available.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


TYPES = ("cli", "mcp", "sdk", "none")
_TESTBED_ROOT = Path(__file__).resolve().parents[2]

# configs/interfaces/<name>/<type>.yaml — config manifests
IFACE_CONFIGS_DIR = _TESTBED_ROOT / "configs" / "interfaces"
# interfaces/<name>/<type>/ — built binary artifacts only
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
    keys: dict[str, str] = field(default_factory=dict)


@dataclass
class InterfaceStatus:
    """Outcome of an interface preflight check."""
    name: str
    type: str
    ok: bool
    installed: bool
    authenticated: bool
    missing_keys: list[str] = field(default_factory=list)
    reason: str = ""
    fix_command: str = ""

    @property
    def message(self) -> str:
        if self.ok:
            return f"{self.name}/{self.type}: ready"
        msg = f"{self.name}/{self.type}: {self.reason}"
        if self.fix_command:
            msg += f"\n  → Run: {self.fix_command}"
        return msg


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def manifest_path(name: str, type_: str) -> Path:
    return IFACE_CONFIGS_DIR / name / f"{type_}.yaml"


def name_type_from_config(config_path: str | Path) -> tuple[str, str]:
    """Infer (name, type) from an interface config path.

    Lets autoresearch/benchmark configs reference an interface by its config
    file (e.g. `configs/interfaces/hopsworks/cli.yaml`) instead of name+mode.
    """
    parts = Path(config_path).parts
    try:
        idx = list(parts).index("interfaces")
        name = parts[idx + 1]
        type_ = Path(parts[idx + 2]).stem
    except (ValueError, IndexError):
        raise ValueError(
            f"Cannot infer interface name/type from {config_path!r}. "
            "Expected a path like configs/interfaces/<name>/<type>.yaml"
        )
    if type_ not in TYPES:
        raise ValueError(f"Unknown interface type {type_!r} from {config_path!r}")
    return name, type_


def bin_dir(name: str, type_: str) -> Path:
    """Directory holding the built binary artifact (no version subfolder)."""
    return IFACE_BINS_DIR / name / type_


def version_dir(version_root: Path, name: str, type_: str, version: int) -> Path:
    """Session-local directory for an improved interface version."""
    return Path(version_root) / "interfaces" / name / type_ / f"v{version}"


def load_manifest(name: str, type_: str) -> dict[str, Any]:
    p = manifest_path(name, type_)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


# ---------------------------------------------------------------------------
# Version override (session-local)
# ---------------------------------------------------------------------------


def _load_version_override(
    version_root: Path | None, name: str, type_: str, version: int
) -> dict[str, Any]:
    """Read a session-local version.yaml, or {} for base (v0) / when absent."""
    if not version or version_root is None:
        return {}
    vp = version_dir(version_root, name, type_, version) / "version.yaml"
    if not vp.exists():
        return {}
    return yaml.safe_load(vp.read_text()) or {}


def _resolved_config(
    name: str, type_: str, version: int, version_root: Path | None
) -> dict[str, Any]:
    """Merge base manifest defaults with a session-local version override."""
    manifest = load_manifest(name, type_)
    override = _load_version_override(version_root, name, type_, version)
    merged: dict[str, Any] = {
        "binary": manifest.get("binary"),
        "runtime_install": manifest.get("runtime_install") or [],
        "mcp_servers": manifest.get("mcp_servers") or {},
        "prompt": manifest.get("prompt"),
    }
    for key in ("binary", "runtime_install", "mcp_servers", "prompt"):
        if key in override:
            merged[key] = override[key]
    return merged


def _interface_dir_for(
    name: str, type_: str, version: int, version_root: Path | None, binary: str | None
) -> Path:
    """The directory to expose as $INTERFACE_DIR — where the binary to use lives.

    A session version that ships its own binary copy uses its own folder;
    otherwise the base binary dir is reused.
    """
    if binary and version and version_root is not None:
        vd = version_dir(version_root, name, type_, version)
        if (vd / binary).exists():
            return vd
    return bin_dir(name, type_)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _auto_prompt(name: str, type_: str, binary: str | None) -> str:
    """Generate a sensible default prompt when the config doesn't supply one."""
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


def _prompt_for(name: str, type_: str, version: int, version_root: Path | None) -> str:
    cfg = _resolved_config(name, type_, version, version_root)
    text = cfg.get("prompt")
    if text:
        return text.strip()
    return _auto_prompt(name, type_, cfg.get("binary"))


def prompt_hash_for(
    name: str, type_: str, version: int | None = None, version_root: Path | None = None
) -> str:
    if name == "none" and type_ == "none":
        return ""
    if not load_manifest(name, type_):
        return ""
    chosen = version or 0
    text = _prompt_for(name, type_, chosen, version_root)
    if not text:
        return ""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Credential keys (declared + stored in the config manifest)
# ---------------------------------------------------------------------------


def keys_for(name: str, type_: str) -> dict[str, str]:
    """Return the manifest's declared credential keys as {name: value}.

    Accepts both a mapping (`keys: {NAME: value}`) and a list
    (`keys: [{name: NAME, value: ...}]`). Missing values normalise to "".
    """
    if name == "none" and type_ == "none":
        return {}
    raw = load_manifest(name, type_).get("keys") or {}
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[str(k)] = "" if v is None else str(v)
    elif isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict) and entry.get("name"):
                v = entry.get("value")
                out[str(entry["name"])] = "" if v is None else str(v)
    return out


def _resolved_keys(name: str, type_: str, env: dict[str, str] | None = None) -> dict[str, str]:
    """Key values to inject into the engineer env: manifest value, else env."""
    base = env if env is not None else os.environ
    out: dict[str, str] = {}
    for k, v in keys_for(name, type_).items():
        out[k] = v or base.get(k, "")
    return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _compute_hash(name: str, type_: str, version: int, version_root: Path | None) -> str:
    """sha256 over manifest bytes + version override bytes + binary bytes + version."""
    h = hashlib.sha256()
    mp = manifest_path(name, type_)
    if mp.exists():
        h.update(mp.read_bytes())
        h.update(b"\0")
    if version and version_root is not None:
        vp = version_dir(version_root, name, type_, version) / "version.yaml"
        if vp.exists():
            h.update(vp.read_bytes())
            h.update(b"\0")
    cfg = _resolved_config(name, type_, version, version_root)
    binary = cfg.get("binary")
    if binary:
        bpath = _interface_dir_for(name, type_, version, version_root, binary) / binary
        if bpath.is_file():
            h.update(bpath.read_bytes())
            h.update(b"\0")
    h.update(f"|v={version}".encode())
    return h.hexdigest()[:8]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def available() -> dict[str, list[str]]:
    """Return {interface_name: [types...]} by scanning config manifests."""
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
            if isinstance(manifest, dict) and manifest:
                out[name_dir.name].add(type_)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _check_known(name: str, type_: str, version: int | None, version_root: Path | None) -> int:
    """Validate type/version and return the chosen version int. Raises ValueError."""
    if type_ not in TYPES:
        raise ValueError(f"Unknown interface type {type_!r}; expected one of {TYPES}")
    manifest = load_manifest(name, type_)
    if not manifest:
        raise ValueError(
            f"No config for {name!r}/{type_!r} at {manifest_path(name, type_)}. "
            f"Create it (configs/interfaces/{name}/{type_}.yaml) — preflight builds the binary."
        )
    chosen = version or 0
    if chosen and version_root is None:
        raise ValueError(
            f"Interface {name!r}/{type_!r} v{chosen} requires --version-root "
            f"(versions live inside an autoresearch session)."
        )
    if chosen and not (version_dir(version_root, name, type_, chosen) / "version.yaml").exists():
        raise ValueError(
            f"Interface {name!r}/{type_!r} has no version {chosen} under "
            f"{version_dir(version_root, name, type_, chosen)}."
        )
    return chosen


def variant_for(
    name: str,
    type_: str,
    version: int | None = None,
    version_root: Path | None = None,
) -> tuple[int, str]:
    """Return (version, hash). version None → 0 (base manifest)."""
    if name == "none" and type_ == "none":
        return 0, ""
    chosen = _check_known(name, type_, version, version_root)
    return chosen, _compute_hash(name, type_, chosen, version_root)


def setup(
    name: str,
    type_: str,
    run_dir: Path,
    venv_python: Path,
    version: int | None = None,
    version_root: Path | None = None,
) -> InterfaceSetup:
    if type_ not in TYPES:
        raise ValueError(f"Unknown interface type {type_!r}; expected one of {TYPES}")

    if name == "none" and type_ == "none":
        return InterfaceSetup(name="none", type="none", prompt_fragment="", version=0, hash="")

    chosen = _check_known(name, type_, version, version_root)
    cfg = _resolved_config(name, type_, chosen, version_root)
    binary = cfg.get("binary")
    runtime_install = cfg.get("runtime_install") or []
    mcp_servers = cfg.get("mcp_servers") or {}
    prompt_fragment = _prompt_for(name, type_, chosen, version_root)
    hash_ = _compute_hash(name, type_, chosen, version_root)
    interface_dir = _interface_dir_for(name, type_, chosen, version_root, binary)

    # Guard: if runtime steps reference $INTERFACE_DIR (pre-built binary), it must
    # exist. Preflight builds it before we get here; this is a backstop.
    if binary and runtime_install and any("$INTERFACE_DIR" in s for s in runtime_install):
        if not (interface_dir / binary).exists():
            raise RuntimeError(
                f"Interface {name!r}/{type_!r} v{chosen} binary '{binary}' not found at "
                f"{interface_dir / binary}. Preflight should have built it; "
                f"check configs/interfaces/{name}/{type_}.yaml install steps."
            )

    if runtime_install:
        _run_install(runtime_install, cwd=run_dir, venv_python=venv_python, interface_dir=interface_dir)

    return InterfaceSetup(
        name=name,
        type=type_,
        prompt_fragment=prompt_fragment,
        version=chosen,
        hash=hash_,
        # `binary` may be a built artifact for any type (e.g. an SDK wheel); only
        # a CLI's binary is an invokable command worth tracking as cli_calls.
        cli_binary=binary if type_ == "cli" else None,
        sdk_module=name if type_ == "sdk" else None,
        mcp_servers=mcp_servers,
        keys=_resolved_keys(name, type_),
    )


def preflight(
    name: str,
    type_: str,
    version: int | None = None,
    version_root: Path | None = None,
    *,
    check_login: bool = True,
    auto_build: bool = True,
    timeout_s: int = 120,
    env: dict[str, str] | None = None,
) -> InterfaceStatus:
    """Build (if needed), then verify an interface is installed + logged in + healthy.

    Interfaces declared in a config are binaries that get built, set up, and
    authenticated at preflight — no manual `banter install` step. Concretely:

      * Build — if the binary is missing, run the config's `install:` steps
        (deterministic, no AI). Disable with auto_build=False.
      * Login — the `auth_command` exits 0 when run non-interactively with the
        declared keys in the environment; for interfaces without an
        `auth_command`, login is satisfied only when every declared key is set.
      * Test — the `test_command` exits 0.
    """
    config_fix = f"check configs/interfaces/{name}/{type_}.yaml (build/install steps)"
    setup_fix = "make setup  (or: banter setup " + f"configs/interfaces/{name}/{type_}.yaml)"

    if name == "none" and type_ == "none":
        return InterfaceStatus(name, type_, ok=True, installed=True, authenticated=True)
    if type_ not in TYPES:
        return InterfaceStatus(
            name, type_, ok=False, installed=False, authenticated=False,
            reason=f"unknown interface type {type_!r}",
        )

    # 1) installed? Build the base binary on demand if it's missing.
    try:
        chosen = _check_known(name, type_, version, version_root)
    except ValueError as e:
        return InterfaceStatus(
            name, type_, ok=False, installed=False, authenticated=False,
            reason=str(e), fix_command=config_fix,
        )
    cfg = _resolved_config(name, type_, chosen, version_root)
    binary = cfg.get("binary")
    runtime_install = cfg.get("runtime_install") or []
    if binary and runtime_install and any("$INTERFACE_DIR" in s for s in runtime_install):
        bpath = _interface_dir_for(name, type_, chosen, version_root, binary) / binary
        if not bpath.exists() and auto_build and load_manifest(name, type_).get("install"):
            try:
                build(name, type_)
            except Exception as e:  # build is shell-out heavy; surface failures
                return InterfaceStatus(
                    name, type_, ok=False, installed=False, authenticated=False,
                    reason=f"build failed: {e}", fix_command=config_fix,
                )
        if not bpath.exists():
            return InterfaceStatus(
                name, type_, ok=False, installed=False, authenticated=False,
                reason=f"binary {binary!r} missing at {bpath}", fix_command=config_fix,
            )

    if not check_login:
        return InterfaceStatus(name, type_, ok=True, installed=True, authenticated=True)

    # 2) login / keys
    keys = keys_for(name, type_)
    base_env = dict(env) if env is not None else dict(os.environ)
    merged_env = dict(base_env)
    missing = []
    for k, declared in keys.items():
        val = declared or base_env.get(k, "")
        if val:
            merged_env[k] = val
        else:
            missing.append(k)
    # Make a freshly-built (not-yet-copied-into-venv) binary callable by auth_command.
    bd = bin_dir(name, type_)
    if bd.is_dir():
        merged_env["PATH"] = f"{bd}{os.pathsep}{merged_env.get('PATH', '')}"

    manifest = load_manifest(name, type_)
    auth_command = manifest.get("auth_command")
    if auth_command:
        if not _run_check(auth_command, merged_env, timeout_s):
            return InterfaceStatus(
                name, type_, ok=False, installed=True, authenticated=False,
                missing_keys=missing,
                reason=f"login check failed (`{auth_command}` returned non-zero or timed out)",
                fix_command=setup_fix,
            )
    elif keys and missing:
        # No auth_command — fall back to requiring all declared keys be present.
        return InterfaceStatus(
            name, type_, ok=False, installed=True, authenticated=False,
            missing_keys=missing,
            reason=f"missing credential key(s): {', '.join(missing)}",
            fix_command=setup_fix,
        )

    # 3) test — deterministic "does it actually run?" check (no AI involved).
    test_command = manifest.get("test_command")
    if test_command and not _run_check(test_command, merged_env, timeout_s):
        return InterfaceStatus(
            name, type_, ok=False, installed=True, authenticated=True,
            missing_keys=missing,
            reason=f"interface did not run reliably (`{test_command}` returned non-zero or timed out)",
            fix_command=config_fix,
        )

    return InterfaceStatus(
        name, type_, ok=True, installed=True, authenticated=True, missing_keys=missing,
    )


def _run_check(command: str, env: dict[str, str], timeout_s: int) -> bool:
    """Run a declared check command non-interactively. True iff it exits 0."""
    try:
        proc = subprocess.run(
            command, shell=True, env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def build(name: str, type_: str) -> Path:
    """Build the interface's binary from its config (deterministic, no AI).

    Clones/updates the source repo and runs the config's `install:` steps with
    $INTERFACE_DIR pointing at the binary dir (interfaces/<name>/<type>/, no
    version subfolder). Idempotent. Returns the binary dir. Run automatically by
    preflight when the binary is missing.
    """
    if name == "none" and type_ == "none":
        return bin_dir("none", "none")
    manifest = load_manifest(name, type_)
    if not manifest:
        raise FileNotFoundError(f"No config at {manifest_path(name, type_)}")

    repo = manifest.get("repo")
    ref = manifest.get("ref", "main")
    install_steps = manifest.get("install") or []
    bins = bin_dir(name, type_)
    bins.mkdir(parents=True, exist_ok=True)
    src = _TESTBED_ROOT / "cache" / "interfaces" / name / type_ / "src"
    venv_bin = _TESTBED_ROOT / ".venv" / "bin"

    if repo:
        if (src / ".git").exists():
            subprocess.run(["git", "-C", str(src), "pull"], check=True)
        else:
            src.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", ref, repo, str(src)],
                check=True,
            )

    if install_steps:
        pip_cache = _TESTBED_ROOT / "cache" / "pip"
        pip_cache.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PATH"] = f"{bins}{os.pathsep}{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(_TESTBED_ROOT / ".venv")
        env["INTERFACE_DIR"] = str(bins)
        env["PIP_CACHE_DIR"] = str(pip_cache)
        cwd = str(src) if src.exists() else str(bins)
        for step in install_steps:
            subprocess.run(step, shell=True, env=env, cwd=cwd, check=True)
    return bins


def _run_install(
    steps: list[str],
    cwd: Path,
    venv_python: Path,
    interface_dir: Path | None = None,
) -> None:
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
