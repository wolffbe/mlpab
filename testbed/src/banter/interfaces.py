"""Resolve a chosen (platform, interface, version) for an engineer run.

Two locations:

    platforms/<platform>/<interface>/config.yaml — CONFIG: where the interface
        lives, how to build/run it, its credential keys, and the base (v0) prompt.
    platforms/<platform>/<interface>/ — BINARY: the built artifact only (e.g.
        platforms/hopsworks/cli/hops). SDK interfaces build nothing (pip-install
        per run) so have no folder here; `none` has none either.

The config manifest carries NO versions; its `prompt:` is v0 (the base). Improved
versions live INSIDE an autoresearch session:

    <version_root>/platforms/<platform>/<interface>/v<n>/version.yaml

(`<version_root>` is the session dir.) A version.yaml overrides the base manifest
(optionally `binary`, `runtime_install`, `mcp_servers`) and may ship its own binary
copy, else the base binary is reused. `prompt:` is the ONE non-overridable field —
frozen to the committed base across versions (the researcher improves interface
SOURCE, not the prompt). v0 always = base manifest, needs no folder.

`banter run` selects a version via `--interface-version <n>` + `--version-root <dir>`
(the session dir). Without a version-root only v0 (base) is available.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


INTERFACES = ("cli", "mcp", "sdk", "none")
_TESTBED_ROOT = Path(__file__).resolve().parents[2]

# Single unified tree. Per interface, config + checked-out source + built binary
# live together — that IS the base (v0) version:
#     platforms/<platform>/<interface>/config.yaml (build/run/keys/prompt)
#     platforms/<platform>/<interface>/...          (source + built artifact)
# A platform also holds its skills + research configs:
#     platforms/<platform>/skills/<bundle>/<version>/<skill>/SKILL.md
#     platforms/<platform>/autoresearch/*.yaml
#     platforms/<platform>/benchmark/*.yaml
PLATFORMS_DIR = _TESTBED_ROOT / "platforms"


@dataclass
class InterfaceSetup:
    platform: str
    interface: str
    prompt_fragment: str
    version: int
    hash: str
    cli_binary: str | None = None
    sdk_module: str | None = None
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    keys: dict[str, str] = field(default_factory=dict)
    serve: list[str] = field(default_factory=list)      # per-run server-start steps
    teardown: list[str] = field(default_factory=list)   # run start+end cleanup steps
    # Outbound hosts THIS interface needs reachable, beyond the testbed baseline
    # (claude API + loopback). Drives the engineer sandbox `allowedDomains`. e.g.
    # `["api.openai.com", "*.openai.com"]` for a cloud SDK; empty for local-only.
    allowed_domains: list[str] = field(default_factory=list)


@dataclass
class InterfaceStatus:
    """Outcome of an interface preflight check."""
    platform: str
    interface: str
    ok: bool
    installed: bool
    authenticated: bool
    missing_keys: list[str] = field(default_factory=list)
    reason: str = ""
    fix_command: str = ""

    @property
    def message(self) -> str:
        if self.ok:
            return f"{self.platform}/{self.interface}: ready"
        msg = f"{self.platform}/{self.interface}: {self.reason}"
        if self.fix_command:
            msg += f"\n  → Run: {self.fix_command}"
        return msg


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


# Per-(platform,interface) override of an interface's home dir. Autoresearch sets
# this (via set_interface_home / `banter run --interface-dir`) so the engineer
# builds + uses the increment's OWN copy
# (results/autoresearch/<run>/<inc>/interface/) instead of the committed
# platforms/<platform>/<interface>/. Safe as process-global: each `banter run`
# process handles exactly one interface.
_INTERFACE_HOME: dict[tuple[str, str], Path] = {}


def set_interface_home(platform: str, interface: str, home: Path) -> None:
    """Point this interface's home at `home` (the per-increment copy) for this
    process — config.yaml, $INTERFACE_DIR, build, install all resolve there."""
    _INTERFACE_HOME[(platform, interface)] = Path(home)


def manifest_path(platform: str, interface: str) -> Path:
    return bin_dir(platform, interface) / "config.yaml"


def platform_interface_from_config(config_path: str | Path) -> tuple[str, str]:
    """Infer (platform, interface) from an interface config path like
    `platforms/<platform>/<interface>/config.yaml`."""
    p = Path(config_path)
    interface = p.parent.name
    platform = p.parent.parent.name
    if interface not in INTERFACES:
        raise ValueError(
            f"Unknown interface {interface!r} from {config_path!r}. "
            "Expected a path like platforms/<platform>/<interface>/config.yaml"
        )
    return platform, interface


def bin_dir(platform: str, interface: str) -> Path:
    """The interface's home: config.yaml + checked-out code + built binary.

    Honors a per-(platform,interface) override from `set_interface_home`
    (autoresearch's per-increment copy); else committed platforms/<platform>/<interface>/.
    """
    return _INTERFACE_HOME.get((platform, interface), PLATFORMS_DIR / platform / interface)


def version_dir(version_root: Path, platform: str, interface: str, version: int) -> Path:
    """Session-local directory for an improved interface version."""
    return Path(version_root) / "platforms" / platform / interface / f"v{version}"


def load_manifest(platform: str, interface: str) -> dict[str, Any]:
    p = manifest_path(platform, interface)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


# ---------------------------------------------------------------------------
# Version override (session-local)
# ---------------------------------------------------------------------------


def _load_version_override(
    version_root: Path | None, platform: str, interface: str, version: int
) -> dict[str, Any]:
    """Read a session-local version.yaml, or {} for base (v0) / when absent."""
    if not version or version_root is None:
        return {}
    vp = version_dir(version_root, platform, interface, version) / "version.yaml"
    if not vp.exists():
        return {}
    return yaml.safe_load(vp.read_text()) or {}


def _resolved_config(
    platform: str, interface: str, version: int, version_root: Path | None
) -> dict[str, Any]:
    """Merge base manifest defaults with a session-local version override."""
    manifest = load_manifest(platform, interface)
    override = _load_version_override(version_root, platform, interface, version)
    merged: dict[str, Any] = {
        "binary": manifest.get("binary"),
        "runtime_install": manifest.get("runtime_install") or [],
        "mcp_servers": manifest.get("mcp_servers") or {},
        "prompt": manifest.get("prompt"),
        # Optional accounting overrides: invokable CLI command (for cli_calls) and
        # importable SDK module (for sdk_calls). Default to `binary` / `platform`
        # below, so existing configs are unchanged.
        "cli_command": manifest.get("cli_command"),
        "sdk_module": manifest.get("sdk_module"),
        # Per-run shell steps in the run venv to START the engineer's servers (e.g.
        # the MCP HTTP server claude connects to at launch), after stale-server
        # cleanup. `teardown` stops them at run start + end.
        "serve": manifest.get("serve") or [],
        "teardown": manifest.get("teardown") or [],
    }
    # `prompt` is deliberately NOT overridable: frozen to the committed base (see
    # `_prompt_for`/`_committed_prompt`) since the researcher improves interface
    # SOURCE, never the prompt. A `prompt:` in a version.yaml override (or an
    # autoresearch interface-dir copy) is ignored.
    for key in ("binary", "runtime_install", "mcp_servers",
                "cli_command", "sdk_module", "serve", "teardown"):
        if key in override:
            merged[key] = override[key]
    return merged


def _interface_dir_for(
    platform: str, interface: str, version: int, version_root: Path | None, binary: str | None
) -> Path:
    """The dir to expose as $INTERFACE_DIR — where the binary to use lives.

    A session version shipping its own binary copy uses its own folder; else the
    base binary dir is reused.
    """
    if binary and version and version_root is not None:
        vd = version_dir(version_root, platform, interface, version)
        if (vd / binary).exists():
            return vd
    return bin_dir(platform, interface)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _auto_prompt(platform: str, interface: str, binary: str | None) -> str:
    """Generate a sensible default prompt when the config doesn't supply one."""
    cap = platform.capitalize()
    if interface == "cli" and binary:
        return (
            f"The {cap} `{binary}` CLI is installed and authenticated. "
            f"Use `{binary} <subcommand>` for all {cap} operations."
        )
    if interface == "sdk":
        return f"The {cap} Python SDK is installed. Import and use it for all {cap} operations."
    if interface == "mcp":
        return f"You have access to the {cap} MCP server. Use the provided MCP tools for all {cap} operations."
    return ""


def _committed_prompt(platform: str, interface: str) -> str | None:
    """The engineer-facing prompt AS COMMITTED at platforms/<p>/<i>/config.yaml.

    Read straight from the committed tree — NOT via bin_dir/load_manifest, which
    honor `set_interface_home` and would pick up an autoresearch interface-dir copy
    whose `prompt:` the researcher may have edited. The prompt is FROZEN across
    versions (the researcher improves interface SOURCE, never the prompt), so edits
    to `prompt:` in a version copy or version.yaml override are ignored.
    """
    p = PLATFORMS_DIR / platform / interface / "config.yaml"
    if not p.exists():
        return None
    m = yaml.safe_load(p.read_text()) or {}
    return m.get("prompt")


def _prompt_for(platform: str, interface: str, version: int, version_root: Path | None) -> str:
    # FROZEN: always the committed base prompt, regardless of version / home
    # redirect — this makes a prompt-only "improvement" a no-op. `is not None`
    # (not truthiness): an intentionally EMPTY committed prompt must also freeze to
    # empty, else the home-redirected copy's edited prompt leaks back in.
    committed = _committed_prompt(platform, interface)
    if committed is not None:
        return committed.strip()
    # No committed prompt key (legacy / dynamically-homed interface): fall back to
    # the resolved config, then an auto-generated default.
    cfg = _resolved_config(platform, interface, version, version_root)
    text = cfg.get("prompt")
    if text:
        return text.strip()
    return _auto_prompt(platform, interface, cfg.get("binary"))


def prompt_hash_for(
    platform: str, interface: str, version: int | None = None, version_root: Path | None = None
) -> str:
    if platform == "none" and interface == "none":
        return ""
    if not load_manifest(platform, interface):
        return ""
    chosen = version or 0
    text = _prompt_for(platform, interface, chosen, version_root)
    if not text:
        return ""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Source fingerprint — "did the interface SOURCE actually change?"
# ---------------------------------------------------------------------------

# Build output + plumbing, not interface SOURCE: excluded from the fingerprint so
# prompt/config tweaks and rebuilt wheels don't count as a "real" change.
_FINGERPRINT_EXCLUDE_DIRS = {
    "__pycache__", ".git", "build", "dist", ".claude", ".github", "Library",
}
_FINGERPRINT_EXCLUDE_SUFFIXES = (".whl", ".pyc", ".pyo", ".egg-link")
# `pyproject.toml` is build-generated; `.banter-src-hash` is our own cache key.
# `config.yaml` is NOT excluded — it's hashed with only the (frozen) `prompt:`
# removed, so a prompt-only edit is a no-op but a real config edit (install steps,
# binary, runtime_install, teardown, …) counts as a change.
_FINGERPRINT_EXCLUDE_NAMES = {"pyproject.toml", ".banter-src-hash"}


def _config_fingerprint_bytes(config_path: Path) -> bytes:
    """Bytes for the root `config.yaml` MINUS the frozen `prompt:` field.

    Lets `source_fingerprint` treat a prompt-only edit as no change while still
    detecting real plumbing edits (install steps, binary, runtime_install,
    teardown, …) — a valid researcher lever. Falls back to raw bytes if the YAML
    can't be parsed.
    """
    try:
        data = yaml.safe_load(config_path.read_text())
        if isinstance(data, dict):
            data.pop("prompt", None)
            return yaml.safe_dump(data, sort_keys=True).encode()
    except Exception:
        pass
    return config_path.read_bytes()


def source_fingerprint(interface_dir: Path) -> str:
    """A stable hash of an interface's SOURCE tree.

    Excludes build artifacts (wheels, `build/`, `*.egg-info`, `__pycache__`), the
    generated `pyproject.toml`, and Claude/VCS plumbing. The root `config.yaml` is
    hashed with its frozen `prompt:` stripped, so two copies with the same
    fingerprint have identical source AND build plumbing — only the prompt or
    rebuilt binaries differ.
    """
    interface_dir = Path(interface_dir)
    h = hashlib.sha256()
    for f in sorted(p for p in interface_dir.rglob("*") if p.is_file()):
        rel = f.relative_to(interface_dir)
        if set(rel.parts) & _FINGERPRINT_EXCLUDE_DIRS:
            continue
        if any(part.endswith(".egg-info") for part in rel.parts):
            continue
        if f.name in _FINGERPRINT_EXCLUDE_NAMES or f.suffix in _FINGERPRINT_EXCLUDE_SUFFIXES:
            continue
        # Root config.yaml: hash everything except the frozen prompt. A nested
        # config.yaml deeper in the tree is hashed normally (it's source).
        payload = (
            _config_fingerprint_bytes(f) if str(rel) == "config.yaml" else f.read_bytes()
        )
        h.update(str(rel).encode())
        h.update(b"\0")
        h.update(payload)
        h.update(b"\0")
    return h.hexdigest()


def assert_source_changed(
    interface_dir: Path, version_label: str | None, fp: str | None = None
) -> None:
    """Refuse an autoresearch IMPROVEMENT version that didn't change the source.

    For `v<N>` (N>0) with a sibling `v<N-1>/interface` present, raise `ValueError`
    if the source is byte-identical to the previous version. Since the prompt is
    frozen and `config.yaml` excluded from the fingerprint, a version that only
    re-worded the prompt (or changed nothing) is caught here — before the engineer
    runs — instead of silently re-measuring `v<N-1>`.

    Baselines (`v0`), non-autoresearch runs (no version label), and the first
    version after a continuation (no sibling on disk) are unconstrained. `fp` may
    carry this dir's already-computed fingerprint (the caller often needs it too)
    to avoid hashing the tree twice.
    """
    if not version_label:
        return
    label = str(version_label).lstrip("v")
    if not label.isdigit():
        return
    n = int(label)
    if n <= 0:
        return
    interface_dir = Path(interface_dir)
    prev = interface_dir.parent.parent / f"v{n - 1}" / "interface"
    if not prev.is_dir():
        return
    cur = fp if fp is not None else source_fingerprint(interface_dir)
    if cur == source_fingerprint(prev):
        raise ValueError(
            f"interface source for {version_label} is identical to v{n - 1} "
            f"(only config.yaml/prompt or build artifacts differ). The engineer "
            f"prompt is FROZEN — to make {version_label} count, improve the "
            f"interface SOURCE under {interface_dir} (e.g. src/python/...) and "
            f"re-run. A prompt-only change is not a valid version."
        )


# ---------------------------------------------------------------------------
# Credential keys (declared + stored in the config manifest)
# ---------------------------------------------------------------------------


def keys_for(platform: str, interface: str) -> dict[str, str]:
    """Return the manifest's declared credential keys as {name: value}.

    Accepts a mapping (`keys: {NAME: value}`), a list of names to read from the env
    (`keys: [NAME1, NAME2]`), or a list of `{name, value}` dicts. A missing/empty
    value normalises to "" and is resolved from the env at run time (see
    `_resolved_keys`).
    """
    if platform == "none" and interface == "none":
        return {}
    raw = load_manifest(platform, interface).get("keys") or {}
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[str(k)] = "" if v is None else str(v)
    elif isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, str):
                out[entry] = ""
            elif isinstance(entry, dict) and entry.get("name"):
                v = entry.get("value")
                out[str(entry["name"])] = "" if v is None else str(v)
    return out


def _resolved_keys(platform: str, interface: str, env: dict[str, str] | None = None) -> dict[str, str]:
    """Key values to inject into the engineer env: manifest value, else env."""
    base = env if env is not None else os.environ
    out: dict[str, str] = {}
    for k, v in keys_for(platform, interface).items():
        out[k] = v or base.get(k, "")
    return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _compute_hash(platform: str, interface: str, version: int, version_root: Path | None) -> str:
    """sha256 over manifest bytes + version override bytes + binary bytes + version."""
    h = hashlib.sha256()
    mp = manifest_path(platform, interface)
    if mp.exists():
        h.update(mp.read_bytes())
        h.update(b"\0")
    if version and version_root is not None:
        vp = version_dir(version_root, platform, interface, version) / "version.yaml"
        if vp.exists():
            h.update(vp.read_bytes())
            h.update(b"\0")
    cfg = _resolved_config(platform, interface, version, version_root)
    binary = cfg.get("binary")
    if binary:
        bpath = _interface_dir_for(platform, interface, version, version_root, binary) / binary
        if bpath.is_file():
            h.update(bpath.read_bytes())
            h.update(b"\0")
    h.update(f"|v={version}".encode())
    return h.hexdigest()[:8]


def _check_known(platform: str, interface: str, version: int | None, version_root: Path | None) -> int:
    """Validate interface/version and return the chosen version int. Raises ValueError."""
    if interface not in INTERFACES:
        raise ValueError(f"Unknown interface {interface!r}; expected one of {INTERFACES}")
    manifest = load_manifest(platform, interface)
    if not manifest:
        raise ValueError(
            f"No config for {platform!r}/{interface!r} at {manifest_path(platform, interface)}. "
            f"Create it (platforms/{platform}/{interface}/config.yaml) — preflight builds the binary."
        )
    chosen = version or 0
    if chosen and version_root is None:
        raise ValueError(
            f"Interface {platform!r}/{interface!r} v{chosen} requires --version-root "
            f"(versions live inside an autoresearch session)."
        )
    if chosen and not (version_dir(version_root, platform, interface, chosen) / "version.yaml").exists():
        raise ValueError(
            f"Interface {platform!r}/{interface!r} has no version {chosen} under "
            f"{version_dir(version_root, platform, interface, chosen)}."
        )
    return chosen


def variant_for(
    platform: str,
    interface: str,
    version: int | None = None,
    version_root: Path | None = None,
) -> tuple[int, str]:
    """Return (version, hash). version None → 0 (base manifest)."""
    if platform == "none" and interface == "none":
        return 0, ""
    chosen = _check_known(platform, interface, version, version_root)
    return chosen, _compute_hash(platform, interface, chosen, version_root)


def setup(
    platform: str,
    interface: str,
    run_dir: Path,
    venv_python: Path,
    version: int | None = None,
    version_root: Path | None = None,
) -> InterfaceSetup:
    if interface not in INTERFACES:
        raise ValueError(f"Unknown interface {interface!r}; expected one of {INTERFACES}")

    if platform == "none" and interface == "none":
        return InterfaceSetup(platform="none", interface="none", prompt_fragment="", version=0, hash="")

    chosen = _check_known(platform, interface, version, version_root)
    cfg = _resolved_config(platform, interface, chosen, version_root)
    binary = cfg.get("binary")
    runtime_install = cfg.get("runtime_install") or []
    mcp_servers = cfg.get("mcp_servers") or {}
    prompt_fragment = _prompt_for(platform, interface, chosen, version_root)
    # The "use the interface as-is" rule is the engineer prompt's default (baked
    # into engineer.md, stripped only for none/none) — not appended here. This
    # fragment carries just the interface's own `prompt:` prose.
    hash_ = _compute_hash(platform, interface, chosen, version_root)
    interface_dir = _interface_dir_for(platform, interface, chosen, version_root, binary)

    # Guard: if runtime steps reference $INTERFACE_DIR (pre-built binary), it must
    # exist. Preflight builds it before here; this is a backstop.
    if binary and runtime_install and any("$INTERFACE_DIR" in s for s in runtime_install):
        if not (interface_dir / binary).exists():
            raise RuntimeError(
                f"Interface {platform!r}/{interface!r} v{chosen} binary '{binary}' not found at "
                f"{interface_dir / binary}. Preflight should have built it; "
                f"check platforms/{platform}/{interface}/config.yaml install steps."
            )

    if runtime_install:
        _run_install(runtime_install, cwd=run_dir, venv_python=venv_python, interface_dir=interface_dir)

    return InterfaceSetup(
        platform=platform,
        interface=interface,
        prompt_fragment=prompt_fragment,
        version=chosen,
        hash=hash_,
        # Resolve BOTH the CLI command and SDK module for EVERY interface (not just
        # the active one), so the engineer hook + command counter can detect and
        # block CROSS-INTERFACE escapes — e.g. `import hopsworks` while the CLI is
        # under test, or a `hops` call while the SDK is. `interface` decides which is
        # "on-interface".
        #   cli_binary: the invokable command (`cli_command`; for a CLI the bare
        #     binary name also works, but a wheel needs the explicit field).
        #   sdk_module: the importable module (`sdk_module`, else the platform name
        #     — true when package == platform name).
        cli_binary=cfg.get("cli_command") or (binary if interface == "cli" else None),
        sdk_module=cfg.get("sdk_module") or platform,
        mcp_servers=mcp_servers,
        keys=_resolved_keys(platform, interface),
        serve=cfg.get("serve") or [],
        teardown=cfg.get("teardown") or [],
        allowed_domains=list(cfg.get("allowed_domains") or []),
    )


def _make_check_venv(target: Path) -> Path:
    """Create an empty venv that shares the base .venv's libraries but holds no
    interface packages — for ephemeral preflight checks (mirrors how each engineer
    run's venv is built). Returns the venv's python.
    """
    base_py = _TESTBED_ROOT / ".venv" / "bin" / "python"
    base = base_py if base_py.exists() else Path(sys.executable)
    subprocess.run(
        [str(base), "-m", "venv", "--system-site-packages", str(target)], check=True
    )
    py = target / "bin" / "python"
    # Expose the base venv's site-packages (shared libs) explicitly, like runner.
    if base_py.exists():
        vsp = next((target / "lib").glob("python*/site-packages"), None)
        bsp = next((_TESTBED_ROOT / ".venv" / "lib").glob("python*/site-packages"), None)
        if vsp and bsp:
            (vsp / "_base_venv.pth").write_text(f"{bsp}\n")
    return py


def _interface_dist_name(cfg: dict, platform: str, binary: str | None) -> str | None:
    """The pip DISTRIBUTION name the interface installs — the package that must NOT
    live in the base .venv. Per-run venvs (clone of base) and the ephemeral check
    venv (--system-site-packages over base) both derive from base; if base already
    holds this distribution at the wheel's version, `pip install <wheel>[extra]`
    reports it satisfied and SKIPS it, so the venv never gets the interface's
    console scripts or extra deps.

    Prefer the built wheel's own name (`<dist>-<version>-...whl`; first
    `-`-delimited token is the normalized distribution); else the declared sdk
    module / platform name (true when package == platform name)."""
    if binary and binary.endswith(".whl"):
        return binary.split("-", 1)[0]
    return cfg.get("sdk_module") or platform


def ensure_base_clean(
    platform: str,
    interface: str,
    version: int | None = None,
    version_root: Path | None = None,
) -> None:
    """Keep the base researcher .venv free of the interface package, so every
    per-run (and preflight check) venv installs the wheel FRESH and complete.

    The base .venv holds only the SHARED libraries each run inherits
    (requirements.txt) — never an interface package. If one leaks in (e.g. a manual
    `pip install` of the wheel), it shadows the per-run install and the run venv
    ends up missing console scripts / extra deps. Uninstalling here restores the
    invariant; no-op once base is clean.
    """
    if platform == "none" and interface == "none":
        return
    base_py = _TESTBED_ROOT / ".venv" / "bin" / "python"
    if not base_py.exists():
        return
    try:
        chosen = _check_known(platform, interface, version, version_root)
        cfg = _resolved_config(platform, interface, chosen, version_root)
    except Exception:
        return
    dist = _interface_dist_name(cfg, platform, cfg.get("binary"))
    if not dist:
        return
    # Installed IN base? `pip show` exits 0 iff the distribution is present.
    present = subprocess.run(
        [str(base_py), "-m", "pip", "show", dist],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not present:
        return
    subprocess.run(
        [str(base_py), "-m", "pip", "uninstall", "-y", dist],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(
        f"[banter] removed interface package {dist!r} from the base .venv — it must "
        f"stay free of interface packages so each run installs the wheel fresh.",
        file=sys.stderr, flush=True,
    )


def preflight(
    platform: str,
    interface: str,
    version: int | None = None,
    version_root: Path | None = None,
    *,
    check_login: bool = True,
    auto_build: bool = True,
    timeout_s: int = 120,
    cleanup_build: bool = False,
    env: dict[str, str] | None = None,
) -> InterfaceStatus:
    """Build + verify an interface. `cleanup_build=True` (session preflight,
    `banter test`) afterwards deletes any build artifacts this created from the
    committed source folder, keeping it source-only — real installs happen per
    version (autoresearch `vX`) or per benchmark run. Default False leaves the
    artifact in place (benchmark runs the engineer against it directly)."""
    try:
        return _preflight_impl(
            platform, interface, version, version_root,
            check_login=check_login, auto_build=auto_build, timeout_s=timeout_s, env=env,
        )
    finally:
        if cleanup_build:
            _clean_build_artifacts(platform, interface)


def _force_rmtree(path: Path) -> None:
    """`shutil.rmtree` that also removes read-only files. A `repo:` clone's `.git`
    objects are read-only, and `rmtree(ignore_errors=True)` would silently skip
    them — leaving a partial `src/` behind. Make every entry writable first."""
    if not path.exists():
        return
    try:
        os.chmod(path, 0o700)            # the top dir itself, so its entries unlink
    except OSError:
        pass
    for root, dirs, files in os.walk(path):
        for name in files + dirs:
            try:
                os.chmod(os.path.join(root, name), 0o700)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def _clean_build_artifacts(platform: str, interface: str) -> None:
    """Remove build artifacts from the committed source folder so it stays
    source-only (config + source). A wheel, build|dist|src dir, egg-info, or the
    configured `binary` in a source folder is always a build artifact."""
    base = PLATFORMS_DIR / platform / interface
    if not base.is_dir():
        return
    for d in ("build", "dist", "src"):
        _force_rmtree(base / d)
    for p in list(base.glob("*.egg-info")):
        _force_rmtree(p)
    for p in list(base.glob("*.whl")):
        p.unlink(missing_ok=True)
    manifest = load_manifest(platform, interface)
    binary = manifest.get("binary")
    if binary and not binary.endswith((".py", ".toml", ".md", ".yaml", ".yml")):
        (base / binary).unlink(missing_ok=True)
    # With build metadata in config.yaml (`package:`), pyproject.toml is
    # build-generated — remove it so the committed folder stays source-only.
    if manifest.get("package"):
        (base / "pyproject.toml").unlink(missing_ok=True)


def _preflight_impl(
    platform: str,
    interface: str,
    version: int | None = None,
    version_root: Path | None = None,
    *,
    check_login: bool = True,
    auto_build: bool = True,
    timeout_s: int = 120,
    env: dict[str, str] | None = None,
) -> InterfaceStatus:
    """Build (if needed), then verify an interface is installed + logged in + healthy.

    Config-declared interfaces are binaries built, set up, and authenticated at
    preflight — no manual `banter install`. Concretely:

      * Build — if the binary is missing, run the config's `install:` steps
        (deterministic, no AI). Disable with auto_build=False.
      * Login — `auth_command` exits 0 run non-interactively with the declared keys
        in env; without an `auth_command`, login is satisfied only when every
        declared key is set.
      * Test — `test_command` exits 0.
    """
    config_fix = f"check platforms/{platform}/{interface}/config.yaml (build/install steps)"
    setup_fix = "make setup  (or: banter setup " + f"platforms/{platform}/{interface}/config.yaml)"

    if platform == "none" and interface == "none":
        return InterfaceStatus(platform, interface, ok=True, installed=True, authenticated=True)
    if interface not in INTERFACES:
        return InterfaceStatus(
            platform, interface, ok=False, installed=False, authenticated=False,
            reason=f"unknown interface {interface!r}",
        )

    # 1) installed? Build the base binary on demand if it's missing.
    try:
        chosen = _check_known(platform, interface, version, version_root)
    except ValueError as e:
        return InterfaceStatus(
            platform, interface, ok=False, installed=False, authenticated=False,
            reason=str(e), fix_command=config_fix,
        )
    cfg = _resolved_config(platform, interface, chosen, version_root)
    binary = cfg.get("binary")
    # The base .venv must never hold the interface package — it would shadow the
    # fresh per-run wheel install (check + run venvs both derive from base). Strip
    # it here so the checks below mirror a real run.
    ensure_base_clean(platform, interface, chosen, version_root)
    runtime_install = cfg.get("runtime_install") or []
    uses_prebuilt = bool(
        binary and runtime_install and any("$INTERFACE_DIR" in s for s in runtime_install)
    )
    bpath = (
        _interface_dir_for(platform, interface, chosen, version_root, binary) / binary
        if uses_prebuilt else None
    )

    # (Re)build only when the artifact is missing. The config's `install:` steps
    # only BUILD the artifact into the interface dir — they don't install it into
    # the base .venv, so base stays free of interface packages and runs don't
    # overlap. The interface installs fresh into a throwaway venv for the checks
    # below, and into each engineer's per-run venv at run time.
    artifact_missing = bpath is not None and not bpath.exists()
    if artifact_missing and auto_build and load_manifest(platform, interface).get("install"):
        try:
            build(platform, interface)
        except Exception as e:  # build is shell-out heavy; surface failures
            return InterfaceStatus(
                platform, interface, ok=False, installed=False, authenticated=False,
                reason=f"build failed: {e}", fix_command=config_fix,
            )
    if bpath is not None and not bpath.exists():
        return InterfaceStatus(
            platform, interface, ok=False, installed=False, authenticated=False,
            reason=f"binary {binary!r} missing at {bpath}", fix_command=config_fix,
        )

    # Verify in an EPHEMERAL venv that shares base libs but installs ONLY this
    # interface (its runtime_install) — exactly what an engineer run gets — then is
    # torn down (keeps base free of interface packages). Session preflight runs only
    # build + `test_command`; LOGIN is verified per challenge
    # (interfaces.login_status). check_login=True (single-challenge form) also checks
    # login here.
    keys = keys_for(platform, interface)
    base_env = dict(env) if env is not None else dict(os.environ)
    merged_env = dict(base_env)
    missing = []
    for k, declared in keys.items():
        val = declared or base_env.get(k, "")
        if val:
            merged_env[k] = val
        else:
            missing.append(k)

    manifest = load_manifest(platform, interface)
    auth_command = manifest.get("auth_command") if check_login else None
    test_command = manifest.get("test_command")
    # No auth_command → login is satisfied only when every declared key is present.
    if check_login and not auth_command and keys and missing:
        return InterfaceStatus(
            platform, interface, ok=False, installed=True, authenticated=False,
            missing_keys=missing,
            reason=f"missing credential key(s): {', '.join(missing)}",
            fix_command=setup_fix,
        )
    # Nothing to verify in a venv (no login to check here, no test) → done.
    if not auth_command and not test_command:
        return InterfaceStatus(
            platform, interface, ok=True, installed=True, authenticated=True, missing_keys=missing,
        )

    # Only stand up a throwaway venv when there's something to install (real
    # interfaces); else run checks against the base bin (cheap, nothing to isolate).
    # The venv installs ONLY this interface and is torn down.
    tmp = None
    try:
        if runtime_install:
            tmp = Path(tempfile.mkdtemp(prefix="banter-preflight-"))
            check_python = _make_check_venv(tmp / "venv")
            try:
                _run_install(
                    runtime_install, cwd=tmp, venv_python=check_python,
                    interface_dir=_interface_dir_for(platform, interface, chosen, version_root, binary),
                )
            except Exception as e:
                return InterfaceStatus(
                    platform, interface, ok=False, installed=False, authenticated=False,
                    reason=f"interface install failed: {e}", fix_command=config_fix,
                )
            check_bin = str(check_python.parent)
            merged_env["VIRTUAL_ENV"] = str(tmp / "venv")
        else:
            check_bin = str(_TESTBED_ROOT / ".venv" / "bin")
        # The interface command/module lives in the (throwaway) venv; expose its bin
        # — and the interface dir, for bare binaries — on PATH for the checks.
        path_parts = [str(bin_dir(platform, interface)), check_bin, merged_env.get("PATH", "")]
        merged_env["PATH"] = os.pathsep.join(p for p in path_parts if p)

        if auth_command and not _run_check(auth_command, merged_env, timeout_s):
            return InterfaceStatus(
                platform, interface, ok=False, installed=True, authenticated=False,
                missing_keys=missing,
                reason=f"login check failed (`{auth_command}` returned non-zero or timed out)",
                fix_command=setup_fix,
            )
        if test_command and not _run_check(test_command, merged_env, timeout_s):
            return InterfaceStatus(
                platform, interface, ok=False, installed=True, authenticated=True,
                missing_keys=missing,
                reason=f"interface did not run reliably (`{test_command}` returned non-zero or timed out)",
                fix_command=config_fix,
            )
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

    return InterfaceStatus(
        platform, interface, ok=True, installed=True, authenticated=True, missing_keys=missing,
    )


def _run_check(command: str, env: dict[str, str], timeout_s: int) -> bool:
    """Run a declared check command non-interactively. True iff it exits 0.

    On failure (non-zero exit, timeout, or OSError) the captured output is printed
    to stderr so the operator can SEE why — else a hanging auth_command looks
    identical to a silent stall.
    """
    import sys as _sys
    try:
        proc = subprocess.run(
            command, shell=True, env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            print(
                f"[banter] check FAILED (exit {proc.returncode}): {command}\n"
                f"--- captured output ---\n{proc.stdout.decode('utf-8', 'replace')}"
                f"\n--- end ---",
                file=_sys.stderr, flush=True,
            )
            return False
        return True
    except subprocess.TimeoutExpired as e:
        out = (e.output or b"").decode("utf-8", "replace") if e.output else "(no output)"
        print(
            f"[banter] check TIMED OUT after {timeout_s}s: {command}\n"
            f"--- captured output before timeout ---\n{out}\n--- end ---",
            file=_sys.stderr, flush=True,
        )
        return False
    except OSError as e:
        print(f"[banter] check OS error: {command}: {e}", file=_sys.stderr, flush=True)
        return False


def login_status(
    platform: str,
    interface: str,
    *,
    venv_python: Path,
    keys: dict[str, str] | None = None,
    timeout_s: int = 30,
) -> InterfaceStatus:
    """Per-challenge login check: run the interface's `auth_command` in the given
    (per-run) venv. Build + test happen once at session preflight; login is
    re-verified for EVERY challenge here (so expired creds are caught mid-session
    and each run authenticates in its own venv).
    """
    if platform == "none" and interface == "none":
        return InterfaceStatus(platform, interface, ok=True, installed=True, authenticated=True)
    setup_fix = "make setup  (or: banter setup " + f"platforms/{platform}/{interface}/config.yaml)"
    declared = keys_for(platform, interface)
    env = dict(os.environ)
    missing = []
    for k, dv in declared.items():
        val = (keys or {}).get(k) or dv or env.get(k, "")
        if val:
            env[k] = val
        else:
            missing.append(k)

    auth_command = load_manifest(platform, interface).get("auth_command")
    if not auth_command:
        # No auth_command → login is satisfied only when all declared keys exist.
        if declared and missing:
            return InterfaceStatus(
                platform, interface, ok=False, installed=True, authenticated=False,
                missing_keys=missing,
                reason=f"missing credential key(s): {', '.join(missing)}", fix_command=setup_fix,
            )
        return InterfaceStatus(platform, interface, ok=True, installed=True, authenticated=True)

    env["PATH"] = os.pathsep.join(
        [str(bin_dir(platform, interface)), str(venv_python.parent), env.get("PATH", "")]
    )
    env["VIRTUAL_ENV"] = str(venv_python.parent.parent)
    if not _run_check(auth_command, env, timeout_s):
        return InterfaceStatus(
            platform, interface, ok=False, installed=True, authenticated=False, missing_keys=missing,
            reason=f"login check failed (`{auth_command}` returned non-zero or timed out)",
            fix_command=setup_fix,
        )
    return InterfaceStatus(
        platform, interface, ok=True, installed=True, authenticated=True, missing_keys=missing,
    )


def _render_pyproject(pkg: dict) -> str:
    """Render a `pyproject.toml` from a config.yaml `package:` block, so build
    metadata lives in config.yaml and the committed folder stays source-only.

    Supported keys: name, version, requires-python, description, dependencies
    (list), scripts (name→target map), optional-dependencies (extra→[deps]),
    packages (list). Build backend: setuptools.
    """
    def arr(xs: list) -> str:
        return "[" + ", ".join(f'"{x}"' for x in xs) + "]"

    lines = [
        "[build-system]",
        'requires = ["setuptools>=68"]',
        'build-backend = "setuptools.build_meta"',
        "",
        "[project]",
        f'name = "{pkg["name"]}"',
        f'version = "{pkg.get("version", "0.1.0")}"',
    ]
    if pkg.get("requires-python"):
        lines.append(f'requires-python = "{pkg["requires-python"]}"')
    if pkg.get("description"):
        lines.append(f'description = "{pkg["description"]}"')
    if pkg.get("dependencies"):
        lines.append(f'dependencies = {arr(pkg["dependencies"])}')
    if pkg.get("scripts"):
        lines += ["", "[project.scripts]"] + [f'{k} = "{v}"' for k, v in pkg["scripts"].items()]
    if pkg.get("optional-dependencies"):
        lines += ["", "[project.optional-dependencies]"] + [
            f"{k} = {arr(v)}" for k, v in pkg["optional-dependencies"].items()
        ]
    if pkg.get("packages"):
        lines += ["", "[tool.setuptools]", f'packages = {arr(pkg["packages"])}']
    return "\n".join(lines) + "\n"


def build(platform: str, interface: str) -> Path:
    """Check out the interface's repo into its folder and build it (no AI).

    Code is checked out INTO platforms/<platform>/<interface>/ — both the checkout
    and the build/artifact location ($INTERFACE_DIR). A `repo:` pointing at a local
    path (e.g. a `fake_repos/...` fake repo) is copied in; a URL is git-cloned. Then
    the config's `install:` steps run there. Idempotent. Returns the interface dir.
    Run automatically by preflight when the binary is missing.
    """
    if platform == "none" and interface == "none":
        return bin_dir("none", "none")
    manifest = load_manifest(platform, interface)
    if not manifest:
        raise FileNotFoundError(f"No config at {manifest_path(platform, interface)}")

    repo = manifest.get("repo")
    ref = manifest.get("ref", "main")
    install_steps = manifest.get("install") or []
    iface_dir = bin_dir(platform, interface)   # config.yaml + (committed) source + artifact
    iface_dir.mkdir(parents=True, exist_ok=True)
    venv_bin = _TESTBED_ROOT / ".venv" / "bin"

    # Where install steps run. No `repo:` → source committed in place (e.g. mlkit).
    # A remote `repo:` is cloned into a `src/` subdir so it doesn't clobber
    # config.yaml; a local path is copied in place.
    src_cwd = iface_dir
    if repo:
        local = _TESTBED_ROOT / repo
        if local.is_dir():
            shutil.copytree(local, iface_dir, dirs_exist_ok=True)
        else:
            src = iface_dir / "src"
            # Pin source to `ref` — a FIXED COMMIT SHA (preferred, reproducible) or
            # a branch/tag. Once checked out we NEVER `git pull`: the tree stays at
            # the pinned commit, so (1) the experiment isn't confounded by upstream
            # drift mid-session, and (2) a researcher's edits to a per-version copy
            # aren't clobbered/merged with origin. To refresh the base, delete src/.
            if not (src / ".git").exists():
                src.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "-C", str(src), "init", "-q"], check=True)
                subprocess.run(
                    ["git", "-C", str(src), "remote", "add", "origin", repo], check=True
                )
                # Fetch just the pinned ref at depth 1 (GitHub serves a reachable
                # SHA this way), then check out detached.
                subprocess.run(
                    ["git", "-C", str(src), "fetch", "--depth", "1", "origin", ref],
                    check=True,
                )
                subprocess.run(
                    ["git", "-C", str(src), "checkout", "-q", "FETCH_HEAD"], check=True
                )
            src_cwd = src
        # Always strip a cloned/copied repo's own agent instructions (.claude/,
        # CLAUDE.md, .mcp.json) — e.g. hopsworks-api ships a .claude/ — so they're
        # never auto-loaded as directives by the engineer/researcher working in or
        # near the interface source.
        from banter import docs as _docs
        _docs.strip_agent_plumbing(src_cwd)

    # `package:` in config.yaml IS the build manifest — write the needed
    # pyproject.toml into the build dir (not committed), so the committed folder
    # stays config.yaml + source code only.
    package = manifest.get("package")
    if package:
        (Path(src_cwd) / "pyproject.toml").write_text(_render_pyproject(package))

    if install_steps:
        env = os.environ.copy()
        env["PATH"] = f"{iface_dir}{os.pathsep}{venv_bin}{os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(_TESTBED_ROOT / ".venv")
        env["INTERFACE_DIR"] = str(iface_dir)
        env["TESTBED_ROOT"] = str(_TESTBED_ROOT)
        # Disable the pip download cache — interface installs are tiny wheels and
        # the cache write path (<testbed>/cache/pip) is unreachable for the
        # researcher/engineer (deny patterns), triggering a pip warning otherwise.
        env["PIP_NO_CACHE_DIR"] = "1"
        for step in install_steps:
            subprocess.run(step, shell=True, env=env, cwd=str(src_cwd), check=True)
    return iface_dir


def _run_install(
    steps: list[str],
    cwd: Path,
    venv_python: Path,
    interface_dir: Path | None = None,
) -> None:
    env = os.environ.copy()
    env["PATH"] = f"{venv_python.parent}:{env['PATH']}"
    env["VIRTUAL_ENV"] = str(venv_python.parent.parent)
    env["TESTBED_ROOT"] = str(_TESTBED_ROOT)
    if interface_dir is not None:
        env["INTERFACE_DIR"] = str(interface_dir)
    # Disable pip's download cache: the shared cache path isn't writable to the
    # engineer (deny patterns), and runtime_install steps are tiny.
    env["PIP_NO_CACHE_DIR"] = "1"
    for step in steps:
        subprocess.run(step, shell=True, cwd=cwd, env=env, check=True)
