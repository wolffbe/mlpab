"""Resolve a chosen (platform, interface) for an agent run.

Two locations:

    configs/platforms/<platform>/<interface>.yaml — CONFIG: where the interface
        lives, how to build/run it, its credential keys, and the agent-facing
        prompt.
    build/<platform>/<interface>/ — BUILD HOME: source checkout + built artifact
        (e.g. build/hopsworks/cli/hops). SDK interfaces build nothing (pip-install
        per run); `none` has neither.

`mlpab run` resolves the committed manifest, builds the artifact at preflight
when missing, and installs the interface fresh into each run's own venv.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mlpab import redact

INTERFACES = ("cli", "mcp", "sdk", "none")
_TESTBED_ROOT = Path(__file__).resolve().parents[2]

# Two-way split:
#     configs/platforms/<platform>/ — the platform's CONFIG FOLDER: flat
#         interface manifests (cli.yaml / mcp.yaml / sdk.yaml), skills.yaml
#         (pointer manifest) + skills trees, and platform infra
#         (setup.py / teardown.py — $MLPAB_PLATFORM_DIR points here).
#     build/<platform>/<interface>/ — gitignored BUILD HOME ($INTERFACE_DIR):
#         source checkouts + built wheels/binaries; recreated on demand.
CONFIGS_DIR = _TESTBED_ROOT / "configs" / "platforms"
BUILD_DIR = _TESTBED_ROOT / "build"


@dataclass
class InterfaceSetup:
    platform: str
    interface: str
    prompt_fragment: str
    hash: str
    # Human identity of the interface build under test (the `version` column in
    # the results CSV): the manifest's pinned `ref:` (git SHA, shortened) /
    # `==X.Y.Z` pip pin / `VER=` pin / rendered package.version; "unknown" when
    # nothing is pinned.
    ref: str = ""
    cli_binary: str | None = None
    cli_subcommand: str | None = None
    # Extra on-interface binaries beyond `cli_binary` (e.g. GCP's `bq`, shipped in
    # the same Cloud SDK as `gcloud`). Each is allowed with ANY subcommand in CLI
    # mode. Empty for the usual single-binary CLIs (aws/az/databricks/hops).
    cli_aux_commands: list[str] = field(default_factory=list)
    sdk_module: str | None = None
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    keys: dict[str, str] = field(default_factory=dict)
    serve: list[str] = field(default_factory=list)  # per-run server-start steps
    teardown: list[str] = field(default_factory=list)  # run start+end cleanup steps
    # Outbound hosts THIS interface needs reachable, beyond the testbed baseline
    # (claude API + loopback). Drives the agent sandbox `allowedDomains`. e.g.
    # `["api.openai.com", "*.openai.com"]` for a cloud SDK; empty for local-only.
    allowed_domains: list[str] = field(default_factory=list)
    # Command patterns that must run OUTSIDE the agent sandbox (drives the
    # sandbox `excludedCommands`). Needed for Go binaries on macOS: Seatbelt
    # blocks trustd, so their TLS verification fails with `x509: OSStatus
    # -26276` (the Claude Code docs name `gh`/`gcloud`/`terraform`; same for
    # `databricks`). Excluded commands lose the domain allowlist, so list only
    # binaries that pin their own host; the PreToolUse hook still applies.
    sandbox_excluded_commands: list[str] = field(default_factory=list)
    # Cost control: the ONLY `ml.<family>.<size>` instance types the agent may
    # request (e.g. the AWS Free Tier set for sagemaker). Surfaced to the hook as
    # TESTBED_INSTANCE_ALLOW, which denies any other instance-type token in a
    # tool call. Empty → no restriction (platforms without instance types).
    instance_allowlist: list[str] = field(default_factory=list)


def _norm_subcommands(value: Any) -> str | None:
    """Manifest `cli_subcommand` accepts one service or a list (e.g.
    `[sagemaker, sagemaker-runtime, s3]`). Normalize to a comma-joined string —
    the wire format of TESTBED_CLI_SUBCOMMAND, which the hook and the command
    counter split back into an allowlist."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        joined = ",".join(s for v in value if (s := str(v).strip()))
        return joined or None
    return str(value).strip() or None


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


def manifest_path(platform: str, interface: str) -> Path:
    """The committed flat manifest: configs/platforms/<p>/<interface>.yaml."""
    return CONFIGS_DIR / platform / f"{interface}.yaml"


def platform_interface_from_config(config_path: str | Path) -> tuple[str, str]:
    """Infer (platform, interface) from an interface config path like
    `configs/platforms/<platform>/<interface>.yaml`."""
    p = Path(config_path)
    interface = p.stem
    platform = p.parent.name
    if interface not in INTERFACES:
        raise ValueError(
            f"Unknown interface {interface!r} from {config_path!r}. "
            "Expected a path like configs/platforms/<platform>/<interface>.yaml"
        )
    return platform, interface


def bin_dir(platform: str, interface: str) -> Path:
    """The interface's BUILD HOME: checked-out code + built binary
    ($INTERFACE_DIR), at the gitignored build/<platform>/<interface>/."""
    return BUILD_DIR / platform / interface


# The PREPARED venv lives inside the build home. `prepare()` materializes it
# ONCE (base libs + this interface's runtime_install); every run CLONES it
# read-only, so runs never pip-install or mutate shared state — and the
# cross-session build barrier is unnecessary. `_PREPARED_STAMP` next to it holds
# the manifest+binary hash the venv was built for, so prepare() is idempotent.
_PREPARED_VENV_NAME = "venv"
_PREPARED_STAMP = ".prepared.hash"


def prepared_venv_dir(platform: str, interface: str) -> Path:
    """The interface's PREPARED venv ($INTERFACE_DIR/venv) — see `prepare()`."""
    return bin_dir(platform, interface) / _PREPARED_VENV_NAME


def venv_site_packages(venv_dir: Path) -> Path | None:
    """A venv's `site-packages` for the RUNNING interpreter's version, PINNED —
    not a wildcard `glob`. A venv (incl. the base .venv) can carry stale
    `lib/pythonX.Y/site-packages` trees from a previous interpreter; `glob`
    returns them in arbitrary order, so a clone could copy a 3.13 tree into a
    3.12 venv (ABI mismatch). Per-run venvs are created from the same interpreter
    that runs mlpab, so its `X.Y` is the correct, unambiguous tree. Falls back to
    the sole globbed tree if the pinned one is absent."""
    name = f"python{sys.version_info.major}.{sys.version_info.minor}"
    pinned = venv_dir / "lib" / name / "site-packages"
    if pinned.is_dir():
        return pinned
    return next((venv_dir / "lib").glob("python*/site-packages"), None)


def _base_venv_fingerprint() -> str:
    """Short hash of the base .venv's installed distributions + python version.

    Folded into the prepared-venv stamp: a prepared venv is a CLONE of the base
    .venv plus the interface, so a change to the base (a new/removed/upgraded
    package, a python bump) must invalidate it — otherwise runs clone stale base
    libs. `*.dist-info` / `*.egg-info` directory names carry the package name AND
    version, so the set of them is a faithful, cheap fingerprint of base."""
    base = _TESTBED_ROOT / ".venv"
    h = hashlib.sha256()
    h.update(f"py{sys.version_info.major}.{sys.version_info.minor}".encode())
    sp = venv_site_packages(base)
    if sp and sp.is_dir():
        names = sorted(p.name for p in sp.iterdir() if p.name.endswith((".dist-info", ".egg-info")))
        h.update(b"\0".join(n.encode() for n in names))
    return h.hexdigest()[:8]


def _prepared_stamp_value(platform: str, interface: str) -> str:
    """What the prepared venv was built FOR: the interface content hash (manifest
    + binary) AND the base-venv fingerprint. Either changing means rebuild."""
    return f"{_compute_hash(platform, interface)}:{_base_venv_fingerprint()}"


_PIN_RE = re.compile(r"==([0-9][\w.\-]*)")
_VER_VAR_RE = re.compile(r"\bVER=([0-9][\w.]*)")


def interface_ref(manifest: dict[str, Any]) -> str:
    """See InterfaceSetup.ref — the version/ref identity of an interface build."""
    # An explicit `version:` in the manifest wins over any derivation.
    if manifest.get("version"):
        return str(manifest["version"])
    ref = manifest.get("ref")
    if ref:
        s = str(ref)
        return s[:12] if re.fullmatch(r"[0-9a-f]{40}", s) else s
    steps = " ".join(
        list(manifest.get("runtime_install") or []) + list(manifest.get("install") or [])
    )
    m = _PIN_RE.search(steps) or _VER_VAR_RE.search(steps)
    if m:
        return m.group(1)
    pkg = manifest.get("package") or {}
    if pkg.get("version"):
        return str(pkg["version"])
    return "unknown"


def load_manifest(platform: str, interface: str) -> dict[str, Any]:
    p = manifest_path(platform, interface)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def _resolved_config(platform: str, interface: str) -> dict[str, Any]:
    """The manifest's run-relevant fields, with defaults normalized."""
    manifest = load_manifest(platform, interface)
    return {
        "binary": manifest.get("binary"),
        "runtime_install": manifest.get("runtime_install") or [],
        # Optional override for the grader's read-client install (see
        # install_for_grader): the checker adapter usually needs only a thin
        # client (e.g. boto3), not the full SDK interface. Defaults to [] →
        # install_for_grader falls back to runtime_install (unchanged behavior).
        "grader_install": manifest.get("grader_install") or [],
        "mcp_servers": manifest.get("mcp_servers") or {},
        "prompt": manifest.get("prompt"),
        # Optional accounting overrides: invokable CLI command (for cli_calls) and
        # importable SDK module (for sdk_calls). Default to `binary` / `platform`
        # below, so existing configs are unchanged.
        "cli_command": manifest.get("cli_command"),
        # Optional subcommand entrypoint(s) — one service or a list (e.g.
        # `[sagemaker, sagemaker-runtime, s3]` with `cli_command: aws`): scopes
        # the CLI interface to `<cli_command> <one of these> …`.
        "cli_subcommand": manifest.get("cli_subcommand"),
        # Optional extra on-interface binaries (e.g. `[bq]` alongside `gcloud`).
        "cli_aux_commands": manifest.get("cli_aux_commands") or [],
        "sdk_module": manifest.get("sdk_module"),
        # Per-run shell steps in the run venv to START the agent's servers (e.g.
        # the MCP HTTP server claude connects to at launch), after stale-server
        # cleanup. `teardown` stops them at run start + end.
        "serve": manifest.get("serve") or [],
        "teardown": manifest.get("teardown") or [],
        # Sandbox/cost controls:
        #   allowed_domains: outbound hosts for the agent sandbox.
        #   instance_allowlist: the only ml.* instance types the agent may
        #     request (e.g. the AWS Free Tier set for sagemaker).
        #   sandbox_excluded_commands: commands run OUTSIDE the sandbox.
        "allowed_domains": manifest.get("allowed_domains") or [],
        "instance_allowlist": manifest.get("instance_allowlist") or [],
        "sandbox_excluded_commands": manifest.get("sandbox_excluded_commands") or [],
    }


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


def _prompt_for(platform: str, interface: str) -> str:
    """The agent-facing prompt: the manifest's `prompt:` (kept verbatim, even
    when intentionally empty), else an auto-generated default."""
    cfg = _resolved_config(platform, interface)
    text = cfg.get("prompt")
    if text is not None:
        return text.strip()
    return _auto_prompt(platform, interface, cfg.get("binary"))


def prompt_hash_for(platform: str, interface: str) -> str:
    if platform == "none" and interface == "none":
        return ""
    if not load_manifest(platform, interface):
        return ""
    text = _prompt_for(platform, interface)
    if not text:
        return ""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


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


def optional_keys(platform: str, interface: str) -> set[str]:
    """Declared keys flagged `optional: true` in the manifest's `keys:` list
    (as `{name: NAME, optional: true}`). These are still resolved and injected
    into the agent env when a value exists, but their ABSENCE does not fail the
    availability gate — used for values a platform `serve:`/setup step provisions
    at run time (e.g. SAGEMAKER_ROLE_ARN, which setup.py creates and exports)."""
    if platform == "none" and interface == "none":
        return set()
    raw = load_manifest(platform, interface).get("keys") or {}
    out: set[str] = set()
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict) and entry.get("name") and entry.get("optional"):
                out.add(str(entry["name"]))
    return out


def _resolved_keys(
    platform: str, interface: str, env: dict[str, str] | None = None
) -> dict[str, str]:
    """Key values to inject into the agent env: manifest value, else env."""
    base = env if env is not None else os.environ
    out: dict[str, str] = {}
    for k, v in keys_for(platform, interface).items():
        out[k] = v or base.get(k, "")
    return {k: v for k, v in out.items() if v}


def missing_keys(platform: str, interface: str, env: dict[str, str] | None = None) -> list[str]:
    """Declared credential keys with no value (neither in the manifest nor the
    env) — the cheap, no-network 'are creds present?' check used by the
    session-start availability gate. Keys flagged `optional` are excluded —
    a platform setup step provisions them at run time."""
    base = env if env is not None else os.environ
    optional = optional_keys(platform, interface)
    return [
        k
        for k, v in keys_for(platform, interface).items()
        if k not in optional and not (v or base.get(k, ""))
    ]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def _compute_hash(platform: str, interface: str) -> str:
    """sha256 over manifest bytes + binary bytes."""
    h = hashlib.sha256()
    mp = manifest_path(platform, interface)
    if mp.exists():
        h.update(mp.read_bytes())
        h.update(b"\0")
    binary = load_manifest(platform, interface).get("binary")
    if binary:
        bpath = bin_dir(platform, interface) / binary
        if bpath.is_file():
            h.update(bpath.read_bytes())
            h.update(b"\0")
    return h.hexdigest()[:8]


def _check_known(platform: str, interface: str) -> None:
    """Validate the interface exists. Raises ValueError."""
    if interface not in INTERFACES:
        raise ValueError(f"Unknown interface {interface!r}; expected one of {INTERFACES}")
    if not load_manifest(platform, interface):
        raise ValueError(
            f"No config for {platform!r}/{interface!r} at {manifest_path(platform, interface)}. "
            f"Create it (configs/platforms/{platform}/{interface}.yaml) — preflight builds the binary."
        )


def variant_for(platform: str, interface: str) -> str:
    """Validate the interface and return its content hash ("" for none/none)."""
    if platform == "none" and interface == "none":
        return ""
    _check_known(platform, interface)
    return _compute_hash(platform, interface)


def install_for_grader(platform: str, run_dir: Path, venv_python: Path) -> None:
    """Install the platform's READ client (the checker adapter in
    evals/adapters/<platform>.py imports it) into the run venv, for grading runs
    whose interface (cli/mcp) didn't already provide it.

    Prefers the SDK manifest's `grader_install` when set (a thin read client —
    e.g. AWS grades through boto3 alone, not the full sagemaker SDK), else falls
    back to the SDK-interface `runtime_install` — that IS the python client the
    adapter reads through for most platforms (e.g. `databricks-sdk`, or the
    hopsworks wheel). Idempotent: pip reports already-satisfied installs as
    no-ops, so an SDK-interface run (client already present) costs nothing.
    Called AFTER the agent finishes, so it never relaxes the agent's
    interface-only confinement.
    """
    sdk_cfg = _resolved_config(platform, "sdk")
    steps = sdk_cfg.get("grader_install") or sdk_cfg.get("runtime_install") or []
    if steps:
        _run_install(
            steps, cwd=run_dir, venv_python=venv_python, interface_dir=bin_dir(platform, "sdk")
        )


def setup(
    platform: str,
    interface: str,
    run_dir: Path,
    venv_python: Path,
    run_install: bool = True,
) -> InterfaceSetup:
    """Resolve the interface for a run. `run_install=True` runs the manifest's
    `runtime_install` into the run venv (the legacy per-run install). Pass
    False when the run venv was cloned from a PREPARED venv (see `prepare()`) —
    the interface is already installed, so there's nothing to install per run."""
    if interface not in INTERFACES:
        raise ValueError(f"Unknown interface {interface!r}; expected one of {INTERFACES}")

    if platform == "none" and interface == "none":
        return InterfaceSetup(platform="none", interface="none", prompt_fragment="", hash="")

    _check_known(platform, interface)
    cfg = _resolved_config(platform, interface)
    binary = cfg.get("binary")
    runtime_install = cfg.get("runtime_install") or []
    mcp_servers = cfg.get("mcp_servers") or {}
    prompt_fragment = _prompt_for(platform, interface)
    # The "use the interface as-is" rule is the agent prompt's default (baked
    # into agent.md, stripped only for none/none) — not appended here. This
    # fragment carries just the interface's own `prompt:` prose.
    hash_ = _compute_hash(platform, interface)
    interface_dir = bin_dir(platform, interface)

    # Guard: if runtime steps reference $INTERFACE_DIR (pre-built binary), it must
    # exist. Preflight builds it before here; this is a backstop. Only relevant
    # when we actually run the install steps (run_install); a prepared-venv clone
    # already baked them in.
    if (
        run_install
        and binary
        and runtime_install
        and any("$INTERFACE_DIR" in s for s in runtime_install)
    ):
        if not (interface_dir / binary).exists():
            raise RuntimeError(
                f"Interface {platform!r}/{interface!r} binary '{binary}' not found at "
                f"{interface_dir / binary}. Preflight should have built it; "
                f"check configs/platforms/{platform}/{interface}.yaml install steps."
            )

    if run_install and runtime_install:
        _run_install(
            runtime_install, cwd=run_dir, venv_python=venv_python, interface_dir=interface_dir
        )

    return InterfaceSetup(
        platform=platform,
        interface=interface,
        ref=interface_ref(load_manifest(platform, interface)),
        prompt_fragment=prompt_fragment,
        hash=hash_,
        # Resolve BOTH the CLI command and SDK module for EVERY interface (not just
        # the active one), so the agent hook + command counter can detect and
        # block CROSS-INTERFACE escapes — e.g. `import hopsworks` while the CLI is
        # under test, or a `hops` call while the SDK is. `interface` decides which is
        # "on-interface".
        #   cli_binary: the invokable command (`cli_command`; for a CLI the bare
        #     binary name also works, but a wheel needs the explicit field).
        #   sdk_module: the importable module (`sdk_module`, else the platform name
        #     — true when package == platform name).
        cli_binary=cfg.get("cli_command") or (binary if interface == "cli" else None),
        cli_subcommand=_norm_subcommands(cfg.get("cli_subcommand")),
        cli_aux_commands=[str(b).strip() for b in (cfg.get("cli_aux_commands") or []) if str(b).strip()],
        sdk_module=cfg.get("sdk_module") or platform,
        mcp_servers=mcp_servers,
        keys=_resolved_keys(platform, interface),
        serve=cfg.get("serve") or [],
        teardown=cfg.get("teardown") or [],
        allowed_domains=list(cfg.get("allowed_domains") or []),
        instance_allowlist=list(cfg.get("instance_allowlist") or []),
        sandbox_excluded_commands=list(cfg.get("sandbox_excluded_commands") or []),
    )


def _make_check_venv(target: Path) -> Path:
    """Create an empty venv that shares the base .venv's libraries but holds no
    interface packages — for ephemeral preflight checks (mirrors how each agent
    run's venv is built). Returns the venv's python.
    """
    base_py = _TESTBED_ROOT / ".venv" / "bin" / "python"
    base = base_py if base_py.exists() else Path(sys.executable)
    subprocess.run([str(base), "-m", "venv", "--system-site-packages", str(target)], check=True)
    py = target / "bin" / "python"
    # Expose the base venv's site-packages (shared libs) explicitly, like runner.
    if base_py.exists():
        vsp = venv_site_packages(target)
        bsp = venv_site_packages(_TESTBED_ROOT / ".venv")
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


def ensure_base_clean(platform: str, interface: str) -> None:
    """Keep the base .venv free of the interface package, so every per-run (and
    preflight check) venv installs the wheel FRESH and complete.

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
        _check_known(platform, interface)
        cfg = _resolved_config(platform, interface)
    except Exception:
        return
    dist = _interface_dist_name(cfg, platform, cfg.get("binary"))
    if not dist:
        return
    # Installed IN base? `pip show` exits 0 iff the distribution is present.
    present = (
        subprocess.run(
            [str(base_py), "-m", "pip", "show", dist],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        ).returncode
        == 0
    )
    if not present:
        return
    subprocess.run(
        [str(base_py), "-m", "pip", "uninstall", "-y", dist],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=120,
    )
    print(
        f"[mlpab] removed interface package {dist!r} from the base .venv — it must "
        f"stay free of interface packages so each run installs the wheel fresh.",
        file=sys.stderr,
        flush=True,
    )


def preflight(
    platform: str,
    interface: str,
    *,
    check_login: bool = True,
    auto_build: bool = True,
    timeout_s: int = 120,
    cleanup_build: bool = False,
    env: dict[str, str] | None = None,
) -> InterfaceStatus:
    """Build + verify an interface. `cleanup_build=True` (`mlpab test`)
    afterwards deletes any build artifacts this created from the committed
    source folder, keeping it source-only. Default False leaves the artifact in
    place (treatment runs put the agent against it directly)."""
    try:
        return _preflight_impl(
            platform,
            interface,
            check_login=check_login,
            auto_build=auto_build,
            timeout_s=timeout_s,
            env=env,
            # `mlpab test` tears the build home down right after, so don't bother
            # materializing a prepared venv only to delete it.
            prepare_venv=not cleanup_build,
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
        os.chmod(path, 0o700)  # the top dir itself, so its entries unlink
    except OSError:
        pass
    for root, dirs, files in os.walk(path):
        for name in files + dirs:
            try:
                os.chmod(os.path.join(root, name), 0o700)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


@contextmanager
def _build_lock(lock_dir: Path):
    """Hold an EXCLUSIVE inter-process lock for the duration of a shared-venv
    rebuild. The same venv path (a platform's plumbing venv, or an interface's
    prepared venv) is shared by every run of that platform/interface, so when
    several runs start as separate processes (e.g. five `mlpab start` tmux
    sessions launched at once) their preflights would otherwise all `_force_rmtree`
    + reinstall the SAME directory concurrently — wiping each other mid-build, so
    a `setup.py verify` in one process imports a half-installed client and dies
    with "No module named '<client>'". The lock serializes the rebuild: the first
    process builds while the rest wait, then everyone re-checks the stamp under
    the lock (double-checked) and reuses the finished venv. The lockfile lives
    NEXT TO the venv (never inside it) so `_force_rmtree(venv_dir)` can't remove
    the fd out from under a waiter."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ".build.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _clean_build_artifacts(platform: str, interface: str) -> None:
    """Remove build artifacts from the interface's build home (build/<p>/<i>) —
    used by `mlpab test` to leave no state behind. A wheel, build|dist|src dir,
    egg-info, or the configured `binary` is always a build artifact."""
    base = BUILD_DIR / platform / interface
    if not base.is_dir():
        return
    for d in ("build", "dist", "src", _PREPARED_VENV_NAME):
        _force_rmtree(base / d)
    (base / _PREPARED_STAMP).unlink(missing_ok=True)
    for p in list(base.glob("*.egg-info")):
        _force_rmtree(p)
    for p in list(base.glob("*.whl")):
        p.unlink(missing_ok=True)
    manifest = load_manifest(platform, interface)
    binary = manifest.get("binary")
    if binary and not binary.endswith((".py", ".toml", ".md", ".yaml", ".yml")):
        (base / binary).unlink(missing_ok=True)
    # With build metadata in config.yaml (`package:`), pyproject.toml is
    # build-generated — remove it so the build folder stays source-only.
    if manifest.get("package"):
        (base / "pyproject.toml").unlink(missing_ok=True)


def _preflight_impl(
    platform: str,
    interface: str,
    *,
    check_login: bool = True,
    auto_build: bool = True,
    timeout_s: int = 120,
    env: dict[str, str] | None = None,
    prepare_venv: bool = True,
) -> InterfaceStatus:
    """Build (if needed), then verify an interface is installed + logged in + healthy.

    Config-declared interfaces are binaries built, set up, and authenticated at
    preflight — no manual `mlpab install`. Concretely:

      * Build — if the binary is missing, run the config's `install:` steps
        (deterministic, no AI). Disable with auto_build=False.
      * Login — `auth_command` exits 0 run non-interactively with the declared keys
        in env; without an `auth_command`, login is satisfied only when every
        declared key is set.
      * Test — `test_command` exits 0.
    """
    config_fix = f"check configs/platforms/{platform}/{interface}.yaml (build/install steps)"
    setup_fix = "make setup  (or: mlpab setup " + f"configs/platforms/{platform}/{interface}.yaml)"

    if platform == "none" and interface == "none":
        return InterfaceStatus(platform, interface, ok=True, installed=True, authenticated=True)
    if interface not in INTERFACES:
        return InterfaceStatus(
            platform,
            interface,
            ok=False,
            installed=False,
            authenticated=False,
            reason=f"unknown interface {interface!r}",
        )

    # 1) installed? Build the binary on demand if it's missing.
    try:
        _check_known(platform, interface)
    except ValueError as e:
        return InterfaceStatus(
            platform,
            interface,
            ok=False,
            installed=False,
            authenticated=False,
            reason=str(e),
            fix_command=config_fix,
        )
    cfg = _resolved_config(platform, interface)
    binary = cfg.get("binary")
    # The base .venv must never hold the interface package — it would shadow the
    # fresh per-run wheel install (check + run venvs both derive from base). Strip
    # it here so the checks below mirror a real run.
    ensure_base_clean(platform, interface)
    runtime_install = cfg.get("runtime_install") or []
    uses_prebuilt = bool(
        binary and runtime_install and any("$INTERFACE_DIR" in s for s in runtime_install)
    )
    bpath = bin_dir(platform, interface) / binary if uses_prebuilt else None

    # (Re)build only when the artifact is missing. The config's `install:` steps
    # only BUILD the artifact into the interface dir — they don't install it into
    # the base .venv, so base stays free of interface packages and runs don't
    # overlap. The interface installs fresh into a throwaway venv for the checks
    # below, and into each agent's per-run venv at run time.
    artifact_missing = bpath is not None and not bpath.exists()
    if artifact_missing and auto_build and load_manifest(platform, interface).get("install"):
        try:
            build(platform, interface)
        except Exception as e:  # build is shell-out heavy; surface failures
            return InterfaceStatus(
                platform,
                interface,
                ok=False,
                installed=False,
                authenticated=False,
                reason=f"build failed: {e}",
                fix_command=config_fix,
            )
    if bpath is not None and not bpath.exists():
        return InterfaceStatus(
            platform,
            interface,
            ok=False,
            installed=False,
            authenticated=False,
            reason=f"binary {binary!r} missing at {bpath}",
            fix_command=config_fix,
        )

    # Materialize the PREPARED venv once, here in the (serial) setup phase — runs
    # then CLONE it read-only, so no run pip-installs or mutates shared state and
    # the cross-session build barrier is unnecessary. Idempotent (hash-stamped).
    if prepare_venv and auto_build:
        try:
            prepare(platform, interface)
        except Exception as e:  # shell-out heavy; surface failures like build
            return InterfaceStatus(
                platform,
                interface,
                ok=False,
                installed=False,
                authenticated=False,
                reason=f"prepare failed: {e}",
                fix_command=config_fix,
            )

    # Verify in an EPHEMERAL venv that shares base libs but installs ONLY this
    # interface (its runtime_install) — exactly what an agent run gets — then is
    # torn down (keeps base free of interface packages). Treatment preflight runs
    # only build + `test_command`; LOGIN is verified per run
    # (interfaces.login_status). check_login=True (single-task form) also checks
    # login here.
    keys = keys_for(platform, interface)
    optional = optional_keys(platform, interface)
    base_env = dict(env) if env is not None else dict(os.environ)
    merged_env = dict(base_env)
    missing = []
    for k, declared in keys.items():
        val = declared or base_env.get(k, "")
        if val:
            merged_env[k] = val
        elif k not in optional:
            missing.append(k)

    manifest = load_manifest(platform, interface)
    auth_command = manifest.get("auth_command") if check_login else None
    test_command = manifest.get("test_command")
    # No auth_command → login is satisfied only when every declared key is present.
    if check_login and not auth_command and keys and missing:
        return InterfaceStatus(
            platform,
            interface,
            ok=False,
            installed=True,
            authenticated=False,
            missing_keys=missing,
            reason=f"missing credential key(s): {', '.join(missing)}",
            fix_command=setup_fix,
        )
    # Nothing to verify in a venv (no login to check here, no test) → done.
    if not auth_command and not test_command:
        return InterfaceStatus(
            platform,
            interface,
            ok=True,
            installed=True,
            authenticated=True,
            missing_keys=missing,
        )

    # Only stand up a throwaway venv when there's something to install (real
    # interfaces); else run checks against the base bin (cheap, nothing to isolate).
    # The venv installs ONLY this interface and is torn down.
    tmp = None
    try:
        if runtime_install:
            tmp = Path(tempfile.mkdtemp(prefix="mlpab-preflight-"))
            check_python = _make_check_venv(tmp / "venv")
            try:
                _run_install(
                    runtime_install,
                    cwd=tmp,
                    venv_python=check_python,
                    interface_dir=bin_dir(platform, interface),
                )
            except Exception as e:
                return InterfaceStatus(
                    platform,
                    interface,
                    ok=False,
                    installed=False,
                    authenticated=False,
                    reason=f"interface install failed: {e}",
                    fix_command=config_fix,
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
                platform,
                interface,
                ok=False,
                installed=True,
                authenticated=False,
                missing_keys=missing,
                reason=f"login check failed (`{auth_command}` returned non-zero or timed out)",
                fix_command=setup_fix,
            )
        if test_command and not _run_check(test_command, merged_env, timeout_s):
            return InterfaceStatus(
                platform,
                interface,
                ok=False,
                installed=True,
                authenticated=True,
                missing_keys=missing,
                reason=f"interface did not run reliably (`{test_command}` returned non-zero or timed out)",
                fix_command=config_fix,
            )
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

    return InterfaceStatus(
        platform,
        interface,
        ok=True,
        installed=True,
        authenticated=True,
        missing_keys=missing,
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
            command,
            shell=True,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            print(
                f"[mlpab] check FAILED (exit {proc.returncode}): {command}\n"
                f"--- captured output ---\n"
                f"{redact.redact(proc.stdout.decode('utf-8', 'replace'))}"
                f"\n--- end ---",
                file=_sys.stderr,
                flush=True,
            )
            return False
        return True
    except subprocess.TimeoutExpired as e:
        out = (e.output or b"").decode("utf-8", "replace") if e.output else "(no output)"
        print(
            f"[mlpab] check TIMED OUT after {timeout_s}s: {command}\n"
            f"--- captured output before timeout ---\n{redact.redact(out)}\n--- end ---",
            file=_sys.stderr,
            flush=True,
        )
        return False
    except OSError as e:
        print(f"[mlpab] check OS error: {command}: {e}", file=_sys.stderr, flush=True)
        return False


def login_status(
    platform: str,
    interface: str,
    *,
    venv_python: Path,
    keys: dict[str, str] | None = None,
    timeout_s: int = 90,
    attempts: int = 3,
) -> InterfaceStatus:
    """Per-run login check: run the interface's `auth_command` in the given
    (per-run) venv. Build + test happen once at treatment preflight; login is
    re-verified for EVERY run here (so expired creds are caught mid-session
    and each run authenticates in its own venv).

    The auth_command makes a LIVE round-trip to the platform (e.g. `hops login`
    authenticates against the Hopsworks cluster). When several configs run in
    parallel against ONE cluster/API key (three `mlpab start` terminals, or
    `concurrency:`), that cluster is hit by N× simultaneous login + project
    create/teardown + agent traffic, so a single login can take far longer than
    a quiet-cluster round-trip. A tight, no-retry check then times out and
    aborts the run before the agent even starts. So the check is retried up to
    `attempts` times with a generous per-attempt `timeout_s` and a short
    backoff — a transient slow/refused login under load is absorbed, while a
    genuinely bad credential still fails on every attempt and aborts cleanly.
    """
    if platform == "none" and interface == "none":
        return InterfaceStatus(platform, interface, ok=True, installed=True, authenticated=True)
    setup_fix = "make setup  (or: mlpab setup " + f"configs/platforms/{platform}/{interface}.yaml)"
    declared = keys_for(platform, interface)
    env = dict(os.environ)
    # Inject EVERY caller-provided value so the login check runs in the same
    # credential env the agent will get — including platform-exported runtime
    # values that aren't declared keys (e.g. a CLOUDSDK_AUTH_ACCESS_TOKEN that
    # setup.py minted from ADC for gcloud). Declared keys are still resolved +
    # missing-checked below.
    for k, v in (keys or {}).items():
        if v:
            env[k] = v
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
                platform,
                interface,
                ok=False,
                installed=True,
                authenticated=False,
                missing_keys=missing,
                reason=f"missing credential key(s): {', '.join(missing)}",
                fix_command=setup_fix,
            )
        return InterfaceStatus(platform, interface, ok=True, installed=True, authenticated=True)

    env["PATH"] = os.pathsep.join(
        [str(bin_dir(platform, interface)), str(venv_python.parent), env.get("PATH", "")]
    )
    env["VIRTUAL_ENV"] = str(venv_python.parent.parent)
    # Retry on transient failure/timeout (cluster slow under parallel load).
    # Exponential-ish backoff (2s, 4s, …) between attempts; the last attempt's
    # captured output is what _run_check already printed for the operator.
    ok = False
    for attempt in range(1, max(1, attempts) + 1):
        if _run_check(auth_command, env, timeout_s):
            ok = True
            break
        if attempt < attempts:
            print(
                f"[mlpab] login check attempt {attempt}/{attempts} failed; "
                f"retrying in {2 * attempt}s …",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(2 * attempt)
    if not ok:
        return InterfaceStatus(
            platform,
            interface,
            ok=False,
            installed=True,
            authenticated=False,
            missing_keys=missing,
            reason=(
                f"login check failed (`{auth_command}` returned non-zero or "
                f"timed out on all {attempts} attempt(s))"
            ),
            fix_command=setup_fix,
        )
    return InterfaceStatus(
        platform,
        interface,
        ok=True,
        installed=True,
        authenticated=True,
        missing_keys=missing,
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
        lines.append(f"dependencies = {arr(pkg['dependencies'])}")
    if pkg.get("scripts"):
        lines += ["", "[project.scripts]"] + [f'{k} = "{v}"' for k, v in pkg["scripts"].items()]
    if pkg.get("optional-dependencies"):
        lines += ["", "[project.optional-dependencies]"] + [
            f"{k} = {arr(v)}" for k, v in pkg["optional-dependencies"].items()
        ]
    if pkg.get("packages"):
        lines += ["", "[tool.setuptools]", f"packages = {arr(pkg['packages'])}"]
    return "\n".join(lines) + "\n"


def build(platform: str, interface: str) -> Path:
    """Check out the interface's repo into its folder and build it (no AI).

    Code is checked out INTO build/<platform>/<interface>/ — both the checkout
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
    iface_dir = bin_dir(platform, interface)  # source checkout + built artifact
    iface_dir.mkdir(parents=True, exist_ok=True)
    venv_bin = _TESTBED_ROOT / ".venv" / "bin"

    # Where install steps run. No `repo:` → source already in place. A remote
    # `repo:` is cloned into a `src/` subdir; a local path is copied in place.
    src_cwd = iface_dir
    if repo:
        local = _TESTBED_ROOT / repo
        if local.is_dir():
            shutil.copytree(local, iface_dir, dirs_exist_ok=True)
        else:
            src = iface_dir / "src"
            # Pin source to `ref` — a FIXED COMMIT SHA (preferred, reproducible) or
            # a branch/tag. Once checked out we NEVER `git pull`: the tree stays at
            # the pinned commit, so the benchmark isn't confounded by upstream
            # drift between runs. To refresh, delete src/.
            if not (src / ".git").exists():
                src.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "-C", str(src), "init", "-q"], check=True, timeout=60)
                subprocess.run(
                    ["git", "-C", str(src), "remote", "add", "origin", repo],
                    check=True,
                    timeout=60,
                )
                # Fetch just the pinned ref at depth 1 (GitHub serves a reachable
                # SHA this way), then check out detached. Network — cap it so a
                # stalled fetch can't hang the build indefinitely.
                subprocess.run(
                    ["git", "-C", str(src), "fetch", "--depth", "1", "origin", ref],
                    check=True,
                    timeout=600,
                )
                subprocess.run(
                    ["git", "-C", str(src), "checkout", "-q", "FETCH_HEAD"],
                    check=True,
                    timeout=120,
                )
            src_cwd = src
        # Always strip a cloned/copied repo's own agent instructions (.claude/,
        # CLAUDE.md, .mcp.json) — e.g. hopsworks-api ships a .claude/ — so they're
        # never auto-loaded as directives by the agent working in or near the
        # interface source.
        from mlpab import docs as _docs

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
        # agent (deny patterns), triggering a pip warning otherwise.
        env["PIP_NO_CACHE_DIR"] = "1"
        for step in install_steps:
            # Generous cap (build/install can be minutes) that still bounds a
            # truly hung step.
            subprocess.run(step, shell=True, env=env, cwd=str(src_cwd), check=True, timeout=1800)
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
    # agent (deny patterns), and runtime_install steps are tiny.
    env["PIP_NO_CACHE_DIR"] = "1"
    for step in steps:
        subprocess.run(step, shell=True, cwd=cwd, env=env, check=True, timeout=1800)


def _materialize_venv(target: Path) -> Path:
    """Create a venv at `target` that FULLY materializes the base .venv's
    site-packages inside it — no `.pth` to an outside path — so it can be CLONED
    into a sandboxed run dir and stand on its own. Returns its python.

    Mirrors runner._make_venv's base path: the per-run venv is later a clone of
    this, so the two must produce the same self-contained layout.
    """
    base_py = _TESTBED_ROOT / ".venv" / "bin" / "python"
    base = base_py if base_py.exists() else Path(sys.executable)
    subprocess.run(
        [str(base), "-m", "venv", "--system-site-packages", str(target)],
        check=True,
        timeout=300,
    )
    py = target / "bin" / "python"
    if base_py.exists():
        vsp = venv_site_packages(target)
        bsp = venv_site_packages(_TESTBED_ROOT / ".venv")
        if vsp and bsp:
            for child in bsp.iterdir():
                dst = vsp / child.name
                if dst.exists():
                    continue
                try:
                    subprocess.run(["cp", "-Rc", str(child), str(dst)], check=True, timeout=600)
                except subprocess.CalledProcessError:
                    subprocess.run(["cp", "-R", str(child), str(dst)], check=True, timeout=600)
    return py


def prepare(platform: str, interface: str, *, force: bool = False) -> Path | None:
    """Build the artifact (if missing) AND materialize the interface's PREPARED
    venv ONCE, so each run only CLONES it (read-only): no per-run pip install, no
    shared-state mutation, no cross-session build barrier. Returns the prepared
    venv dir (None for none/none).

    Idempotent: a prepared venv whose stamp matches the current manifest+binary
    hash is reused. `force=True` rebuilds unconditionally.
    """
    if platform == "none" and interface == "none":
        return None
    _check_known(platform, interface)
    cfg = _resolved_config(platform, interface)
    binary = cfg.get("binary")
    runtime_install = cfg.get("runtime_install") or []
    iface_dir = bin_dir(platform, interface)

    # Build the pinned artifact first (idempotent; only when missing) so the hash
    # below — and the runtime_install steps that reference $INTERFACE_DIR — see it.
    uses_prebuilt = bool(
        binary and runtime_install and any("$INTERFACE_DIR" in s for s in runtime_install)
    )
    if (
        uses_prebuilt
        and not (iface_dir / binary).exists()
        and load_manifest(platform, interface).get("install")
    ):
        build(platform, interface)

    venv_dir = prepared_venv_dir(platform, interface)
    stamp = iface_dir / _PREPARED_STAMP
    want = _prepared_stamp_value(platform, interface)

    def _ready() -> bool:
        return (
            not force
            and (venv_dir / "bin" / "python").exists()
            and stamp.exists()
            and stamp.read_text().strip() == want
        )

    if _ready():
        return venv_dir

    # Serialize the rebuild across processes (parallel `mlpab start` sessions
    # share this prepared venv) and re-check under the lock so a process that
    # already finished the build isn't wiped out from under a concurrent run.
    with _build_lock(iface_dir):
        if _ready():
            return venv_dir
        # (Re)build the prepared venv from scratch — a stale or partial one must
        # not linger. The stamp is written last, so a crash mid-build is
        # re-detected as "not prepared" (missing/mismatched stamp) next time.
        _force_rmtree(venv_dir)
        stamp.unlink(missing_ok=True)
        venv_python = _materialize_venv(venv_dir)
        if runtime_install:
            # cwd = iface_dir so a step's `./venv` IS this prepared venv (the
            # databricks/gcp CLIs copy a binary/SDK tree into `./venv`), and
            # $INTERFACE_DIR = iface_dir (pinned wheels/binaries live there).
            _run_install(
                runtime_install, cwd=iface_dir, venv_python=venv_python, interface_dir=iface_dir
            )
        stamp.write_text(want + "\n")
    return venv_dir


# ---------------------------------------------------------------------------
# Plumbing venv
# ---------------------------------------------------------------------------
# A platform's setup/verify/teardown scripts (configs/platforms/<p>/setup.py,
# teardown.py) talk to the platform through its PYTHON client (e.g. `from
# google.cloud import bigquery`) — this is fixed mlpab plumbing, NOT the agent's
# work. On an SDK-interface run the run venv already has that client, but a
# CLI/MCP run venv does NOT (and gcp's CLI ships as a prebuilt tarball, so its
# run venv has no python client AT ALL → `setup.py verify` died with "No module
# named 'google'", aborting the whole config). Installing the client into the
# run venv would defeat the interface-purity measurement (the agent could then
# import it and bypass the CLI). So plumbing gets its OWN venv, built once at
# build/<platform>/_plumbing/venv, holding exactly the SDK interface's pinned
# client (the SAME deps `install_for_grader` uses to read deliverables back) —
# no new or changed dependency versions. Used by serve/teardown/verify
# REGARDLESS of the interface under test; the agent's run venv stays pure.
_PLUMBING_STAMP = ".plumbing.hash"


def plumbing_venv_dir(platform: str) -> Path:
    """The platform's PLUMBING venv ($BUILD/<platform>/_plumbing/venv) — see
    `prepare_plumbing`."""
    return BUILD_DIR / platform / "_plumbing" / _PREPARED_VENV_NAME


def _plumbing_steps(platform: str) -> list[str]:
    """The install steps that put the platform's python client in the plumbing
    venv: the SDK interface's `grader_install` (a thin read client) or its
    `runtime_install` — identical to `install_for_grader`'s source, since both
    need the same client. Empty when the platform has no SDK interface (then
    there is no plumbing venv and plumbing falls back to the run venv python)."""
    sdk_path = manifest_path(platform, "sdk")
    if not sdk_path.exists():
        return []
    sdk_cfg = _resolved_config(platform, "sdk")
    return list(sdk_cfg.get("grader_install") or sdk_cfg.get("runtime_install") or [])


def prepare_plumbing(platform: str, *, force: bool = False) -> Path | None:
    """Materialize the platform's plumbing venv ONCE (idempotent via a stamp) and
    return its python, or None when the platform needs none (no SDK interface /
    no install steps). Cheap to call per run: an up-to-date venv is reused.

    Built like a prepared venv — a self-contained clone of the base .venv (so it
    stands alone) plus the SDK client. The stamp is the steps + base fingerprint,
    so a base-venv change or an edit to the SDK client pins rebuilds it."""
    if platform == "none":
        return None
    steps = _plumbing_steps(platform)
    if not steps:
        return None
    venv_dir = plumbing_venv_dir(platform)
    stamp = venv_dir.parent / _PLUMBING_STAMP
    h = hashlib.sha256()
    h.update("\0".join(steps).encode())
    h.update(_base_venv_fingerprint().encode())
    want = h.hexdigest()[:16]
    def _ready() -> bool:
        return (
            not force
            and (venv_dir / "bin" / "python").exists()
            and stamp.exists()
            and stamp.read_text().strip() == want
        )

    if _ready():
        return venv_dir / "bin" / "python"
    # Serialize the rebuild across processes (parallel `mlpab start` sessions all
    # share this venv) and re-check under the lock: if another process finished
    # the build while we waited, reuse it instead of wiping it out from under it.
    with _build_lock(venv_dir.parent):
        if _ready():
            return venv_dir / "bin" / "python"
        # Rebuild from scratch — a stale/partial venv must not linger; stamp is
        # written last so a crash mid-build re-detects as "not prepared".
        _force_rmtree(venv_dir)
        stamp.unlink(missing_ok=True)
        venv_python = _materialize_venv(venv_dir)
        _run_install(
            steps,
            cwd=venv_dir.parent,
            venv_python=venv_python,
            interface_dir=bin_dir(platform, "sdk"),
        )
        stamp.write_text(want + "\n")
    return venv_python


def plumbing_python(platform: str) -> Path | None:
    """The platform's plumbing-venv python if already built, else None (caller
    falls back to the run venv python). Does NOT build — use `prepare_plumbing`
    for that."""
    py = plumbing_venv_dir(platform) / "bin" / "python"
    return py if py.exists() else None
