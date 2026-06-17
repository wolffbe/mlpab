.PHONY: help install setup check test lint fmt uninstall venv check-git ml-libs link unlink claude codex vibe \
        run start stop attach status iface-test clean-runs bootstrap-aws bootstrap-azure bootstrap-gcp

# Python sources to lint/format: the two packages + the platform config scripts.
LINT_PATHS := src tests evals configs

# Pin to a Python that orjson/PyO3 supports. PyO3 0.23 caps at 3.13; pip-built
# orjson on 3.14 fails. Override with `make install PYTHON=python3.13`.
PYTHON ?= $(shell command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3.10 || echo python3)
VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
MLPAB := $(VENV)/bin/mlpab

# Where the `mlpab` entrypoint is symlinked so it's on PATH without activating
# the venv. The venv's mlpab script self-activates via its shebang, so a plain
# symlink here is all it takes — no pipx, no host install. ~/.local/bin is the
# standard per-user bin dir and is usually already on PATH.
LOCALBIN := $(HOME)/.local/bin

# Silence the `A new release of pip is available` notice everywhere — make
# install, per-run venv creation, platform `install:` steps. We intentionally
# don't upgrade pip in every venv; just hide the banner.
export PIP_DISABLE_PIP_VERSION_CHECK := 1
export PIP_NO_PYTHON_VERSION_WARNING := 1

# Heavy ML libraries guaranteed available to every agent run live in
# requirements.txt (repo root); they're installed into the base .venv so each
# per-run agent venv inherits them (shared via --system-site-packages) and
# they survive the per-run venv teardown. Edit requirements.txt to change the set.

help:
	@echo "MLPlatformAgentBench (mlpab) — ML-platform benchmarking testbed"
	@echo ""
	@echo "Targets:"
	@echo "  make install   — create venv, install mlpab + evals, link mlpab onto PATH, then run setup"
	@echo "  make setup     — interactive: pick agent engine(s) to authenticate + platform(s) to set up"
	@echo "  make check     — verify a config is runnable: platform + LLM reachable/responsive (CONFIG=...)"
	@echo "  make test      — run the unit tests"
	@echo "  make lint      — ruff check + isort/format check (no changes)"
	@echo "  make fmt       — auto-fix: isort imports, ruff format, ruff --fix"
	@echo "  make uninstall — remove the venv + build artifacts (install state)"
	@echo ""
	@echo "Run a treatment session in tmux (detached from any terminal/Claude):"
	@echo "  mlpab start configs/treatments/hopsworks-cli-sdk-no-skills-opus.yaml"
	@echo "  mlpab status          — list running treatment sessions"
	@echo "  mlpab attach <yaml>   — watch it live (detach: Ctrl-b d)"
	@echo "  mlpab stop   <yaml>   — kill it (the session also dies on its own when the run finishes)"
	@echo ""
	@echo "Run a config inline: mlpab run configs/treatments/<name>.yaml"
	@echo "Clear results:       mlpab clean"

check-git:
	@command -v git >/dev/null 2>&1 || { echo "ERROR: git is not installed. Install git and retry."; exit 1; }
	@echo ">> git OK ($(shell git --version))"

# Enable the committed pre-commit secret guard (.githooks/pre-commit). results/
# and configs/treatments/ are tracked, so this hook is what keeps secrets +
# per-run venvs out of git. Idempotent.
# core.hooksPath is resolved relative to the REPO ROOT, but this project lives
# in a subdir (testbed/) of the repo — so point it at <prefix>.githooks
# (e.g. testbed/.githooks) regardless of where make is run from.
githooks: check-git
	git config core.hooksPath "$$(git rev-parse --show-prefix).githooks"
	@chmod +x .githooks/* 2>/dev/null || true
	@echo ">> pre-commit secret guard enabled (core.hooksPath=$$(git config core.hooksPath))"


install: check-git githooks $(VENV)/.installed link claude codex vibe setup

$(VENV)/.installed: pyproject.toml requirements.txt
	@echo ">> using interpreter: $(PYTHON)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PIP) install -r requirements.txt
	touch $(VENV)/.installed

# Confinement is per-run via .claude/settings.json (sandbox + denies) plus a
# HOME redirect on the env. See `claude_runner._write_settings` for details.

# Claude Code CLI — the PRIMARY agent engine (claude-* models). Best-effort:
# npm channel, official-installer fallback. Auth via `mlpab setup`.
claude:
	@if command -v claude >/dev/null 2>&1; then \
		echo ">> claude OK ($$(claude --version 2>/dev/null))"; \
	elif command -v npm >/dev/null 2>&1; then \
		echo ">> installing @anthropic-ai/claude-code via npm"; npm install -g @anthropic-ai/claude-code; \
	else \
		echo ">> installing claude via official installer"; curl -fsSL https://claude.ai/install.sh | bash || \
		echo ">> WARNING: claude install failed; install Claude Code manually"; \
	fi

# OpenAI Codex CLI — the second agent engine (mlpab dispatches `gpt-*` /
# `*codex*` agent models to it; see codex_runner.py). Best-effort: npm first
# (official channel), Homebrew fallback. Auth comes from OPENAI_API_KEY in .env
# or a one-time `codex login` on the host.
codex:
	@if command -v codex >/dev/null 2>&1; then \
		echo ">> codex OK ($$(codex --version 2>/dev/null))"; \
	elif command -v npm >/dev/null 2>&1; then \
		echo ">> installing @openai/codex via npm"; npm install -g @openai/codex; \
	elif command -v brew >/dev/null 2>&1; then \
		echo ">> installing codex via Homebrew"; brew install codex; \
	else \
		echo ">> WARNING: neither npm nor brew found; install codex manually (npm i -g @openai/codex) to use codex agent models"; \
	fi

# Mistral Vibe CLI — the third agent engine (mlpab dispatches `mistral-*` agent models to it; see mistral_runner.py). Best-effort: uv first
# (official channel), curl-installer fallback. Auth = MISTRAL_API_KEY in .env.
vibe:
	@if command -v vibe >/dev/null 2>&1; then \
		echo ">> vibe OK ($$(vibe --version 2>/dev/null))"; \
	elif command -v uv >/dev/null 2>&1; then \
		echo ">> installing mistral-vibe via uv"; uv tool install mistral-vibe; \
	else \
		echo ">> installing mistral-vibe via curl"; curl -LsSf https://mistral.ai/vibe/install.sh | bash || \
		echo ">> WARNING: vibe install failed; install manually (uv tool install mistral-vibe) to use mistral agent models"; \
	fi

# Install/refresh just the shared libs (requirements.txt) into the existing
# base .venv without rebuilding it — e.g. after editing requirements.txt, or to
# add them to a venv that predates this target.
ml-libs: venv
	$(PIP) install -r requirements.txt

venv: $(VENV)/.installed

# Symlink the venv's mlpab entrypoint onto PATH so `mlpab ...` works from any
# shell without activating the venv. Idempotent; re-points if the venv moved.
link: venv
	@mkdir -p $(LOCALBIN)
	@ln -sf $(abspath $(MLPAB)) $(LOCALBIN)/mlpab
	@echo ">> linked mlpab -> $(LOCALBIN)/mlpab"
	@case ":$$PATH:" in *":$(LOCALBIN):"*) ;; \
		*) echo ">> NOTE: $(LOCALBIN) is not on your PATH — add it to use \`mlpab\` directly";; esac

unlink:
	@rm -f $(LOCALBIN)/mlpab
	@echo ">> removed $(LOCALBIN)/mlpab"

setup: venv
	@$(MLPAB) setup

# Verify a treatment CONFIG can actually run: each platform is reachable +
# responsive (live auth/test) and each model's LLM answers a live probe.
#   make check CONFIG=configs/treatments/aws/aws-claude.yaml
check: venv
	$(MLPAB) check $(CONFIG)

test: venv
	$(PY) -m unittest discover -s tests -p 'test_*.py' -v

# Lint (no changes): ruff check + isort/ruff-format in --check mode. CI-friendly.
lint: venv
	$(VENV)/bin/ruff check $(LINT_PATHS)
	$(VENV)/bin/isort --check-only --diff $(LINT_PATHS)
	$(VENV)/bin/ruff format --check $(LINT_PATHS)

# Auto-fix: sort imports (isort), then format (ruff), then apply ruff's safe
# lint fixes. isort runs FIRST so ruff format only tidies the already-sorted file.
fmt: venv
	$(VENV)/bin/isort $(LINT_PATHS)
	$(VENV)/bin/ruff format $(LINT_PATHS)
	$(VENV)/bin/ruff check --fix $(LINT_PATHS)


uninstall: unlink
	rm -rf $(VENV) build src/mlpab.egg-info src/mlpab/__pycache__ */__pycache__ tests/__pycache__ evals/**/__pycache__
	@echo ">> removed venv + build artifacts. (kept: .env, results/, .azure/, .gcp/ — your data/secrets)"

# --- Agent commands, wrapped (pass CONFIG=path/to/treatment.yaml) -------------
# These call the underlying CLI ($(MLPAB), still usable directly). Examples:
#   make run     CONFIG=configs/treatments/aws/aws-opus-4-8-minimal.yaml
#   make start   CONFIG=configs/treatments/gcp/gcp-opus-4-8-no-skills.yaml
#   make iface-test CONFIG=configs/platforms/azure/sdk.yaml
run: venv
	$(MLPAB) run $(CONFIG) $(ARGS)
start: venv
	$(MLPAB) start $(CONFIG)
stop: venv
	$(MLPAB) stop $(CONFIG)
attach: venv
	$(MLPAB) attach $(CONFIG)
status: venv
	$(MLPAB) status
# `mlpab test` = build + verify ONE interface (distinct from `make test` = unit tests).
iface-test: venv
	$(MLPAB) test $(CONFIG) $(ARGS)
# `mlpab clean` = wipe results/ runs (distinct from `make uninstall`).
clean-runs: venv
	$(MLPAB) clean $(ARGS)

# One-time cloud bootstrap: install CLI, login, create resources, write .env.
# Idempotent. GCP needs a project:  PROJECT=my-proj make bootstrap-gcp
#
# bootstrap-aws prompts for an ADMIN access-key pair (used once, never stored):
# it provisions the SageMaker execution role, the supplement policy, and a
# PowerUser `mlpab-agent` user whose freshly minted key is written to .env. The
# admin identity needs IAM rights to create users/keys/policies/roles — simplest
# is an admin with the AWS-managed AdministratorAccess (or IAMFullAccess). The
# minted agent itself is PowerUser ("everything except IAM").
bootstrap-aws:
	bash configs/platforms/aws/bootstrap.sh
bootstrap-azure:
	bash configs/platforms/azure/bootstrap.sh
bootstrap-gcp:
	bash configs/platforms/gcp/bootstrap.sh
