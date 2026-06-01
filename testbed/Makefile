.PHONY: help install setup test uninstall venv check-git ml-libs openmp link unlink

# Pin to a Python that orjson/PyO3 supports. PyO3 0.23 caps at 3.13; pip-built
# orjson on 3.14 fails. Override with `make install PYTHON=python3.13`.
PYTHON ?= $(shell command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3.10 || echo python3)
VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
BANTER := $(VENV)/bin/banter

# Where the `banter` entrypoint is symlinked so it's on PATH without activating
# the venv. The venv's banter script self-activates via its shebang, so a plain
# symlink here is all it takes — no pipx, no host install. ~/.local/bin is the
# standard per-user bin dir and is usually already on PATH.
LOCALBIN := $(HOME)/.local/bin

# Silence the `A new release of pip is available` notice everywhere — make
# install, per-run venv creation, interface `install:` steps. We intentionally
# don't upgrade pip in every venv; just hide the banner.
export PIP_DISABLE_PIP_VERSION_CHECK := 1
export PIP_NO_PYTHON_VERSION_WARNING := 1

# Heavy ML libraries guaranteed available to every engineer run live in
# requirements.txt (repo root); they're installed into the base .venv so each
# per-run engineer venv inherits them (shared via --system-site-packages) and
# they survive the per-run venv teardown. Edit requirements.txt to change the set.

help:
	@echo "banter — MLE-bench testbed driver"
	@echo ""
	@echo "Targets:"
	@echo "  make install   — create venv, install banter + mle-bench, link banter onto PATH, then run setup"
	@echo "  make setup     — auth for engineer + researcher (Kaggle, Claude)"
	@echo "  make test      — run the unit tests"
	@echo "  make test-integration — run live-claude integration tests (real API calls)"
	@echo "  make uninstall — remove the venv + build artifacts (install state)"
	@echo ""
	@echo "Run a config:  banter run interfaces/<project>/<benchmark|autoresearch>/<name>.yaml"
	@echo "Clear results: banter clean"

check-git:
	@command -v git >/dev/null 2>&1 || { echo "ERROR: git is not installed. Install git and retry."; exit 1; }
	@echo ">> git OK ($(shell git --version))"

install: check-git $(VENV)/.installed link setup

$(VENV)/.installed: pyproject.toml requirements.txt
	@echo ">> using interpreter: $(PYTHON)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	$(PIP) install git+https://github.com/openai/mle-bench.git
	$(MAKE) openmp
	$(PIP) install -r requirements.txt
	touch $(VENV)/.installed

# Confinement is per-run via .claude/settings.json (sandbox + denies) plus a
# HOME redirect on the env. See `claude_runner._write_settings` for details.

# OpenMP runtime (libomp) — xgboost/lightgbm wheels need it (and it isn't
# pip-installable on macOS; they import-fail without it). Installed BEFORE
# requirements.txt. Best-effort: Homebrew on macOS; Linux ships libgomp with the
# toolchain so this is a no-op there.
openmp:
	@if command -v brew >/dev/null 2>&1; then \
		echo ">> ensuring libomp (OpenMP) via Homebrew"; brew list libomp >/dev/null 2>&1 || brew install libomp; \
	else \
		echo ">> brew not found; assuming OpenMP (libgomp) is provided by the system"; \
	fi

# Install/refresh just the shared ML libs (requirements.txt) into the existing
# base .venv without rebuilding it — e.g. after editing requirements.txt, or to
# add them to a venv that predates this target.
ml-libs: venv openmp
	$(PIP) install -r requirements.txt

venv: $(VENV)/.installed

# Symlink the venv's banter entrypoint onto PATH so `banter ...` works from any
# shell without activating the venv. Idempotent; re-points if the venv moved.
link: venv
	@mkdir -p $(LOCALBIN)
	@ln -sf $(abspath $(BANTER)) $(LOCALBIN)/banter
	@echo ">> linked banter -> $(LOCALBIN)/banter"
	@case ":$$PATH:" in *":$(LOCALBIN):"*) ;; \
		*) echo ">> NOTE: $(LOCALBIN) is not on your PATH — add it to use \`banter\` directly";; esac

unlink:
	@rm -f $(LOCALBIN)/banter
	@echo ">> removed $(LOCALBIN)/banter"

setup: venv
	@$(BANTER) setup

test: venv
	$(PY) -m unittest discover -s tests -p 'test_*.py' -v

# Live integration tests — real `claude -p` calls against the API. Files
# use the `live_*.py` prefix so `make test` skips them. Each test auto-skips
# without claude+Keychain, so this target is safe on any host.
test-integration: venv
	$(PY) -m unittest discover -s tests/integration -p 'live_*.py' -v

uninstall: unlink
	rm -rf $(VENV) src/banter.egg-info src/banter/__pycache__ */__pycache__
