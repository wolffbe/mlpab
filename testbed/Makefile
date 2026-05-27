.PHONY: help install setup test clean reset venv check-git

# Pin to a Python that orjson/PyO3 supports. PyO3 0.23 caps at 3.13; pip-built
# orjson on 3.14 fails. Override with `make install PYTHON=python3.13`.
PYTHON ?= $(shell command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3.10 || echo python3)
VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
BANTER := $(VENV)/bin/banter

help:
	@echo "banter — MLE-bench testbed driver"
	@echo ""
	@echo "Targets:"
	@echo "  make install   — create venv, install banter + mle-bench, then run setup"
	@echo "  make setup     — auth for engineer + researcher (Kaggle, Claude) + interface keys"
	@echo "  make test      — run the unit tests"
	@echo "  make clean     — remove .venv and runs/"
	@echo "  make reset     — clean then reinstall"
	@echo ""
	@echo "Run a config:  $(BANTER) configs/<name>.yaml"

check-git:
	@command -v git >/dev/null 2>&1 || { echo "ERROR: git is not installed. Install git and retry."; exit 1; }
	@echo ">> git OK ($(shell git --version))"

install: check-git $(VENV)/.installed setup

$(VENV)/.installed: pyproject.toml
	@echo ">> using interpreter: $(PYTHON)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .
	$(PIP) install git+https://github.com/openai/mle-bench.git
	touch $(VENV)/.installed

venv: $(VENV)/.installed

setup: venv
	@$(BANTER) setup

test: venv
	$(PY) -m unittest discover -s tests -p 'test_*.py' -v

clean:
	rm -rf $(VENV) runs/ src/banter.egg-info src/banter/__pycache__ */__pycache__

reset: clean install
