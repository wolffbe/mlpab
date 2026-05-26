.PHONY: help install setup clean reset venv

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
	@echo "  make setup     — interactive credential setup (Kaggle, Claude Code auth)"
	@echo "  make clean     — remove .venv and runs/"
	@echo "  make reset     — clean then reinstall"

install: $(VENV)/.installed setup

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

clean:
	rm -rf $(VENV) runs/ src/banter.egg-info src/banter/__pycache__ */__pycache__

reset: clean install
