# Makefile for CN Research Project
# Reproducible research paper analysis & extension (IEEE TNSM 2023)

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest

.PHONY: help venv install test lint clean run-all

help:
	@echo "Available make targets:"
	@echo "  venv       : Create Python virtual environment (.venv)"
	@echo "  install    : Install dependencies into .venv"
	@echo "  test       : Run pytest test suite"
	@echo "  clean      : Clean __pycache__, pytest cache, and temporary files"
	@echo "  run-all    : Reproduce all experiments and generate figures/tables"

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov
