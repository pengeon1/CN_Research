# Makefile for CN Research Project
# Multi-Slice Privacy-Aware Traffic Forecasting at RAN Level (IEEE TNSM 2023)
# Replication, Critical Audit, and Resilient Extensions

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest

.PHONY: help venv install test baselines fl clustering improvements figures report slides package clean all

help:
	@echo "Available make targets:"
	@echo "  venv         : Create Python virtual environment (.venv)"
	@echo "  install      : Install pinned dependencies into .venv"
	@echo "  test         : Run pytest test suite"
	@echo "  baselines    : Run DLSTM & Centralized LSTM benchmarks across seeds"
	@echo "  fl           : Run Federated Learning (FALSTM, FPLSTM, sweeps)"
	@echo "  clustering   : Run Clustered FL (IC-Trend, IC-Geo, RC-Fixed, RC-Buggy)"
	@echo "  improvements : Run Improvements A, B, C, D multi-seed ablations"
	@echo "  figures      : Generate all publication figures (PDF/PNG) and LaTeX tables"
	@echo "  report       : Compile docs/report/main.pdf via WeasyPrint"
	@echo "  slides       : Compile docs/slides/presentation.pptx via python-pptx"
	@echo "  package      : Bundle all deliverables into submission.zip"
	@echo "  all          : Complete reproduction pipeline from tests to submission.zip"
	@echo "  clean        : Clean temporary files and caches"

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt

test:
	$(PYTEST) tests/ -v

baselines:
	PYTHONPATH=. $(PYTHON) src/experiments/run_baselines.py --stations 57 --days 10 --epochs 5

fl:
	PYTHONPATH=. $(PYTHON) src/experiments/run_fl.py --stations 57 --days 10 --rounds 6 --local_epochs 1

clustering:
	PYTHONPATH=. $(PYTHON) src/experiments/run_clustering.py --stations 57 --days 10 --clusters 3 --rounds 6 --local_epochs 1

improvements:
	PYTHONPATH=. $(PYTHON) src/experiments/run_improvements.py --stations 57 --days 10 --rounds 6 --local_epochs 1

figures:
	PYTHONPATH=. $(PYTHON) src/experiments/generate_tables_figures.py

report:
	PYTHONPATH=. $(PYTHON) docs/report/build_pdf.py

slides:
	PYTHONPATH=. $(PYTHON) docs/slides/build_deck.py

package: figures report slides
	@echo "Assembling submission.zip..."
	rm -f submission.zip
	zip -r submission.zip \
		README.md Makefile requirements.txt \
		docs/report/main.tex docs/report/main.pdf docs/report/*.tex docs/report/build_pdf.py \
		docs/slides/presentation.pptx docs/slides/build_deck.py \
		docs/video_outline.md docs/ORIGINAL_vs_OURS.md docs/PROGRESS.md \
		docs/notes/ \
		src/ configs/ tests/ results/ figures/ \
		-x "*.pyc" -x "*__pycache__*" -x "*.git*" -x ".venv/*" -x "*.pytest_cache/*"
	@echo "Successfully assembled submission.zip"
	@ls -lh submission.zip

all: test figures report slides package
	@echo "======================================================================"
	@echo "All reproduction, verification, documentation, and packaging COMPLETE!"
	@echo "======================================================================"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov /tmp/test_*
