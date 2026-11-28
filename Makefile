.PHONY: install install-pipeline lint format test verify analyze relabel clean

install:            ## core + dev tools (no deep-learning stack)
	pip install -e ".[dev]"

install-pipeline:   ## everything needed to run Florence-2 / BART-MNLI (CPU wheels)
	pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
	pip install -e ".[pipeline,analysis,review,dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest -m "not slow"

verify:             ## checksums + byte-identical regeneration of the original CSV files
	zsd data verify

relabel:            ## re-score archived strings with BART-MNLI (hours on CPU)
	zsd relabel --mode legacy
	zsd relabel --mode segmented --coverage 0.95 --include-reviewed

analyze:            ## regenerate results/ (summary.json, REPORT.md, figures)
	zsd analyze --n-boot 2000 --seed 2025

clean:
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
