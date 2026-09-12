# avgen developer entry points.
#
# Everything routes through `uv`, so a target behaves the same on a laptop and
# on a CI runner. `make dev` once, then `make test-fast` in the edit loop.

UV ?= uv
PYTHON_SOURCES := src tests examples benchmarks docs
CPU_TESTS := -m "not gpu and not multigpu"
FAST_TESTS := -m "not gpu and not multigpu and not slow and not optional_deps"
GPU_TESTS := -m "gpu or multigpu"
SIM_WORLD_SIZES ?= 8 64 512 1024

.DEFAULT_GOAL := help
.PHONY: help install dev lint format type test test-fast test-gpu docs docs-serve simulate clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install avgen and its runtime dependencies only
	$(UV) sync

dev: ## Install every extra plus the dev and docs groups, and enable pre-commit
	$(UV) sync --all-extras --group dev --group docs
	$(UV) run pre-commit install

lint: ## Lint with ruff, and verify formatting without changing files
	$(UV) run ruff check $(PYTHON_SOURCES)
	$(UV) run ruff format --check $(PYTHON_SOURCES)

format: ## Apply ruff's fixes and formatting in place
	$(UV) run ruff check --fix $(PYTHON_SOURCES)
	$(UV) run ruff format $(PYTHON_SOURCES)

type: ## Type-check src/avgen with mypy --strict
	$(UV) run mypy

test: ## Run every test that does not need a GPU
	$(UV) run pytest $(CPU_TESTS)

test-fast: ## The pre-push subset: no GPU, no slow, no optional deps
	$(UV) run pytest -x -q -n auto $(FAST_TESTS)

test-gpu: ## Run the GPU and multi-GPU tests on this machine
	$(UV) run pytest $(GPU_TESTS)

docs: ## Build the documentation site into ./site (warnings are errors)
	$(UV) run --group docs mkdocs build --strict

docs-serve: ## Serve the documentation with live reload on :8000
	$(UV) run --group docs mkdocs serve

simulate: ## Check the reference parallelism plans at every reference world size
	$(UV) run python .github/scripts/check_reference_plans.py \
		--world-sizes $(SIM_WORLD_SIZES)

clean: ## Remove build artifacts, caches, and the generated site
	rm -rf build dist site .pytest_cache .mypy_cache .ruff_cache htmlcov \
		.coverage .coverage.* coverage.xml docs/reference
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
