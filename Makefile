# Makefile -- Real Estate Price Prediction (CS542, Phases 7–9)
# ===================================================================
# Convenience targets for setup, training, evaluation, tests and lint.
#
# Usage:
#   make setup          # create/update env (conda)
#   make train          # run Phase 6–9 sweep (train + eval + artifacts)
#   make eval           # pretty-print summary from artifacts
#   make test           # run pytest test suite
#   make lint           # run ruff + black
#   make clean          # remove outputs
# ===================================================================

# You can override these on the CLI:
#   make train PYTHON=python3
ENV_NAME   ?= CS542-Project
PYTHON     ?= python
PYTHONPATH ?= src
export PYTHONPATH

# -------------------------------------------------------------------
# Environment / setup
# -------------------------------------------------------------------
.PHONY: setup
setup:
	@echo ">>> Creating or updating conda env '$(ENV_NAME)' from environment.yml"
	@conda env create -f environment.yml -n $(ENV_NAME) 2>/dev/null || \
		conda env update -f environment.yml -n $(ENV_NAME)
	@echo ""
	@echo "Activate the environment with:"
	@echo "    conda activate $(ENV_NAME)"
	@echo ""
	@echo "If you have a pyproject.toml / setup.cfg for 'realestate', install it with:"
	@echo "    pip install -e ."

# -------------------------------------------------------------------
# Training / evaluation
# -------------------------------------------------------------------
.PHONY: train
train:
	@echo ">>> Running Phase 6–9 sweep (train + eval + artifacts) using configs/config.yaml"
	$(PYTHON) -m realestate.train_sweep

# eval: inspect existing artifacts; does NOT retrain
.PHONY: eval
eval:
	@echo ">>> Evaluating latest run artifacts"
	@if [ -f data/outputs/metrics/summary.csv ]; then \
		echo ""; \
		echo "=== metrics/summary.csv (head) ==="; \
		head -n 20 data/outputs/metrics/summary.csv; \
	else \
		echo "No summary.csv found under data/outputs/metrics; run 'make train' first."; \
	fi
	@if [ -d data/outputs/run ]; then \
		LAST_RUN=$$(ls -1 data/outputs/run | sort | tail -n 1); \
		if [ -f "data/outputs/run/$$LAST_RUN/summary.md" ]; then \
			echo ""; \
			echo "=== run/$$LAST_RUN/summary.md (first 40 lines) ==="; \
			head -n 40 "data/outputs/run/$$LAST_RUN/summary.md"; \
		else \
			echo "No summary.md found for latest run ($$LAST_RUN)."; \
		fi; \
	else \
		echo "No run/* directories found under data/outputs/run; run 'make train' first."; \
	fi

# -------------------------------------------------------------------
# Tests and linting
# -------------------------------------------------------------------
.PHONY: test
test:
	@echo ">>> Running pytest"
	pytest -q

.PHONY: lint
lint:
	@echo ">>> Running ruff (lint)"
	ruff check src tests
	@echo ">>> Running black (format)"
	black src tests

# -------------------------------------------------------------------
# Housekeeping
# -------------------------------------------------------------------
.PHONY: clean
clean:
	@echo ">>> Removing outputs/ and intermediate artifacts"
	rm -rf data/outputs
