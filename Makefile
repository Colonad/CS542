# Makefile -- Real Estate Price Prediction (CS542, Phase 7)
# ===================================================================
# Convenience targets for setup, training, evaluation, tests and lint.
#
# Usage:
#   make setup          # create/update env (conda)
#   make train          # run Phase 6 sweep (train + eval)
#   make eval           # pretty-print summary from metrics JSON
#   make test           # run pytest test suite
#   make lint           # run ruff + black
#   make clean          # remove outputs
# ===================================================================

# You can override these on the CLI:
#   make train PYTHON=python3
ENV_NAME ?= CS542-Project-GPU
PYTHON   ?= python

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

# -------------------------------------------------------------------
# Training / evaluation
# -------------------------------------------------------------------
.PHONY: train
train:
	@echo ">>> Running Phase 6 sweep (train + eval) using configs/config.yaml"
	$(PYTHON) -m src.realestate.train_sweep

# (Optional) separate eval step; for now this just re-prints the summary
# from the latest metrics file.
.PHONY: eval
eval:
	@echo ">>> Evaluating latest Phase 6 results"
	$(PYTHON) -m src.realestate.train_sweep | tail -n 40

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
