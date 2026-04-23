.PHONY: install dev test test-cov lint format docker-up docker-down docker-build worker api hermes

PYTHON := python3
POETRY := poetry
SRC_DIR := src/harness
TESTS_DIR := tests

# ---------------------------------------------------------------------------
# Setup & Installation
# ---------------------------------------------------------------------------

install:
	$(POETRY) install --with dev

dev:
	$(POETRY) install --with dev
	cp -n .env.example .env || true
	@echo "Dev environment ready. Edit .env before running services."

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:
	$(POETRY) run pytest $(TESTS_DIR) -q --tb=short

test-cov:
	$(POETRY) run pytest $(TESTS_DIR) \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml \
		-q

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------

lint:
	$(POETRY) run ruff check $(SRC_DIR) $(TESTS_DIR)
	$(POETRY) run mypy $(SRC_DIR)

format:
	$(POETRY) run ruff check --fix $(SRC_DIR) $(TESTS_DIR)
	$(POETRY) run ruff format $(SRC_DIR) $(TESTS_DIR)

# ---------------------------------------------------------------------------
# Docker Compose (infrastructure only)
# ---------------------------------------------------------------------------

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-build:
	docker compose build

# ---------------------------------------------------------------------------
# Application Processes
# ---------------------------------------------------------------------------

api:
	$(POETRY) run uvicorn harness.api.main:create_app \
		--factory \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--log-level info

worker:
	$(POETRY) run python -m harness.workers.agent_worker

hermes:
	$(POETRY) run python -m harness.workers.hermes_worker
