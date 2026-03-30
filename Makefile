PYTHON ?= python3

.PHONY: install install-dev test lint format typecheck precommit run

install:
	uv sync

install-dev:
	uv sync --dev

test:
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy .

precommit:
	uv run pre-commit run --all-files

run:
	uv run python main.py
