#!/usr/bin/env bash
set -e

echo "Running Ruff lint checks..."
ruff check .

echo "Checking Ruff formatting..."
ruff format --check .

echo "Ruff checks passed!"
