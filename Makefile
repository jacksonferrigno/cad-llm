.PHONY: sync sync-minimal test lint fmt check verify dirs info

sync:
	uv sync

sync-minimal:
	uv sync --no-default-groups --no-editable

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff check --fix .
	uv run ruff format .

check: lint test

verify:
	bash scripts/verify_env.sh

dirs:
	uv run cad-llm ensure-dirs

info:
	uv run cad-llm info
