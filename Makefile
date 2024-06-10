.PHONY: install lint format type-check test

install:
	poetry install

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

type-check:
	poetry run mypy .

test:
	poetry run pytest
