.PHONY: tests lint format evals


evals:
	LANGCHAIN_TEST_CACHE=tests/evals/cassettes python -m python -m pytest -p no:asyncio  --max-asyncio-tasks 4 tests/evals

lint:
	python -m ruff check .
	python -m mypy .

format:
	ruff check --select I --fix
	python -m ruff format .
	python -m ruff check . --fix

build:
	poetry build

publish:
	poetry publish --dry-run
