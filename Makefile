format:
	poetry run ruff format pathfinding tests
lint:
	poetry run ruff check --fix pathfinding tests
test:
	poetry run coverage run -m pytest -v -s ./tests && poetry run coverage report -m