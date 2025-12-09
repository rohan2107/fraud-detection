PYTHON ?= python

.PHONY: install install-dev lock-deps lint format test docker-build docker-run

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install pip-tools
	$(PYTHON) -m piptools sync requirements.txt requirements-dev.txt

lock-deps:
	pip-compile -o requirements.txt requirements.in
	pip-compile -o requirements-dev.txt requirements-dev.in

lint:
	ruff check .

format:
	black src tests

test:
	pytest -q

docker-build:
	docker build -t fraud-api .

docker-run:
	docker run --rm -p 8000:8000 --env-file env.example fraud-api


