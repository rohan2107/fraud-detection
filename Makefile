PYTHON ?= python

.PHONY: install install-dev lint format test docker-build docker-run

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install-dev: install
	$(PYTHON) -m pip install -r requirements-dev.txt

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


