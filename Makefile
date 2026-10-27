# One setup command, one run command:
#   make setup   -> venv + dependencies + corpus + index
#   make run     -> start the service on http://localhost:8000
PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup venv corpus index run test eval eval-retrieval eval-generation sweep load clean

setup: venv corpus index

venv:
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -q -r requirements.txt

corpus:
	bash scripts/fetch_corpus.sh

index:
	$(PY) -m librarian.build_index

run:
	$(PY) -m librarian.server

test:
	$(PY) -m pytest -q tests

eval: eval-retrieval eval-generation

eval-retrieval:
	$(PY) eval/validate_qa.py
	$(PY) eval/retrieval_eval.py

eval-generation:
	$(PY) eval/generation_eval.py --judge --tag baseline

sweep:
	$(PY) eval/chunk_sweep.py

load:
	$(PY) eval/load_test.py --url http://localhost:8000 --tag baseline

clean:
	rm -rf index index_sweep results/*.json
