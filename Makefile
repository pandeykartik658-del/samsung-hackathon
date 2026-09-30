# Makefile
# Uses .venv/bin/python when `make venv` has been run, otherwise python3 on PATH.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYTHON_BIN ?= python3
IMAGE ?= theme5
WHISPER_MODEL ?= base.en

.PHONY: help venv install install-core test scenarios bench demo serve models viewer \
        docker-build docker-test docker-demo clean

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-13s %s\n", $$1, $$2}'

venv:  ## create .venv (PYTHON_BIN=python3.10|3.11|3.12)
	$(PYTHON_BIN) -m venv .venv
	.venv/bin/python -m pip install -q --upgrade pip

install:  ## full install: perception extras + test tools
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

install-core:  ## stdlib core + test tools only (no ASR/OCR)
	$(PY) -m pip install -r requirements-dev.txt

test:  ## unit, integration and viewer tests
	$(PY) -m pytest -q tests viewer

scenarios:  ## 9 public-style scenarios through the sim harness
	$(PY) -m sim.run scenarios --report

bench:  ## 60 adversarial scenarios, rubric table + 5 worst failures
	$(PY) -m bench.run_suite bench/scenarios60 --agent theme5.agent:Agent --out bench/runs/latest

demo:  ## scenarios + bench + trace viewer with a real trace
	PYTHON=$(PY) bash run_demo.sh

serve:  ## JSONL events on stdin, JSONL actions on stdout
	$(PY) -m theme5 serve

models:  ## download Whisper weights into ./models (needs Hugging Face access)
	THEME5_MODEL_DIR=models $(PY) scripts/fetch_models.py --model $(WHISPER_MODEL)

docker-build:  ## build the offline image (bakes Whisper weights)
	docker build -t $(IMAGE) --build-arg WHISPER_MODEL=$(WHISPER_MODEL) .

docker-test:  ## run the test suite inside the image
	docker run --rm $(IMAGE) python -m pytest -q -p no:cacheprovider tests viewer

docker-demo:  ## run the demo inside the image
	docker run --rm $(IMAGE)

clean:  ## remove caches and generated runs
	rm -rf runs bench/runs .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
