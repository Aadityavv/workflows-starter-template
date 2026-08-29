.PHONY: setup test lint fixtures demo clean

setup:
	pip install -e ".[dev,llm]"

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

fixtures:
	python tests/fixtures/generate.py

demo:
	djmix analyze tests/fixtures/audio
	djmix mix tests/fixtures/audio --occasion workout --out out/workout.wav
	djmix validate out/workout.plan.json

clean:
	rm -rf out .djmix-cache .pytest_cache .ruff_cache
