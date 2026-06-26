# Convenience targets. Everything also works with plain `python` commands —
# see the README. No dependencies are required.

.PHONY: help demo test batch run clean

help:
	@echo "make demo   - run the guided tour (examples/demo.py)"
	@echo "make test   - run the unittest suite"
	@echo "make batch  - run the built-in batch + metrics"
	@echo "make run Q='calculate 2 + 2'  - run one query"
	@echo "make clean  - remove __pycache__ etc."

demo:
	python examples/demo.py

test:
	python -m unittest discover -s tests

batch:
	python main.py --batch

run:
	python main.py "$(Q)" --trajectory

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info build dist
