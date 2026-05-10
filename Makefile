.PHONY: help install demo eval-all probe report bench-all clean fmt test

PY ?= 3.12

help:
	@echo "make install     create venv + install all extras"
	@echo "make demo        launch the live Streamlit app at localhost:8501"
	@echo "make probe       run 10 sniff-tests per model on real data (~7 min)"
	@echo "make eval-all    re-run all 4 model evals (~25 min on M3)"
	@echo "make report      regenerate docs/RESULTS.md from eval/reports/"
	@echo "make bench-all   re-run all 4 benchmark passes"
	@echo "make test        run pytest + ruff"
	@echo "make fmt         apply ruff fixes"

install:
	uv venv --python $(PY)
	uv pip install -e ".[dev,terramind,prithvi,ttm,live]"
	uv pip install streamlit altair

demo:
	uv run streamlit run app/streamlit_app.py

probe:
	uv run python scripts/probe.py

eval-all:
	uv run riprap-models eval ttm-battery-surge
	uv run riprap-models eval prithvi-pluvial
	uv run riprap-models eval terramind-buildings
	uv run riprap-models eval terramind-lulc
	uv run riprap-models report

bench-all:
	uv run riprap-models bench ttm-battery-surge --n 30
	uv run riprap-models bench prithvi-pluvial --n 5
	uv run riprap-models bench terramind-buildings --n 5
	uv run riprap-models bench terramind-lulc --n 5

report:
	uv run riprap-models report

test:
	uv run pytest tests -q
	uv run ruff check src tests

fmt:
	uv run ruff check src tests --fix

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
	find . -name __pycache__ -prune -exec rm -rf {} +
