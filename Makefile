.PHONY: help install dev test smoke-test schema-test lint format etapa1 train-mlp run-api run-notebook mlflow-ui clean

PY ?= python

help:
	@echo "Targets disponiveis:"
	@echo "  make install       - Instala dependencias (pip install -e .)"
	@echo "  make dev           - Instala dependencias de desenvolvimento"
	@echo "  make test          - Roda toda a suite com cobertura"
	@echo "  make smoke-test    - Roda apenas testes marcados @pytest.mark.smoke"
	@echo "  make schema-test   - Roda apenas testes marcados @pytest.mark.schema"
	@echo "  make lint          - Roda ruff sobre src/ e tests/"
	@echo "  make format        - Formata codigo com black + isort"
	@echo "  make etapa1        - Executa Etapa 1 (EDA + baselines)"
	@echo "  make train-mlp     - Executa Etapa 2 (treina MLP + salva preprocessor)"
	@echo "  make run-api       - Sobe a API FastAPI em http://localhost:8000"
	@echo "  make run-notebook  - Inicia o Jupyter notebook"
	@echo "  make mlflow-ui     - Inicia o MLflow UI em http://localhost:5000"
	@echo "  make clean         - Remove caches (.pytest_cache, __pycache__, build artifacts)"

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest tests/ -v --cov=src --cov-report=term-missing

smoke-test:
	$(PY) -m pytest tests/ -v -m "smoke"

schema-test:
	$(PY) -m pytest tests/ -v -m "schema"

lint:
	$(PY) -m ruff check src/ tests/

format:
	$(PY) -m black src/ tests/ notebooks/
	$(PY) -m isort src/ tests/ notebooks/

etapa1:
	$(PY) -m src.etapa1_eda_baselines

train-mlp:
	$(PY) -m src.train_mlp

run-api:
	$(PY) -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

run-notebook:
	$(PY) -m jupyter notebook notebooks/

mlflow-ui:
	$(PY) -m mlflow ui --host 0.0.0.0 --port 5000

clean:
	$(PY) -c "import shutil, pathlib, sys; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.ruff_cache','.mypy_cache','htmlcov','build','dist']]; print('Limpeza concluida')"
