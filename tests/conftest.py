"""Configuração Global de Testes - pytest"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent


def pytest_configure(config):
    """Configuração inicial do pytest"""
    # Registrar markers customizados
    config.addinivalue_line(
        "markers", "smoke: marca testes críticos (smoke tests)"
    )
    config.addinivalue_line(
        "markers", "schema: marca testes de validação de schema"
    )
    config.addinivalue_line(
        "markers", "integration: marca testes de integração"
    )
    config.addinivalue_line(
        "markers", "e2e: marca testes end-to-end"
    )
    config.addinivalue_line(
        "markers", "slow: marca testes lentos"
    )


@pytest.fixture(scope="session")
def project_root():
    """Retorna raiz do projeto"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_raw_dir():
    """Retorna diretório data/raw"""
    return PROJECT_ROOT / "data" / "raw"


@pytest.fixture(scope="session")
def models_dir():
    """Retorna diretório models"""
    return PROJECT_ROOT / "models"


@pytest.fixture(scope="session")
def logs_dir():
    """Retorna diretório logs"""
    return PROJECT_ROOT / "logs"


@pytest.fixture
def sample_dataframe():
    """Fixture: DataFrame de exemplo"""
    return pd.DataFrame({
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randn(100),
        "feature_3": np.random.randint(0, 5, 100),
        "target": np.random.randint(0, 2, 100)
    })


@pytest.fixture
def sample_numeric_dataframe():
    """Fixture: DataFrame com features apenas numéricas"""
    return pd.DataFrame({
        "f1": np.arange(50, dtype=float),
        "f2": np.random.randn(50),
        "f3": np.random.randn(50)
    })


@pytest.fixture
def sample_categorical_dataframe():
    """Fixture: DataFrame com features apenas categóricas"""
    return pd.DataFrame({
        "cat1": np.random.choice(["A", "B", "C"], 50),
        "cat2": np.random.choice(["X", "Y"], 50),
        "cat3": np.random.choice(["P", "Q", "R", "S"], 50)
    })


@pytest.fixture
def sample_target():
    """Fixture: Target vector (binary)"""
    return np.random.randint(0, 2, 100)


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seeds antes de cada teste"""
    np.random.seed(42)
    yield
    # Garantir que runs MLflow não vazam entre testes
    try:
        import mlflow
        while mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception:
        pass


@pytest.fixture
def caplog_handler(caplog):
    """Capturar logs durante testes"""
    caplog.set_level(logging.DEBUG)
    return caplog


class TimeoutError(Exception):
    """Exceção para timeout de testes"""
    pass


def pytest_collection_modifyitems(config, items):
    """Modificar items coletados (adicionar markers etc)"""
    for item in items:
        # Adicionar marker 'unit' se o test estiver em test_*_unit.py
        if "unit" not in item.nodeid:
            # Adicionar marker default
            pass


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup do ambiente de testes"""
    # Criar diretórios necessários
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "models").mkdir(exist_ok=True)
    (PROJECT_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup após testes
    pass
