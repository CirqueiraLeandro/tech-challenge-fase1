"""Configuração Centralizada do Projeto"""

from pathlib import Path
import os


class Config:
    """Configuração Base"""

    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    DOCS_DIR = PROJECT_ROOT / "docs"
    
    # Dataset Configuration
    DATASET_PATH = DATA_RAW_DIR / "Telco_customer_churn.xlsx"
    DATASET_NAME = "Telco_customer_churn.xlsx"
    
    # Seeds for Reproducibility
    RANDOM_SEED = 42
    NUMPY_SEED = 42
    TORCH_SEED = 42
    SEEDS = {
        "random": 42,
        "numpy": 42,
        "torch": 42,
    }
    
    # Data Configuration
    TEST_SIZE = 0.2
    STRATIFY_TARGET = True
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI = "file:./mlruns"
    EXPERIMENT_NAME = "telco-churn-prediction"
    
    # PyTorch Configuration
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_TITLE = "Telco Churn Prediction API"
    API_VERSION = "1.0.0"
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FILE = LOGS_DIR / "application.log"
    
    # Testing Configuration
    TESTING = False


class DevelopmentConfig(Config):
    """Configuração para Desenvolvimento"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Configuração para Produção"""
    DEBUG = False
    LOG_LEVEL = "WARNING"


class TestingConfig(Config):
    """Configuração para Testes"""
    DEBUG = True
    TESTING = True
    TEST_SIZE = 0.3
    BATCH_SIZE = 8
    EPOCHS = 2


def get_config(environment: str = None) -> Config:
    """Retorna configuração baseada em ambiente
    
    Args:
        environment: Environment name (development, production, testing).
                    If None, uses ENVIRONMENT env var or defaults to development.
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development").lower()
    
    env_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig,
    }
    return env_map.get(environment, DevelopmentConfig)()
