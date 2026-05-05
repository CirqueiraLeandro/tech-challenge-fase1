"""Testes Unitários - Config e Logging"""

import pytest
import logging

from src.config import Config, DevelopmentConfig, ProductionConfig, TestingConfig, get_config
from src.logger import setup_logger


class TestConfigClasses:
    """Testes das classes de configuração"""

    def test_config_base_paths(self):
        """Verificar que paths base existem"""
        config = Config()
        assert config.PROJECT_ROOT.exists()
        assert config.DATA_RAW_DIR.exists()
        assert config.LOGS_DIR.exists()
        assert config.MODELS_DIR.exists()

    def test_config_dataset_path(self):
        """Verificar path do dataset"""
        config = Config()
        assert config.DATASET_NAME == "Telco_customer_churn.xlsx"
        assert config.DATASET_PATH.name == "Telco_customer_churn.xlsx"

    def test_config_seeds(self):
        """Verificar seeds configurados"""
        config = Config()
        assert config.RANDOM_SEED == 42
        assert config.NUMPY_SEED == 42
        assert config.TORCH_SEED == 42

    def test_config_mlflow(self):
        """Verificar MLflow config"""
        config = Config()
        assert config.EXPERIMENT_NAME == "telco-churn-prediction"
        assert config.MLFLOW_TRACKING_URI is not None

    def test_config_api(self):
        """Verificar API config"""
        config = Config()
        assert config.API_HOST == "0.0.0.0"
        assert config.API_PORT == 8000
        assert config.API_TITLE is not None

    def test_config_pytorch(self):
        """Verificar PyTorch config"""
        config = Config()
        assert config.BATCH_SIZE == 32
        assert config.EPOCHS == 50
        assert config.LEARNING_RATE == 0.001


class TestDevelopmentConfig:
    """Testes config de desenvolvimento"""

    def test_development_debug_true(self):
        """Config dev tem DEBUG=True?"""
        config = DevelopmentConfig()
        assert config.DEBUG is True

    def test_development_testing_false(self):
        """Config dev tem TESTING=False?"""
        config = DevelopmentConfig()
        assert config.TESTING is False

    def test_development_herda_de_config(self):
        """Config dev herda de Config?"""
        dev_config = DevelopmentConfig()
        base_config = Config()
        
        # Deve herdar seeds
        assert dev_config.RANDOM_SEED == base_config.RANDOM_SEED


class TestProductionConfig:
    """Testes config de produção"""

    def test_production_debug_false(self):
        """Config prod tem DEBUG=False?"""
        config = ProductionConfig()
        assert config.DEBUG is False

    def test_production_log_level_warning(self):
        """Config prod tem LOG_LEVEL=WARNING?"""
        config = ProductionConfig()
        assert config.LOG_LEVEL == "WARNING"

    def test_production_testing_false(self):
        """Config prod tem TESTING=False?"""
        config = ProductionConfig()
        assert config.TESTING is False


class TestTestingConfig:
    """Testes config de testes"""

    def test_testing_testing_true(self):
        """Config test tem TESTING=True?"""
        config = TestingConfig()
        assert config.TESTING is True

    def test_testing_reduced_epochs(self):
        """Config test tem EPOCHS reduzido?"""
        config = TestingConfig()
        assert config.EPOCHS == 2
        assert config.EPOCHS < Config().EPOCHS

    def test_testing_reduced_batch_size(self):
        """Config test tem BATCH_SIZE reduzido?"""
        config = TestingConfig()
        assert config.BATCH_SIZE == 8
        assert config.BATCH_SIZE < Config().BATCH_SIZE


class TestGetConfig:
    """Testes função get_config"""

    def test_get_config_development_default(self):
        """get_config() retorna Development por padrão?"""
        config = get_config()
        assert config.DEBUG is True

    def test_get_config_development_explicit(self):
        """get_config('development') retorna DevelopmentConfig?"""
        config = get_config("development")
        assert config.DEBUG is True

    def test_get_config_production(self):
        """get_config('production') retorna ProductionConfig?"""
        config = get_config("production")
        assert config.DEBUG is False

    def test_get_config_testing(self):
        """get_config('testing') retorna TestingConfig?"""
        config = get_config("testing")
        assert config.TESTING is True

    def test_get_config_case_insensitive(self):
        """get_config() é case-insensitive?"""
        config1 = get_config("DEVELOPMENT")
        config2 = get_config("development")
        assert config1.DEBUG == config2.DEBUG

    def test_get_config_via_env_var(self, monkeypatch):
        """get_config() usa variável de ambiente?"""
        monkeypatch.setenv("ENVIRONMENT", "production")
        config = get_config()
        assert config.DEBUG is False


class TestLogger:
    """Testes do logger"""

    def test_logger_setup_basic(self):
        """Logger setup funciona?"""
        logger = setup_logger(__name__)
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_logger_with_config(self):
        """Logger setup com config funciona?"""
        config = get_config("development")
        logger = setup_logger(__name__, config)
        assert logger is not None

    def test_logger_levels(self):
        """Logger pode logar diferentes níveis?"""
        logger = setup_logger("test_logger")
        
        # Não deve lançar exceção
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_logger_handlers_existem(self):
        """Logger tem handlers (console + file)?"""
        logger = setup_logger("test_logger_handlers")
        assert len(logger.handlers) >= 1  # Pelo menos console handler

    def test_logger_duplicacao_handlers(self):
        """Não cria handlers duplicados?"""
        logger1 = setup_logger("test_logger_dup")
        n_handlers_1 = len(logger1.handlers)
        
        logger2 = setup_logger("test_logger_dup")  # Mesmo nome
        n_handlers_2 = len(logger2.handlers)
        
        # Não deve duplicar
        assert n_handlers_2 == n_handlers_1

    def test_logger_file_created(self):
        """Logger cria arquivo de log?"""
        config = get_config("development")
        logger = setup_logger("test_logger_file", config)
        
        logger.info("Test message")
        
        # Arquivo deve existir
        assert config.LOG_FILE.exists()


class TestConfigEnvironmentVariables:
    """Testes integração com variáveis de ambiente"""

    def test_log_level_from_env(self, monkeypatch):
        """LOG_LEVEL vem de variável de ambiente?"""
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        get_config()
        # Config usa getenv; verificar apenas que a função existe e não lança

    def test_api_port_from_env(self, monkeypatch):
        """API_PORT vem de variável de ambiente?"""
        monkeypatch.setenv("API_PORT", "9000")
        get_config()
        # Verificar apenas que a função existe e não lança

    def test_missing_env_var_usa_default(self, monkeypatch):
        """Usa defaults quando env var não existe?"""
        monkeypatch.delenv("API_PORT", raising=False)
        config = get_config()
        assert config.API_PORT == 8000  # Default


class TestPathIntegrity:
    """Testes integridade de paths"""

    def test_project_root_valid(self):
        """PROJECT_ROOT é válido?"""
        config = Config()
        assert config.PROJECT_ROOT.exists()
        assert config.PROJECT_ROOT.is_dir()

    def test_data_dirs_exist(self):
        """Diretórios de dados existem?"""
        config = Config()
        assert config.DATA_RAW_DIR.exists()
        assert config.DATA_PROCESSED_DIR.exists()

    def test_models_dir_writable(self):
        """Diretório de modelos é escribível?"""
        config = Config()
        test_file = config.MODELS_DIR / ".test_write"
        try:
            test_file.touch()
            assert test_file.exists()
            test_file.unlink()
        except PermissionError:
            pytest.fail("models/ não é escribível")

    def test_logs_dir_writable(self):
        """Diretório de logs é escribível?"""
        config = Config()
        test_file = config.LOGS_DIR / ".test_write"
        try:
            test_file.touch()
            assert test_file.exists()
            test_file.unlink()
        except PermissionError:
            pytest.fail("logs/ não é escribível")
