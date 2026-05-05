"""Testes Unitários - MLflow Utils"""

import pytest
import pandas as pd

from src.mlflow_utils import MLflowTracker
from src.config import get_config


class TestMLflowTracker:
    """Testes do tracker MLflow"""

    @pytest.fixture
    def config(self):
        return get_config("development")

    @pytest.fixture
    def tracker(self, config):
        return MLflowTracker(config)

    def test_tracker_init(self, tracker):
        """Tracker inicializa?"""
        assert tracker is not None
        assert tracker.config is not None

    def test_tracker_start_run(self, tracker):
        """Pode iniciar um run?"""
        tracker.start_run("test_run")
        # MLflow internally tracks active runs
        tracker.end_run()

    def test_tracker_log_metrics(self, tracker):
        """Pode logar métricas?"""
        tracker.start_run("metrics_run")
        metrics = {
            "accuracy": 0.85,
            "f1": 0.82,
            "auc_roc": 0.88
        }
        tracker.log_metrics(metrics)
        tracker.end_run()

    def test_tracker_log_single_metric(self, tracker):
        """Pode logar métrica individual?"""
        tracker.start_run("single_metric_run")
        tracker.log_metric("loss", 0.15, step=1)
        tracker.log_metric("loss", 0.12, step=2)
        tracker.end_run()

    def test_tracker_log_dataframe(self, tracker):
        """Pode logar DataFrame como artefato?"""
        tracker.start_run("df_run")
        
        df = pd.DataFrame({
            "model": ["dummy", "logistic", "random_forest"],
            "auc_roc": [0.50, 0.85, 0.90]
        })
        
        tracker.log_dataframe_as_artifact(df, "results.csv")
        tracker.end_run()

    def test_tracker_experiment_exists(self, config):
        """Experimento existe?"""
        exp_name = config.EXPERIMENT_NAME
        MLflowTracker.get_experiment_id(exp_name)
        # None é válido se o experimento ainda não foi criado

    def test_tracker_get_runs(self, config):
        """Pode buscar runs?"""
        exp_name = config.EXPERIMENT_NAME
        runs_df = MLflowTracker.get_experiment_runs(exp_name)
        # Pode estar vazio, mas deve retornar DataFrame
        assert isinstance(runs_df, pd.DataFrame)


class TestMLflowTrackerParams:
    """Testes logging de parâmetros"""

    @pytest.fixture
    def config(self):
        return get_config("development")

    @pytest.fixture
    def tracker(self, config):
        return MLflowTracker(config)

    def test_start_run_com_params(self, tracker):
        """Pode loggar parâmetros na inicialização?"""
        params = {
            "model_type": "logistic_regression",
            "max_iter": 1000,
            "random_state": 42
        }
        tracker.start_run("params_run", params)
        tracker.end_run()

    def test_params_tipos_suportados(self, tracker):
        """Parâmetros de diferentes tipos funcionam?"""
        params = {
            "str_param": "value",
            "int_param": 42,
            "float_param": 0.95
        }
        tracker.start_run("mixed_params_run", params)
        tracker.end_run()


class TestMLflowTrackerComparison:
    """Testes comparação de runs"""

    @pytest.fixture
    def config(self):
        return get_config("development")

    def test_compare_runs_returns_dataframe(self, config):
        """Comparação retorna DataFrame?"""
        exp_name = config.EXPERIMENT_NAME
        comparison = MLflowTracker.compare_runs(exp_name)
        assert isinstance(comparison, pd.DataFrame)


class TestMLflowTrackerIntegration:
    """Testes integração completa MLflow"""

    @pytest.fixture
    def config(self):
        return get_config("development")

    def test_full_experiment_workflow(self, config):
        """Workflow completo: start → log → end?"""
        tracker = MLflowTracker(config)
        
        # Start run
        tracker.start_run("integration_run", {"model": "test_model"})
        
        # Log metrics
        tracker.log_metrics({
            "accuracy": 0.80,
            "precision": 0.82,
            "recall": 0.78
        })
        
        # Log incremental metric
        for epoch in range(3):
            tracker.log_metric("train_loss", 1.0 - (epoch * 0.1), step=epoch)
        
        # Log dataframe
        df = pd.DataFrame({"epoch": [0, 1, 2], "loss": [1.0, 0.9, 0.8]})
        tracker.log_dataframe_as_artifact(df, "training_log.csv")
        
        # End run
        tracker.end_run()
