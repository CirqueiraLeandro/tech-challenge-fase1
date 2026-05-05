"""Testes Unitários - Modelos e Baselines (Expandido)"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.baselines import BaselineTrainer


class TestBaselineTrainerSmoke:
    """Smoke tests - Treinamento básico de modelos"""

    @pytest.fixture
    def sample_dataset(self):
        """Criar dataset de exemplo"""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        return X_df, y

    @pytest.mark.smoke
    def test_trainer_init(self):
        """SMOKE: Trainer inicializa?"""
        trainer = BaselineTrainer()
        assert trainer is not None
        assert len(trainer.models) == 0
        assert len(trainer.results) == 0

    @pytest.mark.smoke
    def test_dummy_classifier_treina(self, sample_dataset):
        """SMOKE: DummyClassifier treina?"""
        X, y = sample_dataset
        X_train = X.iloc[:150]
        y_train = y[:150]
        
        trainer = BaselineTrainer()
        model = trainer.train_dummy_classifier(X_train, y_train)
        
        assert model is not None
        assert "dummy" in trainer.models

    @pytest.mark.smoke
    def test_logistic_regression_treina(self, sample_dataset):
        """SMOKE: LogisticRegression treina?"""
        X, y = sample_dataset
        X_train = X.iloc[:150]
        y_train = y[:150]
        
        trainer = BaselineTrainer()
        model = trainer.train_logistic_regression(X_train, y_train)
        
        assert model is not None
        assert "logistic_regression" in trainer.models

    @pytest.mark.smoke
    def test_random_forest_treina(self, sample_dataset):
        """SMOKE: RandomForest treina?"""
        X, y = sample_dataset
        X_train = X.iloc[:150]
        y_train = y[:150]
        
        trainer = BaselineTrainer()
        model = trainer.train_random_forest(X_train, y_train, n_estimators=10)
        
        assert model is not None
        assert "random_forest" in trainer.models


class TestBaselineTrainerSchema:
    """Schema tests - Validação de outputs de modelos"""

    @pytest.fixture
    def trained_models(self):
        """Fixture com modelos treinados"""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        X_train = X_df.iloc[:150]
        y_train = y[:150]
        
        trainer = BaselineTrainer()
        trainer.train_dummy_classifier(X_train, y_train)
        trainer.train_logistic_regression(X_train, y_train)
        trainer.train_random_forest(X_train, y_train, n_estimators=10)
        
        return trainer, X_df, y

    @pytest.mark.schema
    def test_predictions_validos(self, trained_models):
        """SCHEMA: Predições válidas (0/1)?"""
        trainer, X_df, y = trained_models
        X_test = X_df.iloc[150:]
        
        for model_name, model in trainer.models.items():
            y_pred = model.predict(X_test)
            assert set(y_pred).issubset({0, 1}), \
                f"{model_name}: Valores inválidos em predições"

    @pytest.mark.schema
    def test_probabilities_validas(self, trained_models):
        """SCHEMA: Probabilidades em [0,1]?"""
        trainer, X_df, y = trained_models
        X_test = X_df.iloc[150:]
        
        for model_name, model in trainer.models.items():
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                assert (y_proba >= 0).all() and (y_proba <= 1).all(), \
                    f"{model_name}: Probabilidades fora de [0,1]"
                assert y_proba.shape == (len(X_test), 2), \
                    f"{model_name}: Shape incorreto"

    @pytest.mark.schema
    def test_metrics_validas(self, trained_models):
        """SCHEMA: Métricas dentro de intervalo válido?"""
        trainer, X_df, y = trained_models
        X_test = X_df.iloc[150:]
        y_test = y[150:]
        
        for model_name, model in trainer.models.items():
            metrics = trainer.evaluate_model(model_name, model, X_test, y_test)
            
            # Verificar que todas as métricas existem
            required_metrics = ["accuracy", "precision", "recall", "f1", 
                              "auc_roc", "pr_auc", "confusion_matrix", "business_cost"]
            for metric in required_metrics:
                assert metric in metrics, f"{metric} faltando em {model_name}"
            
            # Verificar intervalos
            for metric_name in ["accuracy", "precision", "recall", "f1", "auc_roc", "pr_auc"]:
                val = metrics[metric_name]
                assert 0 <= val <= 1, f"{metric_name}={val} fora de [0,1]"


class TestBaselineTrainerIntegration:
    """Integration tests - Fluxos completos de treinamento"""

    @pytest.fixture
    def complete_dataset(self):
        """Dataset completo para treinamento"""
        X, y = make_classification(
            n_samples=300,
            n_features=15,
            n_informative=8,
            n_redundant=3,
            n_classes=2,
            weights=[0.7, 0.3],  # Desbalanceado (como churn real)
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(15)])
        return X_df, y

    def test_pipeline_treino_avaliacao(self, complete_dataset):
        """INTEG: Pipeline completo treino → avaliação?"""
        X, y = complete_dataset
        X_train = X.iloc[:200]
        y_train = y[:200]
        X_test = X.iloc[200:]
        y_test = y[200:]
        
        trainer = BaselineTrainer(random_state=42)
        
        # Treinar todos os modelos
        trainer.train_dummy_classifier(X_train, y_train)
        trainer.train_logistic_regression(X_train, y_train)
        trainer.train_random_forest(X_train, y_train, n_estimators=50)
        
        # Avaliar todos
        results = trainer.compare_baselines(X_test, y_test)
        
        assert len(results) == 3, "Deveria ter 3 modelos"
        assert len(results.columns) > 0, "Sem colunas de resultado"

    def test_reproducibilidade_seed(self, complete_dataset):
        """INTEG: Seed reproduz resultados?"""
        X, y = complete_dataset
        X_train = X.iloc[:200]
        y_train = y[:200]
        X_test = X.iloc[200:]
        y_test = y[200:]
        
        # Primeiro treino
        trainer1 = BaselineTrainer(random_state=42)
        trainer1.train_logistic_regression(X_train, y_train)
        metrics1 = trainer1.evaluate_model("lr1", trainer1.models["logistic_regression"], 
                                          X_test, y_test)
        
        # Segundo treino com mesma seed
        trainer2 = BaselineTrainer(random_state=42)
        trainer2.train_logistic_regression(X_train, y_train)
        metrics2 = trainer2.evaluate_model("lr2", trainer2.models["logistic_regression"], 
                                          X_test, y_test)
        
        # AUC deve ser idêntico
        assert metrics1["auc_roc"] == metrics2["auc_roc"]

    def test_comparacao_modelos(self, complete_dataset):
        """INTEG: Comparação identifica melhor modelo?"""
        X, y = complete_dataset
        X_train = X.iloc[:200]
        y_train = y[:200]
        X_test = X.iloc[200:]
        y_test = y[200:]
        
        trainer = BaselineTrainer()
        trainer.train_dummy_classifier(X_train, y_train)
        trainer.train_logistic_regression(X_train, y_train)
        trainer.train_random_forest(X_train, y_train, n_estimators=50)
        
        comparison = trainer.compare_baselines(X_test, y_test)
        
        # RandomForest deve ter melhor AUC que DummyClassifier
        rf_auc = comparison[comparison["model_name"] == "random_forest"]["auc_roc"].values[0]
        dummy_auc = comparison[comparison["model_name"] == "dummy"]["auc_roc"].values[0]
        
        assert rf_auc > dummy_auc, "RandomForest deveria ter melhor AUC"

    def test_desbalanceamento_detectado(self, complete_dataset):
        """INTEG: Metrics refletem desbalanceamento?"""
        X, y = complete_dataset
        X_train = X.iloc[:200]
        y_train = y[:200]
        X_test = X.iloc[200:]
        y_test = y[200:]
        
        trainer = BaselineTrainer()
        trainer.train_logistic_regression(X_train, y_train)
        metrics = trainer.evaluate_model("lr", trainer.models["logistic_regression"], 
                                        X_test, y_test)
        
        # Com desbalanceamento, PR-AUC pode diferir de AUC-ROC
        auc_roc = metrics["auc_roc"]
        pr_auc = metrics["pr_auc"]
        
        # Ambas devem existir e ser > 0
        assert auc_roc > 0
        assert pr_auc > 0

    def test_custo_negocio_calculado(self, complete_dataset):
        """INTEG: Custo de negócio calculado corretamente?"""
        X, y = complete_dataset
        X_train = X.iloc[:200]
        y_train = y[:200]
        X_test = X.iloc[200:]
        y_test = y[200:]
        
        trainer = BaselineTrainer()
        trainer.train_logistic_regression(X_train, y_train)
        metrics = trainer.evaluate_model("lr", trainer.models["logistic_regression"], 
                                        X_test, y_test)
        
        # Custo deve ser não-negativo
        cost = metrics["business_cost"]
        assert cost >= 0, "Custo de negócio não pode ser negativo"


class TestBaselineTrainerEdgeCases:
    """Edge case tests - Cenários extremos"""

    def test_treino_pequeno_dataset(self):
        """EDGE: Treina com dataset muito pequeno?"""
        X = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0]
        })
        y = np.array([0, 1, 0])
        
        trainer = BaselineTrainer()
        model = trainer.train_logistic_regression(X, y)
        assert model is not None

    def test_classes_desequilibradas(self):
        """EDGE: Treina com classes muito desequilibradas?"""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            weights=[0.95, 0.05],  # 95-5 split
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        
        trainer = BaselineTrainer()
        trainer.train_logistic_regression(X_df, y)
        metrics = trainer.evaluate_model("lr", trainer.models["logistic_regression"], 
                                        X_df, y)
        
        assert "auc_roc" in metrics
        assert metrics["auc_roc"] >= 0

    def test_features_correlacionadas(self):
        """EDGE: Treina com features altamente correlacionadas?"""
        X = pd.DataFrame({
            "f1": np.arange(100),
            "f2": np.arange(100) * 2,  # Perfectamente correlacionado com f1
            "f3": np.random.randn(100)
        })
        y = (X["f1"] > 50).astype(int).values
        
        trainer = BaselineTrainer()
        trainer.train_random_forest(X, y, n_estimators=10)
        assert "random_forest" in trainer.models
