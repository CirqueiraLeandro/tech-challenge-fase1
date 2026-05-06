"""
Testes para MLP (Etapa 2).

Testes cobrem:
- Arquitetura MLP
- Predictions e predict_proba
- Training loop
- Early stopping
- Validação cruzada
"""

import torch
import numpy as np
import pytest

from src.models.mlp import MLPChurn, create_mlp_model
from src.models.training import MLPTrainer, EarlyStopping, train_with_cross_validation


class TestMLPArchitecture:
    """Testes de arquitetura MLP."""
    
    def test_mlp_init_default(self):
        """MLP com parâmetros padrão (19 features pós leakage fix)."""
        model = MLPChurn()
        assert model.input_dim == 19
        assert model.output_dim == 2
        assert model.hidden_dims == [128, 64, 32]
        assert model.dropout_rate == 0.3
    
    def test_mlp_init_custom(self):
        """MLP com parâmetros customizados."""
        model = MLPChurn(
            input_dim=20,
            hidden_dims=[64, 32],
            output_dim=3,
            dropout_rate=0.5
        )
        assert model.input_dim == 20
        assert model.output_dim == 3
        assert model.hidden_dims == [64, 32]
        assert model.dropout_rate == 0.5
    
    def test_mlp_forward_shape(self):
        """Forward pass retorna shape correto."""
        model = MLPChurn(input_dim=10, output_dim=2)
        X = torch.randn(32, 10)  # batch_size=32, features=10
        output = model.forward(X)
        assert output.shape == (32, 2)
    
    def test_mlp_parameters_count(self):
        """Count de parâmetros > 0."""
        model = MLPChurn()
        params = model.count_parameters()
        assert params > 0
        assert isinstance(params, int)
    
    def test_mlp_predict_shape(self):
        """Predict retorna shape (batch_size,)."""
        model = MLPChurn()
        X = torch.randn(16, 19)
        predictions = model.predict(X)
        assert predictions.shape == (16,)
        assert predictions.dtype == np.int64
    
    def test_mlp_predict_values(self):
        """Predict retorna valores 0 ou 1."""
        model = MLPChurn()
        X = torch.randn(10, 19)
        predictions = model.predict(X)
        assert np.all((predictions == 0) | (predictions == 1))
    
    def test_mlp_predict_proba_shape(self):
        """Predict proba retorna (batch_size, num_classes)."""
        model = MLPChurn()
        X = torch.randn(16, 19)
        proba = model.predict_proba(X)
        assert proba.shape == (16, 2)
    
    def test_mlp_predict_proba_valid_probabilities(self):
        """Probabilidades somam 1 e estão entre 0-1."""
        model = MLPChurn()
        X = torch.randn(10, 19)
        proba = model.predict_proba(X)
        
        # Somam 1
        sums = proba.sum(axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(10))
        
        # Entre 0 e 1
        assert np.all(proba >= 0) and np.all(proba <= 1)
    
    def test_mlp_save_load(self, tmp_path):
        """Salvar e carregar modelo."""
        model1 = MLPChurn()
        save_path = tmp_path / "model.pt"
        model1.save(save_path)

        model2 = MLPChurn()
        model2.load(save_path)

        # BatchNorm em eval() usa running statistics → saídas idênticas
        model1.eval()
        model2.eval()
        X = torch.randn(5, 19)
        with torch.no_grad():
            out1 = model1.forward(X)
            out2 = model2.forward(X)
        torch.testing.assert_close(out1, out2)
    
    def test_mlp_activation_functions(self):
        """Testar diferentes funções de ativação."""
        activations = ["relu", "elu", "tanh", "sigmoid"]
        for act in activations:
            model = MLPChurn(activation=act)
            X = torch.randn(8, 19)
            output = model.forward(X)
            assert output.shape == (8, 2)
    
    def test_create_mlp_model_factory(self):
        """Factory function cria modelo corretamente."""
        model = create_mlp_model(input_dim=20, output_dim=3, device='cpu')
        assert isinstance(model, MLPChurn)
        assert model.input_dim == 20
        assert model.output_dim == 3


class TestEarlyStopping:
    """Testes de early stopping."""
    
    def test_early_stopping_init(self):
        """Inicializar early stopping."""
        es = EarlyStopping(patience=5, delta=0.001)
        assert es.patience == 5
        assert es.delta == 0.001
        assert es.counter == 0
        assert not es.early_stop
    
    def test_early_stopping_improvement(self):
        """Early stopping com melhora."""
        es = EarlyStopping(patience=3, delta=0.01)
        
        # Primeira chamada
        assert not es(0.5)
        
        # Melhora
        assert not es(0.45)
        assert es.counter == 0
    
    def test_early_stopping_no_improvement(self):
        """Early stopping sem melhora."""
        es = EarlyStopping(patience=2, delta=0.01)
        
        # Primeira chamada
        es(0.5)
        
        # Sem melhora
        es(0.49)  # Menos que delta
        assert es.counter == 1
        
        es(0.49)
        assert es.counter == 2
        assert es(0.49)  # Early stop


class TestMLPTrainer:
    """Testes de trainer."""
    
    @pytest.fixture
    def sample_data(self):
        """Gerar dados de amostra."""
        np.random.seed(42)
        X_train = np.random.randn(100, 19).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        X_val = np.random.randn(20, 19).astype(np.float32)
        y_val = np.random.randint(0, 2, 20)
        return X_train, y_train, X_val, y_val
    
    def test_trainer_init(self):
        """Inicializar trainer."""
        model = MLPChurn()
        trainer = MLPTrainer(model, learning_rate=0.001)
        assert trainer.learning_rate == 0.001
        assert trainer.model is not None
    
    def test_trainer_history_structure(self, sample_data):
        """Histórico tem estrutura correta."""
        X_train, y_train, X_val, y_val = sample_data
        
        model = MLPChurn()
        trainer = MLPTrainer(model)
        
        _ = trainer.fit(X_train, y_train, X_val, y_val, epochs=2)
        
        history = trainer.history
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'train_acc' in history
        assert 'val_acc' in history
        assert len(history['train_loss']) > 0
    
    def test_trainer_fit_returns_dict(self, sample_data):
        """Fit retorna histórico."""
        X_train, y_train, X_val, y_val = sample_data
        
        model = MLPChurn()
        trainer = MLPTrainer(model)
        
        result = trainer.fit(X_train, y_train, X_val, y_val, epochs=2)
        assert isinstance(result, dict)
    
    def test_trainer_validation_reduces_loss(self, sample_data):
        """Loss deve reduzir durante treinamento."""
        X_train, y_train, X_val, y_val = sample_data
        
        model = MLPChurn()
        trainer = MLPTrainer(model, patience=20)
        
        trainer.fit(X_train, y_train, X_val, y_val, epochs=10)
        
        # Loss inicial > loss final
        initial_loss = trainer.history['train_loss'][0]
        final_loss = trainer.history['train_loss'][-1]
        assert initial_loss > final_loss


class TestCrossValidation:
    """Testes de validação cruzada."""
    
    @pytest.fixture
    def cv_data(self):
        """Dados para CV."""
        np.random.seed(42)
        X = np.random.randn(200, 19).astype(np.float32)
        y = np.random.randint(0, 2, 200)
        return X, y
    
    def test_cv_returns_models(self, cv_data):
        """CV retorna lista de modelos."""
        X, y = cv_data
        
        models, metrics, results = train_with_cross_validation(
            X, y, n_splits=2, epochs=5, random_state=42
        )
        
        assert len(models) == 2
        assert isinstance(metrics, list)
        assert len(metrics) == 2
    
    def test_cv_results_dataframe(self, cv_data):
        """CV retorna resultados como DataFrame."""
        X, y = cv_data
        
        _, _, results = train_with_cross_validation(
            X, y, n_splits=2, epochs=5, random_state=42
        )
        
        assert hasattr(results, 'shape')
        assert results.shape[0] == 2  # 2 folds
        assert 'final_val_acc' in results.columns


class TestIntegration:
    """Testes de integração."""
    
    def test_full_pipeline(self):
        """Pipeline completo: modelo → treino → predição."""
        # Dados
        np.random.seed(42)
        X_train = np.random.randn(100, 19).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 19).astype(np.float32)
        
        # Treinar
        model = MLPChurn()
        trainer = MLPTrainer(model, patience=10)
        trainer.fit(X_train, y_train, X_train, y_train, epochs=5)
        
        # Predizer
        X_test_t = torch.FloatTensor(X_test)
        pred = model.predict(X_test_t)
        proba = model.predict_proba(X_test_t)
        
        # Validações
        assert pred.shape == (20,)
        assert proba.shape == (20, 2)
        assert np.all((pred == 0) | (pred == 1))
        assert np.all((proba >= 0) & (proba <= 1))
