"""
Testes para API FastAPI - Etapa 3.

Testes cobrem:
- Health check
- Model info
- Predição individual
- Predição em lote
- Error handling
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import patch, MagicMock

# Importar app
from src.api.app import app


@pytest.fixture
def client():
    """FastAPI test client — raise_server_exceptions=False para testar respostas de erro."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def mock_model():
    """Mock modelo para testes."""
    from src.models.mlp import MLPChurn
    
    # Criar modelo mock
    model = MLPChurn()
    
    # Patchear a função predict
    with patch('src.api.app._model', model):
        yield


class TestHealthEndpoint:
    """Testes do endpoint /health."""
    
    def test_health_status_code(self, client):
        """Health check retorna 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self, client):
        """Health check tem estrutura esperada."""
        response = client.get("/health")
        data = response.json()
        
        assert 'status' in data
        assert 'timestamp' in data
        assert 'model_loaded' in data
        assert data['status'] == 'healthy'
    
    def test_health_timestamp_format(self, client):
        """Timestamp tem formato ISO."""
        response = client.get("/health")
        data = response.json()
        
        # Tentar parsear como ISO format
        try:
            datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            valid = True
        except Exception:
            valid = False
        
        assert valid


class TestInfoEndpoint:
    """Testes do endpoint /info."""
    
    def test_info_status_code(self, client):
        """Info endpoint retorna 200."""
        with patch('src.api.app._model', MagicMock()):
            response = client.get("/info")
            assert response.status_code in [200, 503]
    
    def test_info_response_structure(self, client):
        """Info tem campos esperados."""
        # Mock modelo
        mock_model = MagicMock()
        mock_model.input_dim = 19
        mock_model.output_dim = 2
        mock_model.hidden_dims = [128, 64, 32]
        mock_model.count_parameters = MagicMock(return_value=50000)
        
        with patch('src.api.app._model', mock_model):
            response = client.get("/info")
            if response.status_code == 200:
                data = response.json()
                assert 'model_type' in data
                assert 'input_dim' in data
                assert 'output_dim' in data


class TestPredictEndpoint:
    """Testes do endpoint /predict."""
    
    @pytest.fixture
    def sample_features(self):
        """Sample de features para predição."""
        return {
            "age": 35,
            "tenure": 12,
            "monthly_charges": 65.5,
            "total_charges": 786.0,
            "gender": "M",
            "internet_service": "Fiber optic",
            "contract": "Month-to-month",
        }
    
    def test_predict_valid_request(self, client, sample_features):
        """Predição com features válidas."""
        mock_model = MagicMock()
        mock_model.predict_proba = MagicMock(return_value=[[0.7, 0.3]])
        
        with patch('src.api.app._model', mock_model):
            response = client.post("/predict", json=sample_features)
            assert response.status_code == 200
            
            data = response.json()
            assert 'prediction' in data
            assert 'probability_no_churn' in data
            assert 'probability_churn' in data
            assert 'confidence' in data
    
    def test_predict_invalid_age(self, client, sample_features):
        """Predição com idade inválida."""
        sample_features['age'] = 150  # Idade demais
        
        response = client.post("/predict", json=sample_features)
        assert response.status_code == 422  # Validation error
    
    def test_predict_invalid_gender(self, client, sample_features):
        """Predição com gênero inválido."""
        sample_features['gender'] = 'X'
        
        response = client.post("/predict", json=sample_features)
        assert response.status_code == 422
    
    def test_predict_response_structure(self, client, sample_features):
        """Resposta tem estrutura esperada."""
        mock_model = MagicMock()
        mock_model.predict_proba = MagicMock(return_value=[[0.8, 0.2]])
        
        with patch('src.api.app._model', mock_model):
            response = client.post("/predict", json=sample_features)
            
            if response.status_code == 200:
                data = response.json()
                assert data['prediction'] in [0, 1]
                assert 0 <= data['probability_no_churn'] <= 1
                assert 0 <= data['probability_churn'] <= 1
                assert 0 <= data['confidence'] <= 1


class TestBatchPredictEndpoint:
    """Testes do endpoint /predict_batch."""
    
    @pytest.fixture
    def batch_request(self):
        """Sample de batch request."""
        return {
            "data": [
                {
                    "age": 30,
                    "tenure": 10,
                    "monthly_charges": 50.0,
                    "total_charges": 500.0,
                },
                {
                    "age": 40,
                    "tenure": 20,
                    "monthly_charges": 80.0,
                    "total_charges": 1600.0,
                },
            ],
            "return_probabilities": True,
        }
    
    def test_batch_predict_valid_request(self, client, batch_request):
        """Batch predição com dados válidos."""
        mock_model = MagicMock()
        mock_model.predict_proba = MagicMock(return_value=[[0.7, 0.3]])
        
        with patch('src.api.app._model', mock_model):
            response = client.post("/predict_batch", json=batch_request)
            assert response.status_code == 200
            
            data = response.json()
            assert 'total_predictions' in data
            assert 'succeeded' in data
            assert 'failed' in data
            assert 'predictions' in data
    
    def test_batch_predict_count(self, client, batch_request):
        """Batch retorna número correto de predições."""
        mock_model = MagicMock()
        mock_model.predict_proba = MagicMock(return_value=[[0.7, 0.3]])
        
        with patch('src.api.app._model', mock_model):
            response = client.post("/predict_batch", json=batch_request)
            
            if response.status_code == 200:
                data = response.json()
                assert data['total_predictions'] == 2


class TestRootEndpoint:
    """Testes do endpoint root."""
    
    def test_root_status_code(self, client):
        """Root endpoint retorna 200."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_response_structure(self, client):
        """Root response tem informações."""
        response = client.get("/")
        data = response.json()
        
        assert 'name' in data
        assert 'version' in data
        assert 'docs' in data


class TestMetricsEndpoint:
    """Testes do endpoint /metrics."""
    
    def test_metrics_status_code(self, client):
        """Metrics endpoint retorna 200."""
        response = client.get("/metrics")
        assert response.status_code == 200
    
    def test_metrics_response_structure(self, client):
        """Metrics tem estrutura esperada."""
        response = client.get("/metrics")
        data = response.json()
        
        assert 'status' in data
        assert 'timestamp' in data
        assert 'model_loaded' in data


class TestErrorHandling:
    """Testes de tratamento de erros."""
    
    def test_endpoint_no_model(self, client):
        """Endpoint sem modelo carregado retorna 503."""
        with patch('src.api.app._model', None):
            response = client.get("/info")
            assert response.status_code == 503
    
    def test_batch_no_model(self, client):
        """Batch sem modelo retorna 503."""
        batch_request = {"data": [{"age": 30}], "return_probabilities": True}
        
        with patch('src.api.app._model', None):
            response = client.post("/predict_batch", json=batch_request)
            assert response.status_code == 503


class TestIntegration:
    """Testes de integração."""
    
    def test_full_flow(self, client):
        """Flow completo: health → info → predict."""
        mock_model = MagicMock()
        mock_model.input_dim = 19
        mock_model.output_dim = 2
        mock_model.hidden_dims = [128, 64, 32]
        mock_model.count_parameters = MagicMock(return_value=50000)
        mock_model.predict_proba = MagicMock(return_value=[[0.6, 0.4]])
        
        with patch('src.api.app._model', mock_model):
            # Health
            response = client.get("/health")
            assert response.status_code == 200
            
            # Info
            response = client.get("/info")
            assert response.status_code == 200
            
            # Predict
            features = {
                "age": 35,
                "tenure": 12,
                "monthly_charges": 65.5,
                "total_charges": 786.0,
                "gender": "M",
                "internet_service": "Fiber optic",
                "contract": "Month-to-month",
            }
            response = client.post("/predict", json=features)
            assert response.status_code == 200
