"""
Testes para API FastAPI - Etapa 3.

Cobrem:
- Health / Info / Metrics / Root
- Predição individual com 19 features reais (Telco IBM)
- Predição em lote
- Validação de schema
- Error handling (modelo ausente, payload inválido)
"""

import json
import pickle
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.models.mlp import MLPChurn


VALID_PAYLOAD = {
    "Tenure Months": 12,
    "Monthly Charges": 65.5,
    "Total Charges": 786.0,
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
}


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def loaded_artifacts(tmp_path_factory):
    """Carrega artefatos reais salvos pelo train_mlp para os testes que precisam."""
    from src.config import get_config
    config = get_config("production")

    preprocessor_path = config.MODELS_DIR / "preprocessor.pkl"
    metadata_path = config.MODELS_DIR / "mlp_feature_metadata.json"
    model_path = config.MODELS_DIR / "mlp_etapa2.pt"

    if not (preprocessor_path.exists() and metadata_path.exists() and model_path.exists()):
        pytest.skip("Artefatos não encontrados — rode etapa1 + train_mlp antes")

    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    with open(metadata_path) as f:
        metadata = json.load(f)
    model = MLPChurn(input_dim=metadata["input_dim"], hidden_dims=[128, 64, 32])
    model.load(model_path)
    model.eval()

    return {"model": model, "preprocessor": preprocessor, "metadata": metadata}


@pytest.fixture
def patched_app(loaded_artifacts):
    """Patch _model, _preprocessor, _feature_metadata na app."""
    with patch("src.api.app._model", loaded_artifacts["model"]), \
         patch("src.api.app._preprocessor", loaded_artifacts["preprocessor"]), \
         patch("src.api.app._feature_metadata", loaded_artifacts["metadata"]), \
         patch("src.api.app._model_loaded_at", datetime.now()):
        yield


# ============================================================================
# /health, /, /metrics
# ============================================================================

class TestRoot:
    def test_root_status(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_payload(self, client):
        data = client.get("/").json()
        assert data["name"] == "Telco Churn Prediction API"
        assert "version" in data and "docs" in data


class TestHealth:
    def test_health_status(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_payload(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "model_loaded" in data
        # Timestamp parseável
        datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))


class TestMetrics:
    def test_metrics_status(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_payload(self, client):
        data = client.get("/metrics").json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


# ============================================================================
# /info
# ============================================================================

class TestInfo:
    def test_info_no_model_returns_503(self, client):
        with patch("src.api.app._model", None), patch("src.api.app._feature_metadata", None):
            response = client.get("/info")
            assert response.status_code == 503

    def test_info_with_model(self, client, patched_app):
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "MLPChurn"
        assert data["input_dim"] == 19
        assert data["output_dim"] == 2
        assert data["hidden_dims"] == [128, 64, 32]
        assert data["parameters"] == 13410
        assert len(data["feature_order"]) == 19


# ============================================================================
# /predict
# ============================================================================

class TestPredict:
    def test_predict_503_when_no_model(self, client):
        with patch("src.api.app._model", None), patch("src.api.app._preprocessor", None):
            response = client.post("/predict", json=VALID_PAYLOAD)
            assert response.status_code == 503

    def test_predict_real_inference(self, client, patched_app):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability_no_churn"] <= 1
        assert 0 <= data["probability_churn"] <= 1
        # Probabilidades somam ~1
        assert abs(data["probability_no_churn"] + data["probability_churn"] - 1.0) < 1e-5
        assert data["confidence"] == max(
            data["probability_no_churn"], data["probability_churn"]
        )

    def test_predict_invalid_gender(self, client, patched_app):
        bad = VALID_PAYLOAD | {"Gender": "Other"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_predict_invalid_contract(self, client, patched_app):
        bad = VALID_PAYLOAD | {"Contract": "Lifetime"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_predict_missing_field(self, client, patched_app):
        bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "Contract"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_predict_negative_charges_rejected(self, client, patched_app):
        bad = VALID_PAYLOAD | {"Monthly Charges": -10}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_predict_uses_real_features(self, client, patched_app):
        """Cliente fiel à base + Month-to-month tem maior prob de churn que 2-year contract."""
        month_to_month = client.post("/predict", json=VALID_PAYLOAD).json()
        two_year = client.post(
            "/predict", json=VALID_PAYLOAD | {"Contract": "Two year"}
        ).json()
        assert (
            month_to_month["probability_churn"]
            > two_year["probability_churn"]
        ), "Contract real deveria afetar a predição"


# ============================================================================
# /predict_batch
# ============================================================================

class TestPredictBatch:
    def test_batch_503_when_no_model(self, client):
        with patch("src.api.app._model", None), patch("src.api.app._preprocessor", None):
            response = client.post("/predict_batch", json={"data": [VALID_PAYLOAD]})
            assert response.status_code == 503

    def test_batch_predict_two_samples(self, client, patched_app):
        request = {"data": [VALID_PAYLOAD, VALID_PAYLOAD], "return_probabilities": True}
        response = client.post("/predict_batch", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["total_predictions"] == 2
        assert data["succeeded"] == 2
        assert data["failed"] == 0
        for item in data["predictions"]:
            assert item["prediction"] in [0, 1]
            assert "probability_churn" in item

    def test_batch_no_probabilities(self, client, patched_app):
        request = {"data": [VALID_PAYLOAD], "return_probabilities": False}
        data = client.post("/predict_batch", json=request).json()
        assert "probability_churn" not in data["predictions"][0]

    def test_batch_partial_failure(self, client, patched_app):
        bad_sample = {"Gender": "Male"}  # missing fields
        request = {"data": [VALID_PAYLOAD, bad_sample], "return_probabilities": True}
        data = client.post("/predict_batch", json=request).json()
        assert data["total_predictions"] == 2
        assert data["succeeded"] == 1
        assert data["failed"] == 1
        assert "error" in data["predictions"][1]


# ============================================================================
# Latency middleware
# ============================================================================

class TestLatencyHeader:
    def test_x_process_time_header(self, client):
        response = client.get("/health")
        assert "X-Process-Time-Ms" in response.headers
        elapsed_ms = float(response.headers["X-Process-Time-Ms"])
        assert elapsed_ms >= 0
