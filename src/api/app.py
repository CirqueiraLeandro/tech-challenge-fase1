"""
API FastAPI para previsão de churn — Etapa 3.

Endpoints:
- GET  /        : Root info
- GET  /health  : Status da API + modelo
- GET  /info    : Arquitetura do modelo carregado
- GET  /metrics : Estatísticas de uso
- POST /predict : Predição individual com 19 features reais
- POST /predict_batch : Predição em lote
"""

import json
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_config
from src.data import DataPreprocessor
from src.logger import setup_logger
from src.models.mlp import MLPChurn

config = get_config("production")
logger = setup_logger(__name__, config)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Estado global carregado em startup (lifespan)
_model: Optional[MLPChurn] = None
_preprocessor: Optional[DataPreprocessor] = None
_feature_metadata: Optional[Dict] = None
_model_loaded_at: Optional[datetime] = None


# ============================================================================
# LIFESPAN — carrega modelo + preprocessor uma única vez
# ============================================================================

def _load_artifacts() -> bool:
    """Carrega MLP, preprocessor e metadata do disco. Retorna True em sucesso."""
    global _model, _preprocessor, _feature_metadata, _model_loaded_at

    models_dir: Path = config.MODELS_DIR
    model_path = models_dir / "mlp_etapa2.pt"
    preprocessor_path = models_dir / "preprocessor.pkl"
    metadata_path = models_dir / "mlp_feature_metadata.json"

    if not model_path.exists():
        logger.warning(f"Modelo não encontrado: {model_path}")
        return False

    if not preprocessor_path.exists() or not metadata_path.exists():
        logger.warning(
            "Preprocessor ou metadata ausentes — rode `python -m src.train_mlp` "
            "para gerar todos os artefatos"
        )
        return False

    try:
        with open(metadata_path) as f:
            _feature_metadata = json.load(f)

        input_dim = int(_feature_metadata["input_dim"])
        _model = MLPChurn(input_dim=input_dim, hidden_dims=[128, 64, 32])
        _model.load(model_path)
        _model = _model.to(device)
        _model.eval()

        with open(preprocessor_path, "rb") as f:
            _preprocessor = pickle.load(f)

        _model_loaded_at = datetime.now()
        logger.info(
            f"API pronta: MLP({input_dim} features, "
            f"{_model.count_parameters():,} params) + preprocessor"
        )
        return True
    except Exception as e:
        logger.error(f"Falha ao carregar artefatos: {e}")
        _model = None
        _preprocessor = None
        _feature_metadata = None
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando API...")
    _load_artifacts()
    yield
    logger.info("Encerrando API...")


app = FastAPI(
    title="Telco Churn Prediction API",
    description="API para previsão de churn em clientes de telecomunicações",
    version="2.1.0",
    lifespan=lifespan,
)


class LatencyMiddleware(BaseHTTPMiddleware):
    """Adiciona header X-Process-Time-Ms e loga latência de cada request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} ({elapsed_ms:.1f}ms)"
        )
        return response


app.add_middleware(LatencyMiddleware)


# ============================================================================
# SCHEMAS — alinhados às 19 features reais do dataset IBM Telco DAT00148
# ============================================================================

YesNo = Literal["Yes", "No"]
InternetDependent = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    """Features de um cliente — 3 numéricas + 16 categóricas (mesmas usadas no treino)."""

    model_config = ConfigDict(populate_by_name=True)

    # Numeric
    tenure_months: int = Field(..., ge=0, le=100, alias="Tenure Months")
    monthly_charges: float = Field(..., ge=0, alias="Monthly Charges")
    total_charges: float = Field(..., ge=0, alias="Total Charges")

    # Categorical
    gender: Literal["Male", "Female"] = Field(..., alias="Gender")
    senior_citizen: YesNo = Field(..., alias="Senior Citizen")
    partner: YesNo = Field(..., alias="Partner")
    dependents: YesNo = Field(..., alias="Dependents")
    phone_service: YesNo = Field(..., alias="Phone Service")
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(
        ..., alias="Multiple Lines"
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., alias="Internet Service"
    )
    online_security: InternetDependent = Field(..., alias="Online Security")
    online_backup: InternetDependent = Field(..., alias="Online Backup")
    device_protection: InternetDependent = Field(..., alias="Device Protection")
    tech_support: InternetDependent = Field(..., alias="Tech Support")
    streaming_tv: InternetDependent = Field(..., alias="Streaming TV")
    streaming_movies: InternetDependent = Field(..., alias="Streaming Movies")
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., alias="Contract"
    )
    paperless_billing: YesNo = Field(..., alias="Paperless Billing")
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., alias="Payment Method")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = não-churn, 1 = churn")
    probability_no_churn: float = Field(..., ge=0, le=1)
    probability_churn: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    timestamp: datetime


class BatchPredictionRequest(BaseModel):
    data: List[Dict]
    return_probabilities: bool = True


class BatchPredictionItem(BaseModel):
    index: int
    prediction: Optional[int] = None
    probability_no_churn: Optional[float] = None
    probability_churn: Optional[float] = None
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    total_predictions: int
    succeeded: int
    failed: int
    predictions: List[Dict]
    processing_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    model_loaded_at: Optional[datetime] = None
    uptime_seconds: float


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    input_dim: int
    output_dim: int
    hidden_dims: List[int]
    parameters: int
    device: str
    loaded_at: Optional[datetime] = None
    feature_order: List[str]


# ============================================================================
# Inferência real
# ============================================================================

def _features_to_array(payload: Dict) -> np.ndarray:
    """Aplica preprocessor + reordena para o input_dim esperado pelo MLP."""
    if _preprocessor is None or _feature_metadata is None:
        raise RuntimeError("Preprocessor não inicializado")

    cat_cols = _feature_metadata["categorical_features"]
    num_cols = _feature_metadata["numeric_features"]
    feature_order = _feature_metadata["feature_order"]

    df = pd.DataFrame([payload])
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"Features ausentes no payload: {missing}")

    df = _preprocessor.encode_categorical(df, cat_cols, fit=False)
    df = _preprocessor.scale_numeric(df, num_cols, fit=False)
    return df[feature_order].to_numpy(dtype=np.float32)


def _predict_one(payload: Dict) -> Dict:
    X = _features_to_array(payload)
    X_t = torch.FloatTensor(X).to(device)
    proba = _model.predict_proba(X_t)[0]
    pred = int(np.argmax(proba))
    return {
        "prediction": pred,
        "probability_no_churn": float(proba[0]),
        "probability_churn": float(proba[1]),
        "confidence": float(np.max(proba)),
    }


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "Telco Churn Prediction API",
        "version": "2.1.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/info",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        model_loaded=_model is not None and _preprocessor is not None,
        model_loaded_at=_model_loaded_at,
        uptime_seconds=time.time(),
    )


@app.get("/info", response_model=ModelInfo)
async def info():
    if _model is None or _feature_metadata is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    return ModelInfo(
        model_type="MLPChurn",
        input_dim=_model.input_dim,
        output_dim=_model.output_dim,
        hidden_dims=_model.hidden_dims,
        parameters=_model.count_parameters(),
        device=device,
        loaded_at=_model_loaded_at,
        feature_order=_feature_metadata["feature_order"],
    )


@app.get("/metrics")
async def metrics():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": _model is not None and _preprocessor is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CustomerFeatures):
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    try:
        payload = features.model_dump(by_alias=True)
        result = _predict_one(payload)
        logger.info(
            f"Predição: {result['prediction']} | conf={result['confidence']:.4f}"
        )
        return PredictionResponse(timestamp=datetime.now(), **result)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    start = time.time()
    predictions: List[Dict] = []
    succeeded = 0
    failed = 0

    for i, sample in enumerate(request.data):
        try:
            result = _predict_one(sample)
            item = {"index": i, "prediction": result["prediction"]}
            if request.return_probabilities:
                item["probability_no_churn"] = result["probability_no_churn"]
                item["probability_churn"] = result["probability_churn"]
            predictions.append(item)
            succeeded += 1
        except Exception as e:
            logger.error(f"Erro na amostra {i}: {e}")
            failed += 1
            predictions.append({"index": i, "error": str(e)})

    return BatchPredictionResponse(
        total_predictions=len(request.data),
        succeeded=succeeded,
        failed=failed,
        predictions=predictions,
        processing_time_seconds=time.time() - start,
    )


# ============================================================================
# Error handling
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando servidor uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
