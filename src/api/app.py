"""
API FastAPI para previsão de churn - Etapa 3.

Endpoints:
- GET /health: Status da API
- POST /predict: Predição individual
- POST /predict_batch: Predição em lote
- GET /info: Informações do modelo
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
import time
import numpy as np
import torch
from datetime import datetime

from src.config import get_config
from src.data import DataPreprocessor
from src.models.mlp import MLPChurn
from src.logger import setup_logger


# Setup
config = get_config("production")
logger = setup_logger(__name__, config)
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API para previsão de churn em clientes de telecomunicações",
    version="2.0.0"
)


class LatencyMiddleware(BaseHTTPMiddleware):
    """Adiciona header X-Process-Time e loga latência de cada request."""

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

# Config global
config = get_config("production")
device = "cuda" if torch.cuda.is_available() else "cpu"
preprocessor = DataPreprocessor()

# Modelo global
_model: Optional[MLPChurn] = None
_model_loaded_at: Optional[datetime] = None


def load_model():
    """Carregar modelo na memória."""
    global _model, _model_loaded_at

    model_path = config.MODELS_DIR / "mlp_etapa2.pt"

    if model_path.exists():
        try:
            # input_dim=19 após remoção de features com data leakage
            _model = MLPChurn(input_dim=19, hidden_dims=[128, 64, 32])
            _model.load(model_path)
            _model = _model.to(device)
            _model.eval()
            _model_loaded_at = datetime.now()
            logger.info(f"Modelo carregado de {model_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False
    else:
        logger.warning(f"Arquivo de modelo não encontrado: {model_path}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Iniciando API...")
    if load_model():
        logger.info("API pronta para requisições")
    else:
        logger.warning("API iniciada sem modelo carregado")


# ============================================================================
# SCHEMAS
# ============================================================================

class HealthResponse(BaseModel):
    """Response para health check."""
    status: str
    timestamp: datetime
    model_loaded: bool
    model_loaded_at: Optional[datetime] = None
    uptime_seconds: float


class PredictionFeatures(BaseModel):
    """Features para predição individual."""
    age: int = Field(..., ge=18, le=100, description="Idade do cliente")
    tenure: int = Field(..., ge=0, le=72, description="Meses como cliente")
    monthly_charges: float = Field(..., gt=0, description="Custo mensal")
    total_charges: float = Field(..., ge=0, description="Custo total")
    
    # Features categóricas (exemplos)
    gender: str = Field(..., description="Gênero: M/F")
    internet_service: str = Field(..., description="Tipo de serviço de internet")
    contract: str = Field(..., description="Tipo de contrato")
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        if v not in ['M', 'F']:
            raise ValueError('Gênero deve ser M ou F')
        return v


class PredictionResponse(BaseModel):
    """Response de predição."""
    prediction: int = Field(..., description="Classe predita (0: não churn, 1: churn)")
    probability_no_churn: float = Field(..., ge=0, le=1)
    probability_churn: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1, description="Confiança da predição")
    timestamp: datetime


class BatchPredictionRequest(BaseModel):
    """Request para predição em batch."""
    data: List[Dict]
    return_probabilities: bool = True


class BatchPredictionResponse(BaseModel):
    """Response para predição em batch."""
    total_predictions: int
    succeeded: int
    failed: int
    predictions: List[Dict]
    processing_time_seconds: float


class ModelInfo(BaseModel):
    """Informações do modelo."""
    model_type: str
    input_dim: int
    output_dim: int
    hidden_dims: List[int]
    parameters: int
    device: str
    loaded_at: Optional[datetime]


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check da API.
    
    Returns:
        Status da API e modelo
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        model_loaded=_model is not None,
        model_loaded_at=_model_loaded_at,
        uptime_seconds=time.time(),
    )


@app.get("/info", response_model=ModelInfo)
async def model_info():
    """
    Informações do modelo.
    
    Returns:
        Detalhes da arquitetura e status
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    return ModelInfo(
        model_type="MLPChurn",
        input_dim=_model.input_dim,
        output_dim=_model.output_dim,
        hidden_dims=_model.hidden_dims,
        parameters=_model.count_parameters(),
        device=device,
        loaded_at=_model_loaded_at,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: PredictionFeatures):
    """
    Predição individual.
    
    Args:
        features: Features do cliente
        
    Returns:
        Predição e probabilidades
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        logger.info("Predição recebida para cliente")
        
        # Converter para numpy (simplificado - em produção, usar preprocessamento completo)
        X = np.array([[
            features.age,
            features.tenure,
            features.monthly_charges,
            features.total_charges,
        ]], dtype=np.float32)
        
        # Normalizar (usar preprocessador real em produção)
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Pad to 19 features (input_dim pós leakage fix)
        X_padded = np.zeros((1, 19), dtype=np.float32)
        X_padded[:, :4] = X
        
        # Predição
        X_t = torch.FloatTensor(X_padded).to(device)
        proba = _model.predict_proba(X_t)[0]
        pred = np.argmax(proba)
        confidence = np.max(proba)
        
        logger.info(f"Predição: {pred}, Confiança: {confidence:.4f}")
        
        return PredictionResponse(
            prediction=int(pred),
            probability_no_churn=float(proba[0]),
            probability_churn=float(proba[1]),
            confidence=float(confidence),
            timestamp=datetime.now(),
        )
        
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predição em lote.
    
    Args:
        request: Lista de dados para predição
        
    Returns:
        Predições em lote
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    start_time = time.time()
    
    try:
        predictions = []
        succeeded = 0
        failed = 0
        
        for i, sample in enumerate(request.data):
            try:
                # Simplificado - em produção, usar preprocessador completo
                X = np.array([[
                    sample.get('age', 0),
                    sample.get('tenure', 0),
                    sample.get('monthly_charges', 0),
                    sample.get('total_charges', 0),
                ]], dtype=np.float32)
                
                X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
                X_padded = np.zeros((1, 19), dtype=np.float32)
                X_padded[:, :4] = X
                
                X_t = torch.FloatTensor(X_padded).to(device)
                proba = _model.predict_proba(X_t)[0]
                pred = np.argmax(proba)
                
                pred_dict = {
                    'index': i,
                    'prediction': int(pred),
                }
                if request.return_probabilities:
                    pred_dict['probability_no_churn'] = float(proba[0])
                    pred_dict['probability_churn'] = float(proba[1])
                
                predictions.append(pred_dict)
                succeeded += 1
                
            except Exception as e:
                logger.error(f"Erro na amostra {i}: {e}")
                failed += 1
                predictions.append({
                    'index': i,
                    'error': str(e)
                })
        
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            total_predictions=len(request.data),
            succeeded=succeeded,
            failed=failed,
            predictions=predictions,
            processing_time_seconds=processing_time,
        )
        
    except Exception as e:
        logger.error(f"Erro em predição batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """
    Métricas da API (simplificado).
    
    Returns:
        Estatísticas de uso
    """
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": _model is not None,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Telco Churn Prediction API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/info",
    }


# ============================================================================
# Error handling
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler para HTTPException — preserva o status code correto."""
    logger.error(f"HTTP Error {exc.status_code}: {exc.detail}")
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
    logger.info("Iniciando servidor...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
