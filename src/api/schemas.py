"""Schemas Pydantic para request/response da API de predição."""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """Payload de entrada para predição de churn."""
    gender: str = Field(..., description="Gênero do cliente")
    senior_citizen: int = Field(..., ge=0, le=1, description="É cidadão sênior (0/1)")
    partner: str = Field(..., description="Tem parceiro (Yes/No)")
    dependents: str = Field(..., description="Tem dependentes (Yes/No)")
    tenure: float = Field(..., ge=0, description="Meses como cliente")
    phone_service: str = Field(..., description="Possui serviço telefônico (Yes/No)")
    multiple_lines: str = Field(..., description="Múltiplas linhas")
    internet_service: str = Field(..., description="Tipo de serviço de internet")
    online_security: str = Field(..., description="Segurança online")
    online_backup: str = Field(..., description="Backup online")
    device_protection: str = Field(..., description="Proteção de dispositivo")
    tech_support: str = Field(..., description="Suporte técnico")
    streaming_tv: str = Field(..., description="Streaming TV")
    streaming_movies: str = Field(..., description="Streaming filmes")
    contract: str = Field(..., description="Tipo de contrato")
    paperless_billing: str = Field(..., description="Fatura digital (Yes/No)")
    payment_method: str = Field(..., description="Método de pagamento")
    monthly_charges: float = Field(..., ge=0, description="Cobranças mensais")
    total_charges: float = Field(..., ge=0, description="Total cobrado")


class PredictionResponse(BaseModel):
    """Resposta da predição de churn."""
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Probabilidade de churn")
    churn_prediction: bool = Field(..., description="Predição binária de churn")
    confidence: str = Field(..., description="Nível de confiança: low/medium/high")
    model_version: Optional[str] = Field(None, description="Versão do modelo")
