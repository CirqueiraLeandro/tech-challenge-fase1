# Dockerfile para API Telco Churn Prediction

FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY pyproject.toml .

# Instalar dependências Python
RUN pip install --no-cache-dir -e .

# Copiar código
COPY src/ /app/src/
COPY data/ /app/data/
COPY models/ /app/models/

# Criar diretórios
RUN mkdir -p /app/logs

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Comando padrão
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
