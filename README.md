# Telco Churn Prediction - Machine Learning End-to-End

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-130%2B%20passing-brightgreen)](#testes-automatizados)

**Desafio**: Operadora de telecomunicações está perdendo clientes em ritmo acelerado. Necessita prever quais clientes têm risco de churn.

**Solução**: Pipeline de ML profissional end-to-end com:
- Rede neural MLP (PyTorch) com 13,410 parâmetros
- EDA completa + 3 baselines
- API FastAPI v2.0.0 com predição em tempo real
- Monitoramento contínuo e Model Card formal

---

## Dataset e Performance

**Fonte**: [IBM Accelerator Catalog — Telco Customer Churn (DAT00148)](https://accelerator.ca.analytics.ibm.com/bi/?perspective=authoring&pathRef=.public_folders%2FIBM%2BAccelerator%2BCatalog%2FContent%2FDAT00148&id=i9710CF25EF75468D95FFFC7D57D45204&objRef=i9710CF25EF75468D95FFFC7D57D45204&action=run&format=HTML&cmPropStr=%7B%22id%22%3A%22i9710CF25EF75468D95FFFC7D57D45204%22%2C%22type%22%3A%22reportView%22%2C%22defaultName%22%3A%22DAT00148%22%2C%22permissions%22%3A%5B%22execute%22%2C%22read%22%2C%22traverse%22%5D%7D)  
Arquivo: `data/raw/Telco_customer_churn.xlsx`

| Métrica | Valor |
|---------|-------|
| **Clientes** | 7,043 |
| **Features (pós remoção de data leakage)** | 19 (3 numéricas + 16 categóricas) |
| **Churn Rate** | 26.54% (1,869 churn / 5,174 retidos) |
| **Train/Test** | 80/20 estratificado (5,634 / 1,409) |

### Performance dos Modelos (pós correção de data leakage)

| Modelo | Accuracy | F1 | AUC-ROC | PR-AUC |
|--------|----------|----|---------|--------|
| DummyClassifier | 73.46% | 0.00 | 0.500 | 0.265 |
| LogisticRegression | 80.34% | 0.611 | **0.848** | 0.644 |
| RandomForest | 79.35% | 0.565 | 0.834 | 0.627 |
| **MLP PyTorch (Etapa 2)** | 79.28% | 0.556 | 0.837 | 0.649 |

> **Nota sobre data leakage**: As colunas `Churn Score`, `Churn Reason`, `CLTV` e identificadores geográficos foram removidas por causar leakage do target. As métricas acima refletem performance realista.

---

## Quick Start

### 1. Instalação

```bash
git clone <repo>
cd Telecommunications_Industry
pip install -e .
pip install -e ".[dev]"
```

### 2. Executar Pipelines

```bash
# Etapa 1: EDA + Baselines
python -m src.etapa1_eda_baselines

# Etapa 2: Treinar MLP (PyTorch)
python -m src.train_mlp

# Etapa 3: Iniciar API
python -m uvicorn src.api.app:app --reload --port 8000
# Swagger: http://localhost:8000/docs
```

### 3. Executar Testes

```bash
# Todos os testes
python -m pytest tests/ -v --cov=src

# Testes de schema (pandera)
python -m pytest tests/ -m schema

# Testes smoke
python -m pytest tests/ -m smoke
```

---

## Estrutura do Projeto

```
Telecommunications_Industry/
├── src/
│   ├── config.py                    # Configurações centralizadas
│   ├── logger.py                    # Logging estruturado
│   ├── constants.py                 # Constantes do projeto
│   ├── data.py                      # DataLoader + DataPreprocessor
│   ├── baselines.py                 # BaselineTrainer (3 modelos)
│   ├── mlflow_utils.py              # MLflow tracking
│   ├── etapa1_eda_baselines.py      # Script Etapa 1
│   ├── train_mlp.py                 # Script Etapa 2
│   ├── models/
│   │   ├── mlp.py                   # Arquitetura MLP (13.410 params)
│   │   └── training.py              # Loop de treinamento + early stopping
│   ├── api/
│   │   ├── app.py                   # FastAPI v2.0.0
│   │   └── schemas.py               # Pydantic v2 validation
│   ├── preprocessing/
│   │   └── pipeline.py              # Sklearn pipeline (ColumnTransformer)
│   └── monitoring/
│       └── drift.py                 # PSI + feature drift detection
│
├── data/raw/
│   └── Telco_customer_churn.xlsx    # Dataset original (7,043 clientes)
│
├── models/
│   ├── dummyclassifier_baseline.pkl
│   ├── logisticregression_baseline.pkl
│   ├── randomforest_baseline.pkl
│   ├── scaler.pkl
│   ├── mlp_etapa2.pt                # MLP treinado (input_dim=19)
│   ├── etapa1_metadata.json
│   ├── baseline_comparison.csv
│   └── mlp_comparison.csv
│
├── tests/
│   ├── conftest.py
│   ├── test_config_logging.py       # Config + Logging
│   ├── test_data_expanded.py        # DataLoader + DataPreprocessor
│   ├── test_schema_pandera.py       # Schema validation (pandera)
│   ├── test_mlp_etapa2.py           # MLP architecture + training
│   ├── test_models_expanded.py      # Baselines
│   ├── test_api_etapa3.py           # FastAPI endpoints
│   ├── test_mlflow.py               # MLflow tracking
│   ├── test_e2e.py                  # Pipeline básico E2E
│   └── test_integration_e2e.py      # Integration completa
│
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── ML_CANVAS.pdf
│   ├── MODEL_CARD.pdf
│   ├── MONITORING.md
│   └── TECHNICAL_GUIDE.md
│
├── notebooks/
│   └── 01_eda_and_baselines.ipynb
│
├── references/                      # PDFs das aulas por disciplina
├── Makefile
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Etapas de Desenvolvimento

### Etapa 1: Exploração e Baselines

```bash
python -m src.etapa1_eda_baselines
```

- EDA completa: 7,043 clientes, 19 features (pós leakage fix)
- 3 baselines: DummyClassifier, LogisticRegression, RandomForest
- Output: `models/*.pkl` + `etapa1_metadata.json` + `baseline_comparison.csv`

---

### Etapa 2: Rede Neural MLP (PyTorch)

```bash
python -m src.train_mlp
```

- Arquitetura: 19 → 128 → 64 → 32 → 2 (13,410 parâmetros)
- BatchNorm + ReLU + Dropout(0.3)
- Early stopping (patience=10)
- Output: `models/mlp_etapa2.pt` + `models/mlp_comparison.csv`

---

### Etapa 3: API FastAPI

```bash
python -m uvicorn src.api.app:app --reload --port 8000
```

**Endpoints**:

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/` | Root info |
| GET | `/health` | Health check + model status |
| GET | `/info` | Arquitetura do modelo carregado |
| GET | `/metrics` | Estatísticas de uso |
| POST | `/predict` | Predição individual |
| POST | `/predict_batch` | Predição em lote |

**Exemplo `/predict`**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35, "tenure": 12,
    "monthly_charges": 65.5, "total_charges": 786.0,
    "gender": "M", "internet_service": "Fiber optic",
    "contract": "Month-to-month"
  }'
```

**Resposta**:
```json
{
  "prediction": 0,
  "probability_no_churn": 0.887,
  "probability_churn": 0.113,
  "confidence": 0.887,
  "timestamp": "2026-04-25T14:56:08.418976"
}
```

---

## Testes Automatizados

```bash
python -m pytest tests/ -v
```

| Suite | Testes | Descrição |
|-------|--------|-----------|
| test_config_logging.py | 34 | Config + Logging |
| test_data_expanded.py | 22 | DataLoader + Preprocessor |
| test_schema_pandera.py | 12 | Schema validation (pandera) |
| test_mlp_etapa2.py | 21 | Arquitetura MLP |
| test_models_expanded.py | 15 | Baselines |
| test_api_etapa3.py | 18 | FastAPI endpoints |
| test_mlflow.py | 11 | MLflow tracking |
| test_everything.py | ~10 | Integration |
| **Total** | **130+** | **todos passando** |

**Marcadores python -m pytest**:
```bash
python -m pytest tests/ -m smoke    # Smoke tests críticos
python -m pytest tests/ -m schema   # Schema validation (pandera)
```

---

## MLflow Tracking

```bash
python -m mlflow ui --host 0.0.0.0 --port 5000
# Acesso: http://localhost:5000
```

Backend: `file:./mlruns` (local). Experimento: `telco-churn-prediction`.

---

## Arquitetura de Deploy: Real-Time (escolhida)

**Decisão**: inferência em tempo real via API REST. Batch descartado pelo caso de uso.

| Critério | Real-Time ✅ | Batch |
|---|---|---|
| Latência | < 100ms | Horas |
| Integração | CRM / callcenter ao vivo | Data warehouse / relatório |
| Volume do projeto | ~200 contatos/dia | > 1M predições/dia |
| Complexidade | Baixa (REST) | Alta (Airflow/Spark) |
| Freshness | Imediata | Defasagem horas/dias |

**Por que Real-Time?** Agentes de atendimento precisam do risco do cliente *durante a ligação*. O endpoint `/predict_batch` cobre campanhas noturnas sem infra adicional. Detalhes em [docs/TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md).

---

## Docker

```bash
docker-compose up -d
# API: http://localhost:8000/docs
# MLflow: http://localhost:5000
```

---

## Vídeo de Apresentação (STAR)

[![Vídeo STAR — Telco Churn Prediction](https://img.shields.io/badge/YouTube-Apresentação%20STAR-red?logo=youtube)](https://www.youtube.com/watch?v=cD3LQXYjdH0)

Apresentação de 5 minutos no método STAR (Situation, Task, Action, Result) demonstrando o projeto completo.

---

## Documentação

| Documento | Descrição |
|-----------|-----------|
| [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) | Visão geral ponta-a-ponta |
| [docs/ML_CANVAS.pdf](docs/ML_CANVAS.pdf) | Canvas visual do projeto |
| [docs/MODEL_CARD.pdf](docs/MODEL_CARD.pdf) | Limitações, biases, fairness |
| [docs/MONITORING.md](docs/MONITORING.md) | Plano de monitoramento em produção |
| [docs/TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md) | Guia técnico de implementação |

---

## Referências

- [PyTorch](https://pytorch.org)
- [FastAPI](https://fastapi.tiangolo.com)
- [MLflow](https://python -m mlflow.org)
- [Scikit-Learn](https://scikit-learn.org)
- [Pandera](https://pandera.readthedocs.io)

---

Projeto educacional - Telco Churn Prediction (2026)
