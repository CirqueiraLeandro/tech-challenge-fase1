# 📊 PROJECT OVERVIEW — TELCO CHURN PREDICTION

**Projeto**: Previsão de Churn em Telecomunicações — Pipeline ML End-to-End  
**Status**: ✅ Todas as etapas concluídas — Pronto para Produção  
**Disciplinas atendidas**: Fundamentos, APIs, Bibliotecas, Engenharia de Software, Ciclo de Vida

---

## 1️⃣ CONTEXTO E PROBLEMA DE NEGÓCIO

### 1.1 Situação

Uma **operadora de telecomunicações** enfrenta taxa de cancelamento acelerada, causando perda significativa de receita. A diretoria demanda uma solução que:

- **Identifique clientes em risco** antes que cancelem
- **Quantifique probabilidades** de churn com confiança
- **Priorize ações de retenção** baseadas em scores preditivos

### 1.2 Contexto Técnico

```
Dataset: Telco_customer_churn.xlsx
├─ 7.043 clientes
├─ 19 features (pós remoção de data leakage)
│  ├─ Numéricas (3): tenure, monthly_charges, total_charges
│  └─ Categóricas (16): gender, contract, internet_service...
├─ 26.54% churn rate (1,869 churners / 5,174 retidos)
└─ Split: 80/20 estratificado (5.634 treino / 1.409 teste)

Problema: Classificação Binária (Churn: Sim/Não)
Métrica Primária: AUC-ROC
Métrica de Negócio: Custo de churn evitado
```

### 1.3 Impacto de Negócio

| Indicador | Valor |
|---|---|
| Taxa de churn | 26.54% |
| Perda estimada anual | R$ 5 milhões |
| Custo por cliente perdido | R$ 500 |
| Custo por falso positivo | R$ 100 |
| Benefício por churn evitado | R$ 300 |
| Objetivo | Reduzir churn em 15% em 6 meses |

---

## 2️⃣ ETAPA 1: EXPLORAÇÃO E BASELINES

### 2.1 Análise Exploratória (EDA)

#### Limpeza de Dados

```
✅ Total Charges: convertido de object → float64
✅ Missing Values: 11 NaN em TotalCharges → imputados com 0
✅ Duplicatas: 0 encontradas
✅ Data Leakage: 11 colunas removidas (ver seção 2.2)
```

#### Features Utilizadas

```
Numéricas (3):
  tenure, monthly_charges, total_charges

Categóricas (16):
  gender, senior_citizen, partner, dependents,
  phone_service, multiple_lines, internet_service,
  online_security, online_backup, device_protection,
  tech_support, streaming_tv, streaming_movies,
  contract, paperless_billing, payment_method
```

#### Insights de Negócio

```
1. Contract Type   → Mês-a-mês: 42% churn vs 3% (2 anos)
2. Tenure          → < 3 meses: 50% churn; > 24 meses: 5%
3. Internet Service → Fiber Optic: 42% churn vs 7% sem internet
4. Tech Support    → Sem suporte: 40% churn vs 15% com suporte
5. Monthly Charges → Alta correlação com tipo de contrato
```

### 2.2 Data Leakage — Correção Crítica

Versões iniciais incluíam colunas derivadas do target, causando AUC-ROC artificial de ~1.0. Após identificação e remoção:

**Colunas removidas**: `CustomerID`, `Churn Score`, `Churn Reason`, `CLTV`, `Zip Code`, `Latitude`, `Longitude`, `City`, `Lat Long`, `Count`, `Churn Label`

**Resultado**: métricas caíram de 100% → 80% accuracy — valores realistas e utilizáveis em produção.

### 2.3 Modelos Baseline

#### DummyClassifier (referência mínima)
```
Estratégia: sempre prediz classe majoritária (No Churn)
Accuracy: 73.46% | F1: 0.000 | AUC-ROC: 0.500
Função: estabelecer piso mínimo de desempenho
```

#### LogisticRegression (melhor modelo — produção)
```
Config: max_iter=1000, L2, class_weight='balanced'
Accuracy: 80.34% | Precision: 0.643 | Recall: 0.583
F1-Score: 0.611 | AUC-ROC: 0.848 | PR-AUC: 0.644
```

#### RandomForest
```
Config: n_estimators=100, max_depth=15, class_weight='balanced'
Accuracy: 79.35% | F1: 0.565 | AUC-ROC: 0.834 | PR-AUC: 0.627
```

### 2.4 Validação Cruzada

```
5-Fold StratifiedKFold (preserva proporção de classes por fold)
Resultado: modelos estáveis entre folds
```

### 2.5 Data Readiness Checklist

```
✅ Missing values tratados (0 após limpeza)
✅ Tipos de dados corrigidos (Total Charges object → float)
✅ Data leakage identificado e removido
✅ Variável alvo identificada (Churn Value)
✅ Desbalanceamento mitigado (class_weight='balanced')
✅ Reprodutibilidade garantida (seed=42)
✅ Schema Pandera validado (12 verificações automáticas)
```

---

## 3️⃣ ETAPA 2: REDE NEURAL MLP (PYTORCH)

### 3.1 Arquitetura

```
MLPChurn(nn.Module)
  Input:    19 features (3 numéricas + 16 categóricas encoded)
  Hidden 1: Dense(19→128) + BatchNorm + ReLU + Dropout(0.3)
  Hidden 2: Dense(128→64) + BatchNorm + ReLU + Dropout(0.3)
  Hidden 3: Dense(64→32)  + BatchNorm + ReLU + Dropout(0.3)
  Output:   Dense(32→2)   + Softmax → P(no churn), P(churn)

  Total de parâmetros: 13.410
  Dispositivo: CPU
  Arquivo: models/mlp_etapa2.pt
```

### 3.2 Configuração de Treino

```
Optimizer:    Adam (lr=0.001, weight_decay=1e-4)
Loss:         CrossEntropyLoss (class_weight balanceado)
Scheduler:    ReduceLROnPlateau (patience=5, factor=0.5)
Early Stop:   patience=10 épocas
Batch Size:   64
Epochs Máx:   100
```

### 3.3 Comparativo Final de Modelos

| Modelo | Accuracy | F1 | AUC-ROC | PR-AUC |
|---|---|---|---|---|
| DummyClassifier | 73.46% | 0.000 | 0.500 | 0.265 |
| **LogisticRegression** | **80.34%** | **0.611** | **0.848** | **0.644** |
| RandomForest | 79.20% | 0.571 | 0.839 | 0.627 |
| MLP PyTorch | 79.22% | 0.583 | 0.833 | 0.592 |

> LogisticRegression selecionada para produção — melhor AUC-ROC e Recall com menor complexidade.

---

## 4️⃣ ETAPA 3: API FASTAPI

### 4.1 Endpoints

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/health` | Status da API e modelo carregado |
| GET | `/info` | Arquitetura e parâmetros do modelo |
| GET | `/metrics` | Estatísticas de uso em runtime |
| POST | `/predict` | Predição individual com probabilidades |
| POST | `/predict_batch` | Predição em lote (até 10.000 registros) |

### 4.2 Exemplo de Uso

```json
POST /predict
{
  "tenure": 2,
  "monthly_charges": 89.5,
  "total_charges": 179.0,
  "gender": "Female",
  "internet_service": "Fiber optic",
  "contract": "Month-to-month"
}

→ Response 200
{
  "prediction": 1,
  "probability_churn": 0.713,
  "probability_no_churn": 0.287,
  "confidence": 0.713
}
```

### 4.3 Funcionalidades da API

```
✅ Validação Pydantic v2 em todas as entradas
✅ LatencyMiddleware — header X-Process-Time-Ms em toda resposta
✅ Logging estruturado: METHOD /path → STATUS (Xms)
✅ Health check com status do modelo carregado
✅ Predição em lote sem infra adicional
```

### 4.4 Decisão Arquitetural: Real-Time vs Batch

**Escolha**: Real-Time API. Agentes de callcenter precisam do score *durante a ligação* — latência de horas inviabiliza o caso de uso. O endpoint `/predict_batch` cobre campanhas noturnas sem necessidade de Airflow/Spark.

| Critério | Real-Time ✅ | Batch |
|---|---|---|
| Latência | < 100ms | Horas |
| Complexidade | Baixa — REST | Alta — orquestração |
| Volume do projeto | ~200 contatos/dia | > 1M predições/dia |

---

## 5️⃣ TESTES AUTOMATIZADOS

### 5.1 Suites de Testes

| Suite | Arquivo | Testes | Cobertura |
|---|---|---|---|
| Config + Logging | `test_config_logging.py` | 34 | 100% |
| DataLoader + Preprocessor | `test_data_expanded.py` | 22 | 96% |
| Schema Pandera | `test_schema_pandera.py` | 12 | 100% |
| MLP Architecture + Training | `test_mlp_etapa2.py` | 21 | 100% |
| Baselines | `test_models_expanded.py` | 15 | 94% |
| FastAPI Endpoints | `test_api_etapa3.py` | 18 | 80% |
| MLflow Tracking | `test_mlflow.py` | 11 | 89% |
| Integration E2E | `test_integration_e2e.py` | ~10 | — |
| **Total** | | **135+** | **57% global** |

### 5.2 Marcadores (markers)

| Marker | O que executa |
|---|---|
| `smoke` | Testes críticos rápidos (< 5s) |
| `schema` | Validação Pandera (tipos, ranges, leakage) |
| `integration` | Pipeline completo E2E |
| `unit` | Testes unitários isolados |

### 5.3 Schema Pandera — 12 Verificações Automáticas

```
✅ Ranges numéricos (tenure ≥ 0, charges ≥ 0)
✅ Target binário {0, 1}
✅ Churn rate entre 10–50%
✅ Zero valores ausentes
✅ Zero duplicatas
✅ Ausência de colunas de leakage
✅ Valores categóricos dentro do domínio esperado
✅ Mínimo de 5.000 linhas
✅ Mínimo de 15 features
```

### 5.4 Linting

```
ruff check src/ tests/  →  All checks passed! (zero erros)
```

---

## 6️⃣ ARQUITETURA DE ARQUIVOS

### 6.1 Módulos Python (src/)

| Arquivo | Descrição | Etapa |
|---|---|---|
| `config.py` | Configurações centralizadas (paths, seed, env) | Todas |
| `logger.py` | Logging estruturado (arquivo + console, sem print) | Todas |
| `constants.py` | Constantes de features e target | Todas |
| `data.py` | DataLoader + DataPreprocessor reutilizável | 1, 2, 3 |
| `baselines.py` | BaselineTrainer (Dummy, LogReg, RF) | 1 |
| `mlflow_utils.py` | MLflowTracker para experimentos | 1, 2 |
| `etapa1_eda_baselines.py` | Script completo Etapa 1 | 1 |
| `train_mlp.py` | Script Etapa 2 — treino MLP | 2 |
| `models/mlp.py` | Arquitetura MLP (13.410 params) | 2 |
| `models/training.py` | MLPTrainer: fit, validate, early stop | 2 |
| `api/app.py` | FastAPI v2.0.0 — 5 endpoints | 3 |
| `api/schemas.py` | Pydantic v2 schemas (input/output) | 3 |
| `preprocessing/pipeline.py` | ColumnTransformer sklearn | 1, 2, 3 |
| `monitoring/drift.py` | PSI-based drift detection | 3 |

### 6.2 Testes (tests/)

| Arquivo | Tipo | Testes |
|---|---|---|
| `conftest.py` | Setup | Fixtures globais |
| `test_api_etapa3.py` | E2E | 18 |
| `test_config_logging.py` | Unit | 34 |
| `test_data_expanded.py` | Unit | 22 |
| `test_schema_pandera.py` | Schema | 12 |
| `test_mlp_etapa2.py` | Unit | 21 |
| `test_models_expanded.py` | Unit | 15 |
| `test_mlflow.py` | Unit | 11 |
| `test_integration_e2e.py` | E2E | ~10 |

### 6.3 Configuração (raiz)

| Arquivo | Propósito |
|---|---|
| `pyproject.toml` | Single source of truth (deps, linting, pytest) |
| `Makefile` | Automação de tasks de desenvolvimento |
| `Dockerfile` | Build da imagem da API |
| `docker-compose.yml` | API + MLflow UI |
| `.gitignore` | Exclui dados, modelos, caches, logs |

### 6.4 Documentação (docs/)

| Arquivo | Descrição |
|---|---|
| `README.md` | Setup, execução, arquitetura |
| `TECHNICAL_GUIDE.md` | Guia técnico detalhado por módulo |
| `PROJECT_OVERVIEW.md` | Este documento |
| `MODEL_CARD.pdf` | Métricas, limitações, fairness, deploy |
| `MONITORING.md` | Plano de monitoramento em produção |
| `ML_CANVAS.pdf` | Canvas visual do projeto |

### 6.5 Dados e Modelos

| Diretório / Arquivo | Conteúdo | Git |
|---|---|---|
| `data/raw/Telco_customer_churn.xlsx` | Dataset original (7.043 clientes) | Rastreado |
| `models/dummyclassifier_baseline.pkl` | DummyClassifier treinado | Rastreado |
| `models/logisticregression_baseline.pkl` | LogReg treinado (produção) | Rastreado |
| `models/randomforest_baseline.pkl` | RandomForest treinado | Rastreado |
| `models/scaler.pkl` | StandardScaler fitted | Rastreado |
| `models/mlp_etapa2.pt` | MLP PyTorch (13.410 params) | Rastreado |
| `models/etapa1_metadata.json` | Metadata do dataset e resultados | Rastreado |
| `models/baseline_comparison.csv` | Comparativo métricas dos baselines | Rastreado |
| `models/mlp_comparison.csv` | Comparativo MLP vs baselines | Rastreado |
| `mlruns/` | Experimentos MLflow (83+ runs) | Ignorado |

---

## 7️⃣ SISTEMA DE LOGGING

Todos os módulos usam `src/logger.py` — logging estruturado com rotação de arquivo:

```python
from src.logger import setup_logger
logger = setup_logger(__name__, config)

logger.info("Dataset carregado: 7.043 linhas")
logger.warning("PSI > 0.2 em feature contract — avaliar retraining")
logger.error("Modelo não encontrado em models/mlp_etapa2.pt")
```

| Feature | Detalhe |
|---|---|
| Saídas | Console + `logs/application.log` |
| Rotação | 10 MB por arquivo, até 5 backups |
| Ambientes | Development (DEBUG) / Production (INFO) |
| Formato | Timestamp ISO 8601 + módulo + nível + mensagem |

---

## 8️⃣ COMO EXECUTAR O PROJETO

### Setup

```bash
git clone <repo>
cd Telecommunications_Industry
pip install -e .
pip install -e ".[dev]"
```

### Etapa 1 — EDA + Baselines

```bash
python -m src.etapa1_eda_baselines
# Output: models/*.pkl + etapa1_metadata.json
```

### Etapa 2 — MLP PyTorch

```bash
python -m src.train_mlp
# Output: models/mlp_etapa2.pt
```

### Etapa 3 — API

```bash
python -m uvicorn src.api.app:app --reload --port 8000
# Swagger: http://localhost:8000/docs
```

### Testes

```bash
python -m pytest tests/ -v --cov=src        # Suite completa
python -m pytest tests/ -m smoke            # Apenas críticos
python -m pytest tests/ -m schema          # Schema Pandera
```

### MLflow UI

```bash
python -m mlflow ui --host 0.0.0.0 --port 5000
# Acesso: http://localhost:5000
```

---

## 9️⃣ RESULTADOS FINAIS

### Métricas de Performance

| Modelo | Accuracy | F1 | AUC-ROC | PR-AUC | Status |
|---|---|---|---|---|---|
| DummyClassifier | 73.46% | 0.000 | 0.500 | 0.265 | Baseline |
| **LogisticRegression** | **80.34%** | **0.611** | **0.848** | **0.644** | **Produção** |
| RandomForest | 79.20% | 0.571 | 0.839 | 0.627 | Alternativa |
| MLP PyTorch | 79.22% | 0.583 | 0.833 | 0.592 | API (servida) |

### Entregáveis Completos

| Entregável | Status | Detalhe |
|---|---|---|
| EDA + ML Canvas + Baselines | ✅ | `etapa1_eda_baselines.py` |
| MLP PyTorch + MLflow | ✅ | `train_mlp.py`, 83+ runs |
| API FastAPI + 5 endpoints | ✅ | `src/api/app.py` |
| 137 testes automatizados | ✅ | 8 suites, ruff zero erros |
| Model Card (HTML) | ✅ | Métricas, fairness, deploy |
| Documentação completa | ✅ | 7 documentos em `docs/` |
| Commits limpos | ✅ | 8 commits temáticos (6 em prd + 2 em dev) no GitHub |
| Arquitetura deploy documentada | ✅ | Real-Time vs Batch com justificativa |

### Lições Aprendidas

```
✅ LogisticRegression superou MLP em dados tabulares com ~7k amostras
✅ Pandera + testes automáticos previnem erros silenciosos de schema
✅ MLflow essencial para rastrear e comparar experimentos

⚠️ PRINCIPAL LIÇÃO — DATA LEAKAGE:
   Churn Score, Churn Reason e CLTV são calculados APÓS o cancelamento.
   Incluí-los causou AUC-ROC artificial de ~1.0. A correção levou as
   métricas para valores realistas (AUC-ROC 0.848), mas apenas estes
   são válidos para uso em produção.
   Regra: auditar cada feature contra o timeline do evento-alvo.
```

### Mapeamento de Requisitos

| Requisito | Implementação | Status |
|---|---|---|
| Estrutura organizada src/, data/, tests/, docs/ | Completa | ✅ |
| README.md com setup e execução | README.md | ✅ |
| pyproject.toml como single source of truth | pyproject.toml | ✅ |
| Commits limpos e significativos | 8 commits temáticos (6 em prd + 2 em dev) | ✅ |
| .gitignore adequado para ML | .gitignore | ✅ |
| MLP com PyTorch | src/models/mlp.py | ✅ |
| Baselines Scikit-Learn | src/baselines.py | ✅ |
| MLflow tracking | src/mlflow_utils.py | ✅ |
| API FastAPI | src/api/app.py | ✅ |
| Testes (≥ 3 tipos) | Unit, Schema, E2E, Smoke | ✅ |
| Logging estruturado | src/logger.py | ✅ |
| Linting ruff | Zero erros | ✅ |
| Model Card | docs/MODEL_CARD.pdf | ✅ |
| Deploy arquitetura documentada | TECHNICAL_GUIDE.md §8 | ✅ |
| Vídeo STAR 5 min | [YouTube — Apresentação STAR](https://www.youtube.com/watch?v=cD3LQXYjdH0) | ✅ |

---

## 📚 DISCIPLINAS ATENDIDAS

| Disciplina | Implementação |
|---|---|
| **Fundamentos** | Baselines, métricas, PyTorch, trade-offs bias-variance |
| **APIs** | FastAPI, Pydantic v2, middleware, error handling |
| **Bibliotecas** | Scikit-Learn, PyTorch, MLflow, Pandas, Pandera |
| **Eng. Software** | Modularização SOLID, testes, linting, config centralizada |
| **Ciclo de Vida** | ML Canvas, EDA, Model Card, deploy, monitoramento PSI |
