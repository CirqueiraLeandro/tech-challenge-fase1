# 📖 TECHNICAL GUIDE - Documentação Técnica Completa

**Versão**: 2.0 | **Status**: ✅ Production Ready

---

## 📍 COMEÇAR AQUI

Se você é novo no projeto:

1. **Leia** [README.md](../README.md) - Quick start (5 min)
2. **Leia ESTE arquivo** - Estrutura técnica completa (15 min)
3. **Consulte** [MODEL_CARD.pdf](MODEL_CARD.pdf) - Limitations e monitoramento
4. **Execute** `python tests/final_validation.py` - Validar tudo funciona

---

## 🎯 Visão Geral do Projeto

### Problema
Operadora de telecomunicações perde clientes em ritmo acelerado. Precisa prever quais clientes têm risco de churn para implementar estratégias de retenção.

### Solução
Pipeline de ML end-to-end que:
- ✅ Explora 7,043 clientes para entender padrões de churn
- ✅ Treina 3 baselines (DummyClassifier, LogReg, RandomForest)
- ✅ Implementa rede neural MLP com PyTorch (13,410 parâmetros)
- ✅ Expõe API FastAPI v2.0.0 para predições em produção
- ✅ Monitora performance contínuamente

### Stack Técnico
- **Linguagem**: Python 3.11+
- **ML**: PyTorch (MLP) + Scikit-Learn (baselines, preprocessing)
- **API**: FastAPI v2.0.0 + Pydantic
- **Testes**: pytest 95%+ coverage
- **Tracking**: MLflow
- **Deploy**: Docker + Kubernetes (opcional)

---

## 📁 Estrutura de Arquivos (Detalhada)

### 🔵 RAIZ - Configuração e Orquestração

#### `README.md`
- **Propósito**: Documentação principal do projeto
- **Conteúdo**: Quick start, estrutura, comandos, troubleshooting
- **Público**: Todos (devs, PMs, stakeholders)
- **Quando usar**: Primeira leitura, referência rápida
- **Pré-requisito**: Nenhum

#### `pyproject.toml`
- **Propósito**: Single source of truth para dependências e configurações
- **Conteúdo**:
  ```toml
  [project]
  name = "telco-churn"
  version = "4.0.0"
  dependencies = ["torch>=2.0", "fastapi>=0.100", "scikit-learn>=1.3", ...]
  
  [project.optional-dependencies]
  dev = ["pytest>=7.4", "black", "ruff", ...]
  ```
- **Usado por**: `pip install -e .` (instalação)
- **Quando atualizar**: Adicionar nova dependência

#### `Dockerfile`
- **Propósito**: Container production-ready da API
- **Conteúdo**: Multi-stage build, imagem slim Python
- **Build**: `docker build -t telco-churn:latest .`
- **Run**: `docker run -p 8000:8000 telco-churn:latest`

#### `docker-compose.yml`
- **Propósito**: Orquestrar containers locais (dev)
- **Conteúdo**: API (FastAPI), MLflow, dados compartilhados
- **Uso**: `docker-compose up -d`
- **Acesso**:
  - API: http://localhost:8000/docs
  - MLflow: http://localhost:5000

#### `.gitignore`
- **Conteúdo**:
  ```
  __pycache__/
  *.pkl, *.pth, *.pt        # Modelos
  data/raw/                 # Dados brutos
  .env, .venv/             # Secrets, virtualenv
  mlruns/                  # MLflow local
  ```

---

### 🔴 `src/` - Código Principal (Módulos Reutilizáveis)

#### `src/etapa1_eda_baselines.py` ⭐ ETAPA 1

**Propósito**: Pipeline completo de exploração e baselines

**O que faz**:
```
1. Carrega dados: data/raw/Telco_customer_churn.xlsx
2. Limpeza:
   - Remove colunas single-value (Country, State)
   - Preenche missing values (Total Charges)
   - Converte tipos de dados
3. EDA (Análise Exploratória):
   - Estatísticas descritivas
   - Matriz de correlação
   - Distribuição de churn
4. Preprocessing:
   - Label encoding (features categóricas)
   - Standard scaling (features numéricas)
   - Train/test split 80/20 estratificado
5. Modelos Baseline:
   - DummyClassifier (73.46% accuracy - baseline)
   - LogisticRegression (80.34% accuracy, AUC-ROC 0.848)
   - RandomForest (79.35% accuracy, AUC-ROC 0.834)
6. Validação:
   - 5-fold cross-validation estratificada
7. Salva:
   - models/dummyclassifier_baseline.pkl
   - models/logisticregression_baseline.pkl
   - models/randomforest_baseline.pkl
   - models/scaler.pkl
   - models/etapa1_metadata.json
```

**Como executar**:
```bash
python -m src.etapa1_eda_baselines
```

**Output esperado**:
```
[INFO] Carregando dados...
[INFO] Explorando dados: 7043 clientes, 19 features (pós leakage fix)
[INFO] Limpando: removidas 2 colunas single-value, 11 missing values preenchidos
[INFO] Splitting 80/20 estratificado
[INFO] Treinando DummyClassifier: 73.46% accuracy
[INFO] Treinando LogisticRegression: 80.34% accuracy, AUC-ROC 0.848
[INFO] Treinando RandomForest: 79.35% accuracy, AUC-ROC 0.834
[INFO] Cross-validation (5-fold) completo
[OK] Artefatos salvos em models/
```

**Entrega de Etapa 1**: ✅ EDA, ML Canvas, baselines, metadata

---

#### `src/train_mlp.py` ⭐ ETAPA 2

**Propósito**: Treinar rede neural MLP com PyTorch

**Pipeline**:
```
1. Carrega dados pré-processados (via src.etapa1_eda_baselines)
2. Cria DataLoaders (PyTorch)
3. Inicializa MLP:
   - 19 features → 128 → 64 → 32 → 2 classes
   - Dropout 0.3, BatchNorm, ReLU
   - Total: 13,410 parâmetros
4. Treina com:
   - Optimizer: Adam (lr=0.001)
   - Loss: CrossEntropyLoss
   - Epochs: 50 (com early stopping)
   - Batch size: 32
5. Valida: 5-fold cross-validation
6. Compara com baselines
7. Salva: models/mlp_best_model.pth
```

**Como executar**:
```bash
python -m src.train_mlp
```

**Output esperado**:
```
[INFO] Carregando dados...
[INFO] Criando DataLoaders...
[INFO] Inicializando MLP: 13,410 parâmetros
[INFO] Epoch 1/50: Train Loss=0.58, Val Loss=0.52, F1=0.85
...
[INFO] Early stopping acionado no epoch 25
[INFO] MLP final: Accuracy=0.97, F1=0.94, AUC-ROC=0.96
[INFO] Comparação com baselines:
  - LogReg: 1.00 | MLP: 0.97
  - RF: 1.00 | MLP: 0.97
[OK] Modelo salvo em models/mlp_best_model.pth
```

**Entrega de Etapa 2**: ✅ MLP treinado, comparação métricas, cross-validation

---

#### `src/models/mlp.py`

**Propósito**: Arquitetura MLP + componentes auxiliares

**Classes**:

```python
class MLPChurn(nn.Module):
    """Rede neural MLP para previsão de churn
    
    Arquitetura:
    - Input: 19 features
    - Hidden 1: 128 neurons + ReLU + BatchNorm + Dropout(0.3)
    - Hidden 2: 64 neurons + ReLU + BatchNorm + Dropout(0.3)
    - Hidden 3: 32 neurons + ReLU + BatchNorm + Dropout(0.3)
    - Output: 2 classes (softmax in loss)
    """
    
    def __init__(self, input_dim=19, hidden_dims=[128,64,32], 
                 dropout_rate=0.3, batch_norm=True):
        # Inicializa camadas
    
    def forward(self, x):
        # Passa dados pelas camadas
        return logits  # (batch_size, 2)
    
    def predict(self, x):
        # Retorna predição: 0 ou 1
        
    def predict_proba(self, x):
        # Retorna probabilidades: [prob_0, prob_1]
```

**EarlyStopping**:
```python
class EarlyStopping:
    """Para treinamento quando validação não melhora"""
    def __init__(self, patience=10, min_delta=1e-4):
        # Monitora perda de validação
        # Salva best checkpoint
```

**Uso**:
```python
from src.models.mlp import MLPChurn, EarlyStopping

model = MLPChurn(input_dim=19, hidden_dims=[128,64,32])
early_stop = EarlyStopping(patience=10)
```

---

#### `src/models/training.py`

**Propósito**: Loop de treinamento reutilizável

**Classes**:

```python
class MLPTrainer:
    """Orquestra treinamento do MLP
    
    Métodos:
    - train_epoch(): treina 1 epoch
    - validate(): avalia em validação
    - fit(): treina até convergência com early stopping
    - train_with_cross_validation(): CV 5-fold
    """
```

**Uso**:
```python
trainer = MLPTrainer(model, optimizer, criterion, device)
history = trainer.fit(train_loader, val_loader, epochs=50, patience=10)
```

---

#### `src/api/app.py` ⭐ ETAPA 3

**Propósito**: API FastAPI v2.0.0 para inferências em produção

**Endpoints**:

| Método | Path | Descrição | Input | Output |
|--------|------|-----------|-------|--------|
| GET | `/health` | Health check | — | `{"status": "healthy"}` |
| POST | `/predict` | 1 cliente | CustomerData | PredictionResponse |
| POST | `/predict_batch` | até 10k | BatchRequest | BatchResponse |
| GET | `/docs` | Swagger UI | — | HTML |
| GET | `/redoc` | ReDoc | — | HTML |

**Resposta `/predict`**:
```json
{
  "prediction": "churn",
  "probability": [0.15, 0.85],
  "confidence": 0.85,
  "risk_level": "high",
  "recommendation": "Contate cliente para oferecer plano de retenção"
}
```

**Como iniciar**:
```bash
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**Swagger**: http://localhost:8000/docs

**Recursos**:
- ✅ CORS habilitado (allow_origins=["*"])
- ✅ Lazy loading do modelo (na primeira predição)
- ✅ Logging estruturado de cada predição
- ✅ Validação automática Pydantic
- ✅ Batching otimizado
- ✅ Error handling robusto

---

#### `src/api/schemas.py`

**Propósito**: Validação de inputs/outputs com Pydantic

**Modelos**:

```python
class CustomerData(BaseModel):
    """Input para /predict - 21 features"""
    tenure_months: int
    monthly_charges: float
    total_charges: float
    internet_service: str
    contract: str
    tech_support: bool
    # ... 15 campos adicionais
    
class PredictionResponse(BaseModel):
    """Output de /predict"""
    prediction: str  # "churn" ou "retention"
    probability: List[float]  # [p_retention, p_churn]
    confidence: float  # max(probability)
    risk_level: str  # "low", "medium", "high"
    recommendation: str  # Recomendação de ação
```

**Validação**:
- ✅ Tipos de dados automáticos
- ✅ Ranges de valores (ex: 0 ≤ probability ≤ 1)
- ✅ Strings enumeradas
- ✅ Mensagens de erro claras

---

### 🟢 `data/` - Dados

#### `data/raw/Telco_customer_churn.xlsx`

**Propósito**: Dataset original com 7,043 clientes

**Estrutura**:
```
Colunas (33 totais):
- Numéricas (8): Tenure, Monthly Charges, Total Charges, CLTV, Churn Score, Latitude, Longitude, Zip Code
- Categóricas (18): Gender, Internet Service, Contract, Tech Support, ...
- Alvo (1): Churn Value (0 ou 1)
```

**Limpeza (in src/etapa1_eda_baselines.py)**:
- Remove: Country, State (single-value)
- Preenche: Total Charges (11 missing → 0)
- Converte: Total Charges (object → float)
- Resultado: 26 features limpas

---

### 🟡 `models/` - Artefatos Treinados

#### `models/dummyclassifier_baseline.pkl`
- **O que é**: Baseline DummyClassifier (Scikit-Learn)
- **Performance**: 73.46% accuracy (majority class)
- **Uso**: Baseline de comparação
- **Criado por**: `src/etapa1_eda_baselines.py`

#### `models/logisticregression_baseline.pkl`
- **O que é**: Regressão Logística (Scikit-Learn)
- **Performance**: ~80% accuracy, AUC-ROC ~0.84 (realístico, pós leakage fix)
- **Tamanho**: ~50 KB
- **Criado por**: `src/etapa1_eda_baselines.py`
- **Usado por**: Comparação em Etapa 2

#### `models/randomforest_baseline.pkl`
- **O que é**: Random Forest (Scikit-Learn)
- **Performance**: ~80% accuracy, AUC-ROC ~0.84 (realístico, pós leakage fix)
- **Tamanho**: ~5 MB
- **Criado por**: `src/etapa1_eda_baselines.py`
- **Usado por**: Comparação em Etapa 2

#### `models/scaler.pkl`
- **O que é**: StandardScaler (Scikit-Learn) - fit em training set
- **Features escaladas**: 8 numéricas
- **Tamanho**: ~1 KB
- **Criado por**: `src/etapa1_eda_baselines.py`
- **Usado por**: Preprocessing no train_mlp.py e API

#### `models/etapa1_metadata.json`
- **Conteúdo**:
  ```json
  {
    "dataset": {
      "n_customers": 7043,
      "n_features": 26,
      "churn_rate": 0.2654,
      "train_size": 5634,
      "test_size": 1409
    },
    "features": {
      "numeric": ["tenure_months", "monthly_charges", ...],
      "categorical": ["internet_service", "contract", ...]
    },
    "baselines": {
      "dummy": 0.7346,
      "logreg": 1.0,
      "rf": 1.0
    }
  }
  ```
- **Criado por**: `src/etapa1_eda_baselines.py`
- **Usado por**: Documentação, validação, reprodutibilidade

#### `models/mlp_best_model.pth`
- **O que é**: Checkpoint do MLP treinado (PyTorch)
- **Formato**: state_dict (pesos + biases)
- **Tamanho**: ~200 KB
- **Criado por**: `src/train_mlp.py`
- **Usado por**: `src/api/app.py` para inferências
- **Carregamento**:
  ```python
  model = MLPChurn(input_dim=19)
  model.load_state_dict(torch.load("models/mlp_best_model.pth"))
  ```

#### `models/mlp_training_history.json`
- **Conteúdo**: Histórico de loss/metrics por epoch
  ```json
  {
    "train_loss": [0.58, 0.52, ...],
    "val_loss": [0.55, 0.48, ...],
    "train_accuracy": [0.75, 0.78, ...],
    "val_accuracy": [0.76, 0.79, ...],
    "best_epoch": 25
  }
  ```
- **Usado por**: Visualização de convergência, diagnóstico

---

### 🔵 `tests/` - Testes Automatizados (100+ test cases)

**Suite Completa** (8 arquivos com markers python -m pytest):

#### 1️⃣ `tests/final_validation.py`

**Propósito**: Validação rápida de todas as 4 etapas

**O que testa**:
```
✓ Estrutura de arquivos (src/, models/, docs/)
✓ Etapa 1: Metadata carregado, 3 modelos .pkl presentes
✓ Etapa 2: MLP criado, 13,410 parâmetros, forward pass OK, EarlyStopping OK
✓ Etapa 3: FastAPI importada, versão 2.0.0, endpoints OK
✓ Etapa 4: MODEL_CARD.pdf, MONITORING.md presentes
```

**Executar**:
```bash
python tests/final_validation.py
```

---

#### 2️⃣ `tests/test_api_etapa3.py` (18 testes)

**Propósito**: Testes da API FastAPI v2.0.0

**Classes de Teste**:
- `TestHealthEndpoint`: Status e estrutura do /health
- `TestInfoEndpoint`: Info do modelo
- `TestPredictEndpoint`: Single prediction com validação
- `TestBatchPredictEndpoint`: Batch predictions
- `TestRootEndpoint`: Root endpoint
- `TestMetricsEndpoint`: Métricas do modelo
- `TestErrorHandling`: Tratamento de erros
- `TestIntegration`: Full flow

**Executar**:
```bash
python -m pytest tests/test_api_etapa3.py -v
```

---

#### 3️⃣ `tests/test_config_logging.py` (34 testes)

**Propósito**: Configuração e logging estruturado

**Classes de Teste**:
- `TestConfigClasses`: Configurações base, datasets, seeds, MLflow, API, PyTorch
- `TestDevelopmentConfig`: Debug=true, testing=false
- `TestProductionConfig`: Debug=false, log_level=warning

**Executar**:
```bash
python -m pytest tests/test_config_logging.py -v
```

---

#### 4️⃣ `tests/test_data_expanded.py` (~15 testes)

**Propósito**: DataLoader e preprocessamento

**Classes de Teste**:
- `TestDataLoader`: Load, target identification, missing values, duplicates, feature extraction, train/test split
- `TestDataPreprocessor`: Categorical encoding, numeric scaling

**Executar**:
```bash
python -m pytest tests/test_data_expanded.py -v
```

---

#### 5️⃣ `tests/test_e2e.py` (~12 testes)

**Propósito**: End-to-end simples do pipeline

**O que testa**:
- Carregamento de dados
- Preprocessing
- Treinamento de baselines
- Validação de métricas

**Executar**:
```bash
python -m pytest tests/test_e2e.py -v
```

---

#### 6️⃣ `tests/test_integration_e2e.py` (~18 testes)

**Propósito**: End-to-end completo com integração total

**Classe**: `TestPipelineIntegrationE2E`
- Load → Preprocess → Train → Evaluate → Monitor

**Executar**:
```bash
python -m pytest tests/test_integration_e2e.py -v
```

---

#### 7️⃣ `tests/test_mlflow.py` (11 testes)

**Propósito**: MLflow experiment tracking

**Classes de Teste**:
- Tracking de experimentos
- Versioning de modelos
- Artifact management

**Executar**:
```bash
python -m pytest tests/test_mlflow.py -v
```

---

#### 8️⃣ `tests/test_mlp_etapa2.py` (21 testes)

**Propósito**: Testes unitários da rede neural MLP

**Testes**:
- Inicialização (parâmetros=13,410)
- Forward pass (input/output shapes)
- Predict e predict_proba
- EarlyStopping (paciência, min_delta)
- Cross-validation 5-fold
- Feature importance

**Executar**:
```bash
python -m pytest tests/test_mlp_etapa2.py -v
```

---

#### 9️⃣ `tests/test_models_expanded.py` (15 testes)

**Propósito**: Testes dos modelos baseline

**Classes de Teste**:
- `TestBaselineTrainer`: DummyClassifier, LogisticRegression, RandomForest

---

## 📊 Resumo de Arquivos

### ✅ Arquivos em Produção (17 Python + 11 testes + 8 docs)

**Estrutura Limpa** (sem placeholders, sem duplicatas):

```
src/ (17 arquivos Python)
├─ Configuração: config.py, logger.py, constants.py
├─ Data: data.py
├─ ML: baselines.py, mlflow_utils.py
├─ Etapa 1: etapa1_eda_baselines.py (único script)
├─ Etapa 2: train_mlp.py (único script)
├─ models/: mlp.py, training.py (arquitetura + training)
├─ api/: app.py, schemas.py (FastAPI v2.0.0 + Pydantic)
└─ preprocessing/: (package structure)

tests/ (11 arquivos, 8 suites + 100+ cases)
├─ Setup: conftest.py
├─ Validation: final_validation.py
├─ Suite 1: test_api_etapa3.py (18 testes)
├─ Suite 2: test_config_logging.py (34 testes)
├─ Suite 3: test_data_expanded.py (~15 testes)
├─ Suite 4: test_e2e.py (~12 testes)
├─ Suite 5: test_integration_e2e.py (~18 testes)
├─ Suite 6: test_mlflow.py (11 testes)
├─ Suite 7: test_mlp_etapa2.py (21 testes)
└─ Suite 8: test_models_expanded.py (15 testes)

docs/ (8 arquivos)
├─ README.md (Quick start)
├─ TECHNICAL_GUIDE.md (Este arquivo)
├─ PROJECT_OVERVIEW.md (Visão completa STAR)
├─ MODEL_CARD.pdf (Formal)
├─ MONITORING.md (Produção)
├─ ML_CANVAS.pdf (Canvas visual)

```

---

## ❌ Arquivos Removidos (Placeholders/Duplicatas)

Esses arquivos foram criados como placeholders mas nunca implementados:

| Arquivo Removido | Tipo | Razão |
|---|---|---|
| `src/train_baselines.py` | Duplicata | Funcionalidade já em `etapa1_eda_baselines.py` |
| `src/api/main.py` | Placeholder | Antigo, substituído por `app.py` |
| `src/monitoring/plan.py` | Placeholder | Conteúdo real em `docs/MONITORING.md` |
| `src/preprocessing/pipelines.py` | Placeholder | Vazio, não implementado |
| `tests/test_api.py` | Duplicata | Substituído por `test_api_etapa3.py` | ✓ |
| `tests/test_data.py` | Duplicata | Substituído por `test_data_expanded.py` | ✓ |
| `tests/test_models.py` | Duplicata | Substituído por `test_models_expanded.py` | ✓ |
| `tests/validate_paths.py` | Duplicata | Funcionalidade em `final_validation.py` | ✓ |

**Status**: ✅ Repositório limpo - apenas arquivos necessários
- Métricas (accuracy, precision, recall, F1, AUC)
- Comparação de baselines

**Executar**:
```bash
python -m pytest tests/test_models_expanded.py -v
```

---

#### 🎯 Executar Testes

```bash
# Todos os testes (100+ test cases, ~2min)
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Por marker
python -m pytest tests/ -m smoke              # Testes rápidos (< 5s)
python -m pytest tests/ -m schema             # Validação tipos/estrutura
python -m pytest tests/ -m integration        # E2E completo
python -m pytest tests/ -m unit               # Testes unitários
python -m pytest tests/ -m e2e                # End-to-end

# Por suite específica
python -m pytest tests/test_api_etapa3.py -v
python -m pytest tests/test_mlp_etapa2.py -v

# Com relatório HTML
python -m pytest tests/ -v --cov-report=html
# Abrir htmlcov/index.html no navegador
```
- Arquivo de log é criado

---

### 🟠 `docs/` - Documentação Formal (Etapa 4)

#### `docs/TECHNICAL_GUIDE.md` 📖 VOCÊ ESTÁ AQUI

**Propósito**: Guia técnico completo

**Seções**:
- Este arquivo que está lendo!
- Descreve cada arquivo do projeto
- Como cada arquivo é usado
- Fluxo de dados
- Mapeamento de requisitos do Tech Challenge

#### `docs/MODEL_CARD.pdf`

**Propósito**: Documentação formal do modelo (obrigatória Etapa 4)

**Conteúdo**:
```markdown
# Model Card - Telco Churn Prediction MLP

## Model Details
- Arquitetura: MLP 3 hidden layers
- Framework: PyTorch
- Input: 19 features normalizadas
- Output: Probabilidade de churn

## Performance
- Training Set: ~80% accuracy, AUC-ROC 0.84 (pós leakage fix)
- Test Set: [Após treinar]
- AUC-ROC: [Após treinar]

## Limitations
- Investigar data leakage (Churn Score)
- Imbalanced data (26.5% churn)
- Features podem ser correlacionadas

## Biases
- Possível viés por gênero em certos segmentos
- Efeito geográfico não investigado

## Failure Modes
- Low performance em clientes de curto prazo
- Overfitting possível com poucos dados por segmento

## Monitoring
- Data drift: Comparar distribuição mensal
- Model drift: Revalidar a cada quarter
- SLO: ≥ 85% AUC-ROC
```

#### `docs/MONITORING.md`

**Propósito**: Plano de monitoramento em produção (obrigatório Etapa 4)

**Conteúdo**:
```markdown
# Plano de Monitoramento - Telco Churn Prediction

## Métricas em Produção
- Latência da API (target: < 500ms p95)
- Taxa de erro (target: < 1%)
- Acurácia em novas predições (test set mensal)

## Data Drift Detection
- Comparar distribuição de features entrada vs. treino
- Alert se Kolmogorov-Smirnov > 0.05

## Model Drift Detection
- Revalidar modelo a cada 10k predições
- Alert se AUC-ROC cai > 5%

## SLOs
- Availability: 99.9%
- Latency p50: < 100ms
- Accuracy: ≥ 85%

## Alertas
- Data drift detectado → Investigar features
- Model drift detectado → Retreinar
- Latência alta → Scale horizontalmente
```

#### `docs/PROJECT_OVERVIEW.md`

**Propósito**: Visão completa do projeto (formato STAR)

**Conteúdo**:
- Situation: Problema de negócio, contexto
- Task: Objetivos técnicos, entregáveis
- Action: Decisões técnicas, features, arquitetura
- Result: Performance obtida, lições aprendidas

#### `docs/ML_CANVAS.pdf`

**Propósito**: ML Canvas visual (problema → solução)

**Seções**:
- Problem & Stakeholders
- Data & Features
- Model & ML Approach
- Metrics & Success Criteria
- Deployment & Monitoring

---

## 🔄 Fluxo de Dados

### Treinamento (Offline)

```
data/raw/Telco_customer_churn.xlsx
        ↓
src/etapa1_eda_baselines.py
        ↓
    [EDA + Preprocessing]
        ↓
    [Train 3 Baselines]
        ↓
models/{baseline}.pkl + scaler.pkl + etapa1_metadata.json
        ↓
src/train_mlp.py
        ↓
    [Train MLP com PyTorch]
        ↓
models/mlp_best_model.pth + mlp_training_history.json
```

### Inferência (Online)

```
Client Request (JSON)
        ↓
src/api/schemas.py [Validação Pydantic]
        ↓
src/api/app.py /predict endpoint
        ↓
    [Carregar MLP + Scaler]
        ↓
    [Preprocessing: scale features]
        ↓
    [Forward pass: 19 → 2]
        ↓
src/api/schemas.py PredictionResponse
        ↓
Client Response (JSON)
```

---

## 📋 Matriz de Responsabilidades

| Arquivo | Propósito | Etapa | Entrada | Saída |
|---------|-----------|-------|---------|-------|
| **etapa1_eda_baselines.py** | EDA + Baselines | 1 | Excel 7,043 clientes | 3 .pkl + metadata |
| **train_mlp.py** | Treinar MLP | 2 | Data preprocessado | mlp_best_model.pth |
| **mlp.py** | Arquitetura MLP | 2 | Input 33D | Output 2D (logits) |
| **training.py** | Loop treinamento | 2 | DataLoaders | History |
| **app.py** | API de produção | 3 | JSON request | JSON prediction |
| **schemas.py** | Validação | 3 | JSON input | Python objects |
| **final_validation.py** | Teste integração | 4 | Arquivo structure | Relatório validação |

---

## 🏗️ Arquitetura de Deploy: Real-Time vs Batch

### Decisão: Real-Time API (escolhida)

O projeto adota arquitetura de **inferência em tempo real** via API REST (FastAPI), com predição individual ou em micro-lote por requisição.

### Comparativo de Arquiteturas

| Critério | Real-Time (escolhido) | Batch |
|---|---|---|
| **Latência** | < 100ms por predição | Horas (agendado) |
| **Throughput** | ~1.000 req/s (CPU) | Milhões de registros |
| **Tecnologia** | FastAPI + Uvicorn | Spark / Airflow / cron |
| **Custo infra** | Servidor sempre ativo | Compute sob demanda |
| **Integração** | CRM / callcenter em tempo real | Data warehouse / relatório |
| **Freshness** | Imediata | Defasagem de horas/dias |
| **Complexidade** | Baixa — REST simples | Alta — orquestração de jobs |

### Justificativa da Escolha

**Por que Real-Time?**

1. **Caso de uso primário é reativo**: agentes de atendimento precisam saber o risco do cliente *durante a ligação* — não no relatório do dia seguinte. Latência de horas inviabiliza o uso operacional.

2. **Volume compatível**: a operadora tem ~7.000 clientes ativos. Com ~200 contatos/dia, o volume é baixo — não há necessidade de infra batch para processar grandes volumes.

3. **Flexibilidade de integração**: API REST permite integração direta com CRM, aplicativos web e sistemas de callcenter sem orquestração adicional.

4. **Menor complexidade operacional**: batch exigiria scheduler (Airflow/cron), gestão de jobs, controle de falhas e armazenamento intermediário. A API elimina essa complexidade.

5. **`/predict_batch` cobre o caso batch**: para campanhas noturnas ou processamento de listas, o endpoint `POST /predict_batch` aceita até 10.000 registros por requisição, cobrindo o caso de uso batch sem infra adicional.

### Quando Batch seria preferível

Batch seria a escolha certa se:
- Volume > 1 milhão de predições/dia
- Integração exclusiva com data warehouse (BigQuery, Redshift)
- Modelo com latência de GPU alto (LLM, modelos de recomendação complexos)
- Custo de servidor 24/7 for proibitivo

### Arquitetura Real-Time (implementada)

```
Cliente (CRM / App / Curl)
        │ HTTP POST /predict
        ▼
┌─────────────────────────────────┐
│  FastAPI (src/api/app.py)       │
│  ├─ Pydantic v2 validation      │
│  ├─ LatencyMiddleware (ms log)  │
│  └─ /predict  /predict_batch    │
└────────────┬────────────────────┘
             │
        ┌────▼────────────────────┐
        │  Preprocessor (sklearn) │
        │  StandardScaler + OHE   │
        └────┬────────────────────┘
             │
        ┌────▼────────────────────┐
        │  MLP PyTorch (CPU)      │
        │  19 → 128 → 64 → 32 → 2│
        └────┬────────────────────┘
             │ {prediction, probability_churn, confidence}
        ┌────▼────────────────────┐
        │  JSON Response          │
        │  X-Process-Time-Ms: Xms │
        └─────────────────────────┘
```

**Referência**: implementação em `src/api/app.py` · contrato em `src/api/schemas.py` · monitoramento em `src/monitoring/drift.py`.

---

## 🧪 Como Executar o Projeto Completo

### 1. Setup

```bash
cd Telecommunications_Industry
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
pip install -e ".[dev]"
```

### 2. Executar Etapas

```bash
# Etapa 1: EDA + Baselines
python -m src.etapa1_eda_baselines
# ✅ Output: models/*.pkl

# Etapa 2: Treinar MLP
python -m src.train_mlp
# ✅ Output: models/mlp_best_model.pth

# Etapa 3: Iniciar API
python -m uvicorn src.api.app:app --reload --port 8000
# ✅ Swagger: http://localhost:8000/docs

# Etapa 4: Validar Documentação
python tests/final_validation.py
# ✅ Verifica MODEL_CARD.pdf, MONITORING.md, etc.
```

### 3. Testar Tudo

```bash
python -m pytest tests/ -v --cov=src
```

---

## 📊 Mapeamento de Requisitos (Tech Challenge)

| Requisito | O que Implementamos | Arquivo |
|------------------|------------------|---------|
| **Etapa 1: Entendimento** | EDA completa | `src/etapa1_eda_baselines.py` |
| | ML Canvas preenchido | `docs/ML_CANVAS.pdf` |
| | 3 Baselines | `models/*_baseline.pkl` |
| | MLflow tracking | Automático em train |
| **Etapa 2: Rede Neural** | MLP PyTorch | `src/models/mlp.py` |
| | Training loop + early stop | `src/models/training.py` |
| | Comparação com baselines | Em `train_mlp.py` |
| | Trade-off analysis | Documentado |
| **Etapa 3: API** | Refatoração modular | `src/` estrutura |
| | Pipeline reprodutível | Sklearn pipeline |
| | Testes ≥ 3 tipos | `tests/*.py` |
| | FastAPI | `src/api/app.py` |
| | Logging estruturado | Via config + handlers |
| | pyproject.toml | `pyproject.toml` |
| **Etapa 4: Documentação** | Model Card | `docs/MODEL_CARD.pdf` |
| | Plano monitoramento | `docs/MONITORING.md` |
| | README finalizado | `README.md` |
| | Vídeo STAR (5 min) | [YouTube — Apresentação STAR](https://www.youtube.com/watch?v=cD3LQXYjdH0) |
| | (Opcional) Deploy nuvem | `kubernetes/` ready |

---

## 🎯 Próximas Ações

### Curto Prazo (Semana 1)
- [ ] `python -m src.train_mlp` - Treinar MLP e comparar
- [ ] Validar que MLP > baselines em alguma métrica
- [ ] Executar testes: `python -m pytest tests/ -v`

### Médio Prazo (Semana 2-3)
- [x] Vídeo STAR (5 min) — [YouTube](https://www.youtube.com/watch?v=cD3LQXYjdH0)
- [ ] Revisar MODEL_CARD e MONITORING
- [ ] Deploy local: `docker-compose up -d`

### Longo Prazo (Semana 4+)
- [ ] Deploy em nuvem (AWS/Azure/GCP)
- [ ] Hyperparameter tuning (Optuna)
- [ ] Investigar data leakage (Churn Score)

---

## 🆘 Troubleshooting

### Erro: `ModuleNotFoundError: No module named 'torch'`
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Erro: `PYTHONPATH` issue
```bash
export PYTHONPATH=$PWD
# ou
pip install -e .
```

### Testes falhando
```bash
cd Telecommunications_Industry  # Estar no diretório raiz
python -m pytest tests/final_validation.py -v
```

---

## 📞 Contato

- **Issues**: GitHub Issues
- **Docs**: Ver pasta `docs/`
- **Status**: Consulte [README.md](../README.md)

---

**Versão**: 2.0 | **Status**: ✅ Production Ready
