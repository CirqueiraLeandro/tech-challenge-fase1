# PLANO DE MONITORAMENTO - TELCO CHURN API

**Status**: ✅ Produção  
**Responsável**: MLOps Team

---

## 🎯 Objetivo

Monitorar a performance, disponibilidade e confiabilidade do modelo MLP em produção, detectando degradação de performance, data drift e anomalias de infraestrutura.

---

## 📈 MÉTRICAS DE NEGÓCIO

### KPIs Críticos

**1. Redução de Churn Prediction Accuracy**
- SLO: > 78%
- Alertar se: < 75%
- Critério: Calculado daily com ground truth
- Impacto: Reduzir churn em 15% (~R$ 750K economia/ano)

**2. Taxa de Conversão de Ações de Retenção**
- SLO: > 40% das recomendações resultam em retenção
- Medição: Mensal
- Impacto: R$ 300 por cliente retido (TP)

**3. ROI do Modelo**
- SLO: Payback < 6 meses
- Métrica: (Economia - Custo) / Custo inicial
- Ciclo: Trimestral

---

## 🤖 MÉTRICAS DE MODELO

### Performance em Produção

Baseline de referência (test set, pós data leakage fix):
- LogisticRegression: Accuracy 80.3%, AUC-ROC 0.848, F1 0.611
- MLP PyTorch: Accuracy 79.3%, AUC-ROC 0.837, F1 0.556

| Métrica | Baseline | SLO | Warning | Crítico | Frequência |
|---------|----------|-----|---------|---------|-----------|
| **Accuracy** | 80.3% | >78% | <75% | <70% | Daily |
| **Precision** | 64.3% | >60% | <55% | <50% | Daily |
| **Recall** | 58.3% | >55% | <50% | <45% | Daily |
| **F1-Score** | 0.611 | >0.58 | <0.53 | <0.48 | Daily |
| **AUC-ROC** | 0.848 | >0.80 | <0.75 | <0.70 | Daily |
| **PR-AUC** | 0.644 | >0.60 | <0.55 | <0.50 | Daily |

### Custo de Negócio

```
Business Cost = (FN × 500) + (FP × 100) - (TP × 300)

Onde:
- FN: False Negative (cliente deixou e não previmos) = R$ 500 loss
- FP: False Positive (previmos mas não saiu) = R$ 100 cost
- TP: True Positive (prevenimos saída) = R$ 300 value
- TN: True Negative (corretamente não churners) = R$ 0

SLO: Business Cost < R$ 100,000 / mês
Alerta: Business Cost > R$ 150,000
```

---

## 🔍 DATA DRIFT DETECTION

### Testes Estatísticos

**Features Numéricas**: Kolmogorov-Smirnov test
- Threshold: KS statistic > 0.2 = Alerta

**Features Categóricas**: Chi-squared test
- Threshold: p-value < 0.05 = Alerta

**Target Distribution**:
- Baseline: 26.5% churn
- Alerta: ±5% (21.5% - 31.5%)

### Monitoramento com Evidently AI

```python
# Executar daily:
from evidently.metric_preset import DataDriftPreset

dashboard = Dashboard(tabs=[
    DataDriftPreset(),
    TargetDriftPreset(),
    RegressionPerformancePreset(),
])
```

---

## 📊 MÉTRICAS DE INFRAESTRUTURA

### Latência API

| Percentil | SLO | Alerta |
|-----------|-----|--------|
| P50 | <50ms | >100ms |
| P95 | <100ms | >200ms |
| P99 | <200ms | >500ms |
| Max | <1000ms | >2000ms |

### Throughput e Erros

```
Throughput:
- Expected: ~100 req/s
- Alerta: < 50 req/s
- Crítico: < 10 req/s

Taxa de Erro:
- HTTP 5xx: < 0.1% | Alerta: > 1%
- HTTP 4xx: < 1% | Alerta: > 5%
- Timeout: < 0.1% | Alerta: > 0.5%
```

### Recursos

```
CPU: Alerta > 70% | Crítico > 90%
Memory: Alerta > 75% | Crítico > 90%
Disk: Alerta > 80% | Crítico > 95%
Uptime: SLO 99.5% (~3.6 horas downtime/mês)
```

---

## 🚨 SISTEMA DE ALERTAS

### Escalação por Nível

**INFO**: Log apenas
- Accuracy 78-82%
- Latência P95 100-200ms
- CPU 60-70%

**WARNING**: Email + Slack
- Accuracy 75-78%
- Latência P95 200-500ms
- CPU 70-85%
- Data drift PSI 0.1-0.2

**CRITICAL**: SMS + Pagerduty + All Channels
- Accuracy < 75%
- Latência P95 > 1s
- CPU > 85%
- Serviço DOWN
- Data drift PSI > 0.2

---

## 📈 DASHBOARDS

### Prometheus + Grafana

**Dashboard 1: Model Performance**
- Accuracy ao longo do tempo
- Precision vs Recall
- Confusion Matrix
- AUC-ROC curve
- Business Cost (com SLO)

**Dashboard 2: Data Drift**
- Feature distributions
- KS statistic trending
- Target shift detection
- Drift score

**Dashboard 3: Infrastructure**
- Latência P50/P95/P99
- Throughput
- Taxa de erro
- CPU/Memory/Disk

**Dashboard 4: Business Impact**
- Predictions by day
- True Positives trend
- Business Cost  
- ROI acumulado

---

## 🔄 CICLO DE RETRAINAMENTO

### Triggers Automáticos

```
1. Accuracy < 85% por 7 dias
2. Data drift score > 0.6
3. Nova 10K amostras rotuladas
4. Mensal agendado
```

### Processo

```
1. Prepare novo dataset (últimos 30 dias)
2. Validate data quality
3. Train novo modelo (MLP)
4. Validate novo modelo
5. Compare com versão em produção
6. Se melhor: deploy automático
7. Monitor por 7 dias
8. Se OK: promover; se não: rollback
```

---

## 📋 PLAYBOOKS

### Scenario 1: Accuracy Cai para 80%

**0-30 min**: Verificar data drift, anomalias, logs
**30 min-2h**: Analisar features, confusão matriz, mudanças negócio
**Resposta**: Retrainá, hotfix, feature eng, ou rollback

### Scenario 2: Latência P95 > 300ms

**Imediato**: Verificar CPU/Memory, padrão tráfego
**Investigação**: Otimizar código, escalar, caching
**Resposta**: Scale horizontalmente, otimizar, add Redis

### Scenario 3: Taxa de Erro > 1%

**Imediato**: Verificar tipo erro, stack trace
**Investigação**: 5xx/4xx/Timeout?
**Resposta**: Fix bug, tighten validation, otimizar

### Scenario 4: Serviço DOWN

**< 5 min**: Rollback versão anterior
**Investigação**: Qual mudança falhou?
**Comunicação**: Notificar stakeholders, updates 15min

---

## 📞 ESCALAÇÃO

```
Nível 1: Chat Bot (automated)
Nível 2: ML Engineer (< 15 min)
Nível 3: Lead ML Engineer (< 5 min)
Nível 4: Tech Lead (widespread outage)
```

---

## ✅ CHECKLIST

- [ ] Prometheus rodando
- [ ] Grafana com dashboards
- [ ] Alertas configurados
- [ ] Logs centralizados
- [ ] Data drift detection
- [ ] On-call rotation setup
- [ ] Playbooks testados
- [ ] Treinar time
- [ ] Deploy em produção
- [ ] Monitoramento 24/7

---

**Versão**: 1.0  
**Status**: ✅ COMPLETO  

> Implementar step-by-step conforme checklist\n"