#!/usr/bin/env python
"""
Etapa 1: EDA + ML Canvas + Baselines
Projeto: Previsão de Churn em Telecomunicações
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle
import json

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# Configuration
from src.config import get_config
from src.logger import setup_logger

warnings.filterwarnings('ignore')
config = get_config("development")
logger = setup_logger(__name__, config)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent  # Go up one level from src/
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'Telco_customer_churn.xlsx'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

logger.info("="*80)
logger.info("ETAPA 1: EDA + ML CANVAS + BASELINES")
logger.info("="*80)

# ============================================================================
# 1. CARREGAR E LIMPAR DADOS
# ============================================================================
logger.info("\n[1] Loading and cleaning data...")
df = pd.read_excel(DATA_PATH)
logger.info(f"✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Copy to avoid modifying original
df = df.copy()

# Fix Total Charges (convert from object to numeric)
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
missing_total = df['Total Charges'].isna().sum()
if missing_total > 0:
    df.loc[df['Total Charges'].isna(), 'Total Charges'] = 0
    logger.info(f"  - Handled {missing_total} missing Total Charges (filled with 0)")

# Remove colunas administrativas e com data leakage
# Churn Score, Churn Reason e CLTV são derivados do target ou conhecidos só após churn
cols_remove = [
    'CustomerID', 'Count', 'Lat Long', 'Churn Label',
    'Churn Score',   # score derivado do target → leakage direto
    'Churn Reason',  # só existe quando cliente já churnou → label leakage
    'CLTV',          # Customer Lifetime Value correlacionado com target
    'Zip Code', 'Latitude', 'Longitude', 'City',  # IDs geográficos de alta cardinalidade
]
for col in cols_remove:
    if col in df.columns:
        df = df.drop(columns=[col])
        logger.info(f"  - Removed {col}")

# Remove Country and State if only 1 unique value
for col in ['Country', 'State']:
    if col in df.columns and df[col].nunique() == 1:
        df = df.drop(columns=[col])
        logger.info(f"  - Removed {col} (only 1 unique value)")

logger.info(f"✓ After cleaning: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================================
# 2. ANÁLISE EDA
# ============================================================================
logger.info("\n[2] Exploratory Data Analysis...")

target_col = 'Churn Value'
n_churn = (df[target_col] == 1).sum()
n_total = len(df)
churn_rate = n_churn / n_total

logger.info("\nTarget Distribution:")
logger.info(f"  - Total customers: {n_total:,}")
logger.info(f"  - Customers with churn: {n_churn:,}")
logger.info(f"  - Churn rate: {churn_rate:.2%}")
print(f"\n{df[target_col].value_counts().sort_index().to_string()}")

# Feature separation
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != target_col]
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

logger.info("\nFeature Summary:")
logger.info(f"  - Numeric features: {len(numeric_cols)}")
logger.info(f"  - Categorical features: {len(categorical_cols)}")

# Correlation with churn
df_numeric = df[numeric_cols + [target_col]].copy()
corr_matrix = df_numeric.corr()
churn_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)

logger.info("\nTop 5 Features by Correlation with Churn:")
for i, (col, corr_val) in enumerate(churn_corr.head(5).items(), 1):
    logger.info(f"  {i}. {col}: {corr_val:+.4f}")

# Data readiness
logger.info("\nData Readiness Checks:")
n_missing = df.isnull().sum().sum()
n_dupl = df.duplicated().sum()
logger.info(f"  - Missing values: {n_missing}")
logger.info(f"  - Duplicates: {n_dupl}")
logger.info(f"  ✓ Numeric features: {len(numeric_cols)}")
logger.info(f"  ✓ Categorical features: {len(categorical_cols)}")

# ============================================================================
# 3. PREPARAÇÃO PARA MODELAGEM
# ============================================================================
logger.info("\n[3] Feature preparation...")

X = df.drop(columns=[target_col])
y = df[target_col]

# Encode categorical features
X_encoded = X.copy()
label_encoders = {}

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

for col in categorical_features:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

logger.info(f"✓ Encoded {len(categorical_features)} categorical features")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

logger.info(f"✓ Train: {len(X_train):,} | Test: {len(X_test):,}")

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

logger.info(f"✓ Scaled {len(numeric_features)} numeric features")

# ============================================================================
# 4. TREINAMENTO DE BASELINES
# ============================================================================
logger.info("\n[4] Training baselines...")

results = {}

# 1. DummyClassifier
logger.info("\n  [4.1] DummyClassifier (always predicts majority class)...")
dummy = DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED)
dummy.fit(X_train_scaled, y_train)
y_pred_dummy = dummy.predict(X_test_scaled)

acc_dummy = accuracy_score(y_test, y_pred_dummy)
f1_dummy = f1_score(y_test, y_pred_dummy)
prec_dummy = precision_score(y_test, y_pred_dummy)
recall_dummy = recall_score(y_test, y_pred_dummy)

results['DummyClassifier'] = {
    'accuracy': acc_dummy,
    'f1': f1_dummy,
    'precision': prec_dummy,
    'recall': recall_dummy,
    'model': dummy
}

logger.info(f"    Accuracy: {acc_dummy:.4f}")
logger.info(f"    Precision: {prec_dummy:.4f}")
logger.info(f"    Recall: {recall_dummy:.4f}")
logger.info(f"    F1-Score: {f1_dummy:.4f}")

# 2. Logistic Regression
logger.info("\n  [4.2] Logistic Regression (linear baseline)...")
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
recall_lr = recall_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_proba_lr)

results['LogisticRegression'] = {
    'accuracy': acc_lr,
    'f1': f1_lr,
    'precision': prec_lr,
    'recall': recall_lr,
    'auc': auc_lr,
    'model': lr
}

logger.info(f"    Accuracy: {acc_lr:.4f}")
logger.info(f"    Precision: {prec_lr:.4f}")
logger.info(f"    Recall: {recall_lr:.4f}")
logger.info(f"    F1-Score: {f1_lr:.4f}")
logger.info(f"    AUC-ROC: {auc_lr:.4f}")

# 3. Random Forest
logger.info("\n  [4.3] Random Forest (ensemble baseline)...")
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    max_depth=15
)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)

results['RandomForest'] = {
    'accuracy': acc_rf,
    'f1': f1_rf,
    'precision': prec_rf,
    'recall': recall_rf,
    'auc': auc_rf,
    'model': rf
}

logger.info(f"    Accuracy: {acc_rf:.4f}")
logger.info(f"    Precision: {prec_rf:.4f}")
logger.info(f"    Recall: {recall_rf:.4f}")
logger.info(f"    F1-Score: {f1_rf:.4f}")
logger.info(f"    AUC-ROC: {auc_rf:.4f}")

# ============================================================================
# 5. VALIDAÇÃO CRUZADA
# ============================================================================
logger.info("\n[5] Stratified Cross-Validation (5-fold)...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

for model_name in ['LogisticRegression', 'RandomForest']:
    model = results[model_name]['model']
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring='f1')
    
    logger.info(f"\n  {model_name}:")
    logger.info(f"    F1-Scores: {cv_scores.round(4).tolist()}")
    logger.info(f"    Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# 6. RESUMO E SALVAR RESULTADOS
# ============================================================================
logger.info("\n[6] Saving results...")

# Comparison table
comparison_data = []
for model_name, metrics in results.items():
    row = {
        'Model': model_name,
        'Accuracy': f"{metrics['accuracy']:.4f}",
        'Precision': f"{metrics['precision']:.4f}",
        'Recall': f"{metrics['recall']:.4f}",
        'F1-Score': f"{metrics['f1']:.4f}",
        'AUC-ROC': f"{metrics.get('auc', '-'):.4f}" if 'auc' in metrics else '-'
    }
    comparison_data.append(row)

logger.info("\n=== BASELINE COMPARISON ===")
comparison_df = pd.DataFrame(comparison_data)
logger.info(f"\n{comparison_df.to_string(index=False)}")

# Save models
logger.info("\n✓ Saving models...")
for model_name, metrics in results.items():
    model_path = MODELS_DIR / f'{model_name.lower()}_baseline.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(metrics['model'], f)
    logger.info(f"  - {model_path.name}")

# Save scaler
scaler_path = MODELS_DIR / 'scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
logger.info(f"  - {scaler_path.name}")

# Save label encoders + feature order for the API to reproduce preprocessing
label_encoders_path = MODELS_DIR / 'label_encoders.pkl'
with open(label_encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)
logger.info(f"  - {label_encoders_path.name}")

feature_order = X_train_scaled.columns.tolist()
feature_columns_path = MODELS_DIR / 'feature_columns.json'
with open(feature_columns_path, 'w') as f:
    json.dump({
        'feature_order': feature_order,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'input_dim': len(feature_order),
    }, f, indent=2)
logger.info(f"  - {feature_columns_path.name}")

# Save baseline comparison CSV with real test-set metrics
baseline_rows = []
for model_name, metrics in results.items():
    row = {
        'model': model_name,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'auc_roc': metrics.get('auc', np.nan),
    }
    baseline_rows.append(row)
baseline_csv_path = MODELS_DIR / 'baseline_comparison.csv'
pd.DataFrame(baseline_rows).to_csv(baseline_csv_path, index=False)
logger.info(f"  - {baseline_csv_path.name}")

# PR-AUC for LR / RF (requires proba)
pr_auc_lr = average_precision_score(y_test, y_proba_lr)
pr_auc_rf = average_precision_score(y_test, y_proba_rf)

# Save metadata
metadata = {
    'target': target_col,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'label_encoders_count': len(label_encoders),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'churn_rate': float(churn_rate),
    'pr_auc': {
        'LogisticRegression': float(pr_auc_lr),
        'RandomForest': float(pr_auc_rf),
    },
    'results': {
        model: {k: v if isinstance(v, (int, float, str)) else str(type(v).__name__)
                 for k, v in metrics.items()}
        for model, metrics in results.items()
    }
}

metadata_path = MODELS_DIR / 'etapa1_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
logger.info(f"  - {metadata_path.name}")

# ============================================================================
# 7. ML CANVAS SUMMARY
# ============================================================================
logger.info("\n" + "="*80)
logger.info("ML CANVAS SUMMARY")
logger.info("="*80)

ml_canvas = {
    'Business Problem': 'Telecom operator losing customers at accelerated rate',
    'Goal': 'Predict customers at churn risk for proactive retention',
    'Business Metric': 'Cost of churn avoided (customer value vs action cost)',
    'Primary Technical Metric': 'AUC-ROC (balances sensitivity and specificity)',
    'Secondary Metric': 'F1-Score (precision/recall balance)',
    'Target': f'{target_col} (1 = churn, 0 = retention)',
    'Volume': f'{len(df):,} customers',
    'Churn Rate': f'{churn_rate:.2%}',
    'Features': f'{len(numeric_features)} numeric + {len(categorical_features)} categorical',
    'Train/Test Split': f'{len(X_train):,} / {len(X_test):,}',
    'Baseline (Dummy)': f'{acc_dummy:.2%} accuracy',
}

for key, value in ml_canvas.items():
    logger.info(f"{key:25s}: {value}")

logger.info("\n" + "="*80)
logger.info("✓ STAGE 1 COMPLETE")
logger.info("="*80)
logger.info("\nNext Steps:")
logger.info("  [Stage 2] Build MLP with PyTorch and compare with baselines")
logger.info("  [Stage 3] Refactor code into modules and create FastAPI")
logger.info("  [Stage 4] Model Card, documentation, and STAR video")
logger.info("="*80)
