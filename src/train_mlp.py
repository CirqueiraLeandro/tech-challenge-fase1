"""
Pipeline de treinamento MLP - Etapa 2.

Procura:
1. Carregar dados
2. Preprocessar
3. Treinar MLP com validação cruzada
4. Comparar com baselines
5. Rastrear no MLflow
6. Salvar resultados
"""

import json
import pickle

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from src.config import get_config
from src.data import DataLoader, DataPreprocessor
from src.baselines import BaselineTrainer
from src.models.mlp import MLPChurn, create_mlp_model
from src.models.training import MLPTrainer, train_with_cross_validation
from src.logger import setup_logger
from src.mlflow_utils import MLflowTracker


config = get_config("development")
logger = setup_logger(__name__, config)


def evaluate_model(
    model: MLPChurn,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    device: str = "cpu"
) -> dict:
    """
    Avalia modelo MLP.
    
    Args:
        model: Modelo treinado
        X_test: Dados de teste
        y_test: Labels de teste
        model_name: Nome do modelo
        device: cpu/cuda
        
    Returns:
        Dicionário com métricas
    """
    model.eval()

    X_arr = X_test.values if hasattr(X_test, 'values') else X_test
    y_arr = y_test.values if hasattr(y_test, 'values') else y_test

    X_test_t = torch.FloatTensor(X_arr.astype(np.float32)).to(device)
    y_pred = model.predict(X_test_t)
    y_proba = model.predict_proba(X_test_t)[:, 1]

    # Métricas
    accuracy = accuracy_score(y_arr, y_pred)
    precision = precision_score(y_arr, y_pred, zero_division=0)
    recall = recall_score(y_arr, y_pred, zero_division=0)
    f1 = f1_score(y_arr, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_arr, y_proba)
    pr_auc = average_precision_score(y_arr, y_proba)

    # Custo de negócio
    tn = ((y_pred == 0) & (y_arr == 0)).sum()
    fp = ((y_pred == 1) & (y_arr == 0)).sum()
    fn = ((y_pred == 0) & (y_arr == 1)).sum()
    tp = ((y_pred == 1) & (y_arr == 1)).sum()
    
    business_cost = fn * 500 + fp * 100 - tp * 300
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'pr_auc': pr_auc,
        'business_cost': business_cost,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }


def main():
    """Pipeline completo Etapa 2."""
    
    logger.info("=" * 80)
    logger.info("ETAPA 2: MODELAGEM COM REDES NEURAIS (MLP)")
    logger.info("=" * 80)
    
    # Configuração
    config = get_config("development")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # 1. Carregar dados
    logger.info("\n[1/6] Carregando dados...")
    data_loader = DataLoader(config)
    data_loader.load_dataset()
    data_loader.identify_target()  # Identificar coluna target
    X_train, X_test, y_train, y_test = data_loader.train_test_split(test_size=0.2)
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    
    # 2. Preprocessar
    logger.info("\n[2/6] Preprocessando dados...")
    num_cols = data_loader.get_feature_columns()[0]
    cat_cols = data_loader.get_feature_columns()[1]
    
    preprocessor = DataPreprocessor()
    X_train_pre = preprocessor.encode_categorical(X_train, cat_cols, fit=True)
    X_train_pre = preprocessor.scale_numeric(X_train_pre, num_cols, fit=True)
    
    X_test_pre = preprocessor.encode_categorical(X_test, cat_cols, fit=False)
    X_test_pre = preprocessor.scale_numeric(X_test_pre, num_cols, fit=False)
    
    logger.info(f"Preprocessado: {X_train_pre.shape}")
    
    # 3. MLflow setup
    logger.info("\n[3/6] Setup MLflow...")
    tracker = MLflowTracker(config)
    tracker.start_run(
        run_name="etapa_2_mlp_neural_network",
        params={
            "model_type": "MLP",
            "hidden_dims": "[128, 64, 32]",
            "dropout_rate": 0.3,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "early_stopping_patience": 5,
        }
    )
    
    # 4. Treinar MLP com validação cruzada
    logger.info("\n[4/6] Treinando MLP com validação cruzada (5 folds)...")
    models, fold_metrics, cv_results = train_with_cross_validation(
        X_train_pre,
        y_train,
        n_splits=5,
        epochs=100,
        random_state=config.RANDOM_SEED,
        device=device,
    )
    
    logger.info(f"\nCV Accuracy: {cv_results['final_val_acc'].mean():.4f} "
                f"(+/- {cv_results['final_val_acc'].std():.4f})")
    
    input_dim = X_train_pre.shape[1]
    logger.info(f"Input dim do MLP: {input_dim} features")

    # Usar melhor modelo (média de todos)
    final_model = create_mlp_model(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        device=device
    )
    
    # Retreinar no conjunto completo para benchmark
    logger.info("\nRetreinar no conjunto completo para benchmark...")
    trainer = MLPTrainer(final_model, device=device)
    trainer.fit(
        X_train_pre, y_train,
        X_train_pre, y_train,  # Usar treino como val para benchmark
        epochs=100
    )
    
    # 5. Comparar com baselines
    logger.info("\n[5/6] Comparando com baselines...")
    
    # Treinar baselines
    baseline_trainer = BaselineTrainer(random_state=config.RANDOM_SEED)
    baseline_trainer.train_dummy_classifier(X_train_pre, y_train)
    baseline_trainer.train_logistic_regression(X_train_pre, y_train)
    baseline_trainer.train_random_forest(X_train_pre, y_train)
    baseline_models = baseline_trainer.models  # dict: name -> trained model

    # Avaliar todos
    results = []

    # MLP
    mlp_metrics = evaluate_model(final_model, X_test_pre, y_test, "MLP (PyTorch)")
    results.append(mlp_metrics)
    logger.info(f"\nMLP: Accuracy={mlp_metrics['accuracy']:.4f}, "
                f"AUC-ROC={mlp_metrics['auc_roc']:.4f}, "
                f"F1={mlp_metrics['f1']:.4f}")

    # Baselines
    for name, model in baseline_models.items():
        y_pred = model.predict(X_test_pre)
        y_proba = model.predict_proba(X_test_pre)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'model': name.replace('_', ' ').title(),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_proba) if y_proba is not None else 0,
            'pr_auc': average_precision_score(y_test, y_proba) if y_proba is not None else 0,
        }
        
        # Custo de negócio
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        metrics['business_cost'] = fn * 500 + fp * 100 - tp * 300
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['tp'] = tp
        
        results.append(metrics)
        logger.info(f"\n{metrics['model']}: Accuracy={metrics['accuracy']:.4f}, "
                    f"AUC-ROC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}")
    
    # 6. Salvar resultados
    logger.info("\n[6/6] Salvando resultados...")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('auc_roc', ascending=False)
    
    output_file = config.MODELS_DIR / "mlp_comparison.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"Resultados salvos em: {output_file}")
    
    # Rastrear no MLflow
    tracker.log_dataframe_as_artifact(results_df, "mlp_comparison.csv")
    tracker.log_dataframe_as_artifact(cv_results, "cv_results.csv")
    
    metrics_to_log = {
        'mlp_accuracy': mlp_metrics['accuracy'],
        'mlp_auc_roc': mlp_metrics['auc_roc'],
        'mlp_f1': mlp_metrics['f1'],
        'cv_mean_accuracy': cv_results['final_val_acc'].mean(),
        'cv_std_accuracy': cv_results['final_val_acc'].std(),
    }
    tracker.log_metrics(metrics_to_log)
    
    # Salvar modelo
    model_path = config.MODELS_DIR / "mlp_etapa2.pt"
    final_model.save(model_path)
    logger.info(f"Modelo salvo em: {model_path}")

    # Salvar preprocessor para a API reproduzir o pipeline em inferência
    preprocessor_path = config.MODELS_DIR / "preprocessor.pkl"
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info(f"Preprocessor salvo em: {preprocessor_path}")

    feature_metadata_path = config.MODELS_DIR / "mlp_feature_metadata.json"
    with open(feature_metadata_path, "w") as f:
        json.dump({
            "feature_order": X_train_pre.columns.tolist(),
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "input_dim": int(X_train_pre.shape[1]),
        }, f, indent=2)
    logger.info(f"Feature metadata salvo em: {feature_metadata_path}")

    tracker.end_run()
    
    logger.info("\n" + "=" * 80)
    logger.info("ETAPA 2 COMPLETA!")
    logger.info("=" * 80)
    logger.info("\nResultados:")
    print(results_df.to_string(index=False))
    
    return results_df


if __name__ == "__main__":
    results = main()
