"""Testes de Integração End-to-End - Pipeline Completo"""

import pytest
import numpy as np

from src.config import get_config
from src.data import DataLoader, DataPreprocessor
from src.baselines import BaselineTrainer
from src.mlflow_utils import MLflowTracker
from src.logger import setup_logger


class TestPipelineIntegrationE2E:
    """Testes do pipeline completo: Load → Preprocess → Train → Evaluate"""

    @pytest.fixture
    def config(self):
        return get_config("development")

    @pytest.fixture
    def logger(self, config):
        return setup_logger("integration_test", config)

    def test_pipeline_etapa1_completa(self, config, logger):
        """Etapa 1 Completa: Load → Preprocess → Baselines?"""
        
        logger.info("=" * 80)
        logger.info("TEST: ETAPA 1 COMPLETA")
        logger.info("=" * 80)
        
        # 1. LOAD
        logger.info("[1/4] Carregando dados...")
        loader = DataLoader(config)
        df = loader.load_dataset()
        loader.identify_target()
        
        assert len(df) > 0, "Dataset vazio"
        logger.info(f"✓ Dataset carregado: {df.shape}")
        
        # 2. EXPLORE
        logger.info("[2/4] Explorando features...")
        num_cols, cat_cols = loader.get_feature_columns()
        
        assert len(num_cols) > 0, "Sem features numéricas"
        assert len(cat_cols) > 0, "Sem features categóricas"
        logger.info(f"✓ Features: {len(num_cols)} numéricas + {len(cat_cols)} categóricas")
        
        # 3. SPLIT
        logger.info("[3/4] Realizando split...")
        X_train, X_test, y_train, y_test = loader.train_test_split(
            test_size=0.2, stratify=True
        )
        
        assert len(X_train) > len(X_test), "Train deveria ser maior"
        logger.info(f"✓ Split: {len(X_train)} train, {len(X_test)} test")
        
        # 4. PREPROCESS
        logger.info("[4/4] Pré-processando...")
        prep = DataPreprocessor()
        X_train_proc = prep.encode_categorical(X_train, cat_cols, fit=True)
        X_train_proc = prep.scale_numeric(X_train_proc, num_cols, fit=True)
        
        X_test_proc = prep.encode_categorical(X_test, cat_cols, fit=False)
        X_test_proc = prep.scale_numeric(X_test_proc, num_cols, fit=False)
        
        assert X_train_proc is not None, "X_train processado é None"
        assert X_test_proc is not None, "X_test processado é None"
        logger.info("✓ Dados processados")
        
        logger.info("=" * 80)
        logger.info("✓ ETAPA 1 CONCLUÍDA COM SUCESSO")
        logger.info("=" * 80)

    def test_pipeline_baselines_com_mlflow(self, config, logger):
        """Baselines com MLflow tracking?"""
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST: BASELINES COM MLFLOW")
        logger.info("=" * 80)
        
        # Load e prepare
        loader = DataLoader(config)
        loader.load_dataset()
        loader.identify_target()
        num_cols, cat_cols = loader.get_feature_columns()
        
        X_train, X_test, y_train, y_test = loader.train_test_split(test_size=0.2)
        
        # Preprocess
        prep = DataPreprocessor()
        X_train_proc = prep.encode_categorical(X_train, cat_cols, fit=True)
        X_train_proc = prep.scale_numeric(X_train_proc, num_cols, fit=True)
        
        X_test_proc = prep.encode_categorical(X_test, cat_cols, fit=False)
        X_test_proc = prep.scale_numeric(X_test_proc, num_cols, fit=False)
        
        # Train with MLflow
        logger.info("Treinando modelos com MLflow...")
        tracker = MLflowTracker(config)
        trainer = BaselineTrainer(random_state=42)
        
        # DummyClassifier
        logger.info("  • DummyClassifier...")
        tracker.start_run("dummy_baseline", {"model": "DummyClassifier"})
        trainer.train_dummy_classifier(X_train_proc, y_train)
        metrics = trainer.evaluate_model("dummy", trainer.models["dummy"], 
                                        X_test_proc, y_test)
        tracker.log_metrics({k: v for k, v in metrics.items() 
                            if isinstance(v, (int, float))})
        logger.info(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
        tracker.end_run()
        
        # LogisticRegression
        logger.info("  • LogisticRegression...")
        tracker.start_run("logistic_baseline", {"model": "LogisticRegression"})
        trainer.train_logistic_regression(X_train_proc, y_train)
        metrics = trainer.evaluate_model("logistic_regression", 
                                        trainer.models["logistic_regression"], 
                                        X_test_proc, y_test)
        tracker.log_metrics({k: v for k, v in metrics.items() 
                            if isinstance(v, (int, float))})
        logger.info(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
        tracker.end_run()
        
        # RandomForest
        logger.info("  • RandomForest...")
        tracker.start_run("rf_baseline", {"model": "RandomForest", "n_estimators": 100})
        trainer.train_random_forest(X_train_proc, y_train, n_estimators=100)
        metrics = trainer.evaluate_model("random_forest", 
                                        trainer.models["random_forest"], 
                                        X_test_proc, y_test)
        tracker.log_metrics({k: v for k, v in metrics.items() 
                            if isinstance(v, (int, float))})
        logger.info(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
        tracker.end_run()
        
        # Compare
        logger.info("\nComparando baselines...")
        comparison = trainer.compare_baselines(X_test_proc, y_test)
        logger.info(f"\n{comparison.to_string()}")
        
        assert len(comparison) == 3, "Deveria ter 3 modelos"
        
        logger.info("=" * 80)
        logger.info("✓ BASELINES COM MLFLOW CONCLUÍDO COM SUCESSO")
        logger.info("=" * 80)

    def test_reproducibilidade_completa(self, config, logger):
        """Pipeline com seed produz mesmo resultado?"""
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST: REPRODUCIBILIDADE COMPLETA")
        logger.info("=" * 80)
        
        def _run_pipeline(config):
            loader = DataLoader(config)
            loader.load_dataset()
            loader.identify_target()
            X_train, X_test, y_train, y_test = loader.train_test_split(
                random_state=42, test_size=0.2
            )
            _, cat_cols = loader.get_feature_columns()
            prep = DataPreprocessor()
            X_train_p, y_train_p = prep.preprocess(X_train, y_train, fit=True)
            X_test_p, y_test_p = prep.preprocess(X_test, y_test, fit=False)
            trainer = BaselineTrainer(random_state=42)
            trainer.train_logistic_regression(X_train_p, y_train_p)
            metrics = trainer.evaluate_model(
                "lr", trainer.models["logistic_regression"], X_test_p, y_test_p
            )
            return metrics

        # Primeira execução
        logger.info("Execução 1...")
        metrics1 = _run_pipeline(config)

        # Segunda execução
        logger.info("Execução 2...")
        metrics2 = _run_pipeline(config)
        
        # Verificar reproducibilidade
        logger.info(f"Exec 1 AUC-ROC: {metrics1['auc_roc']:.6f}")
        logger.info(f"Exec 2 AUC-ROC: {metrics2['auc_roc']:.6f}")
        
        assert metrics1['auc_roc'] == metrics2['auc_roc'], "AUC-ROC não é reproducível"
        assert metrics1['f1'] == metrics2['f1'], "F1 não é reproducível"
        
        logger.info("=" * 80)
        logger.info("✓ REPRODUCIBILIDADE VERIFICADA")
        logger.info("=" * 80)

    def test_data_quality_checks(self, config, logger):
        """Data quality checks passam?"""
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST: DATA QUALITY CHECKS")
        logger.info("=" * 80)
        
        loader = DataLoader(config)
        df = loader.load_dataset()
        
        checks = {
            "sem_missings": df.isnull().sum().sum() == 0,
            "sem_duplicatas": df.duplicated().sum() == 0,
            "tipos_validos": all(df[col].dtype in [np.int64, np.float64, object] 
                                for col in df.columns),
            "linhas_suficientes": len(df) > 100,
            "colunas_suficientes": len(df.columns) > 5,
        }
        
        logger.info("\nData Quality Checklist:")
        for check_name, result in checks.items():
            status = "✓" if result else "✗"
            logger.info(f"  {status} {check_name}")
        
        assert all(checks.values()), "Alguns checks falharam"
        
        logger.info("=" * 80)
        logger.info("✓ TODOS OS DATA QUALITY CHECKS PASSARAM")
        logger.info("=" * 80)

    def test_model_performance_baseline(self, config, logger):
        """Modelos têm performance razoável?"""
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST: MODEL PERFORMANCE BASELINE")
        logger.info("=" * 80)
        
        # Load
        loader = DataLoader(config)
        loader.load_dataset()
        loader.identify_target()
        num_cols, cat_cols = loader.get_feature_columns()
        
        X_train, X_test, y_train, y_test = loader.train_test_split(test_size=0.2)
        
        # Preprocess
        prep = DataPreprocessor()
        X_train_proc = prep.encode_categorical(X_train, cat_cols, fit=True)
        X_train_proc = prep.scale_numeric(X_train_proc, num_cols, fit=True)
        
        X_test_proc = prep.encode_categorical(X_test, cat_cols, fit=False)
        X_test_proc = prep.scale_numeric(X_test_proc, num_cols, fit=False)
        
        # Train e evaluate
        trainer = BaselineTrainer(random_state=42)
        trainer.train_logistic_regression(X_train_proc, y_train)
        trainer.train_random_forest(X_train_proc, y_train, n_estimators=50)
        
        comparison = trainer.compare_baselines(X_test_proc, y_test)
        
        logger.info("\nPerformance Baselines:")
        for _, row in comparison.iterrows():
            logger.info(f"  {row['model_name']}: AUC-ROC={row['auc_roc']:.4f}")
        
        # Random Forest deve ter melhor AUC que LogReg em geral
        rf_row = comparison[comparison['model_name'] == 'random_forest']
        lr_row = comparison[comparison['model_name'] == 'logistic_regression']
        
        if not rf_row.empty and not lr_row.empty:
            rf_auc = rf_row['auc_roc'].values[0]
            lr_auc = lr_row['auc_roc'].values[0]
            logger.info(f"\nRF melhor que LogReg: {rf_auc > lr_auc}")
        
        logger.info("=" * 80)
        logger.info("✓ MODEL PERFORMANCE BASELINE CONCLUÍDO")
        logger.info("=" * 80)
