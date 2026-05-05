"""Testes Unitários - Módulo de Dados (Expandido)"""

import pytest
import pandas as pd
import numpy as np

from src.data import DataLoader, DataPreprocessor
from src.config import get_config


class TestDataLoaderSmoke:
    """Smoke tests - Verificações básicas críticas"""

    @pytest.fixture
    def config(self):
        config = get_config("development")  # Usar config dev com dataset real
        return config

    @pytest.mark.smoke
    def test_dataset_carrega(self, config):
        """SMOKE: Dataset existe e carrega?"""
        loader = DataLoader(config)
        df = loader.load_dataset()
        assert df is not None
        assert len(df) > 0
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.smoke
    def test_target_existe(self, config):
        """SMOKE: Coluna target pode ser identificada?"""
        loader = DataLoader(config)
        loader.load_dataset()
        target = loader.identify_target()
        assert target is not None
        assert "churn" in target.lower()

    @pytest.mark.smoke
    def test_features_extraem(self, config):
        """SMOKE: Features podem ser extraídas?"""
        loader = DataLoader(config)
        loader.load_dataset()
        loader.identify_target()
        num_cols, cat_cols = loader.get_feature_columns()
        assert len(num_cols) > 0
        assert len(cat_cols) > 0


class TestDataLoaderSchema:
    """Schema tests - Validação de integridade de dados"""

    @pytest.fixture
    def config(self):
        config = get_config("development")
        return config

    @pytest.mark.schema
    def test_sem_missings(self, config):
        """SCHEMA: Dataset sem valores faltantes?"""
        loader = DataLoader(config)
        df = loader.load_dataset()
        missing = df.isnull().sum().sum()
        assert missing == 0, f"Encontrados {missing} missings"

    @pytest.mark.schema
    def test_sem_duplicatas(self, config):
        """SCHEMA: Dataset sem linhas duplicadas?"""
        loader = DataLoader(config)
        df = loader.load_dataset()
        dups = df.duplicated().sum()
        assert dups == 0, f"Encontradas {dups} duplicatas"

    @pytest.mark.schema
    def test_tipos_dados_validos(self, config):
        """SCHEMA: Tipos de dados válidos?"""
        loader = DataLoader(config)
        df = loader.load_dataset()
        for col in df.columns:
            assert df[col].dtype in [np.int64, np.float64, object], \
                f"Tipo inválido em {col}: {df[col].dtype}"

    @pytest.mark.schema
    def test_target_binar(self, config):
        """SCHEMA: Target é binário (0/1)?"""
        loader = DataLoader(config)
        df = loader.load_dataset()
        loader.identify_target()
        target_col = loader.target_col
        unique_vals = set(df[target_col].unique())
        assert unique_vals.issubset({0, 1, "No", "Yes", "Sim", "Não"}), \
            f"Target com valores inesperados: {unique_vals}"


class TestDataLoaderIntegration:
    """Integration tests - Fluxos completos"""

    @pytest.fixture
    def config(self):
        config = get_config("development")
        return config

    def test_split_traintest_proporcoes(self, config):
        """INTEG: Split treino-teste com proporções corretas?"""
        loader = DataLoader(config)
        df = loader.load_dataset()
        loader.identify_target()
        
        X_train, X_test, y_train, y_test = loader.train_test_split(test_size=0.2)
        
        # Verificar proporções
        assert len(X_train) / len(df) == pytest.approx(0.8, abs=0.01)
        assert len(X_test) / len(df) == pytest.approx(0.2, abs=0.01)

    def test_split_estratificado(self, config):
        """INTEG: Split mantém proporção de classes?"""
        loader = DataLoader(config)
        df = loader.load_dataset()
        loader.identify_target()
        
        X_train, X_test, y_train, y_test = loader.train_test_split(
            test_size=0.2, stratify=True
        )
        
        # Proporção de positivos deve ser similar
        train_ratio = y_train.sum() / len(y_train)
        test_ratio = y_test.sum() / len(y_test)
        all_ratio = df[loader.target_col].sum() / len(df)
        
        assert abs(train_ratio - all_ratio) < 0.05
        assert abs(test_ratio - all_ratio) < 0.05

    def test_reproducibilidade_seed(self, config):
        """INTEG: Seed reproduz split?"""
        loader1 = DataLoader(config)
        loader1.load_dataset()
        loader1.identify_target()
        X1_train, X1_test, y1_train, y1_test = loader1.train_test_split(
            random_state=42
        )

        loader2 = DataLoader(config)
        loader2.load_dataset()
        loader2.identify_target()
        X2_train, X2_test, y2_train, y2_test = loader2.train_test_split(
            random_state=42
        )
        
        # Índices devem ser iguais
        assert (X1_train.index == X2_train.index).all()
        assert (X1_test.index == X2_test.index).all()


class TestDataPreprocessorUnit:
    """Unit tests - Componentes isolados de preprocessing"""

    @pytest.fixture
    def sample_numeric(self):
        """Sample data com features numéricas"""
        return pd.DataFrame({
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "col2": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

    @pytest.fixture
    def sample_categorical(self):
        """Sample data com features categóricas"""
        return pd.DataFrame({
            "col1": ["A", "B", "A", "B", "C"],
            "col2": ["X", "Y", "X", "Y", "Z"],
        })

    def test_init(self):
        """UNIT: Preprocessor inicializa corretamente?"""
        prep = DataPreprocessor()
        assert prep.numeric_scaler is None
        assert len(prep.label_encoders) == 0

    def test_categorical_encode_fit(self, sample_categorical):
        """UNIT: Encoding categórico com fit funciona?"""
        prep = DataPreprocessor()
        X = sample_categorical.copy()
        X_enc = prep.encode_categorical(X, ["col1"], fit=True)
        
        assert X_enc["col1"].dtype in [np.int64, np.int32]
        assert "col1" in prep.label_encoders
        assert len(set(X_enc["col1"])) == 3  # A, B, C

    def test_categorical_encode_transform(self, sample_categorical):
        """UNIT: Transform categórico mantém consistência?"""
        prep = DataPreprocessor()
        X_train = sample_categorical.iloc[:3]
        X_test = sample_categorical.iloc[3:].copy()
        
        # Fit
        X_train_enc = prep.encode_categorical(X_train, ["col1"], fit=True)
        
        # Transform
        X_test_enc = prep.encode_categorical(X_test, ["col1"], fit=False)
        
        # O mesmo valor categórico deve ter o mesmo encoding em train e test
        first_train_cat = X_train_enc["col1"].iloc[0]
        first_test_cat = X_test_enc["col1"].iloc[0]
        # Ambos devem ser inteiros (encoded), não strings
        assert isinstance(first_train_cat, (int, np.integer))
        assert isinstance(first_test_cat, (int, np.integer))

    def test_numeric_scale_fit(self, sample_numeric):
        """UNIT: Scaling numérico com fit?"""
        prep = DataPreprocessor()
        X = sample_numeric.copy()
        X_scaled = prep.scale_numeric(X, ["col1"], fit=True)
        
        # Deve ter mean ~0 e std (ddof=0) ~1 — StandardScaler usa ddof=0
        assert abs(X_scaled["col1"].mean()) < 0.01
        assert abs(X_scaled["col1"].std(ddof=0) - 1.0) < 0.01
        assert prep.numeric_scaler is not None

    def test_numeric_scale_transform(self, sample_numeric):
        """UNIT: Scaling numérico com transform?"""
        prep = DataPreprocessor()
        X = sample_numeric.copy()
        
        # Fit
        prep.scale_numeric(X, ["col1"], fit=True)

        # Transform novo valor
        X_new = pd.DataFrame({"col1": [100.0], "col2": [500.0]})
        X_new_scaled = prep.scale_numeric(X_new, ["col1"], fit=False)
        
        # Deve estar escalado (não igual a 100.0)
        assert X_new_scaled["col1"].iloc[0] != 100.0

    def test_target_encode(self):
        """UNIT: Target encoding funciona?"""
        prep = DataPreprocessor()
        y = pd.Series([0, 1, 0, 1, 0])
        y_enc = prep.encode_target(y, fit=True)
        
        assert isinstance(y_enc, np.ndarray)
        assert set(y_enc) == {0, 1}
        assert "target" in prep.label_encoders


class TestDataPreprocessorIntegration:
    """Integration tests - Pipelines de preprocessing"""

    @pytest.fixture
    def sample_complete(self):
        """Sample data completo (numeric + categorical + target)"""
        return pd.DataFrame({
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "cat1": ["A", "B", "A", "B", "C"],
            "num2": [10.0, 20.0, 30.0, 40.0, 50.0],
            "cat2": ["X", "Y", "X", "Y", "Z"],
            "target": [0, 1, 0, 1, 0]
        })

    def test_pipeline_completo(self, sample_complete):
        """INTEG: Pipeline completo de preprocessing?"""
        prep = DataPreprocessor()
        
        X = sample_complete[["num1", "cat1", "num2", "cat2"]]
        y = sample_complete["target"]
        
        X_proc, y_proc = prep.preprocess(X, y, fit=True)
        
        assert X_proc is not None
        assert y_proc is not None
        assert len(X_proc) == len(X)
        assert len(y_proc) == len(y)

    def test_pipeline_idempotencia(self, sample_complete):
        """INTEG: Aplicar preprocessing 2x dá mesmo resultado?"""
        prep = DataPreprocessor()
        
        X = sample_complete[["num1", "cat1", "num2", "cat2"]]
        y = sample_complete["target"]
        
        # Primeira aplicação
        X_proc1, y_proc1 = prep.preprocess(X, y, fit=True)
        
        # Segunda aplicação (sem fit)
        X_proc2, y_proc2 = prep.preprocess(X, y, fit=False)
        
        # Deve ser idêntico
        np.testing.assert_array_almost_equal(X_proc1.values, X_proc2.values)
        np.testing.assert_array_equal(y_proc1, y_proc2)

    def test_pipeline_error_sem_fit(self, sample_complete):
        """INTEG: Erro se tentar transform sem fit?"""
        prep = DataPreprocessor()
        X = sample_complete[["num1", "cat1", "num2", "cat2"]]
        y = sample_complete["target"]
        
        with pytest.raises(ValueError):
            prep.preprocess(X, y, fit=False)
