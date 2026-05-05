import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

_LEAK_COLS = [
    'CustomerID', 'Count', 'Lat Long', 'Churn Label',
    'Churn Score',   # score derivado do target → leakage
    'Churn Reason',  # só existe pós-churn → label leakage
    'CLTV',          # correlacionado com target
    'Zip Code', 'Latitude', 'Longitude', 'City',
]


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.target_col = None

    def load_dataset(self):
        df = pd.read_excel(self.config.DATASET_PATH)
        df = df.drop(columns=[c for c in _LEAK_COLS if c in df.columns])
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)
        df = df.drop_duplicates()
        # Remove Country/State se apenas 1 valor único
        for col in ['Country', 'State']:
            if col in df.columns and df[col].nunique() == 1:
                df = df.drop(columns=[col])
        self.df = df
        return self.df

    def identify_target(self):
        target_cols = [col for col in self.df.columns if 'churn' in col.lower()]
        self.target_col = target_cols[0] if target_cols else None
        return self.target_col
    
    def get_feature_columns(self):
        exclude = {self.target_col} if self.target_col else set()
        numeric_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c not in exclude]
        categorical_cols = [c for c in self.df.select_dtypes(include=['object']).columns if c not in exclude]
        return numeric_cols, categorical_cols
    
    def train_test_split(self, test_size=0.2, random_state=42, stratify=True):
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        if y.dtype == 'object':
            y = (y.str.lower() == 'yes').astype(int)
        stratify_col = y if stratify else None
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_col)

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.numeric_scaler = None  # inicializado ao fit
        self.scaler = StandardScaler()  # alias mantido por compatibilidade
    
    def encode_categorical(self, X, cat_cols, fit=True):
        X_encoded = X.copy()
        for col in cat_cols:
            if col not in X_encoded.columns:
                continue
            # Converter para string para evitar mistura de tipos
            X_encoded[col] = X_encoded[col].astype(str)
            if fit:
                # Label encoding simples - convert strings to integers
                self.label_encoders[col] = LabelEncoder()
                # Fit com todas as categorias possiveis
                unique_vals = list(X_encoded[col].unique())
                self.label_encoders[col].fit(unique_vals)
                X_encoded[col] = self.label_encoders[col].transform(X_encoded[col])
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    try:
                        X_encoded[col] = self.label_encoders[col].transform(X_encoded[col])
                    except ValueError:
                        # Se houver valor novo, usar -1 ou a media
                        encoder = self.label_encoders[col]
                        X_encoded[col] = X_encoded[col].map(
                            lambda x: encoder.transform([x])[0] if x in encoder.classes_ 
                            else 0
                        )
        return X_encoded
    
    def scale_numeric(self, X, num_cols, fit=True):
        X_scaled = X.copy()
        if not num_cols:
            return X_scaled
        if fit:
            self.numeric_scaler = self.scaler
            X_scaled[num_cols] = self.scaler.fit_transform(X_scaled[num_cols])
        else:
            X_scaled[num_cols] = self.scaler.transform(X_scaled[num_cols])
        return X_scaled

    def encode_target(self, y, fit=True):
        """Codifica target para array numpy inteiro."""
        le = LabelEncoder()
        if fit:
            arr = le.fit_transform(y)
            self.label_encoders['target'] = le
        else:
            le = self.label_encoders.get('target')
            arr = le.transform(y) if le else np.array(y)
        return arr.astype(int)

    def preprocess(self, X, y, fit=True):
        """Pipeline completo: encode categoricas + escala numericas + codifica target."""
        if not fit and not self.label_encoders:
            raise ValueError("Preprocessor não foi ajustado (fit=False sem fit anterior).")
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_proc = self.encode_categorical(X, cat_cols, fit=fit)
        X_proc = self.scale_numeric(X_proc, num_cols, fit=fit)
        y_proc = self.encode_target(y, fit=fit)
        return X_proc, y_proc
