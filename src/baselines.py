import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

class BaselineTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def train_dummy_classifier(self, X_train, y_train):
        model = DummyClassifier(strategy='most_frequent', random_state=self.random_state)
        model.fit(X_train, y_train)
        self.models['dummy'] = model
        return model

    def train_logistic_regression(self, X_train, y_train, max_iter=1000):
        model = LogisticRegression(max_iter=max_iter, random_state=self.random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        return model

    def train_random_forest(self, X_train, y_train, n_estimators=100):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=self.random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
        except Exception:
            auc_roc = 0.0
            pr_auc = 0.0
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        business_cost = fp + 5 * fn
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'pr_auc': pr_auc,
            'business_cost': business_cost,
            'confusion_matrix': cm.tolist(),
        }
    
    def compare_baselines(self, X_test, y_test):
        results = []
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model_name, model, X_test, y_test)
            results.append(metrics)
        return pd.DataFrame(results)
