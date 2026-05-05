import mlflow
import pandas as pd

class MLflowTracker:
    def __init__(self, config):
        self.config = config
        # Use local file-based backend instead of remote server
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(config.EXPERIMENT_NAME)
    
    def start_run(self, run_name, params=None):
        mlflow.start_run(run_name=run_name)
        if params:
            mlflow.log_params(params)
    
    def end_run(self):
        mlflow.end_run()
    
    def log_metric(self, key, value, step=None):
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

    def log_dataframe_as_artifact(self, df, filename):
        """Salvar DataFrame como CSV artifact no MLflow."""
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, filename)
            df.to_csv(path, index=False)
            mlflow.log_artifact(path)
    
    @staticmethod
    def get_experiment_id(experiment_name):
        try:
            exp = mlflow.get_experiment_by_name(experiment_name)
            return exp.experiment_id if exp else None
        except Exception:
            return None
    
    @staticmethod
    def get_experiment_runs(experiment_name):
        try:
            exp_id = MLflowTracker.get_experiment_id(experiment_name)
            if exp_id:
                return mlflow.search_runs(experiment_ids=[exp_id])
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def compare_runs(experiment_name):
        """Retorna DataFrame comparando todos os runs do experimento."""
        return MLflowTracker.get_experiment_runs(experiment_name)
