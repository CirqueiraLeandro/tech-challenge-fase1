"""
Training loop para MLP com early stopping, validação cruzada e rastreamento MLflow.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.model_selection import StratifiedKFold

from src.models.mlp import MLPChurn
from src.config import get_config
from src.logger import setup_logger


config = get_config("development")
logger = setup_logger(__name__, config)


class EarlyStopping:
    """Early stopping para evitar overfitting."""
    
    def __init__(self, patience: int = 5, delta: float = 1e-4, verbose: bool = True):
        """
        Args:
            patience: Número de epochs sem melhora antes de parar
            delta: Melhora mínima considerada como progresso
            verbose: Se deve imprimir mensagens
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Retorna True se deve parar.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                logger.info(f"Validação loss melhorou para {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.warning(
                    f"Sem melhora por {self.counter}/{self.patience} epochs"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class MLPTrainer:
    """
    Treinador para MLP com early stopping.
    
    Características:
    - Early stopping baseado em validação
    - Learning rate scheduling
    - Rastreamento de métricas
    - Validação cruzada estratificada
    """
    
    def __init__(
        self,
        model: MLPChurn,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = "cpu",
        patience: int = 5,
        batch_size: int = 32,
    ):
        """
        Args:
            model: Modelo MLP
            learning_rate: Taxa de aprendizado
            weight_decay: L2 regularization
            device: cpu/cuda
            patience: Early stopping patience
            batch_size: Tamanho do batch
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        self.early_stopping = EarlyStopping(patience=patience)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Treina um epoch.
        
        Returns:
            (loss médio, accuracy médio)
        """
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_samples = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Métricas
            predictions = torch.argmax(outputs, dim=1)
            total_loss += loss.item() * X_batch.size(0)
            total_acc += (predictions == y_batch).sum().item()
            total_samples += X_batch.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples

        return avg_loss, avg_acc

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Valida o modelo.
        
        Returns:
            (loss médio, accuracy médio)
        """
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_samples = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                predictions = torch.argmax(outputs, dim=1)
                total_loss += loss.item() * X_batch.size(0)
                total_acc += (predictions == y_batch).sum().item()
                total_samples += X_batch.size(0)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        
        return avg_loss, avg_acc
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Treina o modelo com early stopping.
        
        Args:
            X_train: Dados de treino
            y_train: Labels de treino
            X_val: Dados de validação
            y_val: Labels de validação
            epochs: Número máximo de epochs
            
        Returns:
            Histórico de treinamento
        """
        logger.info(f"Iniciando treinamento com {len(X_train)} amostras")
        logger.info(f"Arquitetura: {self.model.count_parameters()} parâmetros")
        
        # Converter DataFrame para numpy se necessário
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(X_val, 'values'):
            X_val = X_val.values
        if hasattr(y_train, 'values'):
            y_train = y_train.values.ravel()
        if hasattr(y_val, 'values'):
            y_val = y_val.values.ravel()
        
        # Converter para tensores
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        # DataLoaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Treinamento
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Histórico
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )
            
            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping no epoch {epoch+1}")
                break
        
        return self.history
    
    def get_history_df(self) -> pd.DataFrame:
        """Retorna histórico como DataFrame."""
        return pd.DataFrame(self.history)


def train_with_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    epochs: int = 100,
    random_state: int = 42,
    device: str = "cpu",
) -> Tuple[List[MLPChurn], List[Dict], pd.DataFrame]:
    """
    Treina MLP com validação cruzada estratificada.
    
    Args:
        X: Dados
        y: Labels
        n_splits: Número de folds
        epochs: Epochs por fold
        random_state: Seed
        device: cpu/cuda
        
    Returns:
        (modelos, métricas por fold, histórico agregado)
    """
    logger.info(f"Iniciando validação cruzada com {n_splits} folds")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Convert X to numpy if it's a DataFrame
    if hasattr(X, 'values'):
        X_array = X.values
    else:
        X_array = X
    
    # Convert y to numpy if it's a Series
    if hasattr(y, 'values'):
        y_array = y.values
    else:
        y_array = y
    
    models = []
    fold_metrics = []
    histories = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_array, y_array), 1):
        logger.info(f"\n=== FOLD {fold}/{n_splits} ===")
        
        X_train, X_val = X_array[train_idx], X_array[val_idx]
        y_train, y_val = y_array[train_idx], y_array[val_idx]
        
        # Criar e treinar modelo
        model = MLPChurn(input_dim=X_array.shape[1])
        trainer = MLPTrainer(model, device=device)
        history = trainer.fit(X_train, y_train, X_val, y_val, epochs=epochs)
        
        models.append(model)
        histories.append(history)
        
        # Calcular métricas finais
        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(device)
            outputs = model(X_val_t)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            val_acc = np.mean(predictions == y_val)
        
        metrics = {
            'fold': fold,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'final_val_acc': val_acc,
            'best_epoch': len(history['val_loss']),
        }
        fold_metrics.append(metrics)
        
        logger.info(f"Fold {fold}: Validação Accuracy = {val_acc:.4f}")
    
    # Agregado
    metrics_df = pd.DataFrame(fold_metrics)
    logger.info(f"\nMédia de Validação Accuracy: {metrics_df['final_val_acc'].mean():.4f} "
                f"(+/- {metrics_df['final_val_acc'].std():.4f})")
    
    return models, fold_metrics, metrics_df
