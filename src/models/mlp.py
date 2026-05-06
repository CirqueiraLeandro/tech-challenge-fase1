"""
MLP (Multi-Layer Perceptron) para previsão de churn.

Arquitetura:
- Input: 19 features (3 numéricas + 16 categóricas, pós leakage fix)
- Hidden: [128, 64, 32] com ReLU + BatchNorm + Dropout
- Output: 2 classes (binary classification)
- Dropout: 0.3 (regularização)
- Total: 13.410 parâmetros treináveis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from pathlib import Path


class MLPChurn(nn.Module):
    """
    Multi-Layer Perceptron para classificação de churn.
    
    Args:
        input_dim: Número de features de entrada (padrão: 19, pós leakage fix)
        hidden_dims: Lista com dimensões dos hidden layers (padrão: [128, 64, 32])
        output_dim: Número de classes de saída (padrão: 2)
        dropout_rate: Taxa de dropout (padrão: 0.3)
        activation: Função de ativação (padrão: ReLU)
    """

    def __init__(
        self,
        input_dim: int = 19,
        hidden_dims: Optional[list] = None,
        output_dim: int = 2,
        dropout_rate: float = 0.3,
        activation: str = "relu"
    ):
        super(MLPChurn, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Mapeamento de ativações
        activations = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        self.activation = activations.get(activation.lower(), nn.ReLU())
        
        # Construir arquitetura
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation,
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada (batch_size, input_dim)
            
        Returns:
            Tensor de logits (batch_size, output_dim)
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict classes (sem gradiente).
        
        Args:
            x: Tensor de entrada (batch_size, input_dim)
            
        Returns:
            Array de predições (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions.cpu().numpy()
    
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            x: Tensor de entrada (batch_size, input_dim)
            
        Returns:
            Array de probabilidades (batch_size, output_dim)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()
    
    def save(self, path: Path) -> None:
        """Salvar modelo."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
    
    def load(self, path: Path) -> None:
        """Carregar modelo."""
        self.load_state_dict(torch.load(path, map_location='cpu'))
    
    def count_parameters(self) -> int:
        """Contar total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_mlp_model(
    input_dim: int = 19,
    hidden_dims: Optional[list] = None,
    output_dim: int = 2,
    dropout_rate: float = 0.3,
    device: str = "cpu"
) -> MLPChurn:
    """
    Factory function para criar modelo MLP.
    
    Args:
        input_dim: Número de features
        hidden_dims: Dimensões hidden layers
        output_dim: Número de classes
        dropout_rate: Taxa de dropout
        device: Device (cpu/cuda)
        
    Returns:
        MLPChurn model movido para device
    """
    model = MLPChurn(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout_rate=dropout_rate
    )
    model = model.to(device)
    return model