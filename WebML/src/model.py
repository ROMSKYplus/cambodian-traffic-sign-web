"""
Model architecture and loading utilities for Cambodian Traffic Sign Recognition.

Architecture:
    Input: 50 PCA features
    Hidden Layer 1: 100 neurons, ReLU activation
    Hidden Layer 2: 50 neurons, ReLU activation
    Output Layer: 10 neurons (raw logits)

Model files:
    - models/best_ann.pth: PyTorch state dictionary for TrafficSignANN
    - models/pca_ann.pkl: Fitted Scikit-learn PCA transformer (1024 -> 50)
"""

from pathlib import Path
from typing import Tuple, Union, Optional
import os
import joblib
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


class TrafficSignANN(nn.Module):
    """
    Artificial Neural Network for 10-class Cambodian Traffic Sign Recognition.
    
    Accepts 50 PCA-reduced features and produces 10 unnormalized class logits.
    The layer naming `self.network` matches the checkpoint keys:
        - network.0: Linear(50, 100)
        - network.1: ReLU()
        - network.2: Linear(100, 50)
        - network.3: ReLU()
        - network.4: Linear(50, 10)
    """

    def __init__(self, in_features: int = 50, hidden1: int = 100, hidden2: int = 50, num_classes: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 50).
            
        Returns:
            torch.Tensor: Raw logits tensor with shape (batch_size, 10).
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)

    def forward_with_internals(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass returning final logits alongside intermediate layer activations.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, 50).
            
        Returns:
            Tuple[torch.Tensor, dict]: (logits, internals_dict)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Layer 1: Linear(50, 100) + ReLU()
        h1_pre = self.network[0](x)
        h1_post = self.network[1](h1_pre)
        
        # Layer 2: Linear(100, 50) + ReLU()
        h2_pre = self.network[2](h1_post)
        h2_post = self.network[3](h2_pre)
        
        # Output Layer: Linear(50, 10)
        logits = self.network[4](h2_post)
        
        internals = {
            "h1_pre": h1_pre,
            "h1_post": h1_post,
            "h2_pre": h2_pre,
            "h2_post": h2_post,
            "logits": logits,
        }
        return logits, internals


def load_model(
    model_path: Union[str, Path] = "models/best_ann.pth",
    device: str = "cpu"
) -> TrafficSignANN:
    """
    Load the trained PyTorch ANN model from disk.
    
    Args:
        model_path (str | Path): Path to the `best_ann.pth` weights file.
        device (str): Device to map weights onto ('cpu' or 'cuda'/'mps'). Defaults to 'cpu'.
        
    Returns:
        TrafficSignANN: Model loaded with weights and set to eval mode.
        
    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If weights cannot be loaded or dimensions mismatch.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model checkpoint not found at: '{path.resolve()}'. "
            "Please ensure 'best_ann.pth' is placed inside the 'models/' folder."
        )

    try:
        # Load state dictionary securely on CPU
        state_dict = torch.load(path, map_location=device, weights_only=False)
        
        # If saved directly as model or wrapped in dict
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model = TrafficSignANN(in_features=50, hidden1=100, hidden2=50, num_classes=10)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load ANN model from '{path}': {e}") from e


def load_pca(pca_path: Union[str, Path] = "models/pca_ann.pkl") -> PCA:
    """
    Load the fitted Scikit-learn PCA transformer from disk using joblib.
    
    Args:
        pca_path (str | Path): Path to `pca_ann.pkl`.
        
    Returns:
        PCA: Fitted PCA instance with 50 components.
        
    Raises:
        FileNotFoundError: If the PCA file does not exist.
        RuntimeError: If loading fails or components do not match expectations.
    """
    path = Path(pca_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"PCA transformer file not found at: '{path.resolve()}'. "
            "Please ensure 'pca_ann.pkl' is placed inside the 'models/' folder."
        )

    try:
        pca = joblib.load(path)
        
        # Validate transformer properties
        n_components = getattr(pca, "n_components", None) or getattr(pca, "n_components_", None)
        n_features = getattr(pca, "n_features_in_", None)
        
        if n_components != 50:
            raise ValueError(
                f"Expected PCA with 50 components, but loaded PCA has {n_components}."
            )
        if n_features is not None and n_features != 1024:
            raise ValueError(
                f"Expected PCA trained on 1024 input features (32x32 grayscale), but found {n_features}."
            )

        return pca
    except Exception as e:
        raise RuntimeError(f"Failed to load PCA transformer from '{path}': {e}") from e


def get_model_and_pca(
    model_path: Union[str, Path] = "models/best_ann.pth",
    pca_path: Union[str, Path] = "models/pca_ann.pkl",
    device: str = "cpu"
) -> Tuple[TrafficSignANN, PCA]:
    """
    Load and return both the ANN model and the PCA transformer.
    """
    model = load_model(model_path, device=device)
    pca = load_pca(pca_path)
    return model, pca

