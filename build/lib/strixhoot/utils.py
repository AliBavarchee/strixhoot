import os
import json
import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Union, Optional, List
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from tensorflow.keras.models import Model
import lightgbm as lgb
from .data_preprocessor import DataPreprocessor
import random
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA, TruncatedSVD
import lightgbm as lgb
import tensorflow as tf

# Visualization settings
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
COLORS = sns.color_palette()

__all__ = [
    'plot_true_vs_pred',
    'plot_residuals',
    'save_model_artifacts',
    'load_model_artifacts',
    'create_directory',
    'set_random_seeds',
    'DecompositionSwitcher'
]

class DecompositionSwitcher(BaseEstimator, TransformerMixin):
    """
    A simple transformer to wrap a dimensionality reduction estimator.
    This allows the pipeline to switch between, for example, PCA and TruncatedSVD.
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        return self.estimator.transform(X)

    def fit_transform(self, X, y=None):
        return self.estimator.fit_transform(X, y)

def plot_true_vs_pred(y_true, y_pred, output_path):
    """
    Plot a scatter plot of true versus predicted values.
    
    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        output_path (str): File path where the plot will be saved.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted Values")
    plt.savefig(output_path)
    plt.close()

def plot_residuals(y_true, y_pred, output_path):
    """
    Plot the residuals (true - predicted) to diagnose regression performance.
    
    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        output_path (str): File path where the residuals plot will be saved.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.savefig(output_path)
    plt.close()

def save_model_artifacts(model, output_path, filename="model.joblib"):
    """
    Save model artifacts (e.g., a trained model) to a specified file.
    
    Args:
        model: The model or artifact to be saved.
        output_path (str): Directory where the model artifact will be saved.
        filename (str): The name of the file to save the artifact.
    """
    create_directory(output_path)
    file_path = os.path.join(output_path, filename)
    joblib.dump(model, file_path)
    print(f"Model artifacts saved to {file_path}")

def load_model_artifacts(file_path):
    """
    Load model artifacts from a specified file.
    
    Args:
        file_path (str): Path to the saved model artifact.
        
    Returns:
        The loaded model or artifact.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    model = joblib.load(file_path)
    print(f"Model artifacts loaded from {file_path}")
    return model

def create_directory(directory):
    """
    Create a directory if it does not already exist.
    
    Args:
        directory (str): The path of the directory to create.
    """
    os.makedirs(directory, exist_ok=True)

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): The seed value to be set.
    """
    np.random.seed(seed)
    random.seed(seed)
    # Additional libraries (e.g., TensorFlow, PyTorch) can have their seeds set here.
    print(f"Random seeds set to {seed}")

