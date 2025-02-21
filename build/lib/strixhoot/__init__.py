"""
VAE-GAN Boost Keras & TF: Hybrid Generative Modeling with LightGBM Integration
"""

from .data_preprocessor import DataPreprocessor
from .cvae import sampling, CVAE, train_cvae
from .cgan import build_generator, build_discriminator, train_cgan
from .lgbm_regressor import train_lgbm_pipeline
from .lgbm_tuner import tune_lgbm_regressor

from .trainer import main


from .utils import (
    DecompositionSwitcher,
    plot_true_vs_pred,
    plot_residuals,
    save_model_artifacts,
    load_model_artifacts,
    create_directory,
    set_random_seeds
)

#from .config import DEFAULT_PARAMS

__version__ = "0.9.5"
__all__ = [
    'DataPreprocessor',
    'sampling',
    'CVAE',
    'train_cvae',
    'build_generator',
    'build_discriminator',
    'train_cgan',
    'train_lgbm_pipeline',
    'tune_lgbm_regressor',
    'main',
    'DecompositionSwitcher',
    'plot_true_vs_pred',
    'plot_residuals',
    'save_model_artifacts',
    'load_model_artifacts',
    'create_directory',
    'set_random_seeds'
]