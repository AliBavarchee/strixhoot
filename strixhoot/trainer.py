#!pip instrall dill
import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import callbacks
from typing import Dict, Tuple, Optional
import pickle
from .data_preprocessor import DataPreprocessor
from sklearn.model_selection import KFold
#!pip instrall dill
import argparse
import os
import logging
import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import callbacks
from typing import Dict, Tuple, Optional
import pickle
from .data_preprocessor import DataPreprocessor
#from .cvae import CVAE
#from .cgan import CGAN
#from .lgbm_tuner import LightGBMTuner
#from .utils import plot_confusion_matrix, plot_roc_curves, plot_pr_curves

# Import generative model training functions
from .cvae import train_cvae
from .cgan import train_cgan

# Import LGBM regression training and tuning routines
from .lgbm_regressor import train_lgbm_pipeline
from .lgbm_tuner import tune_lgbm_regressor

# Import utility functions
from .utils import set_random_seeds, create_directory

def main(args):
    # Set random seeds for reproducibility
    set_random_seeds(args.seed)
    
    # Create the output directory if it doesn't exist
    create_directory(args.output_path)
    
    # --- Generative Model Training ---
    if args.train_generative:
        gen_output_path = os.path.join(args.output_path, "generative_models")
        create_directory(gen_output_path)
        if args.gen_model in ['cvae', 'both']:
            print("Training CVAE model...")
            # The CVAE training function should handle its own saving of model artifacts
            train_cvae(args.input_path, os.path.join(gen_output_path, "cvae"))
        if args.gen_model in ['cgan', 'both']:
            print("Training CGAN model...")
            # The CGAN training function should handle its own saving of model artifacts
            train_cgan(args.input_path, os.path.join(gen_output_path, "cgan"))
    else:
        print("Skipping generative model training as per configuration.")
    
    # --- LGBMRegressor Training ---
    lgbm_output_path = os.path.join(args.output_path, "lgbm")
    create_directory(lgbm_output_path)
    if args.tune_lgbm:
        print("Tuning LGBMRegressor...")
        tune_lgbm_regressor(
            input_path=args.input_path,
            output_path=lgbm_output_path,
            dim_reducer=args.dim_reducer,
            n_iter=args.n_iter
        )
    else:
        print("Training LGBMRegressor...")
        train_lgbm_pipeline(
            input_path=args.input_path,
            output_path=lgbm_output_path,
            dim_reducer=args.dim_reducer
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid Trainer for vaganboostregg: trains generative models (CVAE, CGAN) and an LGBMRegressor."
    )
    parser.add_argument("--input_path", required=True, help="Path to the input CSV file")
    parser.add_argument("--output_path", required=True, help="Output directory for models and results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gen_model", type=str, choices=["cvae", "cgan", "both", "none"], default="none", 
                        help="Which generative model to train: cvae, cgan, both, or none")
    parser.add_argument("--train_generative", action="store_true", 
                        help="If set, the trainer will train the specified generative model(s)")
    parser.add_argument("--tune_lgbm", action="store_true", 
                        help="If set, perform hyperparameter tuning for LGBMRegressor; otherwise standard training is performed")
    parser.add_argument("--dim_reducer", type=str, choices=["pca", "svd"], default="pca", 
                        help="Dimensionality reduction method for LGBMRegressor")
    parser.add_argument("--n_iter", type=int, default=20, help="Number of iterations for hyperparameter tuning")
    
    args = parser.parse_args()
    
    # Set logging level to suppress excessive LightGBM warnings
    logging.getLogger('lightgbm').setLevel(logging.WARNING)
    
    main(args)


print("Training completed! Best models saved in 'trained_models' directory.")
