import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import pickle
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional, Dict, Any
import warnings
# Import modules from VaganBoost

from .data_preprocessor import DataPreprocessor
#from .lgbm_classifier import main as lgbm_pipeline_main

#from .data_preprocessor import DataPreprocessor  # Standardized preprocessing
#from .lgbm_regressor import train_lgbm_pipeline  # Custom LGBM pipeline
from .utils import DecompositionSwitcher  # PCA/LDA/TruncatedSVD switcher

import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import pickle
from scipy.stats import randint as sp_randint, uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional, Dict, Any
import warnings



import argparse
import os
import numpy as np
import pandas as pd
import joblib
import warnings
import logging
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor
from imblearn.pipeline import Pipeline as imbPipeline

# Import modules from VaganBoost package
from .data_preprocessor import DataPreprocessor
from .lgbm_regressor import train_lgbm_pipeline  # Custom LGBM pipeline
from .utils import DecompositionSwitcher  # PCA/LDA/TruncatedSVD switcher

# Suppress CUDA and LightGBM warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
logging.getLogger('lightgbm').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')


def tune_lgbm_regressor(input_path: str, output_path: str, dim_reducer="pca", n_iter=20):
    """
    Tune hyperparameters for an LGBMRegressor using a full pipeline with preprocessing,
    feature selection, and dimensionality reduction (PCA or SVD). Saves the tuned model
    and tuning results to the specified output directory.
    
    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Directory where tuning results and the best model will be saved.
        dim_reducer (str): Dimensionality reduction method to use; options: "pca" or "svd".
        n_iter (int): Number of iterations for RandomizedSearchCV.
    """
    os.makedirs(output_path, exist_ok=True)

    # Load dataset and separate features/target
    df = pd.read_csv(input_path)
    feature_columns = [col for col in df.columns if col != "label"]
    target_column = "label"

    # Prepare data using the package’s DataPreprocessor
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df, feature_columns, target_column)

    # Set up the dimensionality reducer
    if dim_reducer.lower() == "pca":
        reducer = DecompositionSwitcher(estimator=PCA(n_components=20))
    elif dim_reducer.lower() == "svd":
        reducer = DecompositionSwitcher(estimator=TruncatedSVD(n_components=20))
    else:
        raise ValueError("Invalid dim_reducer. For regression, choose 'pca' or 'svd'.")

    # Build the pipeline (note: SMOTE is omitted for regression)
    pipeline = imbPipeline([
        ('scaler', RobustScaler()),
        ('feature_selector', SelectKBest(mutual_info_regression, k=20)),
        ('dim_reducer', reducer),
        ('regressor', LGBMRegressor(
            objective='regression',
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        ))
    ])

    # Define the hyperparameter search space
    param_dist = {
        'dim_reducer__estimator': [PCA(), TruncatedSVD()],
        'regressor__num_leaves': [127, 255],
        'feature_selector__k': [500, 1000, 1500],
        'regressor__learning_rate': [0.01, 0.1],
        'regressor__n_estimators': [100, 300, 500],
        'regressor__max_depth': [7, 10, 15]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Evaluate best model on the test set
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Best parameters:", search.best_params_)
    print("Best CV score (neg MSE):", search.best_score_)
    print("Test MSE:", mse)
    print("Test R^2:", r2)

    # Save best model and tuning results
    joblib.dump(best_model, os.path.join(output_path, "tuned_model.joblib"))
    results_df = pd.DataFrame(search.cv_results_)
    results_df.to_csv(os.path.join(output_path, "tuning_results.csv"), index=False)
    with open(os.path.join(output_path, "tuning_summary.txt"), "w") as f:
        f.write(f"Best Parameters: {search.best_params_}\n")
        f.write(f"Best CV Score (neg MSE): {search.best_score_}\n")
        f.write(f"Test MSE: {mse}\n")
        f.write(f"Test R^2: {r2}\n")
    print(f"Tuning results saved in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune hyperparameters for LGBMRegressor")
    parser.add_argument("--input_path", required=True, help="Path to input CSV file")
    parser.add_argument("--output_path", required=True, help="Output directory for tuning results")
    parser.add_argument("--dim_reducer", default="pca", help="Dimensionality reduction method: 'pca' or 'svd'")
    parser.add_argument("--n_iter", type=int, default=20, help="Number of iterations for RandomizedSearchCV")
    args = parser.parse_args()

    tune_lgbm_regressor(args.input_path, args.output_path, dim_reducer=args.dim_reducer, n_iter=args.n_iter)
