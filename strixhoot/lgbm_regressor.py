import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA, TruncatedSVD
import umap
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold  # <

import joblib
from joblib import Memory, dump
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, label_binarize
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from lightgbm import LGBMClassifier
from pathlib import Path
from imblearn.pipeline import Pipeline as imbPipeline

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA, TruncatedSVD
import umap
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import joblib
from joblib import Memory, dump
from sklearn.preprocessing import RobustScaler, PolynomialFeatures, label_binarize
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
#from lightgbm import LGBMClassifier
from pathlib import Path
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
# Import from VaganBoost package
from .utils import DecompositionSwitcher, plot_true_vs_pred, plot_residuals, save_model_artifacts, load_model_artifacts, create_directory, set_random_seeds
from .data_preprocessor import DataPreprocessor

# CUDA Warning Suppression Edition
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

import logging
logging.getLogger('lightgbm').setLevel(logging.WARNING)
import warnings
from numba import cuda
try:
    cuda.close()
except:
    pass
warnings.filterwarnings('ignore')



def train_lgbm_pipeline(input_path: str, output_path: str, dim_reducer="pca"):
    """
    Train an LGBMRegressor using a full pipeline with preprocessing, feature selection,
    and dimensionality reduction (PCA or SVD). The script also extracts per-feature weights.
    
    Args:
        input_path: Path to the input CSV file.
        output_path: Directory where the model and results will be saved.
        dim_reducer: Dimensionality reduction method; options: "pca" or "svd".
    """
    os.makedirs(output_path, exist_ok=True)

    # Load dataset and separate features/target
    df = pd.read_csv(input_path)
    feature_columns = [col for col in df.columns if col != "label"]
    target_column = "label"

    # Prepare data using the package’s DataPreprocessor
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df, feature_columns, target_column)
    
    # Set up the dimensionality reducer based on the provided option
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
    
    # Define hyperparameter search space
    param_dist = {
        'dim_reducer__estimator': [PCA(), TruncatedSVD()],
        'regressor__num_leaves': [127, 255],
        'feature_selector__k': [500, 1000, 1500],
        'regressor__learning_rate': [0.01, 0.1],
        'regressor__n_estimators': [100, 300],
        'regressor__max_depth': [7, 10]
    }
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    final_model = search.best_estimator_

    # Save the best model
    joblib.dump(final_model, os.path.join(output_path, "optimized_model.joblib"))

    # ==== Regression Evaluation ====
    y_pred = final_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    report_text = f"Mean Squared Error: {mse:.4f}\nR^2 Score: {r2:.4f}\n"
    with open(os.path.join(output_path, "regression_report.txt"), "w") as f:
        f.write(report_text)
    
    # Plot true versus predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted Values")
    plt.savefig(os.path.join(output_path, "true_vs_pred.png"))
    plt.close()
    
    # ==== Feature Weight Extraction ====
    # Retrieve components from the pipeline
    feature_selector = final_model.named_steps['feature_selector']
    dim_reducer_estimator = final_model.named_steps['dim_reducer'].estimator
    regressor = final_model.named_steps['regressor']
    booster = regressor.booster_

    # Get original feature names from the input CSV
    original_df = pd.read_csv(input_path)
    original_features = [col for col in original_df.columns if col != "label"]
    
    # Get indices of selected features
    selected_mask = feature_selector.get_support()
    selected_indices = np.where(selected_mask)[0]
    
    # Extract decomposition components from the dim_reducer estimator
    components = dim_reducer_estimator.components_
    
    n_components = components.shape[0]
    n_estimators = regressor.n_estimators
    # For regression, we have a single output
    n_outputs = 1
    feature_weights = np.zeros((n_outputs, len(original_features)))
    
    # Sum feature importances from all trees in the booster
    importance_sum = np.zeros(booster.num_feature())
    for it in range(n_estimators):
        imp = booster.feature_importance(importance_type='gain', iteration=it)
        importance_sum += imp
    if len(importance_sum) != n_components:
        raise ValueError("Dimension mismatch between decomposition components and feature importance")
    
    selected_importances = np.dot(importance_sum, components)
    feature_weights[0, selected_indices] = selected_importances
    
    weights_df = pd.DataFrame(
        feature_weights,
        columns=original_features,
        index=["Regression"]
    )
    weights_df.to_csv(os.path.join(output_path, "feature_weights_per_regression.csv"), index=False)
    print(f"Feature weights saved to 'feature_weights_per_regression.csv' in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM for regression with a full pipeline")
    parser.add_argument("--input_path", required=True, help="Path to input CSV file")
    parser.add_argument("--output_path", required=True, help="Output directory for model and results")
    parser.add_argument("--dim_reducer", default="pca", help="Dimensionality reduction method: 'pca' or 'svd'")
    args = parser.parse_args()

    train_lgbm_pipeline(args.input_path, args.output_path, dim_reducer=args.dim_reducer)
