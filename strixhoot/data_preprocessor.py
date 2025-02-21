import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from typing import Tuple, List, Optional, Dict

class DataPreprocessor:
    """
    Handles preprocessing for numerical and categorical data, ensuring consistency across training/testing.
    """

    def __init__(self, scaler_type: str = 'robust', test_size: float = 0.2, random_state: int = 42):
        """
        Initialize data preprocessor.

        Args:
            scaler_type (str): Type of feature scaling ('standard' or 'robust').
            test_size (float): Fraction of data to allocate to test set.
            random_state (int): Seed for reproducibility.
        """
        self.scaler_type = scaler_type
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = None

    def _select_scaler(self):
        """Select and initialize the scaler based on scaler_type."""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError("Invalid scaler type. Choose 'standard' or 'robust'.")

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with the median for numerical columns."""
        return df.fillna(df.median(numeric_only=True))

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns using LabelEncoder."""
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        return df

    def prepare_data(self, df: pd.DataFrame, feature_columns: List[str], target_column: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess data: handle missing values, encode categorical features, scale numerical features.

        Args:
            df (pd.DataFrame): Raw data.
            feature_columns (List[str]): List of feature column names.
            target_column (str): Target column name.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Scaled X_train, X_test, y_train, y_test.
        """
        self.feature_columns = feature_columns
        self.target_column = target_column

        # Handle missing values
        df = self._handle_missing_values(df)

        # Encode categorical variables (if any)
        df = self._encode_categorical(df)

        # Extract features and target
        X = df[feature_columns].values
        y = df[target_column].values

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Scale features
        self.scaler = self._select_scaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Reverse transform scaled features back to original scale.

        Args:
            X_scaled (np.ndarray): Scaled data.

        Returns:
            np.ndarray: Unscaled data.
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted. Call `prepare_data()` first.")
        return self.scaler.inverse_transform(X_scaled)

    def save_scaler(self, filepath: str) -> None:
        """Save the trained scaler."""
        import joblib
        joblib.dump(self.scaler, filepath)

    def load_scaler(self, filepath: str) -> None:
        """Load a previously saved scaler."""
        import joblib
        self.scaler = joblib.load(filepath)
