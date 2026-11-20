"""Preprocessing builders shared by Logistic Regression and Random Forest.

Important: place this preprocessor inside an sklearn Pipeline with the estimator.
Calling fit on the full dataset before splitting would leak validation/test information.
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .feature_config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
except ImportError:  # Supports running this module directly from code/
    from feature_config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def make_preprocessor(scale_numeric: bool):
    """Build a column-aware preprocessor.

    Logistic Regression: scale_numeric=True.
    Random Forest: scale_numeric=False (tree splits do not require scaling).
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
