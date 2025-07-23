import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabpfn import TabPFNRegressor

logger = logging.getLogger(__name__)

MODEL_PORTFOLIO = {
    "lr": LinearRegression,
    "rf": RandomForestRegressor,
    "tabpfn": TabPFNRegressor,
}

SEARCH_SPACES = {
    "lr": {},
    "rf": {
        "n_estimators": [50, 100, 200],
        "max_depth":   [None, 5, 10],
    },
    "tabpfn": {},
}


def extract_meta_features(X):
    n_rows, n_cols = X.shape

    meta_features = {
        "n_rows": n_rows,
        "n_cols": n_cols,
    }

    logger.debug(f"extracted meta features: {meta_features}")

    return meta_features


def select_portfolio(meta_features, portfolio = MODEL_PORTFOLIO):
    chosen_portfolio = list(portfolio.keys())

    # tabpfn is not suitable for big datasets
    if meta_features["n_rows"] > 999:
        chosen_portfolio.remove("tabpfn")

    logger.debug(f"chosen portfolio: {chosen_portfolio}")

    return chosen_portfolio


def build_pipeline(model_name: str, X: pd.DataFrame) -> Pipeline:
    numeric_cols = [c for c in X.columns if X[c].dtype.name not in ("object", "category", "bool")]
    categorical_cols = [c for c in X.columns if X[c].dtype.name in ("object", "category", "bool")]

    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    model = MODEL_PORTFOLIO[model_name]
    estimator = model()

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", estimator),
        ]
    )

    return pipeline


def fit_model(model_name: str, X: pd.DataFrame, y: pd.Series):
    pipe = build_pipeline(model_name, X)
    pipe.fit(X, y)

    logger.info(f"fitted model: {model_name}")

    return pipe


def aggregate_predictions(preds_list):
    aggregated_predictions = np.mean(preds_list, axis=0)

    logger.debug(f"aggregated predictions: {aggregated_predictions.shape}")

    return aggregated_predictions

