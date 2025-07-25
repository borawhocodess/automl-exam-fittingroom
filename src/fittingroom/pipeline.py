import logging

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tabpfn import TabPFNRegressor

logger = logging.getLogger(__name__)

MODEL_PORTFOLIO = {
    "linear": LinearRegression,
    "knn": None,
    "rf": RandomForestRegressor,
    "xgboost": None,
    "lightgbm": None,
    "catboost": None,
    "tabpfn": TabPFNRegressor,
    "nanotabpfn": None,
    "tabdpt": None,
    "realmlp": None,
    "modern_nca": None,
    "tabm": None,
}

SEARCH_SPACES = {
    "linear": {},
    "knn": {},
    "rf": {},
    "xgboost": {},
    "lightgbm": {},
    "catboost": {},
    "tabpfn": {},
    "nanotabpfn": {},
    "tabdpt": {},
    "realmlp": {},
    "modern_nca": {},
    "tabm": {},
}


def select_portfolio(
    meta_features,
    portfolio=MODEL_PORTFOLIO,
    ask_expert_opinion: bool = False,
):
    """
    Select a subset of models based on meta-features.
    If ask_expert_opinion is True, prompt before each modification.
    """
    portfolio_to_choose = {k: v for k, v in portfolio.items() if v is not None}

    logger.debug(f"available portfolio: {list(portfolio_to_choose.keys())}")

    chosen = set(portfolio_to_choose)

    n = meta_features.get("n_instances", 0)
    pct_m = meta_features.get("pct_missing", 0.0)
    num = meta_features.get("n_numeric", 0)
    cat = meta_features.get("n_categorical", 0)
    ratio = num / cat if cat else np.inf
    skew = abs(meta_features.get("target_skew", 0.0))

    # TODO: remove later, for pipeline to work on CPU for now
    if not torch.cuda.is_available() and n > 999:
        chosen.remove("tabpfn")

    def confirm(action_desc: str) -> bool:
        if not ask_expert_opinion:
            return True
        ans = input(f"{action_desc} Proceed? (y/n): ").strip().lower()
        return ans.startswith("y")  # TODO: is everything else no?

    # Rule 1: Drop tabpfn on large datasets
    if n > 10_000 and "tabpfn" in chosen:
        if confirm(f"Dataset has {n} instances, >10k — remove 'tabpfn'?"):
            chosen.remove("tabpfn")

            logger.debug("Removed tabpfn due to large n_instances")

    # Rule 2: Very high-dimensional -> drop MLPs/KNN TODO: WHY?
    if meta_features.get("n_features", 0) > 1000:
        for m in ["realmlp", "modern_nca", "knn"]:
            if m in chosen and confirm(f"Too many features — remove '{m}'?"):
                chosen.remove(m)

                logger.debug(f"Removed {m} due to high n_features")

    # Rule 3: High missingness -> keep only tree-based TODO: WHY?
    if pct_m > 0.2:
        keep = {"catboost", "lightgbm", "xgboost", "randomforest"}
        to_drop = chosen - keep
        for m in list(to_drop):
            if confirm(f"Missingness {pct_m:.2%} >20% — drop '{m}'?"):
                chosen.discard(m)

                logger.debug(f"Removed {m} due to high missingness")

    # Rule 4: Mostly categorical -> drop pure numeric learners TODO: WHY?
    if ratio < 0.5:
        for m in ["realmlp", "modern_nca", "linear", "knn"]:
            if m in chosen and confirm(
                f"Categorical ratio high (num/cat={ratio:.2f}) — drop '{m}'?"
            ):
                chosen.remove(m)

                logger.debug(f"Removed {m} due to high categorical proportion")

    # Rule 5: Highly skewed target -> drop linear/MLP TODO: WHY?
    if skew > 1.0:
        for m in ["linear", "realmlp", "modern_nca"]:
            if m in chosen and confirm(f"Target skew {skew:.2f} >1.0 — drop '{m}'?"):
                chosen.remove(m)

                logger.debug(f"Removed {m} due to high target skew")

    # Ensure at least one per family remains TODO: WHY?
    families = {
        "tree": {"lightgbm", "catboost", "xgboost", "randomforest"},
        "mlp": {"realmlp", "modern_nca", "knn"},  # TODO: is knn mlp?
        "linear": {"linear"},
        # TODO: why tabular models are not in families?
    }
    for fam, models in families.items():
        if not (chosen & models):
            fallback = next(iter(models))
            if confirm(f"No {fam} model left — add fallback '{fallback}'?"):
                # control if it id not none in portfolio
                if fallback in portfolio_to_choose:
                    chosen.add(fallback)

                    logger.debug(f"Added fallback {fallback} for family {fam}")

    chosen_portfolio = list(chosen)

    logger.debug(f"chosen portfolio: {chosen_portfolio}")

    return chosen_portfolio


def build_pipeline(model_name: str, X: pd.DataFrame) -> Pipeline:
    numeric_cols = [
        c for c in X.columns if X[c].dtype.name not in ("object", "category", "bool")
    ]
    categorical_cols = [
        c for c in X.columns if X[c].dtype.name in ("object", "category", "bool")
    ]

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
