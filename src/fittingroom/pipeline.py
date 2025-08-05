import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fittingroom.model_space import MODEL_PORTFOLIO, SEARCH_SPACES

logger = logging.getLogger(__name__)


def select_portfolio(
    meta_features, portfolio=MODEL_PORTFOLIO, ask_expert_opinion: bool = False
) -> list:
    """
    Select a subset of models based on meta-features.
    If ask_expert_opinion is True, prompt before each modification.
    """
    portfolio_to_choose = {k: v for k, v in portfolio.items() if v is not None}

    logger.info(f"Available models: {list(portfolio_to_choose.keys())}")

    hack = list(portfolio_to_choose.keys())

    return hack

    chosen = set(portfolio_to_choose)
    n = meta_features.get("n_instances", 0)
    pct_m = meta_features.get("pct_missing", 0.0)
    num = meta_features.get("n_numeric", 0)
    cat = meta_features.get("n_categorical", 0)
    ratio = num / cat if cat else np.inf
    skew = abs(meta_features.get("target_skew", 0.0))

    def confirm(action_desc: str) -> bool:
        if not ask_expert_opinion:
            return True
        ans = input(f"{action_desc} Proceed? (y/n): ").strip().lower()
        return ans.startswith("y")

    # Rule 1: Drop tabpfn on large datasets
    if n > 10_000 and "tabpfn" in chosen:
        if confirm(f"Dataset has {n} instances, >10k — remove 'tabpfn'?"):
            chosen.remove("tabpfn")
            logger.debug("Removed tabpfn due to large n_instances")

    # Rule 2: Very high-dimensional -> drop MLPs/KNN
    if meta_features.get("n_features", 0) > 1000:
        for m in ["realmlp", "modern_nca", "knn"]:
            if m in chosen and confirm(f"Too many features — remove '{m}'?"):
                chosen.remove(m)
                logger.debug(f"Removed {m} due to high n_features")

    # Rule 3: High missingness -> keep only tree-based
    if pct_m > 0.2:
        keep = {"catboost", "lightgbm", "xgboost", "randomforest", "extratrees"}
        to_drop = chosen - keep
        for m in list(to_drop):
            if confirm(f"Missingness {pct_m:.2%} >20% — drop '{m}'?"):
                chosen.discard(m)
                logger.debug(f"Removed {m} due to high missingness")

    # Rule 4: Mostly categorical -> drop pure numeric learners
    if ratio < 0.5:
        for m in ["realmlp", "modern_nca", "linear", "knn"]:
            if m in chosen and confirm(
                f"Categorical ratio high (num/cat={ratio:.2f}) — drop '{m}'?"
            ):
                chosen.remove(m)
                logger.debug(f"Removed {m} due to high categorical proportion")

    # Rule 5: Highly skewed target -> drop linear/MLP
    if skew > 1.0:
        for m in ["linear", "realmlp", "modern_nca"]:
            if m in chosen and confirm(f"Target skew {skew:.2f} >1.0 — drop '{m}'?"):
                chosen.remove(m)
                logger.debug(f"Removed {m} due to high target skew")

    # Ensure at least one per family remains
    families = {
        "tree": {"lightgbm", "catboost", "xgboost", "randomforest"},
        "mlp": {"realmlp", "modern_nca", "knn"},
        "linear": {"linear"},
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


def build_pipeline(model_name: str, X: pd.DataFrame, params: dict = None) -> Pipeline:
    """
    Builds a sklearn pipeline with preprocessing and model instantiation.

    Args:
        model_name (str): The name of the model from MODEL_PORTFOLIO.
        X (pd.DataFrame): The feature data used to determine column types.
        params (dict, optional): Parameters to pass to the model constructor.

    Returns:
        sklearn.Pipeline: The preprocessing + model pipeline.
    """
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

    model_cls = MODEL_PORTFOLIO[model_name]
    # Always include model-specific static defaults (e.g. LightGBM verbosity)
    static_params = SEARCH_SPACES.get(model_name, {}).get("__static__", {})
    combined_params = {**static_params, **(params or {})}
    estimator = model_cls(**combined_params)

    pipeline = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", estimator),
        ]
    )

    return pipeline


def fit_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    hpo_method: str = "random",  # can be "random", "hyperband", "sh"
    seed: int = 0,
):
    # doing this here to avoid circular import issues
    from .hpo import hpo_search

    model_cls = MODEL_PORTFOLIO[model_name]
    search_space = SEARCH_SPACES.get(model_name, {})

    if not search_space:
        # No hyperparameter optimization: just fit the pipeline normally
        pipe = build_pipeline(model_name, X)
        pipe.fit(X, y)
        logger.info(f"fitted model without HPO: {model_name}")
        return pipe

    if model_name in ["tabm", "realmlp"]:
        return hpo_search(model_cls, model_name, X, y, search_space, hpo_method, seed)

    # Run HPO
    params = hpo_search(
        model_cls=model_cls,
        model_name=model_name,
        X=X,
        y=y,
        search_space=search_space,
        method=hpo_method,
        seed=seed,
    )
    logger.debug(f"best params found from hpo search: {params}")
    pipeline = build_pipeline(model_name, X, params)
    pipeline.fit(X, y)
    pipeline._hpo_params = params
    return pipeline


def aggregate_predictions(preds_list, weights=None) -> pd.Series:
    """
    weighted mean
    """

    preds_arr = np.vstack(preds_list)

    if weights is None:
        return preds_arr.mean(axis=0)

    weights = np.asarray(weights, dtype=float)

    if weights.ndim != 1 or len(weights) != len(preds_list):
        raise ValueError("...")

    weights = np.maximum(weights, 0)

    if weights.sum() == 0:
        weights = np.ones_like(weights)

    weights = weights / weights.sum()

    aggregated_predictions = np.average(preds_arr, axis=0, weights=weights)

    logger.debug(f"aggregated predictions: {aggregated_predictions.shape}")

    return aggregated_predictions
