import logging
from typing import Any, Callable, Dict, Optional

import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .model_space import MODEL_PORTFOLIO
from .pipeline import build_pipeline
from .utils import get_default_constant

logger = logging.getLogger(__name__)


def hpo_search(
    model_cls: Callable[..., BaseEstimator],
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    search_space: Dict[str, Any],
    method: str = "random",  # options: random, sh, hyperband
    fidelity_param: str = "n_estimators",
    seed: int = 7,
):
    """
    run HPO with Optuna using the given strategy and search space.
    """
    if method == "random":
        pruner = None
    elif method == "sh":
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    elif method == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        raise ValueError(f"Unknown HPO method: '{method}'")

    return _run_optuna(
        model_cls,  # callable model
        model_name,  # name of model for the study
        X,
        y,
        search_space,
        seed,
        pruner,
        fidelity_param,
        method,
    )


# TODO: CURRENTLY DOES NO PRUNUNG LOGIC IS WRONG
def _run_optuna(
    model_cls: Callable[..., BaseEstimator],
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    search_space: Dict[str, Any],
    seed: int,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    fidelity_param: str = "n_estimators",
    method: str = "optuna",
    n_trials: int = 30,
):
    logger = logging.getLogger(__name__)
    test_size = get_default_constant("TEST_SIZE")

    # fallback: get name from class if not provided
    if model_name is None:
        model_name = next(
            (k for k, v in MODEL_PORTFOLIO.items() if v == model_cls),
            model_cls.__name__,
        )

    def objective(trial):
        # sample parameters
        params = {}
        for param, spec in search_space.items():
            if param == "__static__":
                continue
            # Handle new-style dict specs with condition
            if isinstance(spec, dict):
                if "condition" in spec:
                    # skip if condition not satisfied
                    cond_key, cond_val = list(spec["condition"].items())[0]
                    if params.get(cond_key) != cond_val:
                        logger.debug(
                            f"Skipping {param} due to condition {cond_key}={cond_val}"
                        )
                        continue
                param_type = spec["type"]
                bounds = spec["bounds"]
            elif isinstance(spec, tuple):
                if isinstance(spec[0], str) and spec[0] in {
                    "int",
                    "float",
                    "float_log",
                    "categorical",
                }:
                    param_type = spec[0]

                    # --- fix starts here ---
                    if param_type == "categorical":
                        bounds = spec[1]  # NOT spec[1:]
                    else:
                        bounds = spec[1:]
                    # --- fix ends here ---
            else:
                param_type = "categorical"
                bounds = spec

            # Suggest based on type
            if param_type == "float_log":
                params[param] = trial.suggest_float(param, *bounds, log=True)
            elif param_type == "float":
                params[param] = trial.suggest_float(param, *bounds)
            elif param_type == "int":
                params[param] = trial.suggest_int(param, *bounds)
            elif param_type == "categorical":
                params[param] = trial.suggest_categorical(param, bounds)
            else:
                raise ValueError(f"Unknown param type: {param_type}")
        # build base pipeline (with dummy model)
        pipe = build_pipeline(model_name, X)

        # inject tuned model
        tuned_model = model_cls(**params)
        pipe.steps[-1] = ("model", tuned_model)

        # split and fit
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        score = r2_score(y_val, preds)

        trial.report(score, step=trial.number)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return score

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name=f"{model_name}_{method}_hpo",
    )

    logger.info(f"[{model_name}] Starting HPO ({method}) with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    logger.info(f"[{model_name}] Best trial: {best_params}")

    # retrain best model on full data
    best_model = model_cls(**best_params)
    pipe = build_pipeline(model_name, X)
    pipe.steps[-1] = ("model", best_model)
    pipe.fit(X, y)

    return pipe
