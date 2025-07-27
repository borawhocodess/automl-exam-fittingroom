import logging
from typing import Any, Callable, Dict, Optional

import optuna
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

from .model_space import MODEL_PORTFOLIO
from .pipeline import build_pipeline
from .utils import get_default_constant
from tabpfn import TabPFNRegressor

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
    elif method == "bayesian":
        return _run_bo(model_cls, model_name, X, y, search_space, seed, method=method)
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

    return


def _run_bo(
    model_cls: Callable[..., BaseEstimator],
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    search_space: Dict[str, Any],
    seed: int,
    method: str = "bayesian",
    n_trials: int = 20,
):

    logger = logging.getLogger(__name__)
    print("A")
    def decode_param_dict(x_dict, search_space):
        # Reverse one-hot encoding to original param dict
        decoded = {}
        for param, spec in search_space.items():
            if param == "__static__":
                continue
            if isinstance(spec, dict):
                if "condition" in spec:
                    cond_key, cond_val = list(spec["condition"].items())[0]
                    if decoded.get(cond_key) != cond_val:
                        continue
                param_type = spec["type"]
                bounds = spec["bounds"]
            else:
                param_type = "categorical"
                bounds = spec
            if param_type == "categorical":
                prefix = param + "_"
                found = False
                for b in bounds:
                    col = f"{param}_{b}"
                    if col in x_dict and x_dict[col] > 0.5:
                        decoded[param] = b
                        found = True
                        break
                if not found and param in x_dict:
                    decoded[param] = x_dict[param]
            else:
                if param in x_dict:
                    decoded[param] = x_dict[param]
        return decoded

    test_size = get_default_constant("TEST_SIZE")
    init_samples = 5

    print(f"Running Bayesian HPO for {model_name} with {n_trials} trials...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")
    def sample_params():
        params = {}
        for param, spec in search_space.items():
            if param == "__static__":
                continue
            if isinstance(spec, dict):
                if "condition" in spec:
                    cond_key, cond_val = list(spec["condition"].items())[0]
                    if params.get(cond_key) != cond_val:
                        continue
                param_type = spec["type"]
                bounds = spec["bounds"]
            else:
                param_type = "categorical"
                bounds = spec

            if param_type == "float_log":
                params[param] = np.exp(
                    np.random.uniform(np.log(bounds[0]), np.log(bounds[1]))
                )
            elif param_type == "float":
                params[param] = np.random.uniform(bounds[0], bounds[1])
            elif param_type == "int":
                params[param] = np.random.randint(bounds[0], bounds[1] + 1)
            elif param_type == "categorical":
                params[param] = np.random.choice(bounds)
        return params

    # Initial random samples
    param_list = [sample_params() for _ in range(init_samples)]
    score_list = []
    
    for p in param_list:
        print("A")
        pipe = build_pipeline(model_name, X_train)
        print("B")
        model = model_cls(**p)
        print("C")
        pipe.steps[-1] = ("model", model)
        print("D")
        print(f"Trying params: {p}")
        pipe.fit(X_train, y_train)
        print("E")
        preds = pipe.predict(X_val)
        print("F")
        score = r2_score(y_val, preds)
        print("G")
        score_list.append(score)

    print(f"Initial samples: {len(param_list)}, scores: {score_list}")
    X_meta = pd.DataFrame(param_list).fillna("None")
    X_meta = pd.get_dummies(X_meta).astype(np.float32)
    y_meta = np.array(score_list)
    print(f"Initial meta features shape: {X_meta.shape}, scores: {y_meta.shape}")
    print(X_meta)
    print(y_meta)
    reg = TabPFNRegressor(device="cpu", fit_mode="low_memory")
    
    print(X_meta.shape, y_meta.shape)
    reg.fit(X_meta, y_meta)
    print("Fitted TabPFNRegressor on initial samples")

    def acq_func(x_np):
        x_df = pd.DataFrame([x_np], columns=X_meta.columns)
        pred = reg.predict(
            x_df.values, output_type="quantiles", quantiles=[0.1, 0.5, 0.9]
        )
        mean, std = pred[1], (pred[2] - pred[0]) / 2
        eta = max(y_meta)
        z = (mean - eta) / std if std > 1e-8 else 0.0
        from scipy.stats import norm

        return -(std * (z * norm.cdf(z) + norm.pdf(z)))

    best_score = -np.inf
    best_param = None

    for _ in range(n_trials - init_samples):
        x0 = X_meta.iloc[np.random.randint(len(X_meta))].values
        res = minimize(acq_func, x0, method="L-BFGS-B")
        x_next = res.x
        x_dict = dict(zip(X_meta.columns, x_next))
        x_decoded = decode_param_dict(x_dict, search_space)

        pipe = build_pipeline(model_name, X_train)
        model = model_cls(**x_decoded)
        pipe.steps[-1] = ("model", model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        score = r2_score(y_val, preds)

        param_list.append(x_decoded)
        score_list.append(score)

        X_meta = pd.DataFrame(param_list).fillna("None")
        X_meta = pd.get_dummies(X_meta).astype(np.float32)
        y_meta = np.array(score_list)
        reg.fit(X_meta, y_meta)

        if score > best_score:
            best_score = score
            best_param = x_decoded

    logger.info(f"Best score with TabPFN-BO: {best_score}")
    logger.info(f"Best params: {best_param}")
