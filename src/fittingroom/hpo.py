import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from fittingroom.model_space import FIDELITY_MAP
from fittingroom.utils import get_default_constant

logger = logging.getLogger(__name__)


def hpo_search(
    model_cls: Callable[..., BaseEstimator],
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    search_space: Dict[str, Any],
    method: str = "random",  # options: "random", "sh", "hyperband"
    seed: int = 7,
    n_trials: int = 30,
):
    """
    Main HPO method. Selects the pruner (and sampler), then kicks off tuning.
    """
    # choose pruner
    if method == "random":
        pruner = None
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif method == "sh":
        pruner = optuna.pruners.SuccessiveHalvingPruner()
        sampler = optuna.samplers.TPESampler(seed=seed)
    elif method == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
        sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        raise ValueError(f"Unknown HPO method: '{method}'")

    return _run_optuna(
        model_cls=model_cls,
        model_name=model_name,
        X=X,
        y=y,
        search_space=search_space,
        seed=seed,
        pruner=pruner,
        sampler=sampler,
        method=method,
        n_trials=n_trials,
    )


def _run_optuna(
    model_cls: Callable[..., BaseEstimator],
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    search_space: Dict[str, Any],
    seed: int,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    method: str = "optuna",
    n_trials: int = 30,
):
    """
    Wraps the Optuna study logic and kicks off the optimization process.
    Holds the main logic for sampling parameters and evaluating the model.
    Uses the provided search space and model class to find the best hyperparameters.
    """
    test_size = get_default_constant("TEST_SIZE")

    study_name = f"{model_name}_{method}_study"
    logger.info(f"Starting HPO for {model_name} using {method}...")

    study = optuna.create_study(
        direction="maximize",  # we maximize R^2 everywhere we report.
        pruner=pruner,
        sampler=sampler,
        study_name=study_name,
    )

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, search_space)
        score = _evaluate_model(
            trial=trial,
            params=params,
            model_cls=model_cls,
            model_name=model_name,
            X=X,
            y=y,
            seed=seed,
            test_size=test_size,
            pruner=pruner,
        )
        return score

    study.optimize(objective, n_trials=n_trials)

    # guard in case all trials were pruned/failed
    try:
        best = study.best_params
        logger.info(f"HPO done for {model_name}. Best score: {study.best_value:.4f}")
        return best
    except ValueError as e:
        logger.warning(f"HPO finished for {model_name}, but no completed trials: {e}")
        return {}


def _sample_params(trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sample parameters using Optuna trial object.
    Handles conditional and static parameters.
    """
    params: Dict[str, Any] = {}

    for name, spec in search_space.items():
        if name == "__static__":
            continue

        condition = spec.get("condition")
        if condition:
            if not all(params.get(k) == v for k, v in condition.items()):
                continue

        param_type = spec.get("type", None)
        bounds = spec.get("bounds", None)
        if not param_type or bounds is None:
            raise ValueError(f"Invalid spec for param '{name}'")

        if param_type == "float":
            params[name] = trial.suggest_float(name, *bounds)
        elif param_type == "float_log":
            params[name] = trial.suggest_float(name, *bounds, log=True)
        elif param_type == "int":
            params[name] = trial.suggest_int(name, *bounds)
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(name, bounds)
        else:
            raise ValueError(f"Unknown param type '{param_type}' for '{name}'")

    # append static (non-tunable) params at the end
    static_params = search_space.get("__static__", {})
    params.update(static_params)

    return params


def _evaluate_model(
    trial: optuna.Trial,
    params: Dict[str, Any],
    model_cls: Callable[..., BaseEstimator],
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    test_size: float,
    pruner: Optional[optuna.pruners.BasePruner] = None,
) -> float:
    """
    Train model with Optuna fidelity-aware logic depending on the pruner.
    If pruner is None, we do not use fidelity and just fit once.
    """
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )
        if pruner is None:
            return _train_no_pruning(
                model_cls=model_cls,
                params=params,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model_name=model_name,
            )
        return training_with_optuna_fidelity(
            model_name=model_name,
            model_cls=model_cls,
            trial=trial,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

    except optuna.exceptions.TrialPruned:
        # bubble up so Optuna accounts it as a pruned trial (not a failure)
        raise
    except Exception as e:
        logger.warning(f"[SKIP] Trial failed for {model_name} with error: {e}")
        # treat unexpected errors as prunable so a bad config doesn't stall the study
        raise optuna.TrialPruned()


def training_with_optuna_fidelity(
    model_name: str,
    model_cls: Callable[..., BaseEstimator],
    trial: optuna.Trial,
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """
    Automatically chooses a fidelity knob per model, trains, and feeds intermediate R^2
    to Optuna so SH/Hyperband can prune. If staging is not feasible it falls back to fitting once.
    Returns the final R^2 on the validation set.
    """
    # TODO: remove if pipeline thing changes
    if model_name == "catboost":
        cat_features = get_catboost_cat_features(X_train)
        params = dict(params)
        params["cat_features"] = cat_features

    fid_key = FIDELITY_MAP.get(model_name, None)
    if fid_key is None:
        return _train_no_pruning(
            model_cls=model_cls,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_name=model_name,
        )

    max_budget = _ensure_fidelity_in_params(model_name, params, fid_key)
    if max_budget <= 0:
        logger.debug(
            f"Non-positive fidelity ({fid_key}={max_budget}) for {model_name}."
        )
        return _train_no_pruning(
            model_cls=model_cls,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_name=model_name,
        )

    logger.info(f"{model_name}: Using fidelity '{fid_key}' with budget={max_budget}.")
    name = model_name.lower()

    # 1) Epoch-based models: use epochs as fidelity, checkpoint linearly
    if name in {"realmlp", "tabm", "modern_nca"}:
        model = model_cls(**{**params, fid_key: max_budget})
        checkpoints = _linear_checkpoint_scheduling(
            max_budget, num_checks=min(10, max(3, max_budget))
        )
        last_score = -np.inf
        total_elapsed = 0.0  # rough timing accumulation

        if _has_method(model, "train_one_epoch"):
            for epoch in range(1, max_budget + 1):
                t0 = time.perf_counter()
                model.train_one_epoch(X_train, y_train)
                if epoch in checkpoints:
                    preds = model.predict(X_val)
                    last_score = _safe_r2(y_val, preds)
                    dt = time.perf_counter() - t0
                    total_elapsed += dt
                    est_saved = _estimate_saved_time(epoch, max_budget, total_elapsed)
                    _report_and_maybe_prune(
                        trial,
                        step=epoch,
                        score=last_score,
                        msg=(
                            f"{model_name} epoch={epoch}/{max_budget} | "
                            f"checkpoint_time={dt:.3f}s, est_saved_if_pruned~{est_saved:.3f}s"
                        ),
                    )
            return last_score

        elif _has_method(model, "partial_fit"):
            for epoch in range(1, max_budget + 1):
                t0 = time.perf_counter()
                model.partial_fit(X_train, y_train)
                if epoch in checkpoints:
                    preds = model.predict(X_val)
                    last_score = _safe_r2(y_val, preds)
                    dt = time.perf_counter() - t0
                    total_elapsed += dt
                    est_saved = _estimate_saved_time(epoch, max_budget, total_elapsed)
                    _report_and_maybe_prune(
                        trial,
                        step=epoch,
                        score=last_score,
                        msg=(
                            f"{model_name} epoch={epoch}/{max_budget} | "
                            f"checkpoint_time={dt:.3f}s, est_saved_if_pruned~{est_saved:.3f}s"
                        ),
                    )
            return last_score

        # no incremental method: single fit (honor epochs if constructor supports it)
        try:
            model = model_cls(**{**params, fid_key: max_budget})
            model.fit(X_train, y_train)
        except TypeError:
            model = model_cls(**params)
            try:
                model.fit(X_train, y_train, epochs=max_budget)
            except TypeError:
                model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = _safe_r2(y_val, preds)
        _report_and_maybe_prune(
            trial,
            step=max_budget,
            score=score,
            msg=f"{model_name} single-shot (no epoch staging)",
        )
        return score

    # 2) tree-based models: LightGBM / XGBoost / CatBoost / RandomForest
    elif name in {"lightgbm", "xgboost", "catboost", "randomforest"}:
        checkpoints = _linear_checkpoint_scheduling(max_budget, num_checks=5)
        last_score = -np.inf
        total_elapsed = 0.0

        if name == "randomforest":
            # incremental tree growth with warm_start.
            step_params = dict(params)
            step_params["warm_start"] = True
            model = model_cls(**step_params)
            prev_step = 0
            for step in checkpoints:
                t0 = time.perf_counter()
                model.set_params(**{fid_key: int(step)})  # grow to 'step' trees
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                score = _safe_r2(y_val, preds)
                last_score = score
                dt = time.perf_counter() - t0
                total_elapsed += dt
                est_saved = _estimate_saved_time(step, max_budget, total_elapsed)
                _report_and_maybe_prune(
                    trial,
                    step=step,
                    score=score,
                    msg=(
                        f"{model_name} trees={step}/{max_budget} (warm_start) | "
                        f"checkpoint_time={dt:.3f}s, est_saved_if_pruned~{est_saved:.3f}s"
                    ),
                )
                prev_step = step
            return last_score

        # boosters: continue training by adding ONLY the delta rounds each checkpoint
        prev_model = None
        prev_step = 0
        for step in checkpoints:
            add = int(step) - int(prev_step)
            if add <= 0:
                prev_step = step
                continue

            t0 = time.perf_counter()
            step_params = dict(params)
            step_params[fid_key] = add  # ADDITIONAL rounds only

            try:
                if name == "xgboost":
                    # continue from previous using xgb_model=Booster; add 'add' rounds
                    model = model_cls(**step_params)
                    if prev_model is None:
                        model.fit(X_train, y_train, verbose=False)
                    else:
                        model.fit(
                            X_train,
                            y_train,
                            xgb_model=prev_model.get_booster(),
                            verbose=False,
                        )

                elif name == "lightgbm":
                    # continue from previous using init_model=Booster; add 'add' rounds
                    model = model_cls(**step_params)
                    if prev_model is None:
                        model.fit(X_train, y_train)
                    else:
                        model.fit(
                            X_train,
                            y_train,
                            init_model=prev_model.booster_,
                        )

                elif name == "catboost":
                    # continue from previous using init_model=prev_model; add 'add' rounds
                    step_params = dict(step_params)
                    step_params.setdefault("allow_writing_files", False)
                    step_params.setdefault("use_best_model", False)

                    model = model_cls(**step_params)
                    if prev_model is None:
                        model.fit(X_train, y_train, verbose=False)
                    else:
                        model.fit(
                            X_train, y_train, init_model=prev_model, verbose=False
                        )

                else:
                    # safety net: fresh fit for unknowns
                    model = model_cls(**step_params)
                    model.fit(X_train, y_train)

            except Exception as e:
                logger.warning(
                    f"{model_name}: incremental fit failed at {fid_key}={step} with error: {e}"
                )
                # let pruner mark this trial as pruned instead of crashing the whole study
                raise optuna.TrialPruned(f"{model_name} failed at step={step}")

            preds = model.predict(X_val)
            score = _safe_r2(y_val, preds)
            last_score = score
            dt = time.perf_counter() - t0
            total_elapsed += dt
            est_saved = _estimate_saved_time(step, max_budget, total_elapsed)

            _report_and_maybe_prune(
                trial,
                step=step,
                score=score,
                msg=(
                    f"{model_name} {fid_key}+={add} (total~{step}/{max_budget}) | "
                    f"checkpoint_time={dt:.3f}s, est_saved_if_pruned~{est_saved:.3f}s"
                ),
            )

            prev_model = model
            prev_step = step

        return last_score

    # 3) other models: fall back
    else:
        logger.debug(f"Unrecognized fidelity control for {model_name}.")
        return _train_no_pruning(
            model_cls=model_cls,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_name=model_name,
        )


def _train_no_pruning(
    model_cls: Callable[..., BaseEstimator],
    params: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "",
) -> float:
    """
    Fallback training when pruning/fidelity logic is disabled.
    """
    # TODO: see 'get_catboost_cat_features'
    if model_name == "catboost":
        cat_features = get_catboost_cat_features(X_train)
        params = dict(params)
        params["cat_features"] = cat_features
    logger.debug(f"Fitting once for {model_name}.")
    model = model_cls(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return _safe_r2(y_val, preds)


def _linear_checkpoint_scheduling(max_step: int, num_checks: int = 5) -> List[int]:
    """
    Make a simple ladder of checkpoint steps: e.g., 20,40,60,80,100 for max_step=100.
    Avoid 0 and duplicates; always include max_step as the last checkpoint.
    """
    max_step = int(max(1, max_step))
    if max_step <= 3 or num_checks <= 1:
        return [max_step]
    raw = np.linspace(max_step / num_checks, max_step, num_checks)
    steps = sorted(set(int(round(x)) for x in raw))
    steps = [s for s in steps if s > 0]
    if steps[-1] != max_step:
        steps.append(max_step)
    return steps


def _has_method(obj: Any, name: str) -> bool:
    return hasattr(obj, name) and callable(getattr(obj, name))


def _safe_r2(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Be defensive with R^2:
      - if y_true is constant, sklearn returns nan and warns
      - so this takes that and gives a very bad score so its pruned out
    """
    try:
        s = float(r2_score(y_true, y_pred))
        if np.isnan(s):
            return -1e9
        return s
    except Exception:
        return -1e9


def _ensure_fidelity_in_params(
    model_name: str, params: Dict[str, Any], fid_key: str
) -> int:
    """
    Ensure params contains the fidelity key; inject a practical default if missing.
    Return the int budget we will target.
    """
    if fid_key in params:
        return int(params[fid_key])

    defaults = {
        "lightgbm": 400,
        "xgboost": 400,
        "catboost": 400,
        "randomforest": 300,
        "realmlp": 50,
        "tabm": 50,
        "modern_nca": 50,
    }
    fallback = defaults.get(model_name.lower(), 100)
    params[fid_key] = fallback
    logger.debug(f"Injected default {fid_key}={fallback} for {model_name}.")
    return fallback


def _report_and_maybe_prune(
    trial: optuna.Trial, step: int, score: float, msg: str
) -> None:
    """
    Single place where we talk to Optuna. We always report R^2 (maximize).
    """
    trial.report(score, step=step)
    logger.debug(f"step={step} | R^2={score:.5f} | {msg}")
    if trial.should_prune():
        logger.info(f"pruned at step={step} with R^2={score:.5f} | {msg}")
        raise optuna.TrialPruned(msg)
    else:
        logger.debug(f"continued at step={step} with R^2={score:.5f} | {msg}")


def get_catboost_cat_features(X: pd.DataFrame) -> List[str]:
    """Return list of categorical column names for CatBoost."""
    return [col for col in X.columns if str(X[col].dtype) == "category"]


def _estimate_saved_time(step: int, max_budget: int, elapsed: float) -> float:
    """
    rough estimation of time: assume per-unit (tree/iter/epoch) time stays about the same
    saved is basically remaining_units * (elapsed / completed_units)
    """
    if step <= 0:
        return 0.0
    per_unit = elapsed / float(step)
    remaining = max(0, int(max_budget) - int(step))
    return max(0.0, remaining * per_unit)
