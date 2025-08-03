import logging
from typing import Callable
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor
from scipy.stats import norm

from sklearn.metrics import r2_score
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


SEARCH_SPACES = {
    "catboost": {
        "iterations": {"type": "int", "bounds": (200, 1000), "__fidelity__": True},
        "learning_rate": {"type": "float_log", "bounds": (1e-3, 0.3)},
        "depth": {"type": "int", "bounds": (4, 10)},
        "l2_leaf_reg": {"type": "float", "bounds": (1.0, 10.0)},
        "border_count": {"type": "int", "bounds": (32, 255)},
        "random_strength": {"type": "float_log", "bounds": (1e-9, 10.0)},
        "rsm": {"type": "float", "bounds": (0.5, 1.0)},
        "leaf_estimation_iterations": {"type": "int", "bounds": (1, 10)},
        "leaf_estimation_method": {
            "type": "categorical",
            "bounds": ["Newton", "Gradient"],
        },
        "grow_policy": {
            "type": "categorical",
            "bounds": ["SymmetricTree", "Depthwise", "Lossguide"],
        },
        "bootstrap_type": {
            "type": "categorical",
            "bounds": ["Bayesian", "Bernoulli", "MVS"],
        },
        "min_data_in_leaf": {"type": "int", "bounds": (1, 20)},
        "bagging_temperature": {
            "type": "float",
            "bounds": (0.0, 1.0),
            "condition": {"bootstrap_type": "Bayesian"},
        },
        "boosting_type": {
            "type": "categorical",
            "bounds": ["Plain", "Ordered"],
            "condition": {"grow_policy": "SymmetricTree"},
        },
    }
}


class TabPFNSurrogate:
    def __init__(self, device="cpu", n_estimators=1):
        self.model = TabPFNRegressor(device=device, n_estimators=1)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, return_std=False):
        if return_std:
            pred = self.model.predict(X, output_type="quantiles", quantiles=[0.1,0.5,0.9])
            
            q10, q50, q90 = pred[0], pred[1], pred[2]
            mean = q50
            std = (q90 - q10) / 2
            return mean, std
        else:
            return self.model.predict(X, output_type="mean")


def UCB(x: np.ndarray, model, kappa: float = 2.0) -> float:
    """Upper Confidence Bound"""
    x = np.array(x).reshape(1, -1)
    mu, std = model.predict(x, return_std=True)
    return mu[0] + kappa * std[0]
    

def PI(x: np.ndarray, model, eta: float, xi: float = 0.01) -> float:
    """Probability of Improvement"""
    x = np.array(x).reshape(1, -1)
    mu, std = model.predict(x, return_std=True)
    if std[0] == 0:
        return 0.0
    z = (mu[0] - eta - xi) / std[0]
    return norm.cdf(z)


def EI(x: float, model, eta: float, xi: float = 0.01) -> float:
    """Expected Improvement"""
    x = np.array(x).reshape(1, -1)

    m, s = model.predict(x, return_std=True)

    s = np.maximum(s, 1e-9)
    Z = (eta - m - xi) / s
    exp_imp = s * (Z * norm.cdf(Z) + norm.pdf(Z))

    return exp_imp


def run_bo_tabpfn(
    X_train,
    y_train,
    X_val,
    y_val,
    max_iter: int = 30,
    model_cls: Callable[..., BaseEstimator] = CatBoostRegressor,
    acquisition: str = "ei",  # "ei", "ucb", or "pi"
    init: int = 5,
    n_restarts: int = 2,
    randomize: bool = True,
    seed: int = 1,
    log_file: str = "bo_log.csv",
):
    """Run the BO loop.

    Parameters
    ----------
    acquisition : (x: float, model: Pipeline, eta: float, **func_params) -> float
        The acquisition function to use

    max_iter : int
        The max number of iterations to run for

    init : int = 25
        Number of points to build initial model with

    randomize: bool = True
        Whether to randomize initial points. If False, samples are lineraly sampled,
        otherwise the are sampled uniformly ad random

    seed : int = 1
        The seed to use for randomization

    Returns
    -------
    np.ndarray
        The evaluated points
    """
    np.random.seed(seed)

    bounds = [
        SEARCH_SPACES["catboost"]["learning_rate"]["bounds"],
        SEARCH_SPACES["catboost"]["l2_leaf_reg"]["bounds"],
        SEARCH_SPACES["catboost"]["depth"]["bounds"]
    ]

    categorical_cols = [
        c for c in X_train.columns if X_train[c].dtype.name in ("object", "category", "bool")
    ]
    
    cat_features = [X_train.columns.get_loc(col) for col in categorical_cols]
    
    # generate the initial hyperparameter values
    x = np.column_stack([
        np.random.uniform(bounds[0][0], bounds[0][1], size=init),  # learning_rate
        np.random.randint(bounds[1][0], bounds[1][1], size=init),   # l2_leaf_reg
        np.random.randint(bounds[2][0], bounds[2][1], size=init)   # depth
      ])
    log_data = []
    ys = []
    for xi in x:
        cb = model_cls(cat_features=cat_features, learning_rate=xi[0], l2_leaf_reg=int(xi[1]),depth=int(xi[2]),verbose=False)
        cb.fit(X_train, y_train)
        r2 = r2_score(y_val, cb.predict(X_val))
        ys.append(r2)
        log_data.append({"iteration": len(log_data), "learning_rate": xi[0], "l2_leaf_reg": int(xi[1]),"depth": int(xi[2]), "r2": r2, "best_r2": max(ys), "init_point":True})

    x = x.astype(np.float32)
    ys = np.asarray(ys, dtype=np.float32)

    # IMPORTANT x and y are not X_train and y_train, but the hyperparameters and the response values

    scaler = StandardScaler()
    model = TabPFNSurrogate(device="cpu", n_estimators=x.shape[1])
    best_model = None
    best_r2 = -np.inf
    for i in range(max_iter - init):  # BO loop
        # Feel free to adjust the hyperparameters
        x_scaled = scaler.fit_transform(x)
        model.fit(x_scaled, ys)

        x_best, score_best = None, np.inf

        for _ in range(n_restarts):
            x0 = np.array([
                np.random.uniform(bounds[0][0], bounds[0][1]),  # learning_rate
                np.random.randint(bounds[1][0], bounds[1][1]),   # l2_leaf_reg
                np.random.randint(bounds[2][0], bounds[2][1])   # depth
                ])
            if acquisition == "ei":
                opt_res = minimize(
                    fun=lambda xp: -EI(x=xp, model=model, eta=max(ys)),
                    x0=x0, bounds=bounds, method="L-BFGS-B", options={"maxfun": 10}
                )
            elif acquisition == "ucb":
                opt_res = minimize(
                    fun=lambda xp: -UCB(x=xp, model=model),
                    x0=x0, bounds=bounds, method="L-BFGS-B", options={"maxfun": 10}
                )
            elif acquisition == "pi":
                opt_res = minimize(
                    fun=lambda xp: -PI(x=xp, model=model, eta=max(ys)),
                    x0=x0, bounds=bounds, method="L-BFGS-B", options={"maxfun": 10}
                )
            else:
                raise ValueError(f"Unknown acquisition: {acquisition}")

            if opt_res.fun < score_best:
                x_best, score_best = np.asarray(opt_res.x).reshape(1, -1), opt_res.fun
                
        x = np.vstack([x, x_best])
        cb = model_cls(cat_features=cat_features, learning_rate=x_best[0][0], l2_leaf_reg=int(x_best[0][1]), depth=int(x_best[0][2]),verbose=False)
        cb.fit(X_train, y_train)
        r2 = r2_score(y_val, cb.predict(X_val))
        if r2 > best_r2:
            best_r2 = r2
            best_model = cb
        ys = np.append(ys, r2)
        log_data.append({"iteration": len(log_data),"learning_rate": x_best[0][0], "l2_leaf_reg": int(x_best[0][1]),"depth": int(x_best[0][2]), "r2": r2, "best_r2": max(ys), "init_point":False})
        logger.debug(f"[{i+1}/{max_iter - init}] Next = learning_rate={x_best[0][0]}, l2_leaf_reg={int(x_best[0][1])}, depth={int(x_best[0][2])}, R²={r2:.4f}")

    log_df = pd.DataFrame(log_data)
    log_df.to_csv(log_file, index=False)
    
    return best_model
