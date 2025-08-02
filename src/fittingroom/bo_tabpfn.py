import logging
from typing import Callable
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor
from scipy.stats import norm

from sklearn.metrics import root_mean_squared_error
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


def EI(x: float, model, eta: float, xi: float = 0.01) -> float:
    """Calculate Expected Improvement.

    Parameters
    ----------
    x : float
        The value to determine the acquisition value at

    model : Pipeline
        The model to predict with

    eta : float
        The best value seen so far

    Returns
    -------
    float
        The expected improment for a given model at x
    """
    # Required as models predict on a list of values normally, not just one
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
    acquisition: Callable = EI,
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
        rmse = root_mean_squared_error(y_val, cb.predict(X_val))
        ys.append(rmse)
        log_data.append({"iteration": len(log_data), "learning_rate": xi[0], "l2_leaf_reg": int(xi[1]),"depth": int(xi[2]), "rmse": rmse, "best_rmse": min(ys), "init_point":True})

    x = x.astype(np.float32)
    ys = np.asarray(ys, dtype=np.float32)

    # IMPORTANT x and y are not X_train and y_train, but the hyperparameters and the response values

    scaler = StandardScaler()
    model = TabPFNSurrogate(device="cpu", n_estimators=x.shape[1])
    best_model = None
    best_rmse = np.inf
    for i in range(max_iter - init):  # BO loop
        # Feel free to adjust the hyperparameters
        x_scaled = scaler.fit_transform(x)
        model.fit(x_scaled, ys)

        # optimize acquisition function
        x_ = 0.0
        y_ = np.inf

        for _ in range(n_restarts):
          x0 = np.array([
              np.random.uniform(bounds[0][0], bounds[0][1]),  # learning_rate
              np.random.randint(bounds[1][0], bounds[1][1]),   # l2_leaf_reg
              np.random.randint(bounds[2][0], bounds[2][1])   # depth
            ])
          opt_res = minimize(
              fun=lambda x: -1 * acquisition(x=x, model=model, eta=min(ys), xi=0.001*((max_iter-init)/(i+1))),
              x0=x0,
              bounds=bounds,
              options={"maxfun": 10},
              method="L-BFGS-B",
          )

          # Update if we found a better value
          y_out = opt_res.fun
          if y_out < y_:
              x_ = np.asarray(opt_res.x, dtype=float).reshape(1, -1)
              y_ = y_out
        x = np.vstack([x, x_])
        cb = model_cls(cat_features=cat_features, learning_rate=x_[0][0], l2_leaf_reg=int(x_[0][1]), depth=int(x_[0][2]),verbose=False)
        cb.fit(X_train, y_train)
        rmse = root_mean_squared_error(y_val, cb.predict(X_val))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = cb
        ys = np.append(ys, rmse)
        log_data.append({"iteration": len(log_data),"learning_rate": xi[0], "l2_leaf_reg": int(xi[1]),"depth": int(xi[2]), "rmse": rmse, "best_rmse": min(ys), "init_point":False})
        logger.debug(f"[{i+1}/{max_iter - init}] Next = learning_rate={x_[0][0]}, l2_leaf_reg={int(x_[0][1])}, depth={int(x_[0][2])}, RMSE={rmse:.4f}")

    log_df = pd.DataFrame(log_data)
    log_df.to_csv(log_file, index=False)
    
    return best_model
