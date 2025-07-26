from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tabpfn import TabPFNRegressor

"""
MODEL_PORTFOLIO = {
    "realmlp": None,
    "tabm": None,
    "catboost": CatBoostRegressor,
    "lightgbm": LGBMRegressor,
    "xgboost": None,
    "modern_nca": None,
    "linear": LinearRegression,
    "knn": None,
    "randomforest": RandomForestRegressor,
    "tabpfn": None,
}"""
MODEL_PORTFOLIO = {
    "realmlp": None,
    "tabm": None,
    "catboost": None,
    "lightgbm": LGBMRegressor,
    "xgboost": None,
    "modern_nca": None,
    "linear": None,
    "knn": None,
    "randomforest": None,
    "tabpfn": None,
}

# place conditionals at the end.
SEARCH_SPACES = {
    "lr": {
        "fit_intercept": {"type": "categorical", "bounds": [True, False]},
        "positive": {"type": "categorical", "bounds": [True, False]},
    },
    "randomforest": {
        "n_estimators": {"type": "categorical", "bounds": [50, 100, 200]},
        "max_depth": {"type": "categorical", "bounds": [None, 5, 10]},
    },
    "tabpfn": {},
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
    },
    "lightgbm": {
        "__static__": {
            "objective": "regression",
            "verbosity": -1,
        },
        "n_estimators": {"type": "int", "bounds": (200, 1000), "__fidelity__": True},
        "learning_rate": {"type": "float_log", "bounds": (1e-2, 0.3)},
        "max_depth": {"type": "int", "bounds": (3, 12)},
        "num_leaves": {"type": "int", "bounds": (10, 256)},
        "min_child_samples": {"type": "int", "bounds": (5, 50)},
        "min_child_weight": {"type": "float_log", "bounds": (1e-3, 10.0)},
        "min_split_gain": {"type": "float", "bounds": (0.0, 0.5)},
        "colsample_bytree": {"type": "float", "bounds": (0.5, 1.0)},
        "reg_alpha": {"type": "float_log", "bounds": (1e-8, 10.0)},
        "reg_lambda": {"type": "float_log", "bounds": (1e-8, 10.0)},
        "boosting_type": {
            "type": "categorical",
            "bounds": ["gbdt", "dart", "goss"],
        },
        "subsample": {
            "type": "float",
            "bounds": (0.5, 1.0),
            "condition": {"boosting_type": "gbdt"},
        },
        "subsample_freq": {
            "type": "int",
            "bounds": (1, 7),
            "condition": {"boosting_type": "gbdt"},
        },
    },
}
