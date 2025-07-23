from __future__ import annotations
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import pandas as pd
import numpy as np
import logging
import optuna

logger = logging.getLogger(__name__)
METRICS = {"r2": r2_score}

class AutoML:
    def __init__(self, seed: int, metric: str = "r2") -> None:
        self.seed = seed
        self.metric = METRICS[metric]
        self._model = None
        self.best_model_type = None
        self.best_params = {}

    def _objective(self, trial, X_train, X_val, y_train, y_val, cat_features):
        model_type = trial.suggest_categorical("model", ["catboost", "lightgbm"])

        if model_type == "catboost": # https://catboost.ai/docs/en/references/training-parameters/
            params = {
                "iterations": trial.suggest_int("catboost:iterations", 200, 1000),
                "learning_rate": trial.suggest_float("catboost:learning_rate", 1e-3, 0.3, log=True),
                "depth": trial.suggest_int("catboost:depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("catboost:l2_leaf_reg", 1, 10.0),
                "bagging_temperature": trial.suggest_float("catboost:bagging_temperature", 0.0, 1.0),
                "border_count": trial.suggest_int("catboost:border_count", 32, 255),
                "random_strength": trial.suggest_float("catboost:random_strength", 1e-9, 10.0, log=True),
                "rsm": trial.suggest_float("catboost:rsm", 0.5, 1.0),
                "leaf_estimation_iterations": trial.suggest_int("catboost:leaf_estimation_iterations", 1, 10),
                "leaf_estimation_method": trial.suggest_categorical("catboost:leaf_estimation_method", ["Newton", "Gradient"]),
                "grow_policy": trial.suggest_categorical("catboost:grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
                "boosting_type": trial.suggest_categorical("catboost:boosting_type", ["Plain", "Ordered"]),
                "bootstrap_type": trial.suggest_categorical("catboost:bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
                "loss_function": "RMSE",
                "min_data_in_leaf": trial.suggest_int("catboost:min_data_in_leaf", 1, 20),
                "verbose": False,
                "random_seed": self.seed,
            }
            if params["bootstrap_type"] != "Bayesian":
                # Remove bagging_temperature since it's not allowed
                params.pop("bagging_temperature", None)
            if params["boosting_type"] == "Ordered" and params["grow_policy"] != "SymmetricTree":
                # Invalid combination; raise an Optuna pruning exception
                raise optuna.TrialPruned("Ordered boosting requires SymmetricTree grow_policy.")

            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, cat_features=cat_features)

        elif model_type == "lightgbm": # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
            for col in cat_features:
                X_train[col] = X_train[col].astype("category")
                X_val[col] = X_val[col].astype("category")

            params = {
                "n_estimators": trial.suggest_int("lightgbm:n_estimators", 200, 1000),
                "learning_rate": trial.suggest_float("lightgbm:learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("lightgbm:max_depth", 3, 12),
                "num_leaves": trial.suggest_int("lightgbm:num_leaves", 20, 300),
                "min_child_samples": trial.suggest_int("lightgbm:min_child_samples", 5, 100),
                "min_child_weight": trial.suggest_float("lightgbm:min_child_weight", 1e-3, 10.0, log=True),
                "subsample": trial.suggest_float("lightgbm:subsample", 0.5, 1.0),
                "subsample_freq": trial.suggest_int("lightgbm:subsample_freq", 1, 7),
                "colsample_bytree": trial.suggest_float("lightgbm:colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("lightgbm:reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("lightgbm:reg_lambda", 1e-8, 10.0, log=True),
                "random_state": self.seed,
                "objective": "regression",
                "boosting_type": trial.suggest_categorical("lightgbm:boosting_type", ["gbdt", "dart", "goss"]),
                "min_split_gain": trial.suggest_float("lightgbm:min_split_gain", 0.0, 1.0),
                "verbosity": -1,
            }

            if params["boosting_type"] == "goss":
                params.pop("subsample", None)
                params.pop("subsample_freq", None)

            model = LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(50), log_evaluation(0)],
            )

        preds = model.predict(X_val)
        return -self.metric(y_val, preds)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> AutoML:
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=self.seed, test_size=0.2)
        cat_features = X.select_dtypes(include=["category", "object"]).columns.tolist()

        logger.info("Starting Optuna hyperparameter tuning...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: self._objective(trial, X_train.copy(), X_val.copy(), y_train, y_val, cat_features),
            n_trials=20
        )

        self.best_model_type = study.best_trial.params["model"]
        self.best_params = study.best_trial.params

        logger.info(f"Best model: {self.best_model_type}")
        logger.info(f"Best params: {self.best_params}")

        if self.best_model_type == "catboost":
            best_params = {k.split("catboost:")[1]: v for k, v in self.best_params.items() if k.startswith("catboost:")}
            best_params["random_seed"] = self.seed
            best_params["loss_function"] = "RMSE"
            best_params["verbose"] = False
            self._model = CatBoostRegressor(**best_params)
            self._model.fit(X, y, cat_features=cat_features)

        elif self.best_model_type == "lightgbm":
            for col in cat_features:
                X[col] = X[col].astype("category")
            best_params = {k.split("lightgbm:")[1]: v for k, v in self.best_params.items() if k.startswith("lightgbm:")}
            best_params["random_state"] = self.seed
            self._model = LGBMRegressor(**best_params)
            self._model.fit(X, y)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ValueError("Model not fitted yet.")
        return self._model.predict(X)
