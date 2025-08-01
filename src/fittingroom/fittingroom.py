from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from fittingroom.meta_learning import extract_meta_features
from fittingroom.pipeline import aggregate_predictions, build_pipeline, fit_model, select_portfolio
from fittingroom.utils import get_default_constant

logger = logging.getLogger(__name__)


class FittingRoom:
    """
    Main AutoML pipeline class for regression on tabular data.
    Wraps portfolio selection, model training, and ensemble prediction.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        ask_expert_opinion: bool = False,
        hpo_method: str = "random",
        add_default_preds_as_features: bool = False,
    ) -> None:
        self.seed = seed if seed is not None else get_default_constant("SEED")
        self._test_size = get_default_constant("TEST_SIZE")
        self._precision = get_default_constant("PRECISION")
        self._models: list = []
        self.ask_expert_opinion = ask_expert_opinion
        self.hpo_method = hpo_method
        self.add_default_preds_as_features = add_default_preds_as_features
        self._models_for_default_preds_as_features: dict = {}

    def _add_default_preds_as_features(
        self,
        X_train,
        y_train,
        X_val,
        portfolio,
    ):
        """
        novelty oder was
        """
        for model_name in portfolio:
            col_name = f"pred_{model_name}"

            try:
                pipe = build_pipeline(model_name, X_train)
            except Exception as e:
                logger.warning(
                    f"could not pipe : {model_name} - {e}"
                )
                continue

            try:
                pipe.fit(X_train, y_train)
            except Exception as e:
                logger.warning(
                    f"skipping {model_name} - {e}"
                )
                continue

            self._models_for_default_preds_as_features[model_name] = pipe

            train_preds = pipe.predict(X_train)
            val_preds = pipe.predict(X_val)

            if col_name in X_train.columns:
                logger.warning(
                    f"overwriting column {col_name}"
                )

            X_train[col_name] = train_preds
            X_val[col_name] = val_preds

            logger.debug(
                f"added: {col_name}"
            )

        logger.info(
            f"added default predictions as features: {list(self._models_for_default_preds_as_features.keys())}"
        )

        return X_train, X_val


    def fit(self, X: pd.DataFrame, y: pd.Series) -> FittingRoom:
        """
        Fit multiple models on the training split and evaluate on validation split.
        Uses meta-features to select a model portfolio dynamically.
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self._test_size, random_state=self.seed
        )

        # get dataset-level info like #missing, skew, etc.
        meta_features = extract_meta_features(X_train, y_train)

        # pick subset of models using rules + optional expert confirmation
        portfolio = select_portfolio(
            meta_features, ask_expert_opinion=self.ask_expert_opinion
        )

        if self.add_default_preds_as_features:
            X_train, X_val = self._add_default_preds_as_features(
                X_train,
                y_train,
                X_val,
                portfolio,
            )


        # fit each model and catch training failures gracefully
        trained_models = []
        for model_name in portfolio:
            try:
                model = fit_model(
                    model_name,
                    X_train,
                    y_train,
                    hpo_method=self.hpo_method,
                    seed=self.seed,
                )
                trained_models.append(model)
            except Exception as e:
                logger.warning(f"Skipping model '{model_name}' due to error: {e}")

        if not trained_models:
            raise RuntimeError("No models were successfully trained.")

        self._models = trained_models

        val_preds_list = [m.predict(X_val) for m in self._models]

        for name, preds in zip(portfolio, val_preds_list):
            model_r2 = r2_score(y_val, preds)

            logger.info(f"validation r2 for {name}: {model_r2:.{self._precision}f}")

        val_preds = aggregate_predictions(val_preds_list)

        val_r2 = r2_score(y_val, val_preds)

        logging.getLogger(__name__).info(f"validation r2 after aggregation: {val_r2:.{self._precision}f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the ensemble of all fitted models.
        Returns averaged predictions.
        """
        if not self._models:
            raise ValueError(".fit() must be called before .predict()")

        if self.add_default_preds_as_features:
            for model_name, pipe in self._models_for_default_preds_as_features.items():
                col_name = f"pred_{model_name}"
                if col_name in X.columns:
                    logger.warning(
                        f"overwriting column {col_name}"
                    )
                X[col_name] = pipe.predict(X)

        # print(X.head())

        test_preds_list = [model.predict(X) for model in self._models]

        aggregated_preds = aggregate_predictions(test_preds_list)

        return aggregated_preds
