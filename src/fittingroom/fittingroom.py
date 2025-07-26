from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from fittingroom.meta_learning import extract_meta_features
from fittingroom.pipeline import aggregate_predictions, fit_model, select_portfolio
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
    ) -> None:
        self.seed = seed if seed is not None else get_default_constant("SEED")
        self._test_size = get_default_constant("TEST_SIZE")
        self._precision = get_default_constant("PRECISION")
        self._models: list = []
        self.ask_expert_opinion = ask_expert_opinion
        self.hpo_method = hpo_method

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

        # get validation predictions from all models
        val_preds_list = [model.predict(X_val) for model in self._models]
        val_preds = aggregate_predictions(val_preds_list)

        # calculate validation R²
        val_r2 = r2_score(y_val, val_preds)
        logger.info(f"Validation R²: {val_r2:.{self._precision}f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the ensemble of all fitted models.
        Returns averaged predictions.
        """
        if not self._models:
            raise ValueError(".fit() must be called before .predict()")

        preds = [model.predict(X) for model in self._models]
        return aggregate_predictions(preds)
