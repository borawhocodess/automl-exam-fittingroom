from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from fittingroom.pipeline import (
    select_portfolio,
    fit_model,
    aggregate_predictions,
)

from fittingroom.meta_learning import extract_meta_features

from fittingroom.utils import get_default_constant

logger = logging.getLogger(__name__)


class FittingRoom:

    def __init__(
        self,
        seed: int,
        ask_expert_opinion: bool = False,
    ) -> None:
        self.seed = seed if seed is not None else get_default_constant("SEED")
        self._test_size = get_default_constant("TEST_SIZE")
        self._precision = get_default_constant("PRECISION")
        self._models: list = []
        self.ask_expert_opinion = ask_expert_opinion

    def fit(
        self,
        X,
        y,
    ):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self._test_size, random_state=self.seed
        )

        meta_features = extract_meta_features(X_train, y_train)
        portfolio = select_portfolio(
            meta_features, ask_expert_opinion=self.ask_expert_opinion
        )

        self._models = [fit_model(name, X_train, y_train) for name in portfolio]

        val_preds_list = [m.predict(X_val) for m in self._models]
        val_preds = aggregate_predictions(val_preds_list)
        val_r2 = r2_score(y_val, val_preds)
        logging.getLogger(__name__).info(f"validation r2: {val_r2:.{self._precision}f}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._models:
            raise ValueError(".fit must be called before predict")

        preds = [m.predict(X) for m in self._models]
        aggregated_preds = aggregate_predictions(preds)
        return aggregated_preds
