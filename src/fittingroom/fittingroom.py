from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline

from fittingroom.bo_tabpfn import run_bo_tabpfn
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
        add_post_hpo_preds_as_features: bool = False,
        use_secondary_hpo: bool = False,
        use_bo_tabpfn_surrogate: bool = False,
    ) -> None:
        self.seed = seed if seed is not None else get_default_constant("SEED")
        self._test_size = get_default_constant("TEST_SIZE")
        self._precision = get_default_constant("PRECISION")
        self._models: list = []
        self._val_scores: list = []
        self.ask_expert_opinion = ask_expert_opinion
        self.hpo_method = hpo_method
        self.add_default_preds_as_features = add_default_preds_as_features
        self.add_post_hpo_preds_as_features = add_post_hpo_preds_as_features
        self._models_for_default_preds_as_features: dict = {}
        self._models_for_post_hpo_preds_as_features: dict = {}
        self._bo_tabpfn_fitted_model = None
        self.use_bo_tabpfn_surrogate = use_bo_tabpfn_surrogate
        self.use_secondary_hpo = use_secondary_hpo

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
        kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=self.seed,
        )

        X_base = X_train.copy(deep=True)

        pending = {}

        for model_name in portfolio:
            col_name = f"pred_{model_name}"

            bag_models = []

            try:
                base_pipe = build_pipeline(model_name, X_train)
            except Exception as e:
                logger.warning(
                    f"could not pipe : {model_name} - {e}"
                )
                continue

            oof_preds = np.zeros(len(X_base))

            for train_idx, val_idx in kf.split(X_base):
                X_train_fold, y_train_fold = X_base.iloc[train_idx], y_train.iloc[train_idx]
                X_val_fold = X_base.iloc[val_idx]

                pipe_fold = clone(base_pipe)
                pipe_fold.fit(X_train_fold, y_train_fold)
                bag_models.append(pipe_fold)
                oof_preds[val_idx] = pipe_fold.predict(X_val_fold)

            self._models_for_default_preds_as_features[model_name] = bag_models

            val_preds = np.mean([m.predict(X_val) for m in bag_models], axis=0)

            if col_name in X_train.columns:
                logger.warning(
                    f"overwriting column {col_name}"
                )

            pending[col_name] = (oof_preds, val_preds, bag_models)

            logger.debug(
                f"calculated OOF: {col_name}"
            )

        for col_name, (oof_preds, val_preds, bag_models) in pending.items():
            X_train[col_name] = oof_preds
            X_val[col_name] = val_preds

        logger.info(
            f"added default OOF predictions as features: {list(self._models_for_default_preds_as_features.keys())}"
        )

        return X_train, X_val

    def _add_post_hpo_preds_as_features(
        self,
        X_train,
        y_train,
        X_val,
        trained_models,
        portfolio,
    ):
        """
        novelty oder was
        """
        kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=self.seed,
        )

        X_base = X_train.copy(deep=True)

        pending = {}

        for model_name, model in zip(portfolio, trained_models):
            col_name = f"pred_post_hpo_{model_name}"

            bag_models = []

            params = getattr(model, "_hpo_params", {})

            try:
                if not params and not isinstance(model, Pipeline):
                    base_pipe = clone(model)
                else:
                    base_pipe = build_pipeline(model_name, X_train, params=params)
            except Exception as e:
                logger.warning(
                    f"could not pipe : {model_name} - {e}"
                )
                continue

            oof_preds = np.zeros(len(X_base))

            for train_idx, val_idx in kf.split(X_base):
                X_train_fold, y_train_fold = X_base.iloc[train_idx], y_train.iloc[train_idx]
                X_val_fold = X_base.iloc[val_idx]

                pipe_fold = clone(base_pipe)
                pipe_fold.fit(X_train_fold, y_train_fold)
                bag_models.append(pipe_fold)
                oof_preds[val_idx] = pipe_fold.predict(X_val_fold)

            self._models_for_post_hpo_preds_as_features[model_name] = bag_models

            val_preds = np.mean([m.predict(X_val) for m in bag_models], axis=0)

            if col_name in X_train.columns:
                logger.warning(
                    f"overwriting column {col_name}"
                )

            pending[col_name] = (oof_preds, val_preds, bag_models)

            logger.debug(
                f"calculated post-HPO OOF: {col_name}"
            )

        for col_name, (oof_preds, val_preds, bag_models) in pending.items():
            X_train[col_name] = oof_preds
            X_val[col_name] = val_preds

        logger.info(
            f"added post-HPO OOF predictions as features: {list(self._models_for_post_hpo_preds_as_features.keys())}"
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

        val_scores = []

        for name, preds in zip(portfolio, val_preds_list):
            model_r2 = r2_score(y_val, preds)
            val_scores.append(model_r2)
            logger.info(f"validation r2 for {name}: {model_r2:.{self._precision}f}")

        self._val_scores = val_scores

        val_preds = aggregate_predictions(val_preds_list, weights=val_scores)

        val_r2 = r2_score(y_val, val_preds)

        logger.info(f"validation r2 after aggregation: {val_r2:.{self._precision}f}")

        if self.add_post_hpo_preds_as_features:
            X_train, X_val = self._add_post_hpo_preds_as_features(
                X_train,
                y_train,
                X_val,
                trained_models,
                portfolio,
            )

        if self.use_bo_tabpfn_surrogate:
            logger.info("Using BO with TabPFN surrogate")

            # Run the Bayesian Optimization with TabPFN surrogate
            bo_tabpfn_trained_model = run_bo_tabpfn(
                X_train,
                y_train,
                X_val,
                y_val,
            )
            self._bo_tabpfn_fitted_model = bo_tabpfn_trained_model

        elif self.use_secondary_hpo and self.add_post_hpo_preds_as_features:
            retrained_models = []
            for model_name in portfolio:
                try:
                    model = fit_model(
                        model_name,
                        X_train,
                        y_train,
                        hpo_method=self.hpo_method,
                        seed=self.seed,
                    )
                    retrained_models.append(model)
                except Exception as e:
                    logger.warning(f"Skipping model '{model_name}' due to error: {e}")

            if retrained_models:
                self._models = retrained_models
                trained_models = retrained_models
            else:
                logger.warning("No models were successfully retrained after HPO.")

        elif self.add_post_hpo_preds_as_features:
            retrained_models = []
            for model_name, first_model in zip(portfolio, trained_models):
                try:
                    params = getattr(first_model, "_hpo_params", {})

                    if not params and not isinstance(first_model, Pipeline):
                        cloned = clone(first_model)
                        cloned.fit(X_train, y_train)
                        retrained_models.append(cloned)
                        continue

                    new_pipeline = build_pipeline(
                        model_name,
                        X_train,
                        params=params,
                    )
                    new_pipeline.fit(X_train, y_train)
                    new_pipeline._hpo_params = params
                    retrained_models.append(new_pipeline)
                except Exception as e:
                    logger.warning(f"Skipping model '{model_name}' due to error: {e}")

            if retrained_models:
                self._models = retrained_models
                trained_models = retrained_models
            else:
                logger.warning("No models retrained in post-HPO phase; keeping first-pass models.")

        if self.add_post_hpo_preds_as_features or self.use_bo_tabpfn_surrogate:
            try:
                val_preds_list = [m.predict(X_val) for m in self._models]
                val_scores = []
                for name, preds in zip(portfolio, val_preds_list):
                    model_r2 = r2_score(y_val, preds)
                    val_scores.append(model_r2)
                    logger.info(f"validation r2 for {name} after post-HPO: {model_r2:.{self._precision}f}")
                self._val_scores = val_scores
                val_preds = aggregate_predictions(val_preds_list, weights=val_scores)
                val_r2 = r2_score(y_val, val_preds)
                logger.info(
                    f"validation r2 after post-HPO aggregation: {val_r2:.{self._precision}f}"
                )
            except Exception as e:
                logger.warning(f"...: {e}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the ensemble of all fitted models.
        Returns averaged predictions.
        """
        if not self._models:
            raise ValueError(".fit() must be called before .predict()")

        X = X.copy(deep=True)

        if self.add_default_preds_as_features:
            for model_name, pipe in self._models_for_default_preds_as_features.items():
                col_name = f"pred_{model_name}"
                if col_name in X.columns:
                    logger.warning(
                        f"overwriting column {col_name}"
                    )
                if isinstance(pipe, list):
                    X[col_name] = np.mean([m.predict(X) for m in pipe], axis=0)
                else:
                    X[col_name] = pipe.predict(X)

        if self.add_post_hpo_preds_as_features:
            for model_name, model in self._models_for_post_hpo_preds_as_features.items():
                col_name = f"pred_post_hpo_{model_name}"
                if col_name in X.columns:
                    logger.warning(
                        f"overwriting column {col_name}"
                    )
                try:
                    if isinstance(model, list):
                        X[col_name] = np.mean([m.predict(X) for m in model], axis=0)
                    else:
                        X[col_name] = model.predict(X)
                except Exception as e:
                    logger.warning(
                        f"... {model_name} - {e}"
                    )

        # print(X.head())

        if self.use_bo_tabpfn_surrogate:
            preds = self._bo_tabpfn_fitted_model.predict(X)
        else:
            test_preds_list = [model.predict(X) for model in self._models]
            preds = aggregate_predictions(test_preds_list, weights=getattr(self, "_val_scores", None))

        return preds
