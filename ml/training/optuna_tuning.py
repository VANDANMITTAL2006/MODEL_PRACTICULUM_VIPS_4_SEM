"""Optuna-based model tuning utilities with graceful fallback."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None

from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None


def _cv_rmse(model_builder, X, y, n_splits: int = 4, random_state: int = 42) -> float:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, valid_idx in cv.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model = model_builder()
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        scores.append(np.sqrt(mean_squared_error(y_valid, pred)))
    return float(np.mean(scores))


def tune_xgboost(X, y, n_trials: int = 40, random_state: int = 42) -> Tuple[Dict, object]:
    if optuna is None:
        baseline = XGBRegressor(random_state=random_state, n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=-1)
        baseline.fit(X, y)
        return {"fallback": True}, baseline

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 900),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 2.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }

        def builder():
            return XGBRegressor(**params)

        return _cv_rmse(builder, X, y)

    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_trial.params
    model = XGBRegressor(**best_params, random_state=random_state, n_jobs=-1, verbosity=0)
    model.fit(X, y)
    return best_params, model


def tune_lightgbm(X, y, n_trials: int = 30, random_state: int = 42):
    if LGBMRegressor is None:
        return {"available": False}, None
    if optuna is None:
        model = LGBMRegressor(random_state=random_state, n_estimators=400)
        model.fit(X, y)
        return {"fallback": True}, model

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 900),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 255),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
            "random_state": random_state,
        }

        def builder():
            return LGBMRegressor(**params)

        return _cv_rmse(builder, X, y)

    study.optimize(objective, n_trials=n_trials)
    params = study.best_trial.params
    model = LGBMRegressor(**params, random_state=random_state)
    model.fit(X, y)
    return params, model


def tune_catboost(X, y, n_trials: int = 30, random_state: int = 42):
    if CatBoostRegressor is None:
        return {"available": False}, None
    if optuna is None:
        model = CatBoostRegressor(verbose=0, random_seed=random_state, iterations=500)
        model.fit(X, y)
        return {"fallback": True}, model

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "loss_function": "RMSE",
            "verbose": 0,
            "random_seed": random_state,
        }

        def builder():
            return CatBoostRegressor(**params)

        return _cv_rmse(builder, X, y)

    study.optimize(objective, n_trials=n_trials)
    params = study.best_trial.params
    model = CatBoostRegressor(**params, loss_function="RMSE", verbose=0, random_seed=random_state)
    model.fit(X, y)
    return params, model
