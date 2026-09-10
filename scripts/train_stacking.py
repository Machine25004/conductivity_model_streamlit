"""Train a stacking ensemble of RF, XGBoost, CatBoost and SVR.

Dependencies: pandas, numpy, scikit-learn, xgboost, catboost, pymatgen, joblib
Run from the project root:
    python scripts/train_stacking.py

Internal CV folds are group-aware so the meta-features do not see the same
formula in training and validation folds.
"""

from __future__ import annotations

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from common import RANDOM_STATE, train_test_model


def main() -> None:
    model_name = "stacking"

    rf_params = {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 2,
        "min_samples_split": 5,
        "max_features": 0.6,
        "random_state": RANDOM_STATE,
    }
    xgb_params = {
        "n_estimators": 800,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 2,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "eval_metric": "rmse",
    }
    cat_params = {
        "iterations": 1000,
        "learning_rate": 0.06,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSE",
        "random_seed": RANDOM_STATE,
        "allow_writing_files": False,
    }
    svr_params = {"kernel": "rbf", "C": 10.0, "epsilon": 0.05, "gamma": "scale"}

    estimators = [
        (
            "rf",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", RandomForestRegressor(**rf_params, n_jobs=1)),
                ]
            ),
        ),
        (
            "xgb",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", XGBRegressor(**xgb_params, n_jobs=1)),
                ]
            ),
        ),
        (
            "catboost",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "regressor",
                        CatBoostRegressor(
                            **cat_params, thread_count=1, verbose=False
                        ),
                    ),
                ]
            ),
        ),
        (
            "svr",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", SVR(**svr_params)),
                ]
            ),
        ),
    ]

    final_params = {"alpha": 1.0}
    ensemble = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(**final_params),
        cv=GroupKFold(n_splits=5),
        n_jobs=1,
        passthrough=False,
    )

    train_test_model(
        ensemble,
        model_name,
        {
            "base_estimators": [name for name, _ in estimators],
            "final_estimator": "Ridge",
            "final_params": final_params,
            "cv": "GroupKFold(n_splits=5)",
        },
    )


if __name__ == "__main__":
    main()
