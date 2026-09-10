"""Train and evaluate a CatBoost regressor.

Dependencies: pandas, numpy, catboost, pymatgen, joblib
Install CatBoost with:
    pip install catboost
Run from the project root:
    python scripts/train_catboost.py
"""

from __future__ import annotations

from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import MODEL_DIR, RANDOM_STATE, train_test_model


def main() -> None:
    model_name = "catboost"
    params = {
        "iterations": 1500,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSE",
        "random_seed": RANDOM_STATE,
        "allow_writing_files": False,
    }
    regressor = CatBoostRegressor(**params, thread_count=1, verbose=False)
    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", regressor)])
    train_test_model(pipeline, model_name, params)
    # Keep a native CatBoost artifact in addition to the reusable pipeline.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pipeline.named_steps["regressor"].save_model(MODEL_DIR / "catboost_model.cbm")


if __name__ == "__main__":
    main()
