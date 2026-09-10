"""Train and evaluate an XGBoost gradient-boosted tree regressor.

Dependencies: pandas, numpy, xgboost, pymatgen, joblib
Install XGBoost with:
    pip install xgboost
Run from the project root:
    python scripts/train_xgb.py
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from common import RANDOM_STATE, save_pipeline, train_test_model


def main() -> None:
    model_name = "xgb"
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.04,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
    }
    regressor = XGBRegressor(**params, n_jobs=1, eval_metric="rmse")
    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", regressor)])
    train_test_model(pipeline, model_name, params)
    # Keep a native XGBoost model artifact in addition to the reusable pipeline.
    xgb_model = pipeline.named_steps["regressor"]
    xgb_model.save_model("model/xgb_model.json")


if __name__ == "__main__":
    main()
