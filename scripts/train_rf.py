"""Train and evaluate a Random Forest regressor.

Dependencies: pandas, numpy, scikit-learn, pymatgen, joblib
Run from the project root:
    python scripts/train_rf.py

The pipeline includes a StandardScaler only so every saved artifact has the
same raw-input interface; scaling is not required for tree models.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import RANDOM_STATE, train_test_model


def main() -> None:
    model_name = "rf"
    params = {
        "n_estimators": 600,
        "max_depth": None,
        "min_samples_leaf": 2,
        "min_samples_split": 5,
        "max_features": 0.6,
        "oob_score": False,
        "random_state": RANDOM_STATE,
    }
    regressor = RandomForestRegressor(**params, n_jobs=1)
    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", regressor)])
    train_test_model(pipeline, model_name, params)


if __name__ == "__main__":
    main()
