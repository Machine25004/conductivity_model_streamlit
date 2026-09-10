"""Train and evaluate a support vector regressor with RBF kernel.

Dependencies: pandas, numpy, scikit-learn, pymatgen, joblib
Run from the project root:
    python scripts/train_svr.py

Standard scaling is mandatory for SVR; the scaler is saved inside the pipeline.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import RANDOM_STATE, train_test_model


def main() -> None:
    model_name = "svr"
    params = {
        "kernel": "rbf",
        "C": 20.0,
        "epsilon": 0.05,
        "gamma": "scale",
        "cache_size": 500,
        "random_state": RANDOM_STATE,
    }
    # Note: SVR does not accept random_state in all scikit-learn versions.
    svr_kwargs = {key: value for key, value in params.items() if key != "random_state"}
    regressor = SVR(**svr_kwargs)
    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", regressor)])
    train_test_model(pipeline, model_name, params)


if __name__ == "__main__":
    main()
