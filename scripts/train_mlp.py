"""Train and evaluate a feed-forward neural network (scikit-learn MLP).

Dependencies: pandas, numpy, scikit-learn, pymatgen, joblib
Run from the project root:
    python scripts/train_mlp.py

Features are standardized before fitting because MLPs are sensitive to scale.
"""

from __future__ import annotations

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import RANDOM_STATE, train_test_model


def main() -> None:
    model_name = "mlp"
    params = {
        "hidden_layer_sizes": (256, 128, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-3,
        "batch_size": 64,
        "learning_rate_init": 2e-3,
        "max_iter": 800,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 40,
        "random_state": RANDOM_STATE,
    }
    regressor = MLPRegressor(**params)
    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", regressor)])
    train_test_model(pipeline, model_name, params)


if __name__ == "__main__":
    main()
