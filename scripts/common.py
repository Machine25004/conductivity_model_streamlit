"""Shared data loading, composition descriptor building, splitting and evaluation."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_DATA_PATH = DATA_DIR / "raw_data.xlsx"
FEATURE_CACHE_PATH = DATA_DIR / "features.csv"
METRICS_PATH = RESULTS_DIR / "metrics.csv"

RANDOM_STATE = 42
TEST_FRACTION = 0.20

FORMULA_COL = "formula"
TEMP_COL = "temperature"
TARGET_COL = "lg_conductivity"
RAW_COLUMNS = [FORMULA_COL, TEMP_COL, "conductivity", TARGET_COL]

# Elements treated as (mostly) anionic when computing the non-metal share.
ANION_ELEMENTS = {"O", "S", "Se", "F", "Cl", "Br", "I", "N"}


def _property_getters() -> dict[str, object]:
    """Elementary properties used for Magpie-style weighted statistics."""
    return {
        "electronegativity": lambda e: float(e.X),
        "atomic_mass": lambda e: float(e.atomic_mass),
        "atomic_radius": lambda e: float(e.atomic_radius),
        "mendeleev_number": lambda e: float(e.mendeleev_no),
        "period": lambda e: float(e.row),
        "group": lambda e: float(e.group),
        "electron_affinity": lambda e: float(e.electron_affinity),
        "avg_cationic_radius": lambda e: float(e.average_cationic_radius),
        "oxidation_state": lambda e: float(np.mean(e.common_oxidation_states)),
    }


PROPERTY_GETTERS = _property_getters()
PROPERTY_STATS = ("mean", "min", "max", "range", "std")


def load_raw_data() -> pd.DataFrame:
    """Read the user-provided Excel file and normalize column names."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_PATH}")
    df = pd.read_excel(RAW_DATA_PATH)
    df = df.iloc[:, : len(RAW_COLUMNS)].copy()
    df.columns = RAW_COLUMNS
    df[FORMULA_COL] = df[FORMULA_COL].astype(str).str.strip()
    df = df[df[FORMULA_COL] != ""]
    df = df.dropna(subset=[FORMULA_COL, TEMP_COL, TARGET_COL])
    df = df.reset_index(drop=True)
    return df


def _parse_compositions(formulas: Sequence[str]) -> list[Composition]:
    """Parse every formula; pymatgen errors are surfaced instead of guessing."""
    parsed: list[Composition] = []
    errors = []
    for formula in formulas:
        try:
            parsed.append(Composition(formula))
        except Exception as exc:  # noqa: BLE001 - report the offending formula
            errors.append((formula, str(exc)))
    if errors:
        detail = "; ".join(f"{f}: {msg}" for f, msg in errors[:20])
        raise ValueError(f"pymatgen could not parse {len(errors)} formula(s): {detail}")
    return parsed


def _weighted_std(values: np.ndarray, weights: np.ndarray, mean: float) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0
    variance = float(np.sum(weights * (values - mean) ** 2) / total)
    return float(np.sqrt(max(variance, 0.0)))


def build_descriptor_frame(
    formulas: Sequence[str],
    temperatures: Sequence[float],
    element_symbols: Sequence[str],
) -> pd.DataFrame:
    """Build composition descriptors from pymatgen and append temperature.

    The output contains one-hot element fractions, weighted statistics of
    elementary properties, simple mixture descriptors, and raw temperature.
    """
    if len(formulas) != len(temperatures):
        raise ValueError("formulas and temperatures must have the same length")

    parsed = _parse_compositions(list(formulas))
    symbols_list = []
    for comp in parsed:
        symbols_list.append(sorted(symbol.symbol for symbol in comp.elements))

    props = {
        symbol: {name: getter(Element(symbol)) for name, getter in PROPERTY_GETTERS.items()}
        for symbol in element_symbols
    }

    rows = []
    for comp, symbols, temp in zip(parsed, symbols_list, temperatures):
        frac = comp.fractional_composition
        frac_dict = {symbol: float(frac.get_atomic_fraction(Element(symbol))) for symbol in symbols}
        row: dict[str, float] = {TEMP_COL: float(temp)}

        row["feat_n_elements"] = len(symbols)
        fractions = np.array(list(frac_dict.values()), dtype=float)
        entropy = -float(np.sum(fractions * np.log(fractions)))
        row["feat_entropy"] = entropy
        row["feat_max_fraction"] = float(np.max(fractions))
        row["feat_min_fraction"] = float(np.min(fractions))
        row["feat_anion_fraction"] = float(
            sum(v for symbol, v in frac_dict.items() if symbol in ANION_ELEMENTS)
        )

        for symbol in element_symbols:
            row[f"frac_{symbol}"] = frac_dict.get(symbol, 0.0)

        for prop_name in PROPERTY_GETTERS:
            values = np.array([props[symbol][prop_name] for symbol in symbols], dtype=float)
            weights = np.array([frac_dict[symbol] for symbol in symbols], dtype=float)
            mean_value = float(np.average(values, weights=weights))
            row[f"feat_{prop_name}_mean"] = mean_value
            row[f"feat_{prop_name}_min"] = float(np.min(values))
            row[f"feat_{prop_name}_max"] = float(np.max(values))
            row[f"feat_{prop_name}_range"] = float(np.max(values) - np.min(values))
            row[f"feat_{prop_name}_std"] = _weighted_std(values, weights, mean_value)

        rows.append(row)

    return pd.DataFrame(rows)


def get_feature_columns(reference: pd.DataFrame) -> list[str]:
    """Return columns consumed by models: composition features + temperature."""
    composition_cols = [
        col for col in reference.columns if col.startswith(("frac_", "feat_"))
    ]
    return composition_cols + [TEMP_COL]


def prepare_feature_data(force_rebuild: bool = False) -> pd.DataFrame:
    """Return cleaned rows with cached pymatgen descriptors and split label."""
    if FEATURE_CACHE_PATH.exists() and not force_rebuild:
        df = pd.read_csv(FEATURE_CACHE_PATH)
        if not df.empty and FORMULA_COL in df.columns and "split" in df.columns:
            return df

    raw = load_raw_data()
    compositions = list(raw[FORMULA_COL])
    parsed = _parse_compositions(compositions)
    element_symbols = sorted(
        {symbol.symbol for comp in parsed for symbol in comp.elements},
        key=lambda s: (Element(s).X, s),
    )
    feature_df = build_descriptor_frame(compositions, raw[TEMP_COL].tolist(), element_symbols)
    df = pd.concat(
        [
            raw[[FORMULA_COL, TARGET_COL]].reset_index(drop=True),
            feature_df.reset_index(drop=True),
        ],
        axis=1,
    )
    df["row_id"] = np.arange(len(df))

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df[FORMULA_COL]))
    split_labels = np.array(["train"] * len(df), dtype=object)
    split_labels[test_idx] = "test"
    df["split"] = split_labels
    ordered_cols = pd.Index(
        [FORMULA_COL, TARGET_COL, "row_id", "split"] + list(feature_df.columns)
    ).drop_duplicates()
    df = df[ordered_cols]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURE_CACHE_PATH, index=False)
    return df


def split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by cached labels. No formula appears in both train and test."""
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    return train, test


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def save_pipeline(estimator, model_name: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{model_name}_pipeline.joblib"
    joblib.dump(estimator, path)


def save_predictions(
    model_name: str,
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = test_df[[FORMULA_COL, TEMP_COL, TARGET_COL]].copy()
    out["actual"] = np.asarray(y_test)
    out["predicted"] = np.asarray(y_pred)
    out["residual"] = out["actual"] - out["predicted"]
    out.to_csv(RESULTS_DIR / f"predictions_{model_name}.csv", index=False)


def record_metrics(
    model_name: str,
    params: dict[str, object],
    metrics: dict[str, float],
    elapsed_sec: float,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "model": model_name,
        "params_json": json.dumps(params, sort_keys=True, default=str),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed_sec, 2),
    }
    row.update(metrics)

    if METRICS_PATH.exists():
        existing = pd.read_csv(METRICS_PATH)
        existing = existing[existing["model"] != model_name]
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])
    updated.to_csv(METRICS_PATH, index=False)


def train_test_model(
    estimator,
    model_name: str,
    params: dict[str, object],
) -> tuple[float, dict[str, float]]:
    """Train, evaluate, persist metrics and save the fitted pipeline."""
    start = time.perf_counter()
    df = prepare_feature_data()
    train_df, test_df = split_frame(df)
    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    print(f"[{model_name}] train rows={len(X_train)}, test rows={len(X_test)}, features={len(feature_cols)}")
    estimator.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    train_pred = estimator.predict(X_train)
    test_pred = estimator.predict(X_test)
    train_metrics = metrics_from_predictions(y_train, train_pred)
    test_metrics = metrics_from_predictions(y_test, test_pred)

    metrics = {
        "r2_train": train_metrics["r2"],
        "rmse_train": train_metrics["rmse"],
        "mae_train": train_metrics["mae"],
        "r2_test": test_metrics["r2"],
        "rmse_test": test_metrics["rmse"],
        "mae_test": test_metrics["mae"],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    record_metrics(model_name, params, metrics, elapsed)
    save_pipeline(estimator, model_name)
    save_predictions(model_name, test_df, y_test, test_pred)

    print(
        f"[{model_name}] R2 test={test_metrics['r2']:.4f}, "
        f"RMSE test={test_metrics['rmse']:.4f}, MAE test={test_metrics['mae']:.4f}"
    )
    print(f"[{model_name}] training finished in {elapsed:.1f}s")
    return elapsed, test_metrics


def build_feature_data_for_inputs(
    formulas: Sequence[str],
    temperatures: Sequence[float],
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Build a prediction-ready feature frame aligned with a trained model."""
    all_cols = list(reference.columns)
    element_cols = [col for col in all_cols if col.startswith("frac_")]
    element_symbols = [col[len("frac_") :] for col in element_cols]
    feature_df = build_descriptor_frame(list(formulas), list(temperatures), element_symbols)
    missing = [col for col in get_feature_columns(reference) if col not in feature_df.columns]
    if missing:
        raise ValueError(
            f"Inputs introduce elements outside the training vocabulary; missing columns: {missing}"
        )
    return feature_df[get_feature_columns(reference)]


def summarize_importance(
    estimator,
    model_name: str,
    feature_cols: Iterable[str],
) -> pd.DataFrame | None:
    """Save tree-native feature importance when available."""
    try:
        raw_model = estimator.named_steps["regressor"] if hasattr(estimator, "named_steps") else estimator
        importance = getattr(raw_model, "feature_importances_", None)
    except AttributeError:
        importance = None
    if importance is None:
        return None
    out = pd.DataFrame({"feature": list(feature_cols), "importance": np.asarray(importance)})
    out = out.sort_values("importance", ascending=False).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / f"importance_{model_name}.csv", index=False)
    return out


def fmt(x: float) -> str:
    return f"{x:.4f}"
