"""Exploratory data analysis for the raw conductivity dataset.

Run from the project root:
    python scripts/explore_data.py

Output:
    results/eda_summary.json  - machine-readable summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FEATURE_CACHE_PATH,
    FORMULA_COL,
    RAW_DATA_PATH,
    RESULTS_DIR,
    TARGET_COL,
    TEMP_COL,
    get_feature_columns,
    load_raw_data,
    prepare_feature_data,
)


def main() -> None:
    raw = load_raw_data()
    feature = prepare_feature_data()

    summary: dict[str, object] = {}
    summary["raw_file"] = str(RAW_DATA_PATH)
    summary["raw_rows"] = int(len(raw))
    summary["raw_columns"] = list(raw.columns)
    summary["missing_values"] = {
        str(col): int(raw[col].isna().sum()) for col in raw.columns
    }
    summary["duplicate_rows"] = int(raw.duplicated().sum())
    summary["unique_formulas"] = int(raw[FORMULA_COL].nunique())
    summary["unique_formula_temp_pairs"] = int(
        raw[[FORMULA_COL, TEMP_COL]].drop_duplicates().shape[0]
    )

    parse_failures = []
    for formula in raw[FORMULA_COL]:
        try:
            Composition(formula)
        except Exception as exc:  # noqa: BLE001
            parse_failures.append({"formula": formula, "error": str(exc)})
    summary["pymatgen_parse_failures"] = parse_failures

    conductivity_col = "conductivity"
    target_log10_diff = np.abs(
        np.asarray(raw[TARGET_COL]) - np.log10(np.asarray(raw[conductivity_col]))
    )
    summary["target_log10_consistency"] = {
        "max_abs_diff": float(np.max(target_log10_diff)),
        "mean_abs_diff": float(np.mean(target_log10_diff)),
        "note": "target column is trusted as-is; differences come from rounding in source",
    }

    summary["temperature"] = {
        "min": float(raw[TEMP_COL].min()),
        "max": float(raw[TEMP_COL].max()),
        "unique_values": int(raw[TEMP_COL].nunique()),
        "quantiles": raw[TEMP_COL].quantile([0.25, 0.5, 0.75]).round(2).to_dict(),
    }
    summary["target"] = {
        "min": float(raw[TARGET_COL].min()),
        "max": float(raw[TARGET_COL].max()),
        "mean": float(raw[TARGET_COL].mean()),
        "std": float(raw[TARGET_COL].std()),
        "unit": "lg_conductivity / log10(S/cm)",
    }

    repeated = raw[FORMULA_COL].value_counts()
    summary["rows_per_formula"] = {
        "max": int(repeated.max()),
        "mean": float(repeated.mean()),
        "median": float(repeated.median()),
    }
    summary["formula_with_most_rows"] = repeated.head(10).astype(int).to_dict()

    summary["split"] = feature["split"].value_counts().to_dict()
    summary["split_formula_groups"] = (
        feature.groupby("split")[FORMULA_COL].nunique().to_dict()
    )
    summary["feature_count"] = len(get_feature_columns(feature))
    summary["feature_cache_file"] = str(FEATURE_CACHE_PATH)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "eda_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print("Data shape:", raw.shape)
    print("Columns:", list(raw.columns))
    print("Missing values:", dict(summary["missing_values"]))
    print("Duplicate rows:", summary["duplicate_rows"])
    print("Unique formulas:", summary["unique_formulas"])
    print("pymatgen parse failures:", len(parse_failures))
    print("Temperature range:", summary["temperature"]["min"], "-", summary["temperature"]["max"])
    print("Target range:", summary["target"]["min"], "-", summary["target"]["max"])
    print("Split rows:", summary["split"])
    print("Split formula groups:", summary["split_formula_groups"])
    print(f"Feature count: {summary['feature_count']}")
    print("Saved:", RESULTS_DIR / "eda_summary.json")


if __name__ == "__main__":
    main()
