"""Solid-state electrolyte conductivity prediction web app (Streamlit).

Run from the project root:
    streamlit run app.py

The app reuses the exact descriptor pipeline defined in scripts/common.py and
loads the fitted scikit-learn pipelines (scaler + regressor) from model/.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pymatgen.core import Composition


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common import (  # noqa: E402
    FORMULA_COL,
    RESULTS_DIR,
    TARGET_COL,
    TEMP_COL,
    build_feature_data_for_inputs,
    prepare_feature_data,
)


MODEL_LABELS = {
    "xgb": "XGBoost 梯度提升树",
    "catboost": "CatBoost 梯度提升树",
    "rf": "随机森林 Random Forest",
    "mlp": "神经网络 MLP",
    "svr": "支持向量回归 SVR",
}

EXAMPLES = [
    "Li3PS4",
    "Li6PS5Cl",
    "Li0.5La0.5TiO3",
    "Li7La3Zr2O12",
    "Li1.3Al0.3Ti1.7(PO4)3",
    "(Li2S)0.6(SiS2)0.4",
]


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                linear-gradient(180deg, #060b18 0%, #0a1628 42%, #0d2038 100%);
            color: #dce9f7;
        }
        [data-testid="stHeader"] { background: rgba(6, 11, 24, 0.0); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1424 0%, #0d2033 100%);
            border-right: 1px solid rgba(118, 190, 255, 0.18);
        }
        h1, h2, h3 { color: #eef6ff; letter-spacing: 0px !important; }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 700;
            line-height: 1.15;
            color: #eaf6ff;
            margin: 0 0 0.25rem 0;
            text-shadow: 0 0 18px rgba(56, 189, 248, 0.25);
        }
        .hero-sub {
            color: #9fb7d1;
            font-size: 0.95rem;
            line-height: 1.6;
            margin: 0 0 1.1rem 0;
        }
        .badge-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
        .badge {
            color: #bde7ff;
            background: rgba(23, 106, 175, 0.16);
            border: 1px solid rgba(85, 190, 255, 0.32);
            border-radius: 8px;
            padding: 0.18rem 0.58rem;
            font-size: 0.72rem;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(18, 54, 88, 0.72), rgba(13, 32, 58, 0.85));
            border: 1px solid rgba(95, 195, 255, 0.28);
            border-radius: 8px;
            padding: 0.9rem 1rem 0.8rem 1rem;
            box-shadow: 0 0 22px rgba(31, 164, 255, 0.12), inset 0 0 14px rgba(31, 164, 255, 0.04);
        }
        [data-testid="stMetricLabel"] { color: #93b8dc; }
        [data-testid="stMetricValue"] { color: #f1fbff; }
        [data-testid="stMetricDelta"] { color: #6ee7b7; }
        .result-note {
            margin-top: 0.55rem;
            color: #8da9c6;
            font-size: 0.82rem;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(10, 24, 44, 0.62);
            border: 1px solid rgba(89, 180, 255, 0.18);
            border-radius: 8px;
        }
        .tech-line {
            background: linear-gradient(90deg, #38bdf8, rgba(56, 189, 248, 0));
            height: 1px;
            margin: 0.2rem 0 0.8rem 0;
        }
        .stButton > button, .stFormSubmitButton > button {
            border: 1px solid rgba(88, 187, 255, 0.48);
            background: linear-gradient(180deg, #135d93, #0b3557);
            color: #ecf9ff;
            border-radius: 8px;
            font-weight: 600;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            border-color: #6ed0ff;
            box-shadow: 0 0 16px rgba(56, 189, 248, 0.28);
            color: #ffffff;
        }
        div[data-testid="stTextInput"] input, div[data-baseweb="select"] > div {
            background-color: rgba(9, 22, 42, 0.86);
            color: #e7f4ff;
            border-radius: 8px;
        }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-title">固态电解质离子电导率预测</div>
        <div class="hero-sub">
            输入材料组分与温度，实时获得 lg σ 预测值
        </div>
        <div class="badge-row">
            <span class="badge">固态电解质</span>
            <span class="badge">pymatgen 描述符</span>
            <span class="badge">105 维特征</span>
            <span class="badge">单位 K</span>
        </div>
        <div class="tech-line"></div>
        """,
        unsafe_allow_html=True,
    )


def discover_model_files() -> list[Path]:
    files = sorted(MODEL_DIR.glob("*_pipeline.joblib"))
    return files


@st.cache_resource(show_spinner="Loading composition descriptors...")
def load_feature_reference() -> pd.DataFrame:
    return prepare_feature_data()


@st.cache_resource(show_spinner="Loading ML model...")
def load_model(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def load_metrics() -> pd.DataFrame | None:
    metrics_path = RESULTS_DIR / "metrics.csv"
    if not metrics_path.exists():
        return None
    return pd.read_csv(metrics_path)


def model_key_from_path(path: Path) -> str:
    return path.name.replace("_pipeline.joblib", "")


def canonical_formula(formula: str) -> str:
    return Composition(formula).formula


def validate_temperature(temperature: float) -> list[str]:
    warnings = []
    if temperature < 250:
        warnings.append("温度低于 250 K，已明显偏离训练集区间（278-1146 K），结果属于外推。")
    elif temperature > 1200:
        warnings.append("温度高于 1200 K，已明显偏离训练集区间（278-1146 K），结果属于外推。")
    return warnings


def element_vocabulary(reference: pd.DataFrame) -> set[str]:
    cols = [col for col in reference.columns if col.startswith("frac_")]
    return {col[len("frac_") :] for col in cols}


def predict_single(pipeline, feature_row: pd.DataFrame) -> float:
    prediction = pipeline.predict(feature_row.to_numpy(dtype=float))[0]
    return float(prediction)


def format_scientific(value: float) -> str:
    if value <= 0:
        return f"{value:.4e}"
    return f"{value:.3e}"


def render_prediction(
    pipeline,
    model_display: str,
    formula_input: str,
    temperature: float,
    reference: pd.DataFrame,
) -> None:
    try:
        comp = Composition(formula_input)
        formula_canonical = comp.formula
        unknown = sorted({symbol.symbol for symbol in comp.elements} - element_vocabulary(reference))
        if unknown:
            st.error(
                f"模型训练集中未包含元素：{', '.join(unknown)}。"
                "请更换为训练数据中的 Li、S、P、O、La、Zr、Ti 等元素组分。"
            )
            return
        features = build_feature_data_for_inputs(
            [formula_canonical], [float(temperature)], reference
        )
        pred_lg = predict_single(pipeline, features)
        pred_sigma = 10.0 ** pred_lg
    except ValueError as exc:
        st.error(f"化学式无法解析：{exc}")
        return
    except Exception as exc:  # noqa: BLE001 - show actionable UI error
        st.error(f"特征计算失败：{exc}")
        return

    with st.container(border=True):
        left, right = st.columns([1.35, 1])
        with left:
            st.metric("预测 lg σ", f"{pred_lg:.4f}", help="lg σ = log10(σ)，σ 单位为 S/cm")
            st.caption(
                f"对应离子电导率 σ ≈ {format_scientific(pred_sigma)} S/cm"
            )
        with right:
            st.markdown(
                f"""
                <div style="padding-top:0.2rem; font-size:0.9rem;">
                  <b>组分</b><br>{formula_canonical}<br><br>
                  <b>温度</b><br>{temperature:.1f} K
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""
            <div class="result-note">
              模型：{model_display} · 预测值为 lg σ（S/cm）；示例 σ = 10^(lg σ)。
              电导率越高（数值越接近 0），离子传导性能越好。
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar(model_files: list[Path], metrics: pd.DataFrame | None) -> None:
    st.sidebar.markdown("### 预测模型")
    options = {model_key_from_path(path): str(path) for path in model_files}
    default_key = "xgb" if "xgb" in options else next(iter(options))
    selected_key = st.sidebar.selectbox(
        "选择已训练模型",
        list(options.keys()),
        format_func=lambda key: MODEL_LABELS.get(key, key),
        index=list(options.keys()).index(default_key),
    )
    st.session_state["selected_model_key"] = selected_key

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 模型性能")
    if metrics is not None and not metrics.empty:
        row = metrics[metrics["model"] == selected_key]
        if not row.empty:
            best = row.iloc[0]
            st.sidebar.markdown(
                f"""
                - R²: **{best['r2_test']:.4f}**
                - RMSE: **{best['rmse_test']:.4f}**
                - MAE: **{best['mae_test']:.4f}**
                """,
                unsafe_allow_html=True,
            )
            st.sidebar.caption("按化学式分组的留出集测试结果")
        else:
            st.sidebar.caption("该模型暂无测试指标")
    else:
        st.sidebar.caption("未找到 results/metrics.csv")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 说明")
    st.sidebar.caption(
        "训练域：278-1146 K；组分使用 pymatgen 解析并生成元素分数、"
        "电负性/原子半径等组成统计特征，与训练脚本 scripts/common.py 完全一致。"
    )
    return selected_key


def main() -> None:
    st.set_page_config(
        page_title="Yili Solid-State Conductivity Predictor",
        page_icon=":material/battery_charging_full:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_style()
    render_header()

    model_files = discover_model_files()
    if not model_files:
        st.error("model/ 目录下未找到 *_pipeline.joblib 模型文件，请先运行 scripts/ 下的训练脚本。")
        st.stop()

    metrics = load_metrics()
    selected_key = render_sidebar(model_files, metrics)
    selected_path = Path(
        next(str(path) for path in model_files if model_key_from_path(path) == selected_key)
    )

    try:
        reference = load_feature_reference()
    except Exception as exc:  # noqa: BLE001
        st.error(f"无法加载数据/特征：{exc}")
        st.stop()

    st.markdown("#### 输入参数")
    with st.container(border=True):
        sample_cols = st.columns(min(len(EXAMPLES), 6))
        for col, example in zip(sample_cols, EXAMPLES):
            if col.button(example, width="stretch", key=f"sample_{example}"):
                st.session_state["formula_input"] = example
                st.rerun()

        with st.form("prediction_form", border=False):
            formula_input = st.text_input(
                "材料组分 / 化学式",
                key="formula_input",
                placeholder="例如 Li3PS4、Li6PS5Cl、Li0.5La0.5TiO3",
                help="支持 pymatgen 可解析的化学式写法，如 Li3PS4、Li 3 P S4、(Li2S)0.6(SiS2)0.4。",
            )
            temperature = st.number_input(
                "温度 T",
                min_value=0.0,
                max_value=2000.0,
                value=298.0,
                step=1.0,
                format="%.1f",
                help="绝对温度，单位为开尔文（K）。",
            )
            submitted = st.form_submit_button("开始预测", width="stretch")

    if temperature <= 0:
        st.error("温度必须大于 0 K。")
        return
    for warning in validate_temperature(temperature):
        st.warning(warning)

    if submitted:
        formula_input = (formula_input or "").strip()
        if not formula_input:
            st.error("请输入材料组分，例如 Li3PS4。")
            return
        try:
            canonical_formula(formula_input)
        except Exception:  # noqa: BLE001
            st.error(
                "组分格式无法解析。请使用标准化学式（如 Li3PS4、Li6PS5Cl、"
                "Li0.5La0.5TiO3）或由 pymatgen 支持的写法。"
            )
            return

        try:
            pipeline = load_model(str(selected_path))
        except Exception as exc:  # noqa: BLE001
            st.error(f"模型加载失败：{exc}")
            return
        render_prediction(
            pipeline,
            MODEL_LABELS.get(selected_key, selected_key),
            formula_input,
            temperature,
            reference,
        )

    with st.expander("查看多模型一致性参考"):
        st.caption(
            "以下数值用于观察不同算法是否给出相近结论，属于交叉验证参考，"
            "不是经校准的不确定度区间。"
        )
        formula_input_preview = st.session_state.get("formula_input", "").strip()
        if submitted and formula_input_preview:
            try:
                comp = Composition(formula_input_preview)
                features = build_feature_data_for_inputs(
                    [comp.formula], [float(temperature)], reference
                )
                rows = []
                for path in model_files:
                    key = model_key_from_path(path)
                    pipe = load_model(str(path))
                    value = predict_single(pipe, features)
                    rows.append(
                        {
                            "model": MODEL_LABELS.get(key, key),
                            "pred_lg_sigma": round(value, 4),
                            "sigma_S_per_cm": format_scientific(10.0**value),
                        }
                    )
                compare_df = pd.DataFrame(rows)
                st.dataframe(
                    compare_df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "pred_lg_sigma": st.column_config.NumberColumn(
                            "预测 lg σ", format="%.4f"
                        ),
                    },
                )
                spread = float(compare_df["pred_lg_sigma"].max() - compare_df["pred_lg_sigma"].min())
                st.caption(f"各模型预测跨度：{spread:.3f} lg 单位")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"无法生成一致性参考：{exc}")
        else:
            st.caption("提交一次预测后，可在此查看 RF / XGB / CatBoost / MLP / SVR 的横向结果。")


if __name__ == "__main__":
    main()
