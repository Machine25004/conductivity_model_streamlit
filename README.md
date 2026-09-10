# 固态电解质离子电导率预测与 Web 展示

项目将材料组分（化学式）与温度作为输入，输出 `lg σ`（`log10(S/cm)`）预测值。模型在 `data/raw_data.xlsx`（1057 条，720 个化学式，无缺失/重复）上训练，特征由 pymatgen 从组成生成，与温度共同构成 105 维输入。

## 目录

```text
conductivity_model/
├── app.py                     # Streamlit Web 应用
├── requirements.txt
├── .streamlit/config.toml     # 深色主题
├── data/raw_data.xlsx         # 原始数据
├── data/features.csv          # 特征缓存（可自动重建）
├── model/                     # 训练好的 pipeline 与模型
├── scripts/                   # 训练与特征工程脚本
└── results/                   # 指标、预测值与对比结果
```

## 安装

```bash
python -m pip install -r requirements.txt
```

本机测试环境：Python 3.13，pandas 2.3.3，scikit-learn 1.9.0，pymatgen 2026.5.4，xgboost 3.4.1，catboost 1.2.10，streamlit 1.61.1。

## 启动 Web 应用

```bash
streamlit run app.py
```

浏览器访问 http://localhost:8501 。模型文件位于 `model/*_pipeline.joblib`；Web 应用只加载带预处理器的 pipeline，因此新输入会先走与训练一致的描述符构建和标准化。

Windows 下也可以直接双击 `快捷启动.bat`：脚本会自动启动 Streamlit 并打开浏览器；如果 8501 端口已有服务在运行，则只打开浏览器页面。

应用当前已支持模型：

| 模型 | 测试 R² | 测试 RMSE | 测试 MAE |
| --- | ---: | ---: | ---: |
| XGBoost | 0.7175 | 1.0154 | 0.7071 |
| CatBoost | 0.6807 | 1.0795 | 0.7393 |
| Random Forest | 0.6499 | 1.1303 | 0.8000 |
| MLP | 0.5774 | 1.2419 | 0.8629 |
| SVR | 0.5658 | 1.2588 | 0.8660 |

评价方式为按化学式分组、互不重叠的 80/20 留出集，避免同一组分同时出现在训练集与测试集。

## 特征与模型调用

- `scripts/common.py`：数据加载、pymatgen 化学式解析、组成描述符、分组划分、评估与保存。
- 描述符：54 个元素分数、电负性/原子质量/原子半径/门捷列夫序号/周期/族/电子亲和能/阳离子半径/氧化态等加权统计、熵与阴离子分数，以及连续温度。
- 特征缓存为 `data/features.csv`；若更换原始数据，请删除该缓存后重新运行训练脚本。
- `model/*_pipeline.joblib` 是 `StandardScaler + Regressor` 完整 pipeline；`xgb_model.json`、`catboost_model.cbm` 是原生模型文件，单独加载时不包含标准化器，Web 应用不使用这两个文件。

## 重训

```bash
python scripts/explore_data.py
python scripts/train_rf.py
python scripts/train_mlp.py
python scripts/train_xgb.py
python scripts/train_catboost.py
python scripts/train_svr.py
python scripts/train_stacking.py
```

模型、`results/metrics.csv` 与逐条预测会写入 `model/` 和 `results/`。

## 素材说明

页面未引用外部图片或图标资源，全部视觉由 Streamlit 原生组件与内嵌 CSS 实现，不涉及第三方版权素材。公司名称仅作文字标识，未嵌入未经授权的 logo。
