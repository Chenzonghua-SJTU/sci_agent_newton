import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import signal, stats, optimize
from sklearn import linear_model, metrics, preprocessing
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process(payload: dict) -> dict:
    action = payload.get("action")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    if action != "fit_relationship_model":
        raise ValueError(f"Unsupported action: {action}")

    experiment_ids = params.get("experiment_ids", [])
    target_series = params.get("target_series")
    basis_expressions = params.get("basis_expressions", [])
    prediction_name = params.get("prediction_name")
    residual_name = params.get("residual_name")
    include_intercept = params.get("include_intercept", True)

    if not target_series or not prediction_name or not residual_name:
        raise ValueError("target_series, prediction_name, residual_name must be provided.")
    if not basis_expressions:
        raise ValueError("basis_expressions must be a non-empty list.")
    if not experiment_ids:
        raise ValueError("experiment_ids must be provided.")

    derived_series = []
    figures = []
    metrics_dict = {}
    observation_lines = []

    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment ID '{eid}' not found in payload.")
        exp = experiments[eid]
        series = exp.get("series", {})
        available = exp.get("available_series", list(series.keys()))

        # 检查所需序列是否存在
        if target_series not in series:
            raise ValueError(f"Experiment {eid}: target series '{target_series}' not found.")
        for b in basis_expressions:
            if b not in series:
                raise ValueError(f"Experiment {eid}: basis expression '{b}' not found.")

        t = series.get("t", None)
        if t is None:
            raise ValueError(f"Experiment {eid}: no 't' series available.")

        y = np.array(series[target_series], dtype=float)
        X_list = [np.array(series[b], dtype=float) for b in basis_expressions]
        # 检查长度一致性
        n = len(t)
        if len(y) != n:
            raise ValueError(f"Experiment {eid}: length of target_series ({len(y)}) != t length ({n}).")
        for i, (b, x) in enumerate(zip(basis_expressions, X_list)):
            if len(x) != n:
                raise ValueError(f"Experiment {eid}: basis '{b}' length ({len(x)}) != t length ({n}).")

        X = np.column_stack(X_list)

        # 拟合
        model = linear_model.LinearRegression(fit_intercept=include_intercept)
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred

        # 计算指标
        n_samples = n
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            r2 = 1.0  # 完美预测（平凡情况）
        else:
            r2 = 1.0 - ss_res / ss_tot
        rmse = math.sqrt(ss_res / n_samples)
        mae = np.mean(np.abs(residuals))

        # 记录系数
        coeffs = model.coef_.tolist()
        intercept = model.intercept_ if include_intercept else 0.0
        coeff_dict = {}
        if include_intercept:
            coeff_dict["intercept"] = intercept
        for b, c in zip(basis_expressions, coeffs):
            coeff_dict[b] = c

        # 构建派生序列
        # prediction
        derived_series.append({
            "experiment_id": eid,
            "name": prediction_name,
            "values": y_pred.tolist(),
            "source_name": f"linear_model({', '.join(basis_expressions)})",
            "provenance": "generated data processor: fit_relationship_model",
            "description": f"Prediction of {target_series} using basis {basis_expressions}, intercept={include_intercept}"
        })
        # residual
        derived_series.append({
            "experiment_id": eid,
            "name": residual_name,
            "values": residuals.tolist(),
            "source_name": f"{target_series} - {prediction_name}",
            "provenance": "generated data processor: fit_relationship_model",
            "description": f"Residual of {target_series} fit"
        })

        # 记录指标
        exp_metrics = {}
        for key, val in coeff_dict.items():
            exp_metrics[f"{eid}_coef_{key}"] = val
        exp_metrics[f"{eid}_R2"] = r2
        exp_metrics[f"{eid}_RMSE"] = rmse
        exp_metrics[f"{eid}_MAE"] = mae
        metrics_dict.update(exp_metrics)

        # 构造观察行
        coeff_str = ", ".join([f"{k}={v:.4f}" for k, v in coeff_dict.items()])
        obs_line = (f"Experiment {eid}: {target_series} = {coeff_str}; "
                    f"R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        observation_lines.append(obs_line)

    # 可选的全局查看图：预测 vs 真实
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    for eid in experiment_ids:
        exp = experiments[eid]
        series = exp.get("series", {})
        # 读取刚生成的预测和真实（注意预测序列尚未回到payload，但我们可以在本地计算）
        y_true = np.array(series[target_series], dtype=float)
        # 重新计算预测（重复拟合，但简单起见，我们直接从已有序列获取？不行，因为预测还没加入payload。我们可以重新拟合，但会重复计算。
        # 为简化，我们略过画图，因为这不是强制要求。标准action可以不画图。
        pass
    plt.close(fig)  # 防止内存泄漏

    observation = "执行关系模型拟合（线性回归）:\n" + "\n".join(observation_lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics_dict
    }
