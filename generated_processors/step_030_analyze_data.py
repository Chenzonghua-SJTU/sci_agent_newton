import json
import math
import statistics
from itertools import accumulate
from functools import reduce
from collections import defaultdict, Counter, deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = Path(payload["output_dir"])

    # 实验ID列表：优先从参数获取，否则使用所有实验
    exp_ids = params.get("experiment_ids", list(experiments.keys()))

    # ------------------------------------------------------------
    # 1. 提取每个实验的稳态加速度，并对 constant 实验做指数衰减拟合
    # ------------------------------------------------------------
    data_rows = []          # 用于回归的数据记录
    tau_list = []           # 有效 τ 值列表
    tau_fext_list = []      # 对应的 F_ext
    tau_v0_list = []        # 对应的 v0

    for eid in exp_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        available_series = exp["available_series"]

        F_ext = config["F_ext"]               # 唯一正确的实际外力
        v0 = config["initial_v"]
        force_type = config["force_field_type"]
        t = np.array(series["t"])
        a_center = series.get("a_center", None)
        if a_center is None:
            raise ValueError(f"Experiment {eid} missing series 'a_center'. Available: {available_series}")

        a_center = np.array(a_center)

        # 稳态加速度：最后10个点均值
        steady_a = float(np.mean(a_center[-10:]))

        row = {
            "experiment_id": eid,
            "F_ext": F_ext,
            "v0": v0,
            "steady_a": steady_a,
            "tau": None,
            "A": None,
            "B": None
        }

        # 指数衰减拟合：仅对 constant 实验进行
        if force_type == "constant":
            try:
                # 初始猜测
                a_min = np.min(a_center)
                a_max = np.max(a_center)
                A_init = a_max - a_min
                tau_init = 1.0
                B_init = a_min
                popt, _ = curve_fit(
                    lambda t, A, tau, B: A * np.exp(-t / tau) + B,
                    t, a_center,
                    p0=[A_init, tau_init, B_init],
                    maxfev=5000
                )
                A, tau, B = popt
                if tau > 0:
                    row["tau"] = tau
                    row["A"] = A
                    row["B"] = B
                    tau_list.append(tau)
                    tau_fext_list.append(F_ext)
                    tau_v0_list.append(v0)
            except Exception:
                # 拟合失败，τ 保持 None
                pass

        data_rows.append(row)

    # ------------------------------------------------------------
    # 2. 多元线性回归：steady_a ~ F_ext + v0
    # ------------------------------------------------------------
    X = np.array([[r["F_ext"], r["v0"]] for r in data_rows])
    y = np.array([r["steady_a"] for r in data_rows])
    n, p = X.shape
    dof = n - p - 1

    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    residuals = y - y_pred
    r2 = r2_score(y, y_pred)
    intercept = float(reg.intercept_)
    coef_fext = float(reg.coef_[0])
    coef_v0 = float(reg.coef_[1])

    # 近似 p 值（F 检验）
    if dof > 0 and r2 < 1.0:
        F_stat = r2 / (1.0 - r2) * dof / p
        p_value = 1.0 - f_dist.cdf(F_stat, p, dof)
    else:
        p_value = None

    # 将残差写入每个行
    for i, row in enumerate(data_rows):
        row["residual"] = float(residuals[i])

    # ------------------------------------------------------------
    # 3. τ 与 F_ext, v0 的相关系数
    # ------------------------------------------------------------
    tau_arr = np.array(tau_list)
    tau_fext_arr = np.array(tau_fext_list)
    tau_v0_arr = np.array(tau_v0_list)

    corr_matrix = None
    if len(tau_arr) > 1:
        corr_data = np.vstack([tau_fext_arr, tau_v0_arr, tau_arr])
        corr_matrix = np.corrcoef(corr_data)   # 3x3
    else:
        corr_matrix = None

    # ------------------------------------------------------------
    # 4. 构建观察字符串
    # ------------------------------------------------------------
    lines = []
    lines.append("稳态加速度（最后10个时间点均值）汇总：")
    for r in data_rows:
        lines.append(f"  {r['experiment_id']:8s}  F_ext={r['F_ext']:6.2f}  v0={r['v0']:6.2f}  steady_a={r['steady_a']:.8f}")
    lines.append("")
    lines.append("多元线性回归：steady_a = β0 + β1*F_ext + β2*v0")
    lines.append(f"  截距 β0 = {intercept:.6f}")
    lines.append(f"  β1 (F_ext) = {coef_fext:.6f}")
    lines.append(f"  β2 (v0)    = {coef_v0:.6f}")
    lines.append(f"  R² = {r2:.6f}    n = {n}")
    if p_value is not None:
        lines.append(f"  p-value (F检验) ≈ {p_value:.6e}")
    else:
        lines.append("  无法计算 p-value（自由度不足或 R²=1）")
    lines.append("")
    lines.append("每个实验的回归残差：")
    for r in data_rows:
        lines.append(f"  {r['experiment_id']:8s}  residual = {r['residual']:.6e}")

    lines.append("")
    if corr_matrix is not None:
        lines.append("τ 与 F_ext、v0 的相关系数矩阵（共 {} 个有效 τ）：".format(len(tau_arr)))
        lines.append("              F_ext       v0          τ")
        lines.append(f"  F_ext      {corr_matrix[0,0]:.6f}   {corr_matrix[0,1]:.6f}   {corr_matrix[0,2]:.6f}")
        lines.append(f"  v0         {corr_matrix[1,0]:.6f}   {corr_matrix[1,1]:.6f}   {corr_matrix[1,2]:.6f}")
        lines.append(f"  τ          {corr_matrix[2,0]:.6f}   {corr_matrix[2,1]:.6f}   {corr_matrix[2,2]:.6f}")
        lines.append(f"τ 均值 = {np.mean(tau_arr):.4f}   标准差 = {np.std(tau_arr):.4f}")
    else:
        lines.append("τ 有效实验数不足（<2），无法计算相关系数。")

    observation = "\n".join(lines)

    # ------------------------------------------------------------
    # 5. 可视化：steady_a vs F_ext (颜色=v0) + 预测vs实际
    # ------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    v0_vals = np.array([r["v0"] for r in data_rows])
    sc = ax1.scatter(
        [r["F_ext"] for r in data_rows],
        [r["steady_a"] for r in data_rows],
        c=v0_vals, cmap='viridis', s=80, edgecolors='k', alpha=0.8
    )
    ax1.set_xlabel("F_ext")
    ax1.set_ylabel("steady_a")
    ax1.set_title("Steady acceleration vs External force (color = initial velocity)")
    cbar = plt.colorbar(sc, ax=ax1, label='v0')
    fig1_path = output_dir / "steady_a_vs_Fext.png"
    fig1.savefig(str(fig1_path))
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(y_pred, y, alpha=0.8, c='b', edgecolors='k')
    min_val = min(y.min(), y_pred.min())
    max_val = max(y.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1.5)
    ax2.set_xlabel("Predicted steady_a")
    ax2.set_ylabel("Actual steady_a")
    ax2.set_title("Predicted vs Actual (multivariate regression)")
    fig2_path = output_dir / "steady_a_pred_vs_actual.png"
    fig2.savefig(str(fig2_path))
    plt.close(fig2)

    figures = [str(fig1_path), str(fig2_path)]

    # ------------------------------------------------------------
    # 6. 度量提取
    # ------------------------------------------------------------
    metrics = {
        "regression_coefficients": {
            "intercept": intercept,
            "coef_F_ext": coef_fext,
            "coef_v0": coef_v0
        },
        "R2": r2,
        "p_value_approx": p_value,
        "n_experiments": n,
        "tau_correlation_matrix": corr_matrix.tolist() if corr_matrix is not None else None,
        "tau_mean": float(np.mean(tau_arr)) if len(tau_arr) > 0 else None,
        "tau_std": float(np.std(tau_arr)) if len(tau_arr) > 0 else None,
        "num_tau_valid": len(tau_arr),
        "steady_a_per_experiment": {r["experiment_id"]: r["steady_a"] for r in data_rows},
        "residuals": {r["experiment_id"]: r["residual"] for r in data_rows},
        "tau_per_experiment": {r["experiment_id"]: r["tau"] for r in data_rows}
    }

    # 不返回派生序列（无复用必要）
    derived_series = []

    return {
        "observation": observation,
        "metrics": metrics,
        "figures": figures,
        "derived_series": derived_series
    }
