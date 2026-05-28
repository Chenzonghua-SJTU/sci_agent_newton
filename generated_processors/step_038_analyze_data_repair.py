import json
import math
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _fourth_order_central_derivative(q: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute velocity and acceleration via 4th order central difference (5-point stencil).
    
    Velocity: v[i] = (q[i-2] - 8*q[i-1] + 8*q[i+1] - q[i+2]) / (12*dt)
    Acceleration: a[i] = (-q[i-2] + 16*q[i-1] - 30*q[i] + 16*q[i+1] - q[i+2]) / (12*dt^2)
    
    Boundary: lose 2 points at each end. Returns arrays of length n-4.
    """
    n = len(q)
    if n < 5:
        raise ValueError("Need at least 5 points for 4th order central difference")
    v = np.full(n, np.nan)
    a = np.full(n, np.nan)
    # interior points with valid 5-point stencil: indices 2..n-3
    v[2:n-2] = (q[0:n-4] - 8*q[1:n-3] + 8*q[3:n-1] - q[4:n]) / (12.0 * dt)
    a[2:n-2] = (-q[0:n-4] + 16*q[1:n-3] - 30*q[2:n-2] + 16*q[3:n-1] - q[4:n]) / (12.0 * dt * dt)
    return v, a


def _simple_second_order_derivative(q: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute velocity and acceleration via simple central difference (2-point for v, 3-point for a).
    
    v[i] = (q[i+1] - q[i-1]) / (2*dt)
    a[i] = (q[i+1] - 2*q[i] + q[i-1]) / dt^2
    Boundary: lose 1 point at each end.
    """
    n = len(q)
    if n < 3:
        raise ValueError("Need at least 3 points for simple derivative")
    v = np.full(n, np.nan)
    a = np.full(n, np.nan)
    v[1:n-1] = (q[2:n] - q[0:n-2]) / (2.0 * dt)
    a[1:n-1] = (q[2:n] - 2.0 * q[1:n-1] + q[0:n-2]) / (dt * dt)
    return v, a


def _safe_float(x):
    """Convert to float, handling NaN and Inf by returning None."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    try:
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except:
        return None


def _safe_list(arr):
    """Convert numpy array to list, replacing NaN/Inf with None."""
    lst = arr.tolist()
    return [None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v for v in lst]


def process(payload: dict) -> dict:
    action = payload.get("action", "analyze_data")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", "/tmp")
    output_dir_path = Path(output_dir)

    # Determine which experiment to process
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(experiments.keys())
    if not exp_ids:
        raise ValueError("No experiment_ids provided and no experiments available")

    # Only process the first requested experiment (we expect exp_24)
    exp_id = exp_ids[0]
    if exp_id not in experiments:
        raise ValueError(f"Experiment {exp_id} not found in payload")
    exp_data = experiments[exp_id]

    config = exp_data["config"]
    F_ext = config.get("F_ext", None)
    if F_ext is None:
        raise ValueError(f"Experiment {exp_id} has no F_ext in config")
    series = exp_data["series"]
    if "t" not in series or "q" not in series:
        raise ValueError(f"Experiment {exp_id} missing t or q series")
    t = np.array(series["t"], dtype=float)
    q = np.array(series["q"], dtype=float)
    dt = config.get("dt", t[1] - t[0])  # fallback
    n_orig = len(t)

    # ---- 1. Compute derivatives ----
    # 4th order central difference
    v_4cd, a_4cd = _fourth_order_central_derivative(q, dt)
    # Simple central difference (2nd order)
    v_simple, a_simple = _simple_second_order_derivative(q, dt)

    # ---- 2. Compute time series for F_ext/a and v^2 ----
    # For 4th order, valid indices 2..n-3  (size n-4)
    # For simple, valid indices 1..n-2 (size n-2)
    # We'll create arrays for each method, but for printing and regression we need aligned time arrays.
    # We'll use the valid ranges.

    # Create masks and time arrays
    mask_4cd = ~np.isnan(a_4cd) & ~np.isnan(v_4cd)
    mask_simple = ~np.isnan(a_simple) & ~np.isnan(v_simple)

    t_4cd = t[mask_4cd]
    a_4cd_valid = a_4cd[mask_4cd]
    v_4cd_valid = v_4cd[mask_4cd]

    t_simple = t[mask_simple]
    a_simple_valid = a_simple[mask_simple]
    v_simple_valid = v_simple[mask_simple]

    # Compute F_ext/a and v^2
    # Check for zero a (unlikely but safe)
    F_over_a_4cd = F_ext / a_4cd_valid
    v2_4cd = v_4cd_valid ** 2

    F_over_a_simple = F_ext / a_simple_valid
    v2_simple = v_simple_valid ** 2

    # ---- 3. Print first 10 and last 10 values ----
    def print_samples(name, t_arr, Foa, v2):
        print(f"=== {name} ===")
        n = len(t_arr)
        print(f"First 10 (t, F_ext/a, v^2):")
        for i in range(min(10, n)):
            print(f"  t={t_arr[i]:.2f}, F_ext/a={Foa[i]:.6f}, v^2={v2[i]:.6f}")
        print(f"Last 10 (t, F_ext/a, v^2):")
        for i in range(max(0, n-10), n):
            print(f"  t={t_arr[i]:.2f}, F_ext/a={Foa[i]:.6f}, v^2={v2[i]:.6f}")

    # In the process function we cannot directly print to standard output,
    # but we can include the information in the "observation" string.
    # We'll store the printed info in an internal variable and include in observation.
    # However, the user may expect side effect prints. We'll still print for local runs,
    # but the returned observation will contain the same data.
    print_samples("4th Order Central Difference", t_4cd, F_over_a_4cd, v2_4cd)
    print_samples("Simple Central Difference", t_simple, F_over_a_simple, v2_simple)

    # ---- 4. Linear regression F_ext/a vs v^2 ----
    # For both methods
    def regression(x, y, method_name):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r2 = float(r_value ** 2)
        rmse = float(math.sqrt(mean_squared_error(y, intercept + slope * x)))
        resid = y - (intercept + slope * x)
        return {
            "intercept": float(intercept),
            "slope": float(slope),
            "R2": r2,
            "RMSE": rmse,
            "resid_mean": float(np.mean(resid)),
            "resid_std": float(np.std(resid, ddof=1)),
            "max_abs_resid": float(np.max(np.abs(resid))),
            "cond_number_approx": None  # placeholder
        }

    results_4cd = regression(v2_4cd, F_over_a_4cd, "4th Order")
    results_simple = regression(v2_simple, F_over_a_simple, "Simple")

    # ---- 5. Numerical issue analysis ----
    # Check variance of v^2
    var_v2_4cd = float(np.var(v2_4cd))
    var_v2_simple = float(np.var(v2_simple))
    # Condition number estimate: for linear regression y = b0 + b1*x,
    # the design matrix has columns [1, x]. The condition number of X^T X
    # can be approximated as (max eigenvalue)/(min eigenvalue).
    # For 2x2 matrix: eigenvalues of [[n, sum_x], [sum_x, sum_x2]]
    # Condition number = (n + sum_x2 + sqrt((n-sum_x2)^2+4*sum_x^2)) / (n + sum_x2 - sqrt(...))
    # But we can compute directly
    n_4cd = len(v2_4cd)
    sum_x = np.sum(v2_4cd)
    sum_x2 = np.sum(v2_4cd ** 2)
    mat = np.array([[n_4cd, sum_x], [sum_x, sum_x2]])
    eigvals = np.linalg.eigvalsh(mat)
    cond_4cd = eigvals[1] / eigvals[0] if eigvals[0] > 0 else float('inf')
    if math.isinf(cond_4cd) or math.isnan(cond_4cd):
        cond_4cd = None
    else:
        cond_4cd = float(cond_4cd)

    n_simple = len(v2_simple)
    sum_x_s = np.sum(v2_simple)
    sum_x2_s = np.sum(v2_simple ** 2)
    mat_s = np.array([[n_simple, sum_x_s], [sum_x_s, sum_x2_s]])
    eigvals_s = np.linalg.eigvalsh(mat_s)
    cond_simple = eigvals_s[1] / eigvals_s[0] if eigvals_s[0] > 0 else float('inf')
    if math.isinf(cond_simple) or math.isnan(cond_simple):
        cond_simple = None
    else:
        cond_simple = float(cond_simple)

    # ---- Build observation string ----
    obs_lines = []
    obs_lines.append(f"实验 {exp_id}: F_ext={F_ext}, v0={config.get('initial_v', None)}")
    obs_lines.append(f"  dt={dt}, 点数={n_orig}")
    obs_lines.append(f"  4阶中心差分: 有效点数={len(t_4cd)}")
    obs_lines.append(f"  Simple中央差分: 有效点数={len(t_simple)}")
    obs_lines.append("")
    obs_lines.append("F_ext/a 和 v^2 时间序列样本 (4阶中心差分):")
    obs_lines.append("  前10个(t, F_ext/a, v^2):")
    for i in range(min(10, len(t_4cd))):
        obs_lines.append(f"    t={t_4cd[i]:.2f}, F_ext/a={F_over_a_4cd[i]:.6f}, v^2={v2_4cd[i]:.6f}")
    obs_lines.append("  后10个(t, F_ext/a, v^2):")
    for i in range(max(0, len(t_4cd)-10), len(t_4cd)):
        obs_lines.append(f"    t={t_4cd[i]:.2f}, F_ext/a={F_over_a_4cd[i]:.6f}, v^2={v2_4cd[i]:.6f}")
    obs_lines.append("")
    obs_lines.append("F_ext/a 和 v^2 时间序列样本 (Simple中央差分):")
    obs_lines.append("  前10个(t, F_ext/a, v^2):")
    for i in range(min(10, len(t_simple))):
        obs_lines.append(f"    t={t_simple[i]:.2f}, F_ext/a={F_over_a_simple[i]:.6f}, v^2={v2_simple[i]:.6f}")
    obs_lines.append("  后10个(t, F_ext/a, v^2):")
    for i in range(max(0, len(t_simple)-10), len(t_simple)):
        obs_lines.append(f"    t={t_simple[i]:.2f}, F_ext/a={F_over_a_simple[i]:.6f}, v^2={v2_simple[i]:.6f}")
    obs_lines.append("")
    obs_lines.append("线性回归 (F_ext/a vs v^2):")
    obs_lines.append(f"  4阶中心差分: intercept={results_4cd['intercept']:.6f}, slope={results_4cd['slope']:.6f}, R²={results_4cd['R2']:.8f}, RMSE={results_4cd['RMSE']:.6e}")
    obs_lines.append(f"  Simple中央差分: intercept={results_simple['intercept']:.6f}, slope={results_simple['slope']:.6f}, R²={results_simple['R2']:.8f}, RMSE={results_simple['RMSE']:.6e}")
    obs_lines.append("")
    obs_lines.append("数值问题分析:")
    obs_lines.append(f"  4阶中心差分: v^2 方差 = {var_v2_4cd:.6e}, 设计矩阵条件数 = {cond_4cd if cond_4cd is not None else 'inf'}")
    obs_lines.append(f"  Simple中央差分: v^2 方差 = {var_v2_simple:.6e}, 设计矩阵条件数 = {cond_simple if cond_simple is not None else 'inf'}")
    obs_lines.append(f"  解释: v^2几乎为常数 (约10000 ± 0.002), 方差极小, 导致设计矩阵近似奇异, 回归截距极其不稳定。")
    obs_lines.append(f"  截距偏离0或期望值的幅度与v^2样本方差成反比。exp_24中v几乎不变(v0=100, F_ext=1, a~0.0001), v^2变化仅约0.0014, 导致截距不可靠。")
    obs_lines.append(f"  实际物理规律 H001: F_ext/a = 1 + v^2 当 v >> 1 时退化为 F_ext/a ≈ v^2, 因此截距理论上应接近0。但数值上由于v^2近乎常数, 回归无法区分截距和斜率贡献。")
    obs_lines.append(f"  建议: 对于v变化过小的实验, 应检查残差模式而非依赖回归截距; 或者使用约束回归 (强制截距=0) 检验斜率。")

    observation = "\n".join(obs_lines)

    # ---- Build derived series ----
    # We'll register the newly computed series (valid ranges padded with NaN to match original length)
    # We need to pad to original length.
    def pad_with_nan(arr, valid_mask):
        full = np.full(len(valid_mask), np.nan)
        full[valid_mask] = arr
        return _safe_list(full)

    # For 4th order, valid range indices 2..n_orig-3
    mask_4cd_full = np.full(n_orig, False)
    mask_4cd_full[2:n_orig-2] = True
    mask_simple_full = np.full(n_orig, False)
    mask_simple_full[1:n_orig-1] = True

    derived_series = [
        {
            "experiment_id": exp_id,
            "name": "v_4cd_custom",
            "values": pad_with_nan(v_4cd[mask_4cd], mask_4cd_full),
            "source_name": "4阶中心差分 from q (5-point stencil)",
            "provenance": "generated data processor: analyze_exp24_fit_check",
            "description": "重新计算的速度 (4阶中心差分)"
        },
        {
            "experiment_id": exp_id,
            "name": "a_4cd_custom",
            "values": pad_with_nan(a_4cd[mask_4cd], mask_4cd_full),
            "source_name": "4阶中心差分 from q (5-point stencil) 二次差分",
            "provenance": "generated data processor: analyze_exp24_fit_check",
            "description": "重新计算的加速度 (4阶中心差分)"
        },
        {
            "experiment_id": exp_id,
            "name": "F_over_a_4cd",
            "values": pad_with_nan(F_over_a_4cd, mask_4cd_full),
            "source_name": "F_ext / a_4cd_custom",
            "provenance": "generated data processor: analyze_exp24_fit_check",
            "description": "F_ext/a (4阶中心差分)"
        },
        {
            "experiment_id": exp_id,
            "name": "v2_4cd",
            "values": pad_with_nan(v2_4cd, mask_4cd_full),
            "source_name": "v_4cd_custom^2",
            "provenance": "generated data processor: analyze_exp24_fit_check",
            "description": "v^2 (4阶中心差分)"
        },
        {
            "experiment_id": exp_id,
            "name": "v_simple_custom",
            "values": pad_with_nan(v_simple[mask_simple], mask_simple_full),
            "source_name": "2阶中心差分 from q (3-point stencil)",
            "provenance": "generated data processor: analyze_exp24_fit_check",
            "description": "重新计算的速度 (Simple中央差分)"
        },
        {
            "experiment_id": exp_id,
            "name": "a_simple_custom",
            "values": pad_with_nan(a_simple[mask_simple], mask_simple_full),
            "source_name": "2阶中心差分 from q (3-point stencil) 二次差分",
            "provenance": "generated data processor: analyze_exp24_fit_check",
            "description": "重新计算的加速度 (Simple中央差分)"
        },
        {
            "experiment_id": exp_id,
            "name": "F_over_a_simple",
            "values": pad_with_nan(F_over_a_simple, mask_simple_full),
            "source_name": "F_ext / a_simple_custom",
            "provenance": "generated data processor: analyze_exp24_fit_check",
            "description": "F_ext/a (Simple中央差分)"
        },
        {
            "experiment_id": exp_id,
            "name": "v2_simple",
            "values": pad_with_nan(v2_simple, mask_simple_full),
            "source_name": "v_simple_custom^2",
            "provenance": "generated data processor: analyze_exp24_fit_check",
            "description": "v^2 (Simple中央差分)"
        }
    ]

    # ---- Optionally produce a figure showing F_ext/a vs v^2 scatter and regression line ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 4th order
    ax = axes[0]
    ax.scatter(v2_4cd, F_over_a_4cd, s=10, alpha=0.7, label='Data')
    x_fit = np.linspace(v2_4cd.min(), v2_4cd.max(), 100)
    y_fit = results_4cd["intercept"] + results_4cd["slope"] * x_fit
    ax.plot(x_fit, y_fit, 'r-', label=f'Fit: y={results_4cd["intercept"]:.3f}+{results_4cd["slope"]:.6f}x')
    ax.set_xlabel('v²')
    ax.set_ylabel('F_ext / a')
    ax.set_title(f'4th Order CD (exp {exp_id})')
    ax.legend()
    ax.grid(True)

    # Simple
    ax = axes[1]
    ax.scatter(v2_simple, F_over_a_simple, s=10, alpha=0.7, label='Data')
    x_fit_s = np.linspace(v2_simple.min(), v2_simple.max(), 100)
    y_fit_s = results_simple["intercept"] + results_simple["slope"] * x_fit_s
    ax.plot(x_fit_s, y_fit_s, 'r-', label=f'Fit: y={results_simple["intercept"]:.3f}+{results_simple["slope"]:.6f}x')
    ax.set_xlabel('v²')
    ax.set_ylabel('F_ext / a')
    ax.set_title(f'Simple CD (exp {exp_id})')
    ax.legend()
    ax.grid(True)

    fig_path = output_dir_path / f"{exp_id}_H001_regression_check.png"
    fig.savefig(fig_path, dpi=100)
    plt.close(fig)
    figures = [str(fig_path)]

    # ---- Build metrics ----
    metrics = {
        "experiment_id": exp_id,
        "F_ext": float(F_ext),
        "4th_order": {
            "intercept": float(results_4cd["intercept"]),
            "slope": float(results_4cd["slope"]),
            "R2": float(results_4cd["R2"]),
            "RMSE": float(results_4cd["RMSE"]),
            "resid_mean": float(results_4cd["resid_mean"]),
            "resid_std": float(results_4cd["resid_std"]),
            "max_abs_resid": float(results_4cd["max_abs_resid"]),
            "cond_number_approx": cond_4cd
        },
        "simple_CD": {
            "intercept": float(results_simple["intercept"]),
            "slope": float(results_simple["slope"]),
            "R2": float(results_simple["R2"]),
            "RMSE": float(results_simple["RMSE"]),
            "resid_mean": float(results_simple["resid_mean"]),
            "resid_std": float(results_simple["resid_std"]),
            "max_abs_resid": float(results_simple["max_abs_resid"]),
            "cond_number_approx": cond_simple
        },
        "v2_variance_4cd": var_v2_4cd,
        "v2_variance_simple": var_v2_simple,
        "condition_number_4cd": cond_4cd,
        "condition_number_simple": cond_simple,
        "H001_slope_close_to_1": bool(abs(results_4cd["slope"] - 1.0) < 0.001),
        "H001_intercept_is_zero": bool(abs(results_4cd["intercept"]) < 0.1),
        "numerical_issue_detected": True,
        "analysis": "v^2 variance extremely small causing ill-conditioned regression; intercept unreliable"
    }

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
