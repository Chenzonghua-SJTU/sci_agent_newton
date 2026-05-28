import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy import signal, optimize, stats
import sklearn
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload.get("action", "custom_data_analysis")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = Path(payload.get("output_dir", "."))

    experiment_ids = params.get("experiment_ids", list(experiments.keys()))
    if not experiment_ids:
        experiment_ids = list(experiments.keys())

    # Collect all normalized data for global fit
    all_v = []
    all_y = []  # a / F_ext
    all_exp_ids = []
    derived_series_list = []
    fit_data = {}   # per experiment data for later residual calculation

    for eid in experiment_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        config = exp.get("config", {})
        series = exp.get("series", {})
        available = exp.get("available_series", list(series.keys()))
        t = series.get("t")
        if t is None:
            continue
        dt = config.get("dt", 0.1)
        F_ext = config.get("F_ext", 0.0)
        if F_ext == 0.0:
            continue   # skip zero force experiments

        # Ensure a_new and v_new exist, else generate from q
        if "a_new" in series and "v_new" in series:
            a_new = series["a_new"]
            v_new = series["v_new"]
            if len(a_new) != len(t) or len(v_new) != len(t):
                # might be inconsistent, regenerate
                v_new, a_new = _estimate_kinematics(series["q"], t, dt)
                derived_series_list.append({
                    "experiment_id": eid,
                    "name": "v_new",
                    "values": v_new.tolist(),
                    "source_name": "Savgol filter from q",
                    "provenance": "generated data processor: step_025_custom_data_analysis",
                    "description": "Velocity estimated from q using Savgol filter (window=11, polyorder=3, deriv=1)"
                })
                derived_series_list.append({
                    "experiment_id": eid,
                    "name": "a_new",
                    "values": a_new.tolist(),
                    "source_name": "Savgol filter from q",
                    "provenance": "generated data processor: step_025_custom_data_analysis",
                    "description": "Acceleration estimated from q using Savgol filter (window=11, polyorder=3, deriv=2)"
                })
        else:
            if "q" not in series:
                continue
            v_new, a_new = _estimate_kinematics(series["q"], t, dt)
            derived_series_list.append({
                "experiment_id": eid,
                "name": "v_new",
                "values": v_new.tolist(),
                "source_name": "Savgol filter from q",
                "provenance": "generated data processor: step_025_custom_data_analysis",
                "description": "Velocity estimated from q using Savgol filter (window=11, polyorder=3, deriv=1)"
            })
            derived_series_list.append({
                "experiment_id": eid,
                "name": "a_new",
                "values": a_new.tolist(),
                "source_name": "Savgol filter from q",
                "provenance": "generated data processor: step_025_custom_data_analysis",
                "description": "Acceleration estimated from q using Savgol filter (window=11, polyorder=3, deriv=2)"
            })

        # Now we have a_new, v_new arrays
        y = np.array(a_new, dtype=float) / F_ext
        x = np.array(v_new, dtype=float)
        # Remove any nan/inf
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            continue
        all_v.append(x)
        all_y.append(y)
        all_exp_ids.extend([eid] * len(x))
        fit_data[eid] = {"x": x.copy(), "y": y.copy()}

    if len(all_v) == 0:
        return {
            "observation": "没有有效恒外力实验数据用于全局拟合。",
            "derived_series": derived_series_list,
            "figures": [],
            "metrics": {}
        }

    X = np.concatenate(all_v)
    Y = np.concatenate(all_y)

    # Global fit: y = exp(-b * x)
    def model(x, b):
        return np.exp(-b * x)

    # Initial guess
    # Use linear regression of log(y) vs x to get initial b: log(y) = -b*x
    with np.errstate(divide='ignore', invalid='ignore'):
        logY = np.where(Y > 0, np.log(Y), -1e10)
    valid = np.isfinite(logY) & (logY > -1e9)
    if valid.sum() < 2:
        # fallback
        b0 = 0.7
    else:
        slope, intercept, r_val, p_val, std_err = stats.linregress(X[valid], logY[valid])
        b0 = -slope if slope > 0 else 0.7

    try:
        popt, pcov = optimize.curve_fit(model, X, Y, p0=[b0], bounds=(0, np.inf))
        b_opt = popt[0]
        b_stderr = np.sqrt(pcov[0,0]) if pcov[0,0] > 0 else None
    except Exception as e:
        # fallback to brute force search
        b_candidates = np.linspace(0.01, 5, 500)
        best_err = np.inf
        b_opt = 0.7
        for b_test in b_candidates:
            err = np.mean((Y - model(X, b_test))**2)
            if err < best_err:
                best_err = err
                b_opt = b_test
        b_stderr = None

    Y_pred = model(X, b_opt)
    R2 = r2_score(Y, Y_pred)

    # Per-experiment residual statistics
    residuals = Y - Y_pred
    resid_stats = {}
    exp_ids_arr = np.array(all_exp_ids)
    for eid in np.unique(exp_ids_arr):
        mask = exp_ids_arr == eid
        resid = residuals[mask]
        x_eid = X[mask]
        resid_mean = np.mean(resid)
        resid_std = np.std(resid, ddof=1)
        resid_min = np.min(resid)
        resid_max = np.max(resid)
        # correlation with v
        if len(x_eid) > 1:
            corr = np.corrcoef(x_eid, resid)[0,1]
        else:
            corr = np.nan
        resid_stats[eid] = {
            "residual_mean": float(resid_mean),
            "residual_std": float(resid_std),
            "residual_min": float(resid_min),
            "residual_max": float(resid_max),
            "corr_v_residual": float(corr)
        }

    # Build metrics dict
    metrics = {
        "b_value": float(b_opt),
        "b_stderr": float(b_stderr) if b_stderr is not None else None,
        "global_R2": float(R2),
        "global_RMSE": float(np.sqrt(np.mean(residuals**2)))
    }
    for eid, stats_e in resid_stats.items():
        for k, v in stats_e.items():
            metrics[f"{eid}_{k}"] = v

    # Plot: a/F_ext vs v with global fit curve
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0,1,10))
    color_idx = 0
    for eid in experiment_ids:
        if eid not in fit_data:
            continue
        x_eid = fit_data[eid]["x"]
        y_eid = fit_data[eid]["y"]
        if len(x_eid) == 0:
            continue
        ax.scatter(x_eid, y_eid, s=5, color=colors[color_idx % len(colors)], label=f"{eid} (F_ext={experiments[eid]['config'].get('F_ext','?')})", alpha=0.6)
        color_idx += 1
    # global fit curve
    x_plot = np.linspace(X.min(), X.max(), 300)
    y_plot = model(x_plot, b_opt)
    ax.plot(x_plot, y_plot, 'k-', linewidth=2, label=f'Global exp fit: b={b_opt:.4f}, R²={R2:.4f}')
    ax.set_xlabel("v (velocity)")
    ax.set_ylabel("a / F_ext")
    ax.set_title("Global Exponential Fit: a/F_ext = exp(-b * v)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = output_dir / "global_exp_fit_across_experiments.png"
    fig.savefig(str(fig_path), dpi=300)
    plt.close(fig)

    # Build observation string
    obs_lines = [
        f"全局指数模型拟合完成，使用实验 {experiment_ids}。",
        f"拟合参数 b = {b_opt:.4f} (标准误={b_stderr:.4f})，全局 R² = {R2:.4f}。",
        f"全局 RMSE = {np.sqrt(np.mean(residuals**2)):.4f}。",
        "各实验残差统计："
    ]
    for eid in experiment_ids:
        if eid in resid_stats:
            s = resid_stats[eid]
            obs_lines.append(f"  {eid}: 均值 {s['residual_mean']:.5f}, 标准差 {s['residual_std']:.5f}, "
                             f"范围 [{s['residual_min']:.5f},{s['residual_max']:.5f}], "
                             f"与 v 相关系数 {s['corr_v_residual']:.4f}")
    obs_lines.append(f"图像已保存: {fig_path.name}")

    return {
        "observation": "\n".join(obs_lines),
        "derived_series": derived_series_list,
        "figures": [str(fig_path)],
        "metrics": metrics
    }

def _estimate_kinematics(q_raw: list, t_raw: list, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate velocity and acceleration from position using Savitzky-Golay filter.
    Returns (v_new, a_new) as numpy arrays of same length as q.
    """
    from scipy.signal import savgol_filter
    q = np.array(q_raw, dtype=float)
    window_length = 11
    polyorder = 3
    # Ensure window length is odd and <= len(q)
    if len(q) < window_length:
        window_length = len(q) if len(q) % 2 == 1 else len(q) - 1
        if window_length < 3:
            window_length = 3
    if window_length <= polyorder:
        polyorder = window_length - 1
    v = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt)
    a = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)
    return v, a
