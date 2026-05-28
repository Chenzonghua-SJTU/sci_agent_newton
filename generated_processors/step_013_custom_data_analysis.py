import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn import linear_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _estimate_kinematics(q: List[float], t: List[float], window_length: int = 11,
                         polyorder: int = 3) -> tuple:
    """
    Estimate smoothed velocity and acceleration from position using Savitzky-Golay filter.
    Returns (v_sg, a_sg) as lists.
    """
    n = len(q)
    if n < window_length:
        # fallback to smaller window
        window_length = n if n % 2 == 1 else n - 1
        if window_length < 3:
            raise ValueError(f"Not enough points ({n}) for Savitzky-Golay filter")
    q_arr = np.array(q)
    dt = (t[-1] - t[0]) / (len(t) - 1) if len(t) > 1 else 1.0
    v = signal.savgol_filter(q_arr, window_length, polyorder, deriv=1) / dt
    a = signal.savgol_filter(q_arr, window_length, polyorder, deriv=2) / (dt ** 2)
    return v.tolist(), a.tolist()


def _ols_fit(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Ordinary least squares fit. X: (n_samples, n_features), include bias column if needed.
    Returns dict with 'coef', 'stderr', 'ci_low', 'ci_high', 'r2', 'n', 'p' (rank).
    """
    n, p = X.shape
    coeff, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    # residuals is empty if rank == n? compute RSS manually
    y_pred = X @ coeff
    RSS = np.sum((y - y_pred) ** 2)
    TSS = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - RSS / TSS if TSS > 0 else 0.0
    # degrees of freedom: n - p
    if n > p:
        MSE = RSS / (n - p)
        # covariance matrix
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
            coef_var = MSE * XtX_inv
            stderr = np.sqrt(np.diag(coef_var))
        except np.linalg.LinAlgError:
            stderr = np.full(p, np.nan)
    else:
        stderr = np.full(p, np.nan)
    ci_low = coeff - 1.96 * stderr
    ci_high = coeff + 1.96 * stderr
    return {
        'coef': coeff.tolist(),
        'stderr': stderr.tolist() if not np.any(np.isnan(stderr)) else None,
        'ci_low': ci_low.tolist() if not np.any(np.isnan(ci_low)) else None,
        'ci_high': ci_high.tolist() if not np.any(np.isnan(ci_high)) else None,
        'r2': R2,
        'n': n,
        'p': p
    }


def process(payload: dict) -> dict:
    action = payload.get("action", "custom_data_analysis")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = Path(payload.get("output_dir", "."))

    # Parse experiment IDs
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        # fallback: all available
        exp_ids = list(experiments.keys())
    # Filter to only existing
    exp_ids = [eid for eid in exp_ids if eid in experiments]

    # Parameters for SG filter
    window_length = 11
    polyorder = 3

    derived_series_list = []
    per_exp_results = {}
    all_a = []
    all_v = []
    all_F = []
    all_exp_labels = []

    # First pass: compute v_sg, a_sg for each experiment
    for eid in exp_ids:
        exp_data = experiments[eid]
        config = exp_data.get("config", {})
        series = exp_data.get("series", {})
        available = exp_data.get("available_series", [])

        # Get q and t
        q = series.get("q")
        t = series.get("t")
        if q is None or t is None:
            raise ValueError(f"Experiment {eid}: missing 'q' or 't' series")

        # Ensure lists
        q = list(q)
        t = list(t)
        if len(q) != len(t):
            raise ValueError(f"Experiment {eid}: length mismatch between q ({len(q)}) and t ({len(t)})")

        # Estimate kinematics
        v_sg, a_sg = _estimate_kinematics(q, t, window_length, polyorder)

        # Store derived series
        derived_series_list.append({
            "experiment_id": eid,
            "name": "v_sg",
            "values": v_sg,
            "source_name": f"Savitzky-Golay filter (window={window_length}, polyorder={polyorder}) from q",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "smoothed velocity"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": "a_sg",
            "values": a_sg,
            "source_name": f"Savitzky-Golay filter (window={window_length}, polyorder={polyorder}) from q",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "smoothed acceleration"
        })

        # Get F_ext
        force_field_type = config.get("force_field_type", "free")
        if force_field_type == "free":
            F_ext = 0.0
        else:
            F_ext = float(config.get("F_ext", 0.0))

        # Store for global fit & fitting
        all_a.extend(a_sg)
        all_v.extend(v_sg)
        all_F.extend([F_ext] * len(v_sg))
        all_exp_labels.extend([eid] * len(v_sg))

        per_exp_results[eid] = {
            "F_ext": F_ext,
            "v_sg": v_sg,
            "a_sg": a_sg,
            "q": q,
            "t": t
        }

    # Now perform per-experiment fits
    per_exp_fits = {}
    for eid, data in per_exp_results.items():
        v_arr = np.array(data["v_sg"])
        a_arr = np.array(data["a_sg"])
        F = data["F_ext"]

        # Linear fit a = alpha + beta * v
        lin_result = stats.linregress(v_arr, a_arr)
        alpha_lin = lin_result.intercept
        beta = lin_result.slope
        r2_lin = lin_result.rvalue ** 2
        # Standard errors
        alpha_stderr = lin_result.intercept_stderr
        beta_stderr = lin_result.stderr
        alpha_ci_low = alpha_lin - 1.96 * alpha_stderr
        alpha_ci_high = alpha_lin + 1.96 * alpha_stderr
        beta_ci_low = beta - 1.96 * beta_stderr
        beta_ci_high = beta + 1.96 * beta_stderr

        # Quadratic fit a = alpha + gamma * v^2
        v2_arr = v_arr ** 2
        quad_result = stats.linregress(v2_arr, a_arr)
        alpha_quad = quad_result.intercept
        gamma = quad_result.slope
        r2_quad = quad_result.rvalue ** 2
        alpha_q_stderr = quad_result.intercept_stderr
        gamma_stderr = quad_result.stderr
        alpha_q_ci_low = alpha_quad - 1.96 * alpha_q_stderr
        alpha_q_ci_high = alpha_quad + 1.96 * alpha_q_stderr
        gamma_ci_low = gamma - 1.96 * gamma_stderr
        gamma_ci_high = gamma + 1.96 * gamma_stderr

        per_exp_fits[eid] = {
            "linear": {
                "alpha": alpha_lin,
                "alpha_ci_low": alpha_ci_low,
                "alpha_ci_high": alpha_ci_high,
                "beta": beta,
                "beta_ci_low": beta_ci_low,
                "beta_ci_high": beta_ci_high,
                "R2": r2_lin
            },
            "quadratic": {
                "alpha": alpha_quad,
                "alpha_ci_low": alpha_q_ci_low,
                "alpha_ci_high": alpha_q_ci_high,
                "gamma": gamma,
                "gamma_ci_low": gamma_ci_low,
                "gamma_ci_high": gamma_ci_high,
                "R2": r2_quad
            },
            "alpha_F_diff": {
                "linear": alpha_lin - F,
                "quadratic": alpha_quad - F
            }
        }

    # Global linear model: a = p1 * F_ext + p2 * v_sg + intercept
    # Build design matrix
    X_global = np.column_stack([
        np.array(all_F),
        np.array(all_v),
        np.ones(len(all_F))
    ])
    y_global = np.array(all_a)

    global_fit = _ols_fit(X_global, y_global)
    # Coefficients: p1, p2, intercept
    coef = global_fit['coef']
    p1, p2, intercept = coef[0], coef[1], coef[2] if len(coef) > 2 else 0.0
    ci_low = global_fit['ci_low']
    ci_high = global_fit['ci_high']
    if ci_low:
        p1_ci_low, p2_ci_low, intercept_ci_low = ci_low[0], ci_low[1], ci_low[2]
        p1_ci_high, p2_ci_high, intercept_ci_high = ci_high[0], ci_high[1], ci_high[2]
    else:
        p1_ci_low = p2_ci_low = intercept_ci_low = None
        p1_ci_high = p2_ci_high = intercept_ci_high = None
    global_r2 = global_fit['r2']

    # Prepare observation & metrics
    obs_lines = []
    metrics = {}

    obs_lines.append(f"对所有 {len(exp_ids)} 个实验重新使用 Savitzky-Golay 滤波 (window={window_length}, polyorder={polyorder}, dt=0.1) 从 q(t) 估计平滑速度 v_sg 和加速度 a_sg。")
    for eid, fit in per_exp_fits.items():
        F = per_exp_results[eid]["F_ext"]
        lin = fit["linear"]
        quad = fit["quadratic"]
        obs_lines.append(f"  {eid} (F_ext={F}):")
        obs_lines.append(f"    线性: α={lin['alpha']:.6f} (95%CI [{lin['alpha_ci_low']:.6f},{lin['alpha_ci_high']:.6f}]), β={lin['beta']:.6f} (95%CI [{lin['beta_ci_low']:.6f},{lin['beta_ci_high']:.6f}]), R²={lin['R2']:.6f}")
        obs_lines.append(f"    二次: α={quad['alpha']:.6f} (95%CI [{quad['alpha_ci_low']:.6f},{quad['alpha_ci_high']:.6f}]), γ={quad['gamma']:.6f} (95%CI [{quad['gamma_ci_low']:.6f},{quad['gamma_ci_high']:.6f}]), R²={quad['R2']:.6f}")
        obs_lines.append(f"    α 与 F_ext 的差值: 线性={fit['alpha_F_diff']['linear']:.6f}, 二次={fit['alpha_F_diff']['quadratic']:.6f}")
        metrics[f"{eid}_linear_alpha"] = lin['alpha']
        metrics[f"{eid}_linear_alpha_ci_low"] = lin['alpha_ci_low']
        metrics[f"{eid}_linear_alpha_ci_high"] = lin['alpha_ci_high']
        metrics[f"{eid}_linear_beta"] = lin['beta']
        metrics[f"{eid}_linear_beta_ci_low"] = lin['beta_ci_low']
        metrics[f"{eid}_linear_beta_ci_high"] = lin['beta_ci_high']
        metrics[f"{eid}_linear_R2"] = lin['R2']
        metrics[f"{eid}_quadratic_alpha"] = quad['alpha']
        metrics[f"{eid}_quadratic_alpha_ci_low"] = quad['alpha_ci_low']
        metrics[f"{eid}_quadratic_alpha_ci_high"] = quad['alpha_ci_high']
        metrics[f"{eid}_quadratic_gamma"] = quad['gamma']
        metrics[f"{eid}_quadratic_gamma_ci_low"] = quad['gamma_ci_low']
        metrics[f"{eid}_quadratic_gamma_ci_high"] = quad['gamma_ci_high']
        metrics[f"{eid}_quadratic_R2"] = quad['R2']
        metrics[f"{eid}_alpha_F_diff_linear"] = fit['alpha_F_diff']['linear']
        metrics[f"{eid}_alpha_F_diff_quadratic"] = fit['alpha_F_diff']['quadratic']

    obs_lines.append(f"全局线性模型 a = p1*F_ext + p2*v_sg + intercept:")
    obs_lines.append(f"  p1={p1:.6f} (95%CI [{p1_ci_low:.6f},{p1_ci_high:.6f}])" if p1_ci_low else f"  p1={p1:.6f}")
    obs_lines.append(f"  p2={p2:.6f} (95%CI [{p2_ci_low:.6f},{p2_ci_high:.6f}])" if p2_ci_low else f"  p2={p2:.6f}")
    obs_lines.append(f"  intercept={intercept:.6f} (95%CI [{intercept_ci_low:.6f},{intercept_ci_high:.6f}])" if intercept_ci_low else f"  intercept={intercept:.6f}")
    obs_lines.append(f"  全局 R² = {global_r2:.6f}")

    metrics['global_p1'] = p1
    metrics['global_p1_ci_low'] = p1_ci_low
    metrics['global_p1_ci_high'] = p1_ci_high
    metrics['global_p2'] = p2
    metrics['global_p2_ci_low'] = p2_ci_low
    metrics['global_p2_ci_high'] = p2_ci_high
    metrics['global_intercept'] = intercept
    metrics['global_intercept_ci_low'] = intercept_ci_low
    metrics['global_intercept_ci_high'] = intercept_ci_high
    metrics['global_R2'] = global_r2

    # Generate figures
    figure_paths = []
    # 1. Per experiment scatter plot with linear and quadratic fits
    for eid in exp_ids:
        data = per_exp_results[eid]
        v_arr = np.array(data["v_sg"])
        a_arr = np.array(data["a_sg"])
        F = data["F_ext"]
        fit = per_exp_fits[eid]
        lin = fit["linear"]
        quad = fit["quadratic"]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(v_arr, a_arr, s=10, alpha=0.7, label='data')
        # Linear fit line
        v_sorted = np.sort(v_arr)
        ax.plot(v_sorted, lin['alpha'] + lin['beta'] * v_sorted, 'r-', label=f"linear: α={lin['alpha']:.4f}, β={lin['beta']:.4f}, R²={lin['R2']:.4f}")
        # Quadratic fit line (as a function of v)
        ax.plot(v_sorted, quad['alpha'] + quad['gamma'] * v_sorted**2, 'g--', label=f"quadratic: α={quad['alpha']:.4f}, γ={quad['gamma']:.4f}, R²={quad['R2']:.4f}")
        ax.set_xlabel('v_sg')
        ax.set_ylabel('a_sg')
        ax.set_title(f'{eid} (F_ext={F})')
        ax.legend()
        fig.tight_layout()
        fname = f"{eid}_a_vs_v_fit.png"
        figpath = str(output_dir / fname)
        fig.savefig(figpath, dpi=150)
        plt.close(fig)
        figure_paths.append(figpath)

    # 2. Global model residuals plot
    X_global = np.column_stack([np.array(all_F), np.array(all_v), np.ones(len(all_F))])
    y_pred = X_global @ np.array(global_fit['coef'])
    residuals = np.array(all_a) - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, s=10, alpha=0.5)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel('predicted a')
    ax.set_ylabel('residuals')
    ax.set_title(f'Global model residuals (R²={global_r2:.4f})')
    fig.tight_layout()
    fname_res = "global_fit_residuals.png"
    figpath_res = str(output_dir / fname_res)
    fig.savefig(figpath_res, dpi=150)
    plt.close(fig)
    figure_paths.append(figpath_res)

    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figure_paths,
        "metrics": metrics
    }
