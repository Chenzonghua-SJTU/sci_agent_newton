import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn import linear_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    # Extract parameters
    params = payload["parameters"]
    experiment_ids = params.get("experiment_ids", list(payload["experiments"].keys()))
    output_dir = Path(payload["output_dir"])

    # Validate experiment_ids exist in payload
    experiments = payload["experiments"]
    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload.")

    # Prepare storage for per‑experiment regression results
    per_exp_results = {}  # eid -> {intercept, coef_v, coef_q, r2, rmse}
    f_ext_list = []
    v0_list = []
    q0_list = []

    # Process each experiment
    for eid in experiment_ids:
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        available = exp.get("available_series", list(series.keys()))

        # Required series
        if "v" not in series or "q" not in series or "t" not in series:
            raise ValueError(f"Experiment {eid} missing basic series (t, q, v).")
        t = np.array(series["t"])
        q = np.array(series["q"])
        v = np.array(series["v"])

        # Get residue_aF; if not available, compute as a - F_ext (a must exist)
        if "residue_aF" in series:
            residue_aF = np.array(series["residue_aF"])
        else:
            if "a" not in series:
                raise ValueError(f"Experiment {eid} has no 'a' series to compute residue_aF.")
            a = np.array(series["a"])
            F_ext = config.get("F_ext", config.get("constant_force", 0))
            residue_aF = a - F_ext  # use actual F_ext from config

        # Check lengths
        n = len(t)
        if not (len(q) == n and len(v) == n and len(residue_aF) == n):
            raise ValueError(f"Series length mismatch in {eid}.")

        # Build design matrix: [1, v, q]
        X = np.column_stack([np.ones(n), v, q])
        y = residue_aF

        # Least squares solution
        coeff, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        intercept = coeff[0]
        coef_v = coeff[1]
        coef_q = coeff[2]

        # Predictions and metrics
        y_pred = X @ coeff
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        n_params = 3  # intercept + two slopes
        n_obs = n
        rmse = np.sqrt(ss_res / n_obs)

        # Store
        per_exp_results[eid] = {
            "intercept": float(intercept),
            "coef_v": float(coef_v),
            "coef_q": float(coef_q),
            "r2": float(r2),
            "rmse": float(rmse),
            "n_points": n_obs
        }

        # Experiment parameters
        F_ext = float(config.get("F_ext", 0))
        v0 = float(config.get("initial_v", 0))
        q0 = float(config.get("initial_q", 0))
        f_ext_list.append(F_ext)
        v0_list.append(v0)
        q0_list.append(q0)

    # Now collect coefficient arrays
    intercepts = np.array([per_exp_results[eid]["intercept"] for eid in experiment_ids])
    coef_v_arr = np.array([per_exp_results[eid]["coef_v"] for eid in experiment_ids])
    coef_q_arr = np.array([per_exp_results[eid]["coef_q"] for eid in experiment_ids])
    f_ext_arr = np.array(f_ext_list)
    v0_arr = np.array(v0_list)
    q0_arr = np.array(q0_list)

    # Pearson correlations for each coefficient with each parameter
    correlations = {}
    for coeff_name, coeff_arr in [("intercept", intercepts), ("coef_v", coef_v_arr), ("coef_q", coef_q_arr)]:
        for param_name, param_arr in [("F_ext", f_ext_arr), ("v0", v0_arr), ("q0", q0_arr)]:
            r, p = stats.pearsonr(coeff_arr, param_arr)
            key = f"{coeff_name}_vs_{param_name}"
            correlations[key] = {"r": float(r), "p": float(p)}

    # Generate scatter plots: 3x3 grid (coefficients vs parameters)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    coeff_labels = ["intercept", "coef_v", "coef_q"]
    param_labels = ["F_ext", "v0", "q0"]
    coeff_data = [intercepts, coef_v_arr, coef_q_arr]
    param_data = [f_ext_arr, v0_arr, q0_arr]

    for i, coeff_name in enumerate(coeff_labels):
        for j, param_name in enumerate(param_labels):
            ax = axes[i, j]
            ax.scatter(param_data[j], coeff_data[i], alpha=0.8, edgecolors='k')
            ax.set_xlabel(param_name)
            ax.set_ylabel(coeff_name)
            ax.grid(True, linestyle='--', alpha=0.6)
            # Annotate pearson r and p
            key = f"{coeff_name}_vs_{param_name}"
            r_val = correlations[key]["r"]
            p_val = correlations[key]["p"]
            ax.set_title(f"r={r_val:.3f}, p={p_val:.3g}")

    plt.tight_layout()
    fig_path = output_dir / "coefficients_vs_parameters.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)

    # Build observations
    # Observation 1: per‑experiment coefficients
    coeff_table_summary = []
    for eid in experiment_ids:
        r = per_exp_results[eid]
        coeff_table_summary.append(
            f"{eid}: intercept={r['intercept']:.4f}, coef_v={r['coef_v']:.4f}, "
            f"coef_q={r['coef_q']:.4f}, R²={r['r2']:.4f}, RMSE={r['rmse']:.4f}"
        )
    obs1_summary = "每个恒外力实验 residue_aF ~ v + q 线性回归系数和拟合质量：\n" + "\n".join(coeff_table_summary)
    obs1_metrics = {}
    for eid in experiment_ids:
        r = per_exp_results[eid]
        for k in ["intercept", "coef_v", "coef_q", "r2", "rmse"]:
            obs1_metrics[f"{eid}_{k}"] = r[k]
    obs1_metrics["experiment_count"] = len(experiment_ids)

    # Observation 2: correlations
    corr_lines = [f"{k}: r={v['r']:.4f}, p={v['p']:.4f}" for k, v in correlations.items()]
    obs2_summary = "各系数与实验参数（F_ext, v0, q0）的Pearson相关系数：\n" + "\n".join(corr_lines)
    obs2_metrics = {}
    for k, v in correlations.items():
        obs2_metrics[f"{k}_r"] = v["r"]
        obs2_metrics[f"{k}_p"] = v["p"]

    # Build final result
    result = {
        "observation": f"完成了12个恒外力实验的residue_aF~v+q线性回归。R²范围:{min([r['r2'] for r in per_exp_results.values()]):.4f}–{max([r['r2'] for r in per_exp_results.values()]):.4f}。相关系数已计算并绘图。",
        "observations": [
            {
                "summary": obs1_summary,
                "source_data_refs": [f"{eid}:residue_aF" for eid in experiment_ids] + [f"{eid}:v" for eid in experiment_ids] + [f"{eid}:q" for eid in experiment_ids],
                "metrics": obs1_metrics
            },
            {
                "summary": obs2_summary,
                "source_data_refs": [f"{eid}:config" for eid in experiment_ids],
                "metrics": obs2_metrics
            }
        ],
        "validations": [],
        "derived_series": [],
        "figures": [str(fig_path)],
        "metrics": {
            "experiment_count": len(experiment_ids),
            "intercept_range": [float(np.min(intercepts)), float(np.max(intercepts))],
            "coef_v_range": [float(np.min(coef_v_arr)), float(np.max(coef_v_arr))],
            "coef_q_range": [float(np.min(coef_q_arr)), float(np.max(coef_q_arr))],
            "r2_range": [min([r['r2'] for r in per_exp_results.values()]), max([r['r2'] for r in per_exp_results.values()])]
        }
    }

    return result
