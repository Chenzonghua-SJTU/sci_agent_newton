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
from scipy.optimize import curve_fit
from scipy import signal
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload.get("action")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = Path(payload.get("output_dir", "."))
    experiment_ids = params.get("experiment_ids", list(experiments.keys()))
    analysis_goal = params.get("analysis_goal", "")
    expected_outputs = params.get("expected_outputs", [])

    if action != "analyze_data":
        raise ValueError(f"Unknown action: {action}")

    # Prepare containers
    constant_force_data = []  # (v, ratio, F_ext, exp_id)
    free_fall_data = []       # (acc, exp_id)
    initial_ratios = {}
    free_acc_stats = {}

    # Process each requested experiment
    for eid in experiment_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        config = exp.get("config", {})
        F_ext = config.get("F_ext", 0.0)
        force_type = config.get("force_field_type", "free")
        series = exp.get("series", {})
        available = exp.get("available_series", [])

        # Determine acceleration and velocity series to use
        # Prefer central difference versions, fall back to plain
        acc_key = None
        vel_key = None
        for candidate in [f"acceleration_central_{eid}", f"acceleration_{eid}"]:
            if candidate in series:
                acc_key = candidate
                break
        for candidate in [f"velocity_central_{eid}", f"velocity_{eid}"]:
            if candidate in series:
                vel_key = candidate
                break
        t_key = "t"
        if t_key not in series:
            continue
        t = np.array(series[t_key])
        if acc_key is None or vel_key is None:
            # If no derived acceleration/velocity, compute from q(t)
            q = np.array(series.get("q", []))
            if len(q) != len(t):
                continue
            dt = t[1] - t[0] if len(t) > 1 else 0.1
            v = np.zeros_like(t)
            a = np.zeros_like(t)
            v[1:-1] = (q[2:] - q[:-2]) / (2 * dt)
            v[0] = (q[1] - q[0]) / dt
            v[-1] = (q[-1] - q[-2]) / dt
            a[1:-1] = (v[2:] - v[:-2]) / (2 * dt)
            a[0] = (v[1] - v[0]) / dt
            a[-1] = (v[-1] - v[-2]) / dt
        else:
            v = np.array(series[vel_key])
            a = np.array(series[acc_key])
            # Ensure same length as t
            if len(v) != len(t) or len(a) != len(t):
                # Try to align by truncating t to match
                min_len = min(len(t), len(v), len(a))
                t = t[:min_len]
                v = v[:min_len]
                a = a[:min_len]

        # Exclude boundary points (first 5 and last 5) for stability
        if len(t) > 10:
            idx_start = 5
            idx_end = -5
            t_inner = t[idx_start:idx_end]
            v_inner = v[idx_start:idx_end]
            a_inner = a[idx_start:idx_end]
        else:
            t_inner = t
            v_inner = v
            a_inner = a

        # Check if F_ext is zero (free motion)
        if abs(F_ext) < 1e-12:
            free_fall_data.append((a_inner, eid))
            free_acc_stats[eid] = {
                "mean_acc": float(np.mean(a_inner)),
                "std_acc": float(np.std(a_inner)),
                "max_abs_acc": float(np.max(np.abs(a_inner)))
            }
            continue

        # Constant force experiments
        # Compute ratio a / F_ext
        ratio = a_inner / F_ext
        # Store data
        constant_force_data.append((v_inner, ratio, F_ext, eid))

        # Initial point check: take first 3 points (or if too short, first point)
        n_initial = min(3, len(t))
        # Use original (not inner) for initial check to capture t≈0
        if len(t) >= n_initial:
            t_init = t[:n_initial]
            a_init = a[:n_initial]
            v_init = v[:n_initial]
            # Avoid division by zero if F_ext is 0 (already handled)
            ratio_init = a_init / F_ext
            # Report mean and closeness to 1
            initial_ratios[eid] = {
                "mean_ratio": float(np.mean(ratio_init)),
                "std_ratio": float(np.std(ratio_init)),
                "close_to_one": bool(abs(np.mean(ratio_init) - 1.0) < 0.05)
            }

    # 1) Scatter plot of a/F_ext vs v (constant force experiments)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(constant_force_data)))
    for idx, (v_arr, ratio_arr, F, eid) in enumerate(constant_force_data):
        ax1.scatter(v_arr, ratio_arr, s=8, alpha=0.7, color=colors[idx],
                    label=f"{eid} (F_ext={F})")
    ax1.set_xlabel("Velocity v")
    ax1.set_ylabel("a / F_ext")
    ax1.set_title("a/F_ext vs v for all constant force experiments")
    ax1.legend(loc='best', fontsize=8)
    fig1.tight_layout()
    scatter_path = str(output_dir / "ratio_vs_v_scatter.png")
    fig1.savefig(scatter_path, dpi=150)
    plt.close(fig1)

    # 2) Global fitting of a/F_ext = sign(v) * f(|v|)
    # Stack all constant force data into arrays
    all_v = []
    all_ratio = []
    all_F = []
    all_eid = []
    for v_arr, ratio_arr, F, eid in constant_force_data:
        # Use absolute velocity for magnitude models
        abs_v = np.abs(v_arr)
        all_v.extend(abs_v)
        all_ratio.extend(ratio_arr * np.sign(v_arr) * (1 if F > 0 else -1))  # sign correct for direction
        all_F.extend([F] * len(v_arr))
        all_eid.extend([eid] * len(v_arr))
    all_v = np.array(all_v)
    all_ratio = np.array(all_ratio)

    # Helper: ensure positive v for models
    def ratio_linear(v_abs, alpha1, beta1):
        return alpha1 + beta1 * v_abs

    def ratio_exponential(v_abs, A, tau):
        return A * np.exp(-v_abs / tau)

    def ratio_rational(v_abs, b):
        return 1.0 / (1.0 + b * v_abs)

    # Global fit results
    models = {}
    residuals = {}

    # Linear
    try:
        popt_lin, pcov_lin = curve_fit(ratio_linear, all_v, all_ratio, p0=[1.0, -0.1])
        pred_lin = ratio_linear(all_v, *popt_lin)
        rmse_lin = math.sqrt(mean_squared_error(all_ratio, pred_lin))
        models["linear"] = {
            "alpha": float(popt_lin[0]),
            "beta": float(popt_lin[1]),
            "rmse_global": float(rmse_lin)
        }
    except Exception as e:
        models["linear"] = {"error": str(e)}

    # Exponential
    try:
        popt_exp, pcov_exp = curve_fit(ratio_exponential, all_v, all_ratio,
                                       p0=[1.0, 1.0], bounds=([0, 0], [10, 100]))
        pred_exp = ratio_exponential(all_v, *popt_exp)
        rmse_exp = math.sqrt(mean_squared_error(all_ratio, pred_exp))
        models["exponential"] = {
            "A": float(popt_exp[0]),
            "tau": float(popt_exp[1]),
            "rmse_global": float(rmse_exp)
        }
    except Exception as e:
        models["exponential"] = {"error": str(e)}

    # Rational
    try:
        popt_rat, pcov_rat = curve_fit(ratio_rational, all_v, all_ratio, p0=[0.5])
        pred_rat = ratio_rational(all_v, *popt_rat)
        rmse_rat = math.sqrt(mean_squared_error(all_ratio, pred_rat))
        models["rational"] = {
            "b": float(popt_rat[0]),
            "rmse_global": float(rmse_rat)
        }
    except Exception as e:
        models["rational"] = {"error": str(e)}

    # Per-experiment residuals and RMSE
    per_exp_residuals = {}
    all_preds = {}
    if "error" not in models.get("linear", {}):
        all_preds["linear"] = pred_lin
    if "error" not in models.get("exponential", {}):
        all_preds["exponential"] = pred_exp
    if "error" not in models.get("rational", {}):
        all_preds["rational"] = pred_rat

    for model_name, pred in all_preds.items():
        # map per experiment
        exp_map = collections.defaultdict(list)
        exp_pred_map = collections.defaultdict(list)
        for i, eid in enumerate(all_eid):
            exp_map[eid].append(all_ratio[i])
            exp_pred_map[eid].append(pred[i])
        per_exp = {}
        for eid in exp_map:
            y_true = np.array(exp_map[eid])
            y_pred = np.array(exp_pred_map[eid])
            rmse = math.sqrt(mean_squared_error(y_true, y_pred))
            residual = (y_true - y_pred).tolist()
            per_exp[eid] = {
                "rmse": float(rmse),
                "residual_mean": float(np.mean(y_true - y_pred)),
                "residual_std": float(np.std(y_true - y_pred))
            }
        per_exp_residuals[model_name] = per_exp

    # 3) Plot global fits with data points (on a single axis)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # Plot all data as scatter
    for idx, (v_arr, ratio_arr, F, eid) in enumerate(constant_force_data):
        ax2.scatter(np.abs(v_arr), ratio_arr * np.sign(v_arr) * (1 if F > 0 else -1),
                    s=4, alpha=0.4, color=colors[idx], label=f"{eid}")
    # Plot fits over sorted v range
    v_sort = np.sort(all_v)
    if "linear" in models and "error" not in models["linear"]:
        ax2.plot(v_sort, ratio_linear(v_sort, *popt_lin), 'r-', lw=2, label="Linear")
    if "exponential" in models and "error" not in models["exponential"]:
        ax2.plot(v_sort, ratio_exponential(v_sort, *popt_exp), 'g-', lw=2, label="Exponential")
    if "rational" in models and "error" not in models["rational"]:
        ax2.plot(v_sort, ratio_rational(v_sort, *popt_rat), 'b-', lw=2, label="Rational")
    ax2.set_xlabel("|v|")
    ax2.set_ylabel("a/F_ext (with sign correction)")
    ax2.set_title("Global fits of a/F_ext = f(|v|)*sign(v)")
    ax2.legend(loc='best', fontsize=8)
    fig2.tight_layout()
    fit_path = str(output_dir / "global_fits.png")
    fig2.savefig(fit_path, dpi=150)
    plt.close(fig2)

    # 4) Initial point analysis
    initial_analysis = {}
    for eid, info in initial_ratios.items():
        initial_analysis[eid] = info

    # 5) Free motion check
    free_motion_check = {}
    for eid, stats in free_acc_stats.items():
        free_motion_check[eid] = {
            "mean_acc": stats["mean_acc"],
            "is_zero": bool(abs(stats["mean_acc"]) < 1e-10),
            "max_abs_acc": stats["max_abs_acc"]
        }

    # Build observation string
    obs_parts = [
        "对所有恒定外力实验进行了 a/F_ext vs v 分析。",
        f"共处理 {len(constant_force_data)} 个恒定外力实验，{len(free_fall_data)} 个自由运动实验。",
        f"全局拟合结果："
    ]
    for mname, mres in models.items():
        if "error" in mres:
            obs_parts.append(f"  {mname}: 拟合失败 ({mres['error']})")
        else:
            obs_parts.append(f"  {mname}: RMSE={mres['rmse_global']:.5f}")

    # Per-experiment residuals summary
    if "linear" in per_exp_residuals:
        lin_rmse_list = [v["rmse"] for v in per_exp_residuals["linear"].values()]
        if lin_rmse_list:
            obs_parts.append(f"  Linear: 各实验RMSE范围 {min(lin_rmse_list):.5f} ~ {max(lin_rmse_list):.5f}")
    if "exponential" in per_exp_residuals:
        exp_rmse_list = [v["rmse"] for v in per_exp_residuals["exponential"].values()]
        if exp_rmse_list:
            obs_parts.append(f"  Exponential: 各实验RMSE范围 {min(exp_rmse_list):.5f} ~ {max(exp_rmse_list):.5f}")
    if "rational" in per_exp_residuals:
        rat_rmse_list = [v["rmse"] for v in per_exp_residuals["rational"].values()]
        if rat_rmse_list:
            obs_parts.append(f"  Rational: 各实验RMSE范围 {min(rat_rmse_list):.5f} ~ {max(rat_rmse_list):.5f}")

    # Initial point summary
    close_count = sum(1 for v in initial_ratios.values() if v["close_to_one"])
    obs_parts.append(f"初始点检查：{close_count}/{len(initial_ratios)} 个实验的初始 a/F_ext 接近1（阈值0.05）。")
    for eid, info in initial_ratios.items():
        obs_parts.append(f"  {eid}: 平均ratio={info['mean_ratio']:.4f}")

    # Free motion check
    zero_free = sum(1 for v in free_motion_check.values() if v["is_zero"])
    obs_parts.append(f"自由运动检查：{zero_free}/{len(free_motion_check)} 个实验加速度均值接近零。")
    for eid, v in free_motion_check.items():
        obs_parts.append(f"  {eid}: mean_acc={v['mean_acc']:.2e}, max_abs={v['max_abs_acc']:.2e}")

    observation = "\n".join(obs_parts)

    # Assemble metrics
    metrics = {
        "global_fits": models,
        "per_experiment_residuals": per_exp_residuals,
        "initial_ratio_check": initial_analysis,
        "free_motion_check": free_motion_check
    }

    figures = [scatter_path, fit_path]

    return {
        "observation": observation,
        "derived_series": [],
        "figures": figures,
        "metrics": metrics
    }
