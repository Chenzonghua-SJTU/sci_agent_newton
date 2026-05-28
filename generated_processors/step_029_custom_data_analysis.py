import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = Path(payload.get("output_dir", "."))

    # Extract experiment ids from parameters, fallback to all available
    exp_ids = params.get("experiment_ids", params.get("experiment_id", list(experiments.keys())))
    if not exp_ids:
        exp_ids = list(experiments.keys())
    # Filter to only those that exist
    exp_ids = [eid for eid in exp_ids if eid in experiments]

    # Optionally: prefer a_sg/v_sg, fallback to a_new/v_new
    optional_series = params.get("optional_series", ["a_sg", "v_sg", "a_new", "v_new"])

    # --- Helper function to get preferred series ---
    def get_series(exp_id: str, preferred: List[str], fallback: List[str]) -> Optional[np.ndarray]:
        exp = experiments[exp_id]
        avail = exp.get("available_series", [])
        series_dict = exp.get("series", {})
        # Check preferred first
        for name in preferred:
            if name in avail and name in series_dict:
                arr = np.array(series_dict[name], dtype=float)
                if len(arr) > 0:
                    return arr
        # Then fallback
        for name in fallback:
            if name in avail and name in series_dict:
                arr = np.array(series_dict[name], dtype=float)
                if len(arr) > 0:
                    return arr
        return None

    # --- Collect data across experiments ---
    all_v = []
    all_a = []
    all_F = []
    all_exp_ids = []
    exp_data = {}  # per experiment: v, a, F_ext

    for eid in exp_ids:
        exp = experiments[eid]
        config = exp.get("config", {})
        force_field_type = config.get("force_field_type", "")
        F_ext = config.get("F_ext", 0.0)
        if force_field_type == "free":
            F_ext = 0.0

        # get velocity
        v = get_series(eid, ["v_sg"], ["v_new", "v_est_sg"])
        a = get_series(eid, ["a_sg"], ["a_new", "a_est_sg"])

        if v is None or a is None:
            raise ValueError(f"Experiment {eid}: cannot find suitable velocity and acceleration series. "
                             f"Available: {exp.get('available_series', [])}")

        if len(v) != len(a):
            raise ValueError(f"Experiment {eid}: velocity and acceleration length mismatch: {len(v)} vs {len(a)}")

        # Only include experiments with non-zero F_ext for meaningful fit (skip free)
        if F_ext == 0.0:
            continue  # skip free experiments (should not happen in this action)

        all_v.extend(v.tolist())
        all_a.extend(a.tolist())
        all_F.extend([F_ext] * len(v))
        all_exp_ids.extend([eid] * len(v))
        exp_data[eid] = {"v": v, "a": a, "F_ext": F_ext}

    if len(all_v) == 0:
        raise ValueError("No valid data points collected for fitting.")

    v_global = np.array(all_v)
    a_global = np.array(all_a)
    F_global = np.array(all_F)

    # --- Define models ---
    models = {
        "exp": {
            "func": lambda v, b: F_global * np.exp(-b * v),  # uses F_global from closure
            "params": ["b"],
            "bounds": ([0], [np.inf]),
            "p0": [0.5]
        },
        "inverse_linear": {
            "func": lambda v, c: F_global / (1 + c * v),
            "params": ["c"],
            "bounds": ([0], [np.inf]),
            "p0": [0.2]
        },
        "power_law": {
            "func": lambda v, k, p: F_global - k * v ** p,
            "params": ["k", "p"],
            "bounds": ([0, 0], [np.inf, 10]),  # p up to 10 to avoid overflow
            "p0": [0.3, 1.0]
        },
        "linear": {
            "func": lambda v, d: F_global - d * v,
            "params": ["d"],
            "bounds": ([0], [np.inf]),
            "p0": [0.2]
        },
        "linear_alt": {
            "func": lambda v, e: F_global * (1 - e * v),
            "params": ["e"],
            "bounds": ([0], [np.inf]),
            "p0": [0.1]
        }
    }

    # We need to provide a wrapper that uses the original F_global for each data point.
    # However, curve_fit expects callable f(x, *params) returning y. Here x is v, and F_global depends on the
    # index (since different experiments have different F_ext). Currently the lambdas use F_global from outer scope,
    # but that's the global array, not the per-point value. Actually the lambda captures the variable F_global
    # which is a NumPy array, so when called with v, it uses the whole array, which works because v and F_global
    # are same length. That's okay because the function is called on the whole set at once. However, curve_fit
    # expects the function to be vectorized along the first dimension. So this is fine.

    # But we need to ensure that for power_law and others, v^p might overflow for large v.
    # We'll clip v to avoid huge numbers.

    results = {}

    # For each model, perform global non-linear fit
    fit_masks = {}  # store mask for finite values
    for model_name, model_info in models.items():
        func = model_info["func"]
        p0 = model_info["p0"]
        bounds = model_info["bounds"]
        try:
            # Check if any v is extremely negative? v can be negative (exp_08).
            # We need to be careful for power law: v^p is problematic for negative v and non-integer p.
            # For model "power_law", we should only fit on non-negative v? Or allow but handle complex? 
            # Since v can be negative, we restrict to v >= 0 for models with v^p.
            if model_name == "power_law":
                mask = v_global >= 0
                if np.sum(mask) < 10:
                    raise ValueError("Not enough non-negative velocity points for power law fit.")
                v_fit = v_global[mask]
                a_fit = a_global[mask]
                F_fit = F_global[mask]
                # Need a new lambda that uses F_fit
                func_local = lambda v, k, p: F_fit - k * v ** p
                popt, pcov = curve_fit(func_local, v_fit, a_fit, p0=p0, bounds=bounds, maxfev=10000)
                pred = func_local(v_fit, *popt)
                # Map back to full length for residual calculation
                pred_full = np.full_like(a_global, np.nan)
                pred_full[mask] = pred
                residual_full = a_global - pred_full
                # Use only mask for R² calculation
                ss_res = np.sum((a_fit - pred)**2)
                ss_tot = np.sum((a_fit - np.mean(a_fit))**2)
                R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                RMSE = np.sqrt(np.mean((a_fit - pred)**2))
                # For residual-v correlation, use points in mask
                residual_local = a_fit - pred
                corr_v_res = np.corrcoef(v_fit, residual_local)[0,1] if len(residual_local) > 2 else np.nan
            else:
                # General fit for all points
                # For inverse_linear and linear_alt, need to ensure no divide by zero
                # For linear_alt, v can be large, but fine.
                # For exp, exp(-b*v) works for all real v.
                popt, pcov = curve_fit(func, v_global, a_global, p0=p0, bounds=bounds, maxfev=10000)
                pred = func(v_global, *popt)
                ss_res = np.sum((a_global - pred)**2)
                ss_tot = np.sum((a_global - np.mean(a_global))**2)
                R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                RMSE = np.sqrt(np.mean((a_global - pred)**2))
                residual = a_global - pred
                corr_v_res = np.corrcoef(v_global, residual)[0,1] if len(residual) > 2 else np.nan
                # For per-experiment residual correlation, we calculate later
                popt_list = popt.tolist()
                # Compute parameter std errors
                perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full_like(popt, np.nan)
                perr_list = perr.tolist()
                # Store results
                model_result = {
                    "params": {model_info["params"][i]: popt_list[i] for i in range(len(popt_list))},
                    "param_stderr": {model_info["params"][i]: perr_list[i] for i in range(len(perr_list))},
                    "R2": R2,
                    "RMSE": RMSE,
                    "corr_v_residual": corr_v_res
                }
                # Calculate per-experiment residual correlations and statistics
                exp_corr = {}
                exp_residual_stats = {}
                for eid in exp_data:
                    mask_exp = (np.array(all_exp_ids) == eid)
                    v_exp = v_global[mask_exp]
                    a_exp = a_global[mask_exp]
                    pred_exp = func(v_exp, *popt)
                    resid_exp = a_exp - pred_exp
                    if len(resid_exp) > 2:
                        corr = np.corrcoef(v_exp, resid_exp)[0,1]
                    else:
                        corr = np.nan
                    exp_corr[f"{eid}_corr_v_residual"] = corr
                    exp_residual_stats[f"{eid}_residual_mean"] = float(np.mean(resid_exp))
                    exp_residual_stats[f"{eid}_residual_std"] = float(np.std(resid_exp))
                    exp_residual_stats[f"{eid}_residual_min"] = float(np.min(resid_exp))
                    exp_residual_stats[f"{eid}_residual_max"] = float(np.max(resid_exp))
                model_result["exp_residual_stats"] = exp_residual_stats
                model_result["exp_corr_v_residual"] = exp_corr
                results[model_name] = model_result
                continue  # already stored

            # For power_law we need to finish storing
            popt_list = popt.tolist()
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.full_like(popt, np.nan)
            perr_list = perr.tolist()
            model_result = {
                "params": {model_info["params"][i]: popt_list[i] for i in range(len(popt_list))},
                "param_stderr": {model_info["params"][i]: perr_list[i] for i in range(len(perr_list))},
                "R2": R2,
                "RMSE": RMSE,
                "corr_v_residual": corr_v_res
            }
            # Per-experiment residual on full data (but only for non-negative v)
            exp_corr = {}
            exp_residual_stats = {}
            for eid in exp_data:
                mask_exp = (np.array(all_exp_ids) == eid)
                v_exp = v_global[mask_exp]
                a_exp = a_global[mask_exp]
                # Use the same mask condition
                mask_local = mask_exp & mask
                if np.sum(mask_local) > 2:
                    v_local = v_global[mask_local]
                    a_local = a_global[mask_local]
                    F_local = F_global[mask_local]
                    pred_local = F_local - popt[0] * v_local ** popt[1]
                    resid_local = a_local - pred_local
                    corr = np.corrcoef(v_local, resid_local)[0,1]
                else:
                    corr = np.nan
                exp_corr[f"{eid}_corr_v_residual"] = corr
                # For stats, compute across all points (with NaN for masked)
                resid_full = a_exp - (F_global[mask_exp] - popt[0] * v_exp ** popt[1])  # careful: v_exp may contain negative
                # But best to compute only where valid
                resid_valid = resid_full[mask_exp & mask]
                if len(resid_valid) > 0:
                    exp_residual_stats[f"{eid}_residual_mean"] = float(np.mean(resid_valid))
                    exp_residual_stats[f"{eid}_residual_std"] = float(np.std(resid_valid))
                    exp_residual_stats[f"{eid}_residual_min"] = float(np.min(resid_valid))
                    exp_residual_stats[f"{eid}_residual_max"] = float(np.max(resid_valid))
                else:
                    for key in ["mean","std","min","max"]:
                        exp_residual_stats[f"{eid}_residual_{key}"] = np.nan
            model_result["exp_residual_stats"] = exp_residual_stats
            model_result["exp_corr_v_residual"] = exp_corr
            results[model_name] = model_result
        except Exception as e:
            results[model_name] = {"error": str(e)}

    # --- Plotting ---
    # Scatter plot of all experiments' a vs v (different colors per experiment)
    # Then overlay each model's fitted curve per experiment.
    exp_colors = {}
    cmap = plt.cm.tab10
    for i, eid in enumerate(exp_data.keys()):
        exp_colors[eid] = cmap(i % 10)

    # Create one figure with subplots (2x3 or 3x2) for each model
    model_names = list(models.keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()[:n_models]
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        result = results.get(model_name, {})
        # Plot scatter per experiment
        for eid in exp_data:
            ed = exp_data[eid]
            vv = ed["v"]
            aa = ed["a"]
            ax.scatter(vv, aa, color=exp_colors[eid], s=3, alpha=0.6, label=eid)
        # Plot model curves
        if "params" in result and "error" not in result:
            v_plot = np.linspace(min(v_global), max(v_global), 200)
            # For each experiment, the model curve depends on F_ext
            for eid in exp_data:
                ed = exp_data[eid]
                F0 = ed["F_ext"]
                # Build func for this experiment with same parameters
                params = result["params"]
                try:
                    if model_name == "exp":
                        pred = F0 * np.exp(-params["b"] * v_plot)
                    elif model_name == "inverse_linear":
                        pred = F0 / (1 + params["c"] * v_plot)
                    elif model_name == "power_law":
                        k, p = params["k"], params["p"]
                        # Only plot for v >=0
                        mask_plot = v_plot >= 0
                        v_valid = v_plot[mask_plot]
                        if len(v_valid) > 0:
                            pred_valid = F0 - k * v_valid ** p
                            ax.plot(v_valid, pred_valid, color=exp_colors[eid], linestyle='--', linewidth=0.5)
                    elif model_name == "linear":
                        d = params["d"]
                        pred = F0 - d * v_plot
                    elif model_name == "linear_alt":
                        e_ = params["e"]
                        pred = F0 * (1 - e_ * v_plot)
                    else:
                        continue
                    if model_name != "power_law":  # already plotted for power_law
                        ax.plot(v_plot, pred, color=exp_colors[eid], linestyle='--', linewidth=0.5)
                except Exception:
                    pass  # skip if issue
        ax.set_title(f"Model: {model_name}\nR²={result.get('R2', 'N/A'):.4f}" if "R2" in result else f"Model: {model_name} (error)")
        ax.set_xlabel("v")
        ax.set_ylabel("a")
        ax.legend(fontsize=6)
    plt.tight_layout()
    fig_path = output_dir / "model_fits_comparison.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)

    # Also create a separate plot: all scatter with best model curve (if identifiable)
    # We'll pick the model with highest R² among successful fits
    best_model = None
    best_R2 = -np.inf
    for mn, res in results.items():
        if "R2" in res and not np.isnan(res["R2"]) and res["R2"] > best_R2:
            best_R2 = res["R2"]
            best_model = mn
    if best_model is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for eid in exp_data:
            ed = exp_data[eid]
            ax2.scatter(ed["v"], ed["a"], color=exp_colors[eid], s=3, alpha=0.6, label=eid)
        # overlay best model curves per experiment
        res_best = results[best_model]
        params = res_best["params"]
        v_plot = np.linspace(min(v_global), max(v_global), 200)
        for eid in exp_data:
            ed = exp_data[eid]
            F0 = ed["F_ext"]
            try:
                if best_model == "exp":
                    pred = F0 * np.exp(-params["b"] * v_plot)
                elif best_model == "inverse_linear":
                    pred = F0 / (1 + params["c"] * v_plot)
                elif best_model == "power_law":
                    k, p = params["k"], params["p"]
                    mask_plot = v_plot >= 0
                    v_valid = v_plot[mask_plot]
                    if len(v_valid) > 0:
                        pred_valid = F0 - k * v_valid ** p
                        ax2.plot(v_valid, pred_valid, color=exp_colors[eid], linestyle='-', linewidth=1)
                elif best_model == "linear":
                    d = params["d"]
                    pred = F0 - d * v_plot
                elif best_model == "linear_alt":
                    e_ = params["e"]
                    pred = F0 * (1 - e_ * v_plot)
                else:
                    continue
                if best_model != "power_law":
                    ax2.plot(v_plot, pred, color=exp_colors[eid], linestyle='-', linewidth=1)
            except Exception:
                pass
        ax2.set_title(f"Scatter with best model: {best_model} (R²={best_R2:.4f})")
        ax2.set_xlabel("v")
        ax2.set_ylabel("a")
        ax2.legend()
        fig_path2 = output_dir / "best_model_overlay.png"
        fig2.savefig(str(fig_path2), dpi=150)
        plt.close(fig2)
    else:
        fig_path2 = None

    # --- Build observation and metrics ---
    observation_parts = []
    observation_parts.append("对恒外力实验 {} 进行了加速度a与速度v的全局模型拟合分析。".format(", ".join(exp_data.keys())))
    observation_parts.append("使用的a序列：优先a_sg，否则a_new；v序列：优先v_sg，否则v_new。")
    observation_parts.append("拟合了5种模型：")

    metrics = {}
    for model_name in model_names:
        res = results.get(model_name, {})
        if "error" in res:
            observation_parts.append(f"  {model_name}: {res['error']}")
            metrics[f"{model_name}_error"] = res["error"]
            continue
        params_str = ", ".join([f"{k}={v:.4f}" for k,v in res["params"].items()])
        param_stderr_str = ", ".join([f"std_{k}={v:.4f}" for k,v in res["param_stderr"].items()])
        r2 = res.get("R2", np.nan)
        rmse = res.get("RMSE", np.nan)
        corr = res.get("corr_v_residual", np.nan)
        observation_parts.append(f"  {model_name}: params=({params_str}), stderr=({param_stderr_str}), "
                                 f"R²={r2:.4f}, RMSE={rmse:.4f}, 残差-v相关系数={corr:.4f}")
        # Add per-experiment residual stats
        for eid in exp_data:
            exp_stats = res.get("exp_residual_stats", {}).get(f"{eid}_residual_mean", None)
            if exp_stats is not None:
                observation_parts.append(f"    {eid}: residual mean={exp_stats:.4f}")
        # Store metrics
        for k, v in res.get("params", {}).items():
            metrics[f"{model_name}_{k}"] = v
        for k, v in res.get("param_stderr", {}).items():
            metrics[f"{model_name}_{k}_stderr"] = v
        metrics[f"{model_name}_R2"] = r2
        metrics[f"{model_name}_RMSE"] = rmse
        metrics[f"{model_name}_corr_v_residual"] = corr
        # per experiment residuals already in res, but we can put selected ones
        for eid in exp_data:
            exp_res = res.get("exp_residual_stats", {})
            for stat in ["mean","std","min","max"]:
                key = f"{eid}_residual_{stat}"
                if key in exp_res:
                    metrics[f"{model_name}_{eid}_residual_{stat}"] = exp_res[key]
            corr_key = f"{eid}_corr_v_residual"
            if corr_key in res.get("exp_corr_v_residual", {}):
                metrics[f"{model_name}_{eid}_corr_v_residual"] = res["exp_corr_v_residual"][corr_key]

    # Determine if any model stands out
    if best_model is not None:
        best_res = results[best_model]
        best_corr = best_res.get("corr_v_residual", np.nan)
        # Check if corr is close to 0 (say |corr| < 0.3) and R² high
        if best_R2 > 0.9 and abs(best_corr) < 0.3:
            candidate_expr = ""
            if best_model == "exp":
                candidate_expr = f"a = F_ext * exp(-{best_res['params']['b']:.4f} * v)"
            elif best_model == "inverse_linear":
                candidate_expr = f"a = F_ext / (1 + {best_res['params']['c']:.4f} * v)"
            elif best_model == "power_law":
                k, p = best_res['params']['k'], best_res['params']['p']
                candidate_expr = f"a = F_ext - {k:.4f} * v^{p:.4f}"
            elif best_model == "linear":
                d = best_res['params']['d']
                candidate_expr = f"a = F_ext - {d:.4f} * v"
            elif best_model == "linear_alt":
                e_ = best_res['params']['e']
                candidate_expr = f"a = F_ext * (1 - {e_:.4f} * v)"
            if candidate_expr:
                observation_parts.append(f"最佳候选表达式（基于高R²和低残差相关）：{candidate_expr}")
                metrics["candidate_expression"] = candidate_expr
        else:
            observation_parts.append(f"当前最佳模型是 {best_model} (R²={best_R2:.4f})，但残差-v相关系数为{best_corr:.4f}，可能存在系统偏差，需进一步验证。")

    observation = "\n".join(observation_parts)

    # Figures list
    figures = [str(fig_path)]
    if fig_path2 is not None:
        figures.append(str(fig_path2))

    return {
        "observation": observation,
        "derived_series": [],  # No new series created
        "figures": figures,
        "metrics": metrics
    }
