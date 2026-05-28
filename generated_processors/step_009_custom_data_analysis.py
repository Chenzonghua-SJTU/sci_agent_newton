import json
import math
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Tuple

def process(payload: dict) -> dict:
    # Extract parameters
    params = payload.get("parameters", {})
    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        experiment_ids = list(payload.get("experiments", {}).keys())
    
    experiments = payload.get("experiments", {})
    output_dir = Path(payload.get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure we have the required experiments
    missing = [eid for eid in experiment_ids if eid not in experiments]
    if missing:
        raise ValueError(f"Missing experiments: {missing}")

    # Savitzky-Golay parameters (consistent with earlier steps)
    window_length = 7
    polyorder = 3
    dt = 0.1  # will be taken from experiment config or default

    results = {}
    derived_series_list = []
    figures = []
    metrics = {}

    # helper to compute v_sg, a_sg from q(t)
    def compute_sg_series(exp_id, exp_data):
        t = np.array(exp_data["series"]["t"], dtype=float)
        q = np.array(exp_data["series"]["q"], dtype=float)
        config = exp_data.get("config", {})
        dt_val = config.get("dt", dt) if config else dt
        # Use savgol for v and a
        v_sg = signal.savgol_filter(q, window_length=window_length, polyorder=polyorder, deriv=1, delta=dt_val)
        a_sg = signal.savgol_filter(q, window_length=window_length, polyorder=polyorder, deriv=2, delta=dt_val)
        return t, q, v_sg, a_sg

    # Collect data for each experiment
    exp_data_dict = {}
    for eid in experiment_ids:
        exp = experiments[eid]
        series = exp.get("series", {})
        available = exp.get("available_series", [])
        config = exp.get("config", {})
        
        # Determine F_ext
        F_ext = config.get("F_ext", 0.0) if config else 0.0
        # Use existing v_sg and a_sg if available, otherwise compute
        if "v_sg" in series and "a_sg" in series:
            t = np.array(series["t"], dtype=float)
            q = np.array(series["q"], dtype=float) if "q" in series else None
            v_sg = np.array(series["v_sg"], dtype=float)
            a_sg = np.array(series["a_sg"], dtype=float)
        else:
            t, q, v_sg, a_sg = compute_sg_series(eid, exp)
            # Optionally register the new series later
            derived_series_list.append({
                "experiment_id": eid,
                "name": "v_sg",
                "values": v_sg.tolist(),
                "source_name": f"savgol_filter(window={window_length}, polyorder={polyorder}, deriv=1, delta={config.get('dt', 0.1)})",
                "provenance": "generated data processor: custom_data_analysis",
                "description": "velocity from Savitzky-Golay filter"
            })
            derived_series_list.append({
                "experiment_id": eid,
                "name": "a_sg",
                "values": a_sg.tolist(),
                "source_name": f"savgol_filter(window={window_length}, polyorder={polyorder}, deriv=2, delta={config.get('dt', 0.1)})",
                "provenance": "generated data processor: custom_data_analysis",
                "description": "acceleration from Savitzky-Golay filter"
            })
        # Ensure q is available (not all experiments have q in series if not registered)
        if "q" not in series:
            # compute from original? but original should always have q
            # fallback: if not present, we might need to read from original experiment data
            # we assume it's there
            raise ValueError(f"Experiment {eid} missing 'q' series.")
        q = np.array(series["q"], dtype=float)
        t = np.array(series["t"], dtype=float)

        exp_data_dict[eid] = {
            "t": t,
            "q": q,
            "v_sg": v_sg,
            "a_sg": a_sg,
            "F_ext": F_ext,
            "config": config
        }

    # --- 1. a_sg vs v_sg: linear and quadratic fit ---
    fit_linear_results = {}
    fit_quad_results = {}
    for eid in experiment_ids:
        d = exp_data_dict[eid]
        v = d["v_sg"]
        a = d["a_sg"]
        # Remove possible NaNs (from edge effects of savgol)
        mask = ~(np.isnan(v) | np.isnan(a))
        v_clean = v[mask]
        a_clean = a[mask]
        if len(v_clean) < 3:
            continue
        # Linear: a = alpha + beta * v
        A = np.vstack([np.ones_like(v_clean), v_clean]).T
        coeff_lin, resid_lin, _, _ = np.linalg.lstsq(A, a_clean, rcond=None)
        alpha_lin, beta_lin = coeff_lin
        # R^2
        ss_res = resid_lin[0] if len(resid_lin) > 0 else np.sum((a_clean - A @ coeff_lin)**2)
        ss_tot = np.sum((a_clean - np.mean(a_clean))**2)
        r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        # Quadratic: a = alpha + gamma * v^2
        v2 = v_clean ** 2
        A2 = np.vstack([np.ones_like(v2), v2]).T
        coeff_quad, resid_quad, _, _ = np.linalg.lstsq(A2, a_clean, rcond=None)
        alpha_quad, gamma_quad = coeff_quad
        ss_res_q = resid_quad[0] if len(resid_quad) > 0 else np.sum((a_clean - A2 @ coeff_quad)**2)
        r2_quad = 1 - ss_res_q / ss_tot if ss_tot > 0 else 0.0
        fit_linear_results[eid] = {
            "alpha": alpha_lin,
            "beta": beta_lin,
            "R2": r2_lin
        }
        fit_quad_results[eid] = {
            "alpha": alpha_quad,
            "gamma": gamma_quad,
            "R2": r2_quad
        }
        # Store metrics
        metrics[f"{eid}_lin_alpha"] = alpha_lin
        metrics[f"{eid}_lin_beta"] = beta_lin
        metrics[f"{eid}_lin_R2"] = r2_lin
        metrics[f"{eid}_quad_alpha"] = alpha_quad
        metrics[f"{eid}_quad_gamma"] = gamma_quad
        metrics[f"{eid}_quad_R2"] = r2_quad

    # --- 2. a_sg vs q scatter plot ---
    for eid in experiment_ids:
        d = exp_data_dict[eid]
        v = d["v_sg"]
        a = d["a_sg"]
        q_plot = d["q"]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(q_plot, a, s=8, alpha=0.7, label=f"{eid} (F_ext={d['F_ext']})")
        ax.set_xlabel("q")
        ax.set_ylabel("a_sg")
        ax.set_title(f"a_sg vs q - {eid}")
        ax.grid(True)
        fname = f"a_sg_vs_q_{eid}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        figures.append(str(output_dir / fname))

    # --- 3. Quadratic fit of q(t) ---
    q_fit_results = {}
    for eid in experiment_ids:
        d = exp_data_dict[eid]
        t_vals = d["t"]
        q_vals = d["q"]
        # quadratic: q = c0 + c1*t + c2*t^2
        coeffs = np.polyfit(t_vals, q_vals, 2)  # returns c2, c1, c0
        c2, c1, c0 = coeffs
        # predicted
        q_pred = np.polyval(coeffs, t_vals)
        ss_res = np.sum((q_vals - q_pred)**2)
        ss_tot = np.sum((q_vals - np.mean(q_vals))**2)
        r2_q = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        q_fit_results[eid] = {
            "c0": c0,
            "c1": c1,
            "c2": c2,
            "R2": r2_q
        }
        metrics[f"{eid}_c2"] = c2
        metrics[f"{eid}_q_fit_R2"] = r2_q
        # compare c2 with F_ext
        F_ext = d["F_ext"]
        if abs(F_ext) > 1e-12:
            ratio_c2_F = c2 / F_ext
            metrics[f"{eid}_c2_over_F_ext"] = ratio_c2_F
        else:
            metrics[f"{eid}_c2_over_F_ext"] = None  # not defined

    # --- 4. Construct a_res = a_sg - F_ext, then fit vs v and v^2 ---
    a_res_fits = {}
    for eid in experiment_ids:
        d = exp_data_dict[eid]
        F_ext = d["F_ext"]
        a_sg = d["a_sg"]
        v_sg = d["v_sg"]
        mask = ~(np.isnan(a_sg) | np.isnan(v_sg))
        a_clean = a_sg[mask]
        v_clean = v_sg[mask]
        a_res = a_clean - F_ext
        # Store a_res as derived series (full length, fill NaNs with interpolation? We'll output masked version?)
        # Better to keep full length and put NaN for masked? We'll put the computed values for non-masked and np.nan for masked.
        # To keep length same as original series, we create full array with NaN, then fill.
        a_res_full = np.full_like(a_sg, np.nan)
        a_res_full[mask] = a_res
        # Add derived series
        derived_series_list.append({
            "experiment_id": eid,
            "name": "a_res",
            "values": a_res_full.tolist(),
            "source_name": f"a_sg - F_ext (F_ext={F_ext})",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "Residual acceleration after subtracting external force F_ext"
        })
        # Fit a_res = beta * v
        # Linear through origin? The goal says "a_res = β*v" meaning no intercept.
        # We'll do regression without intercept.
        X_v = v_clean.reshape(-1,1)
        model_v = LinearRegression(fit_intercept=False)
        model_v.fit(X_v, a_res)
        beta = model_v.coef_[0]
        # R^2 for no-intercept: use total sum of squares about zero? or about mean? Standard R2 not meaningful without intercept.
        # We'll compute R2 as 1 - SS_res/SS_tot (with SS_tot about zero? better about mean). Use sklearn's score.
        # For no-intercept, sklearn uses total sum of squares centered? We'll compute manually:
        ss_res_v = np.sum((a_res - model_v.predict(X_v))**2)
        ss_tot_v = np.sum((a_res - np.mean(a_res))**2)
        r2_v = 1 - ss_res_v / ss_tot_v if ss_tot_v > 0 else 0.0
        # Fit a_res = gamma * v^2
        v2 = v_clean ** 2
        X_v2 = v2.reshape(-1,1)
        model_v2 = LinearRegression(fit_intercept=False)
        model_v2.fit(X_v2, a_res)
        gamma = model_v2.coef_[0]
        ss_res_v2 = np.sum((a_res - model_v2.predict(X_v2))**2)
        r2_v2 = 1 - ss_res_v2 / ss_tot_v if ss_tot_v > 0 else 0.0
        a_res_fits[eid] = {
            "beta": beta,
            "beta_R2": r2_v,
            "gamma": gamma,
            "gamma_R2": r2_v2
        }
        metrics[f"{eid}_a_res_beta"] = beta
        metrics[f"{eid}_a_res_beta_R2"] = r2_v
        metrics[f"{eid}_a_res_gamma"] = gamma
        metrics[f"{eid}_a_res_gamma_R2"] = r2_v2

    # --- 5. Statistical summaries ---
    for eid in experiment_ids:
        d = exp_data_dict[eid]
        a = d["a_sg"]
        v = d["v_sg"]
        mask = ~(np.isnan(a) | np.isnan(v))
        a_clean = a[mask]
        v_clean = v[mask]
        metrics[f"{eid}_a_sg_mean"] = float(np.mean(a_clean))
        metrics[f"{eid}_a_sg_std"] = float(np.std(a_clean))
        metrics[f"{eid}_v_sg_mean"] = float(np.mean(v_clean))
        metrics[f"{eid}_v_sg_std"] = float(np.std(v_clean))

    # --- Generate comparison plots ---
    # (a) a_sg vs v_sg with linear and quadratic fits per experiment
    # We'll create one figure per experiment showing both fits.
    for eid in experiment_ids:
        d = exp_data_dict[eid]
        v = d["v_sg"]
        a = d["a_sg"]
        mask = ~(np.isnan(v) | np.isnan(a))
        vp = v[mask]
        ap = a[mask]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(vp, ap, s=8, alpha=0.7, label=f"{eid} data (F_ext={d['F_ext']})")
        # Generate sorted points for fit lines
        v_sort = np.sort(vp)
        # linear fit line
        if eid in fit_linear_results:
            lin = fit_linear_results[eid]
            a_pred_lin = lin["alpha"] + lin["beta"] * v_sort
            ax.plot(v_sort, a_pred_lin, 'r-', label=f"linear: α={lin['alpha']:.4f}, β={lin['beta']:.4f}, R2={lin['R2']:.4f}")
        # quadratic fit (a = alpha + gamma*v^2)
        if eid in fit_quad_results:
            quad = fit_quad_results[eid]
            a_pred_quad = quad["alpha"] + quad["gamma"] * v_sort**2
            ax.plot(v_sort, a_pred_quad, 'g--', label=f"quad: α={quad['alpha']:.4f}, γ={quad['gamma']:.4f}, R2={quad['R2']:.4f}")
        ax.set_xlabel("v_sg")
        ax.set_ylabel("a_sg")
        ax.set_title(f"a_sg vs v_sg - {eid}")
        ax.legend(fontsize=7)
        ax.grid(True)
        fname = f"a_vs_v_fits_{eid}.png"
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        figures.append(str(output_dir / fname))

    # (b) a_res vs v (or v^2) cross-experiment comparison
    # We'll create one figure with all experiments: a_res vs v with beta fits (through origin)
    fig, ax = plt.subplots(figsize=(8,5))
    colors = plt.cm.tab10(np.linspace(0,1,len(experiment_ids)))
    for idx, eid in enumerate(experiment_ids):
        d = exp_data_dict[eid]
        v = d["v_sg"]
        a = d["a_sg"]
        mask = ~(np.isnan(v) | np.isnan(a))
        vp = v[mask]
        ap = a[mask]
        a_res = ap - d["F_ext"]
        ax.scatter(vp, a_res, s=8, alpha=0.6, color=colors[idx], label=f"{eid} (F_ext={d['F_ext']})")
        # beta fit
        beta_val = a_res_fits[eid]["beta"]
        gamma_val = a_res_fits[eid]["gamma"]
        # line for beta*v
        v_line = np.linspace(vp.min(), vp.max(), 100)
        ax.plot(v_line, beta_val * v_line, '--', color=colors[idx], label=f"{eid} β={beta_val:.4f}")
    ax.set_xlabel("v_sg")
    ax.set_ylabel("a_res = a_sg - F_ext")
    ax.set_title("a_res vs v_sg with linear β fits (through origin)")
    ax.legend(fontsize=6)
    ax.grid(True)
    fname = "a_res_vs_v_beta_comparison.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures.append(str(output_dir / fname))

    # (c) a_res vs v^2 with gamma fits
    fig, ax = plt.subplots(figsize=(8,5))
    for idx, eid in enumerate(experiment_ids):
        d = exp_data_dict[eid]
        v = d["v_sg"]
        a = d["a_sg"]
        mask = ~(np.isnan(v) | np.isnan(a))
        vp = v[mask]
        ap = a[mask]
        a_res = ap - d["F_ext"]
        v2 = vp**2
        ax.scatter(v2, a_res, s=8, alpha=0.6, color=colors[idx], label=f"{eid} (F_ext={d['F_ext']})")
        gamma_val = a_res_fits[eid]["gamma"]
        v_line = np.linspace(v2.min(), v2.max(), 100)
        ax.plot(v_line, gamma_val * v_line, '--', color=colors[idx], label=f"{eid} γ={gamma_val:.4f}")
    ax.set_xlabel("v_sg^2")
    ax.set_ylabel("a_res")
    ax.set_title("a_res vs v_sg^2 with quadratic γ fits (through origin)")
    ax.legend(fontsize=6)
    ax.grid(True)
    fname = "a_res_vs_v2_gamma_comparison.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures.append(str(output_dir / fname))

    # Collect cross-experiment comparison metrics for β and γ
    beta_values = [a_res_fits[eid]["beta"] for eid in experiment_ids if eid in a_res_fits]
    gamma_values = [a_res_fits[eid]["gamma"] for eid in experiment_ids if eid in a_res_fits]
    if len(beta_values) >= 2:
        metrics["beta_cross_mean"] = float(np.mean(beta_values))
        metrics["beta_cross_std"] = float(np.std(beta_values, ddof=1))
        metrics["beta_cross_rel_std"] = float(np.std(beta_values, ddof=1) / abs(np.mean(beta_values))) if abs(np.mean(beta_values)) > 1e-12 else None
    if len(gamma_values) >= 2:
        metrics["gamma_cross_mean"] = float(np.mean(gamma_values))
        metrics["gamma_cross_std"] = float(np.std(gamma_values, ddof=1))
        metrics["gamma_cross_rel_std"] = float(np.std(gamma_values, ddof=1) / abs(np.mean(gamma_values))) if abs(np.mean(gamma_values)) > 1e-12 else None

    # Also compute average c2/F_ext for constant force experiments (excluding free)
    const_exps = [eid for eid in experiment_ids if abs(exp_data_dict[eid]["F_ext"]) > 1e-12]
    c2_ratios = [metrics.get(f"{eid}_c2_over_F_ext") for eid in const_exps if metrics.get(f"{eid}_c2_over_F_ext") is not None]
    if c2_ratios:
        metrics["c2_over_F_ext_mean"] = float(np.mean(c2_ratios))
        metrics["c2_over_F_ext_std"] = float(np.std(c2_ratios, ddof=1))

    # Build observation text
    obs_lines = []
    obs_lines.append("对所有四个实验进行了自定义分析。")
    # Summarize fits
    obs_lines.append("1. a_sg vs v_sg 拟合结果：")
    for eid in experiment_ids:
        lin = fit_linear_results.get(eid, {})
        quad = fit_quad_results.get(eid, {})
        obs_lines.append(f"   {eid}: 线性 α={lin.get('alpha','N/A'):.4f}, β={lin.get('beta','N/A'):.4f}, R²={lin.get('R2','N/A'):.4f}; "
                         f"二次 α={quad.get('alpha','N/A'):.4f}, γ={quad.get('gamma','N/A'):.4f}, R²={quad.get('R2','N/A'):.4f}")
    obs_lines.append("2. a_sg vs q 散点图已保存。")
    obs_lines.append("3. q(t) 二次拟合结果 (c2 为二次项系数)：")
    for eid in experiment_ids:
        qf = q_fit_results.get(eid, {})
        c2 = qf.get("c2", None)
        F_ext = exp_data_dict[eid]["F_ext"]
        if c2 is not None:
            obs_lines.append(f"   {eid}: c2={c2:.6f}, R²={qf.get('R2','N/A'):.4f}, F_ext={F_ext}")
            if abs(F_ext) > 1e-12:
                ratio = metrics.get(f"{eid}_c2_over_F_ext")
                obs_lines.append(f"        c2/F_ext = {ratio:.6f}")
    # c2/F_ext consistency
    if const_exps:
        ratios_str = ", ".join([f"{eid}={metrics.get(eid+'_c2_over_F_ext','N/A'):.4f}" for eid in const_exps])
        obs_lines.append(f"   恒外力实验 c2/F_ext: {ratios_str}")
        if c2_ratios:
            obs_lines.append(f"   平均 c2/F_ext = {np.mean(c2_ratios):.4f} ± {np.std(c2_ratios, ddof=1):.4f}")
    obs_lines.append("4. 派生量 a_res = a_sg - F_ext 已计算。拟合 a_res = β*v 和 a_res = γ*v^2 结果：")
    for eid in experiment_ids:
        ar = a_res_fits.get(eid, {})
        obs_lines.append(f"   {eid}: β={ar.get('beta','N/A'):.4f} (R²={ar.get('beta_R2','N/A'):.4f}), γ={ar.get('gamma','N/A'):.4f} (R²={ar.get('gamma_R2','N/A'):.4f})")
    # cross-comparison
    if beta_values:
        obs_lines.append(f"   β 跨实验: 均值={np.mean(beta_values):.4f}, 样本标准差={np.std(beta_values,ddof=1):.4f}, 相对标准差={np.std(beta_values,ddof=1)/abs(np.mean(beta_values)):.4f}" if abs(np.mean(beta_values))>1e-12 else "β 均值接近0")
    if gamma_values:
        obs_lines.append(f"   γ 跨实验: 均值={np.mean(gamma_values):.4f}, 样本标准差={np.std(gamma_values,ddof=1):.4f}, 相对标准差={np.std(gamma_values,ddof=1)/abs(np.mean(gamma_values)):.4f}" if abs(np.mean(gamma_values))>1e-12 else "γ 均值接近0")
    obs_lines.append("5. a_sg 和 v_sg 统计量：")
    for eid in experiment_ids:
        obs_lines.append(f"   {eid}: a_sg 均值={metrics.get(eid+'_a_sg_mean','N/A'):.4f}, 标准差={metrics.get(eid+'_a_sg_std','N/A'):.4f}; v_sg 均值={metrics.get(eid+'_v_sg_mean','N/A'):.4f}, 标准差={metrics.get(eid+'_v_sg_std','N/A'):.4f}")
    obs_lines.append("图像包括每个实验的 a_sg vs v_sg 拟合图、a_sg vs q 散点图、以及 a_res 跨实验比较图。")
    obs_lines.append("派生序列 a_res 已为每个实验返回。")
    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics
    }
