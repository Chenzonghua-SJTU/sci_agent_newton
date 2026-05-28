import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import os
import json
import traceback

def _estimate_kinematics_from_q(t, q, window=11, polyorder=3):
    """Estimate v and a from q(t) using Savitzky-Golay filter."""
    dt = t[1] - t[0] if len(t) > 1 else 0.1
    q_smooth = savgol_filter(q, window, polyorder)
    v = savgol_filter(q, window, polyorder, deriv=1) / dt
    a = savgol_filter(q, window, polyorder, deriv=2) / (dt**2)
    return q_smooth, v, a

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    parameters = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", "/tmp")
    os.makedirs(output_dir, exist_ok=True)

    analysis_goal = parameters.get("analysis_goal", "")
    exp_ids = parameters.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(experiments.keys())

    # Validate experiments exist
    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")

    # Collect all data from constant-force experiments (force_field_type=constant)
    all_v = []
    all_drag = []
    all_F = []
    per_exp_data = {}  # eid -> (v, drag, F_ext)

    for eid in exp_ids:
        exp = experiments[eid]
        config = exp.get("config", {})
        series = exp.get("series", {})
        available = exp.get("available_series", [])
        F_ext = config.get("F_ext", 0.0)
        # Ensure we have v_est and drag
        if "v_est" in available and "drag" in available:
            v = np.array(series["v_est"])
            drag = np.array(series["drag"])
        else:
            # Estimate from q
            t = np.array(series["t"])
            q = np.array(series["q"])
            # Choose window based on length
            n = len(t)
            window = min(n, 11)
            if window % 2 == 0:
                window += 1
            polyorder = min(3, window-1)
            q_smooth, v, a = _estimate_kinematics_from_q(t, q, window=window, polyorder=polyorder)
            drag = F_ext - a
            # Add as derived series (will be registered in main loop later)
            # But for now, use v and drag
        # Filter out non-positive velocities for power law
        mask = v > 0
        v_ok = v[mask]
        drag_ok = drag[mask]
        if len(v_ok) < 5:
            continue
        all_v.append(v_ok)
        all_drag.append(drag_ok)
        all_F.append(np.full_like(v_ok, F_ext))
        per_exp_data[eid] = (v_ok, drag_ok, F_ext)

    if len(all_v) == 0:
        return {"observation": "无有效数据", "derived_series": [], "figures": [], "metrics": {}}

    # Concatenate all data
    v_all = np.concatenate(all_v)
    drag_all = np.concatenate(all_drag)
    F_all = np.concatenate(all_F)

    # 1) Global power law: drag = c * v^b
    def power_law(v, c, b):
        return c * (v ** b)

    p0 = [0.5, 0.5]  # initial guess
    try:
        popt_power, pcov_power = curve_fit(power_law, v_all, drag_all, p0=p0, maxfev=10000)
        c_opt, b_opt = popt_power
        drag_pred_power = power_law(v_all, c_opt, b_opt)
        residuals_power = drag_all - drag_pred_power
        ss_res = np.sum(residuals_power ** 2)
        ss_tot = np.sum((drag_all - np.mean(drag_all)) ** 2)
        r2_power = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse_power = np.sqrt(np.mean(residuals_power ** 2))
    except Exception as e:
        c_opt, b_opt = np.nan, np.nan
        drag_pred_power = np.full_like(drag_all, np.nan)
        r2_power = np.nan
        rmse_power = np.nan

    # 2) Linear+sqrt: drag = p * v + q * sqrt(v)
    # Design matrix: [v, sqrt(v)]
    A = np.column_stack([v_all, np.sqrt(v_all)])
    try:
        coeff_lin, _, _, _ = np.linalg.lstsq(A, drag_all, rcond=None)
        p_opt, q_opt = coeff_lin[0], coeff_lin[1]
        drag_pred_lin = A @ coeff_lin
        residuals_lin = drag_all - drag_pred_lin
        ss_res_lin = np.sum(residuals_lin ** 2)
        r2_lin = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0.0
        rmse_lin = np.sqrt(np.mean(residuals_lin ** 2))
    except Exception as e:
        p_opt, q_opt = np.nan, np.nan
        drag_pred_lin = np.full_like(drag_all, np.nan)
        r2_lin = np.nan
        rmse_lin = np.nan

    # 3) Per-experiment predictions for derived series (optional)
    derived_series = []
    for eid in exp_ids:
        if eid not in per_exp_data:
            continue
        v_exp, drag_exp, F_exp = per_exp_data[eid]
        drag_pred_power_exp = power_law(v_exp, c_opt, b_opt) if not np.isnan(c_opt) else np.full_like(v_exp, np.nan)
        drag_pred_lin_exp = np.column_stack([v_exp, np.sqrt(v_exp)]) @ [p_opt, q_opt] if not np.isnan(p_opt) else np.full_like(v_exp, np.nan)
        # We will add predictions as series
        derived_series.append({
            "experiment_id": eid,
            "name": "drag_power_fit",
            "values": drag_pred_power_exp.tolist(),
            "source_name": "global power law fit: drag = c*v^b",
            "provenance": "generated data processor: custom_data_analysis"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "drag_lin_sqrt_fit",
            "values": drag_pred_lin_exp.tolist(),
            "source_name": "global linear+sqrt fit: drag = p*v + q*sqrt(v)",
            "provenance": "generated data processor: custom_data_analysis"
        })
        # Also compute a_pred = F_ext - drag_pred for plotting a vs v? optional
        a_pred_power = F_exp - drag_pred_power_exp
        derived_series.append({
            "experiment_id": eid,
            "name": "a_power_fit",
            "values": a_pred_power.tolist(),
            "source_name": "global power law model a = F_ext - c*v^b",
            "provenance": "generated data processor: custom_data_analysis"
        })

    # 4) Plots
    figures = []

    # ---- Global fit plots ----
    # Plot1: drag vs v with both fits
    fig, ax = plt.subplots(figsize=(8, 6))
    # Scatter all data, color by F_ext
    unique_F = sorted(set(F_all))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(unique_F)))
    for f, col in zip(unique_F, colors):
        mask_f = F_all == f
        ax.scatter(v_all[mask_f], drag_all[mask_f], c=[col], s=10, alpha=0.6, label=f'F_ext={f}')
    # Plot fits
    v_grid = np.linspace(min(v_all)*0.9, max(v_all)*1.1, 200)
    if not np.isnan(c_opt):
        ax.plot(v_grid, power_law(v_grid, c_opt, b_opt), 'k-', lw=2, label=f'Power law: c={c_opt:.4f}, b={b_opt:.4f}')
    if not np.isnan(p_opt):
        ax.plot(v_grid, p_opt*v_grid + q_opt*np.sqrt(v_grid), 'r--', lw=2, label=f'Linear+sqrt: p={p_opt:.4f}, q={q_opt:.4f}')
    ax.set_xlabel('v (m/s)')
    ax.set_ylabel('Drag force')
    ax.set_title('Global drag vs velocity fits')
    ax.legend()
    fig_path = os.path.join(output_dir, 'global_drag_vs_v_fits.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    figures.append(fig_path)

    # Plot2: Residuals for both models vs v
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes
    if not np.isnan(c_opt):
        ax1.scatter(v_all, residuals_power, s=5, alpha=0.5, c='blue')
        ax1.axhline(0, color='gray', lw=1)
        ax1.set_xlabel('v')
        ax1.set_ylabel('Residuals')
        ax1.set_title(f'Power law residuals (R²={r2_power:.4f}, RMSE={rmse_power:.4f})')
    if not np.isnan(p_opt):
        ax2.scatter(v_all, residuals_lin, s=5, alpha=0.5, c='red')
        ax2.axhline(0, color='gray', lw=1)
        ax2.set_xlabel('v')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'Linear+sqrt residuals (R²={r2_lin:.4f}, RMSE={rmse_lin:.4f})')
    fig_path = os.path.join(output_dir, 'global_residuals.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    figures.append(fig_path)

    # Plot3: Overlay each experiment separately with color by F_ext
    fig, ax = plt.subplots(figsize=(8, 6))
    for eid, (v_exp, drag_exp, F_exp) in per_exp_data.items():
        ax.scatter(v_exp, drag_exp, s=20, alpha=0.6, label=f'{eid} (F={F_exp})')
    ax.set_xlabel('v')
    ax.set_ylabel('Drag')
    ax.set_title('Per-experiment drag vs v (colored by experiment)')
    ax.legend(fontsize=8)
    fig_path = os.path.join(output_dir, 'per_experiment_drag_vs_v.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    figures.append(fig_path)

    # Plot4: Check if drag curves coincide by plotting drag/F_ext vs v? Or just the previous plot.
    # For completeness, plot drag/F_ext vs v to see independence
    fig, ax = plt.subplots(figsize=(8, 6))
    for eid, (v_exp, drag_exp, F_exp) in per_exp_data.items():
        if F_exp > 0:
            ax.scatter(v_exp, drag_exp / F_exp, s=20, alpha=0.6, label=f'{eid} (F={F_exp})')
    ax.set_xlabel('v')
    ax.set_ylabel('Drag / F_ext')
    ax.set_title('Normalized drag (drag/F_ext) vs velocity')
    ax.legend(fontsize=8)
    fig_path = os.path.join(output_dir, 'normalized_drag_vs_v.png')
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    figures.append(fig_path)

    # 5) Build observation
    obs_parts = []
    obs_parts.append("对实验 {} 进行了全局联合建模。".format(', '.join(exp_ids)))
    if not np.isnan(c_opt):
        obs_parts.append(f"幂律模型 drag = c * v^b 拟合结果：c={c_opt:.4f}, b={b_opt:.4f}, R²={r2_power:.4f}, RMSE={rmse_power:.4f}。")
    else:
        obs_parts.append("幂律模型拟合失败。")
    if not np.isnan(p_opt):
        obs_parts.append(f"线性+平方根模型 drag = p*v + q*sqrt(v) 拟合结果：p={p_opt:.4f}, q={q_opt:.4f}, R²={r2_lin:.4f}, RMSE={rmse_lin:.4f}。")
    else:
        obs_parts.append("线性+平方根模型拟合失败。")
    # Compare R²
    if not np.isnan(r2_power) and not np.isnan(r2_lin):
        if r2_power > r2_lin:
            obs_parts.append(f"幂律模型的R²({r2_power:.4f})略高于线性+平方根模型({r2_lin:.4f})，但差距不大（<0.05? 实际差值={r2_power-r2_lin:.4f}）。")
        else:
            obs_parts.append(f"线性+平方根模型的R²({r2_lin:.4f})略高于幂律模型({r2_power:.4f})，但差距不大（差值={r2_lin-r2_power:.4f}）。")
    obs_parts.append("不同F_ext实验的drag vs v散点图已绘制，重叠程度可目视判断阻力是否独立于外加力场。")
    obs_parts.append("残差图已保存。建议决策LLM结合数值和图形选择最简阻力形式。")
    observation = "\n".join(obs_parts)

    metrics = {
        "global_power_b": float(b_opt) if not np.isnan(b_opt) else None,
        "global_power_c": float(c_opt) if not np.isnan(c_opt) else None,
        "global_power_R2": float(r2_power) if not np.isnan(r2_power) else None,
        "global_power_RMSE": float(rmse_power) if not np.isnan(rmse_power) else None,
        "global_lin_sqrt_p": float(p_opt) if not np.isnan(p_opt) else None,
        "global_lin_sqrt_q": float(q_opt) if not np.isnan(q_opt) else None,
        "global_lin_sqrt_R2": float(r2_lin) if not np.isnan(r2_lin) else None,
        "global_lin_sqrt_RMSE": float(rmse_lin) if not np.isnan(rmse_lin) else None,
        "n_data_points": len(v_all)
    }

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
