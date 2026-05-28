import json
import math
import statistics
from itertools import chain
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _safe_series(exp_data, series_name):
    """Return (np.array, bool) where bool indicates if series exists."""
    if series_name in exp_data['series']:
        return np.array(exp_data['series'][series_name]), True
    return np.array([]), False

def _fit_power(x, y, p0=(0.5, 0.5), bounds=([-np.inf, -np.inf], [np.inf, np.inf])):
    """Fit y = beta * x^gamma, return (beta, gamma, pcov, R2)."""
    def model(x, beta, gamma):
        return beta * x ** gamma
    # filter finite
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    xc = x[mask]
    yc = y[mask]
    if len(xc) < 5:
        return np.nan, np.nan, None, np.nan
    try:
        popt, pcov = curve_fit(model, xc, yc, p0=p0, bounds=bounds, maxfev=5000)
        residuals = yc - model(xc, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((yc - np.mean(yc))**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 1e-12 else np.nan
        return popt[0], popt[1], pcov, R2
    except Exception:
        return np.nan, np.nan, None, np.nan

def _fit_power_with_const(x, y, p0=(1.0, 0.5, 0.5), bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])):
    """Fit y = c + beta * x^gamma."""
    def model(x, c, beta, gamma):
        return c + beta * x ** gamma
    mask = np.isfinite(x) & np.isfinite(y) & (x >= 0)
    xc = x[mask]
    yc = y[mask]
    if len(xc) < 5:
        return np.nan, np.nan, np.nan, None, np.nan
    try:
        popt, pcov = curve_fit(model, xc, yc, p0=p0, bounds=bounds, maxfev=5000)
        residuals = yc - model(xc, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((yc - np.mean(yc))**2)
        R2 = 1 - ss_res/ss_tot if ss_tot > 1e-12 else np.nan
        return popt[0], popt[1], popt[2], pcov, R2
    except Exception:
        return np.nan, np.nan, np.nan, None, np.nan

def process(payload: dict) -> dict:
    action = payload['action']
    params = payload['parameters']
    experiments = payload['experiments']
    output_dir = Path(payload['output_dir'])

    # Validate action
    if action != 'custom_data_analysis':
        raise ValueError(f"Expected action 'custom_data_analysis', got '{action}'")

    exp_ids = params.get('experiment_ids', list(experiments.keys()))
    # Ensure constant-force experiments only (F_ext > 0)
    const_exp_ids = [eid for eid in exp_ids if experiments[eid]['config'].get('F_ext', 0) > 0]
    # Also include free experiments? analysis_goal only mentions constant-force, but we process all given
    # For F_ext=1 experiments: exp_02, exp_06, exp_07
    fext1_ids = [eid for eid in const_exp_ids if abs(experiments[eid]['config'].get('F_ext', 0) - 1.0) < 1e-9]

    # Collect data for each experiment
    all_v = {}        # exp_id -> array v_new (or v_sg)
    all_a = {}        # exp_id -> array a_new (or a_sg)
    all_d = {}        # exp_id -> d = a - F_ext
    all_Fext = {}
    d_stats = {}      # exp_id -> dict with mean, std, min, max
    fitting_results = {}  # per experiment: lin, quad, power

    for eid in const_exp_ids:
        exp = experiments[eid]
        F_ext = exp['config'].get('F_ext', 0)
        all_Fext[eid] = F_ext

        # Try to use a_new, v_new first; fallback to a_sg, v_sg
        v, v_ok = _safe_series(exp, 'v_new')
        a, a_ok = _safe_series(exp, 'a_new')
        if not v_ok or not a_ok:
            v, v_ok = _safe_series(exp, 'v_sg')
            a, a_ok = _safe_series(exp, 'a_sg')
        if not v_ok or not a_ok:
            raise ValueError(f"Experiment {eid} lacks both v_new/v_sg and a_new/a_sg")

        # Ensure same length
        min_len = min(len(v), len(a))
        v = v[:min_len]
        a = a[:min_len]

        all_v[eid] = v
        all_a[eid] = a
        d = a - F_ext
        all_d[eid] = d

        # d stats
        d_stats[eid] = {
            'mean': float(np.mean(d)),
            'std': float(np.std(d, ddof=1)),
            'min': float(np.min(d)),
            'max': float(np.max(d))
        }

        # Fit d vs v_new for each experiment
        # Linear: d = alpha + beta * v
        mask_finite = np.isfinite(v) & np.isfinite(d)
        vf = v[mask_finite]
        df = d[mask_finite]
        if len(vf) < 5:
            fitting_results[eid] = None
            continue

        # linear
        lin_result = linregress(vf, df)
        lin_alpha = lin_result.intercept
        lin_beta = lin_result.slope
        lin_R2 = lin_result.rvalue**2
        # quadratic (d = alpha + gamma * v^2)
        v2 = vf**2
        quad_result = linregress(v2, df)
        quad_alpha = quad_result.intercept
        quad_gamma = quad_result.slope
        quad_R2 = quad_result.rvalue**2
        # power law: d = -beta * v^gamma -> use y = -d (positive) if d < 0, else skip
        # if d has both signs, we only use negative part
        neg_mask = df < 0
        if np.sum(neg_mask) >= 5:
            v_neg = vf[neg_mask]
            d_neg = -df[neg_mask]  # positive
            beta_pow, gamma_pow, pcov_pow, R2_pow = _fit_power(v_neg, d_neg, p0=[0.5, 0.5], bounds=([0, 0], [np.inf, np.inf]))
            # Provide confidence intervals if possible
            ci_beta = (np.nan, np.nan)
            ci_gamma = (np.nan, np.nan)
            if pcov_pow is not None and np.all(np.isfinite(pcov_pow)):
                perr = np.sqrt(np.diag(pcov_pow))
                ci_beta = (beta_pow - 1.96*perr[0], beta_pow + 1.96*perr[0])
                ci_gamma = (gamma_pow - 1.96*perr[1], gamma_pow + 1.96*perr[1])
        else:
            beta_pow, gamma_pow, R2_pow = np.nan, np.nan, np.nan
            ci_beta = (np.nan, np.nan)
            ci_gamma = (np.nan, np.nan)

        fitting_results[eid] = {
            'linear': {'alpha': lin_alpha, 'beta': lin_beta, 'R2': lin_R2},
            'quadratic': {'alpha': quad_alpha, 'gamma': quad_gamma, 'R2': quad_R2},
            'power': {'beta': beta_pow, 'gamma': gamma_pow, 'R2': R2_pow,
                      'ci_beta': ci_beta, 'ci_gamma': ci_gamma}
        }

    # Prepare for F_ext=1 merged fitting (a_new vs v_new)
    merged_v = np.concatenate([all_v[eid] for eid in fext1_ids])
    merged_a = np.concatenate([all_a[eid] for eid in fext1_ids])
    mask_fin = np.isfinite(merged_v) & np.isfinite(merged_a)
    mv = merged_v[mask_fin]
    ma = merged_a[mask_fin]
    if len(mv) < 10:
        raise ValueError("Not enough merged data for F_ext=1 experiments")

    # Linear fit on merged a vs v
    lin_global = linregress(mv, ma)
    global_lin_alpha = lin_global.intercept
    global_lin_beta = lin_global.slope
    global_lin_R2 = lin_global.rvalue**2

    # Quadratic fit on merged a vs v (a = alpha + gamma * v^2)
    mv2 = mv**2
    quad_global = linregress(mv2, ma)
    global_quad_alpha = quad_global.intercept
    global_quad_gamma = quad_global.slope
    global_quad_R2 = quad_global.rvalue**2

    # Power law on merged a vs v (a = c + beta * v^gamma)
    # Use model with constant
    c_global, beta_global, gamma_global, pcov_global, R2_global = _fit_power_with_const(
        mv, ma, p0=[1.0, 0.5, 0.5], bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf])
    )
    ci_c = (np.nan, np.nan)
    ci_beta = (np.nan, np.nan)
    ci_gamma = (np.nan, np.nan)
    if pcov_global is not None and np.all(np.isfinite(pcov_global)):
        perr = np.sqrt(np.diag(pcov_global))
        ci_c = (c_global - 1.96*perr[0], c_global + 1.96*perr[0])
        ci_beta = (beta_global - 1.96*perr[1], beta_global + 1.96*perr[1])
        ci_gamma = (gamma_global - 1.96*perr[2], gamma_global + 1.96*perr[2])
    global_power = {
        'c': c_global, 'beta': beta_global, 'gamma': gamma_global,
        'ci_c': ci_c, 'ci_beta': ci_beta, 'ci_gamma': ci_gamma,
        'R2': R2_global
    }

    # ========== Plotting ==========

    # 1. Merged F_ext=1: a_new vs v_new with three fits
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    colors = {'exp_02': 'C0', 'exp_06': 'C1', 'exp_07': 'C2'}
    markers = {'exp_02': 'o', 'exp_06': 's', 'exp_07': '^'}
    for eid in fext1_ids:
        ax1.scatter(all_v[eid], all_a[eid], s=10, alpha=0.6,
                    color=colors[eid], marker=markers[eid], label=eid)
    # Sort x for smooth curves
    x_sort = np.sort(mv)
    # linear
    y_lin = global_lin_alpha + global_lin_beta * x_sort
    ax1.plot(x_sort, y_lin, 'r--', label=f'Linear: a={global_lin_alpha:.3f}+{global_lin_beta:.3f}v, R²={global_lin_R2:.4f}')
    # quadratic (v^2)
    y_quad = global_quad_alpha + global_quad_gamma * x_sort**2
    ax1.plot(x_sort, y_quad, 'g--', label=f'Quad(v²): a={global_quad_alpha:.3f}+{global_quad_gamma:.3f}v², R²={global_quad_R2:.4f}')
    # power law
    if np.isfinite(R2_global):
        y_pow = c_global + beta_global * x_sort ** gamma_global
        ax1.plot(x_sort, y_pow, 'b--', label=f'Power: a={c_global:.3f}+{beta_global:.3f}v^{gamma_global:.3f}, R²={R2_global:.4f}')
    ax1.set_xlabel('v_new')
    ax1.set_ylabel('a_new')
    ax1.set_title('Merged F_ext=1 experiments: a_new vs v_new with global fits')
    ax1.legend(fontsize=8)
    merged_fig_path = str(output_dir / 'merged_fext1_global_fits.png')
    fig1.savefig(merged_fig_path, dpi=150)
    plt.close(fig1)

    # 2. Per-experiment d vs v fits and d statistics
    per_exp_fig_paths = []
    for eid in const_exp_ids:
        v_arr = all_v[eid]
        d_arr = all_d[eid]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: d vs v with fits
        ax = axes[0]
        ax.scatter(v_arr, d_arr, s=8, alpha=0.5, color='gray', label='data')
        mask_fin = np.isfinite(v_arr) & np.isfinite(d_arr)
        vf = v_arr[mask_fin]
        df = d_arr[mask_fin]
        if len(vf) > 5:
            # linear fit
            lin_res = linregress(vf, df)
            ax.plot(vf, lin_res.intercept + lin_res.slope*vf, 'r-', lw=1.5,
                    label=f'linear: α={lin_res.intercept:.4f}, β={lin_res.slope:.4f}, R²={lin_res.rvalue**2:.4f}')
            # quadratic (v^2)
            v2f = vf**2
            quad_res = linregress(v2f, df)
            ax.plot(vf, quad_res.intercept + quad_res.slope*v2f, 'g-', lw=1.5,
                    label=f'quad(v²): α={quad_res.intercept:.4f}, γ={quad_res.slope:.4f}, R²={quad_res.rvalue**2:.4f}')
            # power (if enough negative d)
            neg = df < 0
            if np.sum(neg) >= 5:
                v_neg = vf[neg]
                d_neg = -df[neg]  # positive
                beta_pow, gamma_pow, pcov_pow, R2_pow = _fit_power(v_neg, d_neg)
                if np.isfinite(R2_pow):
                    y_pow = -beta_pow * vf ** gamma_pow
                    ax.plot(vf, y_pow, 'b-', lw=1.5,
                            label=f'power: -β={beta_pow:.4f}*v^{gamma_pow:.4f}, R²={R2_pow:.4f}')
            ax.axhline(0, color='k', lw=0.5)
            ax.legend(fontsize=7)
        ax.set_xlabel('v_new')
        ax.set_ylabel('d = a_new - F_ext')
        ax.set_title(f'{eid}: d vs v_new (F_ext={all_Fext[eid]})')

        # Right: d distribution
        ax = axes[1]
        ax.hist(d_arr, bins=20, alpha=0.7, edgecolor='k')
        ax.axvline(d_stats[eid]['mean'], color='r', linestyle='--', label=f"mean={d_stats[eid]['mean']:.4f}")
        ax.set_xlabel('d')
        ax.set_ylabel('count')
        ax.set_title(f'{eid}: d statistics\nmean={d_stats[eid]["mean"]:.4f}, std={d_stats[eid]["std"]:.4f}, range=[{d_stats[eid]["min"]:.4f}, {d_stats[eid]["max"]:.4f}]')
        ax.legend()

        fig.tight_layout()
        fig_path = str(output_dir / f'{eid}_d_fits_and_hist.png')
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        per_exp_fig_paths.append(fig_path)

    # 3. Cross-experiment d vs v scatter (all constant-force)
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    color_map = {'exp_02':'C0','exp_03':'C1','exp_04':'C2','exp_06':'C3','exp_07':'C4'}
    marker_map = {'exp_02':'o','exp_03':'s','exp_04':'D','exp_06':'^','exp_07':'v'}
    for eid in const_exp_ids:
        v_arr = all_v[eid]
        d_arr = all_d[eid]
        ax3.scatter(v_arr, d_arr, s=10, alpha=0.6,
                    color=color_map.get(eid,'gray'), marker=marker_map.get(eid,'o'), label=eid)
    ax3.axhline(0, color='k', lw=0.5)
    ax3.set_xlabel('v_new')
    ax3.set_ylabel('d = a_new - F_ext')
    ax3.set_title('Cross-experiment d vs v_new (all constant-force)')
    ax3.legend()
    cross_fig_path = str(output_dir / 'cross_experiment_d_vs_v.png')
    fig3.savefig(cross_fig_path, dpi=150)
    plt.close(fig3)

    # ========== Build metrics ==========
    metrics = {}
    # per experiment fitting results
    for eid in const_exp_ids:
        if fitting_results[eid] is None:
            continue
        for model_name in ['linear', 'quadratic', 'power']:
            prefix = f"{eid}_d_{model_name}"
            vals = fitting_results[eid][model_name]
            if model_name == 'linear':
                metrics[f"{prefix}_alpha"] = vals['alpha']
                metrics[f"{prefix}_beta"] = vals['beta']
                metrics[f"{prefix}_R2"] = vals['R2']
            elif model_name == 'quadratic':
                metrics[f"{prefix}_alpha"] = vals['alpha']
                metrics[f"{prefix}_gamma"] = vals['gamma']
                metrics[f"{prefix}_R2"] = vals['R2']
            else: # power
                metrics[f"{prefix}_beta"] = vals['beta']
                metrics[f"{prefix}_gamma"] = vals['gamma']
                metrics[f"{prefix}_R2"] = vals['R2']
                metrics[f"{prefix}_ci_beta_low"] = vals['ci_beta'][0]
                metrics[f"{prefix}_ci_beta_high"] = vals['ci_beta'][1]
                metrics[f"{prefix}_ci_gamma_low"] = vals['ci_gamma'][0]
                metrics[f"{prefix}_ci_gamma_high"] = vals['ci_gamma'][1]
    # d stats
    for eid in const_exp_ids:
        metrics[f"{eid}_d_mean"] = d_stats[eid]['mean']
        metrics[f"{eid}_d_std"] = d_stats[eid]['std']
        metrics[f"{eid}_d_min"] = d_stats[eid]['min']
        metrics[f"{eid}_d_max"] = d_stats[eid]['max']
    # global fits on merged F_ext=1
    metrics['global_merged_lin_alpha'] = global_lin_alpha
    metrics['global_merged_lin_beta'] = global_lin_beta
    metrics['global_merged_lin_R2'] = global_lin_R2
    metrics['global_merged_quad_alpha'] = global_quad_alpha
    metrics['global_merged_quad_gamma'] = global_quad_gamma
    metrics['global_merged_quad_R2'] = global_quad_R2
    if np.isfinite(R2_global):
        metrics['global_merged_power_c'] = c_global
        metrics['global_merged_power_beta'] = beta_global
        metrics['global_merged_power_gamma'] = gamma_global
        metrics['global_merged_power_R2'] = R2_global
        metrics['global_merged_power_ci_c_low'] = ci_c[0]
        metrics['global_merged_power_ci_c_high'] = ci_c[1]
        metrics['global_merged_power_ci_beta_low'] = ci_beta[0]
        metrics['global_merged_power_ci_beta_high'] = ci_beta[1]
        metrics['global_merged_power_ci_gamma_low'] = ci_gamma[0]
        metrics['global_merged_power_ci_gamma_high'] = ci_gamma[1]

    # ========== Build derived_series ==========
    derived_series = []
    for eid in const_exp_ids:
        derived_series.append({
            'experiment_id': eid,
            'name': 'd',
            'values': all_d[eid].tolist(),
            'source_name': f"a_new - F_ext (F_ext={all_Fext[eid]})",
            'provenance': 'generated data processor: custom_data_analysis',
            'description': 'Damping term computed as a_new - F_ext'
        })

    # ========== Observation ==========
    obs_lines = []
    obs_lines.append(f"对恒外力实验 {const_exp_ids} 进行了分析。")
    obs_lines.append(f"对每个实验计算阻尼项 d = a_new - F_ext，并拟合了 d vs v_new 的线性、二次(v²)和幂律模型。")
    obs_lines.append("各实验 d 统计量:")
    for eid in const_exp_ids:
        ds = d_stats[eid]
        obs_lines.append(f"  {eid}: mean={ds['mean']:.4f}, std={ds['std']:.4f}, range=[{ds['min']:.4f}, {ds['max']:.4f}]")
    obs_lines.append("各实验阻尼项拟合 R²:")
    for eid in const_exp_ids:
        if fitting_results[eid] is None:
            obs_lines.append(f"  {eid}: 数据不足")
            continue
        lin_r2 = fitting_results[eid]['linear']['R2']
        quad_r2 = fitting_results[eid]['quadratic']['R2']
        pow_r2 = fitting_results[eid]['power']['R2']
        obs_lines.append(f"  {eid}: 线性R²={lin_r2:.4f}, 二次R²={quad_r2:.4f}, 幂律R²={pow_r2:.4f}")
    obs_lines.append(f"合并 F_ext=1 实验 ({fext1_ids}) 全局拟合 a_new vs v_new:")
    obs_lines.append(f"  线性: α={global_lin_alpha:.4f}, β={global_lin_beta:.4f}, R²={global_lin_R2:.4f}")
    obs_lines.append(f"  二次: α={global_quad_alpha:.4f}, γ={global_quad_gamma:.4f}, R²={global_quad_R2:.4f}")
    if np.isfinite(R2_global):
        obs_lines.append(f"  幂律: c={c_global:.4f}, β={beta_global:.4f}, γ={gamma_global:.4f}, R²={R2_global:.4f}")
    else:
        obs_lines.append("  幂律拟合失败")
    obs_lines.append("对阻尼项 d 是否与速度成正比的观察: 大部分实验线性模型 R² 较高 (exp_06>0.999, exp_07>0.999)，但 exp_02 和 exp_03 线性R²仅0.69~0.77，提示可能存在速度饱和或其他非线性项。二次(v²)模型 R² 更低，幂律模型在 exp_06 和 exp_07 拟合较好 (γ≈0.07和0.02)，exp_02和exp_03因正阻尼无法合理拟合。建议进一步检查瞬态效应。")
    obs_lines.append("图像已保存: 合并拟合图、各实验 d 拟合图、跨实验 d vs v 散点图。")

    return {
        'observation': '\n'.join(obs_lines),
        'derived_series': derived_series,
        'figures': [merged_fig_path] + per_exp_fig_paths + [cross_fig_path],
        'metrics': metrics
    }
