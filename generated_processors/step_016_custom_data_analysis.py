import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _compute_sg(data: np.ndarray, window: int = 11, polyorder: int = 3, dt: float = 0.1,
                deriv: int = 0) -> np.ndarray:
    """Apply Savitzky-Golay filter to smooth or differentiate a 1D signal."""
    if len(data) < window:
        return data.copy()
    return savgol_filter(data, window_length=window, polyorder=polyorder, deriv=deriv, delta=dt)


def _ensure_series(exp: dict, name: str, data_config: dict) -> np.ndarray:
    """Return existing series or compute from q if not present."""
    if name in exp.get('series', {}):
        return np.array(exp['series'][name])
    # fallback: compute from q
    q = np.array(exp['series'].get('q', []))
    if len(q) == 0:
        raise ValueError(f"Experiment {exp.get('_id', '?')} has no q series to derive {name}")
    t = np.array(exp['series'].get('t', []))
    if len(t) == 0:
        dt = data_config.get('dt', 0.1)
    else:
        dt = t[1] - t[0] if len(t) > 1 else data_config.get('dt', 0.1)
    window = 11
    polyorder = 3
    if name == 'v_sg':
        return _compute_sg(q, window, polyorder, dt, deriv=1)
    elif name == 'a_sg':
        return _compute_sg(q, window, polyorder, dt, deriv=2)
    else:
        raise ValueError(f"Unknown series name {name}")


def _linear_model(v, A, B):
    return A - B * v


def _quad_model(v, A, C):
    return A - C * v ** 2


def _power_model(v, beta, gamma, F_ext):
    return F_ext - beta * (v ** gamma)


def _fit_and_metrics(x, y, model_func, p0, bounds=(-np.inf, np.inf)):
    """Fit model via curve_fit and return (popt, pcov, R2)."""
    popt, pcov = optimize.curve_fit(
        model_func, x, y, p0=p0, maxfev=10000, bounds=bounds
    )
    y_pred = model_func(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # 95% CI from covariance
    perr = np.sqrt(np.diag(pcov))
    ci = np.array([popt - 1.96 * perr, popt + 1.96 * perr]).T
    return popt, ci, r2


def process(payload: dict) -> dict:
    action = payload.get("action", "custom_data_analysis")
    params = payload.get("parameters", {})
    experiments_raw = payload.get("experiments", {})
    output_dir = payload.get("output_dir", "/tmp")

    # Determine which experiments to process
    exp_ids = params.get("experiment_ids", [])
    # analysis_goal also mentions exp_01 and exp_05 for zero check
    all_exp_ids = list(experiments_raw.keys())
    target_ids = [eid for eid in all_exp_ids if eid in exp_ids or eid in ['exp_01', 'exp_05']]
    if not target_ids:
        target_ids = all_exp_ids

    # Prepare data for each experiment
    exp_data = {}
    for eid in target_ids:
        exp = experiments_raw[eid]
        config = exp.get('config', {})
        F_ext = config.get('F_ext', 0.0)
        force_type = config.get('force_field_type', '')

        # Ensure v_sg and a_sg exist
        v_sg = _ensure_series(exp, 'v_sg', config)
        a_sg = _ensure_series(exp, 'a_sg', config)
        t = np.array(exp['series'].get('t', np.linspace(0, (len(v_sg)-1)*config.get('dt', 0.1), len(v_sg))))

        exp_data[eid] = {
            'F_ext': F_ext,
            'force_type': force_type,
            't': t,
            'v_sg': v_sg,
            'a_sg': a_sg,
        }

    # Identify constant-force experiments (F_ext > 0)
    constant_exps = {eid: d for eid, d in exp_data.items()
                     if d['force_type'] == 'constant' and d['F_ext'] > 0}

    # ---------------------------
    # 1. Fitting for each constant force experiment
    # ---------------------------
    fit_results = {}
    d_series = {}  # damping d = a_sg - F_ext
    for eid, d in constant_exps.items():
        v = d['v_sg']
        a = d['a_sg']
        F = d['F_ext']

        # Linear fit a = A - B*v
        p0_lin = [np.mean(a), 0.1]
        bounds_lin = ([-np.inf, -np.inf], [np.inf, np.inf])
        try:
            (A_lin, B_lin), ci_lin, r2_lin = _fit_and_metrics(v, a, _linear_model, p0_lin)
        except Exception:
            A_lin, B_lin, ci_lin, r2_lin = np.nan, np.nan, [[np.nan, np.nan], [np.nan, np.nan]], np.nan

        # Quadratic fit a = A - C*v^2
        p0_quad = [np.mean(a), 0.01]
        try:
            (A_quad, C_quad), ci_quad, r2_quad = _fit_and_metrics(v, a, _quad_model, p0_quad)
        except Exception:
            A_quad, C_quad, ci_quad, r2_quad = np.nan, np.nan, [[np.nan, np.nan], [np.nan, np.nan]], np.nan

        # Power-law fit a = F - beta * v^gamma, fixed alpha = F
        # Use log-log initialization: beta ~ (F - a)/v^gamma, gamma roughly from linear slope in log-log
        # Better: provide reasonable p0
        p0_power = [0.5, 1.0]  # beta, gamma
        # Bounds: beta > 0, gamma > 0 (avoid v^0 issues)
        bounds_power = ([1e-10, 0.0], [np.inf, 5.0])
        try:
            popt_power, pcov_power = optimize.curve_fit(
                lambda v, beta, g: _power_model(v, beta, g, F), v, a,
                p0=p0_power, bounds=bounds_power, maxfev=10000
            )
            beta_power, gamma_power = popt_power
            y_pred = _power_model(v, beta_power, gamma_power, F)
            ss_res = np.sum((a - y_pred) ** 2)
            ss_tot = np.sum((a - np.mean(a)) ** 2)
            r2_power = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            perr = np.sqrt(np.diag(pcov_power))
            ci_power = np.array([popt_power - 1.96 * perr, popt_power + 1.96 * perr]).T
        except Exception:
            beta_power, gamma_power = np.nan, np.nan
            ci_power = [[np.nan, np.nan], [np.nan, np.nan]]
            r2_power = np.nan

        fit_results[eid] = {
            'linear': {'A': A_lin, 'B': B_lin, 'ci_A': ci_lin[0].tolist(), 'ci_B': ci_lin[1].tolist(), 'R2': r2_lin},
            'quadratic': {'A': A_quad, 'C': C_quad, 'ci_A': ci_quad[0].tolist(), 'ci_C': ci_quad[1].tolist(), 'R2': r2_quad},
            'power': {'beta': beta_power, 'gamma': gamma_power, 'ci_beta': ci_power[0].tolist(), 'ci_gamma': ci_power[1].tolist(), 'R2': r2_power}
        }

        # Damping term
        d_series[eid] = a - F

    # ---------------------------
    # 2. Graphics: combined a_sg vs v_sg scatter for constant experiments
    # ---------------------------
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for i, (eid, d_exp) in enumerate(constant_exps.items()):
        ax1.scatter(d_exp['v_sg'], d_exp['a_sg'], c=colors[i % len(colors)],
                    marker=markers[i % len(markers)], label=eid, s=10, alpha=0.7)
    ax1.set_xlabel('v_sg')
    ax1.set_ylabel('a_sg')
    ax1.set_title('a_sg vs v_sg for constant-force experiments')
    ax1.legend()
    scatter_path = str(Path(output_dir) / 'constant_force_scatter.png')
    fig1.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # ---------------------------
    # 3. Per-experiment fit plots (optional but good)
    # ---------------------------
    fit_fig_paths = []
    for eid in constant_exps.keys():
        v = constant_exps[eid]['v_sg']
        a = constant_exps[eid]['a_sg']
        F = constant_exps[eid]['F_ext']
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(v, a, label='data', s=10, alpha=0.6, c='grey')
        # Linear fit
        if not np.isnan(fit_results[eid]['linear']['A']):
            v_sort = np.sort(v)
            ax.plot(v_sort, _linear_model(v_sort, fit_results[eid]['linear']['A'],
                                          fit_results[eid]['linear']['B']),
                    '--', label=f"Linear R²={fit_results[eid]['linear']['R2']:.3f}")
        # Quadratic fit
        if not np.isnan(fit_results[eid]['quadratic']['A']):
            ax.plot(v_sort, _quad_model(v_sort, fit_results[eid]['quadratic']['A'],
                                        fit_results[eid]['quadratic']['C']),
                    '-.', label=f"Quad R²={fit_results[eid]['quadratic']['R2']:.3f}")
        # Power fit
        if not np.isnan(fit_results[eid]['power']['beta']):
            ax.plot(v_sort, _power_model(v_sort, fit_results[eid]['power']['beta'],
                                         fit_results[eid]['power']['gamma'], F),
                    ':', label=f"Power R²={fit_results[eid]['power']['R2']:.3f}")
        ax.set_xlabel('v_sg')
        ax.set_ylabel('a_sg')
        ax.set_title(f'{eid} (F_ext={F})')
        ax.legend(fontsize=8)
        path = str(Path(output_dir) / f'{eid}_fits.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        fit_fig_paths.append(path)

    # ---------------------------
    # 4. Damping d vs v_sg combined scatter
    # ---------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for i, (eid, d_val) in enumerate(d_series.items()):
        v = constant_exps[eid]['v_sg']
        ax2.scatter(v, d_val, c=colors[i % len(colors)], marker=markers[i % len(markers)],
                    label=eid, s=10, alpha=0.7)
    ax2.set_xlabel('v_sg')
    ax2.set_ylabel('d = a_sg - F_ext')
    ax2.set_title('Damping term d vs v_sg')
    ax2.legend()
    d_scatter_path = str(Path(output_dir) / 'd_vs_v_scatter.png')
    fig2.savefig(d_scatter_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # ---------------------------
    # 5. Global fit of d = -beta * v^gamma on combined data
    # ---------------------------
    all_v = np.concatenate([constant_exps[eid]['v_sg'] for eid in constant_exps])
    all_d = np.concatenate([d_series[eid] for eid in constant_exps])
    # Filter negative v (should not happen but safe)
    mask_v = all_v > 0
    all_v = all_v[mask_v]
    all_d = all_d[mask_v]
    # Fit d = -beta * v^gamma => model: d = -beta * v^gamma
    def d_model(v, beta, gamma):
        return -beta * (v ** gamma)
    try:
        p0_d = [0.5, 1.0]
        bounds_d = ([1e-10, 0.0], [np.inf, 3.0])
        popt_d, pcov_d = optimize.curve_fit(d_model, all_v, all_d, p0=p0_d, bounds=bounds_d, maxfev=10000)
        beta_global, gamma_global = popt_d
        y_pred_d = d_model(all_v, beta_global, gamma_global)
        ss_res_d = np.sum((all_d - y_pred_d) ** 2)
        ss_tot_d = np.sum((all_d - np.mean(all_d)) ** 2)
        r2_global = 1 - ss_res_d / ss_tot_d if ss_tot_d > 0 else 0.0
        perr_d = np.sqrt(np.diag(pcov_d))
        ci_global = np.array([popt_d - 1.96 * perr_d, popt_d + 1.96 * perr_d]).T
    except Exception:
        beta_global, gamma_global = np.nan, np.nan
        ci_global = [[np.nan, np.nan], [np.nan, np.nan]]
        r2_global = np.nan

    # ---------------------------
    # 6. Check exp_01 and exp_05 a_sg ~ 0
    # ---------------------------
    zero_check = {}
    for eid in ['exp_01', 'exp_05']:
        if eid in exp_data:
            a = exp_data[eid]['a_sg']
            mean_a = np.mean(a)
            std_a = np.std(a, ddof=1)
            # t-test against 0
            if len(a) > 1 and std_a > 0:
                t_stat, p_val = stats.ttest_1samp(a, 0.0)
            else:
                t_stat, p_val = 0.0, 1.0
            zero_check[eid] = {'mean': mean_a, 'std': std_a, 'p_value': p_val}

    # ---------------------------
    # Build metrics
    # ---------------------------
    metrics = {}
    for eid in constant_exps:
        prefix = eid
        m = fit_results[eid]
        metrics[f'{prefix}_lin_A'] = m['linear']['A']
        metrics[f'{prefix}_lin_A_ci_low'] = m['linear']['ci_A'][0]
        metrics[f'{prefix}_lin_A_ci_high'] = m['linear']['ci_A'][1]
        metrics[f'{prefix}_lin_B'] = m['linear']['B']
        metrics[f'{prefix}_lin_B_ci_low'] = m['linear']['ci_B'][0]
        metrics[f'{prefix}_lin_B_ci_high'] = m['linear']['ci_B'][1]
        metrics[f'{prefix}_lin_R2'] = m['linear']['R2']
        metrics[f'{prefix}_quad_A'] = m['quadratic']['A']
        metrics[f'{prefix}_quad_A_ci_low'] = m['quadratic']['ci_A'][0]
        metrics[f'{prefix}_quad_A_ci_high'] = m['quadratic']['ci_A'][1]
        metrics[f'{prefix}_quad_C'] = m['quadratic']['C']
        metrics[f'{prefix}_quad_C_ci_low'] = m['quadratic']['ci_C'][0]
        metrics[f'{prefix}_quad_C_ci_high'] = m['quadratic']['ci_C'][1]
        metrics[f'{prefix}_quad_R2'] = m['quadratic']['R2']
        metrics[f'{prefix}_power_beta'] = m['power']['beta']
        metrics[f'{prefix}_power_beta_ci_low'] = m['power']['ci_beta'][0]
        metrics[f'{prefix}_power_beta_ci_high'] = m['power']['ci_beta'][1]
        metrics[f'{prefix}_power_gamma'] = m['power']['gamma']
        metrics[f'{prefix}_power_gamma_ci_low'] = m['power']['ci_gamma'][0]
        metrics[f'{prefix}_power_gamma_ci_high'] = m['power']['ci_gamma'][1]
        metrics[f'{prefix}_power_R2'] = m['power']['R2']
    metrics['global_d_beta'] = beta_global
    metrics['global_d_beta_ci_low'] = ci_global[0][0]
    metrics['global_d_beta_ci_high'] = ci_global[0][1]
    metrics['global_d_gamma'] = gamma_global
    metrics['global_d_gamma_ci_low'] = ci_global[1][0]
    metrics['global_d_gamma_ci_high'] = ci_global[1][1]
    metrics['global_d_R2'] = r2_global
    for eid in zero_check:
        metrics[f'{eid}_a_sg_mean'] = zero_check[eid]['mean']
        metrics[f'{eid}_a_sg_std'] = zero_check[eid]['std']
        metrics[f'{eid}_a_sg_t_test_pvalue'] = zero_check[eid]['p_value']

    # ---------------------------
    # Derived series: return damping d for each constant experiment
    # ---------------------------
    derived_series = []
    for eid in constant_exps:
        values = d_series[eid].tolist()
        derived_series.append({
            'experiment_id': eid,
            'name': 'damping',
            'values': values,
            'source_name': 'a_sg - F_ext',
            'provenance': 'generated data processor: custom_data_analysis',
            'description': 'Damping term d = a_sg - F_ext'
        })

    # ---------------------------
    # Observation summary
    # ---------------------------
    obs_lines = []
    obs_lines.append("对所有恒外力实验（exp_02,03,04,06,07）进行了统一分析。")
    obs_lines.append(f"1) 在同一张散点图绘制了每个实验的 a_sg vs v_sg（已保存）。")
    obs_lines.append("2) 每个实验拟合了三个模型：线性 a=A-B*v，二次 a=A-C*v^2，幂律 a=F_ext - β*v^γ。拟合系数及95% CI 见 metrics。")
    obs_lines.append("3) 计算了阻尼项 d = a_sg - F_ext，绘制了 d vs v_sg 散点图。")
    obs_lines.append(f"4) 合并所有恒外力实验的 d 和 v 数据拟合 d = -β*v^γ，β={beta_global:.4f}, γ={gamma_global:.4f}, R²={r2_global:.4f}。")
    for eid, chk in zero_check.items():
        obs_lines.append(f"5) {eid}: a_sg 均值={chk['mean']:.6f}, 标准差={chk['std']:.6f}, t检验p值={chk['p_value']:.4f}（接近零" + ("成立" if chk['p_value'] > 0.05 else "不成立") + "）。")
    obs_lines.append("详细拟合结果和图像已保存。")

    # Assemble figures list
    figures = [scatter_path, d_scatter_path] + fit_fig_paths

    return {
        'observation': '\n'.join(obs_lines),
        'derived_series': derived_series,
        'figures': figures,
        'metrics': metrics
    }
