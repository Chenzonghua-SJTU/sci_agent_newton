import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = payload['parameters']
    exp_ids = params.get('experiment_ids', ['exp_03', 'exp_04'])
    experiments = payload['experiments']
    output_dir = Path(payload['output_dir'])

    derived_series: List[Dict] = []
    metrics: Dict[str, float] = {}
    figures: List[str] = []
    data: Dict[str, Dict[str, np.ndarray]] = {}

    # Validate experiment_ids
    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload['experiments']")
        exp = experiments[eid]
        if 'q' not in exp['series']:
            raise ValueError(f"Experiment {eid} has no series 'q'")
        if 't' not in exp['series']:
            raise ValueError(f"Experiment {eid} has no series 't'")
        t = np.array(exp['series']['t'])
        q = np.array(exp['series']['q'])
        if len(q) < 15:
            raise ValueError(f"Experiment {eid} has only {len(q)} points, cannot use SG window=15")
        dt = t[1] - t[0]

        # Compute velocity and acceleration using SG filter
        v = savgol_filter(q, window_length=15, polyorder=3, deriv=1, delta=dt)
        a = savgol_filter(q, window_length=15, polyorder=3, deriv=2, delta=dt)

        data[eid] = {
            't': t,
            'v': v,
            'a': a
        }

        derived_series.append({
            'experiment_id': eid,
            'name': 'v_sg_long',
            'values': v.tolist(),
            'source_name': 'Savitzky-Golay filter (window=15, polyorder=3, deriv=1) on q, delta=dt',
            'provenance': 'custom_data_analysis: long window kinematics',
            'description': 'Velocity estimated with longer SG window (15,3)'
        })
        derived_series.append({
            'experiment_id': eid,
            'name': 'a_sg_long',
            'values': a.tolist(),
            'source_name': 'Savitzky-Golay filter (window=15, polyorder=3, deriv=2) on q, delta=dt',
            'provenance': 'custom_data_analysis: long window kinematics',
            'description': 'Acceleration estimated with longer SG window (15,3)'
        })

    # Fit a = alpha - beta * v for each experiment
    fit_results = {}
    for eid in exp_ids:
        v = data[eid]['v']
        a = data[eid]['a']
        coeffs = np.polyfit(v, a, 1)  # [m, c] meaning a = m*v + c
        m, c = coeffs[0], coeffs[1]
        alpha = c
        beta = -m  # because a = c + m*v = alpha - beta*v => alpha=c, beta=-m
        pred = np.polyval(coeffs, v)
        residuals = a - pred
        rmse = np.sqrt(np.mean(residuals**2))
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((a - np.mean(a))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float('nan')

        fit_results[eid] = {
            'alpha': alpha,
            'beta': beta,
            'm': m,
            'c': c,
            'rmse': rmse,
            'r2': r2
        }
        metrics[f'{eid}_alpha'] = alpha
        metrics[f'{eid}_beta'] = beta
        metrics[f'{eid}_rmse'] = rmse
        metrics[f'{eid}_r2'] = r2

    # Compare alpha and beta across experiments
    if len(exp_ids) == 2:
        e1, e2 = exp_ids[0], exp_ids[1]
        alpha_diff = abs(fit_results[e1]['alpha'] - fit_results[e2]['alpha'])
        beta_diff = abs(fit_results[e1]['beta'] - fit_results[e2]['beta'])
        metrics['alpha_diff'] = alpha_diff
        metrics['beta_diff'] = beta_diff
        if fit_results[e2]['beta'] != 0:
            metrics['beta_ratio'] = fit_results[e1]['beta'] / fit_results[e2]['beta']
        else:
            metrics['beta_ratio'] = None

        # Add cross-experiment metrics
        metrics['cross_alpha_diff'] = alpha_diff
        metrics['cross_beta_diff'] = beta_diff

    # Plot a vs v scatter with both experiments and fitted lines
    colors = {'exp_03': 'blue', 'exp_04': 'red'}
    fig, ax = plt.subplots(figsize=(8, 6))
    for eid in exp_ids:
        v = data[eid]['v']
        a = data[eid]['a']
        ax.scatter(v, a, s=10, color=colors[eid], label=eid, alpha=0.7)
        v_sort = np.sort(v)
        a_fit = np.polyval([fit_results[eid]['m'], fit_results[eid]['c']], v_sort)
        ax.plot(v_sort, a_fit, color=colors[eid], linestyle='--',
                label=f'{eid} fit: a={fit_results[eid]["alpha"]:.4f} - {fit_results[eid]["beta"]:.4f}*v')
    ax.set_xlabel('v (m/s)')
    ax.set_ylabel('a (m/s²)')
    ax.set_title('a vs v (SG window=15, polyorder=3)')
    ax.legend()
    fig.tight_layout()
    scatter_path = output_dir / 'a_vs_v_sg_long.png'
    fig.savefig(str(scatter_path), dpi=150)
    plt.close(fig)
    figures.append(str(scatter_path))

    # Check a_sg_long trend over time: plot a vs t and compute linear slope
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for eid in exp_ids:
        t = data[eid]['t']
        a = data[eid]['a']
        ax2.plot(t, a, color=colors[eid], label=eid)
        # linear fit a = slope*t + intercept
        coeffs_a_t = np.polyfit(t, a, 1)
        slope = coeffs_a_t[0]
        intercept = coeffs_a_t[1]
        metrics[f'{eid}_a_vs_t_slope'] = slope
        metrics[f'{eid}_a_vs_t_intercept'] = intercept
        ax2.text(0.05, 0.9 - 0.1 * list(exp_ids).index(eid),
                 f'{eid} slope={slope:.6f}', transform=ax2.transAxes,
                 color=colors[eid], fontsize=9)
    ax2.set_xlabel('t (s)')
    ax2.set_ylabel('a_sg_long (m/s²)')
    ax2.set_title('a_sg_long vs Time')
    ax2.legend()
    fig2.tight_layout()
    trend_path = output_dir / 'a_sg_long_vs_t.png'
    fig2.savefig(str(trend_path), dpi=150)
    plt.close(fig2)
    figures.append(str(trend_path))

    # Build observation text
    lines = []
    for eid in exp_ids:
        v = data[eid]['v']
        a = data[eid]['a']
        lines.append(
            f"实验 {eid}: v_sg_long min={v.min():.6f}, max={v.max():.6f}, mean={v.mean():.6f}; "
            f"a_sg_long min={a.min():.6f}, max={a.max():.6f}, mean={a.mean():.6f}."
        )
        fr = fit_results[eid]
        lines.append(
            f"  拟合 a = alpha - beta*v: alpha={fr['alpha']:.6f}, beta={fr['beta']:.6f}, "
            f"R²={fr['r2']:.6f}, RMSE={fr['rmse']:.6e}."
        )
    if len(exp_ids) == 2:
        lines.append(f"跨实验比较: alpha差异={alpha_diff:.6f}, beta差异={beta_diff:.6f}.")
    lines.append("a_sg_long随时间变化的线性斜率已记录在metrics中。")
    observation = "使用SG窗口15、多项式3从q重新估计速度和加速度，得到v_sg_long和a_sg_long。" \
                  "对每个实验拟合线性模型a=alpha-beta*v。" + "\n".join(lines)

    return {
        'observation': observation,
        'derived_series': derived_series,
        'figures': figures,
        'metrics': metrics
    }
