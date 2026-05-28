import json, math, statistics, itertools, functools, collections, pathlib, typing
import numpy as np
import pandas as pd
import scipy.signal
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    # --- parameter extraction -------------------------------------------------
    params = payload.get('parameters', {})
    exp_ids = params.get('experiment_ids', [])
    if not exp_ids:
        exp_ids = list(payload['experiments'].keys())
    if 'exp_01' not in exp_ids:
        raise ValueError("This analysis requires experiment 'exp_01', but it is not in experiment_ids.")
    
    exps = payload['experiments']
    exp = exps['exp_01']
    series = exp['series']
    
    t = np.array(series['t'], dtype=float)
    q = np.array(series['q'], dtype=float)
    dt = t[1] - t[0]
    
    # --- 1. linear fit q vs t -------------------------------------------------
    coeffs_linear = np.polyfit(t, q, 1)
    slope = coeffs_linear[0]
    intercept = coeffs_linear[1]
    q_linear_fit = np.polyval(coeffs_linear, t)
    residuals_linear = q - q_linear_fit
    ss_res = np.sum(residuals_linear ** 2)
    ss_tot = np.sum((q - np.mean(q)) ** 2)
    r2_linear = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # --- 2. estimate kinematics (savgol) --------------------------------------
    window_length = min(7, len(t) // 2 * 2 + 1)   # ensure odd
    polyorder = 2
    q_smooth = scipy.signal.savgol_filter(q, window_length, polyorder, deriv=0)
    v_est = scipy.signal.savgol_filter(q, window_length, polyorder, deriv=1, delta=dt)
    a_est = scipy.signal.savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)
    
    # --- 3. statistics --------------------------------------------------------
    v_mean = float(np.mean(v_est))
    v_std = float(np.std(v_est))
    v_min = float(np.min(v_est))
    v_max = float(np.max(v_est))
    a_mean = float(np.mean(a_est))
    a_std = float(np.std(a_est))
    a_min = float(np.min(a_est))
    a_max = float(np.max(a_est))
    
    # --- 4. further fitting if trend or velocity non-zero --------------------
    slope_nonzero = abs(slope) > 1e-8
    velocity_nonzero = abs(v_mean) > 1e-8
    fitted_extra = {}
    if slope_nonzero or velocity_nonzero:
        coeffs_quad = np.polyfit(t, q, 2)
        a2, b2, c2 = coeffs_quad[0], coeffs_quad[1], coeffs_quad[2]
        q_quad_fit = np.polyval(coeffs_quad, t)
        residuals_quad = q - q_quad_fit
        ss_res_quad = np.sum(residuals_quad ** 2)
        r2_quad = 1.0 - ss_res_quad / ss_tot if ss_tot > 0 else 0.0
        fitted_extra = {
            'quadratic_coeff_a': float(a2),
            'quadratic_coeff_b': float(b2),
            'quadratic_coeff_c': float(c2),
            'quadratic_R2': float(r2_quad)
        }
    
    # --- 5. figures -----------------------------------------------------------
    output_dir = payload['output_dir']
    fig_paths = []
    
    # fig1: q vs t scatter + linear fit
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.scatter(t, q, s=20, c='blue', label='raw q')
    ax1.plot(t, q_linear_fit, 'r-', label=f'linear (slope={float(slope):.6f})')
    ax1.set_xlabel('t')
    ax1.set_ylabel('q')
    ax1.set_title('Position q vs Time with Linear Fit')
    ax1.legend()
    fig1.tight_layout()
    p1 = pathlib.Path(output_dir) / 'exp01_q_vs_t.png'
    fig1.savefig(str(p1))
    plt.close(fig1)
    fig_paths.append(str(p1))
    
    # fig2: velocity vs t
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(t, v_est, 'g-', label='v (savgol deriv=1)')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('t')
    ax2.set_ylabel('v')
    ax2.set_title('Estimated Velocity vs Time')
    ax2.legend()
    fig2.tight_layout()
    p2 = pathlib.Path(output_dir) / 'exp01_v_vs_t.png'
    fig2.savefig(str(p2))
    plt.close(fig2)
    fig_paths.append(str(p2))
    
    # fig3: acceleration vs t
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(t, a_est, 'm-', label='a (savgol deriv=2)')
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax3.set_xlabel('t')
    ax3.set_ylabel('a')
    ax3.set_title('Estimated Acceleration vs Time')
    ax3.legend()
    fig3.tight_layout()
    p3 = pathlib.Path(output_dir) / 'exp01_a_vs_t.png'
    fig3.savefig(str(p3))
    plt.close(fig3)
    fig_paths.append(str(p3))
    
    # --- 6. derived series ----------------------------------------------------
    derived = [
        {
            'experiment_id': 'exp_01',
            'name': 'q_smooth',
            'values': q_smooth.tolist(),
            'source_name': 'savgol_filter smoothed q',
            'provenance': 'generated data processor: custom_data_analysis',
            'description': 'Smoothed position using Savitzky–Golay filter (window=%d, polyorder=2)' % window_length
        },
        {
            'experiment_id': 'exp_01',
            'name': 'v_est',
            'values': v_est.tolist(),
            'source_name': 'savgol_filter derivative order 1',
            'provenance': 'generated data processor: custom_data_analysis',
            'description': 'Estimated velocity via Savitzky–Golay filter (deriv=1)'
        },
        {
            'experiment_id': 'exp_01',
            'name': 'a_est',
            'values': a_est.tolist(),
            'source_name': 'savgol_filter derivative order 2',
            'provenance': 'generated data processor: custom_data_analysis',
            'description': 'Estimated acceleration via Savitzky–Golay filter (deriv=2)'
        },
        {
            'experiment_id': 'exp_01',
            'name': 'q_linear_fit_residual',
            'values': residuals_linear.tolist(),
            'source_name': 'q - (slope*t + intercept)',
            'provenance': 'generated data processor: custom_data_analysis',
            'description': 'Residual after linear fit q(t) = slope*t + intercept'
        }
    ]
    if fitted_extra:
        derived.append({
            'experiment_id': 'exp_01',
            'name': 'q_quad_fit_residual',
            'values': residuals_quad.tolist(),
            'source_name': 'q - (a*t^2 + b*t + c)',
            'provenance': 'generated data processor: custom_data_analysis',
            'description': 'Residual after quadratic fit q(t) = a*t^2 + b*t + c'
        })
    
    # --- 7. metrics -----------------------------------------------------------
    metrics = {
        'linear_slope': float(slope),
        'linear_intercept': float(intercept),
        'linear_R2': float(r2_linear),
        'v_mean': v_mean,
        'v_std': v_std,
        'v_min': v_min,
        'v_max': v_max,
        'a_mean': a_mean,
        'a_std': a_std,
        'a_min': a_min,
        'a_max': a_max
    }
    metrics.update(fitted_extra)
    
    # --- 8. observation -------------------------------------------------------
    obs_parts = [
        f"对实验 exp_01 进行了 q(t) 的线性拟合：斜率={float(slope):.6f}，截距={float(intercept):.6f}，R²={float(r2_linear):.6f}。",
        f"使用 Savitzky-Golay 滤波（window={window_length}, polyorder=2）估计速度和加速度。",
        f"速度统计：均值={v_mean:.6f}，标准差={v_std:.6f}，范围=[{v_min:.6f}, {v_max:.6f}]。",
        f"加速度统计：均值={a_mean:.6f}，标准差={a_std:.6f}，范围=[{a_min:.6f}, {a_max:.6f}]。"
    ]
    if fitted_extra:
        a2_val = fitted_extra['quadratic_coeff_a']
        b2_val = fitted_extra['quadratic_coeff_b']
        c2_val = fitted_extra['quadratic_coeff_c']
        obs_parts.append(
            f"由于线性趋势非零或速度非零，进一步拟合了二次模型：q = {a2_val:.6f}*t^2 + {b2_val:.6f}*t + {c2_val:.6f}，R²={fitted_extra['quadratic_R2']:.6f}。"
        )
    else:
        obs_parts.append("线性趋势和速度均值接近零，未进行进一步拟合。")
    obs_parts.append("已生成 q-t 散点图（含线性拟合）、速度-t 图和加速度-t 图。")
    observation = " ".join(obs_parts)
    
    return {
        'observation': observation,
        'derived_series': derived,
        'figures': fig_paths,
        'metrics': metrics
    }
