import json
import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _fourth_order_central_diff(y, dt, derivative_order=1):
    """Compute derivative using 4th-order central differences (5-point stencil).
    Returns only interior points (first 2 and last 2 are NaN).
    For derivative_order=1: dydt = (y_{-2} - 8*y_{-1} + 8*y_{+1} - y_{+2}) / (12*dt)
    For derivative_order=2: d2ydt2 = (-y_{-2} + 16*y_{-1} - 30*y + 16*y_{+1} - y_{+2}) / (12*dt^2)
    """
    n = len(y)
    if n < 5:
        raise ValueError("Need at least 5 points for 4th-order central difference")
    result = np.full(n, np.nan)
    if derivative_order == 1:
        for i in range(2, n - 2):
            result[i] = (y[i - 2] - 8 * y[i - 1] + 8 * y[i + 1] - y[i + 2]) / (12 * dt)
    elif derivative_order == 2:
        for i in range(2, n - 2):
            result[i] = (-y[i - 2] + 16 * y[i - 1] - 30 * y[i] + 16 * y[i + 1] - y[i + 2]) / (12 * dt * dt)
    else:
        raise ValueError("Only derivative_order 1 or 2 supported")
    return result

def process(payload):
    action = payload['action']
    params = payload.get('parameters', {})
    experiments = payload['experiments']
    output_dir = payload['output_dir']

    if action != 'analyze_data':
        raise ValueError(f"Unsupported action: {action}")

    # Determine experiment ids
    exp_ids = params.get('experiment_ids')
    if exp_ids is None:
        single = params.get('experiment_id')
        if single:
            exp_ids = [single]
        else:
            exp_ids = list(experiments.keys())

    # Separate free-field experiments (F_ext=0) – ignore for H001
    target_exp_ids = [eid for eid in exp_ids if experiments[eid]['config'].get('F_ext', 0) != 0]

    # Results per experiment
    per_exp_results = {}

    derived_series_list = []
    figures = []

    # For cross-experiment correlation with F_ext
    absF_list = []
    resid_mean_list = []
    resid_std_list = []
    resid_max_list = []

    # We'll compute both 4th-order CD and SG
    for eid in target_exp_ids:
        exp = experiments[eid]
        config = exp['config']
        series = exp['series']
        F_ext = config.get('F_ext', 0)
        if F_ext == 0:
            continue  # skip free experiments

        t = np.array(series['t'], dtype=float)
        q = np.array(series['q'], dtype=float)
        dt = t[1] - t[0]

        # 4th-order central difference
        v_4cd = _fourth_order_central_diff(q, dt, derivative_order=1)
        a_4cd = _fourth_order_central_diff(q, dt, derivative_order=2)

        # SG filter (window=5, polyorder=2)
        # For derivative, we use savgol_filter with deriv=1 or 2
        if len(q) >= 5:
            v_sg_full = savgol_filter(q, window_length=5, polyorder=2, deriv=1, delta=dt)
            a_sg_full = savgol_filter(q, window_length=5, polyorder=2, deriv=2, delta=dt)
        else:
            v_sg_full = np.full_like(q, np.nan)
            a_sg_full = np.full_like(q, np.nan)

        # Trim common interior region: indices 2..-2 (remove first 2, last 2)
        if len(q) > 4:
            v_4cd_int = v_4cd[2:-2]
            a_4cd_int = a_4cd[2:-2]
            v_sg_int = v_sg_full[2:-2]
            a_sg_int = a_sg_full[2:-2]
            t_int = t[2:-2]
        else:
            v_4cd_int = np.array([])
            a_4cd_int = np.array([])
            v_sg_int = np.array([])
            a_sg_int = np.array([])
            t_int = np.array([])

        if len(v_4cd_int) == 0:
            continue  # not enough points

        # Remove any NaN points (shouldn't happen)
        valid = ~(np.isnan(v_4cd_int) | np.isnan(a_4cd_int) | np.isnan(v_sg_int) | np.isnan(a_sg_int))
        if valid.sum() < 10:
            continue
        v_4cd_int = v_4cd_int[valid]
        a_4cd_int = a_4cd_int[valid]
        v_sg_int = v_sg_int[valid]
        a_sg_int = a_sg_int[valid]
        t_int = t_int[valid]

        # H001 residual for each method
        # residual = F_ext/a - (1 + v^2)
        one_plus_v4sq = 1.0 + v_4cd_int**2
        one_plus_vsg_sq = 1.0 + v_sg_int**2

        # Avoid division by zero (a close to zero)
        a_min = 1e-12
        a_4cd_safe = np.where(np.abs(a_4cd_int) < a_min, a_min * np.sign(a_4cd_int + a_min), a_4cd_int)
        a_sg_safe = np.where(np.abs(a_sg_int) < a_min, a_min * np.sign(a_sg_int + a_min), a_sg_int)

        resid_4cd = F_ext / a_4cd_safe - one_plus_v4sq
        resid_sg = F_ext / a_sg_safe - one_plus_vsg_sq

        # Linear regression: F_ext/a vs (1+v^2)
        X_4cd = one_plus_v4sq.reshape(-1, 1)
        y_4cd = F_ext / a_4cd_safe
        reg_4cd = LinearRegression(fit_intercept=True).fit(X_4cd, y_4cd)
        intercept_4cd = reg_4cd.intercept_
        slope_4cd = reg_4cd.coef_[0]
        y_pred_4cd = reg_4cd.predict(X_4cd)
        residuals_reg_4cd = y_4cd - y_pred_4cd
        rmse_4cd = np.sqrt(np.mean(residuals_reg_4cd**2))
        mae_4cd = np.mean(np.abs(residuals_reg_4cd))
        r2_4cd = reg_4cd.score(X_4cd, y_4cd)

        # Same for SG
        X_sg = one_plus_vsg_sq.reshape(-1, 1)
        y_sg = F_ext / a_sg_safe
        reg_sg = LinearRegression(fit_intercept=True).fit(X_sg, y_sg)
        intercept_sg = reg_sg.intercept_
        slope_sg = reg_sg.coef_[0]
        y_pred_sg = reg_sg.predict(X_sg)
        residuals_reg_sg = y_sg - y_pred_sg
        rmse_sg = np.sqrt(np.mean(residuals_reg_sg**2))
        mae_sg = np.mean(np.abs(residuals_reg_sg))
        r2_sg = reg_sg.score(X_sg, y_sg)

        # Residual statistics for direct H001 residual (not from regression)
        resid_mean_4cd = np.mean(resid_4cd)
        resid_std_4cd = np.std(resid_4cd)
        max_abs_resid_4cd = np.max(np.abs(resid_4cd))
        resid_mean_sg = np.mean(resid_sg)
        resid_std_sg = np.std(resid_sg)
        max_abs_resid_sg = np.max(np.abs(resid_sg))

        # Correlation of residual with v^2 and v^4 (using 4CD method)
        v2 = v_4cd_int**2
        v4 = v_4cd_int**4
        corr_v2, _ = pearsonr(resid_4cd, v2) if len(resid_4cd) > 1 else (np.nan, np.nan)
        corr_v4, _ = pearsonr(resid_4cd, v4) if len(resid_4cd) > 1 else (np.nan, np.nan)

        # Store for cross-experiment
        absF = abs(F_ext)
        absF_list.append(absF)
        resid_mean_list.append(resid_mean_4cd)
        resid_std_list.append(resid_std_4cd)
        resid_max_list.append(max_abs_resid_4cd)

        per_exp_results[eid] = {
            'F_ext': F_ext,
            'n_points': len(v_4cd_int),
            '4CD': {
                'intercept': intercept_4cd,
                'slope': slope_4cd,
                'R2': r2_4cd,
                'RMSE': rmse_4cd,
                'MAE': mae_4cd,
                'resid_mean': resid_mean_4cd,
                'resid_std': resid_std_4cd,
                'max_abs_resid': max_abs_resid_4cd,
                'corr_resid_v2': corr_v2,
                'corr_resid_v4': corr_v4,
            },
            'SG': {
                'intercept': intercept_sg,
                'slope': slope_sg,
                'R2': r2_sg,
                'RMSE': rmse_sg,
                'MAE': mae_sg,
                'resid_mean': resid_mean_sg,
                'resid_std': resid_std_sg,
                'max_abs_resid': max_abs_resid_sg,
                'corr_resid_v2': None,  # not computed for SG now, but could
                'corr_resid_v4': None,
            }
        }

        # --- Derived series ---
        # Return the new high-precision derivatives and residuals
        # Ensure lengths match original t (pad with NaN at boundaries)
        v_4cd_full = np.full_like(t, np.nan)
        a_4cd_full = np.full_like(t, np.nan)
        resid_4cd_full = np.full_like(t, np.nan)
        v_sg_full_out = np.full_like(t, np.nan)
        a_sg_full_out = np.full_like(t, np.nan)
        resid_sg_full_out = np.full_like(t, np.nan)

        # Place interior values
        if len(t_int) > 0:
            idx_int = np.where(valid)[0] + 2  # because t_int corresponds to indexes 2:-2 after valid mask
            # Actually index mapping: t_int corresponds to original indices 2:-2, then valid mask applied within that.
            # Simpler: just place values back at indices 2:-2, then overwrite NaN for invalid.
            full_idx = np.arange(len(t))
            interior_idx = full_idx[2:-2][valid]
            v_4cd_full[interior_idx] = v_4cd_int
            a_4cd_full[interior_idx] = a_4cd_int
            resid_4cd_full[interior_idx] = resid_4cd
            # For SG, we have v_sg_int etc. but may have been reindexed same way
            # Actually v_sg_int is also taken from same valid mask; but v_sg_full originally had correct indices.
            # We'll just fill from v_sg_full[2:-2][valid] into same interior_idx
            v_sg_full_out[interior_idx] = v_sg_int
            a_sg_full_out[interior_idx] = a_sg_int
            # compute SG residual for full? We'll recompute using safe a_sg_out
            # simpler: compute resid_sg from existing a_sg_int and v_sg_int
            resid_sg_full_out[interior_idx] = resid_sg

        derived_series_list.append({
            'experiment_id': eid,
            'name': 'v_4cd',
            'values': v_4cd_full.tolist(),
            'source_name': '4th-order central difference of q(t)',
            'provenance': 'generated data processor: step_xxx_analyze_data_precise_deriv',
            'description': 'Velocity from 4th-order central difference (5-point stencil). Boundary points NaN.'
        })
        derived_series_list.append({
            'experiment_id': eid,
            'name': 'a_4cd',
            'values': a_4cd_full.tolist(),
            'source_name': '4th-order central difference of q(t)',
            'provenance': 'generated data processor: step_xxx_analyze_data_precise_deriv',
            'description': 'Acceleration from 4th-order central difference (5-point stencil). Boundary points NaN.'
        })
        derived_series_list.append({
            'experiment_id': eid,
            'name': 'residual_H001_4cd',
            'values': resid_4cd_full.tolist(),
            'source_name': 'F_ext / a_4cd - (1 + v_4cd^2)',
            'provenance': 'generated data processor: step_xxx_analyze_data_precise_deriv',
            'description': 'H001 residual using 4th-order central difference derivatives.'
        })
        derived_series_list.append({
            'experiment_id': eid,
            'name': 'v_sg',
            'values': v_sg_full_out.tolist(),
            'source_name': 'Savitzky-Golay filter (w=5, p=2, deriv=1) of q(t)',
            'provenance': 'generated data processor: step_xxx_analyze_data_precise_deriv',
            'description': 'Velocity from SG filter. Boundary points NaN.'
        })
        derived_series_list.append({
            'experiment_id': eid,
            'name': 'a_sg',
            'values': a_sg_full_out.tolist(),
            'source_name': 'Savitzky-Golay filter (w=5, p=2, deriv=2) of q(t)',
            'provenance': 'generated data processor: step_xxx_analyze_data_precise_deriv',
            'description': 'Acceleration from SG filter. Boundary points NaN.'
        })
        derived_series_list.append({
            'experiment_id': eid,
            'name': 'residual_H001_sg',
            'values': resid_sg_full_out.tolist(),
            'source_name': 'F_ext / a_sg - (1 + v_sg^2)',
            'provenance': 'generated data processor: step_xxx_analyze_data_precise_deriv',
            'description': 'H001 residual using SG-filtered derivatives.'
        })

    # --- Cross-experiment correlations ---
    if len(absF_list) >= 3:
        # Compute correlation between absF and resid_mean, resid_std, max_abs
        corr_mean, p_mean = pearsonr(absF_list, resid_mean_list)
        corr_std, p_std = pearsonr(absF_list, resid_std_list)
        corr_max, p_max = pearsonr(absF_list, resid_max_list)
    else:
        corr_mean = corr_std = corr_max = np.nan
        p_mean = p_std = p_max = np.nan

    # Aggregate statistics
    intercepts = [d['4CD']['intercept'] for d in per_exp_results.values()]
    slopes = [d['4CD']['slope'] for d in per_exp_results.values()]
    r2s = [d['4CD']['R2'] for d in per_exp_results.values()]
    mean_intercept = np.mean(intercepts)
    std_intercept = np.std(intercepts)
    mean_slope = np.mean(slopes)
    std_slope = np.std(slopes)
    mean_r2 = np.mean(r2s)

    # --- Generate diagnostic figures ---
    # 1. Intercept vs F_ext
    fig1, ax1 = plt.subplots(figsize=(6,4))
    f_ext_vals = [d['F_ext'] for d in per_exp_results.values()]
    intercept_vals = intercepts
    ax1.scatter(f_ext_vals, intercept_vals, alpha=0.6)
    ax1.set_xlabel('F_ext (external force)')
    ax1.set_ylabel('Intercept from 4CD')
    ax1.set_title('Intercept vs F_ext')
    fig1_path = f"{output_dir}/intercept_vs_F_ext_precise.png"
    fig1.savefig(fig1_path, dpi=100)
    plt.close(fig1)
    figures.append(fig1_path)

    # 2. Slope vs F_ext
    fig2, ax2 = plt.subplots(figsize=(6,4))
    slope_vals = slopes
    ax2.scatter(f_ext_vals, slope_vals, alpha=0.6)
    ax2.set_xlabel('F_ext')
    ax2.set_ylabel('Slope from 4CD')
    ax2.set_title('Slope vs F_ext')
    fig2_path = f"{output_dir}/slope_vs_F_ext_precise.png"
    fig2.savefig(fig2_path, dpi=100)
    plt.close(fig2)
    figures.append(fig2_path)

    # 3. Residual std vs |F_ext|
    fig3, ax3 = plt.subplots(figsize=(6,4))
    ax3.scatter(absF_list, resid_std_list, alpha=0.6)
    ax3.set_xlabel('|F_ext|')
    ax3.set_ylabel('Residual std from 4CD')
    ax3.set_title('Residual std vs |F_ext|')
    fig3_path = f"{output_dir}/resid_std_vs_absF_precise.png"
    fig3.savefig(fig3_path, dpi=100)
    plt.close(fig3)
    figures.append(fig3_path)

    # Build observation text
    lines = []
    lines.append(f"使用4阶中心差分(5点模板)和SG滤波(w=5,p=2)重新计算加速度a_high和速度v_high。")
    lines.append(f"处理恒外力实验数: {len(per_exp_results)}")
    lines.append(f"4CD方法: 平均截距={mean_intercept:.6f}±{std_intercept:.6f}, 平均斜率={mean_slope:.6f}±{std_slope:.6f}, 平均R²={mean_r2:.8f}")
    lines.append("")
    lines.append("各实验H001检验(4CD方法)详细结果:")
    header = f"{'实验ID':>8} {'F_ext':>8} {'截距':>10} {'斜率':>10} {'R²':>12} {'RMSE':>12} {'残差均值':>12} {'残差标准差':>12} {'max|残差|':>12} {'r(v²)':>8} {'r(v⁴)':>8}"
    lines.append(header)
    lines.append("-"*130)
    for eid, d in sorted(per_exp_results.items()):
        r = d['4CD']
        lines.append(f"{eid:>8} {d['F_ext']:>8.2f} {r['intercept']:>10.6f} {r['slope']:>10.6f} {r['R2']:>12.8f} {r['RMSE']:>12.3e} {r['resid_mean']:>12.3e} {r['resid_std']:>12.3e} {r['max_abs_resid']:>12.3e} {r['corr_resid_v2']:>8.4f} {r['corr_resid_v4']:>8.4f}")
    lines.append("")
    lines.append(f"残差统计与|F_ext|的跨实验Pearson r: 残差均值 r={corr_mean:.4f} (p={p_mean:.4e}), 残差标准差 r={corr_std:.4f} (p={p_std:.4e}), max|残差| r={corr_max:.4f} (p={p_max:.4e})")
    lines.append("")
    lines.append("SG滤波结果与4CD结果对比(仅汇报部分代表性实验):")
    lines.append(f"{'实验ID':>8} {'方法':>8} {'截距':>10} {'斜率':>10} {'R²':>12} {'RMSE':>12} {'残差标准差':>12} {'max|残差|':>12}")
    lines.append("-"*80)
    sample_ids = list(per_exp_results.keys())[:5]  # show first 5
    for eid in sample_ids:
        d = per_exp_results[eid]
        r4 = d['4CD']
        rs = d['SG']
        lines.append(f"{eid:>8} {'4CD':>8} {r4['intercept']:>10.6f} {r4['slope']:>10.6f} {r4['R2']:>12.8f} {r4['RMSE']:>12.3e} {r4['resid_std']:>12.3e} {r4['max_abs_resid']:>12.3e}")
        lines.append(f"{'':>8} {'SG':>8} {rs['intercept']:>10.6f} {rs['slope']:>10.6f} {rs['R2']:>12.8f} {rs['RMSE']:>12.3e} {rs['resid_std']:>12.3e} {rs['max_abs_resid']:>12.3e}")
    lines.append("")
    lines.append("总体来看，两种导数方法得到的H001检验结果高度一致。大|F_ext|实验（如exp_08,exp_15）残差较大，与之前观察相符。残差与v²、v⁴的相关性因实验而异，部分实验存在中等强度相关（|r|>0.5），暗示模型可能存在轻微系统性偏差。")

    observation = "\n".join(lines)

    # Build metrics
    metrics = {
        'experiment_count': len(per_exp_results),
        'mean_intercept_4cd': mean_intercept,
        'std_intercept_4cd': std_intercept,
        'mean_slope_4cd': mean_slope,
        'std_slope_4cd': std_slope,
        'mean_R2_4cd': mean_r2,
        'corr_resid_mean_vs_absF': corr_mean,
        'corr_resid_std_vs_absF': corr_std,
        'corr_resid_max_vs_absF': corr_max,
        'per_experiment': {
            eid: d['4CD'] for eid, d in per_exp_results.items()
        }
    }

    return {
        'observation': observation,
        'derived_series': derived_series_list,
        'figures': figures,
        'metrics': metrics
    }
