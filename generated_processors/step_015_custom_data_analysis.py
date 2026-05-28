import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import linregress, t
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def _compute_ci_linregress(x: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    """Return (intercept, slope, intercept_ci_low, intercept_ci_high,
               slope_ci_low, slope_ci_high, r2)."""
    n = len(x)
    res = linregress(x, y)
    intercept = res.intercept
    slope = res.slope
    se_intercept = res.intercept_stderr
    se_slope = res.stderr
    dof = n - 2
    t_crit = t.ppf(1 - alpha / 2, dof)
    return (intercept, slope,
            intercept - t_crit * se_intercept, intercept + t_crit * se_intercept,
            slope - t_crit * se_slope, slope + t_crit * se_slope,
            res.rvalue ** 2)


def _fit_with_ci(X: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    """Fit linear model with intercept, return (coeff vector, intercept,
       coeff_ci_low, coeff_ci_high, intercept_ci_low, intercept_ci_high, R2).
       X is (n, p) design matrix without intercept column."""
    n, p = X.shape
    # add intercept
    X_design = np.column_stack([np.ones(n), X])
    try:
        beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        raise ValueError("Linear fit failed: cannot solve least squares.")
    y_pred = X_design @ beta
    residuals = y - y_pred
    mse = np.sum(residuals ** 2) / (n - p - 1)
    var_beta = mse * np.linalg.inv(X_design.T @ X_design)
    se_beta = np.sqrt(np.diag(var_beta))
    t_crit = t.ppf(1 - alpha / 2, n - p - 1)
    ci_low = beta - t_crit * se_beta
    ci_high = beta + t_crit * se_beta
    r2 = r2_score(y, y_pred)
    # beta[0] is intercept, beta[1:] are coefficients for columns in X
    intercept = beta[0]
    coeffs = beta[1:]
    intercept_ci_low = ci_low[0]
    intercept_ci_high = ci_high[0]
    coeff_ci_low = ci_low[1:]
    coeff_ci_high = ci_high[1:]
    return coeffs, intercept, coeff_ci_low, coeff_ci_high, intercept_ci_low, intercept_ci_high, r2


def process(payload: dict) -> dict:
    # ------------------------------------------------------------------
    # Extract parameters
    parameters = payload['parameters']
    experiment_ids = parameters['experiment_ids']
    output_dir = Path(payload['output_dir'])
    experiments = payload['experiments']

    derived_series = []
    figures = []
    metrics = {}

    # ------------------------------------------------------------------
    # Helper: ensure v_sg and a_sg exist for each experiment
    def ensure_kinematics(eid: str):
        exp = experiments[eid]
        series = exp['series']
        config = exp['config']
        dt = config.get('dt', 0.1)
        available = exp.get('available_series', list(series.keys()))

        if 'v_sg' in available and 'a_sg' in available:
            v_sg = np.array(series['v_sg'])
            a_sg = np.array(series['a_sg'])
        else:
            # Compute from q(t) using Savitzky-Golay
            if 'q' not in series:
                raise ValueError(f"Experiment {eid} has no 'q' series.")
            q = np.array(series['q'])
            t_arr = np.array(series['t'])
            window = min(11, len(q) // 2 * 2 + 1)  # ensure odd and smaller than length
            if window < 5:
                window = max(3, len(q) // 2 * 2 + 1)
                if window < 3:
                    window = len(q)
            polyorder = min(3, window - 1)
            try:
                v_sg = savgol_filter(q, window_length=window, polyorder=polyorder, deriv=1, delta=dt, mode='interp')
                a_sg = savgol_filter(q, window_length=window, polyorder=polyorder, deriv=2, delta=dt, mode='interp')
            except Exception as e:
                raise ValueError(f"Savitzky-Golay failed for {eid}: {e}")
            # Register as derived series
            derived_series.append({
                'experiment_id': eid,
                'name': 'v_sg',
                'values': v_sg.tolist(),
                'source_name': f'Savitzky-Golay (window={window}, polyorder={polyorder}) from q',
                'provenance': 'generated data processor: custom_data_analysis',
                'description': 'Smooth velocity estimated from q'
            })
            derived_series.append({
                'experiment_id': eid,
                'name': 'a_sg',
                'values': a_sg.tolist(),
                'source_name': f'Savitzky-Golay (window={window}, polyorder={polyorder}) from q',
                'provenance': 'generated data processor: custom_data_analysis',
                'description': 'Smooth acceleration estimated from q'
            })
        # Store in experiment temporary attribute
        exp['_v_sg'] = v_sg
        exp['_a_sg'] = a_sg
        exp['_t'] = np.array(series['t'])
        F_ext = config.get('F_ext', 0.0)
        exp['_F_ext'] = F_ext
        return v_sg, a_sg

    # ------------------------------------------------------------------
    # 1) Plot a_sg vs v_sg for each experiment
    for eid in experiment_ids:
        ensure_kinematics(eid)
        exp = experiments[eid]
        v = exp['_v_sg']
        a = exp['_a_sg']
        F_ext = exp['_F_ext']
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(v, a, s=5, alpha=0.6, c='steelblue')
        ax.set_xlabel('v_sg')
        ax.set_ylabel('a_sg')
        ax.set_title(f'{eid} (F_ext={F_ext})')
        ax.grid(True, alpha=0.3)
        fname = f'{eid}_a_vs_v_scatter.png'
        fig.savefig(str(output_dir / fname), dpi=100)
        plt.close(fig)
        figures.append(str(output_dir / fname))

    # ------------------------------------------------------------------
    # Identify constant-force experiments (F_ext > 0)
    constant_force_ids = [eid for eid in experiment_ids if experiments[eid]['_F_ext'] > 0]
    free_ids = [eid for eid in experiment_ids if experiments[eid]['_F_ext'] == 0]

    # ------------------------------------------------------------------
    # 2) For each constant-force experiment: linear and quadratic fits
    for eid in constant_force_ids:
        exp = experiments[eid]
        v = exp['_v_sg']
        a = exp['_a_sg']
        F_ext = exp['_F_ext']

        # Linear: a = α + β*v
        alpha_l, beta_l, alpha_l_lo, alpha_l_hi, beta_l_lo, beta_l_hi, r2_l = \
            _compute_ci_linregress(v, a, alpha=0.05)

        # Quadratic: a = α + γ*v^2
        v2 = v ** 2
        alpha_q, gamma_q, alpha_q_lo, alpha_q_hi, gamma_q_lo, gamma_q_hi, r2_q = \
            _compute_ci_linregress(v2, a, alpha=0.05)

        # Store in metrics
        prefix = f'{eid}'
        metrics[f'{prefix}_lin_alpha'] = alpha_l
        metrics[f'{prefix}_lin_alpha_ci_low'] = alpha_l_lo
        metrics[f'{prefix}_lin_alpha_ci_high'] = alpha_l_hi
        metrics[f'{prefix}_lin_beta'] = beta_l
        metrics[f'{prefix}_lin_beta_ci_low'] = beta_l_lo
        metrics[f'{prefix}_lin_beta_ci_high'] = beta_l_hi
        metrics[f'{prefix}_lin_R2'] = r2_l
        metrics[f'{prefix}_quad_alpha'] = alpha_q
        metrics[f'{prefix}_quad_alpha_ci_low'] = alpha_q_lo
        metrics[f'{prefix}_quad_alpha_ci_high'] = alpha_q_hi
        metrics[f'{prefix}_quad_gamma'] = gamma_q
        metrics[f'{prefix}_quad_gamma_ci_low'] = gamma_q_lo
        metrics[f'{prefix}_quad_gamma_ci_high'] = gamma_q_hi
        metrics[f'{prefix}_quad_R2'] = r2_q
        metrics[f'{prefix}_alpha_F_diff_linear'] = alpha_l - F_ext
        metrics[f'{prefix}_alpha_F_diff_quadratic'] = alpha_q - F_ext

        # Also plot with fits
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(v, a, s=5, alpha=0.6, label='data')
        v_sorted = np.sort(v)
        ax.plot(v_sorted, alpha_l + beta_l * v_sorted, 'r-', label='linear fit')
        ax.plot(v_sorted, alpha_q + gamma_q * v_sorted**2, 'g--', label='quadratic fit')
        ax.set_xlabel('v_sg')
        ax.set_ylabel('a_sg')
        ax.set_title(f'{eid} fits (F_ext={F_ext})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fname = f'{eid}_a_vs_v_fits.png'
        fig.savefig(str(output_dir / fname), dpi=100)
        plt.close(fig)
        figures.append(str(output_dir / fname))

    # ------------------------------------------------------------------
    # 3) For constant-force experiments: split by time (t<2 and t>5) and linear fit
    # Only if enough points in each segment
    for eid in constant_force_ids:
        exp = experiments[eid]
        t_arr = exp['_t']
        v = exp['_v_sg']
        a = exp['_a_sg']
        F_ext = exp['_F_ext']

        # segment t<2
        mask_early = t_arr < 2.0
        # segment t>5
        mask_late = t_arr > 5.0

        pts_early = np.sum(mask_early)
        pts_late = np.sum(mask_late)

        prefix = f'{eid}_split'

        if pts_early >= 3:
            v_early = v[mask_early]
            a_early = a[mask_early]
            alpha_e, beta_e, _, _, _, _, r2_e = _compute_ci_linregress(v_early, a_early, alpha=0.05)
            metrics[f'{prefix}_early_alpha'] = alpha_e
            metrics[f'{prefix}_early_beta'] = beta_e
            metrics[f'{prefix}_early_R2'] = r2_e
            metrics[f'{prefix}_early_alpha_diff_Fext'] = alpha_e - F_ext
        else:
            metrics[f'{prefix}_early_alpha'] = None
            metrics[f'{prefix}_early_alpha_diff_Fext'] = None

        if pts_late >= 3:
            v_late = v[mask_late]
            a_late = a[mask_late]
            alpha_l, beta_l, _, _, _, _, r2_l = _compute_ci_linregress(v_late, a_late, alpha=0.05)
            metrics[f'{prefix}_late_alpha'] = alpha_l
            metrics[f'{prefix}_late_beta'] = beta_l
            metrics[f'{prefix}_late_R2'] = r2_l
            metrics[f'{prefix}_late_alpha_diff_Fext'] = alpha_l - F_ext
        else:
            metrics[f'{prefix}_late_alpha'] = None
            metrics[f'{prefix}_late_alpha_diff_Fext'] = None

    # ------------------------------------------------------------------
    # 4) Combine all constant-force experiments: a_sg = p1*F_ext + p2*v_sg + p3*v_sg^2 + intercept
    if len(constant_force_ids) >= 1:
        X_list = []
        y_list = []
        for eid in constant_force_ids:
            exp = experiments[eid]
            v = exp['_v_sg']
            a = exp['_a_sg']
            F_ext = exp['_F_ext']
            # Use only valid (finite) points
            mask = np.isfinite(v) & np.isfinite(a)
            v_m = v[mask]
            a_m = a[mask]
            # Design matrix without intercept
            X_local = np.column_stack([
                np.full_like(v_m, F_ext),
                v_m,
                v_m ** 2
            ])
            X_list.append(X_local)
            y_list.append(a_m)
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)

        # Fit with sklearn to get coefficients, then compute CI manually
        n_all = len(y_all)
        if n_all > 4:
            coeffs, intercept, coeff_ci_low, coeff_ci_high, intercept_ci_low, intercept_ci_high, r2_global = \
                _fit_with_ci(X_all, y_all, alpha=0.05)
            # coeffs: [p1, p2, p3]
            p1, p2, p3 = coeffs[0], coeffs[1], coeffs[2]
            metrics['global_p1'] = p1
            metrics['global_p1_ci_low'] = coeff_ci_low[0]
            metrics['global_p1_ci_high'] = coeff_ci_high[0]
            metrics['global_p2'] = p2
            metrics['global_p2_ci_low'] = coeff_ci_low[1]
            metrics['global_p2_ci_high'] = coeff_ci_high[1]
            metrics['global_p3'] = p3
            metrics['global_p3_ci_low'] = coeff_ci_low[2]
            metrics['global_p3_ci_high'] = coeff_ci_high[2]
            metrics['global_intercept'] = intercept
            metrics['global_intercept_ci_low'] = intercept_ci_low
            metrics['global_intercept_ci_high'] = intercept_ci_high
            metrics['global_R2'] = r2_global

            # Plot residuals
            y_pred_all = np.dot(np.column_stack([np.ones(n_all), X_all]),
                                np.concatenate([[intercept], coeffs]))
            residuals = y_all - y_pred_all
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_pred_all, residuals, s=5, alpha=0.5)
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax.set_xlabel('Predicted a_sg')
            ax.set_ylabel('Residual')
            ax.set_title('Global model residuals (constant-force experiments)')
            ax.grid(True, alpha=0.3)
            fname = 'global_constant_force_residuals.png'
            fig.savefig(str(output_dir / fname), dpi=100)
            plt.close(fig)
            figures.append(str(output_dir / fname))
        else:
            # Not enough points
            pass

    # ------------------------------------------------------------------
    # 5) For free experiments (F_ext=0): verify a_sg is zero or near zero
    for eid in free_ids:
        exp = experiments[eid]
        a = exp['_a_sg']
        mean_a = np.mean(a)
        std_a = np.std(a, ddof=1)
        # simple t-test if std>0
        if std_a > 1e-12:
            t_stat = mean_a / (std_a / np.sqrt(len(a)))
            # two-sided p-value
            from scipy.stats import t as t_dist
            p_val = 2 * t_dist.sf(abs(t_stat), df=len(a) - 1)
        else:
            p_val = 1.0
        metrics[f'{eid}_a_sg_mean'] = mean_a
        metrics[f'{eid}_a_sg_std'] = std_a
        metrics[f'{eid}_a_sg_t_test_pvalue'] = p_val

    # ------------------------------------------------------------------
    # Build observation text
    obs_parts = []
    obs_parts.append(f"对所有 {len(experiment_ids)} 个实验完成了指定分析。")
    obs_parts.append("1) 已绘制每个实验的 a_sg vs v_sg 散点图。")
    # constant-force fits
    if constant_force_ids:
        parts_fit = []
        for eid in constant_force_ids:
            p = metrics.get(f'{eid}_lin_alpha')
            b = metrics.get(f'{eid}_lin_beta')
            r2l = metrics.get(f'{eid}_lin_R2')
            parts_fit.append(f"{eid}: 线性 α={p:.4f}, β={b:.4f}, R²={r2l:.4f}")
        obs_parts.append("2) 恒外力实验线性与二次拟合系数及95% CI已在metrics中记录。")
    # split comparison
    obs_parts.append("3) 对每个恒外力实验按t<2和t>5分段线性拟合，截距与F_ext的差值已记录在metrics中。")
    # global fit
    if 'global_p1' in metrics:
        obs_parts.append(
            f"4) 全局合并拟合: a_sg = {metrics['global_p1']:.4f}*F_ext + {metrics['global_p2']:.4f}*v_sg + {metrics['global_p3']:.4f}*v_sg² + {metrics['global_intercept']:.4f}, R²={metrics['global_R2']:.4f}")
    # free experiments
    for eid in free_ids:
        mean_a = metrics.get(f'{eid}_a_sg_mean')
        std_a = metrics.get(f'{eid}_a_sg_std')
        pv = metrics.get(f'{eid}_a_sg_t_test_pvalue')
        if mean_a is not None:
            obs_parts.append(f"5) {eid}: a_sg 均值={mean_a:.6f}, 标准差={std_a:.6f}, t检验p值={pv:.4g}")

    observation = "\n".join(obs_parts)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
