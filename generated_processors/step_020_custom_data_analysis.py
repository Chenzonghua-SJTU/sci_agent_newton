import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import ttest_1samp, t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def _get_accel_velocity(exp, prefer_new=True):
    """Return (a, v) arrays for given experiment.
    Prefer a_new/v_new if they exist, else fallback to a_sg/v_sg.
    """
    series = exp['series']
    avail = exp['available_series']
    if prefer_new and 'a_new' in avail and 'v_new' in avail:
        a = np.array(series['a_new'])
        v = np.array(series['v_new'])
    elif 'a_sg' in avail and 'v_sg' in avail:
        a = np.array(series['a_sg'])
        v = np.array(series['v_sg'])
    else:
        raise ValueError(f"No suitable acceleration/velocity series for {exp['config'].get('experiment_id','unknown')}")
    # ensure 1D
    if a.ndim != 1:
        a = a.flatten()
    if v.ndim != 1:
        v = v.flatten()
    return a, v

def _linear_func(v, alpha, beta):
    return alpha + beta * v

def _quad_func(v, alpha, gamma):
    return alpha + gamma * v**2

def _power_law_neg(v, beta, gamma):
    """ d = -beta * v^gamma, beta>0, gamma>0 """
    return -beta * np.power(v, gamma)

def _fit_model(x, y, model_func, p0=None, bounds=(-np.inf, np.inf)):
    """Fit model_func to data (x,y) using curve_fit.
    Returns (popt, pcov, R2, n_eff), where n_eff is number of valid points.
    If fitting fails, returns NaNs.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    n_eff = len(x_clean)
    if n_eff < 3:
        return np.array([np.nan, np.nan]), np.array([[np.nan, np.nan], [np.nan, np.nan]]), np.nan, n_eff
    try:
        popt, pcov = curve_fit(model_func, x_clean, y_clean, p0=p0, bounds=bounds, maxfev=10000)
        y_pred = model_func(x_clean, *popt)
        ss_res = np.sum((y_clean - y_pred)**2)
        ss_tot = np.sum((y_clean - np.mean(y_clean))**2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return popt, pcov, R2, n_eff
    except Exception:
        return np.array([np.nan, np.nan]), np.array([[np.nan, np.nan], [np.nan, np.nan]]), np.nan, n_eff

def _conf_interval(popt, pcov, alpha=0.05):
    """95% confidence interval given popt and pcov.
    Returns list of (low, high) per parameter.
    If pcov has NaNs, returns NaNs.
    """
    n = len(popt)
    try:
        perr = np.sqrt(np.diag(pcov))
        # approximate t-value for large n -> 1.96
        t_val = 1.96
        ci_low = popt - t_val * perr
        ci_high = popt + t_val * perr
        return [(ci_low[i], ci_high[i]) for i in range(n)]
    except Exception:
        return [(np.nan, np.nan)] * n

def _save_figure(fig, filename, output_dir):
    full_path = Path(output_dir) / filename
    fig.savefig(str(full_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(full_path)

def process(payload):
    output_dir = Path(payload['output_dir'])
    experiments = payload['experiments']
    param = payload['parameters']
    
    # Determine which experiments to process
    exp_ids = param.get('experiment_ids', list(experiments.keys()))
    
    # For easy access
    fext_map = {}
    for eid in exp_ids:
        config = experiments[eid]['config']
        fext_map[eid] = config.get('F_ext', 0.0)
    
    derived_series = []
    figures = []
    metrics = {}
    
    # ------------------------------------------------------------
    # 1. Load a and v for all experiments, fallback as needed
    # ------------------------------------------------------------
    a_data = {}
    v_data = {}
    for eid in exp_ids:
        exp = experiments[eid]
        a, v = _get_accel_velocity(exp, prefer_new='a_new' in exp['available_series'])
        a_data[eid] = a
        v_data[eid] = v
    
    # ------------------------------------------------------------
    # 2. No external force experiments (exp_01, exp_05): mean test
    # ------------------------------------------------------------
    free_exp_ids = ['exp_01', 'exp_05']
    for eid in free_exp_ids:
        if eid not in exp_ids:
            continue
        a_vals = a_data[eid]
        mean_val = float(np.mean(a_vals))
        std_val = float(np.std(a_vals, ddof=1))
        # t-test against 0
        if std_val > 0 and len(a_vals) > 1:
            t_stat, p_val = ttest_1samp(a_vals, 0)
        else:
            t_stat, p_val = 0.0, 1.0
        metrics[f'{eid}_a_mean'] = mean_val
        metrics[f'{eid}_a_std'] = std_val
        metrics[f'{eid}_a_t_pvalue'] = float(p_val)
    
    # ------------------------------------------------------------
    # 3. Constant force experiments (F_ext > 0)
    # ------------------------------------------------------------
    const_exp_ids = [eid for eid in exp_ids if fext_map[eid] > 0 and eid in ['exp_02','exp_03','exp_04','exp_06','exp_07']]
    
    # For each constant exp: fit a vs v (linear, quadratic, power) and d vs v
    for eid in const_exp_ids:
        a = a_data[eid]
        v = v_data[eid]
        Fext = fext_map[eid]
        d = a - Fext
        
        # -- a vs v fits (all data) --
        # linear
        popt_lin, pcov_lin, R2_lin, n_lin = _fit_model(v, a, _linear_func, p0=[Fext, -0.2])
        ci_lin = _conf_interval(popt_lin, pcov_lin)
        # quadratic
        popt_quad, pcov_quad, R2_quad, n_quad = _fit_model(v, a, _quad_func, p0=[Fext, -0.05])
        ci_quad = _conf_interval(popt_quad, pcov_quad)
        # power law: a = Fext - beta * v^gamma  (only v>0)
        mask_power = v > 1e-12
        if np.sum(mask_power) >= 10:
            v_pos = v[mask_power]
            d_pos = d[mask_power]
            a_pos = a[mask_power]
            # fit d = -beta * v^gamma
            popt_pow, pcov_pow, R2_pow, n_pow = _fit_model(v_pos, d_pos, _power_law_neg, p0=[0.5, 0.5], bounds=([0, 0], [np.inf, np.inf]))
            ci_pow = _conf_interval(popt_pow, pcov_pow)
        else:
            popt_pow = np.array([np.nan, np.nan])
            pcov_pow = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            R2_pow = np.nan
            n_pow = 0
            ci_pow = [(np.nan, np.nan), (np.nan, np.nan)]
        
        # Store metrics
        metrics[f'{eid}_lin_alpha'] = float(popt_lin[0]) if not np.isnan(popt_lin[0]) else np.nan
        metrics[f'{eid}_lin_beta'] = float(popt_lin[1]) if not np.isnan(popt_lin[1]) else np.nan
        metrics[f'{eid}_lin_R2'] = float(R2_lin) if not np.isnan(R2_lin) else np.nan
        metrics[f'{eid}_quad_alpha'] = float(popt_quad[0]) if not np.isnan(popt_quad[0]) else np.nan
        metrics[f'{eid}_quad_gamma'] = float(popt_quad[1]) if not np.isnan(popt_quad[1]) else np.nan
        metrics[f'{eid}_quad_R2'] = float(R2_quad) if not np.isnan(R2_quad) else np.nan
        metrics[f'{eid}_power_beta'] = float(popt_pow[0]) if not np.isnan(popt_pow[0]) else np.nan
        metrics[f'{eid}_power_gamma'] = float(popt_pow[1]) if not np.isnan(popt_pow[1]) else np.nan
        metrics[f'{eid}_power_R2'] = float(R2_pow) if not np.isnan(R2_pow) else np.nan
        
        # -- d vs v fits --
        # linear
        popt_dlin, pcov_dlin, R2_dlin, n_dlin = _fit_model(v, d, _linear_func, p0=[-Fext, -0.1])
        ci_dlin = _conf_interval(popt_dlin, pcov_dlin)
        # quadratic
        popt_dquad, pcov_dquad, R2_dquad, n_dquad = _fit_model(v, d, _quad_func, p0=[-Fext, -0.05])
        ci_dquad = _conf_interval(popt_dquad, pcov_dquad)
        # power law: d = -beta * v^gamma (only v>0) (already fitted above)
        
        metrics[f'{eid}_d_lin_alpha'] = float(popt_dlin[0]) if not np.isnan(popt_dlin[0]) else np.nan
        metrics[f'{eid}_d_lin_beta'] = float(popt_dlin[1]) if not np.isnan(popt_dlin[1]) else np.nan
        metrics[f'{eid}_d_lin_R2'] = float(R2_dlin) if not np.isnan(R2_dlin) else np.nan
        metrics[f'{eid}_d_quad_alpha'] = float(popt_dquad[0]) if not np.isnan(popt_dquad[0]) else np.nan
        metrics[f'{eid}_d_quad_gamma'] = float(popt_dquad[1]) if not np.isnan(popt_dquad[1]) else np.nan
        metrics[f'{eid}_d_quad_R2'] = float(R2_dquad) if not np.isnan(R2_dquad) else np.nan
        
        # -- Resid plots for a vs v --
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        # Data scatter
        axs[0,0].scatter(v, a, s=5, alpha=0.5)
        axs[0,0].set_xlabel('v')
        axs[0,0].set_ylabel('a')
        axs[0,0].set_title(f'{eid} (F_ext={Fext}) a vs v')
        # Plot fits
        v_sort = np.sort(v)
        if not np.isnan(popt_lin[0]):
            axs[0,0].plot(v_sort, _linear_func(v_sort, *popt_lin), 'r-', label=f'Linear R²={R2_lin:.4f}')
        if not np.isnan(popt_quad[0]):
            axs[0,0].plot(v_sort, _quad_func(v_sort, *popt_quad), 'g-', label=f'Quadratic R²={R2_quad:.4f}')
        if not np.isnan(popt_pow[0]) and np.sum(mask_power) > 0:
            v_pos_sorted = np.sort(v_pos)
            axs[0,0].plot(v_pos_sorted, Fext + _power_law_neg(v_pos_sorted, *popt_pow), 'b-', label=f'Power R²={R2_pow:.4f}')
        axs[0,0].legend(fontsize=8)
        # Residuals linear
        if not np.isnan(popt_lin[0]):
            resid_lin = a - _linear_func(v, *popt_lin)
            axs[0,1].scatter(v, resid_lin, s=5, alpha=0.5)
            axs[0,1].axhline(0, color='k', lw=0.5)
            axs[0,1].set_xlabel('v')
            axs[0,1].set_ylabel('Residual')
            axs[0,1].set_title('Linear residual')
        # Residuals quadratic
        if not np.isnan(popt_quad[0]):
            resid_quad = a - _quad_func(v, *popt_quad)
            axs[1,0].scatter(v, resid_quad, s=5, alpha=0.5)
            axs[1,0].axhline(0, color='k', lw=0.5)
            axs[1,0].set_xlabel('v')
            axs[1,0].set_ylabel('Residual')
            axs[1,0].set_title('Quadratic residual')
        # Residuals power
        if not np.isnan(popt_pow[0]) and np.sum(mask_power) > 0:
            resid_pow = d_pos - _power_law_neg(v_pos, *popt_pow)
            axs[1,1].scatter(v_pos, resid_pow, s=5, alpha=0.5)
            axs[1,1].axhline(0, color='k', lw=0.5)
            axs[1,1].set_xlabel('v')
            axs[1,1].set_ylabel('Residual')
            axs[1,1].set_title('Power residual')
        fig.tight_layout()
        fig_path = _save_figure(fig, f'{eid}_a_vs_v_fits.png', output_dir)
        figures.append(fig_path)
        
        # -- Resid plots for d vs v --
        fig2, axs2 = plt.subplots(2,2, figsize=(12,10))
        axs2[0,0].scatter(v, d, s=5, alpha=0.5)
        axs2[0,0].set_xlabel('v')
        axs2[0,0].set_ylabel('d')
        axs2[0,0].set_title(f'{eid} d vs v')
        if not np.isnan(popt_dlin[0]):
            axs2[0,0].plot(v_sort, _linear_func(v_sort, *popt_dlin), 'r-', label=f'Linear R²={R2_dlin:.4f}')
        if not np.isnan(popt_dquad[0]):
            axs2[0,0].plot(v_sort, _quad_func(v_sort, *popt_dquad), 'g-', label=f'Quadratic R²={R2_dquad:.4f}')
        if not np.isnan(popt_pow[0]) and np.sum(mask_power) > 0:
            axs2[0,0].plot(v_pos_sorted, _power_law_neg(v_pos_sorted, *popt_pow), 'b-', label=f'Power R²={R2_pow:.4f}')
        axs2[0,0].legend(fontsize=8)
        # residuals d linear
        if not np.isnan(popt_dlin[0]):
            resid_dlin = d - _linear_func(v, *popt_dlin)
            axs2[0,1].scatter(v, resid_dlin, s=5, alpha=0.5)
            axs2[0,1].axhline(0, color='k', lw=0.5)
            axs2[0,1].set_title('d linear residual')
        # residuals d quad
        if not np.isnan(popt_dquad[0]):
            resid_dquad = d - _quad_func(v, *popt_dquad)
            axs2[1,0].scatter(v, resid_dquad, s=5, alpha=0.5)
            axs2[1,0].axhline(0, color='k', lw=0.5)
            axs2[1,0].set_title('d quadratic residual')
        # residuals d power
        if not np.isnan(popt_pow[0]) and np.sum(mask_power) > 0:
            resid_dpow = d_pos - _power_law_neg(v_pos, *popt_pow)
            axs2[1,1].scatter(v_pos, resid_dpow, s=5, alpha=0.5)
            axs2[1,1].axhline(0, color='k', lw=0.5)
            axs2[1,1].set_title('d power residual')
        fig2.tight_layout()
        fig2_path = _save_figure(fig2, f'{eid}_d_vs_v_fits.png', output_dir)
        figures.append(fig2_path)
        
        # -- Store derived series: d for this exp (only if not already present) --
        # We'll return d as derived series
        derived_series.append({
            'experiment_id': eid,
            'name': 'd',
            'values': d.tolist(),
            'source_name': f'a_new - F_ext (Fext={Fext})',
            'provenance': 'generated data processor: custom_data_analysis',
            'description': f'Damping term d = a - F_ext for experiment {eid}'
        })
    
    # ------------------------------------------------------------
    # 4. Global fit: a = F_ext - beta * v^gamma over all constant experiments
    # ------------------------------------------------------------
    all_v = []
    all_a = []
    all_F = []
    for eid in const_exp_ids:
        a = a_data[eid]
        v = v_data[eid]
        Fext = fext_map[eid]
        mask = v > 1e-12
        all_v.extend(v[mask].tolist())
        a_sub = a[mask]
        all_a.extend(a_sub.tolist())
        all_F.extend([Fext] * np.sum(mask))
    if len(all_v) >= 10:
        v_global = np.array(all_v)
        a_global = np.array(all_a)
        F_global = np.array(all_F)
        # objective: a - F_ext = -beta * v^gamma
        def global_power(v, F_ext, beta, gamma):
            return F_ext - beta * np.power(v, gamma)
        # wrap for curve_fit: function(y, *params)
        def global_resid(v, F_ext, beta, gamma):
            return global_power(v, F_ext, beta, gamma)
        # We need to provide both v and F_ext as x data. curve_fit expects single x.
        # So we use a wrapper: we pass x = (v, F_ext) and unpack
        def global_model(x, beta, gamma):
            v_arr, F_arr = x
            return F_arr - beta * np.power(v_arr, gamma)
        # pack x as (v_global, F_global)
        x_data = (v_global, F_global)
        popt_global, pcov_global, R2_global, n_global = _fit_model(x_data, a_global, global_model, p0=[0.5, 0.5], bounds=([0, 0], [np.inf, np.inf]))
        if np.isfinite(R2_global):
            ci_global = _conf_interval(popt_global, pcov_global)
            metrics['global_beta'] = float(popt_global[0])
            metrics['global_gamma'] = float(popt_global[1])
            metrics['global_R2'] = float(R2_global)
            metrics['global_ci_beta_low'] = ci_global[0][0]
            metrics['global_ci_beta_high'] = ci_global[0][1]
            metrics['global_ci_gamma_low'] = ci_global[1][0]
            metrics['global_ci_gamma_high'] = ci_global[1][1]
        else:
            metrics['global_beta'] = np.nan
            metrics['global_gamma'] = np.nan
            metrics['global_R2'] = np.nan
        
        # Plot global fit
        fig_global, ax_global = plt.subplots(figsize=(8,6))
        colors = {'exp_02':'blue','exp_03':'orange','exp_04':'green','exp_06':'red','exp_07':'purple'}
        for eid in const_exp_ids:
            mask_v = v_data[eid] > 1e-12
            ax_global.scatter(v_data[eid][mask_v], a_data[eid][mask_v], s=5, label=eid, color=colors.get(eid,'gray'), alpha=0.5)
        if not np.isnan(popt_global[0]):
            v_sort_global = np.sort(v_global)
            a_pred = global_model((v_sort_global, np.ones_like(v_sort_global)*np.mean(F_global)), *popt_global)
            # We can't easily show curve for each F, so just show one with mean F_ext
            ax_global.plot(v_sort_global, a_pred, 'k--', label=f'Global: β={popt_global[0]:.4f}, γ={popt_global[1]:.4f}')
        ax_global.set_xlabel('v')
        ax_global.set_ylabel('a')
        ax_global.set_title('Global fit a = F_ext - β·v^γ')
        ax_global.legend(fontsize=7)
        fig_global_path = _save_figure(fig_global, 'global_power_fit.png', output_dir)
        figures.append(fig_global_path)
    else:
        metrics['global_beta'] = np.nan
        metrics['global_gamma'] = np.nan
        metrics['global_R2'] = np.nan
    
    # ------------------------------------------------------------
    # Build observation string
    # ------------------------------------------------------------
    obs_parts = []
    obs_parts.append(f"处理了 {len(exp_ids)} 个实验：{', '.join(exp_ids)}。")
    # No external force
    for eid in free_exp_ids:
        if eid in exp_ids:
            mean_v = metrics.get(f'{eid}_a_mean', np.nan)
            p_v = metrics.get(f'{eid}_a_t_pvalue', np.nan)
            obs_parts.append(f"{eid} (F_ext=0)：a 的均值={mean_v:.6f}，t检验 p值={p_v:.4f}。")
    # Constant force experiments: highlight best fit among a vs v
    for eid in const_exp_ids:
        R2_lin = metrics.get(f'{eid}_lin_R2', np.nan)
        R2_quad = metrics.get(f'{eid}_quad_R2', np.nan)
        R2_pow = metrics.get(f'{eid}_power_R2', np.nan)
        obs_parts.append(f"{eid} (F_ext={fext_map[eid]})：a vs v 线性R²={R2_lin:.4f}, 二次R²={R2_quad:.4f}, 幂律R²={R2_pow:.4f}。")
        R2_dlin = metrics.get(f'{eid}_d_lin_R2', np.nan)
        R2_dquad = metrics.get(f'{eid}_d_quad_R2', np.nan)
        obs_parts.append(f"  d vs v 线性R²={R2_dlin:.4f}, 二次R²={R2_dquad:.4f}, 幂律R²={R2_pow:.4f}。")
    # Global
    beta_g = metrics.get('global_beta', np.nan)
    gamma_g = metrics.get('global_gamma', np.nan)
    R2_g = metrics.get('global_R2', np.nan)
    obs_parts.append(f"全局幂律 a = F_ext - β·v^γ: β={beta_g:.4f}, γ={gamma_g:.4f}, R²={R2_g:.4f}。")
    
    observation = '\n'.join(obs_parts)
    
    # ------------------------------------------------------------
    # Return
    # ------------------------------------------------------------
    return {
        'observation': observation,
        'derived_series': derived_series,
        'figures': figures,
        'metrics': metrics
    }
