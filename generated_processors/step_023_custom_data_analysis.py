import json
import math
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = payload["action"]
    parameters = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = Path(payload["output_dir"])
    exp_ids = parameters.get("experiment_ids", list(experiments.keys()))
    
    # Validate that all requested experiments exist
    missing = [eid for eid in exp_ids if eid not in experiments]
    if missing:
        raise ValueError(f"Experiment(s) not found: {missing}")
    
    # Helper: extract required series
    def get_series(eid: str, name: str) -> np.ndarray:
        if name not in experiments[eid]["series"]:
            raise ValueError(f"Series '{name}' not available in experiment {eid}")
        return np.array(experiments[eid]["series"][name])
    
    # Results containers
    figures = []
    metrics = {}
    derived_series = []
    observation_lines = []
    
    # Store per-experiment fit results for summary table
    summary_rows = []  # list of dicts for observation text
    
    # Step 1: per-experiment fits
    for eid in exp_ids:
        exp_cfg = experiments[eid]["config"]
        F_ext = exp_cfg.get("F_ext", 0.0)
        if F_ext <= 0:
            observation_lines.append(f"{eid}: F_ext={F_ext}, skipping analysis (requires F_ext>0)")
            continue
        v0 = exp_cfg.get("initial_v", 0.0)
        
        v = get_series(eid, "v_new")
        a = get_series(eid, "a_new")
        t = get_series(eid, "t")
        n = len(v)
        if n < 3:
            raise ValueError(f"Experiment {eid} has insufficient data points ({n})")
        
        # Normalized acceleration
        a_norm = a / F_ext
        
        # ---- exponential model: a = F_ext * exp(-b * v) ----
        def exp_model(vv, b):
            return F_ext * np.exp(-b * vv)
        
        # Initial guess: from linear fit of log(a/F_ext) vs v (if possible)
        # Use robust initial guess: b0 = - (mean(a) - F_ext) / (mean(v)*F_ext?) maybe better use log:
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                valid_mask = (a > 0) & (v >= 0)
                if np.any(valid_mask):
                    log_ratio = np.log(a[valid_mask] / F_ext)
                    v_valid = v[valid_mask]
                    if len(v_valid) > 1:
                        slope, intercept = np.polyfit(v_valid, log_ratio, 1)
                        b0 = max(-slope, 1e-6)
                    else:
                        b0 = 0.1
                else:
                    b0 = 0.1
        except:
            b0 = 0.1
        
        try:
            popt_exp, pcov_exp = curve_fit(exp_model, v, a, p0=[b0], bounds=([1e-12], [np.inf]))
            b_exp = popt_exp[0]
            a_pred_exp = exp_model(v, b_exp)
            residuals_exp = a - a_pred_exp
            ss_res_exp = np.sum(residuals_exp**2)
            ss_tot_exp = np.sum((a - np.mean(a))**2)
            r2_exp = 1 - ss_res_exp / ss_tot_exp if ss_tot_exp > 0 else np.nan
            resid_std_exp = np.std(residuals_exp, ddof=1)  # n-1 degrees but curve_fit uses n-params, fine for report
            
            # 95% CI for b
            if pcov_exp.ndim == 2 and pcov_exp[0][0] > 0:
                se_b = np.sqrt(pcov_exp[0][0])
                ci_low_b = b_exp - 1.96 * se_b
                ci_high_b = b_exp + 1.96 * se_b
            else:
                ci_low_b = ci_high_b = np.nan
        except Exception as e:
            b_exp = np.nan; r2_exp = np.nan; resid_std_exp = np.nan
            ci_low_b = np.nan; ci_high_b = np.nan
            a_pred_exp = np.full(n, np.nan)
        
        metrics[f"{eid}_exp_b"] = b_exp
        metrics[f"{eid}_exp_r2"] = r2_exp
        metrics[f"{eid}_exp_ci_b_low"] = ci_low_b
        metrics[f"{eid}_exp_ci_b_high"] = ci_high_b
        metrics[f"{eid}_exp_resid_std"] = resid_std_exp
        
        # ---- linear model: a = F_ext - k * v  => d = a - F_ext = -k * v ----
        d = a - F_ext
        # Force through origin: solve min sum (d + k*v)^2 => k = - (v @ d) / (v @ v)
        k_lin = -np.dot(v, d) / np.dot(v, v) if np.dot(v, v) > 0 else 0.0
        a_pred_lin = F_ext - k_lin * v
        residuals_lin = a - a_pred_lin
        ss_res_lin = np.sum(residuals_lin**2)
        r2_lin = 1 - ss_res_lin / ss_tot_exp if ss_tot_exp > 0 else np.nan
        resid_std_lin = np.std(residuals_lin, ddof=1)
        
        # 95% CI for k (using linear regression through origin standard error)
        # variance of k: MSE / sum(v^2)
        mse = ss_res_lin / (n - 1) if n > 1 else 0
        se_k = np.sqrt(mse / np.dot(v, v)) if np.dot(v, v) > 0 else np.nan
        ci_low_k = k_lin - 1.96 * se_k if se_k is not None else np.nan
        ci_high_k = k_lin + 1.96 * se_k if se_k is not None else np.nan
        
        metrics[f"{eid}_linear_k"] = k_lin
        metrics[f"{eid}_linear_r2"] = r2_lin
        metrics[f"{eid}_linear_ci_k_low"] = ci_low_k
        metrics[f"{eid}_linear_ci_k_high"] = ci_high_k
        metrics[f"{eid}_linear_resid_std"] = resid_std_lin
        
        # ---- quadratic model: a = F_ext - c * v^2 => d = -c * v^2 ----
        v2 = v ** 2
        c_quad = -np.dot(v2, d) / np.dot(v2, v2) if np.dot(v2, v2) > 0 else 0.0
        a_pred_quad = F_ext - c_quad * v2
        residuals_quad = a - a_pred_quad
        ss_res_quad = np.sum(residuals_quad**2)
        r2_quad = 1 - ss_res_quad / ss_tot_exp if ss_tot_exp > 0 else np.nan
        resid_std_quad = np.std(residuals_quad, ddof=1)
        
        mse_quad = ss_res_quad / (n - 1) if n > 1 else 0
        se_c = np.sqrt(mse_quad / np.dot(v2, v2)) if np.dot(v2, v2) > 0 else np.nan
        ci_low_c = c_quad - 1.96 * se_c if se_c is not None else np.nan
        ci_high_c = c_quad + 1.96 * se_c if se_c is not None else np.nan
        
        metrics[f"{eid}_quad_c"] = c_quad
        metrics[f"{eid}_quad_r2"] = r2_quad
        metrics[f"{eid}_quad_ci_c_low"] = ci_low_c
        metrics[f"{eid}_quad_ci_c_high"] = ci_high_c
        metrics[f"{eid}_quad_resid_std"] = resid_std_quad
        
        # ---- for v0=0 experiments, also check a/F_ext vs v ----
        is_v0_zero = abs(v0) < 1e-9
        if is_v0_zero:
            # Fit exponential to normalized data: a_norm = exp(-b_norm * v)
            def exp_norm(vv, bb):
                return np.exp(-bb * vv)
            try:
                with np.errstate(divide='ignore', invalid='ignore'):
                    valid_mask_norm = (v >= 0) & (a_norm > 0)
                    v_norm = v[valid_mask_norm]
                    a_norm_data = a_norm[valid_mask_norm]
                    if len(v_norm) > 2:
                        # initial guess from log
                        log_a_norm = np.log(a_norm_data)
                        slope_n, _ = np.polyfit(v_norm, log_a_norm, 1)
                        b0_norm = max(-slope_n, 1e-6)
                    else:
                        b0_norm = 0.1
                popt_norm, pcov_norm = curve_fit(exp_norm, v, a_norm, p0=[b0_norm], bounds=([1e-12], [np.inf]))
                b_norm = popt_norm[0]
                a_pred_norm = exp_norm(v, b_norm)
                ss_res_norm = np.sum((a_norm - a_pred_norm)**2)
                ss_tot_norm = np.sum((a_norm - np.mean(a_norm))**2)
                r2_norm = 1 - ss_res_norm / ss_tot_norm if ss_tot_norm > 0 else np.nan
                ci_norm = None
                if pcov_norm.ndim == 2 and pcov_norm[0][0] > 0:
                    se_norm = np.sqrt(pcov_norm[0][0])
                    ci_low_norm = b_norm - 1.96 * se_norm
                    ci_high_norm = b_norm + 1.96 * se_norm
                else:
                    ci_low_norm = ci_high_norm = np.nan
                metrics[f"{eid}_norm_exp_b"] = b_norm
                metrics[f"{eid}_norm_exp_r2"] = r2_norm
                metrics[f"{eid}_norm_exp_ci_b_low"] = ci_low_norm
                metrics[f"{eid}_norm_exp_ci_b_high"] = ci_high_norm
            except Exception as e:
                metrics[f"{eid}_norm_exp_b"] = np.nan
                metrics[f"{eid}_norm_exp_r2"] = np.nan
                metrics[f"{eid}_norm_exp_ci_b_low"] = np.nan
                metrics[f"{eid}_norm_exp_ci_b_high"] = np.nan
                a_pred_norm = np.full(n, np.nan)
        else:
            a_pred_norm = None
        
        # ---- plot for this experiment ----
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(v, a, s=20, alpha=0.7, label='Data (a_new vs v_new)', zorder=3)
        # Sort for smooth curves
        sort_idx = np.argsort(v)
        v_sorted = v[sort_idx]
        
        # Exponential fit (if valid)
        if not np.isnan(b_exp):
            a_sorted_exp = exp_model(v_sorted, b_exp)
            ax.plot(v_sorted, a_sorted_exp, 'r-', lw=2, label=f'Exp fit: a = {F_ext:.2f} * exp(-{b_exp:.4f} v)')
        # Linear fit
        a_sorted_lin = F_ext - k_lin * v_sorted
        ax.plot(v_sorted, a_sorted_lin, 'g--', lw=2, label=f'Linear fit: a = {F_ext:.2f} - {k_lin:.4f} v')
        # Quadratic fit
        a_sorted_quad = F_ext - c_quad * v_sorted**2
        ax.plot(v_sorted, a_sorted_quad, 'b-.', lw=2, label=f'Quad fit: a = {F_ext:.2f} - {c_quad:.4f} v²')
        
        ax.set_xlabel('v_new')
        ax.set_ylabel('a_new')
        ax.set_title(f'{eid} (F_ext={F_ext}, v0={v0}) - Model Comparison')
        ax.legend()
        plt.tight_layout()
        fpath = output_dir / f"{eid}_model_comparison.png"
        fig.savefig(fpath, dpi=100)
        plt.close(fig)
        figures.append(str(fpath))
        
        # For v0=0 experiments, also plot normalized check
        if is_v0_zero and a_pred_norm is not None:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.scatter(v, a_norm, s=20, alpha=0.7, label='a_new / F_ext')
            if not np.isnan(metrics.get(f"{eid}_norm_exp_b", np.nan)):
                a_sorted_norm = exp_norm(v_sorted, b_norm)
                ax2.plot(v_sorted, a_sorted_norm, 'r-', lw=2, label=f'Exp fit: a/F = exp(-{b_norm:.4f} v)')
            ax2.set_xlabel('v_new')
            ax2.set_ylabel('a_new / F_ext')
            ax2.set_title(f'{eid} (v0=0) - Normalized exponential check')
            ax2.legend()
            plt.tight_layout()
            fpath2 = output_dir / f"{eid}_norm_exp_check.png"
            fig2.savefig(fpath2, dpi=100)
            plt.close(fig2)
            figures.append(str(fpath2))
        
        # Record summary row
        row = {
            'experiment': eid,
            'F_ext': F_ext,
            'v0': v0,
            'exp_b': b_exp if not np.isnan(b_exp) else None,
            'exp_R2': r2_exp if not np.isnan(r2_exp) else None,
            'linear_k': k_lin,
            'linear_R2': r2_lin,
            'quad_c': c_quad,
            'quad_R2': r2_quad
        }
        summary_rows.append(row)
    
    # Step 2: global fit (combine all experiments)
    all_v = []
    all_a_norm = []  # a / F_ext
    all_labels = []
    for eid in exp_ids:
        exp_cfg = experiments[eid]["config"]
        F_ext = exp_cfg.get("F_ext", 0.0)
        if F_ext <= 0:
            continue
        v = get_series(eid, "v_new")
        a = get_series(eid, "a_new")
        a_norm = a / F_ext
        all_v.extend(v.tolist())
        all_a_norm.extend(a_norm.tolist())
        all_labels.extend([eid] * len(v))
    
    all_v = np.array(all_v)
    all_a_norm = np.array(all_a_norm)
    
    # Global exponential fit: a_norm = exp(-b_global * v)
    def exp_global(vv, bg):
        return np.exp(-bg * vv)
    
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            valid_global = (all_v >= 0) & (all_a_norm > 0)
            v_g = all_v[valid_global]
            a_g = all_a_norm[valid_global]
            if len(v_g) > 2:
                log_a = np.log(a_g)
                slope_g, _ = np.polyfit(v_g, log_a, 1)
                b0_global = max(-slope_g, 1e-6)
            else:
                b0_global = 0.1
        popt_global, pcov_global = curve_fit(exp_global, all_v, all_a_norm, p0=[b0_global], bounds=([1e-12], [np.inf]))
        b_global = popt_global[0]
        a_pred_global = exp_global(all_v, b_global)
        ss_res_global = np.sum((all_a_norm - a_pred_global)**2)
        ss_tot_global = np.sum((all_a_norm - np.mean(all_a_norm))**2)
        r2_global = 1 - ss_res_global / ss_tot_global if ss_tot_global > 0 else np.nan
        ci_global = None
        if pcov_global.ndim == 2 and pcov_global[0][0] > 0:
            se_global = np.sqrt(pcov_global[0][0])
            ci_low_global = b_global - 1.96 * se_global
            ci_high_global = b_global + 1.96 * se_global
        else:
            ci_low_global = ci_high_global = np.nan
    except Exception as e:
        b_global = np.nan; r2_global = np.nan
        ci_low_global = ci_high_global = np.nan
        a_pred_global = np.full(len(all_v), np.nan)
    
    metrics['global_exp_b'] = b_global
    metrics['global_exp_r2'] = r2_global
    metrics['global_exp_ci_b_low'] = ci_low_global
    metrics['global_exp_ci_b_high'] = ci_high_global
    
    # Global plot
    fig_global, ax_global = plt.subplots(figsize=(9, 7))
    unique_ids = list(dict.fromkeys(all_labels))  # preserve order
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_ids)))
    for idx, eid in enumerate(unique_ids):
        mask = np.array(all_labels) == eid
        ax_global.scatter(all_v[mask], all_a_norm[mask], s=15, alpha=0.6, color=colors[idx], label=f'{eid}')
    # Global fit curve
    sort_idx_global = np.argsort(all_v)
    v_global_sorted = all_v[sort_idx_global]
    if not np.isnan(b_global):
        a_global_sorted = exp_global(v_global_sorted, b_global)
        ax_global.plot(v_global_sorted, a_global_sorted, 'k-', lw=2, label=f'Global exp fit: a/F = exp(-{b_global:.4f} v), R²={r2_global:.3f}')
    ax_global.set_xlabel('v_new')
    ax_global.set_ylabel('a_new / F_ext')
    ax_global.set_title('Global exponential fit (all constant-force experiments)')
    ax_global.legend()
    plt.tight_layout()
    fpath_global = output_dir / "global_exp_fit.png"
    fig_global.savefig(fpath_global, dpi=100)
    plt.close(fig_global)
    figures.append(str(fpath_global))
    
    # Build observation text
    # Summarize per-experiment fits
    obs_lines = []
    obs_lines.append("Per-experiment model fitting results:")
    for row in summary_rows:
        e = row['experiment']
        obs_lines.append(f"  {e} (F_ext={row['F_ext']}, v0={row['v0']}):")
        obs_lines.append(f"    Exponential: b={row['exp_b']:.4f}, R²={row['exp_R2']:.4f}" if row['exp_b'] is not None else "    Exponential: failed")
        obs_lines.append(f"    Linear: k={row['linear_k']:.4f}, R²={row['linear_R2']:.4f}")
        obs_lines.append(f"    Quadratic: c={row['quad_c']:.4f}, R²={row['quad_R2']:.4f}")
    obs_lines.append(f"Global exponential fit (a/F_ext = exp(-b*v)): b_global={b_global:.4f}, R²={r2_global:.4f}" if not np.isnan(b_global) else "Global fit failed.")
    obs_lines.append("95% CI for b (if available) are recorded in metrics with keys *_exp_ci_b_low/_high.")
    obs_lines.append("Figures saved: per-experiment model comparison plots; for v0=0 experiments (exp_02,03,04) additional normalized exponential check plots; global fit plot.")
    observation = "\n".join(obs_lines)
    
    # Return
    return {
        "observation": observation,
        "derived_series": [],  # no new series defined
        "figures": figures,
        "metrics": {
            k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in metrics.items()
        }
    }
