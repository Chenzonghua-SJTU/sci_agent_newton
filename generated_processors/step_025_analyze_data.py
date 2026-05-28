import json
import math
import statistics
import itertools
import functools
import collections
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process(payload: dict) -> dict:
    # extract parameters
    parameters = payload['parameters']
    analysis_mode = parameters.get('analysis_mode', '')
    if analysis_mode != 'maintain_ledger':
        raise ValueError('This implementation only supports maintain_ledger mode.')
    
    experiments = payload['experiments']
    output_dir = payload['output_dir']
    
    # ---------- collect global data for all experiments ----------
    all_t = []      # only for counting, not used
    all_q = []
    all_v = []
    all_a = []
    all_F = []
    exp_keys_global = []
    for eid, exp in experiments.items():
        if 't' not in exp['series'] or 'q' not in exp['series'] or 'v' not in exp['series'] or 'a' not in exp['series']:
            continue
        t_vals = exp['series']['t']
        q_vals = exp['series']['q']
        v_vals = exp['series']['v']
        a_vals = exp['series']['a']
        F_ext = exp['config']['F_ext']
        # all series should have same length
        n = len(t_vals)
        all_t.extend(t_vals)
        all_q.extend(q_vals)
        all_v.extend(v_vals)
        all_a.extend(a_vals)
        all_F.extend([F_ext] * n)
        exp_keys_global.append(eid)
    
    all_t = np.array(all_t)
    all_q = np.array(all_q)
    all_v = np.array(all_v)
    all_a = np.array(all_a)
    all_F = np.array(all_F)
    
    # ---------- global regression helper ----------
    def build_design_matrix(F, v, q, v2, model_idx):
        # model 1 : a ~ F + v + intercept
        if model_idx == 1:
            X = np.column_stack([F, v, np.ones_like(F)])
        elif model_idx == 2:  # a ~ F + v + q
            X = np.column_stack([F, v, q, np.ones_like(F)])
        elif model_idx == 3:  # a ~ F + v + v2
            X = np.column_stack([F, v, v2, np.ones_like(F)])
        elif model_idx == 4:  # a ~ F + v + q + v2
            X = np.column_stack([F, v, q, v2, np.ones_like(F)])
        else:
            raise ValueError(f'Unknown model index {model_idx}')
        return X
    
    def linear_regression_stats(X, y):
        # returns beta, r2, rmse, se
        n, p = X.shape
        beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        SS_res = np.sum((y - y_pred)**2)
        SS_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - SS_res / SS_tot
        rmse = np.sqrt(SS_res / n)
        # standard errors
        MSE = SS_res / (n - p)
        # avoid singular case, pseudo inverse
        try:
            XtX_inv = np.linalg.pinv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)
        cov_beta = MSE * XtX_inv
        se = np.sqrt(np.diag(cov_beta))
        return beta, r2, rmse, se
    
    # iterate four models
    model_labels = [
        'a ~ F_ext + v',
        'a ~ F_ext + v + q',
        'a ~ F_ext + v + v^2',
        'a ~ F_ext + v + q + v^2'
    ]
    model_results = []
    for idx, label in enumerate(model_labels, 1):
        v2_vals = all_v**2
        X = build_design_matrix(all_F, all_v, all_q, v2_vals, idx)
        beta, r2, rmse, se = linear_regression_stats(X, all_a)
        # determine column names
        if idx == 1:
            col_names = ['F_ext', 'v', 'intercept']
        elif idx == 2:
            col_names = ['F_ext', 'v', 'q', 'intercept']
        elif idx == 3:
            col_names = ['F_ext', 'v', 'v^2', 'intercept']
        else:
            col_names = ['F_ext', 'v', 'q', 'v^2', 'intercept']
        model_results.append({
            'model_label': label,
            'beta': beta,
            'se': se,
            'r2': r2,
            'rmse': rmse,
            'col_names': col_names
        })
    
    # ---------- best model selection ----------
    best_idx = np.argmax([m['r2'] for m in model_results])
    best_model = model_results[best_idx]
    
    # ---------- residual plot for best model ----------
    best_X = build_design_matrix(all_F, all_v, all_q, all_v**2, best_idx+1)
    best_pred = best_X @ best_model['beta']
    residuals = all_a - best_pred
    
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Residual (a_true - a_pred)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Residual distribution of best model: {best_model["model_label"]}')
    ax.axvline(0, color='red', linestyle='--')
    fig_path = pathlib.Path(output_dir) / 'best_model_residual_hist.png'
    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    figure_paths = [str(fig_path)]
    
    # ---------- build observations for global regression ----------
    observations = []
    for i, m in enumerate(model_results):
        coef_dict = {}
        se_dict = {}
        for j, cn in enumerate(m['col_names']):
            coef_dict[f'coef_{cn}'] = float(m['beta'][j])
            se_dict[f'se_{cn}'] = float(m['se'][j])
        metrics_entry = {
            'R2': float(m['r2']),
            'RMSE': float(m['rmse']),
            'observation_count': len(all_a),
        }
        metrics_entry.update(coef_dict)
        metrics_entry.update(se_dict)
        summary_text = (
            f"全局多元线性回归模型 '{m['model_label']}': "
            f"R²={m['r2']:.6f}, RMSE={m['rmse']:.6f}, "
            f"系数: {', '.join(f'{cn}={beta_j:.6f}' for cn, beta_j in zip(m['col_names'], m['beta']))}"
        )
        obs = {
            'summary': summary_text,
            'source_data_refs': [f"{eid}:t,q,v,a,F_ext" for eid in exp_keys_global],
            'metrics': metrics_entry
        }
        observations.append(obs)
    
    # ---------- per-experiment a-v regression for constant field ----------
    constant_experiment_ids = []
    for eid, exp in experiments.items():
        if exp['config']['force_field_type'] == 'constant':
            constant_experiment_ids.append(eid)
    
    for eid in constant_experiment_ids:
        exp = experiments[eid]
        v_vals = np.array(exp['series']['v'])
        a_vals = np.array(exp['series']['a'])
        if len(v_vals) < 2:
            continue
        # linear regression a vs v
        slope, intercept, r_value, p_value, std_err = linregress(v_vals, a_vals)
        r2 = r_value**2
        a0 = float(a_vals[0])
        v0 = float(v_vals[0])
        a_end = float(a_vals[-1])
        v_end = float(v_vals[-1])
        F_ext = exp['config']['F_ext']
        summary_text = (
            f"常数场实验 {eid}: F_ext={F_ext}, v0={v0:.6f}, a0={a0:.6f}, "
            f"v_end={v_end:.6f}, a_end={a_end:.6f}, slope_av={slope:.6f}, "
            f"intercept_av={intercept:.6f}, R²_av={r2:.6f}"
        )
        obs = {
            'summary': summary_text,
            'source_data_refs': [f"{eid}:a", f"{eid}:v"],
            'metrics': {
                'F_ext': F_ext,
                'slope_av': float(slope),
                'intercept_av': float(intercept),
                'R2_av': float(r2),
                'a0': a0,
                'v0': v0,
                'a_end': a_end,
                'v_end': v_end
            }
        }
        observations.append(obs)
    
    # ---------- overall metrics ----------
    overall_metrics = {
        'experiment_count': len(experiments),
        'constant_experiment_count': len(constant_experiment_ids),
        'global_regression_count': 4,
        'best_model_index': best_idx + 1,
        'best_model_R2': float(best_model['r2']),
        'best_model_RMSE': float(best_model['rmse']),
        'total_observations': len(observations)
    }
    
    # ---------- observation text ----------
    observation_text = (
        f"完成了对所有 {len(experiments)} 个实验的全局多元线性回归（4个模型），"
        f"最佳模型为 #{best_idx+1}: {best_model['model_label']}，R²={best_model['r2']:.4f}，RMSE={best_model['rmse']:.4f}。"
        f"已对最佳模型绘制残差分布直方图。同时为 {len(constant_experiment_ids)} 个常数场实验逐一计算了a-v线性回归参数。"
        f"共生成 {len(observations)} 条OBS。"
    )
    
    return {
        'observation': observation_text,
        'derived_series': [],
        'observations': observations,
        'figures': figure_paths,
        'metrics': overall_metrics
    }
