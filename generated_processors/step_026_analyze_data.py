import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn import linear_model, metrics, preprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _get_series(exp: dict, series_name: str) -> Optional[np.ndarray]:
    """Get a series by name from experiment, preferring exact match, then fallback to est_ variants."""
    if series_name in exp['series']:
        return np.array(exp['series'][series_name])
    # try estimated names
    exp_id = exp.get('id', '')
    possible = [series_name, f"{series_name}_est_{exp_id}", f"a_est_{exp_id}", f"v_est_{exp_id}"]
    for p in possible:
        if p in exp['series']:
            return np.array(exp['series'][p])
    return None

def process(payload: dict) -> dict:
    action = payload.get('action')
    params = payload.get('parameters', {})
    experiments = payload.get('experiments', {})
    exp_ids = params.get('experiment_ids', None)
    if exp_ids is None:
        exp_ids = list(experiments.keys())
    output_dir = Path(payload.get('output_dir', '.'))
    observations = []
    derived_series = []
    figures = []
    metrics_all = {}

    # Collect configuration and results
    all_configs = {}
    constant_force_exps = []   # F_ext != 0
    free_exps = []            # F_ext == 0
    for eid in exp_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        cfg = exp.get('config', {})
        force_field_type = cfg.get('force_field_type', '')
        F_ext = cfg.get('F_ext', None)
        initial_q = cfg.get('initial_q', None)
        initial_v = cfg.get('initial_v', None)
        all_configs[eid] = {
            'force_field_type': force_field_type,
            'F_ext': F_ext,
            'initial_q': initial_q,
            'initial_v': initial_v
        }
        # Determine if constant force (F_ext != 0) or free (F_ext == 0)
        if force_field_type == 'free' or (F_ext is not None and abs(float(F_ext)) < 1e-12):
            free_exps.append(eid)
        else:
            constant_force_exps.append(eid)

    # 1) Record configuration for all experiments
    config_summary = "实验配置清单:\n"
    for eid in exp_ids:
        c = all_configs.get(eid)
        if c:
            config_summary += f"{eid}: force_field_type={c['force_field_type']}, F_ext={c['F_ext']}, initial_q={c['initial_q']}, initial_v={c['initial_v']}\n"
    observations.append({
        'summary': config_summary,
        'source_data_refs': [f"{eid}:config" for eid in exp_ids if eid in experiments],
        'metrics': {'experiments_count': len(exp_ids)}
    })

    # 2) Process constant force experiments (F_ext != 0)
    const_results = []   # list of dicts
    for eid in constant_force_exps:
        exp = experiments[eid]
        # Get a and v
        a_series = _get_series(exp, 'a')
        v_series = _get_series(exp, 'v')
        if a_series is None or v_series is None:
            # skip if missing
            continue
        # Ensure same length
        n = min(len(a_series), len(v_series))
        a_series = a_series[:n]
        v_series = v_series[:n]
        # Prepare data for regressions
        v2 = v_series ** 2
        # Regression a ~ v
        coeff_v = np.polyfit(v_series, a_series, deg=1)
        slope_v, intercept_v = coeff_v[0], coeff_v[1]
        a_pred_v = np.polyval(coeff_v, v_series)
        r2_v = 1 - np.sum((a_series - a_pred_v)**2) / np.sum((a_series - np.mean(a_series))**2)
        rmse_v = np.sqrt(np.mean((a_series - a_pred_v)**2))
        # Regression a ~ v^2
        coeff_v2 = np.polyfit(v2, a_series, deg=1)
        slope_v2, intercept_v2 = coeff_v2[0], coeff_v2[1]
        a_pred_v2 = np.polyval(coeff_v2, v2)
        r2_v2 = 1 - np.sum((a_series - a_pred_v2)**2) / np.sum((a_series - np.mean(a_series))**2)
        rmse_v2 = np.sqrt(np.mean((a_series - a_pred_v2)**2))
        result = {
            'eid': eid,
            'F_ext': float(exp['config'].get('F_ext', 0)),
            'a_v_slope': round(slope_v, 6),
            'a_v_intercept': round(intercept_v, 6),
            'a_v_r2': round(r2_v, 6),
            'a_v_rmse': round(rmse_v, 6),
            'a_v2_slope': round(slope_v2, 6),
            'a_v2_intercept': round(intercept_v2, 6),
            'a_v2_r2': round(r2_v2, 6),
            'a_v2_rmse': round(rmse_v2, 6),
        }
        const_results.append(result)
        obs = {
            'summary': f"{eid} (F_ext={result['F_ext']}): a~v回归: slope={result['a_v_slope']}, intercept={result['a_v_intercept']}, R²={result['a_v_r2']}, RMSE={result['a_v_rmse']}; a~v²回归: slope={result['a_v2_slope']}, intercept={result['a_v2_intercept']}, R²={result['a_v2_r2']}, RMSE={result['a_v2_rmse']}",
            'source_data_refs': [f"{eid}:a", f"{eid}:v"],
            'metrics': {
                'a_v_slope': result['a_v_slope'],
                'a_v_intercept': result['a_v_intercept'],
                'a_v_r2': result['a_v_r2'],
                'a_v_rmse': result['a_v_rmse'],
                'a_v2_slope': result['a_v2_slope'],
                'a_v2_intercept': result['a_v2_intercept'],
                'a_v2_r2': result['a_v2_r2'],
                'a_v2_rmse': result['a_v2_rmse'],
            }
        }
        observations.append(obs)

    # 3) Group by F_ext and report coefficient similarity
    if const_results:
        # Group by F_ext
        grouped = collections.defaultdict(list)
        for r in const_results:
            grouped[r['F_ext']].append(r)
        for f_ext in sorted(grouped.keys()):
            group = grouped[f_ext]
            # collect slopes and intercepts
            slopes_v = [r['a_v_slope'] for r in group]
            intercepts_v = [r['a_v_intercept'] for r in group]
            slopes_v2 = [r['a_v2_slope'] for r in group]
            intercepts_v2 = [r['a_v2_intercept'] for r in group]
            # compute mean and std
            mean_slope_v = np.mean(slopes_v)
            std_slope_v = np.std(slopes_v, ddof=1) if len(slopes_v)>1 else 0.0
            mean_intercept_v = np.mean(intercepts_v)
            std_intercept_v = np.std(intercepts_v, ddof=1) if len(intercepts_v)>1 else 0.0
            mean_slope_v2 = np.mean(slopes_v2)
            std_slope_v2 = np.std(slopes_v2, ddof=1) if len(slopes_v2)>1 else 0.0
            mean_intercept_v2 = np.mean(intercepts_v2)
            std_intercept_v2 = np.std(intercepts_v2, ddof=1) if len(intercepts_v2)>1 else 0.0
            obs = {
                'summary': f"F_ext={f_ext}组共{len(group)}个实验: a~v系数均值±标准差: slope={mean_slope_v:.6f}±{std_slope_v:.6f}, intercept={mean_intercept_v:.6f}±{std_intercept_v:.6f}; a~v²系数: slope={mean_slope_v2:.6f}±{std_slope_v2:.6f}, intercept={mean_intercept_v2:.6f}±{std_intercept_v2:.6f}",
                'source_data_refs': [f"{r['eid']}:a, {r['eid']}:v" for r in group],
                'metrics': {
                    'F_ext': f_ext,
                    'n_experiments': len(group),
                    'a_v_slope_mean': round(mean_slope_v, 6),
                    'a_v_slope_std': round(std_slope_v, 6),
                    'a_v_intercept_mean': round(mean_intercept_v, 6),
                    'a_v_intercept_std': round(std_intercept_v, 6),
                    'a_v2_slope_mean': round(mean_slope_v2, 6),
                    'a_v2_slope_std': round(std_slope_v2, 6),
                    'a_v2_intercept_mean': round(mean_intercept_v2, 6),
                    'a_v2_intercept_std': round(std_intercept_v2, 6),
                }
            }
            observations.append(obs)

    # 4) Plot a-v scatter for each F_ext value
    # Collect data per F_ext: list of (v, a, label)
    plot_data = collections.defaultdict(list)
    for eid in constant_force_exps:
        exp = experiments[eid]
        a_series = _get_series(exp, 'a')
        v_series = _get_series(exp, 'v')
        if a_series is None or v_series is None:
            continue
        # trim to same length
        n = min(len(a_series), len(v_series))
        a_ = a_series[:n]
        v_ = v_series[:n]
        F_ext = float(exp['config'].get('F_ext', 0))
        plot_data[F_ext].append((v_, a_, eid))

    for F_ext in sorted(plot_data.keys()):
        fig, ax = plt.subplots(figsize=(8, 6))
        for v_arr, a_arr, eid in plot_data[F_ext]:
            ax.scatter(v_arr, a_arr, label=eid, s=4, alpha=0.6)
        ax.set_xlabel('v')
        ax.set_ylabel('a')
        ax.set_title(f'a vs v for F_ext = {F_ext}')
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.tight_layout()
        fname = f"a_v_F_ext_{F_ext}.png"
        fig_path = output_dir / fname
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        figures.append(str(fig_path))
        obs = {
            'summary': f"F_ext={F_ext}的a-v散点图已保存，包含{len(plot_data[F_ext])}个实验。",
            'source_data_refs': [f"{eid}:a,{eid}:v" for _,_,eid in plot_data[F_ext]],
            'metrics': {'figure_path': str(fig_path), 'n_experiments': len(plot_data[F_ext])}
        }
        observations.append(obs)

    # 5) Free motion experiments (F_ext=0)
    if free_exps:
        free_summary_lines = []
        free_metrics = {}
        for eid in free_exps:
            exp = experiments[eid]
            a_series = _get_series(exp, 'a')
            v_series = _get_series(exp, 'v')
            if a_series is None or v_series is None:
                continue
            n = min(len(a_series), len(v_series))
            a_ = a_series[:n]
            v_ = v_series[:n]
            # Regression a ~ v
            coeff = np.polyfit(v_, a_, deg=1)
            slope, intercept = coeff[0], coeff[1]
            a_pred = np.polyval(coeff, v_)
            r2 = 1 - np.sum((a_ - a_pred)**2) / np.sum((a_ - np.mean(a_))**2) if np.std(a_)>0 else 0.0
            rmse = np.sqrt(np.mean((a_ - a_pred)**2))
            free_summary_lines.append(f"{eid}: a~v slope={slope:.6e}, intercept={intercept:.6e}, R²={r2:.6f}, RMSE={rmse:.6e}; a均值={np.mean(a_):.6e}, std={np.std(a_):.6e}")
            free_metrics[f"{eid}_slope"] = round(slope, 10)
            free_metrics[f"{eid}_intercept"] = round(intercept, 10)
            free_metrics[f"{eid}_r2"] = round(r2, 6)
            free_metrics[f"{eid}_rmse"] = round(rmse, 10)
            free_metrics[f"{eid}_a_mean"] = round(np.mean(a_), 10)
            free_metrics[f"{eid}_a_std"] = round(np.std(a_), 10)
        # Plot free motion a vs v
        fig, ax = plt.subplots(figsize=(8, 6))
        for eid in free_exps:
            exp = experiments[eid]
            a_ = _get_series(exp, 'a')
            v_ = _get_series(exp, 'v')
            if a_ is not None and v_ is not None:
                ax.scatter(v_, a_, label=eid, s=4)
        ax.set_xlabel('v')
        ax.set_ylabel('a')
        ax.set_title('Free motion (F_ext=0): a vs v')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fname = "a_v_free_motion.png"
        fig_path = output_dir / fname
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        figures.append(str(fig_path))
        free_summary = "\n".join(free_summary_lines)
        obs = {
            'summary': f"自由运动实验分析:\n{free_summary}",
            'source_data_refs': [f"{eid}:a,{eid}:v" for eid in free_exps],
            'metrics': free_metrics
        }
        observations.append(obs)

    # Build final return
    metrics_all['observation_count'] = len(observations)
    metrics_all['figure_count'] = len(figures)
    metrics_all['constant_force_experiments_processed'] = len(const_results)
    metrics_all['free_experiments_processed'] = len(free_exps)

    return {
        'observation': f"诊断完成: 对{len(exp_ids)}个实验进行了配置确认, {len(const_results)}个恒外力实验进行了a~v和a~v²回归, 按F_ext分组汇报了系数统计, 绘制了各组a-v散点图, 并对{len(free_exps)}个自由运动实验进行了分析。",
        'derived_series': derived_series,
        'observations': observations,
        'figures': figures,
        'metrics': metrics_all
    }
