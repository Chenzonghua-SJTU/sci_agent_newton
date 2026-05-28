import json
import math
import statistics
import itertools
import functools
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
import sklearn.linear_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = Path(payload.get("output_dir", "."))
    
    # 获取实验ID列表
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        # 默认处理所有恒外力实验
        exp_ids = [eid for eid, exp in experiments.items()
                   if exp.get("config", {}).get("force_field_type") == "constant"]
    # 确保所有ID存在
    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload experiments")
    
    # 准备存储结果
    results = {}
    all_v = []
    all_a = []
    
    # 每个实验分析
    needed_series = {"a_new", "v_new"}
    for eid in exp_ids:
        exp = experiments[eid]
        config = exp.get("config", {})
        # 获取F_ext，优先使用F_ext，如果没有则使用constant_force
        F_ext = config.get("F_ext", config.get("constant_force", None))
        if F_ext is None:
            raise ValueError(f"Experiment {eid} has no F_ext or constant_force defined")
        F_ext = float(F_ext)
        
        series = exp.get("series", {})
        if not series:
            raise ValueError(f"Experiment {eid} has no series data")
        
        # 检查需要的序列是否存在
        missing = needed_series - set(series.keys())
        if missing:
            raise ValueError(f"Experiment {eid} missing series: {missing}")
        
        v = np.array(series["v_new"])
        a = np.array(series["a_new"])
        
        # 检查长度
        t = series.get("t", None)
        if t is None:
            raise ValueError(f"Experiment {eid} missing t series")
        n = len(t)
        if len(v) != n or len(a) != n:
            raise ValueError(f"Experiment {eid} series length mismatch")
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(v, a)
        alpha = float(slope)
        beta = float(intercept)
        R2 = float(r_value ** 2)
        
        # 预测和残差
        a_pred = alpha * v + beta
        residual = a - a_pred
        
        # 统计
        residual_mean = float(np.mean(residual))
        residual_std = float(np.std(residual, ddof=1))  # 样本标准差
        residual_min = float(np.min(residual))
        residual_max = float(np.max(residual))
        # 残差与v的相关系数
        corr_v_resid = float(np.corrcoef(v, residual)[0, 1])
        
        # 比较beta与F_ext
        beta_diff = beta - F_ext
        beta_diff_percent = (beta_diff / F_ext) * 100 if F_ext != 0 else 0.0
        
        results[eid] = {
            "alpha": alpha,
            "beta": beta,
            "R2": R2,
            "p_value": p_value,
            "std_err_slope": float(std_err),
            "beta_diff": beta_diff,
            "beta_diff_percent": beta_diff_percent,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "residual_min": residual_min,
            "residual_max": residual_max,
            "corr_v_residual": corr_v_resid,
        }
        
        all_v.extend(v.tolist())
        all_a.extend(a.tolist())
        
        # 保存残差序列（可选，在metrics中已经包含，也可以作为派生序列返回）
        # 但我们这里不返回派生序列，因为主要是统计量
    
    # 全局拟合
    all_v = np.array(all_v)
    all_a = np.array(all_a)
    global_slope, global_intercept, global_r, global_p, global_std = scipy.stats.linregress(all_v, all_a)
    global_alpha = float(global_slope)
    global_beta = float(global_intercept)
    global_R2 = float(global_r ** 2)
    global_a_pred = global_alpha * all_v + global_beta
    global_residual = all_a - global_a_pred
    global_rmse = float(np.sqrt(np.mean(global_residual ** 2)))
    global_residual_mean = float(np.mean(global_residual))
    global_residual_std = float(np.std(global_residual, ddof=1))
    
    # 全局残差与v的相关系数
    global_corr_v_resid = float(np.corrcoef(all_v, global_residual)[0, 1])
    
    # 构建metrics字典
    metrics = {}
    for eid, res in results.items():
        metrics[f"{eid}_alpha"] = res["alpha"]
        metrics[f"{eid}_beta"] = res["beta"]
        metrics[f"{eid}_R2"] = res["R2"]
        metrics[f"{eid}_p_value"] = res["p_value"]
        metrics[f"{eid}_std_err_slope"] = res["std_err_slope"]
        metrics[f"{eid}_beta_diff"] = res["beta_diff"]
        metrics[f"{eid}_beta_diff_percent"] = res["beta_diff_percent"]
        metrics[f"{eid}_residual_mean"] = res["residual_mean"]
        metrics[f"{eid}_residual_std"] = res["residual_std"]
        metrics[f"{eid}_residual_min"] = res["residual_min"]
        metrics[f"{eid}_residual_max"] = res["residual_max"]
        metrics[f"{eid}_corr_v_residual"] = res["corr_v_residual"]
    
    metrics["global_alpha"] = global_alpha
    metrics["global_beta"] = global_beta
    metrics["global_R2"] = global_R2
    metrics["global_RMSE"] = global_rmse
    metrics["global_residual_mean"] = global_residual_mean
    metrics["global_residual_std"] = global_residual_std
    metrics["global_corr_v_residual"] = global_corr_v_resid
    
    # 生成图形（可选）
    figures = []
    
    # 每个实验单独图
    for eid in exp_ids:
        exp = experiments[eid]
        config = exp.get("config", {})
        F_ext = float(config.get("F_ext", config.get("constant_force", 0)))
        v = np.array(exp["series"]["v_new"])
        a = np.array(exp["series"]["a_new"])
        res = results[eid]
        alpha = res["alpha"]
        beta = res["beta"]
        a_pred = alpha * v + beta
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(v, a, s=8, alpha=0.6, label="Data")
        ax.plot(v, a_pred, 'r-', linewidth=2, label=f"Fit: a={alpha:.4f}*v+{beta:.4f}")
        ax.set_xlabel("v_new")
        ax.set_ylabel("a_new")
        ax.set_title(f"Experiment {eid} (F_ext={F_ext}): a vs v linear fit\nR²={res['R2']:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = output_dir / f"{eid}_linear_fit_a_vs_v.png"
        fig.savefig(str(fig_path))
        plt.close(fig)
        figures.append(str(fig_path))
    
    # 全局拟合图
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for idx, eid in enumerate(exp_ids):
        exp = experiments[eid]
        v = np.array(exp["series"]["v_new"])
        a = np.array(exp["series"]["a_new"])
        ax.scatter(v, a, s=6, alpha=0.6, color=colors[idx % len(colors)], label=eid)
    # 全局拟合线
    v_sorted = np.sort(all_v)
    a_global_fit = global_alpha * v_sorted + global_beta
    ax.plot(v_sorted, a_global_fit, 'k-', linewidth=2, label=f"Global: a={global_alpha:.4f}*v+{global_beta:.4f}, R²={global_R2:.4f}")
    ax.set_xlabel("v_new")
    ax.set_ylabel("a_new")
    ax.set_title("Global linear fit across all constant-force experiments")
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    global_fig_path = output_dir / "global_linear_fit_a_vs_v.png"
    fig.savefig(str(global_fig_path))
    plt.close(fig)
    figures.append(str(global_fig_path))
    
    # 残差图（每个实验的残差 vs v）
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for i, eid in enumerate(exp_ids):
        exp = experiments[eid]
        v = np.array(exp["series"]["v_new"])
        a = np.array(exp["series"]["a_new"])
        res = results[eid]
        a_pred = res["alpha"] * v + res["beta"]
        residual = a - a_pred
        ax = axes[i]
        ax.scatter(v, residual, s=10, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax.set_xlabel("v_new")
        ax.set_ylabel("Residual")
        ax.set_title(f"{eid}: Residual (mean={res['residual_mean']:.4f})")
        ax.grid(True, alpha=0.3)
    if len(exp_ids) < 6:
        for j in range(len(exp_ids), 6):
            axes[j].axis('off')
    fig.suptitle("Per-experiment residuals vs v", fontsize=14)
    fig.tight_layout()
    resid_fig_path = output_dir / "residuals_vs_v.png"
    fig.savefig(str(resid_fig_path))
    plt.close(fig)
    figures.append(str(resid_fig_path))
    
    # 构建observation
    lines = []
    lines.append(f"对 {len(exp_ids)} 个恒外力实验进行了线性回归 a = alpha * v + beta。")
    lines.append("各实验拟合结果:")
    for eid in exp_ids:
        res = results[eid]
        exp = experiments[eid]
        F_ext = float(exp["config"].get("F_ext", exp["config"].get("constant_force", 0)))
        lines.append(f"  {eid} (F_ext={F_ext}): alpha={res['alpha']:.4f}, beta={res['beta']:.4f}, R²={res['R2']:.4f}, beta-F_ext={res['beta_diff']:.4f} ({res['beta_diff_percent']:.2f}%), 残差标准差={res['residual_std']:.4f}")
    lines.append(f"全局线性拟合: alpha={global_alpha:.4f}, beta={global_beta:.4f}, R²={global_R2:.4f}, RMSE={global_rmse:.4f}")
    lines.append("各实验残差统计:")
    for eid in exp_ids:
        res = results[eid]
        lines.append(f"  {eid}: mean={res['residual_mean']:.4f}, std={res['residual_std']:.4f}, min={res['residual_min']:.4f}, max={res['residual_max']:.4f}, corr(v,res)={res['corr_v_residual']:.4f}")
    lines.append(f"全局残差与v相关系数: {global_corr_v_resid:.4f}")
    lines.append("图像已保存: 每个实验的a vs v拟合图、全局拟合图、残差分布图。")
    
    observation = "\n".join(lines)
    
    # 返回
    result = {
        "observation": observation,
        "derived_series": [],  # 没有返回新的序列
        "figures": figures,
        "metrics": metrics
    }
    return result
