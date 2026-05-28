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
import scipy
import scipy.optimize
import scipy.stats
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = Path(payload.get("output_dir", "."))

    # 解析参数
    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        experiment_ids = list(experiments.keys())
    
    # 收集数据
    all_v = []
    all_a = []
    all_F = []
    per_exp_data = {}  # exp_id -> (v, a, F_ext)
    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"实验 {eid} 不在 payload 中")
        exp = experiments[eid]
        config = exp["config"]
        # 获取 F_ext，优先使用 fields 中的 F_ext，否则 constant_force 或 fallback 0
        F_ext = config.get("F_ext", None)
        if F_ext is None:
            F_ext = config.get("constant_force", 0.0)
        # 获取 v_new 和 a_new
        series = exp["series"]
        if "v_new" not in series or "a_new" not in series:
            raise ValueError(f"实验 {eid} 缺少 v_new 或 a_new 序列")
        v = np.array(series["v_new"], dtype=float)
        a = np.array(series["a_new"], dtype=float)
        if len(v) == 0 or len(a) == 0:
            raise ValueError(f"实验 {eid} 的 v_new 或 a_new 为空")
        if len(v) != len(a):
            raise ValueError(f"实验 {eid} 的 v_new 和 a_new 长度不一致")
        all_v.extend(v.tolist())
        all_a.extend(a.tolist())
        all_F.extend([F_ext] * len(v))
        per_exp_data[eid] = (v, a, F_ext)
    
    v_arr = np.array(all_v)
    a_arr = np.array(all_a)
    F_arr = np.array(all_F)

    # 定义模型：a = F_ext * exp(-b * v)
    def model_func(xdata, b):
        # xdata: (N,2)  -> v, F_ext
        v = xdata[:, 0]
        F = xdata[:, 1]
        return F * np.exp(-b * v)

    # 构造 xdata: N x 2
    xdata = np.column_stack((v_arr, F_arr))
    ydata = a_arr

    # 非线性最小二乘拟合
    try:
        popt, pcov = scipy.optimize.curve_fit(model_func, xdata, ydata,
                                              p0=[0.7], bounds=([0], [10]),
                                              method='trf', maxfev=5000)
        b_fitted = popt[0]
        b_stderr = np.sqrt(pcov[0, 0]) if pcov.ndim == 2 else float('nan')
    except Exception as e:
        raise ValueError(f"全局指数模型拟合失败: {e}")

    # 计算拟合值及残差
    fitted = model_func(xdata, b_fitted)
    residuals = ydata - fitted

    # 全局 R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    global_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # 每个实验的残差统计
    per_exp_resid_stats = {}
    for eid, (v_exp, a_exp, F_exp) in per_exp_data.items():
        # 从总残差中筛选对应的索引
        # 由于数据顺序是按 experiment_ids 顺序拼接的，我们需要知道每个实验在总数组中的索引区间
        # 其实可以重新计算残差，避免索引混乱
        fitted_exp = F_exp * np.exp(-b_fitted * v_exp)
        resid_exp = a_exp - fitted_exp
        mean_res = float(np.mean(resid_exp))
        std_res = float(np.std(resid_exp, ddof=1) if len(resid_exp) > 1 else 0.0)
        min_res = float(np.min(resid_exp))
        max_res = float(np.max(resid_exp))
        # 检查残差是否与 v 有线性趋势
        corr_v_res = float(np.corrcoef(v_exp, resid_exp)[0, 1]) if len(v_exp) > 1 else 0.0
        per_exp_resid_stats[eid] = {
            "residual_mean": mean_res,
            "residual_std": std_res,
            "residual_min": min_res,
            "residual_max": max_res,
            "corr_v_residual": corr_v_res
        }

    # 绘制残差分布图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    ax2 = axes[1]

    # 散点图：残差 vs v_new，不同实验不同颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', 'D', '^', 'v']
    idx_start = 0
    for i, eid in enumerate(experiment_ids):
        if eid not in per_exp_data:
            continue
        v_exp, a_exp, F_exp = per_exp_data[eid]
        n = len(v_exp)
        resid_chunk = residuals[idx_start: idx_start + n]
        ax1.scatter(v_exp, resid_chunk, c=colors[i % len(colors)],
                   marker=markers[i % len(markers)], label=eid, alpha=0.6, s=10)
        idx_start += n
    ax1.axhline(y=0, color='gray', linestyle='--', lw=1)
    ax1.set_xlabel('v_new')
    ax1.set_ylabel('Residual')
    ax1.set_title(f'Global model residuals (b={b_fitted:.4f})')
    ax1.legend(fontsize=8)

    # 直方图：残差分布
    ax2.hist(residuals, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Density')
    ax2.set_title('Global residual distribution')
    # 叠加正态分布参考线
    mu = np.mean(residuals)
    sigma = np.std(residuals, ddof=1)
    x_plot = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    ax2.plot(x_plot, scipy.stats.norm.pdf(x_plot, mu, sigma), 'r-', lw=2, label=f'Normal fit (μ={mu:.3f}, σ={sigma:.3f})')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    resid_plot_path = output_dir / "global_exp_residual_diagnostics.png"
    fig.savefig(str(resid_plot_path), dpi=150)
    plt.close(fig)

    # 生成观察字符串
    R2_str = f"{global_r2:.4f}"
    b_str = f"{b_fitted:.4f} ± {b_stderr:.4f}"
    obs_lines = [
        f"全局指数模型拟合完成，使用实验 {experiment_ids} 的所有数据点。",
        f"拟合参数 b = {b_str}，全局 R² = {R2_str}。",
        f"残差均方根 (RMSE) = {np.sqrt(np.mean(residuals**2)):.4f}。",
    ]
    obs_lines.append("各实验残差统计:")
    for eid, stats in per_exp_resid_stats.items():
        obs_lines.append(
            f"  {eid}: 均值 {stats['residual_mean']:.4f}, 标准差 {stats['residual_std']:.4f}, "
            f"范围 [{stats['residual_min']:.4f}, {stats['residual_max']:.4f}], "
            f"残差与v相关系数 {stats['corr_v_residual']:.4f}"
        )
    
    # 检查系统趋势
    # 计算残差与v的总体相关系数
    overall_corr = float(np.corrcoef(v_arr, residuals)[0, 1]) if len(v_arr) > 1 else 0.0
    obs_lines.append(f"全部数据残差与v的 Pearson 相关系数: {overall_corr:.4f}")
    if abs(overall_corr) > 0.3:
        obs_lines.append("残差随 v 有较明显的线性趋势，提示指数模型可能不是最佳形式。")
        obs_lines.append("建议考虑修正形式：")
        obs_lines.append("  1) 增加线性阻尼项: a = F_ext * exp(-b*v) - k*v")
        obs_lines.append("  2) 改用幂律阻尼: a = F_ext - beta * v^gamma")
        obs_lines.append("  3) 每个实验单独拟合指数参数b（已证明b不恒定）")
    else:
        obs_lines.append("残差无明显线性趋势，指数模型在该组合数据上基本合理。")
    
    observation = "\n".join(obs_lines)

    # 组装 metrics
    metrics = {
        "b_value": float(b_fitted),
        "b_stderr": float(b_stderr),
        "global_R2": float(global_r2),
        "global_RMSE": float(np.sqrt(np.mean(residuals**2))),
        "residual_overall_corr_with_v": overall_corr,
    }
    for eid, stats in per_exp_resid_stats.items():
        for k, v in stats.items():
            metrics[f"{eid}_{k}"] = v

    # 返回
    return {
        "observation": observation,
        "derived_series": [],  # 本分析不产生新序列
        "figures": [str(resid_plot_path)],
        "metrics": metrics
    }
