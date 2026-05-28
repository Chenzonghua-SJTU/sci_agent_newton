import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn import metrics as skmetrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(experiments.keys())
    optional_series = params.get("optional_series", [])

    # 准备收集结果
    all_fits = []  # 每个实验拟合结果
    all_intercepts = []  # 截距
    all_F = []  # 对应的外力
    all_t = []  # 合并时间序列用于画图（内点时间）
    all_v2 = []
    all_F_over_a = []
    all_resid_ideal = []  # F_ext/a - (1+v^2) 内点
    all_F_for_alpha = []   # 每个点对应的F_ext
    all_exp_labels = []

    # 存储每个实验的内点信息，用于后续构建全长度派生序列
    exp_inner_info = []  # 每个元素: (eid, F_ext, n_total, idx, t_full, resid_ideal_inner)

    for eid in exp_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        config = exp.get("config", {})
        F_ext = config.get("F_ext", 0.0)
        series = exp.get("series", {})
        t_full = np.array(series.get("t", []))
        q = np.array(series.get("q", []))
        a = np.array(series.get("a_est_sg", []))
        v = np.array(series.get("v_est_sg", []))
        if len(a) == 0:
            a = np.array(series.get("a_est", []))
        if len(v) == 0:
            v = np.array(series.get("v_est", []))
        n = len(t_full)
        if n == 0:
            raise ValueError(f"Experiment {eid} has no time series.")
        if len(a) != n or len(v) != n:
            raise ValueError(f"Experiment {eid}: a_est_sg shape mismatch. Have {len(a)}, need {n}.")
        # 排除前2后2边界点
        if n <= 4:
            idx = np.arange(n)
        else:
            idx = np.arange(2, n-2)
        t_inner = t_full[idx]
        a_inner = a[idx]
        v_inner = v[idx]
        a_safe = np.where(np.abs(a_inner) < 1e-15, 1e-15, a_inner)
        F_over_a = F_ext / a_safe
        v2 = v_inner ** 2
        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(v2, F_over_a)
        R2 = r_value ** 2
        residuals = F_over_a - (intercept + slope * v2)
        N = len(v2)
        RMSE = np.sqrt(np.mean(residuals**2))
        MAE = np.mean(np.abs(residuals))
        max_abs_resid = np.max(np.abs(residuals))
        all_fits.append({
            "experiment_id": eid,
            "F_ext": F_ext,
            "n_points": N,
            "intercept": intercept,
            "slope": slope,
            "R2": R2,
            "RMSE": RMSE,
            "MAE": MAE,
            "max_abs_residual": max_abs_resid,
            "residuals": residuals.tolist()
        })
        all_intercepts.append(intercept)
        all_F.append(F_ext)
        all_t.extend(t_inner.tolist())
        all_v2.extend(v2.tolist())
        all_F_over_a.extend(F_over_a.tolist())
        all_exp_labels.extend([eid]*N)
        # 理想残差 (用于修正模型)
        ideal = 1 + v2
        resid_ideal = F_over_a - ideal
        all_resid_ideal.extend(resid_ideal.tolist())
        all_F_for_alpha.extend([F_ext]*N)

        # 存储内点信息，以便后续构建全长度序列
        exp_inner_info.append({
            "eid": eid,
            "F_ext": F_ext,
            "n_total": n,
            "idx": idx.copy(),
            "t_full": t_full.copy(),
            "resid_ideal_inner": resid_ideal.copy()
        })

    # 1. 截距与F_ext的线性关系
    F_arr = np.array(all_F)
    intercept_arr = np.array(all_intercepts)
    slope_int, intercept_int, r_int, p_int, std_err_int = stats.linregress(F_arr, intercept_arr)
    R2_int = r_int ** 2

    # 2. 修正模型: F_ext/a = 1 + v^2 + alpha * F_ext
    x_all = np.array(all_F_for_alpha)
    y_all = np.array(all_resid_ideal)
    # 无截距回归: y = alpha * x
    alpha_fit = np.sum(x_all * y_all) / np.sum(x_all**2)
    N_total = len(x_all)
    dof = N_total - 1
    residuals_alpha = y_all - alpha_fit * x_all
    MSE_alpha = np.sum(residuals_alpha**2) / dof
    se_alpha = np.sqrt(MSE_alpha / np.sum(x_all**2))
    corrected_residuals_inner = residuals_alpha  # 内点长度，顺序与all_resid_ideal一致

    # 分实验计算修正后残差（内点）和RMSE
    exp_corrected_rmse = {}
    exp_corrected_resid_map_inner = {}
    idx_start = 0
    for fit in all_fits:
        eid = fit["experiment_id"]
        N = fit["n_points"]
        seg_resid = corrected_residuals_inner[idx_start:idx_start+N]
        rmse_corr = np.sqrt(np.mean(seg_resid**2))
        exp_corrected_rmse[eid] = rmse_corr
        exp_corrected_resid_map_inner[eid] = seg_resid.copy()
        idx_start += N
    overall_rmse_correction = np.sqrt(np.mean(corrected_residuals_inner**2))

    # 构建修正后残差的派生序列（全长度，边界NaN）
    derived_series_list = []
    for info in exp_inner_info:
        eid = info["eid"]
        n_total = info["n_total"]
        idx = info["idx"]
        t_full = info["t_full"]
        # 从map中获取该实验的修正后残差内点
        if eid in exp_corrected_resid_map_inner:
            seg = exp_corrected_resid_map_inner[eid]
        else:
            seg = np.array([])
        # 构建全长度数组
        full = np.full(n_total, np.nan)
        if len(seg) > 0:
            full[idx] = seg
        derived_series_list.append({
            "experiment_id": eid,
            "name": "corrected_residual_H001_alpha",
            "values": full.tolist(),
            "source_name": "修正模型残差: F_ext/a - 1 - v^2 - alpha*F_ext",
            "provenance": "analyze_data: corrected model fit",
            "description": f"修正后残差 (alpha={alpha_fit:.6f}), 边界点填充NaN"
        })

    # 作图
    fig_paths = []
    # 图1: 每个实验的F_ext/a vs v^2拟合
    n_exp = len(all_fits)
    n_cols = min(4, n_exp)
    n_rows = (n_exp + n_cols - 1) // n_cols
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    if n_rows * n_cols == 1:
        axes1 = [axes1]
    else:
        axes1 = axes1.flatten()
    for i, fit in enumerate(all_fits):
        eid = fit["experiment_id"]
        ax = axes1[i]
        mask = [lab == eid for lab in all_exp_labels]
        x_plot = np.array(all_v2)[mask]
        y_plot = np.array(all_F_over_a)[mask]
        ax.scatter(x_plot, y_plot, s=1, alpha=0.6)
        x_line = np.linspace(x_plot.min(), x_plot.max(), 100)
        y_line = fit["intercept"] + fit["slope"] * x_line
        ax.plot(x_line, y_line, 'r-')
        ax.set_title(f"{eid} F_ext={fit['F_ext']}")
        ax.set_xlabel(r"$v^2$")
        ax.set_ylabel(r"$F_{ext}/a$")
    for j in range(i+1, len(axes1)):
        fig1.delaxes(axes1[j])
    fig1.tight_layout()
    path1 = output_dir / "all_F_over_a_vs_v2_fits.png"
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)
    fig_paths.append(str(path1))

    # 图2: 截距 vs F_ext
    fig2, ax2 = plt.subplots(figsize=(6,5))
    ax2.scatter(F_arr, intercept_arr)
    F_line = np.linspace(F_arr.min(), F_arr.max(), 50)
    ax2.plot(F_line, intercept_int + slope_int*F_line, 'r-')
    ax2.set_xlabel("F_ext")
    ax2.set_ylabel("Intercept of F_ext/a vs v^2")
    ax2.set_title(f"Intercept vs F_ext (R^2={R2_int:.4f})")
    path2 = output_dir / "intercept_vs_F_ext.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    fig_paths.append(str(path2))

    # 图3: 修正后残差 vs v^2 (整体)
    fig3, ax3 = plt.subplots(figsize=(6,5))
    # 用内点数据
    ax3.scatter(np.array(all_v2), corrected_residuals_inner, s=1, alpha=0.3)
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.set_xlabel(r"$v^2$")
    ax3.set_ylabel("Corrected residual")
    ax3.set_title("Corrected residual vs v^2")
    path3 = output_dir / "corrected_residual_vs_v2.png"
    fig3.savefig(path3, dpi=150)
    plt.close(fig3)
    fig_paths.append(str(path3))

    # 图4: 修正后残差 vs t (各实验分开，用内点时间)
    fig4, axes4 = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    if n_rows * n_cols == 1:
        axes4 = [axes4]
    else:
        axes4 = axes4.flatten()
    idx_start = 0
    for i, fit in enumerate(all_fits):
        eid = fit["experiment_id"]
        N = fit["n_points"]
        ax = axes4[i]
        mask = [lab == eid for lab in all_exp_labels]
        t_plot = np.array(all_t)[mask]
        corr_seg = corrected_residuals_inner[idx_start:idx_start+N]
        ax.plot(t_plot, corr_seg, '-', linewidth=0.5)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_title(eid)
        ax.set_xlabel("t")
        ax.set_ylabel("Corrected res.")
        idx_start += N
    for j in range(i+1, len(axes4)):
        fig4.delaxes(axes4[j])
    fig4.tight_layout()
    path4 = output_dir / "corrected_residual_vs_t.png"
    fig4.savefig(path4, dpi=150)
    plt.close(fig4)
    fig_paths.append(str(path4))

    # 构建metrics
    metrics = {
        "experiment_count": len(all_fits),
        "F_ext_intercept_linear": {
            "slope": slope_int,
            "intercept": intercept_int,
            "R2": R2_int,
            "p_value": p_int,
            "std_err_slope": std_err_int
        },
        "alpha_fit": {
            "alpha": alpha_fit,
            "std_error": se_alpha,
            "N_points": N_total
        },
        "overall_corrected_RMSE": overall_rmse_correction,
        "per_experiment_corrected_RMSE": exp_corrected_rmse,
        "F_ext_intercept_correlation": {
            "pearson_r": r_int,
            "p_value": p_int
        },
        "per_experiment_fits": all_fits
    }

    # 构建observation
    obs_lines = []
    obs_lines.append(f"处理了 {len(all_fits)} 个恒外力实验。")
    obs_lines.append("各实验F_ext/a vs v²线性拟合截距、斜率、R²:")
    for fit in all_fits:
        obs_lines.append(f"  {fit['experiment_id']}: F_ext={fit['F_ext']}, intercept={fit['intercept']:.6f}, slope={fit['slope']:.6f}, R²={fit['R2']:.8f}, RMSE={fit['RMSE']:.2e}")
    if len(all_intercepts) > 0:
        mean_int = np.mean(all_intercepts)
        std_int = np.std(all_intercepts, ddof=1)
        mean_sl = np.mean([f["slope"] for f in all_fits])
        std_sl = np.std([f["slope"] for f in all_fits], ddof=1)
        obs_lines.append(f"截距均值={mean_int:.6f}±{std_int:.6f}, 斜率均值={mean_sl:.6f}±{std_sl:.6f}")
    obs_lines.append(f"截距与F_ext线性回归: 斜率={slope_int:.6f}±{std_err_int:.6f}, R²={R2_int:.6f}")
    obs_lines.append(f"修正模型F_ext/a = 1+v²+α·F_ext, α={alpha_fit:.6f}±{se_alpha:.6f}")
    obs_lines.append(f"修正后整体RMSE={overall_rmse_correction:.2e}")
    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": fig_paths,
        "metrics": metrics
    }
