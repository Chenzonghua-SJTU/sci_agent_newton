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
import scipy.signal
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload.get("output_dir", ".")

    # 获取实验列表
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(experiments.keys())
    # 统一转换成 "exp_02" 格式
    exp_id_strs = [f"exp_{e:02d}" if isinstance(e, int) else e for e in exp_ids]

    # 全局收集数据：将所有实验的 (a, v, F_ext) 串联
    all_a = []
    all_v = []
    all_F_ext = []
    # 同时存储每个实验的索引范围，用于后续残差分实验统计
    exp_ranges = {}   # exp_id -> (start, end) 在全局数组中的切片

    global_a_list = []
    global_v_list = []
    global_F_list = []
    segment_starts = []
    current_start = 0

    # 检查所有实验的 t 长度是否一致，用于索引
    for eid in exp_id_strs:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        # 确保必需的 series 存在
        if "a" not in series or "v" not in series or "t" not in series:
            raise ValueError(f"Experiment {eid} missing required series a/v/t")
        a_vals = np.array(series["a"], dtype=float)
        v_vals = np.array(series["v"], dtype=float)
        F_ext = config["F_ext"]  # 使用 F_ext，不是 constant_force
        n = len(a_vals)
        # 追加
        global_a_list.append(a_vals)
        global_v_list.append(v_vals)
        global_F_list.append(np.full(n, F_ext))
        segment_starts.append((eid, current_start, current_start + n))
        current_start += n

    if len(global_a_list) == 0:
        raise ValueError("No valid experiments found")

    # 合并
    all_a = np.concatenate(global_a_list)
    all_v = np.concatenate(global_v_list)
    all_F = np.concatenate(global_F_list)
    n_total = len(all_a)

    # 构建设计矩阵 X = [F_ext, v, 1]
    X = np.column_stack([all_F, all_v, np.ones(n_total)])
    y = all_a

    # 全局 OLS
    # 手动计算 OLS 以获得标准误
    # X.T @ X 可能病态但应可逆
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta_global = XtX_inv @ (X.T @ y)
    residuals_global = y - X @ beta_global
    SS_res = np.sum(residuals_global ** 2)
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    R2_global = 1 - SS_res / SS_tot
    RMSE_global = np.sqrt(np.mean(residuals_global ** 2))
    n_params = 3
    sigma2 = SS_res / (n_total - n_params)
    cov_beta = sigma2 * XtX_inv
    se_global = np.sqrt(np.diag(cov_beta))

    # 残差统计
    res_mean = np.mean(residuals_global)
    res_std = np.std(residuals_global, ddof=0)  # population std
    res_max_abs = np.max(np.abs(residuals_global))

    # 各实验残差 RMSE 计算
    exp_res_rmse = {}
    for eid, start, end in segment_starts:
        exp_res = residuals_global[start:end]
        exp_res_rmse[eid] = float(np.sqrt(np.mean(exp_res ** 2)))

    # -----------------------------------------------------------------
    # 每个实验单独拟合
    single_results = []
    for eid, start, end in segment_starts:
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        a_e = np.array(series["a"], dtype=float)
        v_e = np.array(series["v"], dtype=float)
        F_e = config["F_ext"]
        n_e = len(a_e)
        # 设计矩阵 X_single = [F_e (常数), v_e, 1]
        X_single = np.column_stack([np.full(n_e, F_e), v_e, np.ones(n_e)])
        y_single = a_e
        # 使用 lstsq 处理可能的奇异
        beta_single, res_single, rank, s = np.linalg.lstsq(X_single, y_single, rcond=None)
        # 计算 R² 和 RMSE
        y_pred = X_single @ beta_single
        ss_res_single = np.sum((y_single - y_pred) ** 2)
        ss_tot_single = np.sum((y_single - np.mean(y_single)) ** 2)
        if ss_tot_single == 0:
            r2_single = 1.0 if ss_res_single == 0 else 0.0
        else:
            r2_single = 1 - ss_res_single / ss_tot_single
        rmse_single = np.sqrt(np.mean((y_single - y_pred) ** 2))
        # 系数偏差
        deviation = beta_single - beta_global
        # 记录条件数以检查共线性
        condition = np.linalg.cond(X_single)
        single_results.append({
            "experiment_id": eid,
            "p1": float(beta_single[0]),
            "p2": float(beta_single[1]),
            "p3": float(beta_single[2]),
            "R2": float(r2_single),
            "RMSE": float(rmse_single),
            "condition_number": float(condition),
            "deviation_p1": float(deviation[0]),
            "deviation_p2": float(deviation[1]),
            "deviation_p3": float(deviation[2])
        })

    # -----------------------------------------------------------------
    # 构建 observations 列表
    observations = []

    # 1. 全局回归系数及统计量
    global_obs = {
        "summary": (
            f"全局多元线性回归: a = p1*F_ext + p2*v + p3。"
            f"p1 = {beta_global[0]:.6f} (SE={se_global[0]:.6f}), "
            f"p2 = {beta_global[1]:.6f} (SE={se_global[1]:.6f}), "
            f"p3 = {beta_global[2]:.6f} (SE={se_global[2]:.6f}), "
            f"R² = {R2_global:.6f}, RMSE = {RMSE_global:.6f}, "
            f"样本数 = {n_total}"
        ),
        "source_data_refs": [f"{eid}:a" for eid, _, _ in segment_starts] + 
                            [f"{eid}:v" for eid, _, _ in segment_starts],
        "metrics": {
            "global_p1": float(beta_global[0]),
            "global_p1_se": float(se_global[0]),
            "global_p2": float(beta_global[1]),
            "global_p2_se": float(se_global[1]),
            "global_p3": float(beta_global[2]),
            "global_p3_se": float(se_global[2]),
            "global_R2": float(R2_global),
            "global_RMSE": float(RMSE_global),
            "n_total": int(n_total)
        }
    }
    observations.append(global_obs)

    # 2. 残差统计 (全局)
    res_obs = {
        "summary": (
            f"全局拟合残差统计: 均值 = {res_mean:.8f}, "
            f"标准差 = {res_std:.8f}, 最大绝对值 = {res_max_abs:.8f}"
        ),
        "source_data_refs": ["全局拟合"],
        "metrics": {
            "residual_mean": float(res_mean),
            "residual_std": float(res_std),
            "residual_max_abs": float(res_max_abs)
        }
    }
    observations.append(res_obs)

    # 3. 各实验残差 RMSE 列表（作为一个 observation）
    rmse_list_obs = {
        "summary": "各实验残差 RMSE 列表（按实验 ID 排序）。",
        "source_data_refs": [f"{eid}:residuals" for eid in exp_res_rmse.keys()],
        "metrics": {
            "experiment_residual_rmse": {k: round(v, 8) for k, v in exp_res_rmse.items()}
        }
    }
    observations.append(rmse_list_obs)

    # 4. 每个实验单独拟合结果（每个实验一条 observation）
    for res in single_results:
        eid = res["experiment_id"]
        obs_single = {
            "summary": (
                f"实验 {eid} 单独拟合: a = p1*F_ext + p2*v + p3。"
                f"p1 = {res['p1']:.6f}, p2 = {res['p2']:.6f}, p3 = {res['p3']:.6f}, "
                f"R² = {res['R2']:.6f}, RMSE = {res['RMSE']:.6f}, "
                f"条件数 = {res['condition_number']:.1f}. "
                f"与全局系数偏差: dp1 = {res['deviation_p1']:.6f}, "
                f"dp2 = {res['deviation_p2']:.6f}, dp3 = {res['deviation_p3']:.6f}."
            ),
            "source_data_refs": [f"{eid}:a", f"{eid}:v"],
            "metrics": {
                "p1": res["p1"],
                "p2": res["p2"],
                "p3": res["p3"],
                "R2": res["R2"],
                "RMSE": res["RMSE"],
                "condition_number": res["condition_number"],
                "deviation_p1": res["deviation_p1"],
                "deviation_p2": res["deviation_p2"],
                "deviation_p3": res["deviation_p3"]
            }
        }
        observations.append(obs_single)

    # 总体 observation 字符串（简短的概述）
    observation_summary = (
        f"对 {len(segment_starts)} 个实验进行了全局多元线性回归 a = p1*F_ext + p2*v + p3。"
        f"全局 R²={R2_global:.4f}, RMSE={RMSE_global:.4f}。"
        f"残差均值={res_mean:.6f}, 最大绝对值={res_max_abs:.4f}。"
        f"各实验单独拟合已完成，共产生 {len(observations)} 条 OBS。"
        "注意：单实验中由于 F_ext 为常数，p1 与 p3 存在共线性，系数可能不唯一（最小范数解）。"
    )

    # 可选：绘制全局残差分布图（不强制）
    fig_path = Path(output_dir) / "global_residuals_hist.png"
    plt.figure(figsize=(8, 4))
    plt.hist(residuals_global, bins=50, alpha=0.7)
    plt.xlabel("Residual (a - predicted)")
    plt.ylabel("Frequency")
    plt.title("Global Residual Distribution")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # 构建返回
    result = {
        "observation": observation_summary,
        "derived_series": [],   # 没有派生序列
        "observations": observations,
        "validations": [],
        "figures": [str(fig_path)],
        "metrics": {
            "global_R2": float(R2_global),
            "global_RMSE": float(RMSE_global),
            "residual_mean": float(res_mean),
            "residual_std": float(res_std),
            "residual_max_abs": float(res_max_abs),
            "experiment_count": len(segment_starts),
            "observation_count": len(observations)
        }
    }

    return result
