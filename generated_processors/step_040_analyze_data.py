import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from scipy import stats, signal, optimize
from sklearn import linear_model, metrics, preprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    # 提取参数
    parameters = payload.get("parameters", {})
    analysis_goal = parameters.get("analysis_goal", "")
    experiment_ids = parameters.get("experiment_ids", None)
    optional_series = parameters.get("optional_series", {})
    expected_outputs = parameters.get("expected_outputs", [])

    # 如果没有指定experiment_ids，处理所有实验
    experiments = payload.get("experiments", {})
    if experiment_ids is None:
        experiment_ids = list(experiments.keys())

    # 配置序列名称
    acc_name = optional_series.get("acceleration", "acceleration_central")
    vel_name = optional_series.get("velocity", "velocity_central")

    # 分离恒外力实验和自由实验
    forced_ids = []
    free_ids = []
    for eid in experiment_ids:
        if eid not in experiments:
            continue
        config = experiments[eid].get("config", {})
        F_ext = config.get("F_ext", 0.0)
        if abs(F_ext) > 1e-12:
            forced_ids.append(eid)
        else:
            free_ids.append(eid)

    # 收集恒外力实验数据
    X_list = []
    y_list = []
    forced_experiment_data = {}
    missing_series_info = []

    for eid in forced_ids:
        exp = experiments[eid]
        config = exp.get("config", {})
        F_ext = config.get("F_ext", 0.0)
        series = exp.get("series", {})
        available = exp.get("available_series", [])

        # 选择加速序列
        a_series = series.get(acc_name, None)
        if a_series is None and "acceleration" in available:
            a_series = series.get("acceleration")
        if a_series is None and "acceleration_central" in available:
            a_series = series.get("acceleration_central")
        if a_series is None:
            missing_series_info.append(f"{eid}: missing acceleration series")
            continue

        # 选择速度序列
        v_series = series.get(vel_name, None)
        if v_series is None and "velocity" in available:
            v_series = series.get("velocity")
        if v_series is None and "velocity_central" in available:
            v_series = series.get("velocity_central")
        if v_series is None:
            missing_series_info.append(f"{eid}: missing velocity series")
            continue

        # 确保序列为列表
        a_vals = np.array(a_series, dtype=float)
        v_vals = np.array(v_series, dtype=float)

        # 检查长度一致性
        t_series = series.get("t", [])
        if len(t_series) == 0:
            n_points = len(a_vals)
        else:
            n_points = len(t_series)
        if len(a_vals) != n_points or len(v_vals) != n_points:
            missing_series_info.append(f"{eid}: length mismatch t({n_points}), a({len(a_vals)}), v({len(v_vals)})")
            continue

        # 收集数据
        F_arr = np.full(n_points, F_ext)
        X_feat = np.column_stack([F_arr, v_vals])
        X_list.append(X_feat)
        y_list.append(a_vals)

        forced_experiment_data[eid] = {
            "F_ext": F_ext,
            "a": a_vals,
            "v": v_vals,
            "n_points": n_points
        }

    if len(X_list) == 0:
        raise ValueError("No forced experiments with complete data available for fitting.")

    # 构建全局设计矩阵和目标向量
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    # 多元线性回归： a = alpha * F_ext - beta * v  =>  a = [F_ext, v] @ [alpha, -beta].T
    # 设计矩阵： [F_ext, v]，对应参数 [alpha, -beta]
    # 使用最小二乘法
    n_total = X_all.shape[0]
    # 添加截距? 模型没有截距项，a = alpha*F_ext - beta*v，没有常数项
    # 拟合无截距的多元回归
    X_design = X_all  # 形状 (n, 2)
    # 使用np.linalg.lstsq
    coeff, residuals, rank, s = np.linalg.lstsq(X_design, y_all, rcond=None)
    alpha = coeff[0]
    beta_neg = coeff[1]  # 这是 -beta
    beta = -beta_neg

    # 计算残差
    y_pred = X_design @ coeff
    residuals_vec = y_all - y_pred
    SSE = np.sum(residuals_vec ** 2)
    MSE = SSE / (n_total - 2)  # 两个参数
    global_RMSE = np.sqrt(MSE)

    # 参数标准误
    XtX_inv = np.linalg.inv(X_design.T @ X_design)
    cov_b = MSE * XtX_inv
    se_alpha = np.sqrt(cov_b[0, 0])
    se_beta_neg = np.sqrt(cov_b[1, 1])
    beta_neg = coeff[1]
    # beta = -beta_neg，标准误相同（绝对值）
    se_beta = se_beta_neg

    # 每个实验的 RMSE 和 R²
    per_experiment = []
    for eid, data in forced_experiment_data.items():
        F_ext = data["F_ext"]
        a_e = data["a"]
        v_e = data["v"]
        n_e = len(a_e)
        # 全局模型预测
        pred_e = alpha * F_ext - beta * v_e  # 等价于 [F_ext, v] @ [alpha, -beta]
        # 或者使用 coeff: pred_e = X_design_e @ coeff
        residuals_e = a_e - pred_e
        rmse_e = np.sqrt(np.mean(residuals_e ** 2))
        # R² 相对于该实验的均值（没有中心化？通常R² = 1 - SS_res/SS_tot，SS_tot = sum((y - y_mean)^2)
        y_mean_e = np.mean(a_e)
        SS_tot_e = np.sum((a_e - y_mean_e) ** 2)
        SS_res_e = np.sum(residuals_e ** 2)
        if SS_tot_e > 1e-12:
            r2_e = 1 - SS_res_e / SS_tot_e
        else:
            r2_e = 0.0
        per_experiment.append({
            "experiment": eid,
            "F_ext": F_ext,
            "n_points": n_e,
            "RMSE": float(rmse_e),
            "R2": float(r2_e)
        })

    # 自由实验检查
    free_checks = []
    for eid in free_ids:
        exp = experiments[eid]
        series = exp.get("series", {})
        available = exp.get("available_series", [])
        a_series = series.get(acc_name, None)
        if a_series is None and "acceleration" in available:
            a_series = series.get("acceleration")
        if a_series is None and "acceleration_central" in available:
            a_series = series.get("acceleration_central")
        if a_series is None:
            free_checks.append({
                "experiment": eid,
                "mean_acceleration": None,
                "max_abs_acceleration": None,
                "is_zero": False,
                "error": "missing acceleration series"
            })
            continue
        a_vals = np.array(a_series, dtype=float)
        mean_acc = np.mean(a_vals)
        max_abs_acc = np.max(np.abs(a_vals))
        is_zero = (max_abs_acc < 1e-10)
        free_checks.append({
            "experiment": eid,
            "mean_acceleration": float(mean_acc),
            "max_abs_acceleration": float(max_abs_acc),
            "is_zero": bool(is_zero)
        })

    # 构造输出
    metrics = {
        "alpha": float(alpha),
        "alpha_std": float(se_alpha),
        "beta": float(beta),
        "beta_std": float(se_beta),
        "global_RMSE": float(global_RMSE),
        "per_experiment": per_experiment,
        "free_experiment_checks": free_checks,
    }

    # 生成 observation
    obs_lines = [
        f"全局线性模型 a = alpha * F_ext - beta * v 拟合完成。",
        f"alpha = {alpha:.6f} ± {se_alpha:.6f}",
        f"beta = {beta:.6f} ± {se_beta:.6f}",
        f"全局 RMSE = {global_RMSE:.6f}",
        f"参与拟合的恒外力实验数: {len(forced_experiment_data)}, 总数据点: {n_total}",
    ]
    # 每个实验 RMSE/R²
    for pe in per_experiment:
        obs_lines.append(f"  {pe['experiment']}: RMSE={pe['RMSE']:.6f}, R²={pe['R2']:.4f}")
    # 自由实验
    obs_lines.append(f"自由实验检查:")
    for fc in free_checks:
        if "error" in fc:
            obs_lines.append(f"  {fc['experiment']}: {fc['error']}")
        else:
            obs_lines.append(f"  {fc['experiment']}: mean_acc={fc['mean_acceleration']:.2e}, max_abs={fc['max_abs_acceleration']:.2e}, is_zero={fc['is_zero']}")
    if missing_series_info:
        obs_lines.append(f"警告: {len(missing_series_info)} 个实验因序列缺失被跳过: {missing_series_info}")

    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "metrics": metrics,
        "derived_series": [],
        "figures": []
    }
