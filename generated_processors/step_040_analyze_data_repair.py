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

        # 辅助函数：在series中查找匹配的序列键（支持实验后缀）
        def find_series(base_name, available_list, series_dict):
            # 首先直接完整匹配
            if base_name in series_dict:
                return series_dict[base_name]
            # 然后尝试查找以 base_name 开头的可用序列
            candidates = [s for s in available_list if s.startswith(base_name)]
            if candidates:
                # 取第一个
                key = candidates[0]
                return series_dict.get(key, None)
            # 最后尝试查找减掉实验编号的版本（例如 acceleration_central 可能直接存在）
            if base_name in available_list:
                return series_dict.get(base_name, None)
            return None

        # 选择加速序列
        a_series = find_series(acc_name, available, series)
        if a_series is None:
            missing_series_info.append(f"{eid}: missing acceleration series")
            continue

        # 选择速度序列
        v_series = find_series(vel_name, available, series)
        if v_series is None:
            missing_series_info.append(f"{eid}: missing velocity series")
            continue

        # 确保序列为列表
        a_vals = np.array(a_series, dtype=float)
        v_vals = np.array(v_series, dtype=float)

        # 检查长度一致性（使用t或a的长度）
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

    # 参数标准误（使用伪逆处理可能的奇异矩阵）
    XtX = X_design.T @ X_design
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # 加入小正则
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(XtX.shape[0]))
    cov_b = MSE * XtX_inv
    se_alpha = np.sqrt(cov_b[0, 0])
    se_beta_neg = np.sqrt(cov_b[1, 1])
    se_beta = se_beta_neg

    # 每个实验的 RMSE 和 R²
    per_experiment = []
    for eid, data in forced_experiment_data.items():
        F_ext = data["F_ext"]
        a_e = data["a"]
        v_e = data["v"]
        n_e = len(a_e)
        # 全局模型预测
        pred_e = alpha * F_ext - beta * v_e
        residuals_e = a_e - pred_e
        rmse_e = np.sqrt(np.mean(residuals_e ** 2))
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
        config = exp.get("config", {})
        F_ext = config.get("F_ext", 0.0)
        series = exp.get("series", {})
        available = exp.get("available_series", [])
        # 查找加速度序列
        a_vals = find_series(acc_name, available, series)
        if a_vals is None:
            free_checks.append({
                "experiment": eid,
                "F_ext": F_ext,
                "mean_acceleration": None,
                "max_abs_acceleration": None,
                "is_zero": False,
                "error": "missing acceleration series"
            })
            continue
        a_vals = np.array(a_vals, dtype=float)
        mean_acc = np.mean(a_vals)
        max_abs_acc = np.max(np.abs(a_vals))
        is_zero = (max_abs_acc < 1e-10)
        free_checks.append({
            "experiment": eid,
            "F_ext": F_ext,
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
    for pe in per_experiment:
        obs_lines.append(f"  {pe['experiment']}: F_ext={pe['F_ext']}, RMSE={pe['RMSE']:.6f}, R²={pe['R2']:.4f}, n={pe['n_points']}")
    obs_lines.append("自由实验检查:")
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
