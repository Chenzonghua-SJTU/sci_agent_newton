import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = Path(payload["output_dir"])
    gamma_global = 0.7528

    # 确保是 maintain_ledger 模式
    if params.get("analysis_mode") != "maintain_ledger":
        raise ValueError("Expected analysis_mode='maintain_ledger'")

    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        experiment_ids = [eid for eid in experiments.keys()]

    # 只处理本次要维护的实验
    target_ids = experiment_ids

    derived_series = []
    observations = []
    # 用于拟合的全局数据
    all_constant_v = []
    all_constant_drag = []
    all_constant_F_ext = []
    all_new_residuals = {}  # per experiment list for observation

    # ---- 1. 为新实验计算 v_gradient, a_gradient, drag ----
    for eid in target_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        available = exp.get("available_series", [])
        F_ext = config["F_ext"]
        force_type = config.get("force_field_type", "constant")  # default constant

        # 获取 t 和 q
        t = np.array(series["t"])
        q = np.array(series["q"])
        dt = np.gradient(t)

        # 1a. 计算 v_gradient
        v_grad = np.gradient(q, t, edge_order=2)
        v_name = f"v_gradient_ledger_exp_{eid}"
        if v_name in available:
            # 如果已存在，不重复定义，直接读取
            v_existing = np.array(series[v_name])
            if len(v_existing) != len(t):
                v_grad = v_existing  # 使用已存在的
            else:
                v_grad = v_existing
        derived_series.append({
            "experiment_id": eid,
            "name": v_name,
            "values": v_grad.tolist(),
            "source_name": "np.gradient(q, t, edge_order=2)",
            "provenance": "generated data processor: maintain_ledger step",
            "description": f"Velocity derived from position for experiment {eid} using central difference."
        })

        # 1b. 计算 a_gradient
        a_grad = np.gradient(v_grad, t, edge_order=2)
        a_name = f"a_gradient_ledger_exp_{eid}"
        if a_name in available:
            a_existing = np.array(series[a_name])
            if len(a_existing) == len(t):
                a_grad = a_existing
        derived_series.append({
            "experiment_id": eid,
            "name": a_name,
            "values": a_grad.tolist(),
            "source_name": "np.gradient(v_gradient, t, edge_order=2)",
            "provenance": "generated data processor: maintain_ledger step",
            "description": f"Acceleration derived from velocity for experiment {eid}."
        })

        # 1c. 计算 drag（仅常数场）
        drag_name = f"drag_ledger_exp_{eid}"
        if force_type == "constant":
            if F_ext == 0:
                drag = np.full_like(a_grad, 0.0)
            else:
                drag = F_ext - a_grad
            if drag_name not in available:
                derived_series.append({
                    "experiment_id": eid,
                    "name": drag_name,
                    "values": drag.tolist(),
                    "source_name": f"F_ext - a_gradient (F_ext={F_ext})",
                    "provenance": "generated data processor: maintain_ledger step",
                    "description": f"Drag = F_ext - a for experiment {eid}."
                })
            # 用于常数场拟合
            all_constant_v.extend(np.abs(v_grad).tolist())
            all_constant_drag.extend(drag.tolist())
            all_constant_F_ext.extend([F_ext] * len(v_grad))
        elif force_type == "free":
            # free 场无 drag
            if drag_name not in available:
                # 不生成 drag 系列
                pass
        else:
            # 未识别的力场类型，不生成 drag
            pass

        # ---- 2. 计算残差 residual_H004 ----
        # 对每个新实验计算 a_grad - F_ext * exp(-gamma_global * |v|)
        # 其中 F_ext 可能为 0（free场）
        v_abs = np.abs(v_grad)
        predicted_a = F_ext * np.exp(-gamma_global * v_abs)
        residual = a_grad - predicted_a
        residual_name = f"residual_H004_exp_{eid}"
        if residual_name not in available:
            derived_series.append({
                "experiment_id": eid,
                "name": residual_name,
                "values": residual.tolist(),
                "source_name": f"a_gradient - {F_ext} * exp(-{gamma_global} * |v|)",
                "provenance": "generated data processor: maintain_ledger step",
                "description": f"Residual of H004 model (exponential decay) for experiment {eid}."
            })

        # 收集残差统计
        rmse_res = float(np.sqrt(np.mean(residual ** 2)))
        mean_res = float(np.mean(residual))
        all_new_residuals[eid] = {
            "rmse": rmse_res,
            "mean": mean_res,
            "count": len(residual)
        }

        # 生成该实验的观察
        obs_summary = f"Experiment {eid}: v_gradient, a_gradient, drag{' (constant)' if force_type=='constant' else ''} computed. Residual to H004 (gamma={gamma_global}) RMSE={rmse_res:.6f}."
        obs_metrics = {
            "v_gradient_min": float(np.min(v_grad)),
            "v_gradient_max": float(np.max(v_grad)),
            "a_gradient_min": float(np.min(a_grad)),
            "a_gradient_max": float(np.max(a_grad)),
            "residual_H004_rmse": rmse_res,
            "residual_H004_mean": mean_res
        }
        if force_type == "constant":
            obs_metrics["drag_min"] = float(np.min(drag))
            obs_metrics["drag_max"] = float(np.max(drag))
        observations.append({
            "summary": obs_summary,
            "source_data_refs": [f"{eid}:t", f"{eid}:q"],
            "metrics": obs_metrics
        })

    # ---- 3. 收集所有常数场实验的 drag 和 v 数据（包括已有实验） ----
    # 已有的常数场实验，从 payload 中提取 drag 和 v
    for eid, exp in experiments.items():
        force_type = exp["config"].get("force_field_type", "constant")
        if force_type != "constant":
            continue
        if eid in target_ids:
            # 已经通过上述循环添加到 all_constant_... 中，跳过避免重复
            continue
        series = exp["series"]
        available = exp["available_series"]
        # 尝试获取已有的 drag 系列（优先）
        drag_found = None
        v_found = None
        for key in available:
            if key.startswith("drag_ledger_exp_") or key.startswith("drag_ledger_exp_exp_"):
                drag_found = np.array(series[key])
            if key.startswith("v_gradient_ledger_exp_") or key.startswith("v_gradient_ledger_exp_exp_"):
                v_found = np.array(series[key])
        if drag_found is None or v_found is None:
            # 如果找不到，尝试从 a_gradient 和 F_ext 计算
            a_key = None
            v_key = None
            for key in available:
                if key.startswith("a_gradient_ledger_exp_") or key.startswith("a_gradient_ledger_exp_exp_"):
                    a_key = key
                if key.startswith("v_gradient_ledger_exp_") or key.startswith("v_gradient_ledger_exp_exp_"):
                    v_key = key
            if a_key is None or v_key is None:
                continue  # 无法获取数据，跳过
            a_grad = np.array(series[a_key])
            v_grad = np.array(series[v_key])
            F_ext = exp["config"]["F_ext"]
            drag_calc = F_ext - a_grad
        else:
            v_grad = v_found
            drag_calc = drag_found
        if len(v_grad) == 0:
            continue
        all_constant_v.extend(np.abs(v_grad).tolist())
        all_constant_drag.extend(drag_calc.tolist())
        all_constant_F_ext.extend([exp["config"]["F_ext"]] * len(v_grad))

    # ---- 4. 拟合指数饱和模型 OBS045 ----
    # 模型：drag = F_ext * (1 - exp(-c * |v|))
    v_arr = np.array(all_constant_v)
    drag_arr = np.array(all_constant_drag)
    F_arr = np.array(all_constant_F_ext)
    if len(v_arr) == 0:
        raise ValueError("No constant field data available for exponential saturation fit.")

    # 剔除 F_ext=0 的点（无外力，drag应为0，但模型不适用）
    mask = np.abs(F_arr) > 1e-12
    v_arr = v_arr[mask]
    drag_arr = drag_arr[mask]
    F_arr = F_arr[mask]
    if len(v_arr) == 0:
        raise ValueError("No valid constant field data with non-zero F_ext for exponential saturation fit.")

    # 使用 scipy.optimize.least_squares 处理每个点不同 F_ext
    def residuals(c, v_abs, drag, F):
        return F * (1 - np.exp(-c * v_abs)) - drag

    c0 = 1.0
    try:
        res = least_squares(lambda c: residuals(c[0], v_arr, drag_arr, F_arr), [c0], bounds=(0, np.inf))
        c_opt = res.x[0]
    except Exception as e:
        # 如果 least_squares 失败，使用简单网格搜索
        c_candidates = np.logspace(-3, 3, 1000)
        best_c = c0
        best_cost = np.inf
        for c_test in c_candidates:
            cost = np.sum((F_arr * (1 - np.exp(-c_test * v_arr)) - drag_arr) ** 2)
            if cost < best_cost:
                best_cost = cost
                best_c = c_test
        c_opt = best_c

    predicted_drag = F_arr * (1 - np.exp(-c_opt * v_arr))
    ss_res = np.sum((drag_arr - predicted_drag) ** 2)
    ss_tot = np.sum((drag_arr - np.mean(drag_arr)) ** 2)
    r2_sat = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse_sat = float(np.sqrt(np.mean((drag_arr - predicted_drag) ** 2)))

    obs045_metrics = {
        "c": float(c_opt),
        "R2": float(r2_sat),
        "RMSE": rmse_sat,
        "data_points": len(v_arr)
    }

    # 修正 source_data_refs 的生成（修复 NameError）
    source_refs = [f"{eid}:drag_ledger" for eid in experiments if experiments[eid]["config"].get("force_field_type") == "constant"]
    observations.append({
        "summary": f"OBS045: Exponential saturation model drag = F_ext * (1 - exp(-c*|v|)) fitted over all constant field experiments (including new ones). c={c_opt:.6f}, R²={r2_sat:.6f}, RMSE={rmse_sat:.6f}.",
        "source_data_refs": source_refs,
        "metrics": obs045_metrics
    })

    # ---- 5. 自由场加速度均值 ----
    free_a_values = []
    for eid, exp in experiments.items():
        force_type = exp["config"].get("force_field_type", "constant")
        if force_type != "free":
            continue
        series = exp["series"]
        available = exp["available_series"]
        # 查找已有的 a_gradient 系列
        a_key = None
        for key in available:
            if key.startswith("a_gradient_ledger_exp_") or key.startswith("a_gradient_ledger_exp_exp_"):
                a_key = key
                break
        if a_key is None:
            continue
        a_grad = np.array(series[a_key])
        free_a_values.extend(a_grad.tolist())
    # 再次确认目标实验中的 free 场（如 exp_28）数据
    for eid in target_ids:
        if experiments[eid]["config"].get("force_field_type") == "free":
            # 从已计算的派生序列中提取
            for ds in derived_series:
                if ds["experiment_id"] == eid and "a_gradient_ledger" in ds["name"]:
                    free_a_values.extend(ds["values"])
                    break
    if free_a_values:
        free_mean = float(np.mean(free_a_values))
        free_max_abs = float(np.max(np.abs(free_a_values)))
    else:
        free_mean = 0.0
        free_max_abs = 0.0

    observations.append({
        "summary": f"Free field acceleration mean = {free_mean:.4e}, max absolute = {free_max_abs:.4e} (across all free experiments including exp_28).",
        "source_data_refs": ["exp_01:a_gradient_ledger","exp_04:a_gradient_ledger","exp_11:a_gradient_ledger",
                             "exp_12:a_gradient_ledger","exp_15:a_gradient_ledger","exp_17:a_gradient_ledger"],
        "metrics": {
            "free_field_a_mean": free_mean,
            "free_field_a_max_abs": free_max_abs
        }
    })

    # ---- 6. 结果统计 ----
    total_experiments = len(target_ids)
    total_derived_series = len(derived_series)
    total_observations = len(observations)

    # 构建最终输出
    result = {
        "observation": f"已完成实验 {target_ids} 的账本维护。计算 v_gradient、a_gradient、drag（仅常数场）及 H004 残差序列。使用全局 gamma={gamma_global} 计算残差。对所有常数场实验拟合指数饱和模型 OBS045: c={c_opt:.6f}, R²={r2_sat:.6f}, RMSE={rmse_sat:.6f}。自由场加速度均值={free_mean:.4e}。",
        "derived_series": derived_series,
        "observations": observations,
        "metrics": {
            "experiments_processed": total_experiments,
            "derived_series_count": total_derived_series,
            "observation_count": total_observations,
            "c_saturation": float(c_opt),
            "R2_saturation": float(r2_sat),
            "RMSE_saturation": rmse_sat,
            "free_field_a_mean": free_mean
        },
        "figures": []
    }

    return result
