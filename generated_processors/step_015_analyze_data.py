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
import scipy
from scipy import signal, stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process(payload: dict) -> dict:
    # ---------- 1. 提取参数 ----------
    parameters = payload["parameters"]
    analysis_mode = parameters.get("analysis_mode", "maintain_ledger")
    exp_ids_param = parameters.get("experiment_ids", None)
    analysis_goal = parameters.get("analysis_goal", "")
    expected_outputs = parameters.get("expected_outputs", [])
    # 根据任务，实验列表已给定，但为了保险，我们也支持全处理
    if exp_ids_param is not None:
        target_ids = set(exp_ids_param)
    else:
        target_ids = set(payload["experiments"].keys())

    # ---------- 2. 筛选 constant 场实验 ----------
    experiments = payload["experiments"]
    constant_exps = {}
    for eid, edata in experiments.items():
        config = edata["config"]
        # 判断是否 constant 场: force_field_type 为 'constant' 且 F_ext 不为 None
        # 自由场的 force_field_type 为 'free'
        if config.get("force_field_type") == "constant" and config.get("F_ext") is not None:
            constant_exps[eid] = edata

    # 任务要求对全部 24 个 constant 实验做回归，但只对 target_ids 中的新实验定义派生序列
    # 我们将对所有 constant 实验进行回归，但只定义尚未有 v/a 的派生序列（主要是 target_ids 中缺失的）
    # 但也要考虑之前已有 v/a 的实验（例如 exp_02~19 可能已有）
    derived_series = []
    regression_results = []  # 每个元素: {'exp_id', 'F_ext', 'v0', 'slope', 'intercept', 'r2', 'rmse'}

    for eid in sorted(constant_exps.keys()):
        edata = constant_exps[eid]
        config = edata["config"]
        series = edata.get("series", {})
        available = set(edata.get("available_series", []))

        t = np.array(series["t"], dtype=float)
        q = np.array(series["q"], dtype=float)

        F_ext = config["F_ext"]
        v0 = config.get("initial_v", None)

        # ---- 获取或定义 v ----
        if "v" in available and "v" in series:
            v_arr = np.array(series["v"], dtype=float)
        else:
            v_arr = np.gradient(q, t, edge_order=2)  # 一阶导数
            if len(v_arr) == len(t):
                derived_series.append({
                    "experiment_id": eid,
                    "name": "v",
                    "values": v_arr.tolist(),
                    "source_name": "np.gradient(q, t, edge_order=2)",
                    "provenance": "generated data processor: maintain_ledger",
                    "description": "velocity derived from position via central difference"
                })

        # ---- 获取或定义 a ----
        if "a" in available and "a" in series:
            a_arr = np.array(series["a"], dtype=float)
        else:
            a_arr = np.gradient(v_arr, t, edge_order=2)  # 二阶导数
            if len(a_arr) == len(t):
                derived_series.append({
                    "experiment_id": eid,
                    "name": "a",
                    "values": a_arr.tolist(),
                    "source_name": "np.gradient(v, t, edge_order=2)",
                    "provenance": "generated data processor: maintain_ledger",
                    "description": "acceleration derived from velocity via central difference"
                })

        # ---- 线性回归 a ~ v ----
        # 确保长度一致
        if len(v_arr) != len(a_arr) or len(v_arr) < 3:
            continue  # 数据异常则跳过
        slope, intercept, r_value, p_value, std_err = stats.linregress(v_arr, a_arr)
        r2 = r_value ** 2
        pred = slope * v_arr + intercept
        rmse = math.sqrt(mean_squared_error(a_arr, pred))

        regression_results.append({
            "exp_id": eid,
            "F_ext": F_ext,
            "v0": v0,
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "rmse": rmse
        })

    # ---------- 3. 跨实验汇总 ----------
    # 提取截距 ~ F_ext 回归
    intercept_data = [(r["F_ext"], r["intercept"]) for r in regression_results]
    F_ext_vals = np.array([d[0] for d in intercept_data])
    intercept_vals = np.array([d[1] for d in intercept_data])
    slope_if, inter_if, r_if, p_if, std_if = stats.linregress(F_ext_vals, intercept_vals)
    r2_if = r_if ** 2
    pred_if = slope_if * F_ext_vals + inter_if
    rmse_if = math.sqrt(mean_squared_error(intercept_vals, pred_if))

    # 提取斜率 ~ v0 回归 (v0 可能有重复，但回归可做)
    slope_data = [(r["v0"], r["slope"]) for r in regression_results]
    v0_vals = np.array([d[0] for d in slope_data])
    slope_vals = np.array([d[1] for d in slope_data])
    slope_sv, inter_sv, r_sv, p_sv, std_sv = stats.linregress(v0_vals, slope_vals)
    r2_sv = r_sv ** 2
    pred_sv = slope_sv * v0_vals + inter_sv
    rmse_sv = math.sqrt(mean_squared_error(slope_vals, pred_sv))

    # ---------- 4. 构造 observations ----------
    observations = []
    for r in regression_results:
        obs = {
            "summary": f"实验 {r['exp_id']} a-v 线性回归：斜率={r['slope']:.6f}, 截距={r['intercept']:.6f}, R²={r['r2']:.6f}, RMSE={r['rmse']:.6f}。F_ext={r['F_ext']}, v0={r['v0']}",
            "source_data_refs": [f"{r['exp_id']}:q", f"{r['exp_id']}:v", f"{r['exp_id']}:a"],
            "metrics": {
                "slope": r["slope"],
                "intercept": r["intercept"],
                "R2": r["r2"],
                "RMSE": r["rmse"],
                "F_ext": r["F_ext"],
                "v0": r["v0"]
            }
        }
        observations.append(obs)

    # 跨实验汇总
    obs_intercept = {
        "summary": f"跨实验截距与F_ext线性回归：斜率={slope_if:.6f}, 截距={inter_if:.6f}, R²={r2_if:.6f}, RMSE={rmse_if:.6f}。基于{len(regression_results)}个constant场实验。",
        "source_data_refs": [f"exp_{eid}" for eid in sorted(constant_exps.keys())],
        "metrics": {
            "slope": slope_if,
            "intercept": inter_if,
            "R2": r2_if,
            "RMSE": rmse_if,
            "n_experiments": len(regression_results)
        }
    }
    observations.append(obs_intercept)

    obs_slope = {
        "summary": f"跨实验斜率与v0线性回归：斜率={slope_sv:.6f}, 截距={inter_sv:.6f}, R²={r2_sv:.6f}, RMSE={rmse_sv:.6f}。基于{len(regression_results)}个constant场实验。",
        "source_data_refs": [f"exp_{eid}" for eid in sorted(constant_exps.keys())],
        "metrics": {
            "slope": slope_sv,
            "intercept": inter_sv,
            "R2": r2_sv,
            "RMSE": rmse_sv,
            "n_experiments": len(regression_results)
        }
    }
    observations.append(obs_slope)

    # ---------- 5. 整体 observation 摘要 ----------
    n_constant = len(regression_results)
    new_series_count = len(derived_series)
    obs_count = len(observations)

    summary_text = (
        f"对 {n_constant} 个 constant 场实验执行 a-v 线性回归，"
        f"为新实验定义了 {new_series_count} 个派生序列（v 和 a）。"
        f"共生成 {obs_count} 条 OBS，包括每个实验的回归参数及跨实验汇总。"
        f"截距~F_ext 回归 R²={r2_if:.4f}，斜率~v0 回归 R²={r2_sv:.4f}。"
        f"未宣布任何定律。"
    )

    # ---------- 6. 输出 metrics ----------
    metrics = {
        "constant_experiment_regressions": n_constant,
        "derived_series_count": new_series_count,
        "observation_count": obs_count,
        "intercept_vs_F_slope": slope_if,
        "intercept_vs_F_R2": r2_if,
        "intercept_vs_F_RMSE": rmse_if,
        "slope_vs_v0_slope": slope_sv,
        "slope_vs_v0_R2": r2_sv,
        "slope_vs_v0_RMSE": rmse_sv
    }

    # ---------- 7. 返回 ----------
    return {
        "observation": summary_text,
        "derived_series": derived_series,
        "observations": observations,
        "figures": [],
        "metrics": metrics
    }
