import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compute_4cd_velocity(q: List[float], dt: float) -> List[float]:
    """4-order central difference velocity from position q."""
    n = len(q)
    v = [float('nan')] * n
    for i in range(2, n - 2):
        v[i] = (-q[i+2] + 8*q[i+1] - 8*q[i-1] + q[i-2]) / (12.0 * dt)
    return v


def compute_4cd_acceleration(v: List[float], dt: float) -> List[float]:
    """4-order central difference acceleration from velocity v."""
    n = len(v)
    a = [float('nan')] * n
    for i in range(2, n - 2):
        a[i] = (-v[i+2] + 8*v[i+1] - 8*v[i-1] + v[i-2]) / (12.0 * dt)
    return a


def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload.get("output_dir", "/tmp")

    # 只处理指定的实验
    eids = params.get("experiment_ids", [])
    if not eids:
        raise ValueError("experiment_ids must be provided")

    results = {}
    derived_series_list = []

    for eid in eids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        t = series["t"]
        q = series["q"]
        dt = config["dt"]
        F_ext = config["F_ext"]

        # 计算 v_4cd, a_4cd
        v_4cd = compute_4cd_velocity(q, dt)
        a_4cd = compute_4cd_acceleration(v_4cd, dt)

        # 有效点（非NaN）
        valid_mask = [not (math.isnan(v_4cd[i]) or math.isnan(a_4cd[i])) for i in range(len(t))]
        idx = [i for i, v in enumerate(valid_mask) if v]
        if len(idx) < 3:
            raise ValueError(f"Not enough valid points for experiment {eid} after 4CD")

        t_valid = [t[i] for i in idx]
        v2 = [v_4cd[i]**2 for i in idx]
        v4 = [v_4cd[i]**4 for i in idx]
        y = [F_ext / a_4cd[i] for i in idx]

        # 线性回归 F_ext/a ~ v^2
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(v2, y)
        R2 = r_value ** 2
        n_pts = len(idx)
        residuals = [y[i] - (intercept + slope * v2[i]) for i in range(n_pts)]
        rmse = math.sqrt(sum(r ** 2 for r in residuals) / n_pts)
        mae = sum(abs(r) for r in residuals) / n_pts

        if n_pts > 2:
            r_res_v2 = sp_stats.pearsonr(residuals, v2)[0]
            r_res_v4 = sp_stats.pearsonr(residuals, v4)[0]
        else:
            r_res_v2 = float('nan')
            r_res_v4 = float('nan')

        results[eid] = {
            "n_points": n_pts,
            "F_ext": F_ext,
            "intercept": intercept,
            "slope": slope,
            "R2": R2,
            "RMSE": rmse,
            "MAE": mae,
            "resid_mean": sum(residuals) / n_pts,
            "resid_std": math.sqrt(sum((r - sum(residuals)/n_pts)**2 for r in residuals) / n_pts),
            "max_abs_resid": max(abs(r) for r in residuals),
            "corr_resid_v2": r_res_v2,
            "corr_resid_v4": r_res_v4,
        }

        # 返回派生序列（v_4cd, a_4cd）
        derived_series_list.append({
            "experiment_id": eid,
            "name": "v_4cd",
            "values": v_4cd,
            "source_name": "4阶中心差分 from q",
            "provenance": "generated data processor: step_036_analyze_data",
            "description": "4-order central difference velocity"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": "a_4cd",
            "values": a_4cd,
            "source_name": "4阶中心差分 from q (二次差分)",
            "provenance": "generated data processor: step_036_analyze_data",
            "description": "4-order central difference acceleration"
        })

    # 构造中文 observation
    obs_lines = []
    obs_lines.append(f"使用4阶中心差分(5点模板)计算a和v，边界各丢失2个点。")
    obs_lines.append(f"处理实验：{', '.join(eids)}。")
    for eid in eids:
        r = results[eid]
        line = (
            f"{eid}: F_ext={r['F_ext']}, "
            f"线性回归 F_ext/a = {r['intercept']:.6f} + {r['slope']:.6f} * v², "
            f"R²={r['R2']:.10f}, RMSE={r['RMSE']:.6e}, MAE={r['MAE']:.6e}, "
            f"残差均值={r['resid_mean']:.6e}, 残差标准差={r['resid_std']:.6e}, "
            f"max|残差|={r['max_abs_resid']:.6e}, "
            f"残差-v²相关系数={r['corr_resid_v2']:.6f}, "
            f"残差-v⁴相关系数={r['corr_resid_v4']:.6f}"
        )
        obs_lines.append(line)
    obs = "\n".join(obs_lines)

    # 不生成图像
    figures = []

    # metrics 包含每个实验的详细结果
    metrics = {
        "experiment_count": 2,
        "exp_23": results["exp_23"],
        "exp_24": results["exp_24"],
    }

    return {
        "observation": obs,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics,
    }
