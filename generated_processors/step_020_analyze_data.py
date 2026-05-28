import json
import math
import statistics
import itertools
import functools
import collections
import typing
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    params = payload.get("parameters", {})
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(payload["experiments"].keys())
    experiments = payload["experiments"]

    # 过滤出 constant 场实验
    valid_ids = []
    for eid in exp_ids:
        if eid not in experiments:
            continue
        config = experiments[eid].get("config", {})
        if config.get("force_field_type") != "constant":
            continue
        series = experiments[eid].get("series", {})
        if "a" not in series or "v" not in series:
            continue
        valid_ids.append(eid)

    results = []
    for eid in valid_ids:
        config = experiments[eid]["config"]
        series = experiments[eid]["series"]
        a = np.array(series["a"])
        v = np.array(series["v"])
        # 检查 q 序列是否存在，用于 a-q 回归
        has_q = "q" in series
        if has_q:
            q = np.array(series["q"])
        else:
            q = None
        F_ext = config["F_ext"]
        v0 = config["initial_v"]

        # a vs v 线性回归
        X_av = v.reshape(-1, 1)
        reg_av = LinearRegression().fit(X_av, a)
        slope_av = reg_av.coef_[0]
        intercept_av = reg_av.intercept_
        r2_av = reg_av.score(X_av, a)
        pred_av = reg_av.predict(X_av)
        rmse_av = math.sqrt(np.mean((a - pred_av)**2))

        # a vs q 线性回归（如果 q 存在）
        if has_q:
            X_aq = q.reshape(-1, 1)
            reg_aq = LinearRegression().fit(X_aq, a)
            slope_aq = reg_aq.coef_[0]
            intercept_aq = reg_aq.intercept_
            r2_aq = reg_aq.score(X_aq, a)
            pred_aq = reg_aq.predict(X_aq)
            rmse_aq = math.sqrt(np.mean((a - pred_aq)**2))
        else:
            slope_aq = np.nan
            intercept_aq = np.nan
            r2_aq = np.nan
            rmse_aq = np.nan

        results.append({
            "exp_id": eid,
            "F_ext": F_ext,
            "v0": v0,
            "slope_av": slope_av,
            "intercept_av": intercept_av,
            "r2_av": r2_av,
            "rmse_av": rmse_av,
            "slope_aq": slope_aq,
            "intercept_aq": intercept_aq,
            "r2_aq": r2_aq,
            "rmse_aq": rmse_aq,
            "has_q": has_q
        })

    if not results:
        raise ValueError("没有有效的 constant 场实验可供分析。")

    # 跨实验多元回归
    F_vals = np.array([r["F_ext"] for r in results])
    v0_vals = np.array([r["v0"] for r in results])
    intercept_vals = np.array([r["intercept_av"] for r in results])
    slope_vals = np.array([r["slope_av"] for r in results])
    slope_q_vals = np.array([r["slope_aq"] for r in results])
    # 只对有有效 q 斜率的实验做回归
    valid_q = [r for r in results if r["has_q"]]
    F_q = np.array([r["F_ext"] for r in valid_q])
    v0_q = np.array([r["v0"] for r in valid_q])
    slope_q_only = np.array([r["slope_aq"] for r in valid_q])

    # 构造交互项矩阵
    def build_X(F, v0):
        return np.column_stack((F, v0, F * v0))

    X_intercept = build_X(F_vals, v0_vals)
    reg_intercept = LinearRegression().fit(X_intercept, intercept_vals)
    r2_intercept = reg_intercept.score(X_intercept, intercept_vals)
    pred_intercept = reg_intercept.predict(X_intercept)
    rmse_intercept = math.sqrt(np.mean((intercept_vals - pred_intercept)**2))
    coef_intercept = reg_intercept.coef_
    intercept_intercept = reg_intercept.intercept_

    X_slope = build_X(F_vals, v0_vals)
    reg_slope = LinearRegression().fit(X_slope, slope_vals)
    r2_slope = reg_slope.score(X_slope, slope_vals)
    pred_slope = reg_slope.predict(X_slope)
    rmse_slope = math.sqrt(np.mean((slope_vals - pred_slope)**2))
    coef_slope = reg_slope.coef_
    intercept_slope = reg_slope.intercept_

    if len(valid_q) >= 3:
        X_slope_q = build_X(F_q, v0_q)
        reg_slope_q = LinearRegression().fit(X_slope_q, slope_q_only)
        r2_slope_q = reg_slope_q.score(X_slope_q, slope_q_only)
        pred_slope_q = reg_slope_q.predict(X_slope_q)
        rmse_slope_q = math.sqrt(np.mean((slope_q_only - pred_slope_q)**2))
        coef_slope_q = reg_slope_q.coef_
        intercept_slope_q = reg_slope_q.intercept_
    else:
        r2_slope_q = np.nan
        rmse_slope_q = np.nan
        coef_slope_q = [np.nan, np.nan, np.nan]
        intercept_slope_q = np.nan

    # 构建 observations 列表
    observations = []

    # 1. 每个实验的 a-v 回归
    for r in results:
        obs = {
            "summary": (
                f"Experiment {r['exp_id']}: a vs v linear regression: "
                f"slope={r['slope_av']:.6f}, intercept={r['intercept_av']:.6f}, "
                f"R2={r['r2_av']:.6f}, RMSE={r['rmse_av']:.2e}."
            ),
            "source_data_refs": [f"{r['exp_id']}:a", f"{r['exp_id']}:v"],
            "metrics": {
                "slope_av": r['slope_av'],
                "intercept_av": r['intercept_av'],
                "r2_av": r['r2_av'],
                "rmse_av": r['rmse_av']
            }
        }
        observations.append(obs)

    # 2. 每个实验的 a-q 回归（若有 q）
    for r in results:
        if not r["has_q"]:
            continue
        obs = {
            "summary": (
                f"Experiment {r['exp_id']}: a vs q linear regression: "
                f"slope={r['slope_aq']:.6f}, intercept={r['intercept_aq']:.6f}, "
                f"R2={r['r2_aq']:.6f}, RMSE={r['rmse_aq']:.2e}."
            ),
            "source_data_refs": [f"{r['exp_id']}:a", f"{r['exp_id']}:q"],
            "metrics": {
                "slope_aq": r['slope_aq'],
                "intercept_aq": r['intercept_aq'],
                "r2_aq": r['r2_aq'],
                "rmse_aq": r['rmse_aq']
            }
        }
        observations.append(obs)

    # 3. 跨实验多元回归：intercept_av ~ F_ext + v0 + F_ext:v0
    observations.append({
        "summary": (
            f"Multiple regression on intercept_av: intercept_av ~ F_ext + v0 + F_ext:v0. "
            f"R2={r2_intercept:.6f}, RMSE={rmse_intercept:.2e}. "
            f"Coefficients: F_ext={coef_intercept[0]:.6f}, v0={coef_intercept[1]:.6f}, "
            f"F_ext:v0={coef_intercept[2]:.6f}, intercept={intercept_intercept:.6f}."
        ),
        "source_data_refs": [f"{r['exp_id']}:a" for r in results] + [f"{r['exp_id']}:v" for r in results],
        "metrics": {
            "r2": r2_intercept,
            "rmse": rmse_intercept,
            "coef_F_ext": coef_intercept[0],
            "coef_v0": coef_intercept[1],
            "coef_F_ext_v0": coef_intercept[2],
            "intercept": intercept_intercept
        }
    })

    # 4. 跨实验多元回归：slope_av ~ F_ext + v0 + F_ext:v0
    observations.append({
        "summary": (
            f"Multiple regression on slope_av: slope_av ~ F_ext + v0 + F_ext:v0. "
            f"R2={r2_slope:.6f}, RMSE={rmse_slope:.2e}. "
            f"Coefficients: F_ext={coef_slope[0]:.6f}, v0={coef_slope[1]:.6f}, "
            f"F_ext:v0={coef_slope[2]:.6f}, intercept={intercept_slope:.6f}."
        ),
        "source_data_refs": [f"{r['exp_id']}:a" for r in results] + [f"{r['exp_id']}:v" for r in results],
        "metrics": {
            "r2": r2_slope,
            "rmse": rmse_slope,
            "coef_F_ext": coef_slope[0],
            "coef_v0": coef_slope[1],
            "coef_F_ext_v0": coef_slope[2],
            "intercept": intercept_slope
        }
    })

    # 5. 跨实验多元回归：slope_aq ~ F_ext + v0 + F_ext:v0（若有足够实验）
    if len(valid_q) >= 3:
        observations.append({
            "summary": (
                f"Multiple regression on slope_aq: slope_aq ~ F_ext + v0 + F_ext:v0 (using {len(valid_q)} experiments with q). "
                f"R2={r2_slope_q:.6f}, RMSE={rmse_slope_q:.2e}. "
                f"Coefficients: F_ext={coef_slope_q[0]:.6f}, v0={coef_slope_q[1]:.6f}, "
                f"F_ext:v0={coef_slope_q[2]:.6f}, intercept={intercept_slope_q:.6f}."
            ),
            "source_data_refs": [f"{r['exp_id']}:a" for r in valid_q] + [f"{r['exp_id']}:q" for r in valid_q],
            "metrics": {
                "r2": r2_slope_q,
                "rmse": rmse_slope_q,
                "coef_F_ext": coef_slope_q[0],
                "coef_v0": coef_slope_q[1],
                "coef_F_ext_v0": coef_slope_q[2],
                "intercept": intercept_slope_q
            }
        })
    else:
        observations.append({
            "summary": (
                f"Multiple regression on slope_aq: insufficient data (only {len(valid_q)} experiments with q). "
                f"Regression not performed."
            ),
            "source_data_refs": [f"{r['exp_id']}:a" for r in valid_q] + [f"{r['exp_id']}:q" for r in valid_q],
            "metrics": {
                "r2": np.nan,
                "rmse": np.nan,
                "coef_F_ext": np.nan,
                "coef_v0": np.nan,
                "coef_F_ext_v0": np.nan,
                "intercept": np.nan
            }
        })

    # 总体 observation 文本
    overall_obs = (
        f"完成了 {len(results)} 个常数场实验的 a-v 和 a-q 线性回归，"
        f"以及 intercept_av、slope_av 和 slope_aq 对 F_ext、v0 及其交互项的跨实验多元回归。"
        f"详见 observations 列表。"
    )

    return {
        "observation": overall_obs,
        "derived_series": [],
        "observations": observations,
        "figures": [],
        "metrics": {
            "experiment_count": len(results),
            "observation_count": len(observations),
            "intercept_av_regression_r2": r2_intercept,
            "slope_av_regression_r2": r2_slope,
            "slope_aq_regression_r2": r2_slope_q if not math.isnan(r2_slope_q) else None
        }
    }
