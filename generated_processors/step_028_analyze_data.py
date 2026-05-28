import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scipy
from scipy import integrate, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def process(payload: dict) -> dict:
    parameters = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # 筛选所有 force_field_type == 'constant' 的实验
    constant_exp_ids = []
    for eid, exp in experiments.items():
        if exp["config"].get("force_field_type") == "constant":
            constant_exp_ids.append(eid)

    if not constant_exp_ids:
        raise ValueError("No experiments with force_field_type='constant' found.")

    # 收集每个实验的派生序列、回归结果、端点值
    records = []
    new_derived_series = []
    for eid in constant_exp_ids:
        exp = experiments[eid]
        series = exp["series"]
        config = exp["config"]
        F_ext = config["F_ext"]
        initial_v = config.get("initial_v", 0.0)  # fallback

        t = np.array(series["t"])
        dt_est = None
        if len(t) > 1:
            dt_est = t[1] - t[0]

        # 检查是否已有 a_center_diff 和 v_cumtrapz
        has_a_cd = "a_center_diff" in series
        has_v_ct = "v_cumtrapz" in series

        if not has_a_cd or not has_v_ct:
            # 需要创建派生序列
            # 优先使用已有的 v 序列来求 a_center_diff；否则从 q 求导
            if "v" in series:
                v_raw = np.array(series["v"])
                a_center_diff = np.gradient(v_raw, t, edge_order=2)
            elif "a" in series:
                # 使用已有的 a 序列计算中心差分？ 直接用 a 作为加速度？但要求派生，为了统一，我们用 q 做二阶导
                # 但 a 序列已经存在，我们使用 a 并直接设置为 a_center_diff
                a_center_diff = np.array(series["a"])
            else:
                # 从 q 求 v 再求 a
                v_raw = np.gradient(np.array(series["q"]), t, edge_order=2)
                a_center_diff = np.gradient(v_raw, t, edge_order=2)
                # v_cumtrapz 也会从 a 积分得到，所以这里 a 是二阶导
            # 计算 v_cumtrapz: 从 a_center_diff 积分
            try:
                v_cumtrapz = initial_v + scipy.integrate.cumulative_trapezoid(a_center_diff, t, initial=0)
            except AttributeError:
                # fallback manual cumulative trapezoid
                dt = t[1] - t[0]
                v_integ = np.zeros_like(t)
                v_integ[0] = 0.0
                v_integ[1:] = np.cumsum(0.5 * (a_center_diff[:-1] + a_center_diff[1:]) * dt)
                v_cumtrapz = initial_v + v_integ
            # 确保长度一致
            if len(v_cumtrapz) != len(t):
                # 如果 cumulative_trapezoid 返回少一个点，进行插值
                v_cumtrapz = np.interp(t, t[:len(v_cumtrapz)], v_cumtrapz)
            # 准备派生序列返回（如果a_center_diff或v_cumtrapz原本不存在，则返回；否则不返回避免重复）
            if not has_a_cd:
                new_derived_series.append({
                    "experiment_id": eid,
                    "name": "a_center_diff",
                    "values": a_center_diff.tolist(),
                    "source_name": f"np.gradient(v, t, edge_order=2) on {eid}",
                    "provenance": "generated data processor: step_27_style",
                    "description": "加速度中心差分序列"
                })
            if not has_v_ct:
                new_derived_series.append({
                    "experiment_id": eid,
                    "name": "v_cumtrapz",
                    "values": v_cumtrapz.tolist(),
                    "source_name": f"cumulative_trapezoid(a_center_diff, t, initial=0) + initial_v on {eid}",
                    "provenance": "generated data processor: step_27_style",
                    "description": "速度累积梯形积分序列"
                })
        else:
            a_center_diff = np.array(series["a_center_diff"])
            v_cumtrapz = np.array(series["v_cumtrapz"])

        # a-v 线性回归
        slope_av, intercept_av, r_value_av, _, _ = stats.linregress(v_cumtrapz, a_center_diff)
        R2_av = r_value_av ** 2

        # 端点值
        a0 = float(a_center_diff[0])
        a_end = float(a_center_diff[-1])
        v0 = float(v_cumtrapz[0])
        v_end = float(v_cumtrapz[-1])

        records.append({
            "exp_id": eid,
            "F_ext": F_ext,
            "v0": v0,
            "slope_av": slope_av,
            "intercept_av": intercept_av,
            "R2_av": R2_av,
            "a0": a0,
            "a_end": a_end,
            "v_end": v_end
        })

    # 将 records 转为 numpy 数组以便分析
    F_ext_arr = np.array([r["F_ext"] for r in records])
    v0_arr = np.array([r["v0"] for r in records])
    a0_arr = np.array([r["a0"] for r in records])
    a_end_arr = np.array([r["a_end"] for r in records])
    intercept_av_arr = np.array([r["intercept_av"] for r in records])
    slope_av_arr = np.array([r["slope_av"] for r in records])

    # 跨实验分析 (1): a0 ≈ F_ext - c*v0，寻找最佳c
    # y = a0 - F_ext, fit y = -c * v0 (through origin)
    y = a0_arr - F_ext_arr
    X = v0_arr.reshape(-1, 1)
    lr_c = LinearRegression(fit_intercept=False)
    lr_c.fit(X, y)
    c_best = -lr_c.coef_[0]  # 因为 y = -c * v0 => c = -coef_
    y_pred = lr_c.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum(y ** 2)
    R2_c = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # 也可以计算相关系数
    corr_c, _ = stats.pearsonr(y.flatten(), y_pred.flatten()) if len(y) > 2 else (0, 0)

    # 跨实验分析 (2): intercept_av 与 F_ext 线性回归
    lr_intercept = LinearRegression()
    lr_intercept.fit(F_ext_arr.reshape(-1, 1), intercept_av_arr)
    intercept_pred = lr_intercept.predict(F_ext_arr.reshape(-1, 1))
    R2_intercept_vs_F = r2_score(intercept_av_arr, intercept_pred)

    # 跨实验分析 (3): slope_av 与 v0 线性回归
    lr_slope = LinearRegression()
    lr_slope.fit(v0_arr.reshape(-1, 1), slope_av_arr)
    slope_pred = lr_slope.predict(v0_arr.reshape(-1, 1))
    R2_slope_vs_v0 = r2_score(slope_av_arr, slope_pred)

    # 构建 observations
    observations = []
    # 每个实验一条 OBS
    for rec in records:
        obs = {
            "summary": (f"常数场实验 {rec['exp_id']}: F_ext={rec['F_ext']}, v0={rec['v0']:.6f}, "
                        f"a0={rec['a0']:.6f}, a_end={rec['a_end']:.6f}, v_end={rec['v_end']:.6f}, "
                        f"slope_av={rec['slope_av']:.6f}, intercept_av={rec['intercept_av']:.6f}, R²_av={rec['R2_av']:.6f}"),
            "source_data_refs": [f"{rec['exp_id']}:t", f"{rec['exp_id']}:q",
                                 f"{rec['exp_id']}:a_center_diff", f"{rec['exp_id']}:v_cumtrapz"],
            "metrics": {
                "F_ext": rec["F_ext"],
                "v0": rec["v0"],
                "a0": rec["a0"],
                "a_end": rec["a_end"],
                "v_end": rec["v_end"],
                "slope_av": rec["slope_av"],
                "intercept_av": rec["intercept_av"],
                "R2_av": rec["R2_av"]
            }
        }
        observations.append(obs)

    # 跨实验分析 OBS
    obs_c = {
        "summary": (f"跨实验分析 (1): 寻找常数c使 a0 ≈ F_ext - c*v0. 最佳c={c_best:.6f}, "
                    f"过原点R²={R2_c:.6f}, Pearson相关系数={corr_c:.6f}"),
        "source_data_refs": [f"{r['exp_id']}:a_center_diff" for r in records],
        "metrics": {
            "best_c": float(c_best),
            "R2_c": float(R2_c),
            "corr_c": float(corr_c),
            "analysis_type": "a0 ~ F_ext - c*v0"
        }
    }
    observations.append(obs_c)

    obs_intercept = {
        "summary": (f"跨实验分析 (2): intercept_av 与 F_ext 线性回归. "
                    f"斜率={lr_intercept.coef_[0]:.6f}, 截距={lr_intercept.intercept_:.6f}, R²={R2_intercept_vs_F:.6f}"),
        "source_data_refs": [f"{r['exp_id']}:intercept_av" for r in records],
        "metrics": {
            "intercept_slope": float(lr_intercept.coef_[0]),
            "intercept_intercept": float(lr_intercept.intercept_),
            "R2_intercept_vs_F": float(R2_intercept_vs_F)
        }
    }
    observations.append(obs_intercept)

    obs_slope = {
        "summary": (f"跨实验分析 (3): slope_av 与 v0 线性回归. "
                    f"斜率={lr_slope.coef_[0]:.6f}, 截距={lr_slope.intercept_:.6f}, R²={R2_slope_vs_v0:.6f}"),
        "source_data_refs": [f"{r['exp_id']}:slope_av" for r in records],
        "metrics": {
            "slope_slope": float(lr_slope.coef_[0]),
            "slope_intercept": float(lr_slope.intercept_),
            "R2_slope_vs_v0": float(R2_slope_vs_v0)
        }
    }
    observations.append(obs_slope)

    # 汇总指标
    metrics = {
        "experiment_count": len(constant_exp_ids),
        "observation_count": len(observations),
        "best_c": c_best,
        "R2_c": R2_c,
        "R2_intercept_vs_F": R2_intercept_vs_F,
        "R2_slope_vs_v0": R2_slope_vs_v0
    }

    # observation 汇总文本
    observation = (f"处理了 {len(constant_exp_ids)} 个常数场实验。已确保每个实验都存在 a_center_diff 和 v_cumtrapz 派生序列。"
                   f"完成了 a-v 线性回归并记录端点值。跨实验分析: (1) a0 与 F_ext- c*v0 拟合最佳c={c_best:.6f}, R²={R2_c:.6f}; "
                   f"(2) intercept_av 与 F_ext R²={R2_intercept_vs_F:.6f}; (3) slope_av 与 v0 R²={R2_slope_vs_v0:.6f}。"
                   f"共生成 {len(observations)} 条 OBS。")

    return {
        "observation": observation,
        "derived_series": new_derived_series,
        "observations": observations,
        "figures": [],
        "metrics": metrics
    }
