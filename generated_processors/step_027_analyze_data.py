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
import scipy.integrate
import scipy.signal
import sklearn.linear_model
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process(payload: dict) -> dict:
    action = payload["action"]
    parameters = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # 只处理指定的实验
    exp_ids = parameters.get("experiment_ids")
    if not exp_ids:
        exp_ids = list(experiments.keys())

    derived_series = []
    observations = []

    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"实验 {eid} 不存在于 payload 中")
        exp = experiments[eid]
        config = exp["config"]
        available = exp["available_series"]

        if "t" not in available or "q" not in available:
            raise ValueError(f"实验 {eid} 缺少 t 或 q 序列")
        t = np.array(exp["series"]["t"], dtype=float)
        q = np.array(exp["series"]["q"], dtype=float)
        dt = np.mean(np.diff(t))
        initial_v = config.get("initial_v", 0.0)

        # 1) 计算加速度 a（两次中心差分）和速度 v（累积梯形积分）
        # 第一次梯度：得到速度估计（仅用于中间步骤）
        v_est = np.gradient(q, t, edge_order=2)
        a_derived = np.gradient(v_est, t, edge_order=2)

        # 速度 v：从 a 梯形积分得到
        v_derived = initial_v + scipy.integrate.cumulative_trapezoid(
            a_derived, t, initial=0.0
        )

        # 2) a-v 线性回归
        X = v_derived.reshape(-1, 1)
        y = a_derived
        reg = sklearn.linear_model.LinearRegression()
        reg.fit(X, y)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        r2 = r2_score(y, reg.predict(X))

        # 3) 记录端点值
        a0 = a_derived[0]
        a_end = a_derived[-1]
        v0 = initial_v
        v_end = v_derived[-1]

        # 派生序列
        derived_series.append({
            "experiment_id": eid,
            "name": "a_center_diff",
            "values": a_derived.tolist(),
            "source_name": "np.gradient(np.gradient(q,t),t) edge_order=2",
            "provenance": "generated data processor: maintain_ledger",
            "description": "加速度，通过两次中心差分从位移得到"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "v_cumtrapz",
            "values": v_derived.tolist(),
            "source_name": f"initial_v + cumulative_trapezoid(a, t), initial_v={initial_v}",
            "provenance": "generated data processor: maintain_ledger",
            "description": "速度，通过对加速度累积梯形积分得到，使用初始速度"
        })

        # 单个实验的 observation
        obs = {
            "summary": f"实验 {eid}: a-v 线性回归斜率={slope:.6f}, 截距={intercept:.6f}, R²={r2:.6f}; a0={a0:.6f}, a_end={a_end:.6f}, v0={v0:.6f}, v_end={v_end:.6f}",
            "source_data_refs": [f"{eid}:q", f"{eid}:t"],
            "metrics": {
                "slope_av": slope,
                "intercept_av": intercept,
                "r2_av": r2,
                "a0": a0,
                "a_end": a_end,
                "v0": v0,
                "v_end": v_end,
                "force_field_type": config.get("force_field_type", ""),
                "F_ext": config.get("F_ext") if config.get("force_field_type") != "free" else 0.0
            }
        }

        # 自由场特殊检查
        if config.get("force_field_type") == "free":
            a_mean = np.mean(a_derived)
            a_std = np.std(a_derived)
            obs["metrics"]["a_mean"] = a_mean
            obs["metrics"]["a_std"] = a_std
            obs["summary"] += f"; 自由场: a 均值={a_mean:.6f}, 标准差={a_std:.6f} (应接近0)"

        observations.append(obs)

    # 构建返回
    result = {
        "observation": f"处理了 {len(exp_ids)} 个实验（{', '.join(exp_ids)}）。为每个实验创建了派生序列 a_center_diff, v_cumtrapz。完成了 a-v 线性回归并记录端点值。共生成 {len(observations)} 条 OBS。",
        "derived_series": derived_series,
        "observations": observations,
        "validations": [],
        "figures": [],
        "metrics": {
            "experiment_count": len(exp_ids),
            "observation_count": len(observations),
            "free_experiment_ids": [eid for eid in exp_ids if experiments[eid]["config"].get("force_field_type") == "free"]
        }
    }
    return result
