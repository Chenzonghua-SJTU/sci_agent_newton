import os
import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, List

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # 只处理指定的 experiment
    exp_id = params["experiment_id"]
    if exp_id not in experiments:
        raise ValueError(f"Experiment {exp_id} not found in payload.")

    exp = experiments[exp_id]
    source_series = params["source_series"]
    position_name = params["position_name"]
    velocity_name = params["velocity_name"]
    acceleration_name = params["acceleration_name"]
    window_length = params["window_length"]
    polyorder = params["polyorder"]
    overwrite = params["overwrite"]

    # 检查源序列是否存在
    if source_series not in exp["series"]:
        raise ValueError(f"Source series '{source_series}' not found in experiment {exp_id}.")
    q = np.array(exp["series"][source_series])
    t = np.array(exp["series"]["t"])
    n = len(t)
    if len(q) != n:
        raise ValueError(f"Length mismatch: source series {source_series} has {len(q)} points, t has {n}.")

    # 检查是否需要跳过(若已有且不覆盖)
    defines_q = (position_name == source_series)
    series_to_create = []
    if not defines_q:
        # 要生成新的位置序列
        if position_name in exp["available_series"] and not overwrite:
            raise ValueError(f"Position series '{position_name}' already exists and overwrite=False.")
        series_to_create.append(position_name)
    if velocity_name in exp["available_series"] and not overwrite:
        raise ValueError(f"Velocity series '{velocity_name}' already exists and overwrite=False.")
    series_to_create.append(velocity_name)
    if acceleration_name in exp["available_series"] and not overwrite:
        raise ValueError(f"Acceleration series '{acceleration_name}' already exists and overwrite=False.")
    series_to_create.append(acceleration_name)

    # 使用 Savitzky-Golay 滤波同时求导
    if window_length % 2 == 0:
        window_length += 1  # 必须奇数
    if polyorder >= window_length:
        polyorder = window_length - 1

    # 平滑位置
    q_smooth = savgol_filter(q, window_length, polyorder, deriv=0)
    # 速度（一阶导）
    v = savgol_filter(q, window_length, polyorder, deriv=1, delta=t[1]-t[0] if len(t)>1 else 1.0)
    # 加速度（二阶导）
    a = savgol_filter(q, window_length, polyorder, deriv=2, delta=t[1]-t[0] if len(t)>1 else 1.0)

    # 构建 derived_series
    derived_series = []
    if not defines_q:
        derived_series.append({
            "experiment_id": exp_id,
            "name": position_name,
            "values": q_smooth.tolist(),
            "source_name": f"SG_filter({source_series}, window={window_length}, polyorder={polyorder})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Smoothed position from {source_series}"
        })
    derived_series.append({
        "experiment_id": exp_id,
        "name": velocity_name,
        "values": v.tolist(),
        "source_name": f"SG_filter_deriv1({source_series})",
        "provenance": "generated data processor: estimate_kinematics",
        "description": f"Velocity estimated from {source_series}"
    })
    derived_series.append({
        "experiment_id": exp_id,
        "name": acceleration_name,
        "values": a.tolist(),
        "source_name": f"SG_filter_deriv2({source_series})",
        "provenance": "generated data processor: estimate_kinematics",
        "description": f"Acceleration estimated from {source_series}"
    })

    # metrics
    metrics = {
        f"{exp_id}_q_smooth_min": float(np.min(q_smooth)),
        f"{exp_id}_q_smooth_max": float(np.max(q_smooth)),
        f"{exp_id}_q_smooth_mean": float(np.mean(q_smooth)),
        f"{exp_id}_q_smooth_std": float(np.std(q_smooth)),
        f"{exp_id}_v_min": float(np.min(v)),
        f"{exp_id}_v_max": float(np.max(v)),
        f"{exp_id}_v_mean": float(np.mean(v)),
        f"{exp_id}_v_std": float(np.std(v)),
        f"{exp_id}_a_min": float(np.min(a)),
        f"{exp_id}_a_max": float(np.max(a)),
        f"{exp_id}_a_mean": float(np.mean(a)),
        f"{exp_id}_a_std": float(np.std(a)),
    }

    observation = (
        f"运动学估计完成。实验 {exp_id}: "
        f"从 {source_series} 通过 SG 滤波 (窗口 {window_length}, 阶数 {polyorder}) 估计 "
        f"平滑位置 {position_name}, 速度 {velocity_name}, 加速度 {acceleration_name}。"
        f"q_smooth: 最小值 {metrics[f'{exp_id}_q_smooth_min']:.4f}, "
        f"最大值 {metrics[f'{exp_id}_q_smooth_max']:.4f}; "
        f"v: 最小值 {metrics[f'{exp_id}_v_min']:.4f}, "
        f"最大值 {metrics[f'{exp_id}_v_max']:.4f}; "
        f"a: 最小值 {metrics[f'{exp_id}_a_min']:.4f}, "
        f"最大值 {metrics[f'{exp_id}_a_max']:.4f}。"
    )

    result = {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [],
        "metrics": metrics
    }
    return result
