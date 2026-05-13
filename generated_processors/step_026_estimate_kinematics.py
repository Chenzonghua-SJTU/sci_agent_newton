import numpy as np
from scipy.signal import savgol_filter
import os
from typing import Dict, List, Any

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    if action != "estimate_kinematics":
        raise ValueError(f"Unsupported action: {action}")

    # 解析参数
    exp_id = params.get("experiment_id")
    source_series = params.get("source_series", "q")
    pos_name = params.get("position_name", "q_smooth")
    vel_name = params.get("velocity_name", "v_sg")
    acc_name = params.get("acceleration_name", "a_sg")
    window_length = int(params.get("window_length", 5))
    polyorder = int(params.get("polyorder", 2))
    overwrite = params.get("overwrite", True)

    if exp_id is None:
        # 若无指定实验，处理所有有 source_series 的实验
        raise ValueError("experiment_id is required for estimate_kinematics")
    if exp_id not in experiments:
        raise ValueError(f"Experiment {exp_id} not found in payload")
    exp = experiments[exp_id]

    t = exp["series"].get("t")
    q = exp["series"].get(source_series)
    if t is None or q is None:
        raise ValueError(f"Series 't' or '{source_series}' not found in experiment {exp_id}")
    if len(t) != len(q):
        raise ValueError("Length of t and source_series mismatch")
    n = len(t)
    if n < window_length:
        raise ValueError(f"Data length ({n}) < window_length ({window_length})")

    # 检查重复序列
    existing = exp["series"]
    if pos_name in existing or vel_name in existing or acc_name in existing:
        if not overwrite:
            raise ValueError(f"One of {pos_name}, {vel_name}, {acc_name} already exists and overwrite=False")
        # 允许覆盖

    # 计算平滑位置、速度、加速度
    q_smooth = savgol_filter(q, window_length, polyorder, deriv=0)
    v = savgol_filter(q, window_length, polyorder, deriv=1, delta=t[1] - t[0])
    a = savgol_filter(q, window_length, polyorder, deriv=2, delta=t[1] - t[0])

    # 构建 derived_series
    derived = []
    derived.append({
        "experiment_id": exp_id,
        "name": pos_name,
        "values": q_smooth.tolist(),
        "source_name": f"Savitzky-Golay smooth of {source_series} (window={window_length}, poly={polyorder})",
        "provenance": "generated data processor: estimate_kinematics",
        "description": f"Smoothed position using Savitzky-Golay filter"
    })
    derived.append({
        "experiment_id": exp_id,
        "name": vel_name,
        "values": v.tolist(),
        "source_name": f"Savitzky-Golay velocity from {source_series} (window={window_length}, poly={polyorder})",
        "provenance": "generated data processor: estimate_kinematics",
        "description": f"Estimated velocity (1st derivative) via Savitzky-Golay"
    })
    derived.append({
        "experiment_id": exp_id,
        "name": acc_name,
        "values": a.tolist(),
        "source_name": f"Savitzky-Golay acceleration from {source_series} (window={window_length}, poly={polyorder})",
        "provenance": "generated data processor: estimate_kinematics",
        "description": f"Estimated acceleration (2nd derivative) via Savitzky-Golay"
    })

    # 计算简单 metrics
    metrics = {
        f"{vel_name}_min": float(np.min(v)),
        f"{vel_name}_max": float(np.max(v)),
        f"{vel_name}_mean": float(np.mean(v)),
        f"{vel_name}_std": float(np.std(v)),
        f"{acc_name}_min": float(np.min(a)),
        f"{acc_name}_max": float(np.max(a)),
        f"{acc_name}_mean": float(np.mean(a)),
        f"{acc_name}_std": float(np.std(a)),
        "window_length": window_length,
        "polyorder": polyorder
    }

    observation = (
        f"对实验 {exp_id} 使用 Savitzky-Golay 滤波器，从 {source_series} 估计运动学参数。\n"
        f"参数: window_length={window_length}, polyorder={polyorder}.\n"
        f"平滑后的位置序列 '{pos_name}' 已生成。\n"
        f"速度 '{vel_name}': min={metrics[f'{vel_name}_min']:.4f}, max={metrics[f'{vel_name}_max']:.4f}, "
        f"mean={metrics[f'{vel_name}_mean']:.4f}, std={metrics[f'{vel_name}_std']:.4f}.\n"
        f"加速度 '{acc_name}': min={metrics[f'{acc_name}_min']:.4f}, max={metrics[f'{acc_name}_max']:.4f}, "
        f"mean={metrics[f'{acc_name}_mean']:.4f}, std={metrics[f'{acc_name}_std']:.4f}."
    )

    # 无图像
    figures = []

    return {
        "observation": observation,
        "derived_series": derived,
        "figures": figures,
        "metrics": metrics
    }
