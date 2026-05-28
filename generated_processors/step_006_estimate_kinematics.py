import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 确定要处理的实验ID列表
    exp_ids = params.get("experiment_ids", [])
    single_exp = params.get("experiment_id")
    if not exp_ids and single_exp:
        exp_ids = [single_exp]
    if not exp_ids:
        exp_ids = list(experiments.keys())

    if len(exp_ids) != 1:
        return {
            "observation": "estimate_kinematics 目前只支持单个 experiment_id，但提供了多个或没有。",
            "derived_series": [],
            "figures": [],
            "metrics": {}
        }

    eid = exp_ids[0]
    if eid not in experiments:
        raise ValueError(f"Experiment '{eid}' not found in payload.")

    exp = experiments[eid]
    config = exp.get("config", {})
    series = exp.get("series", {})
    available = exp.get("available_series", [])

    source_series_name = params.get("source_series", "q")
    if source_series_name not in series:
        raise ValueError(f"Source series '{source_series_name}' not available for experiment {eid}. Available: {available}")
    q = np.array(series[source_series_name], dtype=float)
    # 获取时间序列
    if "t" in series:
        t = np.array(series["t"], dtype=float)
    else:
        t = np.arange(len(q)) * config.get("dt", 0.1)
    dt = config.get("dt")
    if dt is None or dt <= 0:
        dt = np.median(np.diff(t))
        if dt <= 0:
            raise ValueError(f"Could not determine dt for experiment {eid}.")
    window_length = params.get("window_length", 11)
    polyorder = params.get("polyorder", 3)
    overwrite = params.get("overwrite", True)

    # 检查窗口有效性
    if window_length % 2 == 0:
        window_length += 1  # 确保奇数
    if window_length < polyorder + 2:
        window_length = polyorder + 3
    if window_length > len(q):
        raise ValueError(f"window_length {window_length} > series length {len(q)} for experiment {eid}.")

    # 使用 Savitzky-Golay 滤波器
    q_smooth = savgol_filter(q, window_length, polyorder, deriv=0, delta=dt)
    v_smooth = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt)
    a_smooth = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)

    # 派生序列命名
    pos_name = "q_smooth"
    vel_name = "v_smooth"
    acc_name = "a_smooth"

    derived_series = [
        {
            "experiment_id": eid,
            "name": pos_name,
            "values": q_smooth.tolist(),
            "source_name": f"savgol_filter({source_series_name}, window={window_length}, poly={polyorder}, deriv=0)",
            "provenance": f"generated data processor: estimate_kinematics (step {action})",
            "description": f"Smoothed position using Savitzky-Golay filter (window={window_length}, poly={polyorder})"
        },
        {
            "experiment_id": eid,
            "name": vel_name,
            "values": v_smooth.tolist(),
            "source_name": f"savgol_filter({source_series_name}, window={window_length}, poly={polyorder}, deriv=1)",
            "provenance": f"generated data processor: estimate_kinematics (step {action})",
            "description": "Estimated velocity (first derivative of smoothed position)"
        },
        {
            "experiment_id": eid,
            "name": acc_name,
            "values": a_smooth.tolist(),
            "source_name": f"savgol_filter({source_series_name}, window={window_length}, poly={polyorder}, deriv=2)",
            "provenance": f"generated data processor: estimate_kinematics (step {action})",
            "description": "Estimated acceleration (second derivative of smoothed position)"
        }
    ]

    # 计算 metrics
    metrics = {
        f"{eid}_{pos_name}_min": float(np.min(q_smooth)),
        f"{eid}_{pos_name}_max": float(np.max(q_smooth)),
        f"{eid}_{pos_name}_mean": float(np.mean(q_smooth)),
        f"{eid}_{pos_name}_std": float(np.std(q_smooth)),
        f"{eid}_{vel_name}_min": float(np.min(v_smooth)),
        f"{eid}_{vel_name}_max": float(np.max(v_smooth)),
        f"{eid}_{vel_name}_mean": float(np.mean(v_smooth)),
        f"{eid}_{vel_name}_std": float(np.std(v_smooth)),
        f"{eid}_{acc_name}_min": float(np.min(a_smooth)),
        f"{eid}_{acc_name}_max": float(np.max(a_smooth)),
        f"{eid}_{acc_name}_mean": float(np.mean(a_smooth)),
        f"{eid}_{acc_name}_std": float(np.std(a_smooth)),
        "window_length": window_length,
        "polyorder": polyorder
    }

    # 生成图像
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, q, 'o', markersize=2, alpha=0.5, label='raw q')
    axes[0].plot(t, q_smooth, '-', linewidth=2, label='q_smooth')
    axes[0].set_ylabel('Position')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, v_smooth, '-', linewidth=2, label='v_smooth')
    axes[1].set_ylabel('Velocity')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, a_smooth, '-', linewidth=2, label='a_smooth')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Acceleration')
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(f"Kinematics estimation for {eid} (window={window_length}, poly={polyorder})")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"kinematics_{eid}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    observation = (
        f"运动学估计完成。实验 {eid}：使用 Savitzky-Golay 滤波器（窗口 {window_length}，多项式阶 {polyorder}）从 {source_series_name} 估计了平滑位置 {pos_name}、速度 {vel_name} 和加速度 {acc_name}。\n"
        f"{pos_name}: 最小值 {metrics[f'{eid}_{pos_name}_min']:.6f}, 最大值 {metrics[f'{eid}_{pos_name}_max']:.6f}, "
        f"均值 {metrics[f'{eid}_{pos_name}_mean']:.6f}, 标准差 {metrics[f'{eid}_{pos_name}_std']:.6f}\n"
        f"{vel_name}: 最小值 {metrics[f'{eid}_{vel_name}_min']:.6f}, 最大值 {metrics[f'{eid}_{vel_name}_max']:.6f}, "
        f"均值 {metrics[f'{eid}_{vel_name}_mean']:.6f}, 标准差 {metrics[f'{eid}_{vel_name}_std']:.6f}\n"
        f"{acc_name}: 最小值 {metrics[f'{eid}_{acc_name}_min']:.6f}, 最大值 {metrics[f'{eid}_{acc_name}_max']:.6f}, "
        f"均值 {metrics[f'{eid}_{acc_name}_mean']:.6f}, 标准差 {metrics[f'{eid}_{acc_name}_std']:.6f}\n"
        f"运动学曲线图已保存。"
    )

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [fig_path],
        "metrics": metrics
    }
