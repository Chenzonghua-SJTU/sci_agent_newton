import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 确定要处理的实验 ID
    exp_id = params.get("experiment_id", "")
    if not exp_id:
        raise ValueError("parameters must contain 'experiment_id' for estimate_kinematics action")
    if exp_id not in experiments:
        raise ValueError(f"Experiment '{exp_id}' not found in payload['experiments']")

    exp_data = experiments[exp_id]
    series = exp_data.get("series", {})
    config = exp_data.get("config", {})
    t = np.array(series.get("t", []))
    q = np.array(series.get(params.get("source_series", "q"), []))

    if len(t) == 0 or len(q) == 0:
        raise ValueError(f"Required series 't' and '{params['source_series']}' not found for {exp_id}")

    window_length = int(params.get("window_length", 11))
    polyorder = int(params.get("polyorder", 3))
    position_name = params.get("position_name", "q_smooth")
    velocity_name = params.get("velocity_name", "v_sg")
    acceleration_name = params.get("acceleration_name", "a_sg")

    if window_length % 2 == 0:
        window_length += 1  # must be odd
    if polyorder >= window_length:
        polyorder = window_length - 1

    # 计算平滑位置、速度、加速度
    q_smooth = savgol_filter(q, window_length, polyorder, deriv=0, mode='nearest')
    v = savgol_filter(q, window_length, polyorder, deriv=1, delta=config.get("dt", t[1]-t[0] if len(t)>1 else 0.1), mode='nearest')
    a = savgol_filter(q, window_length, polyorder, deriv=2, delta=config.get("dt", t[1]-t[0] if len(t)>1 else 0.1), mode='nearest')

    # 统计指标
    def stats(arr, name_prefix=""):
        return {
            f"{name_prefix}min": float(np.min(arr)),
            f"{name_prefix}max": float(np.max(arr)),
            f"{name_prefix}mean": float(np.mean(arr)),
            f"{name_prefix}std": float(np.std(arr)),
            f"{name_prefix}start": float(arr[0]),
            f"{name_prefix}end": float(arr[-1])
        }

    metrics = {}
    metrics.update({f"{exp_id}_{position_name}_" + k: v for k, v in stats(q_smooth).items()})
    metrics.update({f"{exp_id}_{velocity_name}_" + k: v for k, v in stats(v, "v_").items()})
    metrics.update({f"{exp_id}_{acceleration_name}_" + k: v for k, v in stats(a, "a_").items()})
    metrics["window_length"] = window_length
    metrics["polyorder"] = polyorder

    # 保存图像
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, q_smooth, label="q_smooth", color='blue')
    axes[0].set_ylabel("Position")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, v, label="v_sg", color='green')
    axes[1].set_ylabel("Velocity")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, a, label="a_sg", color='red')
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Acceleration")
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(f"Kinematics estimation for {exp_id} (SG window={window_length}, poly={polyorder})")
    figure_path = os.path.join(output_dir, f"kinematics_{exp_id}.png")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close(fig)

    # 派生序列
    derived_series = [
        {
            "experiment_id": exp_id,
            "name": position_name,
            "values": q_smooth.tolist(),
            "source_name": f"savgol_filter(q, window={window_length}, poly={polyorder}, deriv=0)",
            "provenance": f"generated data processor: {action} (step estimate_kinematics)",
            "description": "Smoothed position using Savitzky-Golay filter"
        },
        {
            "experiment_id": exp_id,
            "name": velocity_name,
            "values": v.tolist(),
            "source_name": f"savgol_filter(q, window={window_length}, poly={polyorder}, deriv=1)",
            "provenance": f"generated data processor: {action} (step estimate_kinematics)",
            "description": "Estimated velocity via Savitzky-Golay derivative"
        },
        {
            "experiment_id": exp_id,
            "name": acceleration_name,
            "values": a.tolist(),
            "source_name": f"savgol_filter(q, window={window_length}, poly={polyorder}, deriv=2)",
            "provenance": f"generated data processor: {action} (step estimate_kinematics)",
            "description": "Estimated acceleration via Savitzky-Golay second derivative"
        }
    ]

    # 构建观察字符串
    obs = f"运动学估计完成。实验 {exp_id}：使用 Savitzky-Golay 滤波器（窗口 {window_length}，多项式阶 {polyorder}）从 {params['source_series']} 估计了平滑位置 {position_name}、速度 {velocity_name} 和加速度 {acceleration_name}。\n"
    obs += f"{position_name}: 最小值 {metrics[f'{exp_id}_{position_name}_min']:.6f}, 最大值 {metrics[f'{exp_id}_{position_name}_max']:.6f}, 均值 {metrics[f'{exp_id}_{position_name}_mean']:.6f}, 标准差 {metrics[f'{exp_id}_{position_name}_std']:.6f}\n"
    obs += f"{velocity_name}: 最小值 {metrics[f'{exp_id}_{velocity_name}_v_min']:.6f}, 最大值 {metrics[f'{exp_id}_{velocity_name}_v_max']:.6f}, 均值 {metrics[f'{exp_id}_{velocity_name}_v_mean']:.6f}, 标准差 {metrics[f'{exp_id}_{velocity_name}_v_std']:.6f}\n"
    obs += f"{acceleration_name}: 最小值 {metrics[f'{exp_id}_{acceleration_name}_a_min']:.6f}, 最大值 {metrics[f'{exp_id}_{acceleration_name}_a_max']:.6f}, 均值 {metrics[f'{exp_id}_{acceleration_name}_a_mean']:.6f}, 标准差 {metrics[f'{exp_id}_{acceleration_name}_a_std']:.6f}\n"
    obs += "运动学曲线图已保存。"

    return {
        "observation": obs,
        "derived_series": derived_series,
        "figures": [figure_path],
        "metrics": metrics
    }
