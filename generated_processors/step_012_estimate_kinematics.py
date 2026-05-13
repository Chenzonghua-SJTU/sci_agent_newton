import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    params = payload["parameters"]
    exp_id = params["experiment_id"]
    source_series = params["source_series"]
    pos_name = params["position_name"]
    vel_name = params["velocity_name"]
    acc_name = params["acceleration_name"]
    window_length = params["window_length"]
    polyorder = params["polyorder"]
    overwrite = params.get("overwrite", False)

    exp = payload["experiments"][exp_id]
    t = np.array(exp["series"]["t"], dtype=float)
    q = np.array(exp["series"][source_series], dtype=float)

    # 时间步长（优先使用config中的dt）
    config = exp.get("config", {})
    dt = config.get("dt")
    if dt is None or dt <= 0:
        dt = t[1] - t[0] if len(t) > 1 else 0.05

    n = len(t)
    if n < window_length:
        raise ValueError(f"序列长度 {n} 小于窗口长度 {window_length}")

    # Savitzky-Golay 滤波
    q_smooth = savgol_filter(q, window_length, polyorder, deriv=0, mode='interp')
    v = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt, mode='interp')
    a = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt, mode='interp')

    # 构建派生序列
    derived_series = [
        {
            "experiment_id": exp_id,
            "name": pos_name,
            "values": q_smooth.tolist(),
            "source_name": f"Savitzky-Golay平滑{source_series}(w={window_length}, p={polyorder})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"平滑后的位置序列 (window={window_length}, polyorder={polyorder})"
        },
        {
            "experiment_id": exp_id,
            "name": vel_name,
            "values": v.tolist(),
            "source_name": f"Savitzky-Golay一阶导数{source_series}(w={window_length}, p={polyorder})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"估计的速度序列"
        },
        {
            "experiment_id": exp_id,
            "name": acc_name,
            "values": a.tolist(),
            "source_name": f"Savitzky-Golay二阶导数{source_series}(w={window_length}, p={polyorder})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"估计的加速度序列"
        }
    ]

    # 统计指标
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
        "polyorder": polyorder,
    }

    # 构建 observation 文本
    v_min_str = f"{metrics[f'{vel_name}_min']:.6f}"
    v_max_str = f"{metrics[f'{vel_name}_max']:.6f}"
    v_mean_str = f"{metrics[f'{vel_name}_mean']:.6f}"
    v_std_str  = f"{metrics[f'{vel_name}_std']:.6f}"
    a_min_str = f"{metrics[f'{acc_name}_min']:.6f}"
    a_max_str = f"{metrics[f'{acc_name}_max']:.6f}"
    a_mean_str = f"{metrics[f'{acc_name}_mean']:.6f}"
    a_std_str  = f"{metrics[f'{acc_name}_std']:.6f}"

    observation = (
        f"对实验 {exp_id} 使用 Savitzky-Golay 滤波从 {source_series} 估计运动学参数。\n"
        f"参数: window_length={window_length}, polyorder={polyorder}.\n"
        f"平滑后的位置序列 '{pos_name}' 已生成。\n"
        f"速度 '{vel_name}': min={v_min_str}, max={v_max_str}, mean={v_mean_str}, std={v_std_str}.\n"
        f"加速度 '{acc_name}': min={a_min_str}, max={a_max_str}, mean={a_mean_str}, std={a_std_str}."
    )

    # 生成图像
    output_dir = payload["output_dir"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, q, label='原始 ' + source_series, alpha=0.5, linewidth=1)
    axes[0].plot(t, q_smooth, label='平滑后 ' + pos_name, linewidth=2)
    axes[0].set_ylabel('position')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, v, label=vel_name, color='orange')
    axes[1].set_ylabel('velocity')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, a, label=acc_name, color='green')
    axes[2].set_ylabel('acceleration')
    axes[2].set_xlabel('time (s)')
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(f'{exp_id} Kinematic Estimation (window={window_length}, poly={polyorder})')
    plt.tight_layout()
    figure_path = os.path.join(output_dir, f"{exp_id}_kinematics.png")
    plt.savefig(figure_path, dpi=150)
    plt.close(fig)
    figures = [figure_path]

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
