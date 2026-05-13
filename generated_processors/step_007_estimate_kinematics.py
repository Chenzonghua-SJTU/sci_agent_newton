import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # Extract parameters
    experiment_id = params.get("experiment_id", "exp_04")
    source_series = params.get("source_series", "q")
    velocity_name = params.get("velocity_name", "v_smooth")
    acceleration_name = params.get("acceleration_name", "a_smooth")
    window_length = params.get("window_length", 11)
    polyorder = params.get("polyorder", 3)
    overwrite = params.get("overwrite", True)

    # Validate experiment exists
    if experiment_id not in experiments:
        raise ValueError(f"Experiment {experiment_id} not found in payload.")
    exp = experiments[experiment_id]

    # Extract series
    series = exp.get("series", {})
    if source_series not in series:
        raise ValueError(f"Source series '{source_series}' not found in experiment {experiment_id}.")
    q = np.array(series[source_series], dtype=float)
    t = np.array(series.get("t", None))
    if t is None:
        raise ValueError("Time series 't' not found in experiment.")
    dt = t[1] - t[0] if len(t) > 1 else 1.0

    # Validate window_length
    if window_length > len(q):
        raise ValueError(f"window_length ({window_length}) > length of source series ({len(q)}).")
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd.")
    if polyorder >= window_length:
        raise ValueError(f"polyorder ({polyorder}) must be less than window_length ({window_length}).")

    # Savitzky-Golay smoothing and derivatives
    q_smooth = savgol_filter(q, window_length, polyorder, deriv=0, delta=dt)
    v_smooth = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt)
    a_smooth = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)

    # Compute statistics
    v_min = float(np.min(v_smooth))
    v_max = float(np.max(v_smooth))
    v_mean = float(np.mean(v_smooth))
    v_std = float(np.std(v_smooth))
    a_min = float(np.min(a_smooth))
    a_max = float(np.max(a_smooth))
    a_mean = float(np.mean(a_smooth))
    a_std = float(np.std(a_smooth))

    # Build observation
    observation = (
        f"对实验 {experiment_id} 使用 Savitzky-Golay 滤波 (窗口={window_length}, 多项式阶数={polyorder}) "
        f"从 {source_series} 估计平滑位置、速度和加速度序列。\n"
        f"速度 v_smooth: min={v_min:.6f}, max={v_max:.6f}, mean={v_mean:.6f}, std={v_std:.6f}\n"
        f"加速度 a_smooth: min={a_min:.6f}, max={a_max:.6f}, mean={a_mean:.6f}, std={a_std:.6f}\n"
        f"已生成派生序列 q_smooth, {velocity_name}, {acceleration_name}。"
    )

    # Prepare derived series
    q_smooth_name = source_series + "_smooth"
    derived_series = [
        {
            "experiment_id": experiment_id,
            "name": q_smooth_name,
            "values": q_smooth.tolist(),
            "source_name": f"Savitzky-Golay filter from {source_series}",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Smoothed {source_series} (window={window_length}, polyorder={polyorder})"
        },
        {
            "experiment_id": experiment_id,
            "name": velocity_name,
            "values": v_smooth.tolist(),
            "source_name": f"First derivative of {source_series} (Savitzky-Golay)",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Velocity from {source_series} using Savitzky-Golay derivative"
        },
        {
            "experiment_id": experiment_id,
            "name": acceleration_name,
            "values": a_smooth.tolist(),
            "source_name": f"Second derivative of {source_series} (Savitzky-Golay)",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Acceleration from {source_series} using Savitzky-Golay derivative"
        }
    ]

    # Generate figure
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(t, q, 'o', markersize=2, label='raw q')
    axes[0].plot(t, q_smooth, '-', label='smooth q')
    axes[0].set_ylabel('q')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, v_smooth, '-', label='v_smooth')
    axes[1].set_ylabel('v')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, a_smooth, '-', label='a_smooth')
    axes[2].set_ylabel('a')
    axes[2].set_xlabel('t')
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(f"{experiment_id}: Kinematics (Savitzky-Golay window={window_length}, polyorder={polyorder})")
    figure_filename = f"{experiment_id}_kinematics_savgol.png"
    figure_path = os.path.join(output_dir, figure_filename)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()

    # Metrics
    metrics = {
        f"{experiment_id}_v_mean": v_mean,
        f"{experiment_id}_v_std": v_std,
        f"{experiment_id}_v_min": v_min,
        f"{experiment_id}_v_max": v_max,
        f"{experiment_id}_a_mean": a_mean,
        f"{experiment_id}_a_std": a_std,
        f"{experiment_id}_a_min": a_min,
        f"{experiment_id}_a_max": a_max,
        "window_length": window_length,
        "polyorder": polyorder,
        "method": "Savitzky-Golay"
    }

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [figure_path],
        "metrics": metrics
    }
