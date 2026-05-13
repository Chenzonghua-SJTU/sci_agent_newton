import numpy as np
from scipy.signal import savgol_filter

def process(payload: dict) -> dict:
    params = payload["parameters"]
    exp_id = params["experiment_id"]
    source_series = params["source_series"]
    velocity_name = params["velocity_name"]
    acceleration_name = params["acceleration_name"]
    window_length = params["window_length"]
    polyorder = params["polyorder"]

    if exp_id not in payload["experiments"]:
        raise ValueError(f"Experiment {exp_id} not found in payload")
    exp = payload["experiments"][exp_id]

    if source_series not in exp["series"]:
        raise ValueError(f"Source series '{source_series}' not found in experiment {exp_id}")
    q = np.array(exp["series"][source_series], dtype=float)
    t = np.array(exp["series"]["t"], dtype=float)

    if len(q) < window_length:
        raise ValueError(f"Series length {len(q)} is less than window_length {window_length}")
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")
    if polyorder >= window_length:
        raise ValueError(f"polyorder {polyorder} must be less than window_length {window_length}")

    dt = np.median(np.diff(t)) if len(t) > 1 else 0.1
    if dt <= 0:
        raise ValueError(f"Non-positive dt computed from time series: {dt}")

    q_smooth = savgol_filter(q, window_length, polyorder, deriv=0)
    v = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt)
    a = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)

    derived_series = [
        {
            "experiment_id": exp_id,
            "name": "q_smooth",
            "values": q_smooth.tolist(),
            "source_name": f"Savitzky-Golay filter (win={window_length}, order={polyorder}) from {source_series}",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Smoothed position from {source_series}"
        },
        {
            "experiment_id": exp_id,
            "name": velocity_name,
            "values": v.tolist(),
            "source_name": f"Savitzky-Golay filter (win={window_length}, order={polyorder}, deriv=1, dt={dt}) from {source_series}",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Velocity estimated from {source_series}"
        },
        {
            "experiment_id": exp_id,
            "name": acceleration_name,
            "values": a.tolist(),
            "source_name": f"Savitzky-Golay filter (win={window_length}, order={polyorder}, deriv=2, dt={dt}) from {source_series}",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Acceleration estimated from {source_series}"
        }
    ]

    metrics = {
        f"{exp_id}_q_smooth_min": float(np.min(q_smooth)),
        f"{exp_id}_q_smooth_max": float(np.max(q_smooth)),
        f"{exp_id}_q_smooth_mean": float(np.mean(q_smooth)),
        f"{exp_id}_{velocity_name}_min": float(np.min(v)),
        f"{exp_id}_{velocity_name}_max": float(np.max(v)),
        f"{exp_id}_{velocity_name}_mean": float(np.mean(v)),
        f"{exp_id}_{acceleration_name}_min": float(np.min(a)),
        f"{exp_id}_{acceleration_name}_max": float(np.max(a)),
        f"{exp_id}_{acceleration_name}_mean": float(np.mean(a))
    }

    observation = (
        f"对实验 {exp_id} 使用 Savitzky-Golay 滤波（窗口长度={window_length}, 多项式阶数={polyorder}, dt={dt}) "
        f"从 {source_series} 估计出平滑位置 q_smooth、速度 {velocity_name}、加速度 {acceleration_name}。\n"
        f"  q_smooth: min={metrics[f'{exp_id}_q_smooth_min']:.6f}, "
        f"max={metrics[f'{exp_id}_q_smooth_max']:.6f}, "
        f"mean={metrics[f'{exp_id}_q_smooth_mean']:.6f}\n"
        f"  {velocity_name}: min={metrics[f'{exp_id}_{velocity_name}_min']:.6f}, "
        f"max={metrics[f'{exp_id}_{velocity_name}_max']:.6f}, "
        f"mean={metrics[f'{exp_id}_{velocity_name}_mean']:.6f}\n"
        f"  {acceleration_name}: min={metrics[f'{exp_id}_{acceleration_name}_min']:.6f}, "
        f"max={metrics[f'{exp_id}_{acceleration_name}_max']:.6f}, "
        f"mean={metrics[f'{exp_id}_{acceleration_name}_mean']:.6f}"
    )

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [],
        "metrics": metrics
    }
