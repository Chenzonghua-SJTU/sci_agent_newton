import numpy as np
from scipy.signal import savgol_filter
from typing import Any, Dict, List, Tuple
import os

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # Validate parameters
    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        # Fallback: if no experiment_ids, use all
        experiment_ids = list(experiments.keys())
    source = params.get("source_series", "q")
    pos_name = params.get("position_name", "q_smooth")
    vel_name = params.get("velocity_name", "v")
    acc_name = params.get("acceleration_name", "a")
    window_length = params.get("window_length", 11)
    polyorder = params.get("polyorder", 2)
    overwrite = params.get("overwrite", False)

    # Validation
    if window_length < polyorder + 1:
        raise ValueError(f"window_length ({window_length}) must be > polyorder ({polyorder})")
    if window_length % 2 == 0:
        raise ValueError(f"window_length ({window_length}) must be odd")

    derived_series = []
    metrics = {}
    observations_parts = []

    for exp_id in experiment_ids:
        if exp_id not in experiments:
            raise ValueError(f"Experiment {exp_id} not found in payload")
        exp = experiments[exp_id]
        config = exp["config"]
        series = exp["series"]

        if source not in series:
            raise ValueError(f"Source series '{source}' not found in experiment {exp_id}")
        if "t" not in series:
            raise ValueError(f"Time series 't' missing in experiment {exp_id}")

        q = np.array(series[source], dtype=float)
        t = np.array(series["t"], dtype=float)
        n = len(q)
        if n == 0:
            raise ValueError(f"Empty series '{source}' in experiment {exp_id}")
        if len(t) != n:
            raise ValueError(f"Length mismatch between t ({len(t)}) and {source} ({n})")

        # Determine dt
        if "dt" in config:
            dt = float(config["dt"])
        else:
            dt = float(np.median(np.diff(t)))
        if dt <= 0:
            raise ValueError(f"Non-positive dt ({dt}) in experiment {exp_id}")

        # Apply Savitzky-Golay filter
        try:
            q_smooth = savgol_filter(q, window_length, polyorder, deriv=0)
            # Velocity: first derivative, need to divide by dt
            v = savgol_filter(q, window_length, polyorder, deriv=1) / dt
            # Acceleration: second derivative, divide by dt**2
            a = savgol_filter(q, window_length, polyorder, deriv=2) / (dt ** 2)
        except Exception as e:
            raise ValueError(f"Savitzky-Golay filter failed for {exp_id}: {e}") from e

        # Build derived series entries
        # For position series
        derived_series.append({
            "experiment_id": exp_id,
            "name": pos_name,
            "values": q_smooth.tolist(),
            "source_name": f"Savitzky-Golay smoothed {source} (w={window_length}, p={polyorder})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Smoothed position using SG filter with window={window_length}, polyorder={polyorder}"
        })

        # For velocity
        derived_series.append({
            "experiment_id": exp_id,
            "name": vel_name,
            "values": v.tolist(),
            "source_name": f"Savitzky-Golay 1st derivative of {source} (w={window_length}, p={polyorder})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Velocity estimated via SG filter derivative"
        })

        # For acceleration
        derived_series.append({
            "experiment_id": exp_id,
            "name": acc_name,
            "values": a.tolist(),
            "source_name": f"Savitzky-Golay 2nd derivative of {source} (w={window_length}, p={polyorder})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Acceleration estimated via SG filter second derivative"
        })

        # Compute summary metrics
        for name, arr in [(pos_name, q_smooth), (vel_name, v), (acc_name, a)]:
            prefix = f"{exp_id}_{name}"
            metrics[f"{prefix}_min"] = float(np.min(arr))
            metrics[f"{prefix}_max"] = float(np.max(arr))
            metrics[f"{prefix}_mean"] = float(np.mean(arr))
            metrics[f"{prefix}_std"] = float(np.std(arr, ddof=0))

        part = (f"实验 {exp_id}: 从原始 {source} 通过 SG 滤波 (窗口 {window_length}, 阶数 {polyorder}) "
                f"估计 平滑位置 {pos_name}, 速度 {vel_name}, 加速度 {acc_name}。"
                f"{pos_name}: 最小值 {metrics[f'{exp_id}_{pos_name}_min']:.4f}, 最大值 {metrics[f'{exp_id}_{pos_name}_max']:.4f}; "
                f"{vel_name}: 最小值 {metrics[f'{exp_id}_{vel_name}_min']:.4f}, 最大值 {metrics[f'{exp_id}_{vel_name}_max']:.4f}; "
                f"{acc_name}: 最小值 {metrics[f'{exp_id}_{acc_name}_min']:.4f}, 最大值 {metrics[f'{exp_id}_{acc_name}_max']:.4f}。")
        observations_parts.append(part)

    observation = "运动学估计完成。\n" + "\n".join(observations_parts)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [],
        "metrics": metrics
    }
