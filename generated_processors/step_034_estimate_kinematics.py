import numpy as np
from scipy.signal import savgol_filter
from typing import List, Dict, Any


def process(payload: dict) -> dict:
    # Extract parameters
    params = payload.get("parameters", {})
    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        # fallback to single experiment_id
        eid = params.get("experiment_id")
        if eid:
            experiment_ids = [eid]
        else:
            # use all experiments from payload
            experiment_ids = list(payload.get("experiments", {}).keys())

    source_series = params.get("source_series", "q")
    position_name = params.get("position_name", "q_smooth")
    velocity_name = params.get("velocity_name", "v_sg")
    acceleration_name = params.get("acceleration_name", "a_sg")
    overwrite = params.get("overwrite", True)

    experiments: Dict[str, Any] = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # Default filter parameters (must be odd, polyorder less than window_length)
    default_window = 11
    polyorder = 3

    derived_series = []
    metrics = {}
    observations_parts = []

    for exp_id in experiment_ids:
        if exp_id not in experiments:
            raise ValueError(f"Experiment {exp_id} not found in payload")

        exp = experiments[exp_id]
        series = exp.get("series", {})
        available = exp.get("available_series", [])

        t = np.array(series.get("t"))
        q = np.array(series.get(source_series))

        if t is None or q is None:
            raise ValueError(f"Missing t or {source_series} series for {exp_id}")

        n = len(t)
        if n < 5:
            raise ValueError(f"Experiment {exp_id} has too few data points ({n})")

        # Determine Savitzky-Golay filter window length
        # Must be odd, >= polyorder+1, and <= n
        window = default_window
        if window % 2 == 0:
            window += 1
        if window > n:
            window = n if n % 2 == 1 else n - 1
        if window < polyorder + 1:
            window = polyorder + 1
            if window % 2 == 0:
                window += 1

        dt = np.mean(np.diff(t))  # assume uniform time step

        # Compute smoothed position, velocity, acceleration
        q_smooth = savgol_filter(q, window, polyorder, deriv=0, delta=dt)
        v = savgol_filter(q, window, polyorder, deriv=1, delta=dt)
        a = savgol_filter(q, window, polyorder, deriv=2, delta=dt)

        # Convert to lists for output
        q_list = q_smooth.tolist()
        v_list = v.tolist()
        a_list = a.tolist()

        # Prepare derived_series entries (always return even if overwrite is False; the system will handle overwrite logic)
        # We always produce the series; the environment will decide whether to write based on overwrite flag
        src_desc = f"Savitzky‑Golay filter (window={window}, polyorder={polyorder}, dt={dt:.6f}) from {source_series}"

        # Position series (may overwrite original q if name equals source_series)
        derived_series.append({
            "experiment_id": exp_id,
            "name": position_name,
            "values": q_list,
            "source_name": f"smoothed {source_series} using SG filter",
            "provenance": "generated data processor: estimate_kinematics",
            "description": src_desc
        })

        derived_series.append({
            "experiment_id": exp_id,
            "name": velocity_name,
            "values": v_list,
            "source_name": f"velocity from SG filter (derivative of {source_series})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": src_desc
        })

        derived_series.append({
            "experiment_id": exp_id,
            "name": acceleration_name,
            "values": a_list,
            "source_name": f"acceleration from SG filter (2nd derivative of {source_series})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": src_desc
        })

        # Collect metrics for observation
        q_min = float(np.min(q_smooth))
        q_max = float(np.max(q_smooth))
        q_mean = float(np.mean(q_smooth))
        v_min = float(np.min(v))
        v_max = float(np.max(v))
        v_mean = float(np.mean(v))
        a_min = float(np.min(a))
        a_max = float(np.max(a))
        a_mean = float(np.mean(a))

        # Store in metrics dict with prefix
        metrics[f"{exp_id}_{position_name}_min"] = q_min
        metrics[f"{exp_id}_{position_name}_max"] = q_max
        metrics[f"{exp_id}_{position_name}_mean"] = q_mean
        metrics[f"{exp_id}_{velocity_name}_min"] = v_min
        metrics[f"{exp_id}_{velocity_name}_max"] = v_max
        metrics[f"{exp_id}_{velocity_name}_mean"] = v_mean
        metrics[f"{exp_id}_{acceleration_name}_min"] = a_min
        metrics[f"{exp_id}_{acceleration_name}_max"] = a_max
        metrics[f"{exp_id}_{acceleration_name}_mean"] = a_mean

        observations_parts.append(
            f"{exp_id}: 使用 Savitzky‑Golay 滤波（窗口={window}, "
            f"多项式阶数={polyorder}, dt={dt:.6f}）从 {source_series} 估计出平滑位置 {position_name}、"
            f"速度 {velocity_name}、加速度 {acceleration_name}。\n"
            f"  {position_name}: min={q_min:.6f}, max={q_max:.6f}, mean={q_mean:.6f}\n"
            f"  {velocity_name}: min={v_min:.6f}, max={v_max:.6f}, mean={v_mean:.6f}\n"
            f"  {acceleration_name}: min={a_min:.6f}, max={a_max:.6f}, mean={a_mean:.6f}"
        )

    observation = "\n".join(observations_parts)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [],
        "metrics": metrics
    }
