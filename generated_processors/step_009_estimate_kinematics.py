import numpy as np
from scipy.signal import savgol_filter

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    experiment_id = params.get("experiment_id")
    source_series = params.get("source_series", "q")
    pos_name = params.get("position_name", "q_smooth")
    vel_name = params.get("velocity_name", "v")
    acc_name = params.get("acceleration_name", "a")
    window_length = params.get("window_length", 11)
    polyorder = params.get("polyorder", 2)
    overwrite = params.get("overwrite", False)

    if experiment_id not in experiments:
        raise ValueError(f"Experiment {experiment_id} not found.")
    exp = experiments[experiment_id]

    series = exp.get("series", {})
    if source_series not in series:
        raise ValueError(f"Source series '{source_series}' not available for experiment {experiment_id}.")
    if "t" not in series:
        raise ValueError(f"Time series 't' not available for experiment {experiment_id}.")

    t = np.array(series["t"])
    q = np.array(series[source_series])

    if len(q) != len(t):
        raise ValueError(f"Length mismatch: source series has {len(q)} points, time series has {len(t)} points.")

    dt = t[1] - t[0]

    q_smooth = savgol_filter(q, window_length, polyorder, deriv=0)
    v = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt)
    a = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)

    derived_series = []
    for name, values, desc in [
        (pos_name, q_smooth, "Smoothed position"),
        (vel_name, v, "Velocity"),
        (acc_name, a, "Acceleration")
    ]:
        derived_series.append({
            "experiment_id": experiment_id,
            "name": name,
            "values": values.tolist(),
            "source_name": f"SG derivative of {source_series} (window={window_length}, polyorder={polyorder})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": desc
        })

    metrics = {}
    for label, arr in [("q_smooth", q_smooth), ("v", v), ("a", a)]:
        metrics[f"{experiment_id}_{label}_min"] = float(np.min(arr))
        metrics[f"{experiment_id}_{label}_max"] = float(np.max(arr))
        metrics[f"{experiment_id}_{label}_mean"] = float(np.mean(arr))
        metrics[f"{experiment_id}_{label}_std"] = float(np.std(arr))

    observation = (
        f"运动学估计完成。\n"
        f"实验 {experiment_id}: 原始 {source_series} 范围 [{np.min(q):.6f}, {np.max(q):.6f}] "
        f"平滑位置 {pos_name}: min={np.min(q_smooth):.6f}, max={np.max(q_smooth):.6f} "
        f"速度 {vel_name}: min={np.min(v):.6f}, max={np.max(v):.6f} "
        f"加速度 {acc_name}: min={np.min(a):.6f}, max={np.max(a):.6f}"
    )

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [],
        "metrics": metrics
    }
