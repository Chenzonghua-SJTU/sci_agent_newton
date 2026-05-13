import numpy as np
from scipy.signal import savgol_filter

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload.get("output_dir", "")

    exp_id = params["experiment_id"]
    source_series = params["source_series"]
    position_name = params["position_name"]
    velocity_name = params["velocity_name"]
    acceleration_name = params["acceleration_name"]
    window_length = params["window_length"]
    polyorder = params["polyorder"]
    # overwrite flag is noted but not used here; framework handles overwriting

    if exp_id not in experiments:
        raise ValueError(f"Experiment {exp_id} not found in payload")
    exp = experiments[exp_id]
    config = exp["config"]
    series = exp["series"]

    if source_series not in series:
        raise ValueError(f"Source series '{source_series}' not available in experiment {exp_id}. "
                         f"Available: {list(series.keys())}")
    if "t" not in series:
        raise ValueError("Time series 't' is required but not found in experiment data")

    q = np.array(series[source_series])
    t = np.array(series["t"])
    if len(q) != len(t):
        raise ValueError(f"Length mismatch: source series has {len(q)} points, time series has {len(t)} points")
    dt = config.get("dt", t[1] - t[0])  # fallback estimate

    # Savitzky-Golay filter: position (deriv=0), velocity (deriv=1), acceleration (deriv=2)
    q_smooth = savgol_filter(q, window_length, polyorder, deriv=0)
    v_sg = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt)
    a_sg = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)

    derived_series = [
        {
            "experiment_id": exp_id,
            "name": position_name,
            "values": q_smooth.tolist(),
            "source_name": f"Savitzky-Golay filter (win={window_length}, order={polyorder}) from {source_series}",
            "provenance": f"generated data processor: {action}",
            "description": "Smoothed position via SG filter (deriv=0)"
        },
        {
            "experiment_id": exp_id,
            "name": velocity_name,
            "values": v_sg.tolist(),
            "source_name": f"Savitzky-Golay filter (win={window_length}, order={polyorder}, deriv=1, dt={dt}) from {source_series}",
            "provenance": f"generated data processor: {action}",
            "description": "Estimated velocity via SG filter first derivative"
        },
        {
            "experiment_id": exp_id,
            "name": acceleration_name,
            "values": a_sg.tolist(),
            "source_name": f"Savitzky-Golay filter (win={window_length}, order={polyorder}, deriv=2, dt={dt}) from {source_series}",
            "provenance": f"generated data processor: {action}",
            "description": "Estimated acceleration via SG filter second derivative"
        }
    ]

    fmt = lambda x: f"{x:.6f}"
    observation = (
        f"对实验 {exp_id} 使用 Savitzky-Golay 滤波（窗口长度={window_length}, 多项式阶数={polyorder}, dt={dt}) "
        f"从 {source_series} 估计出平滑位置 {position_name}、速度 {velocity_name}、加速度 {acceleration_name}。\n"
        f"  {position_name}: min={fmt(q_smooth.min())}, max={fmt(q_smooth.max())}, mean={fmt(q_smooth.mean())}\n"
        f"  {velocity_name}: min={fmt(v_sg.min())}, max={fmt(v_sg.max())}, mean={fmt(v_sg.mean())}\n"
        f"  {acceleration_name}: min={fmt(a_sg.min())}, max={fmt(a_sg.max())}, mean={fmt(a_sg.mean())}"
    )

    metrics = {
        f"{exp_id}_{position_name}_min": float(q_smooth.min()),
        f"{exp_id}_{position_name}_max": float(q_smooth.max()),
        f"{exp_id}_{position_name}_mean": float(q_smooth.mean()),
        f"{exp_id}_{velocity_name}_min": float(v_sg.min()),
        f"{exp_id}_{velocity_name}_max": float(v_sg.max()),
        f"{exp_id}_{velocity_name}_mean": float(v_sg.mean()),
        f"{exp_id}_{acceleration_name}_min": float(a_sg.min()),
        f"{exp_id}_{acceleration_name}_max": float(a_sg.max()),
        f"{exp_id}_{acceleration_name}_mean": float(a_sg.mean()),
    }

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [],
        "metrics": metrics
    }
