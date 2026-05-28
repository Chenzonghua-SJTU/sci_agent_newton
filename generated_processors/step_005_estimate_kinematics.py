import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # Extract parameters (support both experiment_id list and single)
    exp_ids = params.get("experiment_ids")
    if exp_ids is None:
        single_id = params.get("experiment_id")
        if single_id:
            exp_ids = [single_id]
        else:
            exp_ids = list(experiments.keys())

    source_series = params["source_series"]         # "q"
    pos_name = params["position_name"]               # "q_smooth"
    vel_name = params["velocity_name"]               # "v"
    acc_name = params["acceleration_name"]           # "a"
    window_length = params.get("window_length", 11)
    polyorder = params.get("polyorder", 2)
    overwrite = params.get("overwrite", True)

    derived_series = []
    metrics_dict = {}
    figures = []
    observation_parts = []

    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload experiments.")
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        available = exp.get("available_series", [])

        # Check source series exists
        if source_series not in series:
            raise ValueError(f"Source series '{source_series}' not available in experiment {eid}. Available: {available}")
        if "t" not in series:
            raise ValueError(f"Time series 't' not available in experiment {eid}.")

        t = np.array(series["t"], dtype=float)
        q = np.array(series[source_series], dtype=float)
        dt = config.get("dt", 0.1)   # fallback to 0.1 if not present

        n = len(t)
        if len(q) != n:
            raise ValueError(f"Length of {source_series} ({len(q)}) does not match length of t ({n}) in experiment {eid}.")

        # Check window length
        if window_length > n:
            raise ValueError(f"window_length ({window_length}) is larger than data length ({n}) in experiment {eid}.")
        if window_length % 2 == 0:
            raise ValueError(f"window_length must be odd, got {window_length}.")

        # Apply Savitzky-Golay filter
        q_smooth = savgol_filter(q, window_length, polyorder, deriv=0)
        v = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt)
        a = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)

        # Derive series dicts (even if overwrite=False, we return new values; user will decide)
        derived_series.append({
            "experiment_id": eid,
            "name": pos_name,
            "values": q_smooth.tolist(),
            "source_name": f"savgol_filter({source_series}, window={window_length}, polyorder={polyorder}, deriv=0)",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Smoothed position from {source_series}"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": vel_name,
            "values": v.tolist(),
            "source_name": f"savgol_filter({source_series}, window={window_length}, polyorder={polyorder}, deriv=1, delta={dt})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": "Velocity (1st derivative of smoothed position)"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": acc_name,
            "values": a.tolist(),
            "source_name": f"savgol_filter({source_series}, window={window_length}, polyorder={polyorder}, deriv=2, delta={dt})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": "Acceleration (2nd derivative of smoothed position)"
        })

        # Compute smoothness metric: RMSE between original q and smoothed q_smooth
        rmse = np.sqrt(np.mean((q - q_smooth)**2))
        metrics_dict.setdefault("smooth_rmse_mean", 0.0)
        # We'll update later as mean over experiments
        if "smooth_rmse_list" not in metrics_dict:
            metrics_dict["smooth_rmse_list"] = []
        metrics_dict["smooth_rmse_list"].append(rmse)

        # Build observation part
        obs_part = (f"实验{eid}: 使用参数 window_length={window_length}, polyorder={polyorder}, "
                    f"dt={dt}。平滑后 {pos_name} 与原始 {source_series} 的 RMSE={rmse:.6g}。 "
                    f"已生成 {pos_name}, {vel_name}, {acc_name}。")
        observation_parts.append(obs_part)

        # Generate figure
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(t, q, 'b-', label=f'original {source_series}')
        axes[0].plot(t, q_smooth, 'r--', label=f'smoothed {pos_name}')
        axes[0].set_ylabel('Position')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(t, v, 'g-', label=vel_name)
        axes[1].set_ylabel('Velocity')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(t, a, 'm-', label=acc_name)
        axes[2].set_ylabel('Acceleration')
        axes[2].set_xlabel('Time')
        axes[2].legend()
        axes[2].grid(True)

        fig.suptitle(f'Experiment {eid}: Kinematics estimation from {source_series}')
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"kinematics_{eid}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        figures.append(fig_path)

    # Finalize metrics
    rmse_list = metrics_dict.get("smooth_rmse_list", [])
    if rmse_list:
        metrics_dict["smooth_rmse_mean"] = float(np.mean(rmse_list))
        metrics_dict["smooth_rmse_std"] = float(np.std(rmse_list))
    else:
        metrics_dict["smooth_rmse_mean"] = 0.0
        metrics_dict["smooth_rmse_std"] = 0.0

    observation = "；".join(observation_parts)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics_dict
    }
