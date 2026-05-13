import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # Determine which experiments to process
    if "experiment_id" in params:
        exp_ids = [params["experiment_id"]]
    elif "experiment_ids" in params:
        exp_ids = params["experiment_ids"]
    else:
        exp_ids = list(experiments.keys())

    # Parameters for Savitzky-Golay filter
    window_length = params.get("window_length", 21)
    polyorder = params.get("polyorder", 3)
    source_series = params.get("source_series", "q")
    position_name = params.get("position_name", "q_smooth")
    velocity_name = params.get("velocity_name", "v_est")
    acceleration_name = params.get("acceleration_name", "a_est")
    overwrite = params.get("overwrite", False)

    derived_series = []
    figures = []
    metrics = {}
    observations = []

    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment '{eid}' not found in payload.")
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        available = exp["available_series"]

        # Get time and position data
        if "t" not in series:
            raise ValueError(f"Experiment '{eid}' has no time series 't'.")
        t = np.array(series["t"])
        if source_series not in series:
            raise ValueError(f"Source series '{source_series}' not available for experiment '{eid}'. Available: {available}")
        q = np.array(series[source_series])

        # Determine dt
        if len(t) < 2:
            raise ValueError(f"Experiment '{eid}' time series too short.")
        if config.get("dt") is not None:
            dt = config["dt"]
        else:
            dt = np.mean(np.diff(t))
        # Safeguard: ensure dt is positive and consistent
        if dt <= 0 or np.std(np.diff(t)) / dt > 0.01:
            raise ValueError(f"Non-uniform or invalid time step for experiment '{eid}': dt={dt}, std_diff={np.std(np.diff(t))}")

        # Validate window length
        n_points = len(q)
        if window_length < polyorder + 2:
            window_length = polyorder + 2
        if window_length > n_points:
            window_length = n_points if n_points % 2 == 1 else n_points - 1
        if window_length < polyorder + 2:
            raise ValueError(f"Not enough points ({n_points}) for given window_length and polyorder.")
        if window_length % 2 == 0:
            window_length += 1  # must be odd

        # Compute smoothed position, velocity, acceleration
        q_smooth = savgol_filter(q, window_length, polyorder)
        v = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt)
        a = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)

        # Prepare derived series
        derived_series.append({
            "experiment_id": eid,
            "name": position_name,
            "values": q_smooth.tolist(),
            "source_name": f"Savitzky-Golay smooth of {source_series} (window={window_length}, poly={polyorder})",
            "provenance": f"generated data processor: estimate_kinematics",
            "description": f"Smoothed position from {source_series}"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": velocity_name,
            "values": v.tolist(),
            "source_name": f"Savitzky-Golay 1st derivative of {source_series} (window={window_length}, poly={polyorder})",
            "provenance": f"generated data processor: estimate_kinematics",
            "description": f"Estimated velocity from {source_series}"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": acceleration_name,
            "values": a.tolist(),
            "source_name": f"Savitzky-Golay 2nd derivative of {source_series} (window={window_length}, poly={polyorder})",
            "provenance": f"generated data processor: estimate_kinematics",
            "description": f"Estimated acceleration from {source_series}"
        })

        # Compute statistics
        v_mean = np.mean(v)
        v_std = np.std(v)
        a_mean = np.mean(a)
        a_std = np.std(a)
        metrics[f"{eid}_v_mean"] = float(v_mean)
        metrics[f"{eid}_v_std"] = float(v_std)
        metrics[f"{eid}_a_mean"] = float(a_mean)
        metrics[f"{eid}_a_std"] = float(a_std)

        observations.append(
            f"实验 {eid}: 点数={n_points}, dt={dt:.4f}\n"
            f"  速度: 均值={v_mean:.6f}, 标准差={v_std:.6f}\n"
            f"  加速度: 均值={a_mean:.6f}, 标准差={a_std:.6f}"
        )

        # Plot kinematics
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t, q, 'b.', markersize=1, alpha=0.5, label=f'raw {source_series}')
        axes[0].plot(t, q_smooth, 'r-', lw=1.5, label=f'{position_name} (smoothed)')
        axes[0].set_ylabel('Position')
        axes[0].legend(loc='best')
        axes[0].grid(True)

        axes[1].plot(t, v, 'g-', lw=1.5, label=velocity_name)
        axes[1].set_ylabel('Velocity')
        axes[1].legend(loc='best')
        axes[1].grid(True)

        axes[2].plot(t, a, 'm-', lw=1.5, label=acceleration_name)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Acceleration')
        axes[2].legend(loc='best')
        axes[2].grid(True)

        fig.suptitle(f'Kinematics Estimation for Experiment {eid}')
        fig.tight_layout()

        # Save figure
        fname = os.path.join(output_dir, f"{eid}_kinematics.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        figures.append(fname)

    observation_text = "运动学估计完成。\n" + "\n".join(observations)
    return {
        "observation": observation_text,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
