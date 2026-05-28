import json
import math
import statistics
from itertools import accumulate
from functools import partial
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn import linear_model, metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process(payload: dict) -> dict:
    action = payload["action"]
    parameters = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = Path(payload["output_dir"])

    # ---- validate action ----
    if action != "estimate_kinematics":
        raise ValueError(f"Unsupported action: {action}, expected 'estimate_kinematics'")

    # ---- parse parameters ----
    experiment_id = parameters.get("experiment_id", "")
    source_series = parameters.get("source_series", "q")
    position_name = parameters.get("position_name", "q_smooth")
    velocity_name = parameters.get("velocity_name", "v_sg")
    acceleration_name = parameters.get("acceleration_name", "a_sg")
    window_length = int(parameters.get("window_length", 7))
    polyorder = int(parameters.get("polyorder", 2))
    overwrite = parameters.get("overwrite", False)

    # ---- check experiment exists ----
    if experiment_id not in experiments:
        raise ValueError(f"Experiment '{experiment_id}' not found in payload")
    exp = experiments[experiment_id]

    # ---- check source series ----
    if source_series not in exp.get("available_series", []):
        raise ValueError(f"Source series '{source_series}' not available in experiment {experiment_id}")
    q = np.array(exp["series"][source_series], dtype=float)
    t = np.array(exp["series"]["t"], dtype=float)
    n = len(t)
    if len(q) != n:
        raise ValueError(f"Length mismatch: {source_series} ({len(q)}) vs t ({n})")

    # ---- validate SG parameters ----
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")
    if window_length >= n:
        raise ValueError(f"window_length ({window_length}) must be less than data length ({n})")
    if polyorder >= window_length:
        raise ValueError(f"polyorder ({polyorder}) must be less than window_length ({window_length})")

    # ---- decide whether to compute ----
    derived_series = []
    existing_series = exp.get("available_series", [])
    needs_q_smooth = overwrite or (position_name not in existing_series)
    needs_v = overwrite or (velocity_name not in existing_series)
    needs_a = overwrite or (acceleration_name not in existing_series)

    # ---- compute Savitzky-Golay derivatives ----
    # Savitzky-Golay: deriv=0 -> smoothed; deriv=1 -> velocity (dq/dt); deriv=2 -> acceleration (d²q/dt²)
    # The filter returns values scaled by dt? Actually savgol_filter uses the x-axis only for spacing.
    # We must divide by dt^deriv manually because the filter assumes unit spacing.
    # The correct way: use savgol_coeffs to get coefficients, then apply convolution, scaling by dt.
    # To avoid edge artifacts at boundaries, we use savgol_filter with 'nearest' mode.
    dt_avg = t[1] - t[0] if n > 1 else 1.0
    # Check uniform spacing (optional robustness)
    if n > 1:
        dt_est = np.mean(np.diff(t))
        if abs(dt_est - dt_avg) > 1e-12:
            raise ValueError("Time series is not uniformly spaced; SG filter assumes uniform spacing.")
    dt = dt_avg

    q_smooth = None
    v = None
    a = None

    if needs_q_smooth:
        q_smooth = signal.savgol_filter(q, window_length, polyorder, deriv=0, mode='nearest')
        # Ensure no boundaries become NaN or inf
        q_smooth = np.nan_to_num(q_smooth, nan=0.0, posinf=0.0, neginf=0.0)

    if needs_v:
        # derivative numerator is dq, denominator dt. SG filter returns dq/d(step) assuming unit step.
        v_raw = signal.savgol_filter(q, window_length, polyorder, deriv=1, mode='nearest')
        v = v_raw / dt

    if needs_a:
        a_raw = signal.savgol_filter(q, window_length, polyorder, deriv=2, mode='nearest')
        a = a_raw / (dt * dt)  # second derivative requires dt^2

    # ---- build derived_series list ----
    if needs_q_smooth and q_smooth is not None:
        derived_series.append({
            "experiment_id": experiment_id,
            "name": position_name,
            "values": q_smooth.tolist(),
            "source_name": f"SG filter (window={window_length}, poly={polyorder}) on {source_series}",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Smoothed position from {source_series} using Savitzky-Golay filter"
        })
    if needs_v and v is not None:
        derived_series.append({
            "experiment_id": experiment_id,
            "name": velocity_name,
            "values": v.tolist(),
            "source_name": f"SG filter deriv=1 on {source_series} (scaled by 1/dt)",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Velocity estimated from {source_series} using Savitzky-Golay filter"
        })
    if needs_a and a is not None:
        derived_series.append({
            "experiment_id": experiment_id,
            "name": acceleration_name,
            "values": a.tolist(),
            "source_name": f"SG filter deriv=2 on {source_series} (scaled by 1/dt²)",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Acceleration estimated from {source_series} using Savitzky-Golay filter"
        })

    # ---- compute statistics for each new series ----
    stats_dict = {}
    series_to_describe = {}
    if needs_q_smooth and q_smooth is not None:
        series_to_describe[position_name] = q_smooth
    if needs_v and v is not None:
        series_to_describe[velocity_name] = v
    if needs_a and a is not None:
        series_to_describe[acceleration_name] = a

    for sname, arr in series_to_describe.items():
        # linear regression to get slope
        slope, intercept, _, _, _ = stats.linregress(t, arr)
        stats_dict[sname] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "start": float(arr[0]),
            "end": float(arr[-1]),
            "slope": float(slope)
        }

    # ---- build observation ----
    obs_parts = [f"estimate_kinematics for {experiment_id} using SG filter (window={window_length}, poly={polyorder}) on source '{source_series}'."]
    for sname, st in stats_dict.items():
        obs_parts.append(f"  {sname}: min={st['min']:.6f}, max={st['max']:.6f}, mean={st['mean']:.6f}, std={st['std']:.6f}, start={st['start']:.6f}, end={st['end']:.6f}, slope={st['slope']:.6f}")
    observation = "\n".join(obs_parts)

    # ---- optionally save a figure showing the new series ----
    figures = []
    if any(series_to_describe):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        colors = ['blue', 'green', 'red']
        labels = [position_name, velocity_name, acceleration_name]
        for idx, (sname, arr) in enumerate(series_to_describe.items()):
            ax = axes[idx]
            ax.plot(t, arr, color=colors[idx], label=sname)
            ax.set_ylabel(sname)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Kinematics estimates for {experiment_id}")
        fig.tight_layout()
        fig_path = output_dir / f"kinematics_{experiment_id}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        figures.append(str(fig_path))

    # ---- prepare metrics ----
    metrics_output = {}
    for sname, st in stats_dict.items():
        metrics_output[f"{sname}_min"] = st["min"]
        metrics_output[f"{sname}_max"] = st["max"]
        metrics_output[f"{sname}_mean"] = st["mean"]
        metrics_output[f"{sname}_std"] = st["std"]
        metrics_output[f"{sname}_start"] = st["start"]
        metrics_output[f"{sname}_end"] = st["end"]
        metrics_output[f"{sname}_slope"] = st["slope"]

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics_output
    }
