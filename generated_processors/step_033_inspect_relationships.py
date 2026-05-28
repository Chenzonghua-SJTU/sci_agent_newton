import json
import math
import statistics
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    # Extract parameters
    params = payload["parameters"]
    experiment_ids = params.get("experiment_ids", [])
    x_series = params["x_series"]
    y_series = params["y_series"]
    output_dir = Path(payload["output_dir"])
    experiments = payload["experiments"]
    
    # Validate experiment IDs
    available_ids = list(experiments.keys())
    for eid in experiment_ids:
        if eid not in available_ids:
            raise ValueError(f"Experiment {eid} not found in payload")
        exp = experiments[eid]
        available = exp.get("available_series", list(exp["series"].keys()))
        if x_series not in available:
            raise ValueError(f"Experiment {eid}: missing required series '{x_series}'")
        if y_series not in available:
            raise ValueError(f"Experiment {eid}: missing required series '{y_series}'")
    
    # Prepare results
    observation_lines = []
    metrics = {}
    figures = []
    
    for eid in experiment_ids:
        exp = experiments[eid]
        t = np.array(exp["series"]["t"])
        x = np.array(exp["series"][x_series])
        y = np.array(exp["series"][y_series])
        
        # Basic statistics
        x_stats = {
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)),
            "start": float(x[0]),
            "end": float(x[-1]),
            "slope": float(np.polyfit(t, x, deg=1)[0])
        }
        y_stats = {
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "mean": float(np.mean(y)),
            "std": float(np.std(y, ddof=1)),
            "start": float(y[0]),
            "end": float(y[-1]),
            "slope": float(np.polyfit(t, y, deg=1)[0])
        }
        
        # Pearson correlation
        r, p_value = stats.pearsonr(x, y)
        
        # Build observation line
        line = (
            f"{eid}: {x_series} vs {y_series} | "
            f"x: min={x_stats['min']:.6f}, max={x_stats['max']:.6f}, mean={x_stats['mean']:.6f}, std={x_stats['std']:.6f}, "
            f"start={x_stats['start']:.6f}, end={x_stats['end']:.6f}, slope={x_stats['slope']:.6f} | "
            f"y: min={y_stats['min']:.6f}, max={y_stats['max']:.6f}, mean={y_stats['mean']:.6f}, std={y_stats['std']:.6f}, "
            f"start={y_stats['start']:.6f}, end={y_stats['end']:.6f}, slope={y_stats['slope']:.6f} | "
            f"Pearson r={r:.6f}, p={p_value:.2e}"
        )
        observation_lines.append(line)
        
        # Store metrics
        metrics[f"{eid}_{x_series}_min"] = x_stats["min"]
        metrics[f"{eid}_{x_series}_max"] = x_stats["max"]
        metrics[f"{eid}_{x_series}_mean"] = x_stats["mean"]
        metrics[f"{eid}_{x_series}_std"] = x_stats["std"]
        metrics[f"{eid}_{x_series}_start"] = x_stats["start"]
        metrics[f"{eid}_{x_series}_end"] = x_stats["end"]
        metrics[f"{eid}_{x_series}_slope"] = x_stats["slope"]
        metrics[f"{eid}_{y_series}_min"] = y_stats["min"]
        metrics[f"{eid}_{y_series}_max"] = y_stats["max"]
        metrics[f"{eid}_{y_series}_mean"] = y_stats["mean"]
        metrics[f"{eid}_{y_series}_std"] = y_stats["std"]
        metrics[f"{eid}_{y_series}_start"] = y_stats["start"]
        metrics[f"{eid}_{y_series}_end"] = y_stats["end"]
        metrics[f"{eid}_{y_series}_slope"] = y_stats["slope"]
        metrics[f"{eid}_pearson_r"] = r
        metrics[f"{eid}_pearson_p"] = p_value
        
        # Plot scatter
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, s=8, alpha=0.7, label=f"{y_series} vs {x_series}")
        ax.set_xlabel(x_series)
        ax.set_ylabel(y_series)
        ax.set_title(f"{eid}: {y_series} vs {x_series} (r={r:.4f})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig_path = output_dir / f"{eid}_{x_series}_vs_{y_series}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        figures.append(str(fig_path))
    
    observation = "关系观察统计如下:\n" + "\n".join(observation_lines)
    observation += "\n中性观察: 散点关系可通过 Pearson r 判断线性相关方向，不构造公式。"
    
    return {
        "observation": observation,
        "derived_series": [],
        "figures": figures,
        "metrics": metrics
    }
