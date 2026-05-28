import numpy as np
from scipy import stats
from typing import Dict, List, Any

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", "")

    experiment_ids = params.get("experiment_ids", list(experiments.keys()))
    symbol = params.get("symbol")
    expression = params.get("expression")
    overwrite = params.get("overwrite", False)

    derived_series = []
    metrics = {}
    observations = []

    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
        exp = experiments[eid]
        config = exp.get("config", {})
        series = exp.get("series", {})
        t = series.get("t", None)
        if t is None:
            raise ValueError(f"Experiment {eid}: missing 't' series")

        # Parse expression: square(v_sg) -> function=square, arg_series=v_sg
        # Simple parsing: assume expression is like "function(arg)"
        func_name, arg = expression.split("(")
        arg = arg.rstrip(")")
        func_name = func_name.strip()
        arg = arg.strip()

        # Get the source series or config constant
        if arg == "F_ext":
            source = config.get("F_ext", 0.0)
            # scalar, need to replicate to match t length
            source_values = [float(source)] * len(t)
        else:
            if arg not in series:
                raise ValueError(f"Experiment {eid}: missing series '{arg}' for expression")
            source_values = series[arg]
            if len(source_values) != len(t):
                raise ValueError(f"Experiment {eid}: series '{arg}' length mismatch with t")

        # Apply function
        func_map = {
            "square": lambda x: x * x,
            "cube": lambda x: x * x * x,
            "sqrt": lambda x: np.sqrt(x),
            "log": lambda x: np.log(x),
            "exp": lambda x: np.exp(x),
            "sin": lambda x: np.sin(x),
            "cos": lambda x: np.cos(x),
            "abs": lambda x: abs(x)
        }
        if func_name not in func_map:
            raise ValueError(f"Unsupported function '{func_name}' in expression")

        func = func_map[func_name]
        new_values = [float(func(v)) for v in source_values]

        # Compute statistics
        arr = np.array(new_values)
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr))
        start_v = new_values[0]
        end_v = new_values[-1]
        # slope = (end - start) / (t[-1]-t[0])
        dt = t[-1] - t[0]
        slope_v = (end_v - start_v) / dt if dt != 0 else 0.0

        # Add derived series
        derived_series.append({
            "experiment_id": eid,
            "name": symbol,
            "values": new_values,
            "source_name": expression,
            "provenance": "generated data processor: define_derived_quantity",
            "description": f"派生序列 {symbol} = {expression}"
        })

        # Store metrics
        prefix = f"{eid}_{symbol}"
        metrics[f"{prefix}_min"] = min_v
        metrics[f"{prefix}_max"] = max_v
        metrics[f"{prefix}_mean"] = mean_v
        metrics[f"{prefix}_std"] = std_v
        metrics[f"{prefix}_start"] = start_v
        metrics[f"{prefix}_end"] = end_v
        metrics[f"{prefix}_slope"] = slope_v

        observations.append(
            f"{eid}: {symbol} min={min_v:.6f}, max={max_v:.6f}, "
            f"mean={mean_v:.6f}, std={std_v:.6f}, "
            f"start={start_v:.6f}, end={end_v:.6f}, slope={slope_v:.6f}"
        )

    obs_text = (
        f"为实验 {experiment_ids} 定义派生序列 {symbol} = {expression}。\n"
        + "\n".join(observations)
    )

    return {
        "observation": obs_text,
        "derived_series": derived_series,
        "figures": [],
        "metrics": metrics
    }
