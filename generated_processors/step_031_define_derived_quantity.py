import os
import numpy as np
from typing import Any, Dict, List

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    if action not in ("define_derived_quantity", "test_candidate_expression"):
        raise ValueError(f"Unsupported action: {action}")
    params = payload["parameters"]
    experiment_ids = params.get("experiment_ids", [])
    symbol = params.get("symbol", "")
    expression_str = params.get("expression", "")
    overwrite = params.get("overwrite", False)
    output_dir = payload["output_dir"]
    experiments = payload["experiments"]

    if not experiment_ids:
        experiment_ids = list(experiments.keys())

    # allowed functions for expression evaluation
    func_map = {
        "square": lambda x: np.asarray(x, dtype=float)**2,
        "cube": lambda x: np.asarray(x, dtype=float)**3,
        "sqrt": lambda x: np.sqrt(np.asarray(x, dtype=float)),
        "log": lambda x: np.log(np.asarray(x, dtype=float)),
        "exp": lambda x: np.exp(np.asarray(x, dtype=float)),
        "sin": lambda x: np.sin(np.asarray(x, dtype=float)),
        "cos": lambda x: np.cos(np.asarray(x, dtype=float)),
        "abs": lambda x: np.abs(np.asarray(x, dtype=float)),
    }

    derived_series_list = []
    metrics = {}
    observation_lines = []

    for exp_id in experiment_ids:
        if exp_id not in experiments:
            raise ValueError(f"Experiment {exp_id} not found in payload.")
        exp_data = experiments[exp_id]
        series = exp_data.get("series", {})
        config = exp_data.get("config", {})

        # collect available series names from this experiment
        avail = exp_data.get("available_series", [])
        # also include those already in series dict
        all_names = set(list(series.keys()) + avail)

        # Check that required series for expression exist
        # Parse expression to find variable names (series names not in func_map)
        import re
        # simple tokenizer: keep letters and underscores
        tokens = re.findall(r'[A-Za-z_]\w*', expression_str)
        # remove known functions and math symbols
        known = set(func_map.keys()) | {"F_ext"}
        required_vars = [t for t in tokens if t not in known]

        for var in required_vars:
            if var not in all_names:
                raise ValueError(f"Series '{var}' is required but not available in experiment {exp_id}. Available: {sorted(all_names)}")

        # Build evaluation locals dictionary
        eval_locals = {}
        for var in required_vars:
            arr = np.array(series[var], dtype=float)
            eval_locals[var] = arr
        # add F_ext from config if present
        F_ext = config.get("F_ext", config.get("constant_force", None))
        if F_ext is not None:
            eval_locals["F_ext"] = float(F_ext)
        else:
            # if no F_ext defined, raise error if expression uses it
            if "F_ext" in expression_str:
                raise ValueError(f"Expression uses F_ext but experiment {exp_id} has no F_ext in config.")
            eval_locals["F_ext"] = 0.0

        # add function map
        eval_locals.update(func_map)

        # Evaluate expression using numpy built-in functions
        try:
            result = eval(expression_str, {"__builtins__": None}, eval_locals)
            if not isinstance(result, np.ndarray):
                result = np.array([result], dtype=float)
            # ensure same length as 't' series
            t_arr = np.array(series.get("t", []), dtype=float)
            if len(result) != len(t_arr):
                raise ValueError(f"Result length {len(result)} does not match t length {len(t_arr)}")
            check_values = result.tolist()
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression_str}' for experiment {exp_id}: {e}")

        # Determine overwrite: remove existing symbol if present and overwrite=True
        if symbol in all_names and overwrite:
            # series dict already contains it, we will return new derived series later
            pass

        # Statistics
        arr = np.array(check_values, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        start_val = check_values[0] if len(check_values) > 0 else float('nan')
        end_val = check_values[-1] if len(check_values) > 0 else float('nan')

        # Compare with F_ext if available
        F_ext_val = eval_locals.get("F_ext", None)
        if F_ext_val is not None:
            deviation = mean_val - F_ext_val
            abs_deviation = np.abs(deviation)
        else:
            deviation = None
            abs_deviation = None

        line = f"实验 {exp_id} (F_ext={F_ext_val}): check 均值={mean_val:.6f}, 标准差={std_val:.6f}, 最小值={min_val:.6f}, 最大值={max_val:.6f}"
        if deviation is not None:
            line += f", 与F_ext偏差={deviation:.6e} (绝对偏差={abs_deviation:.6e})"
        observation_lines.append(line)

        # Store metrics
        prefix = f"{exp_id}_check"
        metrics[f"{prefix}_mean"] = mean_val
        metrics[f"{prefix}_std"] = std_val
        metrics[f"{prefix}_min"] = min_val
        metrics[f"{prefix}_max"] = max_val
        metrics[f"{prefix}_start"] = start_val
        metrics[f"{prefix}_end"] = end_val
        if deviation is not None:
            metrics[f"{prefix}_deviation"] = deviation
            metrics[f"{prefix}_abs_deviation"] = abs_deviation

        # Prepare derived series
        source_name = f"根据表达式 {expression_str} 由 {required_vars} 计算"
        provenance = f"generated data processor: {action}"
        derived_series_list.append({
            "experiment_id": exp_id,
            "name": symbol,
            "values": check_values,
            "source_name": source_name,
            "provenance": provenance,
            "description": f"检验量 {symbol}"
        })

        # Optional: save check vs time plot
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig_path = os.path.join(output_dir, f"{exp_id}_{symbol}_vs_t.png")
        plt.figure(figsize=(8, 5))
        plt.plot(t_arr, arr, 'b-', label=f"{symbol} (mean={mean_val:.4f})")
        plt.axhline(y=F_ext_val if F_ext_val is not None else mean_val, color='r', linestyle='--', label=f"F_ext={F_ext_val}" if F_ext_val is not None else "mean")
        plt.xlabel("Time t")
        plt.ylabel(symbol)
        plt.title(f"Experiment {exp_id}: {symbol} = {expression_str}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

    # Build observation
    observation = "define_derived_quantity: 计算派生量 check = " + expression_str + "\n" + "\n".join(observation_lines)

    # figures list
    figures = [os.path.join(output_dir, f"{exp_id}_{symbol}_vs_t.png") for exp_id in experiment_ids]

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics
    }
