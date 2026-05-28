import numpy as np
import os
import json

def _safe_eval_expr(expr: str, variables: dict) -> np.ndarray:
    """
    Evaluate a mathematical expression using numpy functions.
    Supports square, cube, sqrt, log, exp, sin, cos, abs.
    Each variable in expr must be a key in variables dict, values are numpy arrays or scalars.
    """
    # Replace custom functions with numpy equivalents
    expr = expr.replace('square', 'np.square')
    expr = expr.replace('cube', 'np.power')  # will need to handle cube(x) -> np.power(x,3)
    expr = expr.replace('sqrt', 'np.sqrt')
    expr = expr.replace('log', 'np.log')
    expr = expr.replace('exp', 'np.exp')
    expr = expr.replace('sin', 'np.sin')
    expr = expr.replace('cos', 'np.cos')
    expr = expr.replace('abs', 'np.abs')
    # cube(x) becomes np.power(x,3) after replacement, need to adjust since cube(x) -> np.power(x,3)
    # Actually we replace 'cube' with 'np.power' but then the argument (x) will be left, so we need to handle
    # but this is approximate; for safety we keep a direct map using custom function
    # Alternatively, define cube = lambda x: x**3 in namespace
    # Use a safer approach: define a wrapper namespace
    namespace = {
        'np': np,
        'square': np.square,
        'cube': lambda x: np.power(x, 3),
        'sqrt': np.sqrt,
        'log': np.log,
        'exp': np.exp,
        'sin': np.sin,
        'cos': np.cos,
        'abs': np.abs,
    }
    # Add variables
    namespace.update(variables)
    return eval(expr, {"__builtins__": {}}, namespace)


def process(payload: dict) -> dict:
    # 1. Extract action and parameters
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    if action != "define_derived_quantity":
        raise ValueError(f"Action mismatch: expected 'define_derived_quantity', got '{action}'")

    experiment_ids = params.get("experiment_ids", [])
    symbol = params.get("symbol", "LHS")
    expression = params.get("expression", "")
    description = params.get("description", "")
    overwrite = params.get("overwrite", False)

    if not expression:
        raise ValueError("expression is required")

    derived_series = []
    figures = []
    metrics = {}
    observation_lines = []

    # Process each requested experiment
    for exp_id in experiment_ids:
        exp_data = experiments.get(exp_id)
        if exp_data is None:
            observation_lines.append(f"实验 {exp_id} 不存在，跳过")
            continue

        config = exp_data.get("config", {})
        series = exp_data.get("series", {})
        available = exp_data.get("available_series", [])

        # Check overwrite condition
        if not overwrite and symbol in available:
            observation_lines.append(f"实验 {exp_id} 已存在序列 {symbol}（overwrite=False），跳过计算")
            # If series exists, we can still compute metrics from existing values
            if symbol in series:
                vals = np.array(series[symbol])
                F_ext = config.get("F_ext", 0)
                dev_mean = np.mean(vals) - F_ext
                metrics[f"{exp_id}_{symbol}_mean"] = float(np.mean(vals))
                metrics[f"{exp_id}_{symbol}_std"] = float(np.std(vals))
                metrics[f"{exp_id}_{symbol}_min"] = float(np.min(vals))
                metrics[f"{exp_id}_{symbol}_max"] = float(np.max(vals))
                metrics[f"{exp_id}_{symbol}_dev_mean"] = float(dev_mean)
            continue

        # Get F_ext from config
        F_ext = config.get("F_ext", None)
        if F_ext is None:
            observation_lines.append(f"实验 {exp_id} 缺少 F_ext 参数，跳过")
            continue

        # Identify necessary series from expression by parsing variable names
        # We'll simply try to evaluate and catch NameError
        variables_needed = []
        # Check available series: we need a_sg11, v_sg11, t maybe
        # Better: use a regex or just check common names: a_sg11, v_sg11
        # We'll test with a default set
        # Hardcoded check
        for var_name in ['a_sg11', 'v_sg11']:
            if var_name not in series:
                observation_lines.append(f"实验 {exp_id} 缺少序列 {var_name}，跳过")
                continue
        # Also need t for length check
        if 't' not in series:
            observation_lines.append(f"实验 {exp_id} 缺少时间序列 t，跳过")
            continue

        # Extract series values as numpy arrays
        t = np.array(series['t'])
        a_sg11 = np.array(series['a_sg11'])
        v_sg11 = np.array(series['v_sg11'])

        # Check length consistency
        n = len(t)
        if len(a_sg11) != n or len(v_sg11) != n:
            observation_lines.append(f"实验 {exp_id} 序列长度不一致: t={n}, a_sg11={len(a_sg11)}, v_sg11={len(v_sg11)}，跳过")
            continue

        # Build evaluation context
        context = {
            'a_sg11': a_sg11,
            'v_sg11': v_sg11,
            'F_ext': F_ext,
        }

        try:
            lhs = _safe_eval_expr(expression, context)
        except Exception as e:
            observation_lines.append(f"实验 {exp_id} 表达式计算失败: {e}")
            continue

        # Convert to list
        lhs_list = lhs.tolist() if hasattr(lhs, 'tolist') else list(lhs)

        # Register derived series
        derived_series.append({
            "experiment_id": exp_id,
            "name": symbol,
            "values": lhs_list,
            "source_name": f"expression: {expression}",
            "provenance": "generated data processor: define_derived_quantity",
            "description": description
        })

        # Compute metrics
        mean_val = float(np.mean(lhs))
        std_val = float(np.std(lhs))
        min_val = float(np.min(lhs))
        max_val = float(np.max(lhs))
        dev_mean = mean_val - F_ext
        # Also compute absolute deviation mean
        abs_dev_mean = float(np.mean(np.abs(lhs - F_ext)))

        metrics[f"{exp_id}_{symbol}_mean"] = mean_val
        metrics[f"{exp_id}_{symbol}_std"] = std_val
        metrics[f"{exp_id}_{symbol}_min"] = min_val
        metrics[f"{exp_id}_{symbol}_max"] = max_val
        metrics[f"{exp_id}_{symbol}_dev_from_F_ext_mean"] = dev_mean
        metrics[f"{exp_id}_{symbol}_abs_dev_from_F_ext_mean"] = abs_dev_mean

        observation_lines.append(
            f"实验 {exp_id}: {symbol} mean={mean_val:.6f}, std={std_val:.6f}, min={min_val:.6f}, max={max_val:.6f}, "
            f"dev from F_ext={dev_mean:.6f}, abs_dev_mean={abs_dev_mean:.6f}"
        )

        # Plot LHS vs t
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, lhs, label=f'{symbol} ({exp_id})', linewidth=1.5)
        ax.axhline(y=F_ext, color='r', linestyle='--', alpha=0.7, label=f'F_ext = {F_ext}')
        ax.set_xlabel('t')
        ax.set_ylabel(symbol)
        ax.set_title(f'{symbol} vs t for {exp_id}\nExpression: {expression}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fname = f"{symbol}_vs_t_{exp_id}.png"
        fpath = os.path.join(output_dir, fname)
        plt.tight_layout()
        plt.savefig(fpath, dpi=100)
        plt.close()
        figures.append(fpath)

    # Build observation
    if observation_lines:
        observation = "define_derived_quantity 执行结果：\n" + "\n".join(observation_lines)
    else:
        observation = "未处理任何实验。"

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
