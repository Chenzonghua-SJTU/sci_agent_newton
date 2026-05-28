import json
import numpy as np
from typing import Any, Dict, List

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = payload["action"]
    if action != "define_derived_quantity":
        raise ValueError(f"Unsupported action: {action}")
    params = payload["parameters"]
    experiment_ids = params.get("experiment_ids", list(payload["experiments"].keys()))
    symbol = params["symbol"]
    expression = params["expression"]
    description = params.get("description", "")
    overwrite = params.get("overwrite", False)

    # 只支持 square(v_sg) 这种简单形式，可根据需要扩展
    if expression != "square(v_sg)":
        raise NotImplementedError(f"Expression '{expression}' not implemented; only 'square(v_sg)' is supported")

    experiments = payload["experiments"]
    derived_series: List[Dict[str, Any]] = []
    all_metrics: Dict[str, float] = {}
    observations_lines = []

    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
        exp = experiments[eid]
        series = exp.get("series", {})
        available = exp.get("available_series", [])
        # 检查是否有 v_sg 序列
        if "v_sg" not in series:
            raise ValueError(f"Experiment {eid}: required series 'v_sg' not found in series; available: {available}")

        v_sg = np.array(series["v_sg"], dtype=float)
        if len(v_sg) == 0:
            raise ValueError(f"Experiment {eid}: v_sg series is empty")

        t = np.array(series.get("t", []), dtype=float)
        if len(t) != len(v_sg):
            raise ValueError(f"Experiment {eid}: t series length {len(t)} does not match v_sg length {len(v_sg)}")

        v_sq = v_sg ** 2

        # 统计指标
        v_sq_min = float(np.min(v_sq))
        v_sq_max = float(np.max(v_sq))
        v_sq_mean = float(np.mean(v_sq))
        v_sq_std = float(np.std(v_sq, ddof=0))
        v_sq_start = float(v_sq[0])
        v_sq_end = float(v_sq[-1])
        if len(t) > 1:
            slope = (v_sq[-1] - v_sq[0]) / (t[-1] - t[0])
        else:
            slope = 0.0

        # 记录指标
        prefix = f"{eid}_v_sq_"
        all_metrics[prefix + "min"] = v_sq_min
        all_metrics[prefix + "max"] = v_sq_max
        all_metrics[prefix + "mean"] = v_sq_mean
        all_metrics[prefix + "std"] = v_sq_std
        all_metrics[prefix + "start"] = v_sq_start
        all_metrics[prefix + "end"] = v_sq_end
        all_metrics[prefix + "slope"] = slope

        observations_lines.append(
            f"{eid}: v_sq min={v_sq_min:.6f}, max={v_sq_max:.6f}, mean={v_sq_mean:.6f}, std={v_sq_std:.6f}, "
            f"start={v_sq_start:.6f}, end={v_sq_end:.6f}, slope={slope:.6f}"
        )

        # 准备派生序列
        derived_series.append({
            "experiment_id": eid,
            "name": symbol,
            "values": v_sq.tolist(),
            "source_name": "square(v_sg)",
            "provenance": "generated data processor: define_derived_quantity",
            "description": description
        })

    observation = (
        f"为实验 {experiment_ids} 定义派生序列 {symbol} = square(v_sg)。\n"
        + "\n".join(observations_lines)
    )

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [],
        "metrics": all_metrics
    }
