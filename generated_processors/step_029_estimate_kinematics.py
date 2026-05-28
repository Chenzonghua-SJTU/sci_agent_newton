import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, List, Any

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    experiment_ids = params.get("experiment_ids", params.get("experiment_id"))
    if experiment_ids is None:
        experiment_ids = list(experiments.keys())
    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]

    source_series = params.get("source_series", "q")
    position_name = params.get("position_name", "q_smooth")
    velocity_name = params.get("velocity_name", "v_smooth")
    acceleration_name = params.get("acceleration_name", "a_smooth")
    window_length = int(params.get("window_length", 11))
    polyorder = int(params.get("polyorder", 3))
    overwrite = params.get("overwrite", True)

    # 检查 window_length 必须为奇数
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")
    if window_length < polyorder + 1:
        raise ValueError(f"window_length must be > polyorder, got window_length={window_length}, polyorder={polyorder}")

    derived_series = []
    metrics = {}

    for exp_id in experiment_ids:
        if exp_id not in experiments:
            raise ValueError(f"Experiment {exp_id} not found in payload")
        exp = experiments[exp_id]
        config = exp.get("config", {})
        series_dict = exp.get("series", {})
        available = exp.get("available_series", [])

        # 获取原始 q 和 t
        if source_series not in series_dict:
            raise ValueError(f"Source series '{source_series}' not available in experiment {exp_id}")
        q = np.array(series_dict[source_series], dtype=float)
        t = np.array(series_dict.get("t", []), dtype=float)
        if len(t) == 0:
            raise ValueError(f"t series not available in experiment {exp_id}")

        # 计算 dt
        if len(t) > 1:
            dt = np.mean(np.diff(t))
        else:
            dt = config.get("dt", 0.1)  # fallback
        if dt <= 0:
            raise ValueError(f"Non-positive dt={dt} for experiment {exp_id}")

        # 检查窗口长度
        n = len(q)
        if window_length > n:
            raise ValueError(f"window_length={window_length} > number of points={n} in experiment {exp_id}")

        # 计算平滑位置、速度、加速度
        q_smooth = savgol_filter(q, window_length, polyorder, deriv=0, mode='interp')
        v_smooth = savgol_filter(q, window_length, polyorder, deriv=1, mode='interp') / dt
        a_smooth = savgol_filter(q, window_length, polyorder, deriv=2, mode='interp') / (dt ** 2)

        # 构造派生序列
        def add_series(name, values):
            derived_series.append({
                "experiment_id": exp_id,
                "name": name,
                "values": values.tolist(),
                "source_name": f"Savitzky-Golay {source_series} → {name} (window={window_length}, poly={polyorder})",
                "provenance": f"estimate_kinematics: {action}",
                "description": f"Estimated {name} from {source_series}"
            })

        add_series(position_name, q_smooth)
        add_series(velocity_name, v_smooth)
        add_series(acceleration_name, a_smooth)

        # 记录 metrics
        for label, arr in [("q_smooth", q_smooth), ("v_smooth", v_smooth), ("a_smooth", a_smooth)]:
            prefix = f"{exp_id}_{label}"
            metrics[f"{prefix}_min"] = float(np.min(arr))
            metrics[f"{prefix}_max"] = float(np.max(arr))
            metrics[f"{prefix}_mean"] = float(np.mean(arr))
            metrics[f"{prefix}_std"] = float(np.std(arr))

    # 构造 observation
    obs_lines = ["运动学估计完成。"]
    for exp_id in experiment_ids:
        pos = metrics.get(f"{exp_id}_q_smooth_min", None)
        if pos is not None:
            obs_lines.append(
                f"实验 {exp_id}: "
                f"{position_name}: 最小值 {metrics[f'{exp_id}_q_smooth_min']:.4f}, "
                f"最大值 {metrics[f'{exp_id}_q_smooth_max']:.4f}; "
                f"{velocity_name}: 最小值 {metrics[f'{exp_id}_v_smooth_min']:.4f}, "
                f"最大值 {metrics[f'{exp_id}_v_smooth_max']:.4f}; "
                f"{acceleration_name}: 最小值 {metrics[f'{exp_id}_a_smooth_min']:.4f}, "
                f"最大值 {metrics[f'{exp_id}_a_smooth_max']:.4f}。"
            )
    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [],
        "metrics": metrics
    }
