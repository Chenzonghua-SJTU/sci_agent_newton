import numpy as np
from scipy.signal import savgol_filter
import os
import json

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    parameters = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 处理参数
    experiment_id = parameters.get("experiment_id", "exp_07")
    source_series = parameters.get("source_series", "q")
    position_name = parameters.get("position_name", "q_sg21")
    velocity_name = parameters.get("velocity_name", "v_sg21")
    acceleration_name = parameters.get("acceleration_name", "a_sg21")
    window_length = int(parameters.get("window_length", 21))
    polyorder = int(parameters.get("polyorder", 2))
    overwrite = parameters.get("overwrite", True)

    if experiment_id not in experiments:
        raise ValueError(f"Experiment {experiment_id} not found in payload")
    exp = experiments[experiment_id]
    series = exp.get("series", {})
    available = exp.get("available_series", [])
    config = exp.get("config", {})

    # 检查源序列是否存在
    if source_series not in series:
        raise ValueError(f"Source series '{source_series}' not found in experiment {experiment_id}")
    q = np.array(series[source_series], dtype=np.float64)
    t = np.array(series.get("t", None), dtype=np.float64)
    if t is None:
        raise ValueError(f"Time series 't' not found in experiment {experiment_id}")
    n = len(q)
    if len(t) != n:
        raise ValueError(f"Source series and time series have different lengths ({len(q)} vs {len(t)})")

    # 检查窗口长度有效性
    if window_length > n:
        raise ValueError(f"window_length ({window_length}) exceeds series length ({n})")
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")
    if polyorder >= window_length:
        raise ValueError(f"polyorder ({polyorder}) must be less than window_length ({window_length})")

    # 计算时间步长
    if n > 1:
        dt = (t[-1] - t[0]) / (n - 1)
    else:
        dt = 1.0
    if dt <= 0:
        raise ValueError(f"Invalid time step dt={dt}")

    # 应用 Savitzky-Golay 滤波器
    # 平滑位置（deriv=0）
    q_smooth = savgol_filter(q, window_length, polyorder, deriv=0, mode='mirror')
    # 速度（deriv=1）注意需要除以 dt
    v = savgol_filter(q, window_length, polyorder, deriv=1, mode='mirror') / dt
    # 加速度（deriv=2）需要除以 dt^2
    a = savgol_filter(q, window_length, polyorder, deriv=2, mode='mirror') / (dt ** 2)

    # 构建派生序列
    derived_series = []
    # 位置序列
    ps = {
        "experiment_id": experiment_id,
        "name": position_name,
        "values": q_smooth.tolist(),
        "source_name": f"Savitzky-Golay filtered {source_series} (window={window_length}, polyorder={polyorder})",
        "provenance": "generated data processor: estimate_kinematics",
        "description": f"Smoothed position using Savitzky-Golay filter (window={window_length}, polyorder={polyorder})"
    }
    derived_series.append(ps)

    # 速度序列
    vs = {
        "experiment_id": experiment_id,
        "name": velocity_name,
        "values": v.tolist(),
        "source_name": f"Savitzky-Golay derivative of {source_series} (window={window_length}, polyorder={polyorder})",
        "provenance": "generated data processor: estimate_kinematics",
        "description": f"Velocity estimated via Savitzky-Golay filter (window={window_length}, polyorder={polyorder})"
    }
    derived_series.append(vs)

    # 加速度序列
    acs = {
        "experiment_id": experiment_id,
        "name": acceleration_name,
        "values": a.tolist(),
        "source_name": f"Savitzky-Golay second derivative of {source_series} (window={window_length}, polyorder={polyorder})",
        "provenance": "generated data processor: estimate_kinematics",
        "description": f"Acceleration estimated via Savitzky-Golay filter (window={window_length}, polyorder={polyorder})"
    }
    derived_series.append(acs)

    # 统计信息
    observation = (
        f"对实验 {experiment_id} 使用 Savitzky-Golay 滤波器（窗口{window_length}，阶数{polyorder}）从 {source_series} 估计运动学参数。"
        f"生成了序列 {position_name}（平滑位置）、{velocity_name}（速度）、{acceleration_name}（加速度）。"
        f"时间步长 dt={dt:.6f}。"
        f"序列长度 {n}。"
    )
    # 添加统计到 observation
    stats = {}
    for arr, name in [(q_smooth, position_name), (v, velocity_name), (a, acceleration_name)]:
        stats[name] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr))
        }
    stat_str = "; ".join([f"{k}: min={v['min']:.6f}, max={v['max']:.6f}, mean={v['mean']:.6f}, std={v['std']:.6f}" for k,v in stats.items()])
    observation += f" 统计摘要: {stat_str}"

    # 可选 metrics
    metrics = {
        f"{experiment_id}_{position_name}_min": stats[position_name]["min"],
        f"{experiment_id}_{position_name}_max": stats[position_name]["max"],
        f"{experiment_id}_{velocity_name}_min": stats[velocity_name]["min"],
        f"{experiment_id}_{velocity_name}_max": stats[velocity_name]["max"],
        f"{experiment_id}_{acceleration_name}_min": stats[acceleration_name]["min"],
        f"{experiment_id}_{acceleration_name}_max": stats[acceleration_name]["max"],
    }

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [],
        "metrics": metrics
    }
