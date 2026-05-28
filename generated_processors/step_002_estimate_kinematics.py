import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, List, Any, Optional

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", "")

    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        # 如果没有显式指定，处理所有实验
        experiment_ids = list(experiments.keys())

    source_series = params.get("source_series", "q")
    position_name = params.get("position_name", "q")
    velocity_name = params.get("velocity_name", "v")
    acceleration_name = params.get("acceleration_name", "a")
    window_length = params.get("window_length", 5)
    polyorder = params.get("polyorder", 2)
    overwrite = params.get("overwrite", False)

    observation_parts = []
    all_derived_series = []
    all_figures = []
    all_metrics = {}

    for exp_id in experiment_ids:
        if exp_id not in experiments:
            raise ValueError(f"Experiment {exp_id} not found in payload.")
        exp_data = experiments[exp_id]
        config = exp_data.get("config", {})
        series = exp_data.get("series", {})
        available = exp_data.get("available_series", [])

        # 获取源数据
        if source_series not in series:
            raise ValueError(f"Source series '{source_series}' not available in experiment {exp_id}.")
        q = np.array(series[source_series], dtype=float)
        if len(q) == 0:
            raise ValueError(f"Source series '{source_series}' is empty in experiment {exp_id}.")

        # 获取时间 t
        t = None
        if "t" in series:
            t = np.array(series["t"], dtype=float)
        elif "t" in available:
            raise ValueError(f"Time series 't' not found in series, but listed in available_series.")
        else:
            # 尝试从 config 获取 dt 并构造 t
            dt = config.get("dt", None)
            if dt is not None:
                t = np.arange(len(q)) * dt
            else:
                raise ValueError(f"Cannot determine time vector for experiment {exp_id}. Please ensure 't' series is present or provide dt in config.")

        # 确保 q 和 t 长度一致
        if len(t) != len(q):
            raise ValueError(f"Length mismatch: t({len(t)}) vs {source_series}({len(q)}) in experiment {exp_id}.")

        # 均匀采样检查（非严格）
        dt_avg = np.mean(np.diff(t))
        if not np.allclose(np.diff(t), dt_avg, atol=1e-12):
            # 允许微小误差，但严重不均匀应警告
            max_dev = np.max(np.abs(np.diff(t) - dt_avg))
            if max_dev > 1e-9:
                raise ValueError(f"Time vector in experiment {exp_id} is not uniformly spaced (max deviation {max_dev:.2e}). Savgol requires uniform spacing.")

        # 计算平滑位置、速度、加速度（使用 savgol_filter）
        try:
            q_smooth = savgol_filter(q, window_length, polyorder, deriv=0)
            v = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt_avg)
            a = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt_avg)
        except Exception as e:
            raise ValueError(f"Savgol filter failed for experiment {exp_id}: {e}")

        # 准备输出序列列表（检查覆盖条件）
        derived_items = []

        # 平滑位置
        if position_name and position_name not in available:
            derived_items.append({
                "experiment_id": exp_id,
                "name": position_name,
                "values": q_smooth.tolist(),
                "source_name": f"savgol_filter(window={window_length}, polyorder={polyorder}) on {source_series}",
                "provenance": f"estimate_kinematics: window={window_length}, polyorder={polyorder}",
                "description": f"Smoothed position from {source_series} using SG filter"
            })
        elif position_name and position_name in available and not overwrite:
            # 已存在且不覆盖，则跳过
            pass
        elif position_name and position_name in available and overwrite:
            # 强制覆盖（无论如何我们都返回同名序列，由外部决定替换）
            derived_items.append({
                "experiment_id": exp_id,
                "name": position_name,
                "values": q_smooth.tolist(),
                "source_name": f"savgol_filter(window={window_length}, polyorder={polyorder}) on {source_series} (overwrite)",
                "provenance": f"estimate_kinematics: window={window_length}, polyorder={polyorder}, overwrite=True",
                "description": f"Smoothed position from {source_series} using SG filter (overwrites previous)"
            })

        # 速度
        if velocity_name and velocity_name not in available:
            derived_items.append({
                "experiment_id": exp_id,
                "name": velocity_name,
                "values": v.tolist(),
                "source_name": f"savgol_filter(deriv=1, window={window_length}, polyorder={polyorder}) on {source_series}",
                "provenance": f"estimate_kinematics: window={window_length}, polyorder={polyorder}",
                "description": f"Velocity estimated from {source_series} using SG filter"
            })
        elif velocity_name and velocity_name in available and not overwrite:
            pass
        elif velocity_name and velocity_name in available and overwrite:
            derived_items.append({
                "experiment_id": exp_id,
                "name": velocity_name,
                "values": v.tolist(),
                "source_name": f"savgol_filter(deriv=1) on {source_series} (overwrite)",
                "provenance": f"estimate_kinematics: window={window_length}, polyorder={polyorder}, overwrite=True",
                "description": f"Velocity estimated from {source_series} (overwrites previous)"
            })

        # 加速度
        if acceleration_name and acceleration_name not in available:
            derived_items.append({
                "experiment_id": exp_id,
                "name": acceleration_name,
                "values": a.tolist(),
                "source_name": f"savgol_filter(deriv=2, window={window_length}, polyorder={polyorder}) on {source_series}",
                "provenance": f"estimate_kinematics: window={window_length}, polyorder={polyorder}",
                "description": f"Acceleration estimated from {source_series} using SG filter"
            })
        elif acceleration_name and acceleration_name in available and not overwrite:
            pass
        elif acceleration_name and acceleration_name in available and overwrite:
            derived_items.append({
                "experiment_id": exp_id,
                "name": acceleration_name,
                "values": a.tolist(),
                "source_name": f"savgol_filter(deriv=2) on {source_series} (overwrite)",
                "provenance": f"estimate_kinematics: window={window_length}, polyorder={polyorder}, overwrite=True",
                "description": f"Acceleration estimated from {source_series} (overwrites previous)"
            })

        # 构建实验观察
        exp_obs_parts = [f"实验 {exp_id}:"]
        # 生成简短统计
        q_range = (q.min(), q.max())
        v_range = (v.min(), v.max())
        a_range = (a.min(), a.max())
        exp_obs_parts.append(f"原始 {source_series} 范围 [{q_range[0]:.6f}, {q_range[1]:.6f}]")
        # 记录生成的序列
        for item in derived_items:
            if item["name"] == velocity_name:
                exp_obs_parts.append(f"速度 {velocity_name}: min={v_range[0]:.6f}, max={v_range[1]:.6f}")
            elif item["name"] == acceleration_name:
                exp_obs_parts.append(f"加速度 {acceleration_name}: min={a_range[0]:.6f}, max={a_range[1]:.6f}")
            elif item["name"] == position_name:
                qs = np.array(item["values"])
                exp_obs_parts.append(f"平滑位置 {position_name}: min={qs.min():.6f}, max={qs.max():.6f}")
        # 如果没有生成任何序列，说明已存在且不覆盖
        if not derived_items:
            exp_obs_parts.append("所有目标序列已存在且 overwrite=False，未生成新序列")

        observation_parts.append(" ".join(exp_obs_parts))
        all_derived_series.extend(derived_items)

        # 可选的 metrics（不必须）
        # 对每个实验记录速度和加速度的范围
        all_metrics[f"{exp_id}_v_min"] = float(v.min())
        all_metrics[f"{exp_id}_v_max"] = float(v.max())
        all_metrics[f"{exp_id}_a_min"] = float(a.min())
        all_metrics[f"{exp_id}_a_max"] = float(a.max())

    observation = " | ".join(observation_parts)

    return {
        "observation": observation,
        "derived_series": all_derived_series,
        "figures": all_figures,
        "metrics": all_metrics
    }
