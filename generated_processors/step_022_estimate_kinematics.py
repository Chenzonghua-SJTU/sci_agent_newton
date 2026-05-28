import numpy as np
from scipy.signal import savgol_filter

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # 提取参数
    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        # 如果只有单个实验
        if "experiment_id" in params:
            experiment_ids = [params["experiment_id"]]
        else:
            experiment_ids = list(experiments.keys())

    source_series = params.get("source_series", "q")
    position_name = params.get("position_name", "q")
    velocity_name = params.get("velocity_name", "v")
    acceleration_name = params.get("acceleration_name", "a")
    window_length = params.get("window_length", 21)
    polyorder = params.get("polyorder", 3)
    overwrite = params.get("overwrite", True)

    # 检查窗口参数合法性
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")
    if polyorder >= window_length:
        raise ValueError(
            f"polyorder ({polyorder}) must be less than window_length ({window_length})"
        )

    derived_series = []
    metrics = {}
    observation_lines = ["运动学估计完成。"]
    for exp_id in experiment_ids:
        if exp_id not in experiments:
            raise ValueError(f"Experiment {exp_id} not found in payload")
        exp = experiments[exp_id]
        config = exp.get("config", {})
        series = exp.get("series", {})
        if source_series not in series:
            raise ValueError(
                f"Experiment {exp_id}: source_series '{source_series}' not found in available series"
            )
        q = np.array(series[source_series], dtype=float)
        t = np.array(series.get("t", []), dtype=float)
        if len(t) == 0:
            raise ValueError(f"Experiment {exp_id}: 't' series not found")
        n = len(t)
        if len(q) != n:
            raise ValueError(f"Experiment {exp_id}: source_series length {len(q)} != t length {n}")
        if window_length > n:
            raise ValueError(
                f"Experiment {exp_id}: window_length ({window_length}) exceeds data length ({n})"
            )

        # 采样间隔
        dt = t[1] - t[0] if n > 1 else 1.0

        # Savitzky-Golay 滤波
        q_smooth = savgol_filter(q, window_length, polyorder, deriv=0, mode="interp")
        v = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt, mode="interp")
        a = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt, mode="interp")

        # 构建派生序列（覆盖原序列）
        pos_series = {
            "experiment_id": exp_id,
            "name": position_name,
            "values": q_smooth.tolist(),
            "source_name": f"SG_filter_deriv0({source_series}, window={window_length}, polyorder={polyorder})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"平滑位置（SG滤波，窗口{window_length}，阶数{polyorder}）"
        }
        vel_series = {
            "experiment_id": exp_id,
            "name": velocity_name,
            "values": v.tolist(),
            "source_name": f"SG_filter_deriv1({source_series}, window={window_length}, polyorder={polyorder})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"速度（SG滤波导数，窗口{window_length}，阶数{polyorder}）"
        }
        acc_series = {
            "experiment_id": exp_id,
            "name": acceleration_name,
            "values": a.tolist(),
            "source_name": f"SG_filter_deriv2({source_series}, window={window_length}, polyorder={polyorder})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"加速度（SG滤波二阶导数，窗口{window_length}，阶数{polyorder}）"
        }
        derived_series.extend([pos_series, vel_series, acc_series])

        # 统计量
        q_min, q_max, q_mean, q_std = float(np.min(q_smooth)), float(np.max(q_smooth)), float(np.mean(q_smooth)), float(np.std(q_smooth))
        v_min, v_max, v_mean, v_std = float(np.min(v)), float(np.max(v)), float(np.mean(v)), float(np.std(v))
        a_min, a_max, a_mean, a_std = float(np.min(a)), float(np.max(a)), float(np.mean(a)), float(np.std(a))

        metrics[f"{exp_id}_{position_name}_min"] = q_min
        metrics[f"{exp_id}_{position_name}_max"] = q_max
        metrics[f"{exp_id}_{position_name}_mean"] = q_mean
        metrics[f"{exp_id}_{position_name}_std"] = q_std
        metrics[f"{exp_id}_{velocity_name}_min"] = v_min
        metrics[f"{exp_id}_{velocity_name}_max"] = v_max
        metrics[f"{exp_id}_{velocity_name}_mean"] = v_mean
        metrics[f"{exp_id}_{velocity_name}_std"] = v_std
        metrics[f"{exp_id}_{acceleration_name}_min"] = a_min
        metrics[f"{exp_id}_{acceleration_name}_max"] = a_max
        metrics[f"{exp_id}_{acceleration_name}_mean"] = a_mean
        metrics[f"{exp_id}_{acceleration_name}_std"] = a_std

        obs = (
            f"实验 {exp_id}: 从 {source_series} 通过 SG 滤波 (窗口 {window_length}, 阶数 {polyorder}) 估计 "
            f"平滑位置 {position_name}, 速度 {velocity_name}, 加速度 {acceleration_name}。"
            f"{position_name}: 最小值 {q_min:.4f}, 最大值 {q_max:.4f}; "
            f"{velocity_name}: 最小值 {v_min:.4f}, 最大值 {v_max:.4f}; "
            f"{acceleration_name}: 最小值 {a_min:.4f}, 最大值 {a_max:.4f}。"
        )
        observation_lines.append(obs)

    # 如果 overwrite==False 但参数要求覆盖，实际处理中我们忽略已有的派生序列，按 overwrite 逻辑返回
    # 观察信息
    observation = "\n".join(observation_lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [],
        "metrics": metrics
    }
