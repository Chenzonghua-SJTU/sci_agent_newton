import numpy as np
from scipy.signal import savgol_filter
import warnings

def process(payload: dict) -> dict:
    # --- 参数解析 ---
    params = payload["parameters"]
    exp_id = params["experiment_id"]
    source_series_name = params["source_series"]          # 原始位置序列名
    position_name = params["position_name"]                # 输出的位置（平滑后）序列名
    velocity_name = params["velocity_name"]                # 输出的速度序列名
    acceleration_name = params["acceleration_name"]        # 输出的加速度序列名
    window_length = int(params["window_length"])
    polyorder = int(params["polyorder"])
    overwrite = params.get("overwrite", False)

    # --- 获取实验数据 ---
    if exp_id not in payload["experiments"]:
        raise ValueError(f"实验 {exp_id} 不存在于 payload 中")
    exp = payload["experiments"][exp_id]

    if source_series_name not in exp["series"]:
        raise ValueError(f"源序列 {source_series_name} 在实验 {exp_id} 中不存在")
    q = np.array(exp["series"][source_series_name], dtype=float)

    if "t" not in exp["series"]:
        raise ValueError(f"时间序列 t 在实验 {exp_id} 中不存在")
    t = np.array(exp["series"]["t"], dtype=float)

    n = len(q)
    if n < window_length:
        raise ValueError(f"序列长度 {n} 小于窗口长度 {window_length}")

    # --- 时间步长（假设均匀） ---
    dt = t[1] - t[0]
    if dt <= 0:
        raise ValueError("时间步长必须为正")

    # --- 使用 Savitzky-Golay 滤波计算导数 ---
    q_smooth = savgol_filter(q, window_length, polyorder, deriv=0)
    v = savgol_filter(q, window_length, polyorder, deriv=1) / dt
    a = savgol_filter(q, window_length, polyorder, deriv=2) / (dt ** 2)

    # --- 构建结果数据 ---
    # 源信息文本
    source_desc = (f"Savitzky-Golay filter (window={window_length}, poly={polyorder}) "
                   f"on {exp_id}:{source_series_name}")

    derived_series = []

    # 位置序列（平滑后）
    derived_series.append({
        "experiment_id": exp_id,
        "name": position_name,
        "values": q_smooth.tolist(),
        "source_name": source_desc,
        "provenance": "generated data processor: estimate_kinematics",
        "description": f"Smoothed position via SG filter (win={window_length}, poly={polyorder})"
    })

    # 速度序列
    derived_series.append({
        "experiment_id": exp_id,
        "name": velocity_name,
        "values": v.tolist(),
        "source_name": source_desc,
        "provenance": "generated data processor: estimate_kinematics",
        "description": f"Velocity estimated as 1st derivative of {source_series_name} (SG filter)"
    })

    # 加速度序列
    derived_series.append({
        "experiment_id": exp_id,
        "name": acceleration_name,
        "values": a.tolist(),
        "source_name": source_desc,
        "provenance": "generated data processor: estimate_kinematics",
        "description": f"Acceleration estimated as 2nd derivative of {source_series_name} (SG filter)"
    })

    # --- 统计信息用于 observation & metrics ---
    def stats(arr, name):
        return {
            f"{name}_min": float(np.min(arr)),
            f"{name}_max": float(np.max(arr)),
            f"{name}_mean": float(np.mean(arr)),
            f"{name}_std": float(np.std(arr)),
            f"{name}_start": float(arr[0]),
            f"{name}_end": float(arr[-1]),
            f"{name}_slope": float((arr[-1] - arr[0]) / (t[-1] - t[0])) if len(arr) >= 2 else float('nan')
        }

    metrics = {}
    metrics.update(stats(q_smooth, position_name))
    metrics.update(stats(v, velocity_name))
    metrics.update(stats(a, acceleration_name))

    observation = (
        f"实验 {exp_id}: 从序列 {source_series_name} 使用 Savitzky-Golay 滤波 "
        f"(窗口={window_length}, 多项式阶数={polyorder}) 估计得到位置（平滑后）、速度、加速度序列。\n"
        f"位置序列 {position_name}: 均值={metrics[f'{position_name}_mean']:.6f}, "
        f"范围=[{metrics[f'{position_name}_min']:.6f}, {metrics[f'{position_name}_max']:.6f}], "
        f"起点={metrics[f'{position_name}_start']:.6f}, 终点={metrics[f'{position_name}_end']:.6f}\n"
        f"速度序列 {velocity_name}: 均值={metrics[f'{velocity_name}_mean']:.6f}, "
        f"范围=[{metrics[f'{velocity_name}_min']:.6f}, {metrics[f'{velocity_name}_max']:.6f}], "
        f"起点={metrics[f'{velocity_name}_start']:.6f}, 终点={metrics[f'{velocity_name}_end']:.6f}\n"
        f"加速度序列 {acceleration_name}: 均值={metrics[f'{acceleration_name}_mean']:.6f}, "
        f"范围=[{metrics[f'{acceleration_name}_min']:.6f}, {metrics[f'{acceleration_name}_max']:.6f}], "
        f"起点={metrics[f'{acceleration_name}_start']:.6f}, 终点={metrics[f'{acceleration_name}_end']:.6f}"
    )

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [],
        "metrics": metrics
    }
