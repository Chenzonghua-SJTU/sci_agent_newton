import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional, Any

def _get_dt(experiment: dict) -> float:
    """从实验配置中提取 dt 步长。"""
    config = experiment.get("config", {})
    # 优先从 config 中的 dt 字段获取
    dt = config.get("dt", None)
    if dt is not None:
        return dt
    # 否则从 t 序列计算平均间隔
    t_vals = experiment["series"].get("t", [])
    if len(t_vals) < 2:
        raise ValueError("无法确定时间步长，t 序列太短")
    return np.mean(np.diff(t_vals))

def _check_series_exists(experiment: dict, series_name: str) -> bool:
    """检查实验是否已存在指定名称的序列。"""
    return series_name in experiment.get("available_series", []) or series_name in experiment.get("series", {})

def process(payload: dict) -> dict:
    parameters = payload["parameters"]
    experiment_id = parameters.get("experiment_id")
    source_series_name = parameters.get("source_series", "q")
    position_name = parameters.get("position_name", "q_smooth")
    velocity_name = parameters.get("velocity_name", "v_sg")
    acceleration_name = parameters.get("acceleration_name", "a_sg")
    window_length = int(parameters.get("window_length", 11))
    polyorder = int(parameters.get("polyorder", 3))
    overwrite = bool(parameters.get("overwrite", False))

    experiments = payload.get("experiments", {})
    if experiment_id not in experiments:
        raise ValueError(f"找不到实验 {experiment_id}")

    experiment = experiments[experiment_id]
    series = experiment.get("series", {})
    t = np.array(series.get("t", []))
    source = np.array(series.get(source_series_name, []))

    if len(t) == 0 or len(source) == 0:
        raise ValueError(f"实验 {experiment_id} 缺少时间序列或源序列 '{source_series_name}'")
    if len(t) != len(source):
        raise ValueError("时间序列和源序列长度不匹配")

    # 检查是否已存在且不覆盖
    if not overwrite:
        existing = []
        for name in [position_name, velocity_name, acceleration_name]:
            if _check_series_exists(experiment, name):
                existing.append(name)
        if existing:
            raise ValueError(f"目标序列 {existing} 已存在，且 overwrite=False，无法执行。如需覆盖请设置 overwrite=True")

    # 获取时间步长
    dt = _get_dt(experiment)

    # 应用 Savitzky-Golay 滤波器
    # 确保窗口长度不超过数据长度且为奇数
    n = len(t)
    if window_length % 2 == 0:
        window_length += 1  # 强制奇数
    if window_length > n:
        window_length = n if n % 2 == 1 else n - 1
        if window_length < 3:
            raise ValueError("数据点太少，无法应用 Savitzky-Golay 滤波")

    try:
        q_smooth = signal.savgol_filter(source, window_length, polyorder, deriv=0)
        v_sg = signal.savgol_filter(source, window_length, polyorder, deriv=1, delta=dt)
        a_sg = signal.savgol_filter(source, window_length, polyorder, deriv=2, delta=dt)
    except Exception as e:
        raise ValueError(f"Savitzky-Golay 滤波失败: {e}")

    # 转换为列表
    q_smooth_list = q_smooth.tolist()
    v_sg_list = v_sg.tolist()
    a_sg_list = a_sg.tolist()

    # 构建 derived_series
    derived_series = [
        {
            "experiment_id": experiment_id,
            "name": position_name,
            "values": q_smooth_list,
            "source_name": f"Savitzky-Golay filter (win={window_length}, order={polyorder}) from {source_series_name}",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"平滑位置，基于 {source_series_name} 的 Savitzky-Golay 滤波"
        },
        {
            "experiment_id": experiment_id,
            "name": velocity_name,
            "values": v_sg_list,
            "source_name": f"Savitzky-Golay filter (win={window_length}, order={polyorder}, deriv=1, dt={dt}) from {source_series_name}",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"速度，基于 {source_series_name} 的一阶导数 Savitzky-Golay 滤波"
        },
        {
            "experiment_id": experiment_id,
            "name": acceleration_name,
            "values": a_sg_list,
            "source_name": f"Savitzky-Golay filter (win={window_length}, order={polyorder}, deriv=2, dt={dt}) from {source_series_name}",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"加速度，基于 {source_series_name} 的二阶导数 Savitzky-Golay 滤波"
        }
    ]

    # 计算统计指标
    metrics = {}
    for name, values in [(position_name, q_smooth_list), (velocity_name, v_sg_list), (acceleration_name, a_sg_list)]:
        arr = np.array(values)
        metrics[f"{experiment_id}_{name}_min"] = float(np.min(arr))
        metrics[f"{experiment_id}_{name}_max"] = float(np.max(arr))
        metrics[f"{experiment_id}_{name}_mean"] = float(np.mean(arr))

    # 观察信息
    observation = (
        f"对实验 {experiment_id} 使用 Savitzky-Golay 滤波（窗口长度={window_length}, "
        f"多项式阶数={polyorder}, dt={dt}) 从 {source_series_name} 估计出平滑位置 {position_name}、"
        f"速度 {velocity_name}、加速度 {acceleration_name}。\n"
        f"  {position_name}: min={metrics[f'{experiment_id}_{position_name}_min']:.6f}, "
        f"max={metrics[f'{experiment_id}_{position_name}_max']:.6f}, "
        f"mean={metrics[f'{experiment_id}_{position_name}_mean']:.6f}\n"
        f"  {velocity_name}: min={metrics[f'{experiment_id}_{velocity_name}_min']:.6f}, "
        f"max={metrics[f'{experiment_id}_{velocity_name}_max']:.6f}, "
        f"mean={metrics[f'{experiment_id}_{velocity_name}_mean']:.6f}\n"
        f"  {acceleration_name}: min={metrics[f'{experiment_id}_{acceleration_name}_min']:.6f}, "
        f"max={metrics[f'{experiment_id}_{acceleration_name}_max']:.6f}, "
        f"mean={metrics[f'{experiment_id}_{acceleration_name}_mean']:.6f}"
    )

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [],
        "metrics": metrics
    }
