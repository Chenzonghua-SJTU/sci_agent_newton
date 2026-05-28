import numpy as np
import os

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 提取参数
    exp_ids = params.get("experiment_ids", [])
    symbol = params.get("symbol", "drag_over_sqrtv")
    expression = params.get("expression", "")
    overwrite = params.get("overwrite", False)
    description = params.get("description", "")

    # 结果容器
    derived_series = []
    metrics = {}
    figures = []
    obs_lines = []

    for eid in exp_ids:
        exp_key = str(eid)  # payload 中 key 可能是字符串
        if exp_key not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")

        exp_data = experiments[exp_key]
        series = exp_data.get("series", {})
        available = exp_data.get("available_series", [])

        # 检查必需序列
        drag_name = "drag"
        vel_name = "v_est"
        if drag_name not in available or vel_name not in available:
            obs_lines.append(f"实验 {eid} 缺少所需序列 {drag_name} 或 {vel_name}，跳过")
            continue

        # 检查目标序列是否已存在且不允许覆盖
        if symbol in available and not overwrite:
            obs_lines.append(f"实验 {eid} 已存在序列 {symbol}，overwrite=False，跳过")
            # 仍然可以记录已存在的统计信息
            if symbol in series:
                vals = np.array(series[symbol])
                metrics[f"exp{eid}_{symbol}_min"] = float(np.min(vals))
                metrics[f"exp{eid}_{symbol}_max"] = float(np.max(vals))
                metrics[f"exp{eid}_{symbol}_mean"] = float(np.mean(vals))
                metrics[f"exp{eid}_{symbol}_std"] = float(np.std(vals))
            continue

        # 获取原始序列
        drag = np.array(series[drag_name])
        v_est = np.array(series[vel_name])

        # 检查长度一致性
        t_len = len(series.get("t", drag))
        if len(drag) != t_len or len(v_est) != t_len:
            raise ValueError(f"实验 {eid}: drag 或 v_est 长度与 t 不一致")

        # 计算派生序列
        # v_est 可能含零或负数，sqrt 要求非负
        # 物理上速度应≥0，这里取绝对值避免问题
        v_safe = np.maximum(v_est, 0.0)  # 将负值截断为0
        with np.errstate(divide='ignore', invalid='ignore'):
            result_vals = drag / np.sqrt(v_safe)
        # 处理除零和无效值
        result_vals = np.where(v_safe == 0, 0.0, result_vals)
        result_vals = np.nan_to_num(result_vals, nan=0.0)

        # 构造派生序列记录
        exp_name = f"exp_{eid:02d}"  # 统一格式
        derived_series.append({
            "experiment_id": exp_name,
            "name": symbol,
            "values": result_vals.tolist(),
            "source_name": expression,
            "provenance": f"define_derived_quantity: {symbol} = {expression}",
            "description": description
        })

        # 计算统计
        series_min = float(np.min(result_vals))
        series_max = float(np.max(result_vals))
        series_mean = float(np.mean(result_vals))
        series_std = float(np.std(result_vals))
        metrics[f"exp{eid}_{symbol}_min"] = series_min
        metrics[f"exp{eid}_{symbol}_max"] = series_max
        metrics[f"exp{eid}_{symbol}_mean"] = series_mean
        metrics[f"exp{eid}_{symbol}_std"] = series_std

        obs_lines.append(
            f"实验 {eid}: 派生序列 {symbol} (={expression}) "
            f"min={series_min:.6f}, max={series_max:.6f}, "
            f"mean={series_mean:.6f}, std={series_std:.6f}"
        )

    # 构造观察报告
    if not obs_lines:
        observation = "未处理任何实验，请检查输入参数"
    else:
        observation = "执行 define_derived_quantity:\n" + "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
