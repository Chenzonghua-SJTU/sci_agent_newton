import numpy as np
from scipy.signal import savgol_filter
import os

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    output_dir = payload.get("output_dir", ".")
    experiments = payload.get("experiments", {})

    # 提取参数
    source_series = params.get("source_series", "q")
    position_name = params.get("position_name", "q_smooth")
    velocity_name = params.get("velocity_name", "v_sg")
    acceleration_name = params.get("acceleration_name", "a_sg")
    window_length = params.get("window_length", 5)
    polyorder = params.get("polyorder", 2)
    overwrite = params.get("overwrite", True)
    experiment_ids = params.get("experiment_ids", list(experiments.keys()))

    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")
    if polyorder >= window_length:
        raise ValueError(f"polyorder ({polyorder}) must be less than window_length ({window_length})")

    derived_series = []
    processed_exps = []

    for eid in experiment_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        series_dict = exp.get("series", {})
        t = np.array(series_dict.get("t", []))
        q = np.array(series_dict.get(source_series, []))

        if len(t) == 0:
            raise ValueError(f"Experiment {eid}: t series is empty.")
        if len(q) == 0:
            raise ValueError(f"Experiment {eid}: source series '{source_series}' is empty.")
        if len(t) != len(q):
            raise ValueError(f"Experiment {eid}: length mismatch between t ({len(t)}) and {source_series} ({len(q)}).")
        if len(q) < window_length:
            raise ValueError(f"Experiment {eid}: series length ({len(q)}) is shorter than window_length ({window_length}).")

        # 平滑位置
        q_smooth = savgol_filter(q, window_length, polyorder, mode='interp')

        # 估计速度 (一阶导)
        v_sg = savgol_filter(q_smooth, window_length, polyorder, deriv=1, delta=t[1] - t[0] if len(t) > 1 else 1.0, mode='interp')

        # 估计加速度 (二阶导)
        a_sg = savgol_filter(q_smooth, window_length, polyorder, deriv=2, delta=t[1] - t[0] if len(t) > 1 else 1.0, mode='interp')

        # 准备派生序列
        new_sequences = {
            position_name: q_smooth.tolist(),
            velocity_name: v_sg.tolist(),
            acceleration_name: a_sg.tolist()
        }

        for sname, svalues in new_sequences.items():
            # 检查是否已有同名序列
            already_exists = sname in series_dict
            if already_exists and not overwrite:
                continue
            derived_series.append({
                "experiment_id": eid,
                "name": sname,
                "values": svalues,
                "source_name": f"Savitzky-Golay filter (window={window_length}, polyorder={polyorder}) from '{source_series}'",
                "provenance": "generated data processor: estimate_kinematics",
                "description": f"Smoothed position / velocity / acceleration estimated via Savitzky-Golay filter."
            })

        processed_exps.append(eid)

    # 构建 observation
    n_processed = len(processed_exps)
    obs_parts = [
        f"对 {n_processed} 个实验 ({', '.join(processed_exps)}) 使用 Savitzky-Golay 滤波 "
        f"(窗口={window_length}, 多项式阶={polyorder}) 从序列 '{source_series}' 估计了平滑位置 '{position_name}'、"
        f"速度 '{velocity_name}' 和加速度 '{acceleration_name}'。"
    ]

    # 可选的统计概要（仅对第一个实验示例）
    if processed_exps:
        first_eid = processed_exps[0]
        first_exp = experiments[first_eid]
        t_first = np.array(first_exp["series"].get("t", []))
        v_first = None
        for ds in derived_series:
            if ds["experiment_id"] == first_eid and ds["name"] == velocity_name:
                v_first = np.array(ds["values"])
                break
        if v_first is not None and len(v_first) > 0:
            obs_parts.append(
                f"示例实验 {first_eid}: 速度均值={v_first.mean():.4f}, 标准差={v_first.std():.4f}。"
            )

    observation = " ".join(obs_parts)

    metrics = {}
    if processed_exps:
        # 计算每个实验的速度和加速度均值作为指标
        for eid in processed_exps:
            for ds in derived_series:
                if ds["experiment_id"] == eid and ds["name"] == velocity_name:
                    v_vals = np.array(ds["values"])
                    metrics[f"{eid}_v_mean"] = float(v_vals.mean())
                    metrics[f"{eid}_v_std"] = float(v_vals.std())
                if ds["experiment_id"] == eid and ds["name"] == acceleration_name:
                    a_vals = np.array(ds["values"])
                    metrics[f"{eid}_a_mean"] = float(a_vals.mean())
                    metrics[f"{eid}_a_std"] = float(a_vals.std())

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [],
        "metrics": metrics
    }
