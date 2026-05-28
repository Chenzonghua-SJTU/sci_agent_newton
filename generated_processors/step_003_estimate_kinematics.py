import numpy as np
from scipy.signal import savgol_filter
import os, json

def process(payload: dict) -> dict:
    action = payload["action"]
    parameters = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload.get("output_dir", ".")

    # 提取参数
    exp_ids = parameters.get("experiment_ids", parameters.get("experiment_id", None))
    if exp_ids is None:
        exp_ids = list(experiments.keys())
    elif isinstance(exp_ids, str):
        exp_ids = [exp_ids]
    source_series = parameters.get("source_series", "q")
    pos_name = parameters.get("position_name", "q_smooth")
    vel_name = parameters.get("velocity_name", "v")
    acc_name = parameters.get("acceleration_name", "a")
    window_length = int(parameters.get("window_length", 11))
    polyorder = int(parameters.get("polyorder", 2))
    overwrite = parameters.get("overwrite", False)

    # 参数校验
    if window_length % 2 == 0:
        raise ValueError(f"window_length ({window_length}) 必须是奇数")
    if polyorder >= window_length:
        raise ValueError(f"polyorder ({polyorder}) 必须小于 window_length ({window_length})")

    derived_series = []
    metrics = {}
    observation_parts = []

    # 收集所有实验的平滑前后 RMSE
    pos_rmse_list = []

    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"实验 {eid} 不存在于 payload 中")
        exp = experiments[eid]
        series = exp["series"]
        avail = exp.get("available_series", list(series.keys()))
        t = np.array(series.get("t", []), dtype=float)
        src = np.array(series.get(source_series, None), dtype=float)
        if src is None:
            raise ValueError(f"实验 {eid} 中不存在序列 {source_series}")
        n = len(t)
        if len(src) != n:
            raise ValueError(f"实验 {eid} 中 {source_series} 长度 ({len(src)}) 与 t 长度 ({n}) 不一致")
        if n < window_length:
            raise ValueError(f"实验 {eid} 点数 ({n}) 小于 window_length ({window_length})")

        # 检查是否有同名序列并决定覆盖
        if pos_name in avail and not overwrite:
            raise ValueError(f"实验 {eid} 已存在序列 {pos_name} 且 overwrite=False，请先设置 overwrite=True 或改名")
        if vel_name in avail and not overwrite:
            raise ValueError(f"实验 {eid} 已存在序列 {vel_name} 且 overwrite=False")
        if acc_name in avail and not overwrite:
            raise ValueError(f"实验 {eid} 已存在序列 {acc_name} 且 overwrite=False")

        # 计算平滑
        q_smooth = savgol_filter(src, window_length, polyorder, deriv=0, mode='interp')
        v = savgol_filter(src, window_length, polyorder, deriv=1, delta=exp["config"].get("dt", 0.1), mode='interp')
        a = savgol_filter(src, window_length, polyorder, deriv=2, delta=exp["config"].get("dt", 0.1), mode='interp')

        # 计算平滑前后 RMSE (仅对位置)
        rmse = np.sqrt(np.mean((src - q_smooth)**2))
        pos_rmse_list.append(rmse)

        # 添加派生序列
        derived_series.append({
            "experiment_id": eid,
            "name": pos_name,
            "values": q_smooth.tolist(),
            "source_name": f"savgol_filter({source_series}, window={window_length}, polyorder={polyorder}, deriv=0)",
            "provenance": "generated data processor: estimate_kinematics",
            "description": "经 Savitzky-Golay 平滑后的位置"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": vel_name,
            "values": v.tolist(),
            "source_name": f"savgol_filter({source_series}, window={window_length}, polyorder={polyorder}, deriv=1, delta=dt)",
            "provenance": "generated data processor: estimate_kinematics",
            "description": "由平滑位置估计的速度（一阶导数）"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": acc_name,
            "values": a.tolist(),
            "source_name": f"savgol_filter({source_series}, window={window_length}, polyorder={polyorder}, deriv=2, delta=dt)",
            "provenance": "generated data processor: estimate_kinematics",
            "description": "由平滑位置估计的加速度（二阶导数）"
        })

        # 记录观察
        observation_parts.append(f"实验 {eid}: {source_series} 平滑后 q_smooth RMSE={rmse:.6f}")

    # 全局指标
    if pos_rmse_list:
        metrics["smooth_rmse_mean"] = float(np.mean(pos_rmse_list))
        metrics["smooth_rmse_std"] = float(np.std(pos_rmse_list))
        metrics["smooth_rmse_list"] = [float(x) for x in pos_rmse_list]
    observation = "；".join(observation_parts)
    if observation:
        observation += "。"
    else:
        observation = "未处理任何实验。"
    observation += f"使用参数 window_length={window_length}, polyorder={polyorder}。已生成 {pos_name}, {vel_name}, {acc_name}。"

    # 可选：生成可视化
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = []
    # 为每个实验绘制四子图
    for eid in exp_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        t = np.array(exp["series"]["t"], dtype=float)
        src = np.array(exp["series"][source_series], dtype=float)
        qs = np.array([ds["values"] for ds in derived_series if ds["experiment_id"]==eid and ds["name"]==pos_name][0])
        v_arr = np.array([ds["values"] for ds in derived_series if ds["experiment_id"]==eid and ds["name"]==vel_name][0])
        a_arr = np.array([ds["values"] for ds in derived_series if ds["experiment_id"]==eid and ds["name"]==acc_name][0])
        fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
        axs[0].plot(t, src, 'b-', label=f'raw {source_series}')
        axs[0].plot(t, qs, 'r--', label=f'{pos_name}')
        axs[0].set_ylabel(source_series)
        axs[0].legend()
        axs[1].plot(t, qs, 'g-', label=pos_name)
        axs[1].set_ylabel(pos_name)
        axs[1].legend()
        axs[2].plot(t, v_arr, 'm-', label=vel_name)
        axs[2].set_ylabel(vel_name)
        axs[2].legend()
        axs[3].plot(t, a_arr, 'c-', label=acc_name)
        axs[3].set_xlabel('t')
        axs[3].set_ylabel(acc_name)
        axs[3].legend()
        fig.suptitle(f'{eid} - Kinematics estimation')
        fname = f'kinematics_{eid}.png'
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)
    if figures:
        observation += f" 图像已保存至 {figures}。"

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
