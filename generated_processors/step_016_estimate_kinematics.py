import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload.get("action", "estimate_kinematics")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 获取实验列表
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        # 如果未指定，尝试单个 experiment_id
        single = params.get("experiment_id")
        if single:
            exp_ids = [single]
        else:
            exp_ids = list(experiments.keys())

    window_length = int(params.get("window_length", 13))
    polyorder = int(params.get("polyorder", 2))
    source_series = params.get("source_series", "q")
    overwrite = params.get("overwrite", True)

    derived_series = []
    figures = []
    metrics = {}
    observation_lines = []

    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"实验 {eid} 不存在于 payload 数据中。")

        exp = experiments[eid]
        config = exp.get("config", {})
        series = exp.get("series", {})
        available_series = exp.get("available_series", [])

        # 检查源序列
        if source_series not in series:
            raise ValueError(f"实验 {eid} 中找不到序列 {source_series}。可用序列: {available_series}")

        q = np.array(series[source_series], dtype=float)
        t_key = "t"
        if t_key not in series:
            # 如果 t 不是标准名，尝试找时间序列
            t_key = [s for s in available_series if s.startswith("t")][0]
        t = np.array(series[t_key], dtype=float)

        # 时间步长 dt
        dt = config.get("dt", None)
        if dt is None or dt == 0:
            # 从 t 序列差分估计
            dt = np.mean(np.diff(t))

        n = len(q)
        if n < window_length:
            raise ValueError(f"实验 {eid} 的序列长度 {n} 小于窗口长度 {window_length}。")

        # 平滑位置
        q_smooth = savgol_filter(q, window_length, polyorder)
        # 速度（一阶导数）
        v = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt)
        # 加速度（二阶导数）
        a = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)

        # 计算平滑 RMSE
        rmse = float(np.sqrt(np.mean((q - q_smooth)**2)))
        metrics[f"{eid}_smooth_rmse"] = rmse

        # 构造派生序列
        pos_name = "q_smooth"
        vel_name = "v"
        acc_name = "a"
        # 如果 overwrite=False 且系列已存在，则跳过（此处我们始终生成，需要由外层决定覆盖）
        # 但按参数，overwrite=True，所以直接生成
        derived_series.append({
            "experiment_id": eid,
            "name": pos_name,
            "values": q_smooth.tolist(),
            "source_name": f"Savitzky-Golay (window={window_length}, polyorder={polyorder}) smoothing of {source_series}",
            "provenance": f"generated data processor: {__file__}",
            "description": f"Smoothed position from {source_series} using savgol_filter"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": vel_name,
            "values": v.tolist(),
            "source_name": f"Savitzky-Golay first derivative of {source_series} (window={window_length})",
            "provenance": f"generated data processor: {__file__}",
            "description": f"Velocity estimated via savgol_filter derivative"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": acc_name,
            "values": a.tolist(),
            "source_name": f"Savitzky-Golay second derivative of {source_series} (window={window_length})",
            "provenance": f"generated data processor: {__file__}",
            "description": f"Acceleration estimated via savgol_filter second derivative"
        })

        # 记录观察
        observation_lines.append(
            f"实验 {eid}: {source_series} 平滑后 {pos_name} RMSE={rmse:.6f}；使用参数 window_length={window_length}, polyorder={polyorder}, dt={dt:.4f}。已生成 {pos_name}, {vel_name}, {acc_name}。"
        )

        # 绘图
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(t, q, 'b-', label=f'original {source_series}', alpha=0.6)
        axes[0].plot(t, q_smooth, 'r-', label=f'{pos_name} (smooth)', linewidth=2)
        axes[0].set_ylabel('Position')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(t, v, 'g-', label=vel_name)
        axes[1].set_ylabel('Velocity')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(t, a, 'm-', label=acc_name)
        axes[2].set_ylabel('Acceleration')
        axes[2].legend()
        axes[2].grid(True)

        axes[2].set_xlabel('Time')
        fig.suptitle(f'Kinematics for {eid} (window={window_length}, polyorder={polyorder})')
        plt.tight_layout()

        fig_path = os.path.join(output_dir, f"kinematics_{eid}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        figures.append(fig_path)

    # 汇总观察
    observation = "；".join(observation_lines)
    observation = f"对实验 {exp_ids} 进行了运动学估计。" + observation

    # 可选：附加整体指标（均值标准差等）
    rmse_list = [metrics[f"{eid}_smooth_rmse"] for eid in exp_ids]
    if rmse_list:
        metrics["smooth_rmse_mean"] = float(np.mean(rmse_list))
        metrics["smooth_rmse_std"] = float(np.std(rmse_list))
        metrics["smooth_rmse_list"] = rmse_list

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
