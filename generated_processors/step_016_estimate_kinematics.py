import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

def process(payload: dict) -> dict:
    action = payload.get("action", "estimate_kinematics")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 解析参数
    exp_ids = params.get("experiment_ids", params.get("experiment_id", None))
    if exp_ids is None:
        exp_ids = list(experiments.keys())
    elif isinstance(exp_ids, str):
        exp_ids = [exp_ids]

    overwrite = params.get("overwrite", True)
    window_length = params.get("window_length", 21)
    polyorder = params.get("polyorder", 3)
    source_series = params.get("source_series", "q")
    position_name = params.get("position_name", "q_smooth")
    velocity_name = params.get("velocity_name", "v_est")
    acceleration_name = params.get("acceleration_name", "a_est")

    derived_series = []
    figures = []
    metrics = {}
    obs_parts = []

    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload['experiments']")
        exp = experiments[eid]
        series = exp.get("series", {})
        if source_series not in series:
            raise ValueError(f"Source series '{source_series}' not available in experiment {eid}")

        t = np.array(series.get("t", []))
        q = np.array(series.get(source_series, []))
        if len(t) == 0 or len(q) == 0:
            raise ValueError(f"Experiment {eid}: t or {source_series} series is empty")

        if len(t) != len(q):
            raise ValueError(f"Experiment {eid}: t and {source_series} series lengths differ")
        n = len(t)
        dt = t[1] - t[0] if n > 1 else 1.0

        # 应用 Savitzky-Golay 滤波器
        q_smooth = savgol_filter(q, window_length=window_length, polyorder=polyorder, deriv=0)
        v = savgol_filter(q, window_length=window_length, polyorder=polyorder, deriv=1, delta=dt)
        a = savgol_filter(q, window_length=window_length, polyorder=polyorder, deriv=2, delta=dt)

        # 注册派生序列
        derived_series.append({
            "experiment_id": eid,
            "name": position_name,
            "values": q_smooth.tolist(),
            "source_name": f"Savitzky-Golay filter (window={window_length}, poly={polyorder}) applied to {source_series}",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"Smoothed position from {source_series} using Savitzky-Golay"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": velocity_name,
            "values": v.tolist(),
            "source_name": f"Savitzky-Golay filter (window={window_length}, poly={polyorder}, deriv=1) of {source_series}",
            "provenance": "generated data processor: estimate_kinematics",
            "description": "Estimated velocity (first derivative of position)"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": acceleration_name,
            "values": a.tolist(),
            "source_name": f"Savitzky-Golay filter (window={window_length}, poly={polyorder}, deriv=2) of {source_series}",
            "provenance": "generated data processor: estimate_kinematics",
            "description": "Estimated acceleration (second derivative of position)"
        })

        # 统计
        v_mean = float(np.mean(v))
        v_std = float(np.std(v, ddof=1)) if n > 1 else 0.0
        a_mean = float(np.mean(a))
        a_std = float(np.std(a, ddof=1)) if n > 1 else 0.0

        metrics[f"{eid}_v_mean"] = v_mean
        metrics[f"{eid}_v_std"] = v_std
        metrics[f"{eid}_a_mean"] = a_mean
        metrics[f"{eid}_a_std"] = a_std

        # 图像
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t, q_smooth, label=f'{position_name} (smooth)', color='green')
        axes[0].plot(t, q, label=f'{source_series} (raw)', alpha=0.5, color='gray')
        axes[0].set_ylabel('Position')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, v, label=velocity_name, color='blue')
        axes[1].set_ylabel('Velocity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, a, label=acceleration_name, color='red')
        axes[2].set_ylabel('Acceleration')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f"Experiment {eid}: Kinematics Estimation (Savitzky-Golay w={window_length}, p={polyorder})")
        fig.tight_layout()
        figure_path = os.path.join(output_dir, f"{eid}_kinematics.png")
        fig.savefig(figure_path, dpi=100)
        plt.close(fig)
        figures.append(figure_path)

        obs_parts.append(
            f"实验 {eid}: 点数={n}, dt={dt:.4f}\n"
            f"  速度: 均值={v_mean:.6f}, 标准差={v_std:.6f}\n"
            f"  加速度: 均值={a_mean:.6f}, 标准差={a_std:.6f}\n"
        )

    observation = "运动学估计完成。\n" + "\n".join(obs_parts) + f"\n图像已保存: {', '.join(figures)}"
    if overwrite:
        observation += "\n已覆盖同名序列（overwrite=True）。"

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
