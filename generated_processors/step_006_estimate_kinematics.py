import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload["action"]
    parameters = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload.get("output_dir", ".")

    experiment_ids = parameters.get("experiment_ids", [])
    if not experiment_ids:
        raise ValueError("参数 experiment_ids 不能为空")

    source_series = parameters.get("source_series", "q")
    position_name = parameters.get("position_name")
    velocity_name = parameters.get("velocity_name", "v_est")
    acceleration_name = parameters.get("acceleration_name", "a_est")
    window_length = parameters.get("window_length", 21)
    polyorder = parameters.get("polyorder", 3)
    overwrite = parameters.get("overwrite", True)

    derived_series = []
    figures = []
    metrics = {}
    observation_lines = []

    for exp_id in experiment_ids:
        if exp_id not in experiments:
            raise ValueError(f"实验 {exp_id} 不在 payload 中")
        exp = experiments[exp_id]
        config = exp["config"]
        series = exp["series"]
        available = exp.get("available_series", [])

        # 获取时间序列和 dt
        if "t" not in series:
            raise ValueError(f"实验 {exp_id} 缺少时间序列 't'")
        t = np.array(series["t"])
        n_points = len(t)
        # 从 config 中读取 dt，如果不存在则从 t 差值推断
        dt = config.get("dt", None)
        if dt is None:
            dt = np.median(np.diff(t))
        if dt <= 0:
            raise ValueError(f"实验 {exp_id} 的时间步长 dt 无效 (dt={dt})")

        # 获取源序列
        if source_series not in series:
            raise ValueError(f"实验 {exp_id} 缺少源序列 '{source_series}'")
        y = np.array(series[source_series])

        if len(y) != n_points:
            raise ValueError(f"实验 {exp_id} 的源序列长度 ({len(y)}) 与 t 长度 ({n_points}) 不一致")

        # 检查窗口参数
        if window_length >= n_points:
            raise ValueError(f"window_length ({window_length}) 必须小于数据点数 ({n_points})")
        if polyorder >= window_length:
            raise ValueError(f"polyorder ({polyorder}) 必须小于 window_length ({window_length})")

        # 计算平滑位置（如果指定了 position_name）
        if position_name is not None:
            q_smooth = savgol_filter(y, window_length, polyorder, deriv=0, delta=dt)
            derived_series.append({
                "experiment_id": exp_id,
                "name": position_name,
                "values": q_smooth.tolist(),
                "source_name": f"Savitzky-Golay filter (window={window_length}, polyorder={polyorder})",
                "provenance": "generated data processor: estimate_kinematics",
                "description": f"平滑后的位置序列，从 {source_series} 估计"
            })

        # 计算速度
        v = savgol_filter(y, window_length, polyorder, deriv=1, delta=dt)
        derived_series.append({
            "experiment_id": exp_id,
            "name": velocity_name,
            "values": v.tolist(),
            "source_name": f"Savitzky-Golay filter first derivative (window={window_length}, polyorder={polyorder})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"从 {source_series} 估计的速度序列"
        })

        # 计算加速度
        a = savgol_filter(y, window_length, polyorder, deriv=2, delta=dt)
        derived_series.append({
            "experiment_id": exp_id,
            "name": acceleration_name,
            "values": a.tolist(),
            "source_name": f"Savitzky-Golay filter second derivative (window={window_length}, polyorder={polyorder})",
            "provenance": "generated data processor: estimate_kinematics",
            "description": f"从 {source_series} 估计的加速度序列"
        })

        # 统计
        v_mean = float(np.mean(v))
        v_std = float(np.std(v))
        a_mean = float(np.mean(a))
        a_std = float(np.std(a))

        metrics[f"{exp_id}_v_mean"] = v_mean
        metrics[f"{exp_id}_v_std"] = v_std
        metrics[f"{exp_id}_a_mean"] = a_mean
        metrics[f"{exp_id}_a_std"] = a_std

        observation_lines.append(
            f"实验 {exp_id}: 点数={n_points}, dt={dt:.4f}\n"
            f"  速度: 均值={v_mean:.6f}, 标准差={v_std:.6f}\n"
            f"  加速度: 均值={a_mean:.6f}, 标准差={a_std:.6f}"
        )

        # 绘制速度与加速度图
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].plot(t, v, 'b-', label=f"{velocity_name}")
        axes[0].set_ylabel("速度")
        axes[0].set_title(f"{exp_id}: 估计速度")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(t, a, 'r-', label=f"{acceleration_name}")
        axes[1].set_xlabel("时间 (s)")
        axes[1].set_ylabel("加速度")
        axes[1].set_title(f"{exp_id}: 估计加速度")
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        figname = f"{exp_id}_kinematics.png"
        figpath = os.path.join(output_dir, figname)
        plt.savefig(figpath)
        plt.close(fig)
        figures.append(figpath)
        observation_lines.append(f"  图像已保存: {figpath}")

    observation = "运动学估计完成。\n" + "\n".join(observation_lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
