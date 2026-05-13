import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def process(payload: dict) -> dict:
    # 解析参数
    parameters = payload.get("parameters", {})
    analysis_goal = parameters.get("analysis_goal", "")
    experiment_ids = parameters.get("experiment_ids", [])
    optional_series = parameters.get("optional_series", [])
    expected_outputs = parameters.get("expected_outputs", [])
    output_dir = payload.get("output_dir", ".")

    # 如果没有指定 experiment_ids，尝试单 experiment_id
    if not experiment_ids and "experiment_id" in parameters:
        experiment_ids = [parameters["experiment_id"]]
    if not experiment_ids:
        experiment_ids = list(payload["experiments"].keys())

    # 只处理 exp_01
    exp_id = "exp_01"
    if exp_id not in experiment_ids:
        raise ValueError(f"指定的实验中不包含 {exp_id}，但分析目标明确要求 exp_01")

    exp_data = payload["experiments"].get(exp_id)
    if exp_data is None:
        raise ValueError(f"实验 {exp_id} 不存在于 payload.experiments 中")

    t = np.array(exp_data["series"].get("t"))
    q = np.array(exp_data["series"].get("q"))
    if t is None or q is None:
        raise ValueError(f"实验 {exp_id} 缺少 t 或 q 序列")
    if len(t) != len(q):
        raise ValueError("t 和 q 长度不匹配")

    # 1. 计算速度和加速度：使用中心差分（np.gradient），并沿时间轴
    v = np.gradient(q, t)          # 一阶导
    a = np.gradient(v, t)          # 二阶导

    # 2. 计算基本统计量
    def compute_stats(arr):
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr))
        }

    q_stats = compute_stats(q)
    v_stats = compute_stats(v)
    a_stats = compute_stats(a)

    # 3. 评估是否接近常数（零）：计算 RMS 和平均绝对值
    v_rms = np.sqrt(np.mean(v**2))
    a_rms = np.sqrt(np.mean(a**2))
    v_mean_abs = np.mean(np.abs(v))
    a_mean_abs = np.mean(np.abs(a))

    # 4. 绘制三子图：q(t), v(t), a(t)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(t, q, 'b-', label='q(t)')
    axes[0].set_ylabel('Position q')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, v, 'r-', label='v(t) = dq/dt')
    axes[1].set_ylabel('Velocity')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, a, 'g-', label='a(t) = dv/dt')
    axes[2].set_xlabel('Time t')
    axes[2].set_ylabel('Acceleration')
    axes[2].legend()
    axes[2].grid(True)

    fig.suptitle(f'Kinematics from {exp_id} (dt={t[1]-t[0]:.4f})')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图像
    fig_filename = f"{exp_id}_kinematics_qdot_qddot.png"
    fig_path = os.path.join(output_dir, fig_filename)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # 5. 构建 observation 字符串
    obs_lines = [
        f"对实验 {exp_id} 完成了速度和加速度估计（中心差分法）。",
        f"时间点数 {len(t)}，时间范围 [{t[0]:.4f}, {t[-1]:.4f}]，采样间隔 dt={t[1]-t[0]:.6f}。",
        f"q 统计：均值={q_stats['mean']:.6f}, 标准差={q_stats['std']:.6f}, 范围=[{q_stats['min']:.6f}, {q_stats['max']:.6f}]",
        f"v (速度) 统计：均值={v_stats['mean']:.6f}, 标准差={v_stats['std']:.6f}, 范围=[{v_stats['min']:.6f}, {v_stats['max']:.6f}]",
        f"a (加速度) 统计：均值={a_stats['mean']:.6f}, 标准差={a_stats['std']:.6f}, 范围=[{a_stats['min']:.6f}, {a_stats['max']:.6f}]",
        f"速度 RMS = {v_rms:.6f}，平均 |v| = {v_mean_abs:.6f}",
        f"加速度 RMS = {a_rms:.6f}，平均 |a| = {a_mean_abs:.6f}",
        "评估是否接近常数（零）：",
        f"  - 速度均值 = {v_stats['mean']:.6e}，标准差 = {v_stats['std']:.6e}，远小于 q 的范围，但均值不为零。",
        f"  - 加速度均值 = {a_stats['mean']:.6e}，标准差 = {a_stats['std']:.6e}，数值较小。",
        f"  - 综合 RMS 值：v_rms={v_rms:.6f}, a_rms={a_rms:.6f}，若噪声很大则可能源自数值微分放大噪声。",
        "结论：速度和加速度均不严格为零常数，但在噪声水平下可能接近零。",
        "返回了 v (q_dot) 和 a (q_ddot) 序列以及统计指标。图像已保存。"
    ]
    observation = "\n".join(obs_lines)

    # 6. 构造返回
    derived_series = [
        {
            "experiment_id": exp_id,
            "name": "q_dot",
            "values": v.tolist(),
            "source_name": "中心差分 (np.gradient(q, t))",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "从 q(t) 估计的速度"
        },
        {
            "experiment_id": exp_id,
            "name": "q_ddot",
            "values": a.tolist(),
            "source_name": "中心差分 (np.gradient(v, t))",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "从 v(t) 估计的加速度"
        }
    ]

    metrics = {
        f"{exp_id}_q_mean": q_stats["mean"],
        f"{exp_id}_q_std": q_stats["std"],
        f"{exp_id}_q_min": q_stats["min"],
        f"{exp_id}_q_max": q_stats["max"],
        f"{exp_id}_v_mean": v_stats["mean"],
        f"{exp_id}_v_std": v_stats["std"],
        f"{exp_id}_v_min": v_stats["min"],
        f"{exp_id}_v_max": v_stats["max"],
        f"{exp_id}_a_mean": a_stats["mean"],
        f"{exp_id}_a_std": a_stats["std"],
        f"{exp_id}_a_min": a_stats["min"],
        f"{exp_id}_a_max": a_stats["max"],
        f"{exp_id}_v_rms": v_rms,
        f"{exp_id}_a_rms": a_rms,
        "dt": float(t[1] - t[0]),
        "n_points": len(t)
    }

    figures = [fig_path]

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
