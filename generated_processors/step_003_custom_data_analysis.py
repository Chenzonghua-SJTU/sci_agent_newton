import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def process(payload: dict) -> dict:
    # 仅处理 exp_02
    exp_id = "exp_02"
    if exp_id not in payload["experiments"]:
        raise ValueError(f"Experiment {exp_id} not found in payload")
    exp = payload["experiments"][exp_id]
    t = np.array(exp["series"]["t"])
    q = np.array(exp["series"]["q"])
    if len(t) != len(q):
        raise ValueError(f"Length mismatch: t={len(t)}, q={len(q)}")
    dt = t[1] - t[0]
    if not np.isclose(dt, float(exp["config"]["dt"]), rtol=1e-6):
        dt = float(exp["config"]["dt"])

    # 参数
    window = 5
    polyorder = 2
    # 使用 Savitzky-Golay 滤波估计速度和加速度
    v = savgol_filter(q, window, polyorder, deriv=1, delta=dt)
    a = savgol_filter(q, window, polyorder, deriv=2, delta=dt)

    # 统计量
    v_mean = float(np.mean(v))
    v_std = float(np.std(v, ddof=1))
    a_mean = float(np.mean(a))
    a_std = float(np.std(a, ddof=1))

    # 判断是否恒定（阈值 1e-10）
    v_constant = v_std < 1e-10
    a_constant = a_std < 1e-10

    # 绘制图像
    output_dir = payload.get("output_dir", ".")
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(t, q, 'b-', label='q')
    axes[0].set_ylabel('Position')
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(t, v, 'r-', label='v')
    axes[1].set_ylabel('Velocity')
    axes[1].grid(True)
    axes[1].legend()
    axes[2].plot(t, a, 'g-', label='a')
    axes[2].set_ylabel('Acceleration')
    axes[2].set_xlabel('Time')
    axes[2].grid(True)
    axes[2].legend()
    fig.suptitle(f'Kinematics for {exp_id}')
    plt.tight_layout()
    fig_path = f"{output_dir}/kinematics_{exp_id}.png"
    fig.savefig(fig_path)
    plt.close(fig)

    # 派生序列
    derived_series = [
        {
            "experiment_id": exp_id,
            "name": "velocity_series",
            "values": v.tolist(),
            "source_name": f"Savgol deriv=1, window={window}, order={polyorder} on q(t)",
            "provenance": "custom_data_analysis processor",
            "description": "速度序列（从位置经Savitzky-Golay滤波估计）"
        },
        {
            "experiment_id": exp_id,
            "name": "acceleration_series",
            "values": a.tolist(),
            "source_name": f"Savgol deriv=2, window={window}, order={polyorder} on q(t)",
            "provenance": "custom_data_analysis processor",
            "description": "加速度序列（从位置经Savitzky-Golay滤波估计）"
        }
    ]

    # 指标
    metrics = {
        "v_mean": v_mean,
        "v_std": v_std,
        "a_mean": a_mean,
        "a_std": a_std,
        "v_constant": v_constant,
        "a_constant": a_constant
    }

    # 观察报告
    observation = (
        f"对实验 {exp_id}（自由场景, F_ext=0, v0=1）从 q(t) 使用 Savitzky-Golay 滤波 "
        f"(窗口={window}, 阶数={polyorder}) 估计了速度和加速度。\n"
        f"速度均值 = {v_mean:.6f}, 标准差 = {v_std:.6e}\n"
        f"加速度均值 = {a_mean:.6e}, 标准差 = {a_std:.6e}\n"
        f"速度是否恒定（std<1e-10）: {v_constant}\n"
        f"加速度是否恒定（std<1e-10）: {a_constant}\n"
        f"新序列 'velocity_series' 和 'acceleration_series' 已返回。\n"
        f"图像已保存至 {fig_path}。"
    )

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [fig_path],
        "metrics": metrics
    }
