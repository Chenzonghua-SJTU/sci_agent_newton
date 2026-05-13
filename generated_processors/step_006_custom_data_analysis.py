import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        experiment_ids = list(experiments.keys())

    analysis_goal = params.get("analysis_goal", "")

    if len(experiment_ids) != 1 or experiment_ids[0] != "exp_04":
        raise ValueError(f"当前 action 只处理 exp_04，但收到: {experiment_ids}")

    exp_id = "exp_04"
    exp_data = experiments[exp_id]
    series = exp_data["series"]
    config = exp_data["config"]

    t = np.array(series["t"])
    q = np.array(series["q"])

    if len(t) != len(q):
        raise ValueError(f"t 和 q 长度不一致: {len(t)} vs {len(q)}")

    dt = config.get("dt")
    if dt is None:
        dt = t[1] - t[0] if len(t) > 1 else 1.0

    # 中心差分计算速度和加速度
    v = np.gradient(q, dt)
    a = np.gradient(v, dt)

    # 加速度统计
    a_mean = float(np.mean(a))
    a_std = float(np.std(a, ddof=0))

    # 检查加速度是否接近常数：如果标准差相对于均值（绝对值）很小，或者绝对值很小则视为常数
    # 只报告数值，不做判定
    if abs(a_mean) > 1e-12:
        rel_std = a_std / abs(a_mean)
    else:
        rel_std = a_std  # 均值接近0时用绝对值

    # q 对 t^2 的线性拟合
    t2 = t ** 2
    coeff_t2 = np.polyfit(t2, q, 1)  # [slope, intercept]
    slope_t2 = float(coeff_t2[0])
    intercept_t2 = float(coeff_t2[1])
    q_pred_t2 = np.polyval(coeff_t2, t2)
    residual_t2 = q - q_pred_t2
    mse_t2 = float(np.mean(residual_t2 ** 2))
    rmse_t2 = float(np.sqrt(mse_t2))

    # q 对 t 的二次多项式拟合
    coeff_quad = np.polyfit(t, q, 2)  # [a2, a1, a0]
    q_pred_quad = np.polyval(coeff_quad, t)
    residual_quad = q - q_pred_quad
    mse_quad = float(np.mean(residual_quad ** 2))
    rmse_quad = float(np.sqrt(mse_quad))

    # 构建派生序列
    derived_series = [
        {
            "experiment_id": exp_id,
            "name": "v_central_diff",
            "values": v.tolist(),
            "source_name": "np.gradient(q, dt)",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "中心差分法估计的速度序列"
        },
        {
            "experiment_id": exp_id,
            "name": "a_central_diff",
            "values": a.tolist(),
            "source_name": "np.gradient(v, dt)",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "中心差分法估计的加速度序列"
        }
    ]

    # 画图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # q-t
    axes[0, 0].plot(t, q, 'b-', label='q(t)')
    axes[0, 0].set_xlabel('t')
    axes[0, 0].set_ylabel('q')
    axes[0, 0].set_title('Position vs Time')
    axes[0, 0].grid(True)

    # v-t
    axes[0, 1].plot(t, v, 'g-', label='v (central diff)')
    axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('v')
    axes[0, 1].set_title('Velocity vs Time')
    axes[0, 1].grid(True)

    # a-t
    axes[1, 0].plot(t, a, 'r-', label='a (central diff)')
    axes[1, 0].axhline(y=a_mean, color='k', linestyle='--', label=f'mean={a_mean:.4f}')
    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('a')
    axes[1, 0].set_title(f'Acceleration vs Time (std={a_std:.4f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # q vs t^2 拟合与二次拟合对比
    axes[1, 1].scatter(t, q, s=10, color='blue', label='data', alpha=0.6)
    axes[1, 1].plot(t, q_pred_t2, 'r-', label=f'linear t^2 fit (slope={slope_t2:.4f})')
    axes[1, 1].plot(t, q_pred_quad, 'g--', label=f'quadratic fit')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('q')
    axes[1, 1].set_title('Fits')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{exp_id}_kinematics_fits.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)

    # 返回 metrics
    metrics = {
        f"{exp_id}_a_mean": a_mean,
        f"{exp_id}_a_std": a_std,
        f"{exp_id}_a_rel_std": rel_std,
        f"{exp_id}_linear_t2_slope": slope_t2,
        f"{exp_id}_linear_t2_intercept": intercept_t2,
        f"{exp_id}_linear_t2_mse": mse_t2,
        f"{exp_id}_linear_t2_rmse": rmse_t2,
        f"{exp_id}_quadratic_coeff_a2": float(coeff_quad[0]),
        f"{exp_id}_quadratic_coeff_a1": float(coeff_quad[1]),
        f"{exp_id}_quadratic_coeff_a0": float(coeff_quad[2]),
        f"{exp_id}_quadratic_mse": mse_quad,
        f"{exp_id}_quadratic_rmse": rmse_quad,
        "window_length": len(t),
        "method": "central_difference"
    }

    observation = (
        f"对实验 {exp_id} (constant force, F_ext={config.get('constant_force', 'N/A')}, "
        f"q0={config.get('initial_q', 'N/A')}, v0={config.get('initial_v', 'N/A')}) "
        f"的位置序列 q(t) 进行了运动学分析。\n"
        f"使用中心差分法 (np.gradient, dt={dt}) 估计速度和加速度序列。\n"
        f"加速度均值 = {a_mean:.6f}, 标准差 = {a_std:.6f}, 相对标准差 = {rel_std:.6f}。\n"
        f"q(t) 对 t^2 的线性拟合斜率 = {slope_t2:.6f}, 截距 = {intercept_t2:.6f}, "
        f"MSE = {mse_t2:.6e}, RMSE = {rmse_t2:.6e}。\n"
        f"二次多项式拟合系数: a2={coeff_quad[0]:.6f}, a1={coeff_quad[1]:.6f}, a0={coeff_quad[2]:.6f}, "
        f"MSE = {mse_quad:.6e}, RMSE = {rmse_quad:.6e}。\n"
        f"已生成派生序列 v_central_diff, a_central_diff 及运动学拟合图。"
    )

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [fig_path],
        "metrics": metrics
    }
