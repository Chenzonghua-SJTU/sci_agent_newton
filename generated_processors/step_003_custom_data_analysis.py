import os
import json
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload.get("action", "custom_data_analysis")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 确定实验 ID 列表
    exp_ids = params.get("experiment_ids", None)
    if exp_ids is None:
        single_id = params.get("experiment_id", None)
        if single_id is not None:
            exp_ids = [single_id]
        else:
            exp_ids = list(experiments.keys())

    analysis_goal = params.get("analysis_goal", "")
    expected_outputs = params.get("expected_outputs", [])

    derived_series = []
    metrics = {}
    figures = []

    for eid in exp_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        series = exp.get("series", {})
        if "t" not in series or "q" not in series:
            continue
        t = np.array(series["t"], dtype=float)
        q = np.array(series["q"], dtype=float)
        dt = t[1] - t[0]  # 假设均匀采样

        # 使用中心差分（np.gradient）得到与 t 等长的速度、加速度
        # 这与题目中“一阶差分（速度）”概念一致，且便于返回派生序列
        v = np.gradient(q, dt)          # 速度，长度 = len(t)
        a = np.gradient(v, dt)          # 加速度，长度 = len(t)

        # 统计量
        v_mean = float(np.mean(v))
        v_std = float(np.std(v, ddof=1))
        a_mean = float(np.mean(a))
        a_std = float(np.std(a, ddof=1))

        # 线性拟合残差 MSE
        coeffs = np.polyfit(t, q, 1)        # q = slope * t + intercept
        q_fit = np.polyval(coeffs, t)
        residual_mse = float(np.mean((q - q_fit)**2))

        # 存储 metrics
        metrics[f"{eid}_v_mean"] = v_mean
        metrics[f"{eid}_v_std"] = v_std
        metrics[f"{eid}_a_mean"] = a_mean
        metrics[f"{eid}_a_std"] = a_std
        metrics[f"{eid}_linear_residual_MSE"] = residual_mse

        # 派生序列（中心差分得到的速度和加速度）
        derived_series.append({
            "experiment_id": eid,
            "name": "v_central_diff",
            "values": v.tolist(),
            "source_name": "np.gradient(q, dt)",
            "provenance": "generated data processor: custom_data_analysis_step",
            "description": "中心差分估计的速度，长度与t一致"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "a_central_diff",
            "values": a.tolist(),
            "source_name": "np.gradient(v, dt)",
            "provenance": "generated data processor: custom_data_analysis_step",
            "description": "中心差分估计的加速度，长度与t一致"
        })

        # 绘图（可选，增加观察丰富性）
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        # q-t
        axes[0,0].plot(t, q, 'b-', label='q(t)')
        axes[0,0].plot(t, q_fit, 'r--', label='linear fit')
        axes[0,0].set_xlabel('t')
        axes[0,0].set_ylabel('q')
        axes[0,0].set_title(f'{eid}: q vs t')
        axes[0,0].legend()
        axes[0,0].grid(True)

        # v-t
        axes[0,1].plot(t, v, 'g-', label='v (central diff)')
        axes[0,1].axhline(y=v_mean, color='r', linestyle='--', label=f'mean={v_mean:.6f}')
        axes[0,1].set_xlabel('t')
        axes[0,1].set_ylabel('v')
        axes[0,1].set_title(f'{eid}: v vs t')
        axes[0,1].legend()
        axes[0,1].grid(True)

        # a-t
        axes[1,0].plot(t, a, 'm-', label='a (central diff)')
        axes[1,0].axhline(y=a_mean, color='r', linestyle='--', label=f'mean={a_mean:.6f}')
        axes[1,0].set_xlabel('t')
        axes[1,0].set_ylabel('a')
        axes[1,0].set_title(f'{eid}: a vs t')
        axes[1,0].legend()
        axes[1,0].grid(True)

        # residual plot
        residual = q - q_fit
        axes[1,1].plot(t, residual, 'ko', markersize=2)
        axes[1,1].axhline(y=0, color='gray', linestyle='--')
        axes[1,1].set_xlabel('t')
        axes[1,1].set_ylabel('residual')
        axes[1,1].set_title(f'{eid}: residual (MSE={residual_mse:.3e})')
        axes[1,1].grid(True)

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"{eid}_kinematics.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        figures.append(fig_path)

    # 构建 observation 文本
    obs_parts = [f"对实验 {exp_ids} 执行自定义数据分析："]
    obs_parts.append("使用中心差分(np.gradient)从q(t)估计速度v和加速度a序列（长度与t相同），并计算统计量、线性拟合残差MSE。")
    for eid in exp_ids:
        if eid not in experiments:
            continue
        vm = metrics.get(f"{eid}_v_mean")
        vs = metrics.get(f"{eid}_v_std")
        am = metrics.get(f"{eid}_a_mean")
        as_ = metrics.get(f"{eid}_a_std")
        rmse = metrics.get(f"{eid}_linear_residual_MSE")
        obs_parts.append(f"  {eid}: v_mean={vm:.6f}, v_std={vs:.6e}; a_mean={am:.6e}, a_std={as_:.6e}; 线性残差MSE={rmse:.6e}")
    obs_parts.append("速度/加速度序列和图像已返回，可供后续分析使用。")
    observation = "\n".join(obs_parts)

    result = {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
    return result
