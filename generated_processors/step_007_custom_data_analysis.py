import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload.get("action", "custom_data_analysis")
    parameters = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 只处理指定的实验
    exp_ids = parameters.get("experiment_ids", [])
    if not exp_ids:
        raise ValueError("No experiment_ids provided")

    # 存储结果
    derived_series_list = []
    figures_list = []
    metrics = {}
    observations = []

    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment '{eid}' not found in payload")
        exp_data = experiments[eid]
        series = exp_data.get("series", {})
        available = exp_data.get("available_series", [])
        t = np.array(series.get("t", []), dtype=float)
        q = np.array(series.get("q", []), dtype=float)
        if len(t) == 0 or len(q) == 0:
            raise ValueError(f"Experiment '{eid}' has empty t or q series")

        # 时间步长（假设均匀）
        dt = t[1] - t[0]
        n = len(t)

        # ---------- 二次多项式拟合 ----------
        coeffs = np.polyfit(t, q, 2)  # [a, b, c] for a*t^2 + b*t + c
        a_coeff, b_coeff, c_coeff = coeffs
        q_fit = a_coeff * t**2 + b_coeff * t + c_coeff
        residuals = q - q_fit
        residual_std = np.std(residuals)

        # ---------- 中心差分法计算加速度 ----------
        # 二阶中心差分: (q[i+1] - 2*q[i] + q[i-1]) / dt^2
        a_cd = np.full_like(q, np.nan)
        a_cd[1:-1] = (q[2:] - 2*q[1:-1] + q[:-2]) / (dt**2)
        # 端点用相邻值填充
        a_cd[0] = a_cd[1] if n > 1 else 0.0
        a_cd[-1] = a_cd[-2] if n > 1 else 0.0

        # 加速度统计
        acc_mean = np.nanmean(a_cd)
        acc_std = np.nanstd(a_cd)

        # ---------- 绘制加速度随时间变化图 ----------
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t, a_cd, 'b-', linewidth=1.5, label='a(t) (center difference)')
        ax.axhline(y=acc_mean, color='r', linestyle='--', linewidth=1, label=f'mean = {acc_mean:.4f}')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Acceleration a')
        ax.set_title(f'Experiment {eid}: Acceleration from Central Difference')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig_path = os.path.join(output_dir, f"{eid}_acceleration_center_diff.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        figures_list.append(fig_path)

        # ---------- 构建派生序列 ----------
        # 加速度序列
        derived_series_list.append({
            "experiment_id": eid,
            "name": "a_center_diff",
            "values": a_cd.tolist(),
            "source_name": "Central difference second derivative of q(t)",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "Acceleration computed via center difference (non-Savitzky-Golay)"
        })
        # 拟合残差序列
        derived_series_list.append({
            "experiment_id": eid,
            "name": "fit_residual",
            "values": residuals.tolist(),
            "source_name": "Quadratic fit residual = q - (a*t^2 + b*t + c)",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "Residual of quadratic polynomial fit to q(t)"
        })

        # ---------- 更新指标 ----------
        metrics[f"{eid}_fit_a"] = float(a_coeff)
        metrics[f"{eid}_fit_b"] = float(b_coeff)
        metrics[f"{eid}_fit_c"] = float(c_coeff)
        metrics[f"{eid}_residual_std"] = float(residual_std)
        metrics[f"{eid}_acc_mean"] = float(acc_mean)
        metrics[f"{eid}_acc_std"] = float(acc_std)

        # ---------- 观察文本 ----------
        obs = (
            f"实验 {eid} 分析完成。\n"
            f"  二次拟合系数: t^2系数 = {a_coeff:.6f}, t系数 = {b_coeff:.6f}, 常数项 = {c_coeff:.6f}\n"
            f"  残差标准差 = {residual_std:.6e}\n"
            f"  中心差分法加速度: 均值 = {acc_mean:.6f}, 标准差 = {acc_std:.6f}\n"
            f"  加速度是否近似常数: {'是' if acc_std < 0.1 * max(abs(acc_mean), 1e-6) else '否'}\n"
            f"  图像已保存: {fig_path}"
        )
        observations.append(obs)

    return {
        "observation": "\n".join(observations),
        "derived_series": derived_series_list,
        "figures": figures_list,
        "metrics": metrics
    }
