import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload.get("output_dir", ".")

    # 确定要处理的实验
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        exp_id = params.get("experiment_id")
        if exp_id:
            exp_ids = [exp_id]
        else:
            exp_ids = list(experiments.keys())

    # 只保留在 experiments 中存在的实验
    exp_ids = [eid for eid in exp_ids if eid in experiments]
    if not exp_ids:
        raise ValueError("No valid experiments found.")

    window_length = 11
    polyorder = 3
    dt = None

    derived_series_list = []
    figures = []
    metrics = {}
    observations_parts = []
    a_means = {}
    quad_a_vals = {}
    force_vals = {}

    for eid in exp_ids:
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        t = np.array(series["t"])
        q = np.array(series["q"])
        if dt is None and len(t) > 1:
            dt = t[1] - t[0]
        if dt is None:
            dt = 0.1

        # 用 Savitzky-Golay 滤波同时估计平滑位置、速度、加速度
        q_smooth = savgol_filter(q, window_length=window_length, polyorder=polyorder, deriv=0)
        v_smooth = savgol_filter(q, window_length=window_length, polyorder=polyorder, deriv=1, delta=dt)
        a_smooth = savgol_filter(q, window_length=window_length, polyorder=polyorder, deriv=2, delta=dt)

        a_mean = float(np.mean(a_smooth))
        a_std = float(np.std(a_smooth))
        a_means[eid] = a_mean

        # 获取外力
        F_ext = config.get("constant_force")
        if F_ext is None:
            F_ext = config.get("F_ext")
        if F_ext is None:
            F_ext = 0.0
        force_vals[eid] = float(F_ext)

        # q_smooth vs t^2 线性拟合
        t2 = t ** 2
        coeffs_linear = np.polyfit(t2, q_smooth, 1)  # [k, b]
        k_linear = float(coeffs_linear[0])
        b_linear = float(coeffs_linear[1])
        q_pred_linear = np.polyval(coeffs_linear, t2)
        mse_linear = float(np.mean((q_smooth - q_pred_linear) ** 2))

        # 二次多项式拟合 q_smooth vs t
        coeffs_quad = np.polyfit(t, q_smooth, 2)  # [a, b, c]
        a_quad, b_quad, c_quad = [float(v) for v in coeffs_quad]
        q_pred_quad = np.polyval(coeffs_quad, t)
        mse_quad = float(np.mean((q_smooth - q_pred_quad) ** 2))
        quad_a_vals[eid] = a_quad

        # 记录 metrics
        metrics[f"{eid}_a_mean"] = a_mean
        metrics[f"{eid}_a_std"] = a_std
        metrics[f"{eid}_linear_k"] = k_linear
        metrics[f"{eid}_linear_b"] = b_linear
        metrics[f"{eid}_linear_mse"] = mse_linear
        metrics[f"{eid}_quad_a"] = a_quad
        metrics[f"{eid}_quad_b"] = b_quad
        metrics[f"{eid}_quad_c"] = c_quad
        metrics[f"{eid}_quad_mse"] = mse_quad
        metrics[f"{eid}_F_ext"] = F_ext

        # 添加派生序列
        derived_series_list.append({
            "experiment_id": eid,
            "name": "q_smooth",
            "values": q_smooth.tolist(),
            "source_name": f"Savitzky-Golay filter (window={window_length}, polyorder={polyorder}) applied to q",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"Smooth position from SG filter (w={window_length}, p={polyorder})"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": "v_smooth",
            "values": v_smooth.tolist(),
            "source_name": f"First derivative via Savitzky-Golay filter (window={window_length}, polyorder={polyorder})",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"Velocity from SG derivative (w={window_length}, p={polyorder})"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": "a_smooth",
            "values": a_smooth.tolist(),
            "source_name": f"Second derivative via Savitzky-Golay filter (window={window_length}, polyorder={polyorder})",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"Acceleration from SG second derivative (w={window_length}, p={polyorder})"
        })

        # 观察文本
        observations_parts.append(
            f"实验 {eid} (F_ext={F_ext}): "
            f"a_smooth 均值={a_mean:.6f}, 标准差={a_std:.6f}; "
            f"q vs t^2 线性拟合 k={k_linear:.6f}, MSE={mse_linear:.6e}; "
            f"二次拟合系数 a={a_quad:.6f}, b={b_quad:.6f}, c={c_quad:.6f}, MSE={mse_quad:.6e}."
        )

        # 绘图：每个实验的运动学图和拟合图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        # 位置
        axes[0, 0].plot(t, q, 'b-', alpha=0.3, label='q raw')
        axes[0, 0].plot(t, q_smooth, 'r-', label='q_smooth')
        axes[0, 0].set_xlabel('t')
        axes[0, 0].set_ylabel('q')
        axes[0, 0].legend()
        axes[0, 0].set_title(f'{eid}: Position')

        # 速度
        axes[0, 1].plot(t, v_smooth, 'g-', label='v_smooth')
        axes[0, 1].set_xlabel('t')
        axes[0, 1].set_ylabel('v')
        axes[0, 1].legend()
        axes[0, 1].set_title('Velocity')

        # 加速度
        axes[1, 0].plot(t, a_smooth, 'm-', label='a_smooth')
        axes[1, 0].axhline(y=a_mean, color='k', linestyle='--', alpha=0.5, label=f'mean={a_mean:.3f}')
        axes[1, 0].fill_between(t, a_mean - a_std, a_mean + a_std, alpha=0.2, color='gray')
        axes[1, 0].set_xlabel('t')
        axes[1, 0].set_ylabel('a')
        axes[1, 0].legend()
        axes[1, 0].set_title('Acceleration')

        # 拟合比较
        axes[1, 1].plot(t, q_smooth, 'b-', label='q_smooth')
        axes[1, 1].plot(t, q_pred_linear, 'r--', label=f'linear vs t^2: k={k_linear:.4f}')
        axes[1, 1].plot(t, q_pred_quad, 'g:', label=f'quadratic: a={a_quad:.4f}')
        axes[1, 1].set_xlabel('t')
        axes[1, 1].set_ylabel('q')
        axes[1, 1].legend()
        axes[1, 1].set_title('Fits')

        plt.tight_layout()
        figname = f"{eid}_kinematics_fits_w{window_length}_p{polyorder}.png"
        figpath = os.path.join(output_dir, figname)
        plt.savefig(figpath, dpi=150)
        plt.close(fig)
        figures.append(figpath)

    # 两实验比较（仅当两个都处理了）
    if "exp_04" in exp_ids and "exp_05" in exp_ids:
        F04 = force_vals["exp_04"]
        F05 = force_vals["exp_05"]
        ratio_a = a_means["exp_05"] / a_means["exp_04"] if a_means["exp_04"] != 0 else float('nan')
        expected_ratio = F05 / F04 if F04 != 0 else float('nan')
        ratio_quad_a = quad_a_vals["exp_05"] / quad_a_vals["exp_04"] if quad_a_vals["exp_04"] != 0 else float('nan')
        observations_parts.append(
            f"比较: exp_05(F={F05}) vs exp_04(F={F04}): "
            f"加速度均值比值 = {ratio_a:.4f} (期望力比值 = {expected_ratio:.2f}); "
            f"二次项系数比值 = {ratio_quad_a:.4f}."
        )
        metrics["a_mean_ratio_exp05_exp04"] = ratio_a
        metrics["expected_F_ratio"] = expected_ratio
        metrics["quad_a_ratio_exp05_exp04"] = ratio_quad_a

        # 制两实验加速度对比图
        fig2, ax = plt.subplots(figsize=(8, 5))
        for eid in ["exp_04", "exp_05"]:
            exp = experiments[eid]
            t = np.array(exp["series"]["t"])
            q = np.array(exp["series"]["q"])
            # 由于同一个函数内，可以直接重新算（开销小）
            a_s = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)
            ax.plot(t, a_s, label=f"{eid} (F={force_vals[eid]})")
        ax.set_xlabel('t')
        ax.set_ylabel('a_smooth')
        ax.legend()
        ax.set_title('Acceleration comparison exp_04 vs exp_05')
        figname2 = "exp04_exp05_acceleration_comparison.png"
        figpath2 = os.path.join(output_dir, figname2)
        plt.savefig(figpath2, dpi=150)
        plt.close(fig2)
        figures.append(figpath2)

    observation = "\n".join(observations_parts)

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics
    }
