import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiment_ids = params.get("experiment_ids", [])
    analysis_goal = params.get("analysis_goal", "")
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 只处理指定的实验
    if not experiment_ids:
        experiment_ids = list(experiments.keys())

    # 收集每个实验的数据
    plot_data = []   # 用于绘图：每个元素是 (v_sg, drag, label, color)
    metrics = {}
    derived_series_list = []
    figures = []

    # 颜色映射
    color_map = {'exp_02': 'blue', 'exp_03': 'green', 'exp_04': 'orange', 'exp_05': 'red'}

    # 存储每个实验的拟合结果用于合并拟合（如果需要）
    all_v = []
    all_drag = []
    exp_labels = []

    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload.")
        exp = experiments[eid]
        config = exp.get("config", {})
        series = exp.get("series", {})
        available = exp.get("available_series", [])

        # 确保 a_sg, v_sg 存在
        if "a_sg" not in series or "v_sg" not in series:
            raise ValueError(f"Experiment {eid} missing a_sg or v_sg series.")
        a_sg = np.array(series["a_sg"])
        v_sg = np.array(series["v_sg"])
        t = np.array(series["t"])

        # 读取外力 F_ext
        force_field_type = config.get("force_field_type", "")
        if force_field_type == "free":
            F_ext = 0.0
        else:
            F_ext = config.get("constant_force", None)
            if F_ext is None:
                raise ValueError(f"Experiment {eid}: constant_force not found in config.")
            F_ext = float(F_ext)

        # 计算 drag = F_ext - a_sg
        drag = F_ext - a_sg

        # 返回 drag 序列（覆盖已有）
        derived_series_list.append({
            "experiment_id": eid,
            "name": "drag",
            "values": drag.tolist(),
            "source_name": f"drag = {F_ext} - a_sg",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"Drag force calculated as F_ext - a_sg for experiment {eid}"
        })

        # 收集数据用于合并拟合
        all_v.extend(v_sg.tolist())
        all_drag.extend(drag.tolist())
        exp_labels.extend([eid]*len(v_sg))

        # 拟合：线性模型 drag = c0 + c1*v_sg
        coeffs_linear = np.polyfit(v_sg, drag, 1)   # coeffs: [c1, c0]
        c1_lin, c0_lin = coeffs_linear[0], coeffs_linear[1]
        pred_linear = np.polyval(coeffs_linear, v_sg)
        resid_linear = drag - pred_linear
        ss_res_linear = np.sum(resid_linear**2)
        ss_tot_linear = np.sum((drag - np.mean(drag))**2)
        r2_linear = 1 - ss_res_linear / ss_tot_linear if ss_tot_linear > 0 else 0.0
        resid_std_linear = np.std(resid_linear, ddof=2)  # 有2个参数

        # 返回线性残差序列
        derived_series_list.append({
            "experiment_id": eid,
            "name": "linear_residual",
            "values": resid_linear.tolist(),
            "source_name": f"drag - (c0 + c1*v_sg) from linear fit",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"Residual of linear drag vs v_sg fit for {eid}"
        })

        # 二次模型 drag = c0 + c1*v_sg + c2*v_sg^2
        coeffs_quad = np.polyfit(v_sg, drag, 2)   # [c2, c1, c0]
        c2_q, c1_q, c0_q = coeffs_quad[0], coeffs_quad[1], coeffs_quad[2]
        pred_quad = np.polyval(coeffs_quad, v_sg)
        resid_quad = drag - pred_quad
        ss_res_quad = np.sum(resid_quad**2)
        ss_tot_quad = np.sum((drag - np.mean(drag))**2)
        r2_quad = 1 - ss_res_quad / ss_tot_quad if ss_tot_quad > 0 else 0.0
        resid_std_quad = np.std(resid_quad, ddof=3)  # 3个参数

        # 返回二次残差序列
        derived_series_list.append({
            "experiment_id": eid,
            "name": "quad_residual",
            "values": resid_quad.tolist(),
            "source_name": f"drag - (c0 + c1*v_sg + c2*v_sg^2) from quadratic fit",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"Residual of quadratic drag vs v_sg fit for {eid}"
        })

        # 检查 drag/v_sg 是否近似常数
        v_sg_safe = np.where(np.abs(v_sg) < 1e-12, np.nan, v_sg)  # 避免除零
        drag_over_v = drag / v_sg_safe
        valid_mask = ~np.isnan(drag_over_v)
        if np.sum(valid_mask) > 0:
            mean_dov = np.mean(drag_over_v[valid_mask])
            std_dov = np.std(drag_over_v[valid_mask])
            dov_note = f"drag/v_sg: mean={mean_dov:.4f}, std={std_dov:.4f}"
        else:
            mean_dov = np.nan
            std_dov = np.nan
            dov_note = "drag/v_sg: all v_sg near zero, cannot compute"

        # 检查 drag 与 v_sg^2 的线性关系
        v2 = v_sg**2
        coeffs_v2 = np.polyfit(v2, drag, 1)   # drag = k*v^2 + c
        k_v2, c_v2 = coeffs_v2[0], coeffs_v2[1]
        pred_v2 = np.polyval(coeffs_v2, v2)
        resid_v2 = drag - pred_v2
        ss_res_v2 = np.sum(resid_v2**2)
        ss_tot_v2 = np.sum((drag - np.mean(drag))**2)
        r2_v2 = 1 - ss_res_v2 / ss_tot_v2 if ss_tot_v2 > 0 else 0.0
        resid_std_v2 = np.std(resid_v2, ddof=2)

        # 记录指标
        prefix = eid
        metrics[f"{prefix}_linear_c0"] = c0_lin
        metrics[f"{prefix}_linear_c1"] = c1_lin
        metrics[f"{prefix}_linear_R2"] = r2_linear
        metrics[f"{prefix}_linear_resid_std"] = resid_std_linear
        metrics[f"{prefix}_quad_c0"] = c0_q
        metrics[f"{prefix}_quad_c1"] = c1_q
        metrics[f"{prefix}_quad_c2"] = c2_q
        metrics[f"{prefix}_quad_R2"] = r2_quad
        metrics[f"{prefix}_quad_resid_std"] = resid_std_quad
        metrics[f"{prefix}_drag_over_v_mean"] = mean_dov
        metrics[f"{prefix}_drag_over_v_std"] = std_dov
        metrics[f"{prefix}_drag_vs_v2_k"] = k_v2
        metrics[f"{prefix}_drag_vs_v2_c"] = c_v2
        metrics[f"{prefix}_drag_vs_v2_R2"] = r2_v2
        metrics[f"{prefix}_drag_vs_v2_resid_std"] = resid_std_v2

        # 准备绘图数据
        color = color_map.get(eid, 'gray')
        plot_data.append((v_sg, drag, eid, color))

        # 构建 observation 文本（每个实验的统计）
        # 将在后面汇总

    # 输出每个实验的统计（在observation中）
    obs_lines = []
    obs_lines.append("对实验 {} 计算 drag = F_ext - a_sg，并拟合 drag vs v_sg。".format(experiment_ids))
    obs_lines.append("")
    for eid in experiment_ids:
        if eid not in metrics:
            continue
        prefix = eid
        c0_lin = metrics[f"{prefix}_linear_c0"]
        c1_lin = metrics[f"{prefix}_linear_c1"]
        r2_lin = metrics[f"{prefix}_linear_R2"]
        rs_lin = metrics[f"{prefix}_linear_resid_std"]
        c0_q = metrics[f"{prefix}_quad_c0"]
        c1_q = metrics[f"{prefix}_quad_c1"]
        c2_q = metrics[f"{prefix}_quad_c2"]
        r2_q = metrics[f"{prefix}_quad_R2"]
        rs_q = metrics[f"{prefix}_quad_resid_std"]
        dov_mean = metrics[f"{prefix}_drag_over_v_mean"]
        dov_std = metrics[f"{prefix}_drag_over_v_std"]
        k_v2 = metrics[f"{prefix}_drag_vs_v2_k"]
        c_v2 = metrics[f"{prefix}_drag_vs_v2_c"]
        r2_v2 = metrics[f"{prefix}_drag_vs_v2_R2"]
        rs_v2 = metrics[f"{prefix}_drag_vs_v2_resid_std"]
        obs_lines.append(f"实验 {eid}:")
        obs_lines.append(f"  线性模型 drag = {c0_lin:.4f} + {c1_lin:.4f}*v_sg, R²={r2_lin:.4f}, 残差标准差={rs_lin:.4f}")
        obs_lines.append(f"  二次模型 drag = {c0_q:.4f} + {c1_q:.4f}*v_sg + {c2_q:.4f}*v_sg², R²={r2_q:.4f}, 残差标准差={rs_q:.4f}")
        obs_lines.append(f"  drag/v_sg: 均值={dov_mean:.4f}, 标准差={dov_std:.4f}")
        obs_lines.append(f"  drag vs v_sg²线性拟合: drag = {c_v2:.4f} + {k_v2:.4f}*v_sg², R²={r2_v2:.4f}, 残差标准差={rs_v2:.4f}")
        obs_lines.append("")

    # 合并所有实验的拟合（整体线性/二次）
    if len(all_v) > 0:
        all_v_arr = np.array(all_v)
        all_drag_arr = np.array(all_drag)
        # 全局线性
        coeffs_global_lin = np.polyfit(all_v_arr, all_drag_arr, 1)
        c1_gl, c0_gl = coeffs_global_lin[0], coeffs_global_lin[1]
        pred_gl_lin = np.polyval(coeffs_global_lin, all_v_arr)
        resid_gl_lin = all_drag_arr - pred_gl_lin
        ss_res_gl_lin = np.sum(resid_gl_lin**2)
        ss_tot_gl_lin = np.sum((all_drag_arr - np.mean(all_drag_arr))**2)
        r2_gl_lin = 1 - ss_res_gl_lin / ss_tot_gl_lin if ss_tot_gl_lin > 0 else 0.0
        resid_std_gl_lin = np.std(resid_gl_lin, ddof=2)
        # 全局二次
        coeffs_global_quad = np.polyfit(all_v_arr, all_drag_arr, 2)
        c2_gl, c1_gl, c0_gl = coeffs_global_quad[0], coeffs_global_quad[1], coeffs_global_quad[2]
        pred_gl_quad = np.polyval(coeffs_global_quad, all_v_arr)
        resid_gl_quad = all_drag_arr - pred_gl_quad
        ss_res_gl_quad = np.sum(resid_gl_quad**2)
        ss_tot_gl_quad = np.sum((all_drag_arr - np.mean(all_drag_arr))**2)
        r2_gl_quad = 1 - ss_res_gl_quad / ss_tot_gl_quad if ss_tot_gl_quad > 0 else 0.0
        resid_std_gl_quad = np.std(resid_gl_quad, ddof=3)
        obs_lines.append("合并所有数据点的全局拟合:")
        obs_lines.append(f"  线性: drag={c0_gl:.4f}+{c1_gl:.4f}*v_sg, R²={r2_gl_lin:.4f}, 残差标准差={resid_std_gl_lin:.4f}")
        obs_lines.append(f"  二次: drag={c0_gl:.4f}+{c1_gl:.4f}*v_sg+{c2_gl:.4f}*v_sg², R²={r2_gl_quad:.4f}, 残差标准差={resid_std_gl_quad:.4f}")
        # 记录全局指标
        metrics["global_linear_c0"] = c0_gl
        metrics["global_linear_c1"] = c1_gl
        metrics["global_linear_R2"] = r2_gl_lin
        metrics["global_linear_resid_std"] = resid_std_gl_lin
        metrics["global_quad_c0"] = c0_gl
        metrics["global_quad_c1"] = c1_gl
        metrics["global_quad_c2"] = c2_gl
        metrics["global_quad_R2"] = r2_gl_quad
        metrics["global_quad_resid_std"] = resid_std_gl_quad
    else:
        obs_lines.append("无有效数据用于全局拟合。")

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    # 为每个实验绘制散点
    for v, d, label, color in plot_data:
        ax.scatter(v, d, label=label, color=color, s=20, alpha=0.7)
    # 绘制每个实验的拟合曲线（使用该实验的拟合系数）
    # 为了图清晰，只绘制全局拟合曲线，或者每个实验的？决定绘制全局拟合曲线
    v_grid = np.linspace(min(all_v), max(all_v), 300) if len(all_v) > 0 else np.array([])
    if len(all_v) > 0:
        # 全局线性
        ax.plot(v_grid, np.polyval([c1_gl, c0_gl], v_grid), 'k--', label=f'Global linear (R²={r2_gl_lin:.3f})', linewidth=1)
        # 全局二次
        ax.plot(v_grid, np.polyval([c2_gl, c1_gl, c0_gl], v_grid), 'k-', label=f'Global quadratic (R²={r2_gl_quad:.3f})', linewidth=1)
    ax.set_xlabel('v_sg')
    ax.set_ylabel('drag')
    ax.set_title('Drag vs v_sg for each experiment with global fits')
    ax.legend()
    fig_path = os.path.join(output_dir, "drag_vs_v_sg_scatter_with_fits.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    figures.append(fig_path)

    # 残差图：每个实验的线性残差和二次残差
    fig2, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for eid in experiment_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        series = exp.get("series", {})
        if "linear_residual" in series:
            resid_lin = np.array(series["linear_residual"])
        else:
            # 从返回的derived_series中取？但还没返回给payload，所以这里不能直接使用。我们直接使用本地计算的residual（local变量）
            # 但是我们已经计算过，需要存储。简单起见，重新计算一遍
            a_sg = np.array(series["a_sg"])
            v_sg = np.array(series["v_sg"])
            F_ext = float(config.get("constant_force", 0))
            drag = F_ext - a_sg
            coeffs_linear = np.polyfit(v_sg, drag, 1)
            resid_lin = drag - np.polyval(coeffs_linear, v_sg)
            coeffs_quad = np.polyfit(v_sg, drag, 2)
            resid_quad = drag - np.polyval(coeffs_quad, v_sg)
        color = color_map.get(eid, 'gray')
        axes[0].scatter(v_sg, resid_lin, label=eid, color=color, s=10, alpha=0.7)
        axes[1].scatter(v_sg, resid_quad, label=eid, color=color, s=10, alpha=0.7)
    axes[0].axhline(0, color='gray', linestyle='--')
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[0].set_ylabel('Linear residual')
    axes[1].set_xlabel('v_sg')
    axes[1].set_ylabel('Quadratic residual')
    axes[0].legend()
    axes[1].legend()
    axes[0].set_title('Residuals of drag vs v_sg fits')
    resid_path = os.path.join(output_dir, "drag_vs_v_sg_residuals.png")
    plt.tight_layout()
    plt.savefig(resid_path, dpi=150)
    plt.close()
    figures.append(resid_path)

    observation = "\n".join(obs_lines)

    result = {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics
    }
    return result
