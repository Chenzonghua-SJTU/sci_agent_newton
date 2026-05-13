import os
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    parameters = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 获取实验ID列表
    eids = parameters.get("experiment_ids", list(experiments.keys()))
    # 检查所有请求的实验是否存在
    for eid in eids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload.")
        exp = experiments[eid]
        if "a_sg" not in exp["series"] or "v_sg" not in exp["series"]:
            raise ValueError(f"Experiment {eid} missing required series: a_sg or v_sg.")

    # 收集所有恒定外力实验 (exp_02, exp_03, exp_04, exp_05) 的 drag 和 v_sg
    def get_drag_and_v(exp):
        a_sg = np.array(exp["series"]["a_sg"])
        v_sg = np.array(exp["series"]["v_sg"])
        # 获取外力 F_ext：优先 config["F_ext"]，其次 config["constant_force"]
        config = exp.get("config", {})
        if "F_ext" in config:
            F_ext = config["F_ext"]
        elif "constant_force" in config:
            F_ext = config["constant_force"]
        else:
            # 若都无，尝试从实验描述中推理（备选）
            raise ValueError(f"Experiment {eid} missing F_ext in config.")
        # 如果已有 drag 序列，直接使用；否则计算
        if "drag" in exp["series"]:
            drag = np.array(exp["series"]["drag"])
        else:
            drag = F_ext - a_sg
        return drag, v_sg

    const_friction_eids = ["exp_02", "exp_03", "exp_04", "exp_05"]
    # 验证所有恒定外力实验都有数据
    for eid in const_friction_eids:
        if eid not in experiments:
            raise ValueError(f"Required constant force experiment {eid} missing.")
    # 收集数据
    all_drag = []
    all_v = []
    exp_labels = []
    exp_data = {}
    for eid in const_friction_eids:
        drag, v = get_drag_and_v(experiments[eid])
        all_drag.extend(drag.tolist())
        all_v.extend(v.tolist())
        exp_labels.extend([eid] * len(drag))
        exp_data[eid] = {"drag": drag, "v": v}

    all_drag = np.array(all_drag)
    all_v = np.array(all_v)
    exp_labels = np.array(exp_labels)

    # 拟合 drag = k * v^n
    def power_law(v, k, n):
        return k * v ** n

    # 对全数据拟合
    # 为稳定拟合，添加小的正则化，排除 v=0 的点
    mask = all_v > 0
    if mask.sum() < 3:
        raise ValueError("Not enough positive v points for power law fit.")
    v_pos = all_v[mask]
    drag_pos = all_drag[mask]

    try:
        popt, pcov = curve_fit(power_law, v_pos, drag_pos, p0=[1.0, 1.0], maxfev=10000)
        k_fit, n_fit = popt
        # 计算预测值
        pred_drag = power_law(v_pos, k_fit, n_fit)
        # 计算 R² (用正速度部分)
        ss_res = np.sum((drag_pos - pred_drag) ** 2)
        ss_tot = np.sum((drag_pos - np.mean(drag_pos)) ** 2)
        r2_power = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        # 残差标准差
        resid_std_power = np.std(drag_pos - pred_drag)
    except Exception as e:
        k_fit = np.nan
        n_fit = np.nan
        r2_power = np.nan
        resid_std_power = np.nan
        pred_drag = None

    # 计算 drag/v 统计（每个实验和全局）
    drag_over_v_global = all_drag / (all_v + 1e-12)  # 避免除以0
    drag_over_v_mean_global = np.mean(drag_over_v_global)
    drag_over_v_std_global = np.std(drag_over_v_global)

    per_exp_stats = {}
    for eid in const_friction_eids:
        d = exp_data[eid]["drag"]
        v = exp_data[eid]["v"]
        ratio = d / (v + 1e-12)
        per_exp_stats[eid] = {
            "mean_ratio": float(np.mean(ratio)),
            "std_ratio": float(np.std(ratio))
        }

    # 检查 exp_07 的 drag 是否为 0
    exp07 = experiments.get("exp_07")
    drag07_msg = ""
    if exp07 is not None:
        # 尝试获取 a_sg
        if "a_sg" in exp07["series"]:
            a_sg_07 = np.array(exp07["series"]["a_sg"])
            # 自由场实际外力为 0
            drag_07 = 0.0 - a_sg_07  # F_actual = 0
            drag_mean_07 = np.mean(drag_07)
            drag_std_07 = np.std(drag_07)
            drag07_msg = f"exp_07 drag 均值 = {drag_mean_07:.6f}，标准差 = {drag_std_07:.6f}"
            # 同时检查是否接近 0
            if abs(drag_mean_07) < 1e-6:
                drag07_msg += "，近似为 0。"
            else:
                drag07_msg += "，不接近 0。"
        else:
            drag07_msg = "exp_07 缺少 a_sg 序列，无法计算 drag。"

    # 绘制散点图
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'exp_02': 'blue', 'exp_03': 'green', 'exp_04': 'orange', 'exp_05': 'red'}
    for eid in const_friction_eids:
        d = exp_data[eid]["drag"]
        v = exp_data[eid]["v"]
        ax.scatter(v, d, c=colors[eid], label=eid, s=10, alpha=0.7)
    # 叠加全局拟合曲线（如成功）
    if not np.isnan(k_fit):
        v_sorted = np.sort(all_v)
        drag_fit = power_law(v_sorted, k_fit, n_fit)
        ax.plot(v_sorted, drag_fit, 'k-', label=f"drag={k_fit:.4f}*v^{n_fit:.4f}", linewidth=2)
    ax.set_xlabel("v_sg")
    ax.set_ylabel("drag (F_ext - a_sg)")
    ax.set_title("Drag vs Velocity (Constant Force Experiments)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_path = os.path.join(output_dir, "drag_vs_v_power_law.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 组装 observation
    obs_lines = []
    obs_lines.append("对恒定外力实验 exp_02-05 收集 drag 与 v_sg 数据，合并后拟合 power law drag = k * v^n。")
    if not np.isnan(k_fit):
        obs_lines.append(f"拟合结果: k = {k_fit:.6f}, n = {n_fit:.6f}, R² = {r2_power:.4f}, 残差标准差 = {resid_std_power:.6f}")
    else:
        obs_lines.append("Power law 拟合失败。")
    obs_lines.append("各实验 drag/v 统计:")
    for eid, stats in per_exp_stats.items():
        obs_lines.append(f"  {eid}: 均值={stats['mean_ratio']:.4f}, 标准差={stats['std_ratio']:.4f}")
    obs_lines.append(f"全局 drag/v 均值={drag_over_v_mean_global:.4f}, 标准差={drag_over_v_std_global:.4f}")
    obs_lines.append(drag07_msg)
    observation = "\n".join(obs_lines)

    # 组装 metrics
    metrics = {
        "k_power_law": k_fit if not np.isnan(k_fit) else None,
        "n_power_law": n_fit if not np.isnan(n_fit) else None,
        "r2_power_law": r2_power if not np.isnan(r2_power) else None,
        "residual_std_power_law": resid_std_power if not np.isnan(resid_std_power) else None,
        "drag_over_v_mean_global": drag_over_v_mean_global,
        "drag_over_v_std_global": drag_over_v_std_global,
    }
    for eid, stats in per_exp_stats.items():
        metrics[f"{eid}_drag_over_v_mean"] = stats["mean_ratio"]
        metrics[f"{eid}_drag_over_v_std"] = stats["std_ratio"]

    # 派生序列：无新序列，drag 已存在
    derived_series = []

    figures = [fig_path]

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics,
    }
