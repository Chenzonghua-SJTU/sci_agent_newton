import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    exp_ids = params.get("experiment_ids", list(experiments.keys()))
    # 只保留在experiments中且属于constant外力场的实验（F_ext > 0）
    target_ids = []
    for eid in exp_ids:
        if eid not in experiments:
            continue
        cfg = experiments[eid].get("config", {})
        F = cfg.get("F_ext", cfg.get("constant_force", 0))
        if F <= 0:
            continue
        target_ids.append(eid)

    # 收集每个实验的 (v, ratio, F_ext)
    data_by_exp = {}
    for eid in target_ids:
        exp = experiments[eid]
        series = exp.get("series", {})
        available = exp.get("available_series", [])
        cfg = exp.get("config", {})
        F_ext = cfg.get("F_ext", cfg.get("constant_force", 0))

        # 优先使用已有的ratio_drag_over_F，否则计算
        if "ratio_drag_over_F" in available:
            ratio = np.array(series["ratio_drag_over_F"])
        else:
            drag = np.array(series.get("drag", []))
            if len(drag) == 0:
                continue
            ratio = drag / F_ext

        # 速度
        v_key = "v_est" if "v_est" in available else "velocity"
        if v_key not in available:
            continue
        v = np.array(series[v_key])

        # 只保留有效数据（v>0, ratio有效）
        mask = (v > 0) & np.isfinite(ratio)
        v = v[mask]
        ratio = ratio[mask]
        if len(v) < 5:
            continue

        data_by_exp[eid] = {
            "v": v,
            "ratio": ratio,
            "F_ext": F_ext,
            "label": f"{eid} (F={F_ext})"
        }

    if len(data_by_exp) == 0:
        raise ValueError("没有可用的恒外力实验数据")

    # 定义拟合函数
    def model1(v, k):
        return 1 - np.exp(-k * v)

    def model2(v, a, b):
        return a * v + b * np.sqrt(v)

    # 每个实验拟合 k
    per_exp_k = {}
    per_exp_r2 = {}
    for eid, d in data_by_exp.items():
        v = d["v"]
        ratio = d["ratio"]
        try:
            popt, pcov = curve_fit(model1, v, ratio, p0=[1.0], bounds=(0, np.inf))
            k = popt[0]
            pred = model1(v, k)
            ss_res = np.sum((ratio - pred) ** 2)
            ss_tot = np.sum((ratio - np.mean(ratio)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            per_exp_k[eid] = k
            per_exp_r2[eid] = r2
        except Exception:
            continue

    # 统计k
    k_values = np.array(list(per_exp_k.values()))
    if len(k_values) == 0:
        raise ValueError("没有成功拟合出k")
    k_mean = float(np.mean(k_values))
    k_std = float(np.std(k_values, ddof=1)) if len(k_values) > 1 else 0.0
    k_rel_std = k_std / k_mean if k_mean > 0 else np.nan

    # 全局拟合k_global
    all_v = np.concatenate([d["v"] for d in data_by_exp.values()])
    all_ratio = np.concatenate([d["ratio"] for d in data_by_exp.values()])
    try:
        popt_global, _ = curve_fit(model1, all_v, all_ratio, p0=[1.0], bounds=(0, np.inf))
        k_global = popt_global[0]
        pred_global = model1(all_v, k_global)
        ss_res = np.sum((all_ratio - pred_global) ** 2)
        ss_tot = np.sum((all_ratio - np.mean(all_ratio)) ** 2)
        r2_global = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    except Exception:
        k_global = None
        r2_global = None

    # 全局线性+平方根拟合
    X = np.column_stack([all_v, np.sqrt(all_v)])
    try:
        coeff, res, rank, sv = np.linalg.lstsq(X, all_ratio, rcond=None)
        a_lin, b_sqrt = coeff[0], coeff[1]
        pred_lin_sqrt = X @ coeff
        ss_res_ls = np.sum((all_ratio - pred_lin_sqrt) ** 2)
        ss_tot_ls = np.sum((all_ratio - np.mean(all_ratio)) ** 2)
        r2_lin_sqrt = 1 - ss_res_ls / ss_tot_ls if ss_tot_ls > 0 else 0
    except Exception:
        a_lin, b_sqrt, r2_lin_sqrt = None, None, None

    # 绘图1：所有实验散点 + 全局拟合曲线
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_by_exp)))
    for idx, (eid, d) in enumerate(data_by_exp.items()):
        ax1.scatter(d["v"], d["ratio"], s=10, color=colors[idx], label=d["label"], alpha=0.7)
    # 全局拟合曲线
    v_grid = np.linspace(0, max(all_v)*1.05, 200)
    if k_global is not None:
        ax1.plot(v_grid, model1(v_grid, k_global), 'k-', linewidth=2, label=f'global fit: 1-exp(-{k_global:.4f}v)')
    ax1.set_xlabel('v')
    ax1.set_ylabel('drag / F_ext')
    ax1.set_title('All experiments: drag/F_ext vs v with global fit')
    ax1.legend()
    ax1.grid(True)
    fig1_path = os.path.join(output_dir, 'all_exp_drag_over_F_vs_v_global_fit.png')
    fig1.savefig(fig1_path)
    plt.close(fig1)

    # 绘图2：每个实验单独拟合曲线 vs 全局拟合
    fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for idx, (eid, d) in enumerate(data_by_exp.items()):
        ax = axes[idx]
        v = d["v"]
        ratio = d["ratio"]
        ax.scatter(v, ratio, s=5, color='blue', alpha=0.5, label='data')
        # 单独拟合
        if eid in per_exp_k:
            k_local = per_exp_k[eid]
            ax.plot(v_grid, model1(v_grid, k_local), 'r-', linewidth=2, label=f'local k={k_local:.4f}')
        # 全局拟合
        if k_global is not None:
            ax.plot(v_grid, model1(v_grid, k_global), 'k--', linewidth=1.5, label=f'global k={k_global:.4f}')
        ax.set_title(d["label"])
        ax.set_xlabel('v')
        ax.set_ylabel('drag/F_ext')
        ax.legend(fontsize=8)
        ax.grid(True)
    # 隐藏多余子图
    for idx in range(len(data_by_exp), len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, 'per_exp_drag_over_F_fit_comparison.png')
    fig2.savefig(fig2_path)
    plt.close(fig2)

    # 构建 observation
    obs_lines = []
    obs_lines.append(f"对 {len(data_by_exp)} 个恒外力实验进行了 drag/F_ext = 1 - exp(-k*v) 拟合。")
    obs_lines.append("各实验 k 值：")
    for eid in sorted(per_exp_k.keys()):
        obs_lines.append(f"  {eid}: k={per_exp_k[eid]:.4f}, R²={per_exp_r2[eid]:.4f}")
    obs_lines.append(f"k 均值 = {k_mean:.4f}, 样本标准差 = {k_std:.4f}, 相对标准差 = {k_rel_std:.4f}")
    if k_global is not None:
        obs_lines.append(f"全局联合拟合 k_global = {k_global:.4f}, R² = {r2_global:.4f}")
    else:
        obs_lines.append("全局拟合失败。")
    if a_lin is not None:
        obs_lines.append(f"对比模型 drag/F_ext = a*v + b*sqrt(v): a={a_lin:.4f}, b={b_sqrt:.4f}, 全局 R² = {r2_lin_sqrt:.4f}")
    else:
        obs_lines.append("线性+平方根拟合失败。")
    observation = "\n".join(obs_lines)

    # 构建 metrics
    metrics = {}
    metrics["k_mean"] = k_mean
    metrics["k_std"] = k_std
    metrics["k_rel_std"] = k_rel_std
    if k_global is not None:
        metrics["k_global"] = k_global
        metrics["global_R2"] = r2_global
    for eid in per_exp_k:
        metrics[f"{eid}_k"] = per_exp_k[eid]
        metrics[f"{eid}_R2"] = per_exp_r2[eid]
    if a_lin is not None:
        metrics["lin_sqrt_a"] = a_lin
        metrics["lin_sqrt_b"] = b_sqrt
        metrics["lin_sqrt_R2"] = r2_lin_sqrt

    # derived_series 不需要新增
    derived_series = []

    figures = [fig1_path, fig2_path]

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
