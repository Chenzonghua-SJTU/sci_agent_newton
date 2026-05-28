import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from copy import deepcopy
import os

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 提取实验 ID 列表
    experiment_ids = params.get("experiment_ids", list(experiments.keys()))
    if not experiment_ids:
        experiment_ids = list(experiments.keys())

    # 结果容器
    results = []
    k_per_exp = []
    f_ext_per_exp = []
    r2_per_exp = []

    # 定义饱和模型 drag/F_ext = 1 - exp(-k * v)
    def saturation_model(v, k):
        return 1 - np.exp(-k * v)

    # 定义替代模型 drag/F_ext = v / (v + v0)
    def alt_model(v, v0):
        return v / (v + v0)

    # 逐个实验处理
    for exp_id in experiment_ids:
        exp_data = experiments.get(exp_id)
        if exp_data is None:
            continue
        config = exp_data.get("config", {})
        f_ext = config.get("F_ext", config.get("constant_force", None))
        if f_ext is None or f_ext == 0:
            # 非恒外力实验跳过
            continue
        series = exp_data.get("series", {})
        # 检查必要的序列
        v_est = np.array(series.get("v_est", series.get("velocity", None)))
        drag = np.array(series.get("drag", None))
        if v_est is None or drag is None:
            # 尝试通过 a_est 计算 drag = F_ext - a_est
            a_est = series.get("a_est", None)
            if a_est is not None and f_ext is not None:
                drag = f_ext - np.array(a_est)
                v_est = np.array(series.get("v_est", series.get("velocity", None)))
            else:
                continue  # 数据不足以计算
        # 确保数据类型
        v_est = np.asarray(v_est, dtype=float)
        drag = np.asarray(drag, dtype=float)
        if len(v_est) == 0 or len(drag) == 0:
            continue
        # 计算 ratio_drag_over_F
        ratio = drag / f_ext
        # 剔除无效值
        mask = np.isfinite(v_est) & np.isfinite(ratio) & (v_est > 0)
        v_clean = v_est[mask]
        ratio_clean = ratio[mask]
        if len(v_clean) < 5:
            continue
        # 拟合饱和模型
        try:
            popt_sat, pcov_sat = curve_fit(saturation_model, v_clean, ratio_clean, p0=[0.5], maxfev=5000)
            k_val = popt_sat[0]
            # 计算 R²
            y_pred_sat = saturation_model(v_clean, k_val)
            ss_res = np.sum((ratio_clean - y_pred_sat) ** 2)
            ss_tot = np.sum((ratio_clean - np.mean(ratio_clean)) ** 2)
            r2_sat = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        except Exception:
            k_val = np.nan
            r2_sat = 0
        # 拟合替代模型 (drag/F_ext = v/(v+v0))
        try:
            popt_alt, pcov_alt = curve_fit(alt_model, v_clean, ratio_clean, p0=[1.0], maxfev=5000)
            v0_val = popt_alt[0]
            y_pred_alt = alt_model(v_clean, v0_val)
            ss_res_alt = np.sum((ratio_clean - y_pred_alt) ** 2)
            r2_alt = 1 - ss_res_alt / ss_tot if ss_tot > 0 else 0
        except Exception:
            v0_val = np.nan
            r2_alt = 0
        # 记录
        results.append({
            "exp_id": exp_id,
            "F_ext": f_ext,
            "k": k_val,
            "R2_sat": r2_sat,
            "v0": v0_val,
            "R2_alt": r2_alt,
            "v_clean": v_clean.tolist(),
            "ratio_clean": ratio_clean.tolist()
        })
        if np.isfinite(k_val):
            k_per_exp.append(k_val)
            f_ext_per_exp.append(f_ext)
            r2_per_exp.append(r2_sat)

    if len(k_per_exp) == 0:
        return {"observation": "无有效实验数据", "metrics": {}, "figures": []}
    # 计算 k 统计
    k_array = np.array(k_per_exp)
    k_mean = np.mean(k_array)
    k_std = np.std(k_array, ddof=1)
    k_rel_std = k_std / k_mean if k_mean != 0 else np.inf
    # k vs F_ext 线性回归
    f_array = np.array(f_ext_per_exp)
    if len(f_array) >= 2:
        A = np.vstack([f_array, np.ones_like(f_array)]).T
        coeff, residuals, rank, s = np.linalg.lstsq(A, k_array, rcond=None)
        slope_k = coeff[0]
        intercept_k = coeff[1]
        # R² of linear fit
        k_pred = slope_k * f_array + intercept_k
        ss_res_k = np.sum((k_array - k_pred) ** 2)
        ss_tot_k = np.sum((k_array - np.mean(k_array)) ** 2)
        r2_k_vs_F = 1 - ss_res_k / ss_tot_k if ss_tot_k > 0 else 0
    else:
        slope_k = 0
        intercept_k = np.mean(k_array)
        r2_k_vs_F = 0

    # 判断 k 是否接近常数（相对标准差 < 20% 认为变化不大）
    k_is_constant = k_rel_std < 0.2

    # 构建观察字符串
    obs_lines = []
    obs_lines.append(f"处理了 {len(results)} 个恒外力实验: {[r['exp_id'] for r in results]}")
    for r in results:
        obs_lines.append(
            f"  {r['exp_id']} (F_ext={r['F_ext']}): k={r['k']:.4f}, R²_sat={r['R2_sat']:.4f}, "
            f"v0={r['v0']:.4f}, R²_alt={r['R2_alt']:.4f}"
        )
    obs_lines.append(f"k 统计: 均值={k_mean:.4f}, 样本标准差={k_std:.4f}, 相对标准差={k_rel_std:.4f}")
    obs_lines.append(f"k 与 F_ext 线性回归: 斜率={slope_k:.4f}, 截距={intercept_k:.4f}, R²={r2_k_vs_F:.4f}")
    if k_is_constant:
        obs_lines.append(f"k 变化不大 (相对标准差 {k_rel_std:.2%} < 20%)，建议使用统一 k 均值 = {k_mean:.4f}")
    else:
        obs_lines.append(f"k 有明显趋势，尝试 v/(v+v0) 模型")
        v0_vals = [r['v0'] for r in results if np.isfinite(r['v0'])]
        if v0_vals:
            v0_mean = np.mean(v0_vals)
            v0_std = np.std(v0_vals, ddof=1)
            obs_lines.append( f"替代模型 v0 均值={v0_mean:.4f}, 标准差={v0_std:.4f}")

    # 绘图
    figures = []
    # 1. 每个实验的拟合对比图
    fig1, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, r in enumerate(results):
        ax = axes[i]
        v = np.array(r["v_clean"])
        ratio = np.array(r["ratio_clean"])
        ax.scatter(v, ratio, s=10, label="data", alpha=0.6)
        v_sorted = np.sort(v)
        if np.isfinite(r["k"]):
            ax.plot(v_sorted, saturation_model(v_sorted, r["k"]), 'r-', label=f"saturation k={r['k']:.3f}")
        if np.isfinite(r["v0"]):
            ax.plot(v_sorted, alt_model(v_sorted, r["v0"]), 'g--', label=f"v/(v+v0) v0={r['v0']:.3f}")
        ax.set_title(f"{r['exp_id']} F_ext={r['F_ext']}")
        ax.set_xlabel("v_est")
        ax.set_ylabel("drag/F_ext")
        ax.legend(fontsize=8)
    for j in range(len(results), len(axes)):
        axes[j].axis("off")
    fig1.tight_layout()
    fig1_path = os.path.join(output_dir, "per_exp_saturation_fit.png")
    fig1.savefig(fig1_path)
    plt.close(fig1)
    figures.append(fig1_path)

    # 2. k vs F_ext 图
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(f_ext_per_exp, k_per_exp, c='blue', label='k value')
    if len(f_array) >= 2:
        f_line = np.linspace(min(f_array), max(f_array), 100)
        k_line = slope_k * f_line + intercept_k
        ax2.plot(f_line, k_line, 'r--', label=f'linear fit slope={slope_k:.3f}')
    ax2.set_xlabel("F_ext")
    ax2.set_ylabel("k")
    ax2.set_title("k vs F_ext")
    ax2.legend()
    fig2_path = os.path.join(output_dir, "k_vs_F_ext.png")
    fig2.savefig(fig2_path)
    plt.close(fig2)
    figures.append(fig2_path)

    # 3. 所有实验散点 + 全局拟合（使用统一 k 均值或 k vs F 模型）
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for r in results:
        v = np.array(r["v_clean"])
        ratio = np.array(r["ratio_clean"])
        ax3.scatter(v, ratio, s=5, label=r['exp_id'], alpha=0.5)
    if k_is_constant:
        v_global = np.linspace(0, max([max(r['v_clean']) for r in results]), 200)
        ax3.plot(v_global, saturation_model(v_global, k_mean), 'k-', lw=2, label=f'global k={k_mean:.3f}')
    else:
        # 使用每个实验的 v0 均值画全局替代曲线
        v0_mean_val = np.mean([r['v0'] for r in results if np.isfinite(r['v0'])])
        v_global = np.linspace(0, max([max(r['v_clean']) for r in results]), 200)
        ax3.plot(v_global, alt_model(v_global, v0_mean_val), 'k--', lw=2, label=f'global v0={v0_mean_val:.3f}')
    ax3.set_xlabel("v_est")
    ax3.set_ylabel("drag/F_ext")
    ax3.set_title("All experiments with best model")
    ax3.legend()
    fig3_path = os.path.join(output_dir, "global_saturation_fit.png")
    fig3.savefig(fig3_path)
    plt.close(fig3)
    figures.append(fig3_path)

    # 构造 metrics
    metrics = {
        "k_mean": k_mean,
        "k_std": k_std,
        "k_rel_std": k_rel_std,
        "k_vs_F_slope": slope_k,
        "k_vs_F_intercept": intercept_k,
        "k_vs_F_R2": r2_k_vs_F,
        "k_is_constant": int(k_is_constant)
    }
    for r in results:
        metrics[f"{r['exp_id']}_k"] = r['k']
        metrics[f"{r['exp_id']}_R2_sat"] = r['R2_sat']
        metrics[f"{r['exp_id']}_v0"] = r['v0']
        metrics[f"{r['exp_id']}_R2_alt"] = r['R2_alt']
    # 派生序列：residual 计算（使用拟合最好的模型）
    derived_series = []
    if k_is_constant:
        best_model = lambda v: saturation_model(v, k_mean)
        best_name = f"saturation_k={k_mean:.4f}"
    else:
        v0_mean_val = np.mean([r['v0'] for r in results if np.isfinite(r['v0'])])
        best_model = lambda v: alt_model(v, v0_mean_val)
        best_name = f"v/(v+{v0_mean_val:.4f})"

    for r in results:
        exp_id = r['exp_id']
        v_clean = np.array(r['v_clean'])
        ratio_clean = np.array(r['ratio_clean'])
        # 需要返回与原始序列等长的残差，但只能返回原始长度（包括未使用的点）
        # 从原始数据中获取完整 v_est 和 ratio 序列 (未过滤的)
        exp_data = experiments.get(exp_id)
        if exp_data is None:
            continue
        series = exp_data.get("series", {})
        v_orig = np.array(series.get("v_est", series.get("velocity", [])))
        # 构建原始 drag/F_ext
        f_ext = exp_data.get("config", {}).get("F_ext", 0)
        if f_ext == 0:
            continue
        drag_orig = np.array(series.get("drag", None))
        if drag_orig is None:
            a_orig = series.get("a_est", None)
            if a_orig is not None:
                drag_orig = f_ext - np.array(a_orig)
            else:
                continue
        ratio_orig = drag_orig / f_ext
        residual_orig = ratio_orig - best_model(v_orig)
        # 只保留有限值
        mask = np.isfinite(v_orig) & np.isfinite(residual_orig)
        residual_values = np.full_like(v_orig, np.nan)
        residual_values[mask] = residual_orig[mask]
        derived_series.append({
            "experiment_id": exp_id,
            "name": "residual_saturation_v3",
            "values": residual_values.tolist(),
            "source_name": f"residual = drag/F_ext - {best_name}",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"Residual using best model from current analysis"
        })

    # 构造观察
    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
