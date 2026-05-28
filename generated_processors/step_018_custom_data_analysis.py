import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import scipy
from scipy import optimize
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    # -------------------------------------------------------------------------
    # 1. 提取参数与实验数据
    # -------------------------------------------------------------------------
    parameters = payload.get("parameters", {})
    analysis_goal = parameters.get("analysis_goal", "")
    exp_ids = parameters.get("experiment_ids", [])
    optional_series = parameters.get("optional_series", [])
    output_dir = Path(payload["output_dir"])
    experiments = payload.get("experiments", {})

    # 如果没有指定实验ID，处理所有
    if not exp_ids:
        exp_ids = list(experiments.keys())

    # 只保留恒外力实验 (force_field_type=constant, F_ext > 0)
    constant_exp_ids = []
    for eid in exp_ids:
        cfg = experiments[eid].get("config", {})
        if cfg.get("force_field_type") == "constant" and cfg.get("F_ext", 0) > 0:
            constant_exp_ids.append(eid)
    if not constant_exp_ids:
        return {"observation": "没有恒外力实验，跳过分析。", "derived_series": [], "figures": [], "metrics": {}}

    # -------------------------------------------------------------------------
    # 2. 辅助函数：从实验获取序列（支持先检查 existing series）
    # -------------------------------------------------------------------------
    def get_series(eid: str, name: str) -> np.ndarray:
        exp = experiments[eid]
        if name in exp.get("series", {}):
            return np.array(exp["series"][name])
        raise ValueError(f"实验 {eid} 中找不到序列 '{name}'")

    def get_t(eid: str) -> np.ndarray:
        return get_series(eid, "t")

    def get_F_ext(eid: str) -> float:
        return experiments[eid]["config"].get("F_ext", 0.0)

    # -------------------------------------------------------------------------
    # 3. 对每个恒外力实验，确认 v_sg, a_sg 存在，否则使用 estimated
    # -------------------------------------------------------------------------
    for eid in constant_exp_ids:
        exp = experiments[eid]
        available = exp.get("available_series", [])
        if "v_sg" not in available or "a_sg" not in available:
            # 尝试从 q 估计
            q = get_series(eid, "q")
            t = get_t(eid)
            dt = exp["config"].get("dt", t[1] - t[0])
            try:
                v = savgol_filter(q, window_length=11, polyorder=3, deriv=1, delta=dt)
                a = savgol_filter(q, window_length=11, polyorder=3, deriv=2, delta=dt)
            except Exception:
                v = np.gradient(q, dt)
                a = np.gradient(v, dt)
            # 临时添加到 series 中 (不修改 payload，仅本地使用)
            exp["series"]["v_sg"] = v.tolist()
            exp["series"]["a_sg"] = a.tolist()
            available.append("v_sg")
            available.append("a_sg")

    # -------------------------------------------------------------------------
    # 4. 执行分析
    # -------------------------------------------------------------------------
    derived_series_list = []
    figures = []
    metrics = {}

    # ---- 4.1 每个实验单独拟合 ----
    all_d_v = {}
    for eid in constant_exp_ids:
        v = get_series(eid, "v_sg")
        a = get_series(eid, "a_sg")
        t = get_t(eid)
        F_ext = get_F_ext(eid)
        d = a - F_ext  # 阻尼项 d = a_sg - F_ext

        # 保存 d 为新派生序列
        derived_series_list.append({
            "experiment_id": eid,
            "name": "d",
            "values": d.tolist(),
            "source_name": f"d = a_sg - F_ext (F_ext={F_ext})",
            "provenance": "generated data processor: step_n_custom_data_analysis",
            "description": "阻尼项"
        })
        all_d_v[eid] = (d, v)

        # 线性拟合 d = α + β * v
        A_lin = np.vstack([np.ones_like(v), v]).T
        coeff_lin, resid_lin, rank_lin, sv_lin = np.linalg.lstsq(A_lin, d, rcond=None)
        d_lin_pred = A_lin @ coeff_lin
        ss_res_lin = np.sum((d - d_lin_pred)**2)
        ss_tot_lin = np.sum((d - np.mean(d))**2)
        r2_lin = 1 - ss_res_lin / (ss_tot_lin + 1e-15)
        # 残差标准差
        resid_std_lin = np.std(d - d_lin_pred)
        # 置信区间 (95% t-based)
        n = len(v)
        mse_lin = resid_lin[0] / (n - 2) if len(resid_lin) > 0 else 0.0
        se_lin = np.sqrt(np.diag(np.linalg.inv(A_lin.T @ A_lin) * mse_lin)) if n>2 else [0,0]
        t_val = scipy.stats.t.ppf(0.975, n-2) if n>2 else 1.96
        ci_lin_alpha = (coeff_lin[0] - t_val*se_lin[0], coeff_lin[0] + t_val*se_lin[0])
        ci_lin_beta  = (coeff_lin[1] - t_val*se_lin[1], coeff_lin[1] + t_val*se_lin[1])

        # 二次拟合 d = α + γ * v^2
        A_quad = np.vstack([np.ones_like(v), v**2]).T
        coeff_quad, resid_quad, rank_quad, sv_quad = np.linalg.lstsq(A_quad, d, rcond=None)
        d_quad_pred = A_quad @ coeff_quad
        ss_res_quad = np.sum((d - d_quad_pred)**2)
        ss_tot_quad = np.sum((d - np.mean(d))**2)
        r2_quad = 1 - ss_res_quad / (ss_tot_quad + 1e-15)
        resid_std_quad = np.std(d - d_quad_pred)
        mse_quad = resid_quad[0] / (n - 2) if len(resid_quad)>0 else 0.0
        se_quad = np.sqrt(np.diag(np.linalg.inv(A_quad.T @ A_quad) * mse_quad)) if n>2 else [0,0]
        ci_quad_alpha = (coeff_quad[0] - t_val*se_quad[0], coeff_quad[0] + t_val*se_quad[0])
        ci_quad_gamma = (coeff_quad[1] - t_val*se_quad[1], coeff_quad[1] + t_val*se_quad[1])

        # 幂律拟合 d = -beta * v^gamma (beta>0, gamma>0)
        def power_law(v, beta, gamma):
            return -beta * (v ** gamma)

        # 初值
        try:
            popt, pcov = scipy.optimize.curve_fit(
                power_law, v, d,
                p0=[0.1, 1.0],
                bounds=([1e-10, 0.001], [np.inf, 3.0]),
                maxfev=5000
            )
            beta_power, gamma_power = popt
            d_power_pred = power_law(v, beta_power, gamma_power)
            ss_res_power = np.sum((d - d_power_pred)**2)
            r2_power = 1 - ss_res_power / (ss_tot_lin + 1e-15)
            resid_std_power = np.std(d - d_power_pred)
            # 置信区间
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else [0,0]
            ci_beta = (beta_power - t_val*perr[0], beta_power + t_val*perr[0])
            ci_gamma = (gamma_power - t_val*perr[1], gamma_power + t_val*perr[1])
        except Exception as e:
            beta_power, gamma_power = np.nan, np.nan
            r2_power = np.nan
            resid_std_power = np.nan
            ci_beta = (np.nan, np.nan)
            ci_gamma = (np.nan, np.nan)

        metrics[f"{eid}_lin_alpha"] = coeff_lin[0]
        metrics[f"{eid}_lin_alpha_ci_low"] = ci_lin_alpha[0]
        metrics[f"{eid}_lin_alpha_ci_high"] = ci_lin_alpha[1]
        metrics[f"{eid}_lin_beta"] = coeff_lin[1]
        metrics[f"{eid}_lin_beta_ci_low"] = ci_lin_beta[0]
        metrics[f"{eid}_lin_beta_ci_high"] = ci_lin_beta[1]
        metrics[f"{eid}_lin_R2"] = r2_lin
        metrics[f"{eid}_lin_resid_std"] = resid_std_lin

        metrics[f"{eid}_quad_alpha"] = coeff_quad[0]
        metrics[f"{eid}_quad_alpha_ci_low"] = ci_quad_alpha[0]
        metrics[f"{eid}_quad_alpha_ci_high"] = ci_quad_alpha[1]
        metrics[f"{eid}_quad_gamma"] = coeff_quad[1]
        metrics[f"{eid}_quad_gamma_ci_low"] = ci_quad_gamma[0]
        metrics[f"{eid}_quad_gamma_ci_high"] = ci_quad_gamma[1]
        metrics[f"{eid}_quad_R2"] = r2_quad
        metrics[f"{eid}_quad_resid_std"] = resid_std_quad

        metrics[f"{eid}_power_beta"] = beta_power
        metrics[f"{eid}_power_gamma"] = gamma_power
        metrics[f"{eid}_power_R2"] = r2_power
        metrics[f"{eid}_power_resid_std"] = resid_std_power
        metrics[f"{eid}_power_ci_beta_low"] = ci_beta[0]
        metrics[f"{eid}_power_ci_beta_high"] = ci_beta[1]
        metrics[f"{eid}_power_ci_gamma_low"] = ci_gamma[0]
        metrics[f"{eid}_power_ci_gamma_high"] = ci_gamma[1]

        # ---- d/v 和 d/v^2 的统计量 ----
        # 避免除零
        small_v = np.abs(v) < 1e-12
        v_safe = np.where(small_v, np.nan, v)
        d_over_v = d / v_safe
        d_over_v2 = d / (v_safe**2)
        metrics[f"{eid}_d_over_v_mean"] = np.nanmean(d_over_v)
        metrics[f"{eid}_d_over_v_std"] = np.nanstd(d_over_v)
        metrics[f"{eid}_d_over_v2_mean"] = np.nanmean(d_over_v2)
        metrics[f"{eid}_d_over_v2_std"] = np.nanstd(d_over_v2)

        # ---- 绘图: d vs v_sg 散点 + 三个拟合曲线 ----
        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(v, d, s=10, alpha=0.6, label="data")
        # 排序用于画曲线
        idx_sorted = np.argsort(v)
        v_sort = v[idx_sorted]
        ax.plot(v_sort, coeff_lin[0] + coeff_lin[1]*v_sort, 'r-', label=f"Linear R²={r2_lin:.3f}")
        ax.plot(v_sort, coeff_quad[0] + coeff_quad[1]*v_sort**2, 'g-', label=f"Quad R²={r2_quad:.3f}")
        if not np.isnan(beta_power):
            ax.plot(v_sort, power_law(v_sort, beta_power, gamma_power), 'b--', label=f"Power: β={beta_power:.3f}, γ={gamma_power:.3f}, R²={r2_power:.3f}")
        ax.set_xlabel("v_sg")
        ax.set_ylabel("d = a_sg - F_ext")
        ax.set_title(f"{eid}: d vs v_sg with fits")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig_path = output_dir / f"{eid}_d_vs_v_fits_power.png"
        fig.savefig(str(fig_path))
        plt.close(fig)
        figures.append(str(fig_path))

    # ---- 4.2 跨实验比较：将所有 d-v 数据合并拟合幂律 ----
    all_d = np.concatenate([all_d_v[eid][0] for eid in constant_exp_ids])
    all_v = np.concatenate([all_d_v[eid][1] for eid in constant_exp_ids])
    # 去掉负 v 或太小的 v
    mask = all_v > 0.01
    all_d_pos = all_d[mask]
    all_v_pos = all_v[mask]

    def global_power(v, beta, gamma):
        return -beta * (v ** gamma)

    try:
        popt_global, pcov_global = scipy.optimize.curve_fit(
            global_power, all_v_pos, all_d_pos,
            p0=[0.1, 1.0],
            bounds=([1e-10, 0.001], [np.inf, 3.0]),
            maxfev=5000
        )
        beta_global, gamma_global = popt_global
        d_global_pred = global_power(all_v_pos, beta_global, gamma_global)
        ss_res_global = np.sum((all_d_pos - d_global_pred)**2)
        ss_tot_global = np.sum((all_d_pos - np.mean(all_d_pos))**2)
        r2_global = 1 - ss_res_global / (ss_tot_global + 1e-15)
        resid_std_global = np.std(all_d_pos - d_global_pred)
        n_global = len(all_v_pos)
        t_val_global = scipy.stats.t.ppf(0.975, n_global-2) if n_global>2 else 1.96
        perr_global = np.sqrt(np.diag(pcov_global)) if pcov_global is not None else [0,0]
        ci_global_beta = (beta_global - t_val_global*perr_global[0], beta_global + t_val_global*perr_global[0])
        ci_global_gamma = (gamma_global - t_val_global*perr_global[1], gamma_global + t_val_global*perr_global[1])
    except Exception as e:
        beta_global, gamma_global = np.nan, np.nan
        r2_global = np.nan
        resid_std_global = np.nan
        ci_global_beta = (np.nan, np.nan)
        ci_global_gamma = (np.nan, np.nan)

    metrics["global_power_beta"] = beta_global
    metrics["global_power_gamma"] = gamma_global
    metrics["global_power_R2"] = r2_global
    metrics["global_power_resid_std"] = resid_std_global
    metrics["global_power_ci_beta_low"] = ci_global_beta[0]
    metrics["global_power_ci_beta_high"] = ci_global_beta[1]
    metrics["global_power_ci_gamma_low"] = ci_global_gamma[0]
    metrics["global_power_ci_gamma_high"] = ci_global_gamma[1]

    # ---- 跨实验 d-v 散点图合并 + 全局拟合 ----
    fig, ax = plt.subplots(figsize=(7,5))
    colors = ['b','g','r','c','m']
    for i, eid in enumerate(constant_exp_ids):
        d_e, v_e = all_d_v[eid]
        ax.scatter(v_e, d_e, s=8, alpha=0.5, color=colors[i % len(colors)], label=eid)
    if not np.isnan(beta_global):
        v_grid = np.linspace(0.01, max(all_v)*1.1, 200)
        ax.plot(v_grid, global_power(v_grid, beta_global, gamma_global), 'k--', 
                label=f"Global: β={beta_global:.3f}, γ={gamma_global:.3f}, R²={r2_global:.3f}")
    ax.set_xlabel("v_sg")
    ax.set_ylabel("d")
    ax.set_title("Cross-experiment d vs v_sg with global power-law fit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = output_dir / "cross_experiment_d_vs_v_global_power.png"
    fig.savefig(str(fig_path))
    plt.close(fig)
    figures.append(str(fig_path))

    # ---- 检查无外力实验 exp_01, exp_05 的 a_sg 是否接近零 ----
    for eid in ['exp_01', 'exp_05']:
        if eid in experiments:
            try:
                a_sg = get_series(eid, "a_sg")
                mean_a = np.mean(a_sg)
                std_a = np.std(a_sg)
                n_a = len(a_sg)
                se_a = std_a / np.sqrt(n_a)
                t_stat = mean_a / (se_a + 1e-15)
                p_value = 2 * (1 - scipy.stats.t.cdf(abs(t_stat), df=n_a-1))
                metrics[f"{eid}_a_sg_mean"] = mean_a
                metrics[f"{eid}_a_sg_std"] = std_a
                metrics[f"{eid}_a_sg_t_test_pvalue"] = p_value
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # 5. 构建 observation 字符串
    # -------------------------------------------------------------------------
    obs_lines = []
    obs_lines.append("对所有恒外力实验进行了 d = a_sg - F_ext 计算，并拟合了线性、二次和幂律模型。")
    obs_lines.append("每个实验的拟合参数（系数、95%CI、R²、残差标准差）已记录在 metrics 中。")
    obs_lines.append("d/v 和 d/v^2 的均值和标准差也已计算。")
    obs_lines.append(f"全局幂律拟合 d = -β * v^γ: β={beta_global:.4f} (CI [{ci_global_beta[0]:.4f},{ci_global_beta[1]:.4f}]), γ={gamma_global:.4f} (CI [{ci_global_gamma[0]:.4f},{ci_global_gamma[1]:.4f}]), R²={r2_global:.4f}")
    # 检查无外力
    for eid in ['exp_01','exp_05']:
        key_mean = f"{eid}_a_sg_mean"
        key_p = f"{eid}_a_sg_t_test_pvalue"
        if key_mean in metrics:
            obs_lines.append(f"{eid}: a_sg 均值={metrics[key_mean]:.6f}, p值={metrics[key_p]:.6f}")

    obs_lines.append("图像已保存：每个实验的 d 拟合图 + 跨实验全局图。")
    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics
    }
