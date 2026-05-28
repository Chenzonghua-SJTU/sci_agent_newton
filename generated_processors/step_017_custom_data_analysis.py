import json
import math
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def _estimate_kinematics(q, dt=0.1, window_length=11, polyorder=3):
    """从位置 q 估计平滑速度和加速度（Savitzky-Golay）"""
    if len(q) < window_length:
        window_length = len(q) if len(q) % 2 == 1 else len(q) - 1
        if window_length < 3:
            raise ValueError("序列太短，无法进行 Savitzky-Golay 滤波")
    q_smooth = savgol_filter(q, window_length, polyorder)
    v = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt)
    a = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)
    return v, a

def _linear_func(v, alpha, beta):
    return alpha + beta * v

def _quadratic_func(v, alpha, gamma):
    return alpha + gamma * v**2

def _fit_and_report(v, a, model, name):
    """拟合模型，返回 (alpha, alpha_ci, beta/beta_ci, gamma/gamma_ci?, R2, residuals)"""
    try:
        popt, pcov = curve_fit(model, v, a, maxfev=10000)
        # 标准误
        se = np.sqrt(np.diag(pcov))
        # 95% 置信区间（正态近似）
        ci_low = popt - 1.96 * se
        ci_high = popt + 1.96 * se
        # R²
        residuals = a - model(v, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((a - np.mean(a))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        return popt, ci_low, ci_high, r2, residuals
    except Exception as e:
        raise ValueError(f"拟合 {name} 失败: {e}")

def process(payload: dict) -> dict:
    parameters = payload["parameters"]
    exp_ids = parameters.get("experiment_ids", [])
    analysis_goal = parameters.get("analysis_goal", "")
    optional_series = parameters.get("optional_series", [])

    experiments = payload["experiments"]
    output_dir = Path(payload["output_dir"])

    # 根据 analysis_goal 确定要处理的实验：
    # 恒外力实验：给定的 exp_ids（通常是恒外力实验）
    # 另外需要检查无外力实验 exp_01 和 exp_05
    constant_force_exps = [eid for eid in exp_ids if eid in experiments]
    check_free_exps = ["exp_01", "exp_05"]
    all_processing_exps = constant_force_exps + [eid for eid in check_free_exps if eid in experiments and eid not in constant_force_exps]

    metrics = {}
    derived_series = []
    figures = []

    # 首先确保所有处理的实验都有 v_sg 和 a_sg
    for eid in all_processing_exps:
        exp = experiments[eid]
        series = exp["series"]
        config = exp["config"]
        if "v_sg" not in series or "a_sg" not in series:
            # 从 q 估计
            q = series.get("q", None)
            if q is None:
                raise ValueError(f"实验 {eid} 缺少系列序列 q，无法估计运动学")
            t = series["t"]
            dt = config.get("dt", 0.1)
            v_est, a_est = _estimate_kinematics(q, dt)
            # 添加为派生序列
            derived_series.append({
                "experiment_id": eid,
                "name": "v_sg",
                "values": v_est.tolist(),
                "source_name": "Savitzky-Golay (window=11, polyorder=3) from q",
                "provenance": "generated data processor: custom_data_analysis",
                "description": "平滑速度估计"
            })
            derived_series.append({
                "experiment_id": eid,
                "name": "a_sg",
                "values": a_est.tolist(),
                "source_name": "Savitzky-Golay (window=11, polyorder=3) from q",
                "provenance": "generated data processor: custom_data_analysis",
                "description": "平滑加速度估计"
            })
            # 注：我们不会修改 payload，但返回的 derived_series 会通知上层

    # ---------- 1. 恒外力实验拟合 a_sg vs v_sg ----------
    for eid in constant_force_exps:
        exp = experiments[eid]
        config = exp["config"]
        F_ext = config.get("F_ext", 0.0)
        series = exp["series"]
        v = np.array(series["v_sg"])
        a = np.array(series["a_sg"])
        t = np.array(series["t"])

        # 线性拟合 a = alpha + beta * v
        try:
            popt_lin, ci_low_lin, ci_high_lin, r2_lin, resid_lin = _fit_and_report(v, a, _linear_func, "线性")
            alpha_lin, beta_lin = popt_lin
        except Exception as e:
            alpha_lin = np.nan; beta_lin = np.nan; ci_low_lin = [np.nan, np.nan]; ci_high_lin = [np.nan, np.nan]; r2_lin = np.nan; resid_lin = np.full_like(a, np.nan)

        # 二次拟合 a = alpha + gamma * v^2
        try:
            popt_quad, ci_low_quad, ci_high_quad, r2_quad, resid_quad = _fit_and_report(v, a, _quadratic_func, "二次")
            alpha_quad, gamma_quad = popt_quad
        except Exception as e:
            alpha_quad = np.nan; gamma_quad = np.nan; ci_low_quad = [np.nan, np.nan]; ci_high_quad = [np.nan, np.nan]; r2_quad = np.nan; resid_quad = np.full_like(a, np.nan)

        # 记录指标
        metrics.update({
            f"{eid}_lin_alpha": alpha_lin,
            f"{eid}_lin_alpha_ci_low": ci_low_lin[0],
            f"{eid}_lin_alpha_ci_high": ci_high_lin[0],
            f"{eid}_lin_beta": beta_lin,
            f"{eid}_lin_beta_ci_low": ci_low_lin[1] if len(ci_low_lin) > 1 else np.nan,
            f"{eid}_lin_beta_ci_high": ci_high_lin[1] if len(ci_high_lin) > 1 else np.nan,
            f"{eid}_lin_R2": r2_lin,
            f"{eid}_quad_alpha": alpha_quad,
            f"{eid}_quad_alpha_ci_low": ci_low_quad[0],
            f"{eid}_quad_alpha_ci_high": ci_high_quad[0],
            f"{eid}_quad_gamma": gamma_quad,
            f"{eid}_quad_gamma_ci_low": ci_low_quad[1] if len(ci_low_quad) > 1 else np.nan,
            f"{eid}_quad_gamma_ci_high": ci_high_quad[1] if len(ci_high_quad) > 1 else np.nan,
            f"{eid}_quad_R2": r2_quad,
        })

        # 残差统计
        for tag, resid in [("lin", resid_lin), ("quad", resid_quad)]:
            if np.any(np.isfinite(resid)):
                metrics[f"{eid}_resid_{tag}_mean"] = np.nanmean(resid)
                metrics[f"{eid}_resid_{tag}_std"] = np.nanstd(resid)
                metrics[f"{eid}_resid_{tag}_max_abs"] = np.nanmax(np.abs(resid))
            else:
                metrics[f"{eid}_resid_{tag}_mean"] = np.nan
                metrics[f"{eid}_resid_{tag}_std"] = np.nan
                metrics[f"{eid}_resid_{tag}_max_abs"] = np.nan

        # 绘制散点图叠加拟合曲线
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(v, a, alpha=0.6, label=f"{eid} data (F_ext={F_ext})")
        v_sorted = np.sort(v)
        if not np.isnan(alpha_lin) and not np.isnan(beta_lin):
            ax.plot(v_sorted, _linear_func(v_sorted, alpha_lin, beta_lin), 'r-', label=f"linear: α={alpha_lin:.4f}, β={beta_lin:.4f}, R²={r2_lin:.4f}")
        if not np.isnan(alpha_quad) and not np.isnan(gamma_quad):
            ax.plot(v_sorted, _quadratic_func(v_sorted, alpha_quad, gamma_quad), 'g--', label=f"quadratic: α={alpha_quad:.4f}, γ={gamma_quad:.4f}, R²={r2_quad:.4f}")
        ax.set_xlabel("v_sg")
        ax.set_ylabel("a_sg")
        ax.set_title(f"{eid}: a_sg vs v_sg fits")
        ax.legend()
        figname = f"{eid}_a_vs_v_fits.png"
        fig_path = output_dir / figname
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append(str(fig_path))

        # ---------- 2. 阻尼项 d = a_sg - F_ext 的拟合 ----------
        d = a - F_ext
        # 线性拟合 d = alpha_d + beta_d * v
        try:
            popt_d_lin, ci_d_lin_low, ci_d_lin_high, r2_d_lin, resid_d_lin = _fit_and_report(v, d, _linear_func, "d线性")
            alpha_d_lin, beta_d_lin = popt_d_lin
        except Exception as e:
            alpha_d_lin = np.nan; beta_d_lin = np.nan; ci_d_lin_low = [np.nan, np.nan]; ci_d_lin_high = [np.nan, np.nan]; r2_d_lin = np.nan; resid_d_lin = np.full_like(d, np.nan)

        # 二次拟合 d = alpha_d + gamma_d * v^2
        try:
            popt_d_quad, ci_d_quad_low, ci_d_quad_high, r2_d_quad, resid_d_quad = _fit_and_report(v, d, _quadratic_func, "d二次")
            alpha_d_quad, gamma_d_quad = popt_d_quad
        except Exception as e:
            alpha_d_quad = np.nan; gamma_d_quad = np.nan; ci_d_quad_low = [np.nan, np.nan]; ci_d_quad_high = [np.nan, np.nan]; r2_d_quad = np.nan; resid_d_quad = np.full_like(d, np.nan)

        # 记录指标
        metrics.update({
            f"{eid}_d_lin_alpha": alpha_d_lin,
            f"{eid}_d_lin_alpha_ci_low": ci_d_lin_low[0],
            f"{eid}_d_lin_alpha_ci_high": ci_d_lin_high[0],
            f"{eid}_d_lin_beta": beta_d_lin,
            f"{eid}_d_lin_beta_ci_low": ci_d_lin_low[1] if len(ci_d_lin_low) > 1 else np.nan,
            f"{eid}_d_lin_beta_ci_high": ci_d_lin_high[1] if len(ci_d_lin_high) > 1 else np.nan,
            f"{eid}_d_lin_R2": r2_d_lin,
            f"{eid}_d_quad_alpha": alpha_d_quad,
            f"{eid}_d_quad_alpha_ci_low": ci_d_quad_low[0],
            f"{eid}_d_quad_alpha_ci_high": ci_d_quad_high[0],
            f"{eid}_d_quad_gamma": gamma_d_quad,
            f"{eid}_d_quad_gamma_ci_low": ci_d_quad_low[1] if len(ci_d_quad_low) > 1 else np.nan,
            f"{eid}_d_quad_gamma_ci_high": ci_d_quad_high[1] if len(ci_d_quad_high) > 1 else np.nan,
            f"{eid}_d_quad_R2": r2_d_quad,
        })

        # 残差统计
        for tag, resid in [("d_lin", resid_d_lin), ("d_quad", resid_d_quad)]:
            if np.any(np.isfinite(resid)):
                metrics[f"{eid}_resid_{tag}_mean"] = np.nanmean(resid)
                metrics[f"{eid}_resid_{tag}_std"] = np.nanstd(resid)
                metrics[f"{eid}_resid_{tag}_max_abs"] = np.nanmax(np.abs(resid))
            else:
                metrics[f"{eid}_resid_{tag}_mean"] = np.nan
                metrics[f"{eid}_resid_{tag}_std"] = np.nan
                metrics[f"{eid}_resid_{tag}_max_abs"] = np.nan

        # 绘制 d vs v 散点图
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(v, d, alpha=0.6, label=f"d = a_sg - F_ext")
        if not np.isnan(alpha_d_lin) and not np.isnan(beta_d_lin):
            ax.plot(v_sorted, _linear_func(v_sorted, alpha_d_lin, beta_d_lin), 'r-', label=f"linear: α={alpha_d_lin:.4f}, β={beta_d_lin:.4f}, R²={r2_d_lin:.4f}")
        if not np.isnan(alpha_d_quad) and not np.isnan(gamma_d_quad):
            ax.plot(v_sorted, _quadratic_func(v_sorted, alpha_d_quad, gamma_d_quad), 'g--', label=f"quadratic: α={alpha_d_quad:.4f}, γ={gamma_d_quad:.4f}, R²={r2_d_quad:.4f}")
        ax.set_xlabel("v_sg")
        ax.set_ylabel("d = a_sg - F_ext")
        ax.set_title(f"{eid}: damping term d vs v")
        ax.legend()
        figname = f"{eid}_d_vs_v_fits.png"
        fig_path = output_dir / figname
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures.append(str(fig_path))

        # 将阻尼项 d 注册为派生序列
        derived_series.append({
            "experiment_id": eid,
            "name": "d",
            "values": d.tolist(),
            "source_name": f"d = a_sg - F_ext (F_ext={F_ext})",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "阻尼项：加速度减去外力（假设质量=1）"
        })

    # ---------- 3. 检查无外力实验 a_sg 均值是否接近 0 ----------
    for eid in check_free_exps:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        series = exp["series"]
        if "a_sg" not in series:
            continue
        a_sg = np.array(series["a_sg"])
        mean_a = np.mean(a_sg)
        std_a = np.std(a_sg)
        # t 检验：H0: 均值 = 0
        if std_a > 0 and len(a_sg) > 1:
            t_stat, p_value = stats.ttest_1samp(a_sg, 0.0)
        else:
            t_stat = 0.0
            p_value = 1.0
        metrics[f"{eid}_a_sg_mean"] = mean_a
        metrics[f"{eid}_a_sg_std"] = std_a
        metrics[f"{eid}_a_sg_t_test_pvalue"] = p_value

    # ---------- 构建 observation ----------
    obs_parts = []
    obs_parts.append("对所有恒外力实验（{}）分别进行了 a_sg 与 v_sg 的线性与二次拟合，并绘制了散点与拟合曲线叠加图。".format(", ".join(constant_force_exps)))
    obs_parts.append("对所有恒外力实验计算了阻尼项 d = a_sg - F_ext，并进行了线性和二次拟合，曲线图已保存。")
    if check_free_exps:
        exp01_mean = metrics.get("exp_01_a_sg_mean")
        exp01_p = metrics.get("exp_01_a_sg_t_test_pvalue")
        exp05_mean = metrics.get("exp_05_a_sg_mean")
        exp05_p = metrics.get("exp_05_a_sg_t_test_pvalue")
        obs_parts.append(f"检查了无外力实验 exp_01 和 exp_05 的 a_sg 均值：exp_01 均值={exp01_mean:.6f} (p={exp01_p:.4f})，exp_05 均值={exp05_mean:.6f} (p={exp05_p:.4f})。")
    obs_parts.append("详细拟合系数、置信区间和 R² 可参见 metrics 字典。")
    observation = "。".join(obs_parts)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
