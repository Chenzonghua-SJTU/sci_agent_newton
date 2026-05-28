import json
import math
import statistics
from itertools import accumulate
from functools import reduce
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _central_diff_velocity(q: np.ndarray, dt: float) -> np.ndarray:
    """中心差分求速度，前后点用前向/后向差分填充"""
    v = np.full_like(q, np.nan)
    v[0] = (q[1] - q[0]) / dt
    v[-1] = (q[-1] - q[-2]) / dt
    v[1:-1] = (q[2:] - q[:-2]) / (2.0 * dt)
    return v


def _central_diff_acceleration(v: np.ndarray, dt: float) -> np.ndarray:
    """中心差分求加速度，前后两点用NaN填充"""
    a = np.full_like(v, np.nan)
    a[2:-2] = (v[3:-1] - v[1:-3]) / (2.0 * dt)
    return a


def _exp_model(v, a0, tau):
    return a0 * np.exp(-v / tau)


def process(payload: dict) -> dict:
    parameters = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = Path(payload["output_dir"])

    # 需要处理的实验ID列表，追加自由实验
    exp_ids = parameters.get("experiment_ids", list(experiments.keys()))
    free_exp_ids = {"exp_02", "exp_08", "exp_16"}
    exp_ids = list(set(exp_ids).union(free_exp_ids))

    # 结果容器
    fit_results: List[dict] = []
    free_checks: List[dict] = []

    # 收集用于全局模型的数据 (只恒外力实验，且v方向与F_ext一致)
    global_v = []
    global_a = []
    global_F = []
    global_v0 = []

    for exp_id in exp_ids:
        exp = experiments[exp_id]
        config = exp["config"]
        series = exp["series"]

        t = np.array(series["t"])
        q = np.array(series["q"])
        F_ext = config.get("F_ext", 0.0)
        force_type = config.get("force_field_type", "")
        v0 = config.get("initial_v", 0.0)
        dt = config.get("dt", t[1] - t[0])

        # 获取或计算加速度和速度
        if "acceleration_central" in series and "velocity_central" in series:
            a_raw = np.array(series["acceleration_central"])
            v_raw = np.array(series["velocity_central"])
        else:
            v_raw = _central_diff_velocity(q, dt)
            a_raw = _central_diff_acceleration(v_raw, dt)

        # 剔除边界5个点
        if len(t) > 10:
            idx = slice(5, -5)
        else:
            idx = slice(0, len(t))
        v_int = v_raw[idx]
        a_int = a_raw[idx]

        # 去掉NaN和Inf
        mask = np.isfinite(v_int) & np.isfinite(a_int)
        v_clean = v_int[mask]
        a_clean = a_int[mask]

        if len(v_clean) < 3:
            # 数据太少
            continue

        # ---- 自由运动实验验证 ----
        if force_type == "free" or abs(F_ext) < 1e-12:
            mean_a = float(np.mean(a_clean))
            max_abs_a = float(np.max(np.abs(a_clean)))
            is_zero = abs(mean_a) < 1e-12
            free_checks.append({
                "experiment": exp_id,
                "F_ext": F_ext,
                "v0": v0,
                "mean_acceleration": mean_a,
                "max_abs_acceleration": max_abs_a,
                "is_zero": is_zero
            })
            continue  # 自由实验不参与拟合

        # ---- 恒外力实验：指数拟合 ----
        # 确保 v 为正（通过翻转）
        v_mean = np.mean(v_clean)
        if v_mean < 0:
            v_fit = -v_clean
            a_fit = -a_clean
            flipped = True
        else:
            v_fit = v_clean
            a_fit = a_clean
            flipped = False

        # 拟合 a = a0 * exp(-v/tau)
        try:
            a0_guess = float(np.max(np.abs(a_fit)))
            if a0_guess <= 0:
                a0_guess = 1.0
            popt, pcov = curve_fit(
                _exp_model, v_fit, a_fit,
                p0=[a0_guess, 1.0],
                maxfev=20000
            )
            a0_fit, tau_fit = popt
            pred = _exp_model(v_fit, a0_fit, tau_fit)
            residuals = a_fit - pred
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            ss_res = float(np.sum(residuals ** 2))
            ss_tot = float(np.sum((a_fit - np.mean(a_fit)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

            # 如果翻转，恢复 a0 的符号
            if flipped:
                a0_fit = -a0_fit

            fit_results.append({
                "experiment": exp_id,
                "F_ext": F_ext,
                "v0": v0,
                "a0_fit": a0_fit,
                "tau_fit": tau_fit,
                "RMSE": rmse,
                "R2": r2,
                "n_points": len(v_fit),
                "error": None
            })

            # 如果 v 方向与 F_ext 符号一致（或翻转后一致），收集全局数据
            # 统一规则：F_ext>0 时 v_fit>0，F_ext<0 时 v_fit<0
            # 由于我们已经保证 v_fit>0，所以只有 F_ext>0 的点才保留
            if F_ext > 0:
                global_v.extend(v_fit.tolist())
                global_a.extend(a_fit.tolist())  # a_fit 此时为正
                global_F.extend([F_ext] * len(v_fit))
                global_v0.extend([v0] * len(v_fit))

        except Exception as e:
            fit_results.append({
                "experiment": exp_id,
                "F_ext": F_ext,
                "v0": v0,
                "a0_fit": None,
                "tau_fit": None,
                "RMSE": None,
                "R2": None,
                "n_points": 0,
                "error": str(e)
            })

    # ---- tau 统计 ----
    tau_vals = [r["tau_fit"] for r in fit_results if r["tau_fit"] is not None]
    tau_mean = float(np.mean(tau_vals)) if tau_vals else None
    tau_std = float(np.std(tau_vals)) if tau_vals else None

    # ---- a0 与 F_ext、v0 的关系 ----
    valid_fits = [r for r in fit_results if r["a0_fit"] is not None]
    regression_a0_F = None
    regression_a0_v0 = None
    regression_a0_multi = None

    if len(valid_fits) >= 3:
        F_vals = np.array([r["F_ext"] for r in valid_fits])
        v0_vals = np.array([r["v0"] for r in valid_fits])
        a0_vals = np.array([r["a0_fit"] for r in valid_fits])

        # a0 ~ F_ext
        reg_F = LinearRegression().fit(F_vals.reshape(-1, 1), a0_vals)
        regression_a0_F = {
            "slope": float(reg_F.coef_[0]),
            "intercept": float(reg_F.intercept_),
            "R2": float(reg_F.score(F_vals.reshape(-1, 1), a0_vals))
        }

        # a0 ~ v0
        if np.unique(v0_vals).size > 1:
            reg_v0 = LinearRegression().fit(v0_vals.reshape(-1, 1), a0_vals)
            regression_a0_v0 = {
                "slope": float(reg_v0.coef_[0]),
                "intercept": float(reg_v0.intercept_),
                "R2": float(reg_v0.score(v0_vals.reshape(-1, 1), a0_vals))
            }
        else:
            regression_a0_v0 = {"slope": 0.0, "intercept": float(np.mean(a0_vals)), "R2": 0.0}

        # 双变量 a0 = c1*F_ext + c2*v0 + intercept
        X_multi = np.column_stack([F_vals, v0_vals])
        reg_multi = LinearRegression().fit(X_multi, a0_vals)
        regression_a0_multi = {
            "c1": float(reg_multi.coef_[0]),
            "c2": float(reg_multi.coef_[1]),
            "intercept": float(reg_multi.intercept_),
            "R2": float(reg_multi.score(X_multi, a0_vals))
        }

    # ---- 全局模型拟合 a = (c1*F_ext + c2*v0) * exp(-v/tau) ----
    global_fit_result = None
    if len(global_v) > 100:
        v_arr = np.array(global_v)
        a_arr = np.array(global_a)
        F_arr = np.array(global_F)
        v0_arr = np.array(global_v0)
        # 构造自变量：c1*F_ext + c2*v0
        # 但这里我们直接拟合参数 c1, c2, tau
        def global_model(data, c1, c2, tau):
            F, v0, v = data
            return (c1 * F + c2 * v0) * np.exp(-v / tau)

        try:
            p0 = [1.0, 0.0, 10.0]
            popt_global, _ = curve_fit(
                lambda x, c1, c2, tau: global_model(x, c1, c2, tau),
                (F_arr, v0_arr, v_arr),
                a_arr, p0=p0, maxfev=20000
            )
            pred_global = global_model((F_arr, v0_arr, v_arr), *popt_global)
            residuals_global = a_arr - pred_global
            rmse_global = float(np.sqrt(np.mean(residuals_global ** 2)))
            ss_res = float(np.sum(residuals_global ** 2))
            ss_tot = float(np.sum((a_arr - np.mean(a_arr)) ** 2))
            r2_global = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            global_fit_result = {
                "c1": float(popt_global[0]),
                "c2": float(popt_global[1]),
                "tau": float(popt_global[2]),
                "RMSE": rmse_global,
                "R2": r2_global,
                "n_points": len(v_arr)
            }
        except Exception as e:
            global_fit_result = {"error": str(e)}

    # ---- 绘图 ----
    base_name = "step_analyze_data"
    fig_files = []

    # 图1：每个实验的 a-v 拟合曲线（挑选几个典型）
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    i = -1  # 初始化，防止 valid_fits 为空
    for i, entry in enumerate(valid_fits[:12]):
        ax = axes[i]
        exp_id = entry["experiment"]
        # 获取之前存储的 v_fit, a_fit 数据（需要重新从实验获取）
        exp = experiments[exp_id]
        config = exp["config"]
        series = exp["series"]
        t = np.array(series["t"])
        q = np.array(series["q"])
        if "acceleration_central" in series and "velocity_central" in series:
            a_all = np.array(series["acceleration_central"])
            v_all = np.array(series["velocity_central"])
        else:
            v_all = _central_diff_velocity(q, dt)
            a_all = _central_diff_acceleration(v_all, dt)
        # 内部点
        if len(t) > 10:
            sidx = slice(5, -5)
        else:
            sidx = slice(0, len(t))
        v_plot = v_all[sidx]
        a_plot = a_all[sidx]
        mask = np.isfinite(v_plot) & np.isfinite(a_plot)
        v_plot = v_plot[mask]
        a_plot = a_plot[mask]

        # 确保 v 为正用于画拟合线
        v_mean_plot = np.mean(v_plot)
        if v_mean_plot < 0:
            v_plot_for_line = np.linspace(np.min(-v_plot), np.max(-v_plot), 100)
            a_line = entry["a0_fit"] * np.exp(-v_plot_for_line / entry["tau_fit"])
        else:
            v_plot_for_line = np.linspace(np.min(v_plot), np.max(v_plot), 100)
            a_line = entry["a0_fit"] * np.exp(-v_plot_for_line / entry["tau_fit"])

        ax.scatter(v_plot, a_plot, s=5, alpha=0.6, label="data")
        ax.plot(v_plot_for_line, a_line, 'r-', label="exp fit")
        ax.set_title(f"{exp_id} (F={entry['F_ext']})")
        ax.set_xlabel("v")
        ax.set_ylabel("a")
        ax.legend(fontsize=6)

    for j in range(max(i + 1, 0), 12):
        axes[j].axis("off")
    plt.tight_layout()
    fig_path = output_dir / f"{base_name}_per_experiment_fits.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    fig_files.append(str(fig_path))

    # 图2：a0 vs F_ext
    if len(valid_fits) > 2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        F_vals_plot = np.array([r["F_ext"] for r in valid_fits])
        a0_vals_plot = np.array([r["a0_fit"] for r in valid_fits])
        ax2.scatter(F_vals_plot, a0_vals_plot, c='blue')
        # 线性拟合线
        if regression_a0_F:
            F_line = np.linspace(F_vals_plot.min(), F_vals_plot.max(), 100)
            a0_line = regression_a0_F["slope"] * F_line + regression_a0_F["intercept"]
            ax2.plot(F_line, a0_line, 'r--', label=f'R²={regression_a0_F["R2"]:.3f}')
        ax2.set_xlabel("F_ext")
        ax2.set_ylabel("a0 (fit)")
        ax2.set_title("a0 vs F_ext")
        ax2.legend()
        fig_path2 = output_dir / f"{base_name}_a0_vs_Fext.png"
        plt.savefig(fig_path2, dpi=150)
        plt.close()
        fig_files.append(str(fig_path2))

    # 图3：a0 vs v0 (如果v0有变化)
    if len(valid_fits) > 2 and len(set(r["v0"] for r in valid_fits)) > 1:
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        v0_vals_plot = np.array([r["v0"] for r in valid_fits])
        ax3.scatter(v0_vals_plot, a0_vals_plot, c='green')
        if regression_a0_v0:
            v0_line = np.linspace(v0_vals_plot.min(), v0_vals_plot.max(), 100)
            a0_line_v0 = regression_a0_v0["slope"] * v0_line + regression_a0_v0["intercept"]
            ax3.plot(v0_line, a0_line_v0, 'r--', label=f'R²={regression_a0_v0["R2"]:.3f}')
        ax3.set_xlabel("v0")
        ax3.set_ylabel("a0 (fit)")
        ax3.set_title("a0 vs v0")
        ax3.legend()
        fig_path3 = output_dir / f"{base_name}_a0_vs_v0.png"
        plt.savefig(fig_path3, dpi=150)
        plt.close()
        fig_files.append(str(fig_path3))

    # 图4：tau 分布直方图
    if tau_vals:
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        ax4.hist(tau_vals, bins=min(10, len(tau_vals)), edgecolor='black')
        ax4.axvline(tau_mean, color='r', linestyle='--', label=f'mean={tau_mean:.3f}')
        ax4.set_xlabel("tau")
        ax4.set_ylabel("frequency")
        ax4.set_title(f"tau distribution (std={tau_std:.4f})")
        ax4.legend()
        fig_path4 = output_dir / f"{base_name}_tau_hist.png"
        plt.savefig(fig_path4, dpi=150)
        plt.close()
        fig_files.append(str(fig_path4))

    # ---- 构建 observation ----
    obs_parts = []
    obs_parts.append(f"处理了 {len(fit_results)} 个恒外力实验的指数拟合。")
    # 自由检查
    passed_free = [r for r in free_checks if r["is_zero"]]
    obs_parts.append(f"自由运动检查：{len(passed_free)}/{len(free_checks)} 实验加速度均值<1e-12。")
    if tau_mean is not None:
        obs_parts.append(f"tau 均值 = {tau_mean:.4f}, 标准差 = {tau_std:.4f}。")
    # 回归
    if regression_a0_F:
        obs_parts.append(f"a0 vs F_ext 回归：斜率={regression_a0_F['slope']:.4f}, 截距={regression_a0_F['intercept']:.4f}, R²={regression_a0_F['R2']:.4f}。")
    if regression_a0_v0:
        obs_parts.append(f"a0 vs v0 回归：斜率={regression_a0_v0['slope']:.4f}, 截距={regression_a0_v0['intercept']:.4f}, R²={regression_a0_v0['R2']:.4f}。")
    if regression_a0_multi:
        obs_parts.append(f"双变量回归 a0 = c1*F_ext + c2*v0 + intercept：c1={regression_a0_multi['c1']:.4f}, c2={regression_a0_multi['c2']:.4f}, intercept={regression_a0_multi['intercept']:.4f}, R²={regression_a0_multi['R2']:.4f}。")
    if global_fit_result and "error" not in global_fit_result:
        obs_parts.append(f"全局模型 (a = (c1*F_ext + c2*v0)*exp(-v/tau)) 使用 {global_fit_result['n_points']} 个点（F_ext>0）：c1={global_fit_result['c1']:.4f}, c2={global_fit_result['c2']:.4f}, tau={global_fit_result['tau']:.4f}, RMSE={global_fit_result['RMSE']:.5f}, R²={global_fit_result['R2']:.4f}。")
    elif global_fit_result and "error" in global_fit_result:
        obs_parts.append(f"全局模型拟合失败：{global_fit_result['error']}")

    observation = " ".join(obs_parts)

    # ---- 构建 metrics ----
    metrics = {
        "per_experiment_fits": fit_results,
        "tau_mean": tau_mean,
        "tau_std": tau_std,
        "free_experiment_checks": free_checks,
        "a0_vs_F_ext_regression": regression_a0_F,
        "a0_vs_v0_regression": regression_a0_v0,
        "a0_multi_regression": regression_a0_multi,
        "global_model_fit": global_fit_result
    }

    return {
        "observation": observation,
        "derived_series": [],
        "figures": fig_files,
        "metrics": metrics
    }
