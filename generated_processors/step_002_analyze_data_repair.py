import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, signal
from sklearn import metrics as sk_metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = Path(payload.get("output_dir", "."))

    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(experiments.keys())

    # 只处理 exp_02 和 exp_03
    target_ids = [eid for eid in exp_ids if eid in ("exp_02", "exp_03")]
    if not target_ids:
        raise ValueError("No valid experiment IDs (exp_02 or exp_03) found in parameters.")

    derived_series_list = []
    figures = []
    obs_parts = []
    metrics = {}

    # 平滑参数
    window_length = 11
    polyorder = 2

    def get_v_a_series(exp_id: str, t: np.ndarray, q: np.ndarray, available: list) -> Tuple[np.ndarray, np.ndarray]:
        """获取速度 v 和加速度 a，优先使用已有的平滑序列"""
        v_name = f"v_smoothed_{window_length}_{exp_id}"
        a_name = f"a_smoothed_{window_length}_{exp_id}"
        if v_name in available and a_name in available:
            # 提取已有序列
            v = np.array(experiments[exp_id]["series"][v_name])
            a = np.array(experiments[exp_id]["series"][a_name])
            return v, a
        # 否则自己从 q 计算
        q_smooth = signal.savgol_filter(q, window_length, polyorder)
        # 速度：先对 q 平滑，再中心差分
        v = signal.savgol_filter(q, window_length, polyorder, deriv=1, delta=t[1]-t[0])
        # 加速度：对 v 再求导（或直接对 q 求二阶导）
        a = signal.savgol_filter(q, window_length, polyorder, deriv=2, delta=t[1]-t[0])
        return v, a

    # 处理 exp_02
    for exp_id in target_ids:
        exp_data = experiments.get(exp_id, {})
        if not exp_data:
            raise ValueError(f"Experiment {exp_id} not found in payload.")
        series = exp_data.get("series", {})
        available = exp_data.get("available_series", [])
        t = np.array(series.get("t"))
        q = np.array(series.get("q"))
        if t is None or q is None:
            raise ValueError(f"Experiment {exp_id} missing 't' or 'q' series.")

        dt = t[1] - t[0]
        v, a = get_v_a_series(exp_id, t, q, available)

        # 存储 v 和 a 作为派生序列
        derived_series_list.append({
            "experiment_id": exp_id,
            "name": f"velocity_{exp_id}",
            "values": v.tolist(),
            "source_name": f"savgol_filter(window={window_length}, polyorder={polyorder}, deriv=1)",
            "provenance": "generated data processor: step_031_analyze_data",
            "description": f"一阶导数（速度） from q(t) using Savitzky-Golay filter"
        })
        derived_series_list.append({
            "experiment_id": exp_id,
            "name": f"acceleration_{exp_id}",
            "values": a.tolist(),
            "source_name": f"savgol_filter(window={window_length}, polyorder={polyorder}, deriv=2)",
            "provenance": "generated data processor: step_031_analyze_data",
            "description": f"二阶导数（加速度） from q(t) using Savitzky-Golay filter"
        })

        # 绘制 velocity vs t
        fig_v, ax_v = plt.subplots(figsize=(6,4))
        ax_v.plot(t, v, 'b-', label='v(t)')
        ax_v.set_xlabel('t')
        ax_v.set_ylabel('velocity')
        ax_v.set_title(f'velocity vs t ({exp_id})')
        ax_v.legend()
        fname_v = output_dir / f"velocity_vs_t_{exp_id}.png"
        fig_v.savefig(fname_v)
        plt.close(fig_v)
        figures.append(str(fname_v))

        # 绘制 acceleration vs t
        fig_a, ax_a = plt.subplots(figsize=(6,4))
        ax_a.plot(t, a, 'r-', label='a(t)')
        ax_a.set_xlabel('t')
        ax_a.set_ylabel('acceleration')
        ax_a.set_title(f'acceleration vs t ({exp_id})')
        ax_a.legend()
        fname_a = output_dir / f"acceleration_vs_t_{exp_id}.png"
        fig_a.savefig(fname_a)
        plt.close(fig_a)
        figures.append(str(fname_a))

        # 对 exp_02: 线性拟合残差验证
        if exp_id == "exp_02":
            # 线性模型 q = slope * t + intercept
            A = np.vstack([t, np.ones_like(t)]).T
            slope, intercept = np.linalg.lstsq(A, q, rcond=None)[0]
            q_pred_linear = slope * t + intercept
            residuals_linear = q - q_pred_linear
            rmse_linear = np.sqrt(np.mean(residuals_linear**2))
            mae_linear = np.mean(np.abs(residuals_linear))
            r2_linear = 1 - np.sum(residuals_linear**2) / np.sum((q - np.mean(q))**2)

            metrics[f"linear_fit_{exp_id}"] = {
                "slope": slope,
                "intercept": intercept,
                "RMSE": rmse_linear,
                "MAE": mae_linear,
                "R2": r2_linear
            }
            derived_series_list.append({
                "experiment_id": exp_id,
                "name": f"residual_linear_{exp_id}",
                "values": residuals_linear.tolist(),
                "source_name": f"q - (slope*t+intercept)",
                "provenance": "generated data processor: step_031_analyze_data",
                "description": f"线性拟合残差（q - (slope*t+intercept)）"
            })

            # 绘制残差图
            fig_res, ax_res = plt.subplots(figsize=(6,4))
            ax_res.plot(t, residuals_linear, 'o', markersize=2, label='residual')
            ax_res.axhline(0, color='gray', linestyle='--')
            ax_res.set_xlabel('t')
            ax_res.set_ylabel('residual (q - linear)')
            ax_res.set_title(f'Linear fit residuals ({exp_id})')
            ax_res.legend()
            fname_res = output_dir / f"linear_residuals_{exp_id}.png"
            fig_res.savefig(fname_res)
            plt.close(fig_res)
            figures.append(str(fname_res))

            obs_parts.append(f"{exp_id}: 线性拟合 slope={slope:.6f}, intercept={intercept:.6f}, RMSE={rmse_linear:.6f}, MAE={mae_linear:.6f}, R²={r2_linear:.6f}")

        # 对 exp_03: 幂律拟合 q = c * t^p
        if exp_id == "exp_03":
            # 去掉 t=0 的点以避免 log(0)
            mask = t > 0
            t_pos = t[mask]
            q_pos = q[mask]
            # 初始猜测：通过 log-log 线性回归
            log_t = np.log(t_pos)
            log_q = np.log(q_pos)
            poly_coeff = np.polyfit(log_t, log_q, 1)
            p_init = poly_coeff[0]
            c_init = np.exp(poly_coeff[1])

            def power_law(t, c, p):
                return c * t**p

            power_fit_succeeded = False
            residuals_power = None
            try:
                popt, pcov = optimize.curve_fit(power_law, t_pos, q_pos, p0=[c_init, p_init])
                c_opt, p_opt = popt
                perr = np.sqrt(np.diag(pcov))
                power_fit_succeeded = True
            except Exception as e:
                # fallback to least squares on log scale
                log_q_fit = poly_coeff[0] * log_t + poly_coeff[1]
                q_fit = np.exp(log_q_fit)
                residuals_power = q_pos - q_fit
                rmse_power = np.sqrt(np.mean(residuals_power**2))
                # store approximate params
                c_opt, p_opt = np.exp(poly_coeff[1]), poly_coeff[0]
                perr = [0, 0]
                obs_parts.append(f"{exp_id}: 幂律拟合 log-log 线性回归 c={c_opt:.6f}, p={p_opt:.6f}, RMSE={rmse_power:.6f} (curve_fit failed)")
            else:
                q_pred_power = power_law(t_pos, c_opt, p_opt)
                residuals_power = q_pos - q_pred_power
                rmse_power = np.sqrt(np.mean(residuals_power**2))
                mae_power = np.mean(np.abs(residuals_power))
                r2_power = 1 - np.sum(residuals_power**2) / np.sum((q_pos - np.mean(q_pos))**2)

                metrics[f"power_law_fit_{exp_id}"] = {
                    "c": c_opt,
                    "c_error": perr[0],
                    "p": p_opt,
                    "p_error": perr[1],
                    "RMSE": rmse_power,
                    "MAE": mae_power,
                    "R2": r2_power
                }

                obs_parts.append(f"{exp_id}: 幂律拟合 q={c_opt:.6f} * t^{p_opt:.6f}, c_err={perr[0]:.6f}, p_err={perr[1]:.6f}, RMSE={rmse_power:.6f}, MAE={mae_power:.6f}, R²={r2_power:.6f}")

                # 保存残差序列（只包含 t>0 的点，但需要与原始长度一致，填充 NaN for t=0）
                full_residuals = np.full_like(t, np.nan)
                full_residuals[mask] = residuals_power
                derived_series_list.append({
                    "experiment_id": exp_id,
                    "name": f"residual_powerlaw_{exp_id}",
                    "values": full_residuals.tolist(),
                    "source_name": f"q - c*t^p (c={c_opt:.6f}, p={p_opt:.6f})",
                    "provenance": "generated data processor: step_031_analyze_data",
                    "description": "幂律拟合残差，t=0 处为 NaN"
                })

            # 绘制幂律拟合图（无论曲线拟合是否成功都绘制）
            fig_pow, ax_pow = plt.subplots(figsize=(6,4))
            ax_pow.plot(t_pos, q_pos, 'o', markersize=2, label='data')
            t_fit = np.linspace(t_pos.min(), t_pos.max(), 200)
            q_fit = power_law(t_fit, c_opt, p_opt)
            ax_pow.plot(t_fit, q_fit, 'r-', label=f'fit: q={c_opt:.4f}*t^{p_opt:.4f}')
            ax_pow.set_xlabel('t')
            ax_pow.set_ylabel('q')
            ax_pow.set_title(f'Power law fit ({exp_id})')
            ax_pow.legend()
            fname_pow = output_dir / f"powerlaw_fit_{exp_id}.png"
            fig_pow.savefig(fname_pow)
            plt.close(fig_pow)
            figures.append(str(fname_pow))

            # 绘制残差图（仅在成功拟合且有残差时）
            if power_fit_succeeded and residuals_power is not None and len(residuals_power) > 0:
                fig_res_pow, ax_res_pow = plt.subplots(figsize=(6,4))
                ax_res_pow.plot(t_pos, residuals_power, 'o', markersize=2, label='residual')
                ax_res_pow.axhline(0, color='gray', linestyle='--')
                ax_res_pow.set_xlabel('t')
                ax_res_pow.set_ylabel('residual')
                ax_res_pow.set_title(f'Power law residuals ({exp_id})')
                ax_res_pow.legend()
                fname_res_pow = output_dir / f"powerlaw_residuals_{exp_id}.png"
                fig_res_pow.savefig(fname_res_pow)
                plt.close(fig_res_pow)
                figures.append(str(fname_res_pow))

    # 汇总 observation
    obs_header = f"对实验 {target_ids} 进行分析。使用 Savitzky-Golay 滤波器 (window={window_length}, polyorder={polyorder}) 从 q(t) 估计速度和加速度。\n"
    obs = obs_header + "\n".join(obs_parts)
    obs += "\n已生成速度-时间图和加速度-时间图，以及必要的拟合残差图。派生序列包括速度、加速度、线性拟合残差（exp_02）、幂律拟合残差（exp_03）。"

    return {
        "observation": obs,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics
    }
