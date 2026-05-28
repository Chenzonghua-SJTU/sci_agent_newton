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
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _estimate_kinematics(t: np.ndarray, q: np.ndarray, window: int = 5, polyorder: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用Savitzky-Golay滤波和中心差分估计光滑位置、速度、加速度。
    返回 (q_smooth, v, a)，保留原始长度，边界点因滤波和差分可能有较大误差。
    """
    if len(t) < window:
        return q, np.full_like(t, np.nan), np.full_like(t, np.nan)
    dt = t[1] - t[0]  # 假设均匀采样
    q_smooth = savgol_filter(q, window_length=window, polyorder=polyorder)
    v = np.gradient(q_smooth, dt)
    a = np.gradient(v, dt)
    return q_smooth, v, a

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")
    analysis_mode = params.get("analysis_mode", "")
    experiment_ids = params.get("experiment_ids", [])
    expected_outputs = params.get("expected_outputs", [])

    if analysis_mode != "test_hypothesis":
        raise ValueError("This code only handles test_hypothesis mode")

    # 根据实验ID列表过滤
    if not experiment_ids:
        experiment_ids = list(experiments.keys())
    
    # 存储结果
    fits = []
    free_checks = []
    residual_series_dict = {}  # {exp_id: {'t': list, 'residual': list}}
    figures = []
    derived_series_list = []
    all_experiments_analyzed = True

    # 为每个实验计算运动学并分析
    for exp_id in experiment_ids:
        if exp_id not in experiments:
            raise ValueError(f"Experiment {exp_id} not found in payload")
        exp = experiments[exp_id]
        config = exp["config"]
        series = exp["series"]
        available = exp.get("available_series", [])

        # 获取控制参数
        force_field_type = config.get("force_field_type", "")
        F_ext = config.get("F_ext", 0.0)  # 官方外力
        # 获取 t 和 q 序列
        t = np.array(series["t"], dtype=float)
        q = np.array(series["q"], dtype=float)
        n_total = len(t)

        # 运动学估计
        q_smooth, v_est, a_est = _estimate_kinematics(t, q, window=5, polyorder=2)
        # 排除首尾各2个边界点（由于savgol和梯度边界失真）
        boundary = 2
        if n_total <= 2 * boundary:
            inner_slice = slice(0, n_total)
        else:
            inner_slice = slice(boundary, n_total - boundary)
        t_inner = t[inner_slice]
        v_inner = v_est[inner_slice]
        a_inner = a_est[inner_slice]

        # 检查自由实验
        if force_field_type == "free" or abs(F_ext) < 1e-12:
            mean_a = float(np.mean(a_inner))
            std_a = float(np.std(a_inner))
            max_abs_a = float(np.max(np.abs(a_inner)))
            free_checks.append({
                "experiment_id": exp_id,
                "mean_a": mean_a,
                "std_a": std_a,
                "max_abs_a": max_abs_a
            })
            continue

        # 恒外力实验
        # 要求F_ext不为零，且a_est不能太小以免除零
        valid_mask = np.abs(a_inner) > 1e-12
        if np.sum(valid_mask) < 5:
            # 如果有效点数太少，跳过拟合但记录
            fits.append({
                "experiment_id": exp_id,
                "F_ext": F_ext,
                "n_inner": len(t_inner),
                "n_valid": 0,
                "error": "Too few valid points (|a| > 1e-12)"
            })
            continue

        v2 = v_inner[valid_mask] ** 2
        F_over_a = F_ext / a_inner[valid_mask]
        t_valid = t_inner[valid_mask]

        # 线性回归 F_ext/a vs v^2
        slope, intercept, r_value, p_value, std_err = stats.linregress(v2, F_over_a)
        R2 = r_value ** 2
        predicted = intercept + slope * v2
        residuals = F_over_a - predicted
        rmse = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))
        resid_mean = float(np.mean(residuals))
        resid_std = float(np.std(residuals))
        max_abs_resid = float(np.max(np.abs(residuals)))

        fit_result = {
            "experiment_id": exp_id,
            "F_ext": float(F_ext),
            "n_inner": len(t_inner),
            "n_valid": int(np.sum(valid_mask)),
            "intercept": float(intercept),
            "slope": float(slope),
            "R2": float(R2),
            "RMSE": rmse,
            "MAE": mae,
            "resid_mean": resid_mean,
            "resid_std": resid_std,
            "max_abs_resid": max_abs_resid
        }
        fits.append(fit_result)

        # 保存残差时间序列（只保存有效点）
        residual_series_dict[exp_id] = {
            "t": t_valid.tolist(),
            "residual": residuals.tolist()
        }

        # 构建与原始 t 等长的残差序列，非有效点设为 NaN
        full_residual = np.full(n_total, np.nan)
        inner_indices_valid = np.where(valid_mask)[0] + boundary  # 原始索引
        full_residual[inner_indices_valid] = residuals

        derived_series_list.append({
            "experiment_id": exp_id,
            "name": "residual_H001",
            "values": full_residual.tolist(),
            "source_name": "F_ext/a - (intercept + slope * v^2) from linear fit",
            "provenance": "generated data processor: step_006_analyze_data_repair.py (current run)",
            "description": "残差序列 (F_ext/a - fit) 用于H001一致性检查"
        })

        # 绘制散点图和拟合线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # 左图：F_ext/a vs v^2
        ax1.scatter(v2, F_over_a, s=10, alpha=0.6, label='Data')
        v2_sorted = np.sort(v2)
        ax1.plot(v2_sorted, intercept + slope * v2_sorted, 'r-', label=f'Fit: y={intercept:.4f}+{slope:.4f}x')
        ax1.set_xlabel(r'$v^2$')
        ax1.set_ylabel(r'$F_{ext}/a$')
        ax1.set_title(f'{exp_id} (F_ext={F_ext})')
        ax1.legend()
        ax1.text(0.05, 0.95, f'R²={R2:.6f}\nRMSE={rmse:.6f}', transform=ax1.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        # 右图：残差时间序列
        ax2.plot(t_valid, residuals, 'o-', markersize=2, linewidth=0.5)
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_xlabel('t')
        ax2.set_ylabel('Residual')
        ax2.set_title(f'Residual Time Series (RMSE={rmse:.6f})')
        fig.tight_layout()
        figname = f"{exp_id}_H001_linear_fit.png"
        figpath = str(Path(output_dir) / figname)
        fig.savefig(figpath, dpi=150)
        plt.close(fig)
        figures.append(figpath)

    # 自由实验输出
    free_text_lines = []
    for fc in free_checks:
        free_text_lines.append(
            f"  {fc['experiment_id']}: mean_a={fc['mean_a']:.4e}, std_a={fc['std_a']:.4e}, max|a|={fc['max_abs_a']:.4e}"
        )
    free_text = "\n".join(free_text_lines)

    # 恒外力拟合汇总
    if fits:
        fits_text_lines = []
        header = f"{'实验':<12} {'F_ext':<8} {'n_valid':<8} {'截距':<12} {'斜率':<12} {'R²':<10} {'RMSE':<12} {'残差mean':<14} {'残差std':<14} {'max|残差|':<14}"
        fits_text_lines.append(header)
        fits_text_lines.append("-" * len(header))
        for f in fits:
            if "error" in f:
                line = f"{f['experiment_id']:<12} {f.get('F_ext', '?'):<8} {f['n_valid']:<8} {'NA':<12} {'NA':<12} {'NA':<10} {'NA':<12} {'NA':<14} {'NA':<14} {'NA':<14}"
            else:
                line = f"{f['experiment_id']:<12} {f['F_ext']:<8.1f} {f['n_valid']:<8} {f['intercept']:<12.6f} {f['slope']:<12.6f} {f['R2']:<10.6f} {f['RMSE']:<12.6f} {f['resid_mean']:<14.4e} {f['resid_std']:<14.4e} {f['max_abs_resid']:<14.4e}"
            fits_text_lines.append(line)
        fits_text = "\n".join(fits_text_lines)

        # 跨实验一致性统计
        intercepts = [f['intercept'] for f in fits if 'intercept' in f]
        slopes = [f['slope'] for f in fits if 'slope' in f]
        r2s = [f['R2'] for f in fits if 'R2' in f]
        mean_intercept = np.mean(intercepts) if intercepts else np.nan
        std_intercept = np.std(intercepts) if intercepts else np.nan
        mean_slope = np.mean(slopes) if slopes else np.nan
        std_slope = np.std(slopes) if slopes else np.nan
        mean_r2 = np.mean(r2s) if r2s else np.nan
        summary_lines = [
            "",
            f"跨实验统计: 平均截距={mean_intercept:.6f} ± {std_intercept:.6f}, "
            f"平均斜率={mean_slope:.6f} ± {std_slope:.6f}, 平均R²={mean_r2:.6f}"
        ]
        summary_text = "\n".join(summary_lines)
    else:
        fits_text = "（无恒外力实验可供拟合）"
        summary_text = ""
        mean_intercept = mean_slope = mean_r2 = np.nan

    # 构建observation
    obs_parts = [
        f"=== 检验候选规律 H001: F_ext / a = 1 + v² ===",
        f"处理实验: {experiment_ids}",
        f"运动学估计: savgol滤波(window=5, polyorder=2), 中心差分, 排除首尾各{boundary}个边界点",
        "",
        "自由实验加速度检查:",
        free_text,
        "",
        "恒外力实验线性拟合结果 (F_ext/a vs v²):",
        fits_text,
        summary_text
    ]
    observation = "\n".join(obs_parts)

    # 构建metrics
    metrics = {
        "free_experiment_checks": free_checks,
        "forced_experiment_fits": fits,
        "mean_intercept": float(np.nan_to_num(mean_intercept)),
        "std_intercept": float(np.nan_to_num(std_intercept)),
        "mean_slope": float(np.nan_to_num(mean_slope)),
        "std_slope": float(np.nan_to_num(std_slope)),
        "mean_R2": float(np.nan_to_num(mean_r2)),
        "experiment_ids": experiment_ids,
        "residual_series_available": list(residual_series_dict.keys())
    }
    # 判断是否支持H001：截距和斜率是否接近1（容许0.05偏差）
    if not np.isnan(mean_intercept) and not np.isnan(mean_slope):
        supports = (abs(mean_intercept - 1.0) < 0.05) and (abs(mean_slope - 1.0) < 0.05)
        metrics["supports_H001"] = bool(supports)
    else:
        metrics["supports_H001"] = False

    result = {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics
    }
    return result
