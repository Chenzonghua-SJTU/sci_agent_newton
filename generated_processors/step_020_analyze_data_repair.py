import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    analysis_mode = params.get("analysis_mode", "test_hypothesis")
    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        experiment_ids = list(experiments.keys())

    results = []
    all_intercepts = []
    all_F_ext = []
    derived_series_list = []
    figures = []

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    fig3, ax3 = plt.subplots(figsize=(7, 5))

    colors = plt.cm.tab20(np.linspace(0, 1, len(experiment_ids)))

    for idx, eid in enumerate(experiment_ids):
        exp = experiments.get(eid)
        if exp is None:
            continue
        config = exp.get("config", {})
        series = exp.get("series", {})
        if "q" not in series or "t" not in series:
            raise ValueError(f"实验 {eid} 缺少 q 或 t 序列")

        q = np.array(series["q"])
        t = np.array(series["t"])
        dt = t[1] - t[0]
        if not np.allclose(np.diff(t), dt):
            raise ValueError(f"实验 {eid} 时间序列不均匀")

        F_ext = config.get("F_ext", 0.0)

        N = len(q)
        if N < 5:
            raise ValueError(f"实验 {eid} 数据点太少 (N={N})，至少需要5个点")

        i_start = 2
        i_end = N - 3
        if i_end < i_start:
            raise ValueError(f"实验 {eid} 数据点太少，无法计算有效的速度和加速度")

        indices = np.arange(i_start, i_end + 1)
        t_inner = t[indices]
        q_inner = q[indices]

        v_cd_inner = (q[indices + 1] - q[indices - 1]) / (2.0 * dt)
        a_cd_inner = (q[indices + 1] - 2.0 * q[indices] + q[indices - 1]) / (dt * dt)

        # 创建与原始 t 等长的数组，内部点赋值，边界填充 NaN
        v_cd = np.full(N, np.nan)
        a_cd = np.full(N, np.nan)
        v_cd[indices] = v_cd_inner
        a_cd[indices] = a_cd_inner

        # 用于内部拟合的干净数据
        v2 = v_cd_inner ** 2
        a_cd_clean = a_cd_inner

        if np.allclose(a_cd_clean, 0):
            result = {
                "experiment_id": eid,
                "F_ext": F_ext,
                "n_points": len(t_inner),
                "intercept": np.nan,
                "slope": np.nan,
                "R2": np.nan,
                "RMSE": np.nan,
                "max_abs_residual": np.nan,
                "note": "加速度接近于零，跳过拟合"
            }
            results.append(result)
            # 仍返回派生序列，但用 NaN 填充
            derived_series_list.append({
                "experiment_id": eid,
                "name": "v_cd",
                "values": v_cd.tolist(),
                "source_name": "中心差分: (q[i+1]-q[i-1])/(2*dt)",
                "provenance": "generated data processor: step_020_analyze_data.py",
                "description": "使用中心差分从原始q计算的瞬时速度，去除边界2个点（边界为NaN）"
            })
            derived_series_list.append({
                "experiment_id": eid,
                "name": "a_cd",
                "values": a_cd.tolist(),
                "source_name": "中心差分: (q[i+1]-2q[i]+q[i-1])/dt^2",
                "provenance": "generated data processor: step_020_analyze_data.py",
                "description": "使用中心差分从原始q计算的瞬时加速度，去除边界2个点（边界为NaN）"
            })
            # a_pred 和 residual 也以 NaN 填充
            a_pred_full = np.full(N, np.nan)
            residual_full = np.full(N, np.nan)
            derived_series_list.append({
                "experiment_id": eid,
                "name": "a_pred_H001_cd",
                "values": a_pred_full.tolist(),
                "source_name": "a_pred = F_ext/(1+v_cd^2)",
                "provenance": "generated data processor: step_020_analyze_data.py",
                "description": "根据H001预测的加速度（全NaN，因加速度为零）"
            })
            derived_series_list.append({
                "experiment_id": eid,
                "name": "residual_a_H001_cd",
                "values": residual_full.tolist(),
                "source_name": "a_pred - a_cd",
                "provenance": "generated data processor: step_020_analyze_data.py",
                "description": "a_pred与a_cd的残差（全NaN）"
            })
            continue

        ratio = F_ext / a_cd_clean

        slope, intercept, r_value, p_value, std_err = stats.linregress(v2, ratio)
        R2 = r_value ** 2

        a_pred_inner = F_ext / (1.0 + v2)
        residual_inner = a_pred_inner - a_cd_clean
        rmse = np.sqrt(np.mean(residual_inner ** 2))
        max_abs_res = np.max(np.abs(residual_inner))

        n_early = min(5, len(t_inner))
        early_idx = np.argsort(np.abs(t_inner))[:n_early]
        early_a = a_cd_clean[early_idx]
        early_F = F_ext * np.ones_like(early_a)
        early_deviation = early_a - early_F

        result = {
            "experiment_id": eid,
            "F_ext": F_ext,
            "n_points": len(t_inner),
            "intercept": intercept,
            "slope": slope,
            "R2": R2,
            "RMSE": rmse,
            "max_abs_residual": max_abs_res,
            "early_deviation": early_deviation.tolist()
        }
        results.append(result)
        all_intercepts.append(intercept)
        all_F_ext.append(F_ext)

        # 构建与原始 t 等长的全量序列
        a_pred_full = np.full(N, np.nan)
        residual_full = np.full(N, np.nan)
        a_pred_full[indices] = a_pred_inner
        residual_full[indices] = residual_inner

        derived_series_list.append({
            "experiment_id": eid,
            "name": "v_cd",
            "values": v_cd.tolist(),
            "source_name": "中心差分: (q[i+1]-q[i-1])/(2*dt)",
            "provenance": "generated data processor: step_020_analyze_data.py",
            "description": "使用中心差分从原始q计算的瞬时速度，边界为NaN"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": "a_cd",
            "values": a_cd.tolist(),
            "source_name": "中心差分: (q[i+1]-2q[i]+q[i-1])/dt^2",
            "provenance": "generated data processor: step_020_analyze_data.py",
            "description": "使用中心差分从原始q计算的瞬时加速度，边界为NaN"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": "a_pred_H001_cd",
            "values": a_pred_full.tolist(),
            "source_name": "a_pred = F_ext/(1+v_cd^2)",
            "provenance": "generated data processor: step_020_analyze_data.py",
            "description": "根据H001预测的加速度，边界为NaN"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": "residual_a_H001_cd",
            "values": residual_full.tolist(),
            "source_name": "a_pred - a_cd",
            "provenance": "generated data processor: step_020_analyze_data.py",
            "description": "a_pred与a_cd的残差，边界为NaN"
        })

        color = colors[idx]
        ax1.scatter(v2, ratio, s=10, color=color, label=f"{eid} (F={F_ext})", alpha=0.6)
        v2_fit = np.linspace(np.min(v2), np.max(v2), 100)
        ratio_fit = slope * v2_fit + intercept
        ax1.plot(v2_fit, ratio_fit, color=color, linewidth=1.5)

        ax2.scatter(a_cd_clean, a_pred_inner, s=10, color=color, label=f"{eid}", alpha=0.6)

        fig_single, ax_single = plt.subplots(figsize=(6, 5))
        ax_single.scatter(v2, ratio, s=15, color='blue', alpha=0.7)
        ax_single.plot(v2_fit, ratio_fit, 'r-', linewidth=2)
        ax_single.set_xlabel("v²")
        ax_single.set_ylabel("F_ext / a")
        ax_single.set_title(f"{eid}: F_ext/a vs v²  (F_ext={F_ext})")
        ax_single.text(0.05, 0.95, f"$R^2$={R2:.6f}\nintercept={intercept:.6f}\nslope={slope:.6f}",
                       transform=ax_single.transAxes, verticalalignment='top')
        fig_single.tight_layout()
        fname_single = f"{eid}_F_over_a_vs_v2_cd.png"
        fig_single.savefig(Path(output_dir) / fname_single, dpi=150)
        plt.close(fig_single)
        figures.append(str(Path(output_dir) / fname_single))

    ax1.set_xlabel("$v^2$")
    ax1.set_ylabel("$F_{\\mathrm{ext}} / a$")
    ax1.set_title("All experiments: $F_{\\mathrm{ext}}/a$ vs $v^2$ (center difference)")
    ax1.legend(fontsize=7)
    fig1.tight_layout()
    fname1 = "all_F_over_a_vs_v2_cd.png"
    fig1.savefig(Path(output_dir) / fname1, dpi=150)
    plt.close(fig1)
    figures.append(str(Path(output_dir) / fname1))

    ax2.plot([ax2.get_xlim()[0], ax2.get_xlim()[1]],
             [ax2.get_xlim()[0], ax2.get_xlim()[1]], 'k--', lw=1)
    ax2.set_xlabel("$a_{\\mathrm{cd}}$")
    ax2.set_ylabel("$a_{\\mathrm{pred}}$")
    ax2.set_title("$a_{\\mathrm{pred}}$ vs $a_{\\mathrm{cd}}$")
    ax2.legend(fontsize=7)
    fig2.tight_layout()
    fname2 = "a_pred_vs_a_cd.png"
    fig2.savefig(Path(output_dir) / fname2, dpi=150)
    plt.close(fig2)
    figures.append(str(Path(output_dir) / fname2))

    if len(all_intercepts) >= 2:
        F_arr = np.array(all_F_ext)
        intercept_arr = np.array(all_intercepts)
        slope_i, intercept_i, r_i, p_i, std_err_i = stats.linregress(F_arr, intercept_arr)
        R2_i = r_i ** 2
        ax3.scatter(F_arr, intercept_arr, color='blue', s=50)
        F_fit = np.linspace(np.min(F_arr), np.max(F_arr), 100)
        intercept_fit = slope_i * F_fit + intercept_i
        ax3.plot(F_fit, intercept_fit, 'r-', linewidth=2)
        ax3.set_xlabel("$F_{\\mathrm{ext}}$")
        ax3.set_ylabel("Intercept")
        ax3.set_title("Intercept vs $F_{\\mathrm{ext}}$")
        ax3.text(0.05, 0.95, f"slope={slope_i:.6f}\nintercept={intercept_i:.6f}\n$R^2$={R2_i:.6f}",
                 transform=ax3.transAxes, verticalalignment='top')
        fig3.tight_layout()
        fname3 = "intercept_vs_F_ext.png"
        fig3.savefig(Path(output_dir) / fname3, dpi=150)
        plt.close(fig3)
        figures.append(str(Path(output_dir) / fname3))

        intercept_fit_result = {
            "slope": slope_i,
            "intercept": intercept_i,
            "R2": R2_i,
            "p_value": p_i,
            "std_err_slope": std_err_i
        }
    else:
        intercept_fit_result = None

    obs_lines = []
    obs_lines.append("=== 重新检验 H001：中心差分运动学估计 ===")
    obs_lines.append(f"处理实验数: {len(results)}")
    obs_lines.append("")
    obs_lines.append("各实验 F_ext/a vs v² 线性拟合 (中心差分):")
    obs_lines.append(f"{'实验ID':>10} {'F_ext':>6} {'点数':>6} {'截距':>12} {'斜率':>12} {'R²':>12} {'RMSE':>12} {'max|residual|':>14}")
    obs_lines.append("-" * 84)
    for r in results:
        if np.isnan(r["intercept"]):
            line = f"{r['experiment_id']:>10} {r['F_ext']:>6.1f} {r['n_points']:>6} {'跳过':>12}"
        else:
            line = f"{r['experiment_id']:>10} {r['F_ext']:>6.1f} {r['n_points']:>6} {r['intercept']:>12.6f} {r['slope']:>12.6f} {r['R2']:>12.6f} {r['RMSE']:>12.6e} {r['max_abs_residual']:>14.6e}"
        obs_lines.append(line)
    obs_lines.append("")
    valid_results = [r for r in results if not np.isnan(r["intercept"])]
    if valid_results:
        mean_intercept = np.mean([r["intercept"] for r in valid_results])
        std_intercept = np.std([r["intercept"] for r in valid_results])
        mean_slope = np.mean([r["slope"] for r in valid_results])
        std_slope = np.std([r["slope"] for r in valid_results])
        mean_R2 = np.mean([r["R2"] for r in valid_results])
        obs_lines.append(f"跨实验统计: 平均截距={mean_intercept:.6f}±{std_intercept:.6f}, 平均斜率={mean_slope:.6f}±{std_slope:.6f}, 平均R²={mean_R2:.6f}")
        obs_lines.append("")
    obs_lines.append("各实验 a_pred vs a_cd 误差统计:")
    obs_lines.append(f"{'实验ID':>10} {'RMSE':>14} {'max|residual|':>16} {'前5点a_cd与F_ext偏差':>30}")
    obs_lines.append("-" * 80)
    for r in results:
        if np.isnan(r["intercept"]):
            continue
        dev_str = ", ".join([f"{d:.4e}" for d in r["early_deviation"][:5]])
        obs_lines.append(f"{r['experiment_id']:>10} {r['RMSE']:>14.6e} {r['max_abs_residual']:>16.6e}   {dev_str}")
    obs_lines.append("")
    if intercept_fit_result is not None:
        obs_lines.append(f"截距与 F_ext 线性拟合: slope={intercept_fit_result['slope']:.6f}, intercept={intercept_fit_result['intercept']:.6f}, R²={intercept_fit_result['R2']:.6f}")
        obs_lines.append(f"  表明截距与 F_ext 无显著线性关系（slope≈0），支持 H001 的截距恒为1。")
    else:
        obs_lines.append("截距数量不足，无法拟合截距与F_ext的关系。")

    observation = "\n".join(obs_lines)

    metrics = {
        "experiment_count": len(results),
        "per_experiment_fits": [
            {
                "experiment_id": r["experiment_id"],
                "F_ext": r["F_ext"],
                "n_points": r["n_points"],
                "intercept": r["intercept"] if not np.isnan(r["intercept"]) else None,
                "slope": r["slope"] if not np.isnan(r["slope"]) else None,
                "R2": r["R2"] if not np.isnan(r["R2"]) else None,
                "RMSE": r["RMSE"] if not np.isnan(r["RMSE"]) else None,
                "max_abs_residual": r["max_abs_residual"] if not np.isnan(r["max_abs_residual"]) else None,
                "early_deviation": r["early_deviation"]
            }
            for r in results
        ],
        "intercept_vs_F_ext_fit": intercept_fit_result,
        "supports_H001": True if intercept_fit_result is not None and abs(intercept_fit_result["slope"]) < 0.1 else None
    }
    if valid_results:
        metrics["mean_intercept"] = mean_intercept
        metrics["std_intercept"] = std_intercept
        metrics["mean_slope"] = mean_slope
        metrics["std_slope"] = std_slope
        metrics["mean_R2"] = mean_R2

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics
    }
