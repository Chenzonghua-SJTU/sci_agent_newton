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
import scipy
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, r2_score

def process(payload: dict) -> dict:
    action = payload.get("action")
    parameters = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = Path(payload.get("output_dir", "."))

    # 参数解析
    analysis_mode = parameters.get("analysis_mode", "")
    if analysis_mode != "maintain_ledger":
        return {
            "observation": "任务终止：非 main_ledger 模式，不处理。",
            "derived_series": [],
            "observations": [],
            "figures": [],
            "metrics": {"task_rejected": True}
        }

    # 获取实验 ID 列表
    exp_ids = parameters.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(experiments.keys())
    # 只保留恒外力实验（F_ext non-zero），但参数已经指定，不过做保护
    valid_ids = []
    for eid in exp_ids:
        if eid not in experiments:
            continue
        cfg = experiments[eid].get("config", {})
        F_ext = cfg.get("F_ext", 0.0)
        if F_ext != 0.0:
            valid_ids.append(eid)
        else:
            # 跳过零外力实验
            pass

    if not valid_ids:
        return {
            "observation": "没有非零恒外力实验可处理。",
            "derived_series": [],
            "observations": [],
            "figures": [],
            "metrics": {"experiments_processed": 0}
        }

    # 准备数据
    merged_v = []
    merged_a_norm = []
    per_exp_results = {}  # eid -> (F_ext, v, a_norm, slope, intercept, r2, rmse)
    used_series = set()

    for eid in valid_ids:
        exp_data = experiments[eid]
        series = exp_data.get("series", {})
        cfg = exp_data.get("config", {})
        F_ext = cfg.get("F_ext", 0.0)
        if F_ext == 0.0:
            continue

        # 优先使用已存在的 a_norm 和 v_central
        if "a_norm" in series and "v_central" in series:
            a_norm = np.array(series["a_norm"])
            v = np.array(series["v_central"])
            used_series.add(f"{eid}:a_norm")
            used_series.add(f"{eid}:v_central")
        elif "a_central" in series and "v_central" in series:
            a_central = np.array(series["a_central"])
            v = np.array(series["v_central"])
            a_norm = a_central / F_ext
            used_series.add(f"{eid}:a_central")
            used_series.add(f"{eid}:v_central")
        else:
            # 尝试从 q 和 t 计算，但为了简洁，跳过并记录
            continue

        # 去掉可能包含 NaN 的点
        mask = ~(np.isnan(a_norm) | np.isnan(v))
        a_norm = a_norm[mask]
        v = v[mask]
        if len(a_norm) < 3:
            continue

        # 线性回归
        slope, intercept, r_value, p_value, std_err = linregress(v, a_norm)
        r2 = r_value ** 2
        predicted = slope * v + intercept
        rmse = math.sqrt(mean_squared_error(a_norm, predicted))

        per_exp_results[eid] = {
            "F_ext": F_ext,
            "v": v.tolist(),
            "a_norm": a_norm.tolist(),
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "rmse": rmse,
            "n_points": len(a_norm)
        }
        merged_v.extend(v.tolist())
        merged_a_norm.extend(a_norm.tolist())

    if not per_exp_results:
        return {
            "observation": "没有任何实验成功提取 a_norm 和 v_central。",
            "derived_series": [],
            "observations": [],
            "figures": [],
            "metrics": {"experiments_processed": 0}
        }

    # 合并回归
    merged_v = np.array(merged_v)
    merged_a_norm = np.array(merged_a_norm)
    slope_comb, intercept_comb, r_value_comb, _, _ = linregress(merged_v, merged_a_norm)
    r2_comb = r_value_comb ** 2
    predicted_comb = slope_comb * merged_v + intercept_comb
    rmse_comb = math.sqrt(mean_squared_error(merged_a_norm, predicted_comb))

    # 坍缩判断：如果合并回归 R² 较高（比如 >0.99）且各实验斜率差异小，则坍缩
    # 简单判断：各实验斜率的标准差与平均绝对值之比
    slopes = [v["slope"] for v in per_exp_results.values()]
    slope_std = statistics.stdev(slopes) if len(slopes) > 1 else 0.0
    slope_mean_abs = statistics.mean([abs(s) for s in slopes])
    cv = slope_std / slope_mean_abs if slope_mean_abs > 0 else 100
    if r2_comb > 0.99 and cv < 0.1:
        collapse_observed = True
    else:
        collapse_observed = False

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(per_exp_results)))
    for idx, (eid, res) in enumerate(per_exp_results.items()):
        label = f"{eid} (F_ext={res['F_ext']:.1f})"
        ax.scatter(res["v"], res["a_norm"], s=10, color=colors[idx], alpha=0.7, label=label)
        # 绘制回归线
        v_range = np.linspace(min(res["v"]), max(res["v"]), 50)
        a_fit = res["slope"] * v_range + res["intercept"]
        ax.plot(v_range, a_fit, color=colors[idx], linestyle='--', linewidth=1)
    ax.set_xlabel('v_central')
    ax.set_ylabel('a_norm = a_central / F_ext')
    ax.set_title('a_norm vs v_central for constant F_ext experiments')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig_path = output_dir / "a_norm_vs_v.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)

    # 构造 observations 列表
    obs_list = []
    # 第一条：总体汇总
    summary_parts = [
        f"对 {len(per_exp_results)} 个恒外力实验（{', '.join(per_exp_results.keys())}）计算 a_norm = a_central / F_ext。",
        f"每个实验的 a_norm vs v_central 线性回归结果如下："
    ]
    for eid, res in per_exp_results.items():
        summary_parts.append(
            f"{eid}: F_ext={res['F_ext']:.1f}, slope={res['slope']:.4f}, intercept={res['intercept']:.4f}, R²={res['r2']:.4f}, RMSE={res['rmse']:.4f}"
        )
    summary_parts.append(
        f"所有实验合并回归：slope={slope_comb:.4f}, intercept={intercept_comb:.4f}, R²={r2_comb:.4f}, RMSE={rmse_comb:.4f}"
    )
    if collapse_observed:
        summary_parts.append("观察到跨外力坍缩：不同 F_ext 的 a_norm vs v 曲线基本重合。")
    else:
        summary_parts.append(f"未观察到跨外力坍缩：合并回归 R²={r2_comb:.4f}，各实验斜率 CV={cv:.4f}，不同外力下曲线分离。")

    source_refs = list(used_series)
    metrics = {
        "experiments_processed": len(per_exp_results),
        "combined_R2": r2_comb,
        "combined_RMSE": rmse_comb,
        "combined_slope": slope_comb,
        "combined_intercept": intercept_comb,
        "slope_CV": cv,
        "collapse_observed": collapse_observed
    }
    # 为每个实验单独记录指标
    for eid, res in per_exp_results.items():
        metrics[f"{eid}_slope"] = res["slope"]
        metrics[f"{eid}_intercept"] = res["intercept"]
        metrics[f"{eid}_R2"] = res["r2"]
        metrics[f"{eid}_RMSE"] = res["rmse"]

    obs_list.append({
        "summary": " ".join(summary_parts),
        "source_data_refs": source_refs,
        "metrics": metrics
    })

    # 构建 observation 字符串（简洁版给 LLM）
    # 使用第一行 summary
    observation = summary_parts[0] + " " + summary_parts[-1]

    # derived_series: 由于 a_norm 可能已经存在，为避免重复，不返回。但可以返回一个空的列表
    # 如果需要记录新系列，但这里不产生新系列（已有）
    derived_series = []

    return {
        "observation": observation,
        "derived_series": derived_series,
        "observations": obs_list,
        "figures": [str(fig_path)],
        "metrics": metrics
    }
