import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn import linear_model, metrics as skmetrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process(payload: dict) -> dict:
    # ---------- 参数提取 ----------
    action = payload.get("action", "")
    parameters = payload.get("parameters", {})
    experiments_data = payload.get("experiments", {})
    # existing_observations = payload.get("observations", [])  # 已有OBS，不直接使用
    # existing_validations = payload.get("validations", [])
    output_dir = Path(payload.get("output_dir", "."))

    experiment_ids = parameters.get("experiment_ids", None)
    if experiment_ids is None:
        experiment_ids = list(experiments_data.keys())

    # ---------- 检查任务要求：只处理给定实验，但必须包含所有15个 ----------
    # 确保实验存在
    for eid in experiment_ids:
        if eid not in experiments_data:
            raise ValueError(f"Experiment {eid} not found in payload['experiments']")

    # ---------- 收集回归结果 ----------
    regression_results = []  # list of dict per experiment

    for eid in experiment_ids:
        exp = experiments_data[eid]
        config = exp["config"]
        series = exp["series"]
        available = exp.get("available_series", [])

        # 检查a和v是否可用
        if "a" not in available or "v" not in available:
            raise ValueError(f"Experiment {eid}: a or v series not available")
        if "a" not in series or "v" not in series:
            raise ValueError(f"Experiment {eid}: a or v not in series dict")

        v_vals = np.array(series["v"], dtype=float)
        a_vals = np.array(series["a"], dtype=float)

        # 线性回归 a ~ v
        slope, intercept, r_value, p_value, std_err = stats.linregress(v_vals, a_vals)
        r2 = r_value ** 2

        # 提取F_ext和v0
        F_ext = float(config.get("F_ext", 0.0))
        v0 = float(config.get("initial_v", 0.0))
        force_field_type = config.get("force_field_type", "unknown")

        regression_results.append({
            "experiment_id": eid,
            "F_ext": F_ext,
            "v0": v0,
            "force_field_type": force_field_type,
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "p_value": p_value,
            "std_err": std_err
        })

    # ---------- 构建observations ----------
    new_observations = []

    # 1. 每个实验一条OBS
    for res in regression_results:
        eid = res["experiment_id"]
        obs = {
            "summary": f"{eid} a-v线性回归: 斜率={res['slope']:.6f}, 截距={res['intercept']:.6f}, R²={res['r2']:.6f}",
            "source_data_refs": [f"{eid}:v", f"{eid}:a"],
            "metrics": {
                "slope": res["slope"],
                "intercept": res["intercept"],
                "r2": res["r2"],
                "p_value": res["p_value"],
                "F_ext": res["F_ext"],
                "v0": res["v0"],
                "force_field_type": res["force_field_type"]
            }
        }
        new_observations.append(obs)

    # 2. 斜率和截距与F_ext、v0的相关系数（全部15个实验）
    slopes = np.array([r["slope"] for r in regression_results])
    intercepts = np.array([r["intercept"] for r in regression_results])
    F_exts = np.array([r["F_ext"] for r in regression_results])
    v0s = np.array([r["v0"] for r in regression_results])

    # 计算相关系数
    corr_slope_Fext, p_slope_Fext = stats.pearsonr(slopes, F_exts)
    corr_slope_v0, p_slope_v0 = stats.pearsonr(slopes, v0s)
    corr_intercept_Fext, p_intercept_Fext = stats.pearsonr(intercepts, F_exts)
    corr_intercept_v0, p_intercept_v0 = stats.pearsonr(intercepts, v0s)

    corr_obs = {
        "summary": (f"全部15个实验的回归参数相关系数: "
                    f"斜率与F_ext r={corr_slope_Fext:.4f} (p={p_slope_Fext:.4e}), "
                    f"斜率与v0 r={corr_slope_v0:.4f} (p={p_slope_v0:.4e}), "
                    f"截距与F_ext r={corr_intercept_Fext:.4f} (p={p_intercept_Fext:.4e}), "
                    f"截距与v0 r={corr_intercept_v0:.4f} (p={p_intercept_v0:.4e})"),
        "source_data_refs": [f"{eid}:v" for eid in experiment_ids] + [f"{eid}:a" for eid in experiment_ids],
        "metrics": {
            "corr_slope_Fext": corr_slope_Fext,
            "p_slope_Fext": p_slope_Fext,
            "corr_slope_v0": corr_slope_v0,
            "p_slope_v0": p_slope_v0,
            "corr_intercept_Fext": corr_intercept_Fext,
            "p_intercept_Fext": p_intercept_Fext,
            "corr_intercept_v0": corr_intercept_v0,
            "p_intercept_v0": p_intercept_v0,
            "n_experiments": len(regression_results)
        }
    }
    new_observations.append(corr_obs)

    # 3. 自由场实验回归验证（exp_01,04,07）
    free_exp_ids = ["exp_01", "exp_04", "exp_07"]
    free_results = [r for r in regression_results if r["experiment_id"] in free_exp_ids]
    if free_results:
        free_slopes = np.array([r["slope"] for r in free_results])
        free_intercepts = np.array([r["intercept"] for r in free_results])
        slope_mean = np.mean(free_slopes)
        slope_std = np.std(free_slopes, ddof=1) if len(free_slopes) > 1 else 0.0
        intercept_mean = np.mean(free_intercepts)
        intercept_std = np.std(free_intercepts, ddof=1) if len(free_intercepts) > 1 else 0.0
        # 检查是否为零（数值容差1e-12）
        slope_zero = abs(slope_mean) < 1e-12 and slope_std < 1e-12
        intercept_zero = abs(intercept_mean) < 1e-12 and intercept_std < 1e-12
        free_obs = {
            "summary": (f"自由场实验(exp_01,04,07)回归验证: "
                        f"斜率均值={slope_mean:.6e}, 斜率标准差={slope_std:.6e}; "
                        f"截距均值={intercept_mean:.6e}, 截距标准差={intercept_std:.6e}. "
                        f"斜率和截距均接近0(绝对值<1e-12)? {slope_zero and intercept_zero}"),
            "source_data_refs": [f"{eid}:v" for eid in free_exp_ids] + [f"{eid}:a" for eid in free_exp_ids],
            "metrics": {
                "free_experiment_count": len(free_results),
                "slope_mean": slope_mean,
                "slope_std": slope_std,
                "intercept_mean": intercept_mean,
                "intercept_std": intercept_std,
                "slope_near_zero": slope_zero,
                "intercept_near_zero": intercept_zero
            }
        }
        new_observations.append(free_obs)

    # 4. 常数场实验回归参数表格（文本形式）
    const_results = [r for r in regression_results if r["force_field_type"] == "constant"]
    if const_results:
        # 构建表格文本
        header = f"{'exp_id':<10} {'F_ext':<8} {'v0':<8} {'slope':<12} {'intercept':<12} {'R²':<10}"
        lines = [header, "-" * len(header)]
        for r in const_results:
            lines.append(f"{r['experiment_id']:<10} {r['F_ext']:<8.2f} {r['v0']:<8.2f} {r['slope']:<12.6f} {r['intercept']:<12.6f} {r['r2']:<10.6f}")
        table_str = "\n".join(lines)
        table_obs = {
            "summary": f"常数场实验回归参数表格:\n{table_str}",
            "source_data_refs": [f"{r['experiment_id']}:v" for r in const_results] + [f"{r['experiment_id']}:a" for r in const_results],
            "metrics": {
                "constant_experiment_count": len(const_results),
                "regression_table": table_str
            }
        }
        new_observations.append(table_obs)

    # ---------- 生成分组散点图 ----------
    figures = []
    # 图1: 斜率 vs F_ext 按v0分组
    plt.figure(figsize=(8, 6))
    v0_unique = sorted(set(r["v0"] for r in regression_results))
    for v0_val in v0_unique:
        subset = [r for r in regression_results if r["v0"] == v0_val]
        x = [r["F_ext"] for r in subset]
        y = [r["slope"] for r in subset]
        plt.scatter(x, y, label=f"v0={v0_val:.0f}", s=50, alpha=0.8)
    plt.xlabel("F_ext")
    plt.ylabel("a-v regression slope")
    plt.title("Slope vs F_ext (grouped by v0)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    fig1_path = output_dir / "slope_vs_Fext_by_v0.png"
    plt.savefig(str(fig1_path), dpi=150)
    plt.close()
    figures.append(str(fig1_path))

    # 图2: 斜率 vs v0 按F_ext分组
    plt.figure(figsize=(8, 6))
    F_ext_unique = sorted(set(r["F_ext"] for r in regression_results))
    for f_val in F_ext_unique:
        subset = [r for r in regression_results if r["F_ext"] == f_val]
        x = [r["v0"] for r in subset]
        y = [r["slope"] for r in subset]
        plt.scatter(x, y, label=f"F_ext={f_val:.1f}", s=50, alpha=0.8)
    plt.xlabel("v0")
    plt.ylabel("a-v regression slope")
    plt.title("Slope vs v0 (grouped by F_ext)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    fig2_path = output_dir / "slope_vs_v0_by_Fext.png"
    plt.savefig(str(fig2_path), dpi=150)
    plt.close()
    figures.append(str(fig2_path))

    # ---------- 返回结果 ----------
    observation_text = (f"处理了{len(regression_results)}个实验的a-v回归，"
                        f"记录了每个实验的斜率和截距OBS；"
                        f"计算了斜率和截距与F_ext、v0的相关系数；"
                        f"验证了自由场回归接近零；"
                        f"生成了常数场回归参数表格；"
                        f"保存了2张分组散点图。")

    return {
        "observation": observation_text,
        "derived_series": [],  # 没有新的派生序列
        "observations": new_observations,
        "validations": [],      # 本模式不生成验证
        "figures": figures,
        "metrics": {
            "experiments_analyzed": len(regression_results),
            "observation_count": len(new_observations),
            "figure_count": len(figures)
        }
    }
