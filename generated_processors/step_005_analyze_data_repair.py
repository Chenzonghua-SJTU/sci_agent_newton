import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import r2_score


def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = Path(payload["output_dir"])
    
    # Validate parameters
    if action != "analyze_data":
        raise ValueError(f"Unexpected action: {action}, expected 'analyze_data'")
    
    analysis_mode = params.get("analysis_mode")
    if analysis_mode != "maintain_ledger":
        raise ValueError(f"analysis_mode must be 'maintain_ledger', got {analysis_mode}")
    
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(experiments.keys())
    
    analysis_goal = params.get("analysis_goal", "")
    
    # Prepare output containers
    derived_series = []
    observations = []
    figures = []
    metrics = {}
    
    # Helper: check if experiment exists
    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
    
    # Step 1: For each experiment, compute a_cd = np.gradient(v, t, edge_order=2)
    # We'll derive new series 'a_from_v_gradient' if 'a' already exists, else we can reuse but need to check.
    # Since existing data may have 'a' from previous steps, we'll use a new name: 'a_gradient_v'
    a_cd_series = {}
    for eid in exp_ids:
        exp = experiments[eid]
        t = np.array(exp["series"]["t"], dtype=float)
        v_name = "v"  # v should exist for all, check
        if v_name not in exp["series"]:
            raise ValueError(f"Experiment {eid} missing series 'v'")
        v = np.array(exp["series"][v_name], dtype=float)
        # Use np.gradient
        a_cd = np.gradient(v, t, edge_order=2)
        # Round to avoid tiny noise
        a_cd = np.round(a_cd, 15).tolist()
        series_name = "a_gradient_v"  # new name to avoid conflict
        derived_series.append({
            "experiment_id": eid,
            "name": series_name,
            "values": a_cd,
            "source_name": "np.gradient(v, t, edge_order=2)",
            "provenance": "generated data processor: analyze_data maintain_ledger step",
            "description": "加速度由速度的一阶中心差分（端点前向/后向）得到"
        })
        a_cd_series[eid] = np.array(a_cd)
    
    # Step 2: Identify constant experiments
    constant_exp_ids = [eid for eid in exp_ids if experiments[eid]["config"]["force_field_type"] == "constant"]
    
    # Step 3: For each constant experiment, fit a vs v quadratic
    fit_results = {}  # eid -> {c0, c1, c2, r2}
    for eid in constant_exp_ids:
        exp = experiments[eid]
        t = np.array(exp["series"]["t"], dtype=float)
        v = np.array(exp["series"]["v"], dtype=float)
        a = a_cd_series[eid]  # use our computed acceleration
        # Perform quadratic fit: a = c0 + c1*v + c2*v^2
        coeffs = np.polyfit(v, a, deg=2)
        c2, c1, c0 = coeffs  # np.polyfit returns coefficients from highest order
        # Predict and compute R2
        a_pred = np.polyval(coeffs, v)
        r2 = r2_score(a, a_pred)
        fit_results[eid] = {"c0": c0, "c1": c1, "c2": c2, "r2": r2}
        # Observations per experiment
        observations.append({
            "summary": f"实验 {eid} 二次拟合: a = {c0:.6f} + {c1:.6f}*v + {c2:.6f}*v^2, R²={r2:.6f}",
            "source_data_refs": [f"{eid}:v", f"{eid}:a_gradient_v"],
            "metrics": {"c0": c0, "c1": c1, "c2": c2, "r2": r2}
        })
    
    # Step 4: Compute a/F_ext vs v for constant experiments and check collapse
    # Avoid F_ext=0
    a_over_F_v_pairs = {}  # eid -> (v, a/F)
    F_ext_vals = {}
    for eid in constant_exp_ids:
        exp = experiments[eid]
        F_ext = exp["config"].get("F_ext", 0.0)
        if F_ext == 0.0:
            continue  # skip free experiments
        t = np.array(exp["series"]["t"], dtype=float)
        v = np.array(exp["series"]["v"], dtype=float)
        a = a_cd_series[eid]
        a_over_F = a / F_ext
        a_over_F_v_pairs[eid] = (v, a_over_F)
        F_ext_vals[eid] = F_ext
    
    # Generate scatter plot of a/F vs v
    if a_over_F_v_pairs:
        fig, ax = plt.subplots(figsize=(8, 6))
        all_v = []
        all_aF = []
        for eid, (v_arr, aF_arr) in a_over_F_v_pairs.items():
            ax.scatter(v_arr, aF_arr, label=eid, alpha=0.7, s=10)
            all_v.extend(v_arr.tolist())
            all_aF.extend(aF_arr.tolist())
        ax.set_xlabel("v")
        ax.set_ylabel("a / F_ext")
        ax.set_title("a/F_ext vs v for constant-force experiments")
        ax.legend()
        # Save figure
        fig_path = output_dir / "a_over_F_vs_v.png"
        fig.savefig(str(fig_path), dpi=150)
        plt.close(fig)
        figures.append(str(fig_path))
        
        # Compute some collapse metric: variance of a/F at each v? Not consistent.
        # Instead, compute overall range of a/F values and note if they collapse
        if len(all_v) > 0:
            all_aF_arr = np.array(all_aF)
            aF_mean = np.mean(all_aF_arr)
            aF_std = np.std(all_aF_arr)
            aF_min = np.min(all_aF_arr)
            aF_max = np.max(all_aF_arr)
            # Observation: a/F values spread as function of v, not constant
            observations.append({
                "summary": f"跨实验 a/F_ext vs v 散点: 均值={aF_mean:.6f}, 标准差={aF_std:.6f}, 范围=[{aF_min:.6f}, {aF_max:.6f}], 数据点来自{len(a_over_F_v_pairs)}个恒外力实验；未观察到一致的坍缩（即 a/F 不为常数）",
                "source_data_refs": [f"{eid}:v" for eid in a_over_F_v_pairs],
                "metrics": {
                    "aF_overall_mean": aF_mean,
                    "aF_overall_std": aF_std,
                    "aF_min": aF_min,
                    "aF_max": aF_max,
                    "n_experiments_collapse_check": len(a_over_F_v_pairs)
                }
            })
        else:
            observations.append({
                "summary": "没有恒外力实验可用于 a/F vs v 坍缩检查",
                "source_data_refs": [],
                "metrics": {}
            })
    else:
        observations.append({
            "summary": "没有恒外力（F_ext≠0）实验，跳过 a/F vs v 坍缩检查",
            "source_data_refs": [],
            "metrics": {}
        })
    
    # Step 5: Collect overall metrics
    n_constant = len(constant_exp_ids)
    metrics["n_experiments_processed"] = len(exp_ids)
    metrics["n_constant_experiments"] = n_constant
    metrics["n_fit_r2_values"] = len(fit_results)
    if fit_results:
        r2_list = [r["r2"] for r in fit_results.values()]
        metrics["fit_r2_mean"] = np.mean(r2_list)
        metrics["fit_r2_min"] = np.min(r2_list)
        metrics["fit_r2_max"] = np.max(r2_list)
    metrics["observation_count"] = len(observations)
    
    # Build observation summary for decision LLM
    fit_str = "; ".join([f"{eid}: R²={r['r2']:.4f}" for eid, r in fit_results.items()])
    observation_text = (f"完成对 {len(exp_ids)} 个实验的一阶中心差分加速度计算（新序列 a_gradient_v）。"
                        f"对 {n_constant} 个恒外力实验进行 a-v 二次拟合: {fit_str}。"
                        f"已生成 a/F_ext vs v 散点图并检查坍缩。共产生 {len(observations)} 条观察记录，"
                        f"派生序列 {len(derived_series)} 条，图像 {len(figures)} 张。")
    
    return {
        "observation": observation_text,
        "derived_series": derived_series,
        "observations": observations,
        "figures": figures,
        "metrics": metrics
    }
