import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    # Parse payload
    action = payload["action"]
    parameters = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = Path(payload["output_dir"])

    # Analysis mode and goal
    analysis_mode = parameters.get("analysis_mode", "maintain_ledger")
    analysis_goal = parameters.get("analysis_goal", "")
    experiment_ids = parameters.get("experiment_ids", [])
    if not experiment_ids:
        experiment_ids = list(experiments.keys())

    # Store results
    derived_series_list = []
    observations_list = []
    figures_list = []
    global_metrics = {}

    # --------------------------------------------------------------------------
    # 1. Record initial conditions for every requested experiment
    # --------------------------------------------------------------------------
    initial_conditions = {}
    for exp_id in experiment_ids:
        exp_data = experiments[exp_id]
        config = exp_data["config"]
        q0 = config.get("initial_q", None)
        v0 = config.get("initial_v", None)
        initial_conditions[exp_id] = (q0, v0)

    # Create one observation summarizing all initial conditions
    summary_lines = []
    for exp_id, (q0, v0) in initial_conditions.items():
        summary_lines.append(f"{exp_id}: q0={q0}, v0={v0}")
    ic_summary = "实验初始条件提取: " + "; ".join(summary_lines)
    observations_list.append({
        "summary": ic_summary,
        "source_data_refs": [f"{e}:config" for e in experiment_ids],
        "metrics": {"experiment_count": len(experiment_ids)}
    })

    # --------------------------------------------------------------------------
    # 2. Identify constant-force experiments and prepare data
    # --------------------------------------------------------------------------
    constant_exp_ids = []
    for exp_id in experiment_ids:
        config = experiments[exp_id]["config"]
        force_type = config.get("force_field_type", "")
        if force_type == "constant":
            constant_exp_ids.append(exp_id)

    # Ensure required series exist for constant experiments
    for exp_id in constant_exp_ids:
        series = experiments[exp_id]["series"]
        available = experiments[exp_id].get("available_series", [])
        required = ["t", "q", "v", "a"]
        for r in required:
            if r not in available:
                raise ValueError(f"实验 {exp_id} 缺少必需序列 {r}。已有序列: {available}")
        # Ensure (a - F_ext) series is present -> we will calculate if not
        # But if residue_aF already exists, use it; otherwise compute
        if "residue_aF" not in available:
            F_ext = experiments[exp_id]["config"]["F_ext"]
            a = np.array(series["a"])
            residue = a - F_ext
            derived_series_list.append({
                "experiment_id": exp_id,
                "name": "residue_aF",
                "values": residue.tolist(),
                "source_name": f"a - F_ext (F_ext={F_ext})",
                "provenance": "generated data processor: maintain_ledger",
                "description": "计算 (a - F_ext) 以分析外力作用"
            })

    # After deriving, collect all residue_aF series
    # For constant experiments, now we can read residue_aF from series or derived
    # We'll build data arrays for cross-experiment fitting
    all_v = []
    all_residue = []
    for exp_id in constant_exp_ids:
        series = experiments[exp_id]["series"]
        # If we just derived it, it might not be in series yet, but we can read from derived
        # Actually after the loop, derived list contains the new ones but they are not merged back.
        # Easiest: check if residue_aF exists in series; if not, compute on the fly using available a and F_ext
        if "residue_aF" in series:
            residue = np.array(series["residue_aF"])
        else:
            # We already derived it, but we can also recompute to be safe
            F_ext = experiments[exp_id]["config"]["F_ext"]
            a = np.array(series["a"])
            residue = a - F_ext
        v = np.array(series["v"])
        all_v.extend(v.tolist())
        all_residue.extend(residue.tolist())

    if not constant_exp_ids or len(all_v) == 0:
        # No constant experiments – skip analysis
        return {
            "observation": "未发现恒外力实验，跳过相关分析",
            "derived_series": derived_series_list,
            "observations": observations_list,
            "figures": figures_list,
            "metrics": {"constant_experiments_found": 0}
        }

    all_v = np.array(all_v)
    all_residue = np.array(all_residue)

    # --------------------------------------------------------------------------
    # 3. Cross-experiment (a-F_ext) vs v quadratic fit
    # --------------------------------------------------------------------------
    coeffs = np.polyfit(all_v, all_residue, 2)  # [c2, c1, c0]
    c2, c1, c0 = coeffs
    fitted = np.polyval(coeffs, all_v)
    residuals = all_residue - fitted
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((all_residue - np.mean(all_residue))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals**2))

    # Generate figure: all constant experiments scatter + fit curve
    plt.figure(figsize=(8,6))
    plt.scatter(all_v, all_residue, s=10, alpha=0.5, label="Data points (all constant exp.)")
    v_sorted = np.sort(all_v)
    plt.plot(v_sorted, np.polyval(coeffs, v_sorted), 'r-', label=f"Quadratic fit (R²={r2:.4f}, RMSE={rmse:.4f})")
    plt.xlabel("v")
    plt.ylabel("a - F_ext")
    plt.title("Cross-experiment (a-F_ext) vs v with quadratic fit")
    plt.legend()
    plt.grid(True)
    fig1_path = output_dir / "cross_experiment_aF_vs_v.png"
    plt.savefig(str(fig1_path), dpi=150)
    plt.close()
    figures_list.append(str(fig1_path))

    # observation for cross fit
    cross_fit_obs = {
        "summary": f"所有恒外力实验({len(constant_exp_ids)}个) (a-F_ext) vs v 二次拟合: "
                   f"c2={c2:.6f}, c1={c1:.6f}, c0={c0:.6f}, R²={r2:.4f}, RMSE={rmse:.4f}, "
                   f"数据点数={len(all_v)}",
        "source_data_refs": [f"{e}:v,{e}:residue_aF" for e in constant_exp_ids],
        "metrics": {
            "c2": c2, "c1": c1, "c0": c0,
            "R2": r2, "RMSE": rmse,
            "n_points": len(all_v),
            "n_experiments": len(constant_exp_ids)
        }
    }
    observations_list.append(cross_fit_obs)

    # --------------------------------------------------------------------------
    # 4. Per-experiment a vs q linear fit for all constant experiments
    # --------------------------------------------------------------------------
    a_vs_q_results = []
    for exp_id in constant_exp_ids:
        series = experiments[exp_id]["series"]
        q = np.array(series["q"])
        a = np.array(series["a"])
        # Linear fit a = slope * q + intercept
        slope, intercept = np.polyfit(q, a, 1)  # high to low: [slope, intercept]
        fitted_a = slope * q + intercept
        residuals = a - fitted_a
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((a - np.mean(a))**2)
        r2_fit = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
        rmse_fit = np.sqrt(np.mean(residuals**2))
        a_vs_q_results.append({
            "experiment_id": exp_id,
            "slope": slope,
            "intercept": intercept,
            "R2": r2_fit,
            "RMSE": rmse_fit
        })

    # Build a summary observation for a vs q fits
    lines = []
    for res in a_vs_q_results:
        lines.append(f"{res['experiment_id']}: slope={res['slope']:.6f}, intercept={res['intercept']:.6f}, R²={res['R2']:.4f}, RMSE={res['RMSE']:.6f}")
    aq_summary = "各恒外力实验 a vs q 线性拟合结果: " + "; ".join(lines)
    observations_list.append({
        "summary": aq_summary,
        "source_data_refs": [f"{r['experiment_id']}:q,{r['experiment_id']}:a" for r in a_vs_q_results],
        "metrics": {
            "experiment_count": len(a_vs_q_results),
            "details": {r["experiment_id"]: {
                "slope": r["slope"],
                "intercept": r["intercept"],
                "R2": r["R2"],
                "RMSE": r["RMSE"]
            } for r in a_vs_q_results}
        }
    })

    # --------------------------------------------------------------------------
    # 5. Detailed analysis for exp_09 and exp_10 (a vs v and a vs q with plots)
    # --------------------------------------------------------------------------
    special_exps = [e for e in experiment_ids if e in ("exp_09", "exp_10")]
    for exp_id in special_exps:
        if exp_id not in experiments:
            continue
        series = experiments[exp_id]["series"]
        t = np.array(series["t"])
        q = np.array(series["q"])
        v = np.array(series["v"])
        a = np.array(series["a"])
        F_ext = experiments[exp_id]["config"]["F_ext"]

        # a vs v quadratic fit
        coeffs_v = np.polyfit(v, a, 2)
        fitted_a_v = np.polyval(coeffs_v, v)
        res_v = a - fitted_a_v
        ss_res_v = np.sum(res_v**2)
        ss_tot_v = np.sum((a - np.mean(a))**2)
        r2_v = 1 - ss_res_v/ss_tot_v if ss_tot_v > 0 else 0.0
        rmse_v = np.sqrt(np.mean(res_v**2))

        # a vs q linear fit
        coeffs_q = np.polyfit(q, a, 1)  # slope, intercept
        fitted_a_q = coeffs_q[0] * q + coeffs_q[1]
        res_q = a - fitted_a_q
        ss_res_q = np.sum(res_q**2)
        ss_tot_q = np.sum((a - np.mean(a))**2)
        r2_q = 1 - ss_res_q/ss_tot_q if ss_tot_q > 0 else 0.0
        rmse_q = np.sqrt(np.mean(res_q**2))

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        # a vs v
        axes[0].scatter(v, a, s=15, alpha=0.6, label="Data")
        v_sorted = np.sort(v)
        axes[0].plot(v_sorted, np.polyval(coeffs_v, v_sorted), 'r-', 
                     label=f"Quadratic fit (R²={r2_v:.4f}, RMSE={rmse_v:.4f})")
        axes[0].set_xlabel("v")
        axes[0].set_ylabel("a")
        axes[0].set_title(f"{exp_id}: a vs v")
        axes[0].legend()
        axes[0].grid(True)
        # a vs q
        axes[1].scatter(q, a, s=15, alpha=0.6, label="Data")
        q_sorted = np.sort(q)
        axes[1].plot(q_sorted, coeffs_q[0]*q_sorted + coeffs_q[1], 'b-',
                     label=f"Linear fit (R²={r2_q:.4f}, RMSE={rmse_q:.4f})")
        axes[1].set_xlabel("q")
        axes[1].set_ylabel("a")
        axes[1].set_title(f"{exp_id}: a vs q")
        axes[1].legend()
        axes[1].grid(True)
        plt.tight_layout()
        fig_path = output_dir / f"{exp_id}_a_vs_v_and_a_vs_q.png"
        plt.savefig(str(fig_path), dpi=150)
        plt.close()
        figures_list.append(str(fig_path))

        # Observation for this experiment
        obs_special = {
            "summary": f"{exp_id} 详细分析: a vs v 二次拟合 (c2={coeffs_v[0]:.6f}, c1={coeffs_v[1]:.6f}, c0={coeffs_v[2]:.6f}, "
                       f"R²={r2_v:.4f}, RMSE={rmse_v:.4f}); a vs q 线性拟合 (斜率={coeffs_q[0]:.6f}, 截距={coeffs_q[1]:.6f}, "
                       f"R²={r2_q:.4f}, RMSE={rmse_q:.4f}); F_ext={F_ext}",
            "source_data_refs": [f"{exp_id}:q", f"{exp_id}:v", f"{exp_id}:a", f"{exp_id}:config"],
            "metrics": {
                "a_vs_v_c2": coeffs_v[0], "a_vs_v_c1": coeffs_v[1], "a_vs_v_c0": coeffs_v[2],
                "a_vs_v_R2": r2_v, "a_vs_v_RMSE": rmse_v,
                "a_vs_q_slope": coeffs_q[0], "a_vs_q_intercept": coeffs_q[1],
                "a_vs_q_R2": r2_q, "a_vs_q_RMSE": rmse_q,
                "F_ext": F_ext
            }
        }
        observations_list.append(obs_special)

    # --------------------------------------------------------------------------
    # 6. Build final return
    # --------------------------------------------------------------------------
    global_metrics = {
        "constant_experiments_analyzed": len(constant_exp_ids),
        "cross_fit_R2": r2,
        "cross_fit_RMSE": rmse,
        "initial_conditions_recorded": len(initial_conditions),
        "observations_generated": len(observations_list),
        "figures_generated": len(figures_list)
    }

    # Concise observation string for the decision LLM
    obs_text = (
        f"已处理 {len(experiment_ids)} 个实验。"
        f"提取初始条件: {', '.join([f'{k}:({v[0]},{v[1]})' for k,v in initial_conditions.items()])}. "
        f"对所有 {len(constant_exp_ids)} 个恒外力实验进行 (a-F_ext) vs v 整体二次拟合: R²={r2:.4f}, RMSE={rmse:.4f}. "
        f"每个恒外力实验 a vs q 线性拟合已完成。"
        f"对 exp_09 和 exp_10 生成了 a vs v (二次) 和 a vs q (线性) 详细拟合及图像。"
    )

    result = {
        "observation": obs_text,
        "derived_series": derived_series_list,
        "observations": observations_list,
        "figures": figures_list,
        "metrics": global_metrics
    }
    return result
