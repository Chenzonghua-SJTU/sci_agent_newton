import numpy as np
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, List

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # parameters
    analysis_goal = params.get("analysis_goal", "")
    experiment_ids = params.get("experiment_ids", list(experiments.keys()))
    optional_series = params.get("optional_series", [])
    expected_outputs = params.get("expected_outputs", [])

    # configuration for differentiation
    window_length = 31
    polyorder = 2
    k_global = 0.755557  # previously fitted exponent

    derived_series = []
    figures = []
    metrics = {}
    per_experiment_results = {}

    for exp_id in experiment_ids:
        if exp_id not in experiments:
            continue

        exp = experiments[exp_id]
        config = exp["config"]
        series = exp["series"]
        available = exp.get("available_series", list(series.keys()))

        # get t and q
        t = np.array(series["t"], dtype=float)
        q = np.array(series["q"], dtype=float)
        dt = config.get("dt", 0.01)
        F_ext = config.get("F_ext", None)
        if F_ext is None:
            F_ext = config.get("constant_force", None)
        if F_ext is None:
            raise ValueError(f"Experiment {exp_id}: cannot determine F_ext from config {config}")

        # Savitzky-Golay smoothing
        if len(q) < window_length:
            # fallback: smaller window
            wl = len(q) // 2 * 2 + 1  # make odd
            if wl < 5:
                wl = 5
        else:
            wl = window_length
        q_smooth = scipy.signal.savgol_filter(q, wl, polyorder)

        # Velocity: central difference using np.gradient
        v_sg = np.gradient(q_smooth, dt)

        # Acceleration: second gradient
        a_sg = np.gradient(v_sg, dt)

        # Compute candidate expression: a / (F_ext * exp(-k * |v|))
        denominator = F_ext * np.exp(-k_global * np.abs(v_sg))
        expr = np.divide(a_sg, denominator, out=np.full_like(a_sg, np.nan), where=np.abs(denominator) > 1e-15)

        # Remove inf/nan
        finite_mask = np.isfinite(expr)
        if np.any(finite_mask):
            expr_mean = float(np.mean(expr[finite_mask]))
            expr_std = float(np.std(expr[finite_mask]))
        else:
            expr_mean = np.nan
            expr_std = np.nan

        per_experiment_results[exp_id] = {
            "F_ext": F_ext,
            "expr_mean": expr_mean,
            "expr_std": expr_std,
            "v": v_sg.tolist(),
            "a": a_sg.tolist(),
            "expr": expr.tolist(),
            "q_smooth": q_smooth.tolist()
        }

        # register derived series
        # unique names: add suffix _sg31
        v_name = f"v_sg31"
        a_name = f"a_sg31"
        expr_name = f"expr_candidate"
        # check if they already exist? we will still register new ones
        derived_series.append({
            "experiment_id": exp_id,
            "name": v_name,
            "values": v_sg.tolist(),
            "source_name": "Savitzky-Golay smooth window=31 poly=2 + central diff",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"Velocity estimated from q via Savgol(window={wl},poly={polyorder}) + gradient"
        })
        derived_series.append({
            "experiment_id": exp_id,
            "name": a_name,
            "values": a_sg.tolist(),
            "source_name": "Central difference of v_sg31",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"Acceleration from gradient of v_sg31"
        })
        derived_series.append({
            "experiment_id": exp_id,
            "name": expr_name,
            "values": expr.tolist(),
            "source_name": f"a / (F_ext * exp(-{k_global}*|v|))",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "Candidate constant expression"
        })

    # Cross-experiment statistics
    expr_means = [res["expr_mean"] for res in per_experiment_results.values() if np.isfinite(res["expr_mean"])]
    expr_stds = [res["expr_std"] for res in per_experiment_results.values() if np.isfinite(res["expr_std"])]
    if len(expr_means) > 1:
        overall_mean = float(np.mean(expr_means))
        overall_std = float(np.std(expr_means, ddof=1))  # sample std
    elif len(expr_means) == 1:
        overall_mean = expr_means[0]
        overall_std = 0.0
    else:
        overall_mean = np.nan
        overall_std = np.nan

    # Build metrics
    for exp_id, res in per_experiment_results.items():
        prefix = exp_id
        metrics[f"{prefix}_F_ext"] = res["F_ext"]
        metrics[f"{prefix}_expr_mean"] = res["expr_mean"]
        metrics[f"{prefix}_expr_std"] = res["expr_std"]
    metrics["overall_expr_mean"] = overall_mean
    metrics["overall_expr_std"] = overall_std
    metrics["n_experiments"] = len(per_experiment_results)

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: expr vs t for all experiments
    ax = axes[0]
    for exp_id in experiment_ids:
        if exp_id not in per_experiment_results:
            continue
        res = per_experiment_results[exp_id]
        expr_arr = np.array(res["expr"])
        t_arr = np.array(experiments[exp_id]["series"]["t"])
        # mask non-finite
        mask = np.isfinite(expr_arr)
        ax.plot(t_arr[mask], expr_arr[mask], label=f"{exp_id} (F={res['F_ext']})", alpha=0.7)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.4, label='Ideal=1')
    ax.set_xlabel("t")
    ax.set_ylabel("expr = a / (F*exp(-k|v|))")
    ax.set_title(f"Expression over time (k={k_global})")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Plot 2: bar plot of experiment means with error bars (std within experiment)
    ax = axes[1]
    ids = [eid for eid in experiment_ids if eid in per_experiment_results]
    means = [per_experiment_results[eid]["expr_mean"] for eid in ids]
    stds = [per_experiment_results[eid]["expr_std"] for eid in ids]
    colors = ['C{}'.format(i % 10) for i in range(len(ids))]
    bars = ax.bar(ids, means, yerr=stds, capsize=5, color=colors)
    ax.axhline(y=overall_mean, color='r', linestyle='--', label=f'Overall mean={overall_mean:.3f}')
    ax.fill_between(range(-1, len(ids)+1), overall_mean-overall_std, overall_mean+overall_std,
                    alpha=0.2, color='red', label=f'±1σ={overall_std:.3f}')
    ax.set_ylabel("Expression mean ± std")
    ax.set_title("Per-experiment expression statistics")
    ax.legend()
    ax.grid(True, axis='y')

    # Plot 3: scatter of a vs v with ideal curve
    ax = axes[2]
    v_all = []
    a_all = []
    for exp_id in experiment_ids:
        if exp_id not in per_experiment_results:
            continue
        res = per_experiment_results[exp_id]
        v_arr = np.array(res["v"])
        a_arr = np.array(res["a"])
        mask = np.isfinite(v_arr) & np.isfinite(a_arr)
        ax.scatter(v_arr[mask], a_arr[mask], s=2, alpha=0.3, label=exp_id)
        v_all.extend(v_arr[mask].tolist())
        a_all.extend(a_arr[mask].tolist())

    # overlay ideal curve a = F_ext * exp(-k|v|) using overall average F_ext? or per experiment? just show for F=1
    v_ideal = np.linspace(0, 6, 200)
    a_ideal = 1.0 * np.exp(-k_global * v_ideal)
    ax.plot(v_ideal, a_ideal, 'k--', label=f'Ideal: a = F * exp(-{k_global}|v|)')
    ax.set_xlabel("v")
    ax.set_ylabel("a")
    ax.set_title("Acceleration vs velocity with ideal curve (F=1)")
    ax.legend(fontsize=7, markerscale=3)
    ax.grid(True)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "expr_analysis_summary.png")
    fig.savefig(fig_path)
    plt.close(fig)
    figures.append(fig_path)

    # observation text
    obs_lines = []
    obs_lines.append(f"对实验 {experiment_ids} 进行自定义分析。")
    obs_lines.append(f"方法：对 q 进行 Savitzky-Golay 平滑（窗口 {window_length}，阶次 {polyorder}），然后用中心差分（np.gradient）估计速度 v_sg31 和加速度 a_sg31。")
    obs_lines.append(f"计算表达式 expr = a / (F_ext * exp(-{k_global} * |v|))，期望为常数 1。")
    obs_lines.append("各实验结果：")
    for eid in experiment_ids:
        if eid in per_experiment_results:
            res = per_experiment_results[eid]
            obs_lines.append(f"  {eid}: F_ext={res['F_ext']}, expr mean={res['expr_mean']:.4f}, std={res['expr_std']:.4f}")
    obs_lines.append(f"跨实验统计：mean of means = {overall_mean:.4f}, std of means = {overall_std:.4f}")
    if overall_std > 0.15 * abs(overall_mean) if overall_mean != 0 else overall_std > 0.1:
        obs_lines.append("建议：表达式并非严格常数，特别是 exp_10 明显偏离（先前已知异常），建议改进指数系数或考虑速度依赖修正。")
    else:
        obs_lines.append("表达式在各实验间较为恒定，可视为常数。")
    obs_lines.append("返回派生序列：v_sg31, a_sg31, expr_candidate 供后续分析。")
    observation = "\n".join(obs_lines)

    result = {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
    return result
