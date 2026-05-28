import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def process(payload: dict) -> dict:
    action = payload.get("action", "custom_data_analysis")
    params = payload.get("parameters", {})
    experiment_ids = params.get("experiment_ids")
    if not experiment_ids:
        experiment_ids = list(payload["experiments"].keys())
    output_dir = payload.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    # fixed global saturation coefficient from step 32
    k_global = 0.7447

    # collect data
    data_by_exp = {}
    per_exp_stats = {}
    all_v = []
    all_res = []
    all_F = []
    all_exp = []

    for eid in experiment_ids:
        exp = payload["experiments"].get(eid)
        if exp is None:
            continue
        config = exp["config"]
        series = exp["series"]
        F_ext = config.get("F_ext", 0.0)

        # get v_est
        if "v_est" not in series:
            raise ValueError(f"Experiment {eid} lacks v_est series")
        v_est = np.array(series["v_est"])

        # get drag / F_ext ratio
        if "ratio_drag_over_F" in series:
            ratio = np.array(series["ratio_drag_over_F"])
        elif "drag" in series and F_ext != 0.0:
            drag = np.array(series["drag"])
            ratio = drag / F_ext
        else:
            raise ValueError(f"Experiment {eid} lacks both ratio_drag_over_F and (drag with non-zero F_ext)")

        # residual
        predicted = 1.0 - np.exp(-k_global * v_est)
        residual = ratio - predicted

        data_by_exp[eid] = {
            "v_est": v_est,
            "residual": residual,
            "F_ext": F_ext
        }
        per_exp_stats[eid] = {
            "mean": float(np.mean(residual)),
            "std": float(np.std(residual, ddof=1)),
            "F_ext": F_ext
        }

        all_v.extend(v_est.tolist())
        all_res.extend(residual.tolist())
        all_F.extend([F_ext] * len(v_est))
        all_exp.extend([eid] * len(v_est))

    # convert to arrays
    all_v = np.array(all_v)
    all_res = np.array(all_res)
    all_F = np.array(all_F)

    # -------- Plot: residual vs v_est colored by F_ext --------
    sorted_F = sorted(set(all_F))
    if len(sorted_F) > 1:
        norm = plt.Normalize(min(sorted_F), max(sorted_F))
    else:
        norm = plt.Normalize(0, 1)
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(10, 6))
    for eid, data in data_by_exp.items():
        color = cmap(norm(data["F_ext"]))
        ax.scatter(data["v_est"], data["residual"],
                   c=[color], label=f"{eid} (F={data['F_ext']:.1f})",
                   s=5, alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel("v_est")
    ax.set_ylabel("residual = drag/F_ext - (1 - exp(-0.7447 v))")
    ax.set_title("Residual vs v_est colored by F_ext")
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label("F_ext")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    scatter_fig = os.path.join(output_dir, "residual_vs_v_est.png")
    fig.savefig(scatter_fig, dpi=100)
    plt.close(fig)

    # -------- Global fits: linear and quadratic residual ~ v --------
    # linear
    coeff_linear = np.polyfit(all_v, all_res, 1)
    y_pred_lin = np.polyval(coeff_linear, all_v)
    ss_res_lin = np.sum((all_res - y_pred_lin) ** 2)
    ss_tot = np.sum((all_res - np.mean(all_res)) ** 2)
    R2_lin = 1 - ss_res_lin / ss_tot if ss_tot != 0 else 0

    # quadratic
    coeff_quad = np.polyfit(all_v, all_res, 2)
    y_pred_quad = np.polyval(coeff_quad, all_v)
    ss_res_quad = np.sum((all_res - y_pred_quad) ** 2)
    R2_quad = 1 - ss_res_quad / ss_tot if ss_tot != 0 else 0

    # per-experiment linear fits
    fit_by_exp = {}
    for eid, data in data_by_exp.items():
        x = data["v_est"]
        y = data["residual"]
        if len(x) < 2:
            continue
        c = np.polyfit(x, y, 1)
        yp = np.polyval(c, x)
        ssr = np.sum((y - yp) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ssr / sst if sst != 0 else 0
        fit_by_exp[eid] = {"slope": c[0], "intercept": c[1], "R2": r2}

    # overall statistics
    overall_mean = float(np.mean(all_res))
    overall_std = float(np.std(all_res, ddof=1))
    overall_min = float(np.min(all_res))
    overall_max = float(np.max(all_res))

    # -------- Plot with overlay fits --------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for eid, data in data_by_exp.items():
        color = cmap(norm(data["F_ext"]))
        ax2.scatter(data["v_est"], data["residual"],
                    c=[color], label=f"{eid} (F={data['F_ext']:.1f})",
                    s=5, alpha=0.5)
    x_sorted = np.sort(all_v)
    ax2.plot(x_sorted, np.polyval(coeff_linear, x_sorted),
             'k--', label=f"linear (slope={coeff_linear[0]:.4f}, R²={R2_lin:.4f})")
    ax2.plot(x_sorted, np.polyval(coeff_quad, x_sorted),
             'r:', label=f"quadratic (R²={R2_quad:.4f})")
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_xlabel("v_est")
    ax2.set_ylabel("residual")
    ax2.set_title("Residual vs v_est with global fits")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig2.tight_layout()
    fits_fig = os.path.join(output_dir, "residual_fits.png")
    fig2.savefig(fits_fig, dpi=100)
    plt.close(fig2)

    # -------- Build derived series for residuals --------
    derived_series = []
    for eid, data in data_by_exp.items():
        derived_series.append({
            "experiment_id": eid,
            "name": "residual_saturation",
            "values": data["residual"].tolist(),
            "source_name": "residual = drag/F_ext - (1 - exp(-0.7447 * v_est))",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "Residual of saturation model with k=0.7447"
        })

    # -------- Metrics --------
    metrics = {
        "overall_residual_mean": overall_mean,
        "overall_residual_std": overall_std,
        "overall_residual_min": overall_min,
        "overall_residual_max": overall_max,
        "linear_fit_slope": coeff_linear[0],
        "linear_fit_intercept": coeff_linear[1],
        "linear_fit_R2": R2_lin,
        "quadratic_fit_a": coeff_quad[2],
        "quadratic_fit_b": coeff_quad[1],
        "quadratic_fit_c": coeff_quad[0],
        "quadratic_fit_R2": R2_quad,
    }
    for eid, s in per_exp_stats.items():
        metrics[f"{eid}_residual_mean"] = s["mean"]
        metrics[f"{eid}_residual_std"] = s["std"]
    for eid, f in fit_by_exp.items():
        metrics[f"{eid}_linear_residual_slope"] = f["slope"]
        metrics[f"{eid}_linear_residual_R2"] = f["R2"]

    # correlation between residual and F_ext
    try:
        corr, pval = stats.pearsonr(all_F, all_res)
        metrics["residual_F_ext_corr"] = corr
        metrics["residual_F_ext_pval"] = pval
    except Exception:
        corr, pval = None, None

    # -------- Observation text --------
    lines = [
        f"对 {len(experiment_ids)} 个恒外力实验 (F_ext 取值: {sorted_F}) 计算了残差 residual = drag/F_ext - (1 - exp(-0.7447 * v_est))。",
        f"整体残差统计: 均值={overall_mean:.4f}, 标准差={overall_std:.4f}, 最小值={overall_min:.4f}, 最大值={overall_max:.4f}。"
    ]
    for eid in experiment_ids:
        if eid in per_exp_stats:
            s = per_exp_stats[eid]
            lines.append(f"  {eid} (F_ext={s['F_ext']:.1f}): 残差均值={s['mean']:.4f}, 标准差={s['std']:.4f}。")
    lines.append(f"整体残差 vs v_est 线性拟合: slope={coeff_linear[0]:.4f}, intercept={coeff_linear[1]:.4f}, R²={R2_lin:.4f}。")
    lines.append(f"整体残差 vs v_est 二次拟合: a={coeff_quad[2]:.4f}, b={coeff_quad[1]:.4f}, c={coeff_quad[0]:.4f}, R²={R2_quad:.4f}。")
    if abs(coeff_linear[0]) > 0.01:
        lines.append("残差与v存在明显线性趋势，建议在饱和模型中增加与v相关的修正项（例如线性项）。")
    else:
        lines.append("残差无明显线性趋势，但可能存在二次形态或与F_ext相关。")
    if corr is not None:
        lines.append(f"残差与F_ext的Pearson相关系数={corr:.4f}, p-value={pval:.4e}。{'存在显著相关' if pval < 0.05 else '无明显相关'}")
    observation = "\n".join(lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [scatter_fig, fits_fig],
        "metrics": metrics
    }
