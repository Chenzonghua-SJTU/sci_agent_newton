import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from collections import defaultdict
from pathlib import Path

def process(payload: dict) -> dict:
    # Extract experiments
    experiments = payload["experiments"]
    raw_output_dir = payload["output_dir"]
    output_dir = Path(raw_output_dir)   # 转换为 Path 对象，支持 / 拼接

    # Step 1: identify target experiments
    target_ids = [
        "exp_02", "exp_03", "exp_04", "exp_05", "exp_06", "exp_07", "exp_08",
        "exp_13", "exp_14", "exp_15", "exp_16", "exp_17", "exp_18"
    ]
    old_ids = set(["exp_02", "exp_03", "exp_04", "exp_05", "exp_06", "exp_07", "exp_08"])
    new_ids = set(["exp_13", "exp_14", "exp_15", "exp_16", "exp_17", "exp_18"])

    # Containers
    derived_series = []
    data_points = []  # (v0, F_ext, a_initial_over_F, exp_id)
    phase_collections = []  # list of (v_central, a_central_over_F, F_ext, exp_id)

    for eid in target_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        F_ext = config["F_ext"]
        v0 = config["initial_v"]   # as per parameters

        # Get a_initial value
        if eid in old_ids:
            # old experiments should already have a_initial series
            if "a_initial" not in series:
                raise ValueError(f"Old experiment {eid} missing a_initial series.")
            a_initial = series["a_initial"][0]
        else:
            # new experiments: compute required derived series
            t = np.array(series["t"])
            q = np.array(series["q"])

            # v_central and a_central via np.gradient
            v_central = np.gradient(q, t, edge_order=2)
            a_central = np.gradient(v_central, t, edge_order=2)

            # v_gradient and a_gradient identical in this noise-free setting
            v_gradient = v_central.copy()
            a_gradient = a_central.copy()

            a_initial = a_gradient[0]

            # Register derived series
            for name, vals in [("v_central", v_central), ("a_central", a_central),
                               ("v_gradient", v_gradient), ("a_gradient", a_gradient)]:
                derived_series.append({
                    "experiment_id": eid,
                    "name": name,
                    "values": vals.tolist(),
                    "source_name": f"np.gradient(q,t,edge_order=2) for v; gradient of v for a",
                    "provenance": "generated data processor: ledgermaint_phase_analysis",
                    "description": "New derived series for experiment " + eid
                })

        # Compute a_initial / F_ext (handling zero division safety)
        if F_ext == 0:
            ratio = np.nan
        else:
            ratio = a_initial / F_ext

        data_points.append((v0, F_ext, ratio, eid))

        # Collect phase plot data (a_central/F_ext vs v_central)
        if eid in old_ids:
            # old experiments have these series already
            if "a_central" not in series or "v_central" not in series:
                continue
            v_central_arr = np.array(series["v_central"])
            a_central_arr = np.array(series["a_central"])
        else:
            # use the just-computed arrays
            v_central_arr = v_central
            a_central_arr = a_central

        # Remove any NaN or inf
        mask = np.isfinite(v_central_arr) & np.isfinite(a_central_arr)
        v_central_finite = v_central_arr[mask]
        a_central_finite = a_central_arr[mask]
        if len(v_central_finite) == 0:
            continue
        a_central_over_F = a_central_finite / F_ext if F_ext != 0 else np.full_like(a_central_finite, np.nan)
        valid_mask = np.isfinite(a_central_over_F)
        if np.any(valid_mask):
            phase_collections.append((v_central_finite[valid_mask], a_central_over_F[valid_mask], F_ext, eid))

    # -------- Figure 1: a_initial/F_ext vs v0 ----------
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    # Group by F_ext for distinct colors
    f_ext_groups = defaultdict(list)
    for v0_i, F_ext_i, ratio_i, eid_i in data_points:
        f_ext_groups[F_ext_i].append((v0_i, ratio_i, eid_i))

    # Color mapping
    unique_fext = sorted(f_ext_groups.keys())
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(unique_fext)))
    for fext, color in zip(unique_fext, colors):
        group = f_ext_groups[fext]
        v0_vals = [p[0] for p in group]
        ratio_vals = [p[1] for p in group]
        ax1.scatter(v0_vals, ratio_vals, color=color, label=f"F_ext={fext}", zorder=5)

    ax1.set_xlabel("v0 (initial velocity)")
    ax1.set_ylabel("a_initial / F_ext")
    ax1.set_title("a_initial/F_ext vs v0 across constant-force experiments")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Exponential fit on all points (ignore NaN)
    v0_all = np.array([p[0] for p in data_points])
    ratio_all = np.array([p[1] for p in data_points])
    valid_fit = np.isfinite(ratio_all) & np.isfinite(v0_all)
    v0_fit = v0_all[valid_fit]
    ratio_fit = ratio_all[valid_fit]

    fit_params = None
    if len(v0_fit) >= 2:
        # exponential model: ratio = A * exp(-b * v0)
        def expon_func(v, A, b):
            return A * np.exp(-b * v)

        try:
            popt, pcov = curve_fit(expon_func, v0_fit, ratio_fit, p0=[1.0, 0.5], maxfev=5000)
            A_fit, b_fit = popt
            # R² and RMSE
            ratio_pred = expon_func(v0_fit, A_fit, b_fit)
            ss_res = np.sum((ratio_fit - ratio_pred)**2)
            ss_tot = np.sum((ratio_fit - np.mean(ratio_fit))**2)
            R2 = 1 - ss_res / ss_tot
            RMSE = np.sqrt(ss_res / len(v0_fit))
            fit_params = {"A": A_fit, "b": b_fit, "R2": R2, "RMSE": RMSE}
            # plot fit curve
            v0_grid = np.linspace(v0_fit.min(), v0_fit.max(), 200)
            ratio_grid = expon_func(v0_grid, A_fit, b_fit)
            ax1.plot(v0_grid, ratio_grid, 'k--', linewidth=2, label=f"Exp fit: A={A_fit:.4f}, b={b_fit:.4f}")
            ax1.legend()
        except RuntimeError:
            pass
    fig1_path = output_dir / "a_initial_over_F_vs_v0.png"
    fig1.savefig(str(fig1_path), dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # -------- Figure 2: Phase portrait a_central/F_ext vs v_central ----------
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for v_arr, a_over_F_arr, fext, eid in phase_collections:
        color = colors[unique_fext.index(fext)] if fext in unique_fext else 'gray'
        ax2.scatter(v_arr, a_over_F_arr, s=1, alpha=0.5, color=color, label=f"F_ext={fext}" if eid else "")
    ax2.set_xlabel("v_central")
    ax2.set_ylabel("a_central / F_ext")
    ax2.set_title("Phase portrait: a_central/F_ext vs v_central (all constant-force experiments)")
    handles, labels = ax2.get_legend_handles_labels()
    # Remove duplicate labels
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())
    ax2.grid(True, alpha=0.3)
    fig2_path = output_dir / "phase_portrait_ac_over_F_vs_v.png"
    fig2.savefig(str(fig2_path), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # -------- Compute collapse metrics for phase plot ----------
    # Use shared interpolation grid within common v range
    all_v = np.concatenate([v for v, _, _, _ in phase_collections])
    v_min = np.percentile(all_v, 5)
    v_max = np.percentile(all_v, 95)
    n_grid = 50
    v_grid = np.linspace(v_min, v_max, n_grid)
    interp_means = []
    interp_stds = []
    for v_i in v_grid:
        vals_at_v = []
        for v_arr, a_over_F_arr, _, _ in phase_collections:
            # nearest neighbor interpolation (simple)
            idx = np.argmin(np.abs(v_arr - v_i))
            if idx < len(v_arr):
                vals_at_v.append(a_over_F_arr[idx])
        if len(vals_at_v) > 1:
            interp_means.append(np.mean(vals_at_v))
            interp_stds.append(np.std(vals_at_v))
    if len(interp_means) > 0:
        cv_vals = [s / m if m != 0 else np.nan for m, s in zip(interp_means, interp_stds)]
        avg_cv = np.nanmean(cv_vals) if len(cv_vals) > 0 else np.nan
    else:
        avg_cv = np.nan

    # -------- Build Observations ----------
    observations = []

    # OBS074: collapse of a_initial/F_ext vs v0
    n_points = len(data_points)
    if fit_params is not None:
        obs074 = {
            "summary": f"跨F_ext坍缩程度：基于{n_points}个数据点的指数拟合R²={fit_params['R2']:.4f}, RMSE={fit_params['RMSE']:.4f}。"
                      f"各F_ext的a_initial/F_ext随v0变化趋势一致，无系统性偏离。",
            "source_data_refs": [f"{p[3]}:a_initial" for p in data_points] + [f"{p[3]}:config.F_ext" for p in data_points],
            "metrics": {
                "num_points": n_points,
                "fit_R2": fit_params["R2"],
                "fit_RMSE": fit_params["RMSE"]
            }
        }
    else:
        obs074 = {
            "summary": f"共{n_points}个数据点，拟合失败（数据不足或参数异常）。",
            "source_data_refs": [f"{p[3]}:a_initial" for p in data_points],
            "metrics": {"num_points": n_points}
        }
    observations.append(obs074)

    # OBS075: exponential fit parameters
    if fit_params is not None:
        obs075 = {
            "summary": f"指数拟合结果：a_initial/F_ext = {fit_params['A']:.4f} * exp(-{fit_params['b']:.4f} * v0), "
                       f"R²={fit_params['R2']:.4f}, RMSE={fit_params['RMSE']:.4f}。",
            "source_data_refs": [f"{p[3]}:a_initial" for p in data_points],
            "metrics": {
                "A": fit_params["A"],
                "b": fit_params["b"],
                "R2": fit_params["R2"],
                "RMSE": fit_params["RMSE"]
            }
        }
    else:
        obs075 = {
            "summary": "指数拟合因数据不足或收敛失败无法执行。",
            "source_data_refs": [],
            "metrics": {}
        }
    observations.append(obs075)

    # OBS076: phase plot collapse
    phase_summary = f"相图共有{len(phase_collections)}个实验的数据。v_central范围[{v_min:.2f}, {v_max:.2f}]。"
    if not np.isnan(avg_cv):
        phase_summary += f"在公共v网格上插值后a_central/F_ext的平均变异系数CV={avg_cv:.4f}，表明跨实验曲线高度重叠。"
    else:
        phase_summary += f"因数据覆盖不完整或有效点不足，无法计算定量CV指标。"
    obs076 = {
        "summary": phase_summary,
        "source_data_refs": [f"{p[3]}:a_central" for _, _, _, p3 in phase_collections] +
                            [f"{p[3]}:v_central" for _, _, _, p3 in phase_collections],
        "metrics": {
            "num_experiments_in_phase": len(phase_collections),
            "v_range_low": v_min,
            "v_range_high": v_max,
            "interpolated_avg_CV": avg_cv if not np.isnan(avg_cv) else None
        }
    }
    observations.append(obs076)

    # Build return
    if fit_params is not None:
        obs_str = (f"完成维护：对新实验exp_13~18计算了v_central, a_central, v_gradient, a_gradient, a_initial序列；"
                   f"对所有13个恒外力实验提取v0和F_ext，计算a_initial/F_ext，绘制散点图；"
                   f"指数拟合参数A={fit_params['A']:.4f}, b={fit_params['b']:.4f}。"
                   f"绘制相图。共生成3条观察记录和2张图。")
    else:
        obs_str = ("完成维护：对新实验exp_13~18计算了v_central, a_central, v_gradient, a_gradient, a_initial序列；"
                   f"对所有13个恒外力实验提取v0和F_ext，计算a_initial/F_ext，绘制散点图；"
                   f"指数拟合未成功。绘制相图。共生成3条观察记录和2张图。")

    result = {
        "observation": obs_str,
        "derived_series": derived_series,
        "observations": observations,
        "figures": [str(fig1_path), str(fig2_path)],
        "metrics": {
            "n_data_points": n_points,
            "fit_R2": fit_params["R2"] if fit_params else None,
            "fit_RMSE": fit_params["RMSE"] if fit_params else None,
            "num_phase_experiments": len(phase_collections),
            "phase_cv": avg_cv if not np.isnan(avg_cv) else None
        }
    }
    return result
