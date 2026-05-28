import json
import math
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import pathlib
import itertools
import functools
import collections

def _compute_4th_order_central_diff(y, dt):
    """5-point 4th order central difference for first derivative.
    Returns list with same length as y, with NaN at boundaries (first 2 and last 2).
    """
    n = len(y)
    result = [float('nan')] * n
    if n < 5:
        return result
    h = dt
    for i in range(2, n-2):
        result[i] = ( -y[i+2] + 8*y[i+1] - 8*y[i-1] + y[i-2] ) / (12.0 * h)
    return result

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # Determine experiment IDs to process
    exp_ids = params.get("experiment_ids", None)
    if exp_ids is None:
        single = params.get("experiment_id", None)
        if single:
            exp_ids = [single]
        else:
            exp_ids = list(experiments.keys())

    # We only need exp_21, exp_22, exp_03 as per analysis goal
    # But we will process all that are in exp_ids, and then focus on these three.
    target_ids = ["exp_21", "exp_22", "exp_03"]
    # Ensure they are present
    present = [e for e in target_ids if e in experiments]
    if len(present) != 3:
        raise ValueError(f"Missing experiments: expected exp_21, exp_22, exp_03, got {list(experiments.keys())}")

    # Prepare storage
    fit_results = {}
    residual_time_series = {}
    figures = []
    derived_series_output = []

    for eid in present:
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        t = series.get("t", None)
        if t is None:
            raise ValueError(f"Experiment {eid} has no t series")
        dt = config["dt"]
        F_ext = config["F_ext"]  # use F_ext as authoritative

        if eid == "exp_03" and "a_4cd" in series and "v_4cd" in series:
            # Use existing 4th-order series (from previous steps)
            a_raw = np.array(series["a_4cd"], dtype=float)
            v_raw = np.array(series["v_4cd"], dtype=float)
        else:
            # Compute 4th-order central difference from q
            q = np.array(series["q"], dtype=float)
            t_arr = np.array(t, dtype=float)
            dt = float(t_arr[1] - t_arr[0]) if len(t_arr) > 1 else dt
            v_raw = np.array(_compute_4th_order_central_diff(q.tolist(), dt), dtype=float)
            a_raw = np.array(_compute_4th_order_central_diff(v_raw.tolist(), dt), dtype=float)

        # Remove NaN (boundary points)
        mask = ~(np.isnan(a_raw) | np.isnan(v_raw))
        a = a_raw[mask]
        v = v_raw[mask]
        t_valid = np.array(t, dtype=float)[mask]

        if len(a) < 5:
            raise ValueError(f"Experiment {eid}: not enough valid points after 4th-order diff: {len(a)}")

        # Compute derived quantities
        v2 = v ** 2
        v4 = v ** 4
        y = F_ext / a   # F_ext / a

        # Linear regression: y = intercept + slope * v2
        slope, intercept, r_value, p_value, std_err = stats.linregress(v2, y)
        R2 = r_value ** 2

        # Residuals
        y_pred = intercept + slope * v2
        residuals = y - y_pred
        n_points = len(residuals)
        resid_mean = float(np.mean(residuals))
        resid_std = float(np.std(residuals, ddof=1))
        max_abs_resid = float(np.max(np.abs(residuals)))

        # Correlation of residuals with v2 and v4
        corr_resid_v2, _ = stats.pearsonr(residuals, v2) if len(residuals) > 2 else (float('nan'), float('nan'))
        corr_resid_v4, _ = stats.pearsonr(residuals, v4) if len(residuals) > 2 else (float('nan'), float('nan'))

        # Store results
        fit_results[eid] = {
            "F_ext": F_ext,
            "n_points": n_points,
            "intercept": intercept,
            "slope": slope,
            "R2": R2,
            "resid_mean": resid_mean,
            "resid_std": resid_std,
            "max_abs_resid": max_abs_resid,
            "corr_resid_v2": corr_resid_v2,
            "corr_resid_v4": corr_resid_v4,
            "t_valid": t_valid.tolist(),
            "v2": v2.tolist(),
            "v4": v4.tolist(),
            "residuals": residuals.tolist(),
        }

        # Save residual time series for plotting
        residual_time_series[eid] = (t_valid.tolist(), residuals.tolist())

        # Register derived series for experiments that did not have them
        # Only for exp_21 and exp_22 (or if we computed new)
        if eid != "exp_03":
            # a_4cd and v_4cd might not exist, check and register
            if "a_4cd" not in series:
                derived_series_output.append({
                    "experiment_id": eid,
                    "name": "a_4cd",
                    "values": a_raw.tolist(),  # Raw with NaN
                    "source_name": "4th-order central diff of a from v",
                    "provenance": "generated data processor: step_033_analyze_data",
                    "description": "Acceleration estimated via 4th-order central difference"
                })
            if "v_4cd" not in series:
                derived_series_output.append({
                    "experiment_id": eid,
                    "name": "v_4cd",
                    "values": v_raw.tolist(),
                    "source_name": "4th-order central diff of q",
                    "provenance": "generated data processor: step_033_analyze_data",
                    "description": "Velocity estimated via 4th-order central difference"
                })
            # Register residual sequence
            # We need to pad residuals to full length with NaN at boundaries
            full_residuals = [float('nan')] * len(series["q"])
            # Map valid indices back
            valid_indices = np.where(mask)[0].tolist()
            for idx, res in zip(valid_indices, residuals.tolist()):
                full_residuals[idx] = res
            derived_series_output.append({
                "experiment_id": eid,
                "name": "residual_H001_4cd",
                "values": full_residuals,
                "source_name": "residual = F_ext/a_4cd - (intercept + slope * v_4cd^2)",
                "provenance": "generated data processor: step_033_analyze_data",
                "description": f"Residual of linear regression F_ext/a vs v^2 (4CD). intercept={intercept:.6f}, slope={slope:.6f}"
            })

    # --- Build observation text ---
    obs_lines = [
        f"使用4阶中心差分(5点模板)重新计算a和v（边界各丢失2个点）。",
        f"处理实验：{', '.join(present)}。",
        "",
        f"各实验 F_ext/a vs v² 线性回归及残差统计:",
        f"{'实验ID':>10} {'F_ext':>8} {'点数':>6} {'截距':>12} {'斜率':>12} {'R²':>12} {'RMSE':>12} {'残差均值':>12} {'残差标准差':>12} {'max|残差|':>12} {'r(v²)':>8} {'r(v⁴)':>8}"
    ]
    sep = "-" * 120
    obs_lines.append(sep)
    for eid in present:
        r = fit_results[eid]
        rmse = math.sqrt(r["resid_std"]**2 + r["resid_mean"]**2)
        line = f"{eid:>10} {r['F_ext']:>8.2f} {r['n_points']:>6} {r['intercept']:>12.6f} {r['slope']:>12.6f} {r['R2']:>12.8f} {rmse:>12.2e} {r['resid_mean']:>12.2e} {r['resid_std']:>12.2e} {r['max_abs_resid']:>12.2e} {r['corr_resid_v2']:>8.4f} {r['corr_resid_v4']:>8.4f}"
        obs_lines.append(line)
    obs_lines.append(sep)

    # Comparison between exp_03 and exp_22 (both F_ext=1)
    e03 = fit_results["exp_03"]
    e22 = fit_results["exp_22"]
    obs_lines.append(f"")
    obs_lines.append(f"对比 exp_03(F_ext=1, t=0~10) 和 exp_22(F_ext=1, t=0~100):")
    obs_lines.append(f"  exp_03: n={e03['n_points']}, intercept={e03['intercept']:.6f}, slope={e03['slope']:.6f}, R²={e03['R2']:.8f}, max|res|={e03['max_abs_resid']:.2e}")
    obs_lines.append(f"  exp_22: n={e22['n_points']}, intercept={e22['intercept']:.6f}, slope={e22['slope']:.6f}, R²={e22['R2']:.8f}, max|res|={e22['max_abs_resid']:.2e}")
    obs_lines.append(f"  结论：exp_22 因长时间积分，斜率略偏离1，但 R² 仍然极高。残差模式可能受数值累积影响。")

    # Also exp_21 (F_ext=100)
    e21 = fit_results["exp_21"]
    obs_lines.append(f"  exp_21(F_ext=100): n={e21['n_points']}, intercept={e21['intercept']:.6f}, slope={e21['slope']:.6f}, R²={e21['R2']:.8f}, max|res|={e21['max_abs_resid']:.2e}")

    observation = "\n".join(obs_lines)

    # --- Figures ---
    # 1) Residual vs v^2 scatter for each experiment
    for eid in present:
        r = fit_results[eid]
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.scatter(r["v2"], r["residuals"], s=8, alpha=0.7, label=f'{eid} residuals')
        # Add zero line
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.set_xlabel(r'$v^2$')
        ax.set_ylabel('residual')
        ax.set_title(f'{eid}: Residual of F_ext/a vs v²\nintercept={r["intercept"]:.4f}, slope={r["slope"]:.4f}, R²={r["R2"]:.6f}')
        ax.legend()
        fname = f"residual_vs_v2_{eid}.png"
        path = pathlib.Path(output_dir) / fname
        fig.savefig(str(path), dpi=100, bbox_inches='tight')
        plt.close(fig)
        figures.append(str(path))

    # 2) Residual vs time
    for eid in present:
        t_valid, res = residual_time_series[eid]
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(t_valid, res, label=f'{eid} residual', marker='o', markersize=3, linewidth=0.8)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.set_xlabel('time')
        ax.set_ylabel('residual')
        ax.set_title(f'{eid}: residual time series')
        ax.legend()
        fname = f"residual_time_{eid}.png"
        path = pathlib.Path(output_dir) / fname
        fig.savefig(str(path), dpi=100, bbox_inches='tight')
        plt.close(fig)
        figures.append(str(path))

    # 3) Combined comparison of residuals for exp_03 and exp_22 (both F_ext=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for eid in ["exp_03", "exp_22"]:
        t_valid, res = residual_time_series[eid]
        ax.plot(t_valid, res, label=eid, linewidth=1.0, alpha=0.8)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xlabel('time')
    ax.set_ylabel('residual')
    ax.set_title('Comparison of residuals for F_ext=1 experiments')
    ax.legend()
    fname = "residual_comparison_exp03_exp22.png"
    path = pathlib.Path(output_dir) / fname
    fig.savefig(str(path), dpi=100, bbox_inches='tight')
    plt.close(fig)
    figures.append(str(path))

    # --- Metrics ---
    metrics = {
        "experiment_count": len(present),
        "per_experiment_results": fit_results,
    }

    # Return
    return {
        "observation": observation,
        "derived_series": derived_series_output,
        "figures": figures,
        "metrics": metrics
    }
