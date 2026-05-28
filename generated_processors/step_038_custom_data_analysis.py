from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import os

K_GLOBAL = 0.7479  # from step36

def _estimate_velocity_from_q(t: List[float], q: List[float]) -> List[float]:
    """Numerical derivative using central difference, returns array same length as t."""
    arr_t = np.array(t)
    arr_q = np.array(q)
    dt = np.diff(arr_t)
    if np.any(dt <= 0):
        raise ValueError("t must be strictly increasing")
    v = np.zeros_like(arr_q)
    # forward/backward for endpoints
    v[0] = (arr_q[1] - arr_q[0]) / (arr_t[1] - arr_t[0])
    v[-1] = (arr_q[-1] - arr_q[-2]) / (arr_t[-1] - arr_t[-2])
    # central for interior
    v[1:-1] = (arr_q[2:] - arr_q[:-2]) / (arr_t[2:] - arr_t[:-2])
    return v.tolist()

def process(payload: dict) -> dict:
    action = payload.get("action", "custom_data_analysis")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")
    exp_ids = params.get("experiment_ids", list(experiments.keys()))
    if not exp_ids:
        exp_ids = list(experiments.keys())

    derived_series = []
    figures = []
    metrics = {}

    # store per-experiment statistics
    per_exp_stats = {}
    # collect all constant-force experiment residuals and v for later fitting
    all_residuals = []
    all_v = []
    all_F = []
    all_exp_labels = []

    for eid in exp_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        config = exp.get("config", {})
        series = exp.get("series", {})
        available = exp.get("available_series", list(series.keys()))

        # --- get time t for length check ---
        t = series.get("t")
        if t is None:
            continue
        n = len(t)

        # --- get velocity v_est ---
        if "v_est" in series:
            v_est = series["v_est"]
        elif "velocity" in series:
            v_est = series["velocity"]
        elif "q" in series and "t" in series:
            v_est = _estimate_velocity_from_q(series["t"], series["q"])
        else:
            raise ValueError(f"Experiment {eid}: cannot obtain velocity sequence")

        # ensure length matches t
        if len(v_est) != n:
            raise ValueError(f"Experiment {eid}: v_est length {len(v_est)} != t length {n}")

        # --- get F_ext ---
        F_ext = config.get("F_ext", 0.0)
        if F_ext is None:
            F_ext = 0.0
        force_type = config.get("force_field_type", "")

        # --- get drag ---
        # for constant force experiments, drag should exist
        # for free experiments, drag is zero (since a=0 and F_ext=0)
        if "drag" in series:
            drag = series["drag"]
        elif force_type == "free":
            drag = [0.0] * n
        else:
            # attempt to compute: drag = F_ext - a_est if a_est exists, else fallback
            if "a_est" in series:
                a_est = series["a_est"]
                drag_val = [F_ext - a for a in a_est]
                drag = drag_val
            else:
                raise ValueError(f"Experiment {eid}: cannot determine drag. Available: {available}")

        if len(drag) != n:
            raise ValueError(f"Experiment {eid}: drag length {len(drag)} != t length {n}")

        # --- compute residual: drag - F_ext * (1 - exp(-K_GLOBAL * v_est)) ---
        residual = []
        for d, v in zip(drag, v_est):
            res = d - F_ext * (1.0 - np.exp(-K_GLOBAL * v))
            residual.append(res)

        # --- statistics ---
        res_arr = np.array(residual)
        mean_val = float(np.mean(res_arr))
        std_val = float(np.std(res_arr, ddof=1))  # sample std

        per_exp_stats[eid] = {
            "mean": mean_val,
            "std": std_val,
            "F_ext": F_ext
        }

        # --- collect for global analysis ---
        all_residuals.extend(residual)
        all_v.extend(v_est)
        all_F.extend([F_ext] * n)
        all_exp_labels.extend([eid] * n)

        # --- create derived series ---
        derived_series.append({
            "experiment_id": eid,
            "name": f"residual_direct",
            "values": residual,
            "source_name": "drag - F_ext*(1 - exp(-0.7479*v_est))",
            "provenance": "generated data processor: step_038_custom_data_analysis",
            "description": "Residual of direct saturation expression (using k=0.7479)"
        })

    # ---- Build observation text ----
    lines = []
    lines.append(f"全局常数 k = {K_GLOBAL}")
    for eid in exp_ids:
        if eid in per_exp_stats:
            s = per_exp_stats[eid]
            lines.append(f"{eid} (F_ext={s['F_ext']}): 残差均值={s['mean']:.6f}, 标准差={s['std']:.6f}")

    # ---- Global fitting of residual vs v for constant-force experiments only ----
    # Filter constant-force experiments (F_ext > 0)
    idx_cf = [i for i, f in enumerate(all_F) if f > 0]
    if len(idx_cf) > 2:
        v_cf = np.array(all_v)[idx_cf]
        res_cf = np.array(all_residuals)[idx_cf]
        # linear fit
        A = np.vstack([v_cf, np.ones_like(v_cf)]).T
        slope, intercept = np.linalg.lstsq(A, res_cf, rcond=None)[0]
        predicted = slope * v_cf + intercept
        ss_res = np.sum((res_cf - predicted)**2)
        ss_tot = np.sum((res_cf - np.mean(res_cf))**2)
        r2_linear = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        lines.append(f"恒外力实验残差 vs v 线性拟合: slope={slope:.6f}, intercept={intercept:.6f}, R²={r2_linear:.4f}")

        # also compute correlation with F_ext
        F_cf = np.array(all_F)[idx_cf]
        corr_matrix = np.corrcoef(res_cf, F_cf)
        corr_val = corr_matrix[0,1] if corr_matrix.shape == (2,2) else 0.0
        lines.append(f"残差与F_ext相关系数: {corr_val:.4f}")

        # ---- Try correction: residual ≈ -F_ext * exp(-K_GLOBAL * v) ----
        theory = -F_cf * np.exp(-K_GLOBAL * v_cf)
        diff = res_cf - theory
        rmse_theory = float(np.sqrt(np.mean(diff**2)))
        corr_theory = float(np.corrcoef(res_cf, theory)[0,1]) if len(res_cf) > 1 else 0.0
        lines.append(f"修正项尝试: residual ≈ -F_ext * exp(-k*v) 的RMSE={rmse_theory:.6f}, 相关系数={corr_theory:.4f}")

        # ---- Plot: residual vs v for constant-force experiments ----
        fig, ax = plt.subplots(figsize=(10, 6))
        # group by F_ext
        unique_F = sorted(set(all_F))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_F)))
        for Fv, color in zip(unique_F, colors):
            if Fv <= 0:
                continue
            mask = np.array(all_F) == Fv
            v_plot = np.array(all_v)[mask]
            res_plot = np.array(all_residuals)[mask]
            ax.scatter(v_plot, res_plot, s=8, alpha=0.6, color=color, label=f'F_ext={Fv}')
        # plot linear fit line
        if len(v_cf) > 0:
            v_sort = np.sort(v_cf)
            fit_line = slope * v_sort + intercept
            ax.plot(v_sort, fit_line, 'r--', label=f'Linear fit (R²={r2_linear:.3f})')
        ax.set_xlabel('v_est')
        ax.set_ylabel('Residual (drag - F_ext*(1 - exp(-k*v)))')
        ax.set_title('Residual vs v_est for constant-force experiments')
        ax.legend()
        fig_path = os.path.join(output_dir, "residual_direct_vs_v_constant_force.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        figures.append(fig_path)

        # ---- Plot: residual vs F_ext ----
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(F_cf, res_cf, s=10, alpha=0.5, c='blue')
        ax2.set_xlabel('F_ext')
        ax2.set_ylabel('Residual')
        ax2.set_title('Residual vs F_ext')
        fig2_path = os.path.join(output_dir, "residual_direct_vs_F_ext.png")
        fig2.savefig(fig2_path, dpi=150)
        plt.close(fig2)
        figures.append(fig2_path)

        # ---- Plot: compare residual with theory -F_ext * exp(-k*v) ----
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.scatter(theory, res_cf, s=10, alpha=0.5, c='green')
        # perfect line
        lims = [min(theory.min(), res_cf.min()), max(theory.max(), res_cf.max())]
        ax3.plot(lims, lims, 'k--', alpha=0.5)
        ax3.set_xlabel('Theoretical residual: -F_ext * exp(-k*v)')
        ax3.set_ylabel('Actual residual')
        ax3.set_title(f'Correlation = {corr_theory:.3f}, RMSE = {rmse_theory:.5f}')
        fig3_path = os.path.join(output_dir, "residual_vs_theory_correction.png")
        fig3.savefig(fig3_path, dpi=150)
        plt.close(fig3)
        figures.append(fig3_path)

        # store metrics
        metrics["linear_fit_slope"] = float(slope)
        metrics["linear_fit_intercept"] = float(intercept)
        metrics["linear_fit_R2"] = float(r2_linear)
        metrics["residual_F_ext_corr"] = float(corr_val)
        metrics["correction_rmse"] = float(rmse_theory)
        metrics["correction_corr"] = float(corr_theory)

    # ---- Store per-experiment stats in metrics ----
    for eid, s in per_exp_stats.items():
        metrics[f"{eid}_residual_mean"] = s["mean"]
        metrics[f"{eid}_residual_std"] = s["std"]

    observation = "分析结果：\n" + "\n".join(lines)

    # Add derived series for theory correction (for potential reference)
    # We only need to return the derived series computed earlier.
    # No need for additional series here; the residual_direct is already added.

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
