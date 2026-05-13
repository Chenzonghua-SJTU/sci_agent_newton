import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def process(payload: dict) -> dict:
    action = payload.get("action", "custom_data_analysis")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    eids = params.get("experiment_ids", [])
    if not eids:
        raise ValueError("experiment_ids not provided in parameters.")

    for eid in eids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload.")

    all_metrics = {}
    fig_data = {}
    derived_series = []

    for eid in eids:
        exp = experiments[eid]
        config = exp.get("config", {})

        # extract F_ext
        F_ext = config.get("F_ext")
        if F_ext is None:
            F_ext = config.get("constant_force")
        if F_ext is None:
            # fallback hard-coded values (from analysis goal)
            fallback = {"exp_02": 1.0, "exp_03": 10.0, "exp_04": 10.0, "exp_05": 5.0}
            F_ext = fallback.get(eid)
            if F_ext is None:
                raise ValueError(f"F_ext cannot be determined for experiment {eid}.")

        series = exp.get("series", {})
        v_sg = np.array(series.get("v_sg"))
        a_sg = np.array(series.get("a_sg"))
        if v_sg is None or a_sg is None:
            raise ValueError(f"Experiment {eid} missing v_sg or a_sg series.")
        if len(v_sg) != len(a_sg):
            raise ValueError(f"Series length mismatch in experiment {eid}.")

        drag = F_ext - a_sg

        # linear fit: drag = c0 + c1 * v_sg
        coeffs_lin = np.polyfit(v_sg, drag, 1)
        drag_pred_lin = np.polyval(coeffs_lin, v_sg)
        r2_lin = r2_score(drag, drag_pred_lin)
        resid_std_lin = float(np.std(drag - drag_pred_lin, ddof=2))

        # quadratic fit: drag = c0 + c1 * v_sg + c2 * v_sg^2
        coeffs_quad = np.polyfit(v_sg, drag, 2)
        drag_pred_quad = np.polyval(coeffs_quad, v_sg)
        r2_quad = r2_score(drag, drag_pred_quad)
        resid_std_quad = float(np.std(drag - drag_pred_quad, ddof=3))

        # drag / v_sg statistics (exclude points where |v_sg| < 1e-12)
        mask = np.abs(v_sg) > 1e-12
        if np.any(mask):
            ratio = drag[mask] / v_sg[mask]
            ratio_mean = float(np.mean(ratio))
            ratio_std = float(np.std(ratio))
        else:
            ratio_mean = float('nan')
            ratio_std = float('nan')

        prefix = eid + "_"
        all_metrics[prefix + "linear_c0"] = float(coeffs_lin[1])
        all_metrics[prefix + "linear_c1"] = float(coeffs_lin[0])
        all_metrics[prefix + "linear_R2"] = r2_lin
        all_metrics[prefix + "linear_resid_std"] = resid_std_lin
        all_metrics[prefix + "quad_c0"] = float(coeffs_quad[2])
        all_metrics[prefix + "quad_c1"] = float(coeffs_quad[1])
        all_metrics[prefix + "quad_c2"] = float(coeffs_quad[0])
        all_metrics[prefix + "quad_R2"] = r2_quad
        all_metrics[prefix + "quad_resid_std"] = resid_std_quad
        all_metrics[prefix + "drag_over_v_mean"] = ratio_mean
        all_metrics[prefix + "drag_over_v_std"] = ratio_std

        # derived series
        drag_list = drag.tolist()
        lin_resid_list = (drag - drag_pred_lin).tolist()
        quad_resid_list = (drag - drag_pred_quad).tolist()

        derived_series.append({
            "experiment_id": eid,
            "name": "drag",
            "values": drag_list,
            "source_name": f"drag = F_ext - a_sg, F_ext={F_ext}",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"Calculated drag for experiment {eid}"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "linear_residual",
            "values": lin_resid_list,
            "source_name": "drag - (c0 + c1*v_sg) from linear fit",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"Residual of linear fit for experiment {eid}"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "quad_residual",
            "values": quad_resid_list,
            "source_name": "drag - (c0 + c1*v_sg + c2*v_sg^2) from quadratic fit",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"Residual of quadratic fit for experiment {eid}"
        })

        # store for plots
        v_sorted = np.sort(v_sg)
        fig_data[eid] = {
            "v_sg": v_sg.tolist(),
            "drag": drag_list,
            "v_sorted": v_sorted.tolist(),
            "drag_pred_lin": np.polyval(coeffs_lin, v_sorted).tolist(),
            "drag_pred_quad": np.polyval(coeffs_quad, v_sorted).tolist(),
            "lin_resid": lin_resid_list,
            "quad_resid": quad_resid_list,
            "F_ext": F_ext,
            "r2_lin": r2_lin,
            "r2_quad": r2_quad
        }

    # ---- Figures ----
    figures = []

    # 1. Independent fits per experiment
    n_exp = len(eids)
    if n_exp > 0:
        fig, axes = plt.subplots(
            (n_exp + 1) // 2, 2 if n_exp > 1 else 1,
            figsize=(12, 5 * ((n_exp + 1) // 2)),
            squeeze=False
        )
        axes_flat = axes.flatten()
        for idx, eid in enumerate(eids):
            ax = axes_flat[idx]
            d = fig_data[eid]
            ax.scatter(d["v_sg"], d["drag"], s=10, alpha=0.7,
                       label=f"{eid} (F_ext={d['F_ext']})")
            ax.plot(d["v_sorted"], d["drag_pred_lin"], '--',
                    label=f"Linear  R²={d['r2_lin']:.3f}")
            ax.plot(d["v_sorted"], d["drag_pred_quad"], '-',
                    label=f"Quadratic R²={d['r2_quad']:.3f}")
            ax.set_xlabel("v_sg")
            ax.set_ylabel("drag")
            ax.set_title(f"{eid}: drag vs v_sg")
            ax.legend(fontsize=8)
        # hide unused subplots
        for idx in range(len(eids), len(axes_flat)):
            axes_flat[idx].axis('off')
        plt.tight_layout()
        fig1_path = os.path.join(output_dir, "drag_vs_v_sg_independent_fits.png")
        fig.savefig(fig1_path, dpi=150)
        plt.close(fig)
        figures.append(fig1_path)

    # 2. Residual plots
    if n_exp > 0:
        fig, axes = plt.subplots(
            (n_exp + 1) // 2, 2 if n_exp > 1 else 1,
            figsize=(12, 5 * ((n_exp + 1) // 2)),
            squeeze=False
        )
        axes_flat = axes.flatten()
        for idx, eid in enumerate(eids):
            ax = axes_flat[idx]
            d = fig_data[eid]
            ax.scatter(d["v_sg"], d["lin_resid"], s=8, alpha=0.5, label="Linear residual")
            ax.scatter(d["v_sg"], d["quad_resid"], s=8, alpha=0.5, label="Quadratic residual")
            ax.axhline(0, color='gray', linestyle='--')
            ax.set_xlabel("v_sg")
            ax.set_ylabel("Residual")
            ax.set_title(f"{eid}: residuals")
            ax.legend(fontsize=8)
        for idx in range(len(eids), len(axes_flat)):
            axes_flat[idx].axis('off')
        plt.tight_layout()
        fig2_path = os.path.join(output_dir, "drag_vs_v_sg_residuals.png")
        fig.savefig(fig2_path, dpi=150)
        plt.close(fig)
        figures.append(fig2_path)

    # ---- Observation ----
    lines = [f"对实验 {', '.join(eids)} 计算 drag = F_ext - a_sg，进行线性与二次拟合，并计算 drag/v_sg 的均值与标准差。"]
    for eid in eids:
        m = all_metrics
        lines.append(f"\n{eid}:")
        lines.append(f"  线性: drag = {m[eid+'_linear_c0']:.4f} + {m[eid+'_linear_c1']:.4f}·v_sg   R²={m[eid+'_linear_R2']:.4f}  残差标准差={m[eid+'_linear_resid_std']:.4f}")
        lines.append(f"  二次: drag = {m[eid+'_quad_c0']:.4f} + {m[eid+'_quad_c1']:.4f}·v_sg + {m[eid+'_quad_c2']:.4f}·v_sg²   R²={m[eid+'_quad_R2']:.4f}  残差标准差={m[eid+'_quad_resid_std']:.4f}")
        lines.append(f"  drag/v_sg 均值={m[eid+'_drag_over_v_mean']:.4f}  标准差={m[eid+'_drag_over_v_std']:.4f}")
    lines.append("\n跨实验 drag/v_sg 均值比较:")
    for eid in eids:
        lines.append(f"  {eid}: {all_metrics[eid+'_drag_over_v_mean']:.4f} ± {all_metrics[eid+'_drag_over_v_std']:.4f}")
    observation = "\n".join(lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": all_metrics
    }
