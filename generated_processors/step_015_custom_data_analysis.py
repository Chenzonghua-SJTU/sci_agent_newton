import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
from typing import Any, Dict, List

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    parameters = payload["parameters"]
    experiment_ids = parameters.get("experiment_ids", [])
    analysis_goal = parameters.get("analysis_goal", "")
    output_dir = payload["output_dir"]
    experiments = payload["experiments"]

    if not experiment_ids:
        raise ValueError("experiment_ids is empty or missing")

    os.makedirs(output_dir, exist_ok=True)

    derived_series = []
    metrics = {}
    figure_paths = []

    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")

        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        t = series.get("t")
        v_poly = series.get("v_poly")
        a_poly = series.get("a_poly")

        if t is None or v_poly is None or a_poly is None:
            raise ValueError(f"Experiment {eid}: required series 't', 'v_poly', 'a_poly' not all present")

        # Get F_ext from config or default mapping
        # config may have "F_ext" key; if not, infer from "constant_force" or "force_field_type"
        F_ext = config.get("F_ext", None)
        if F_ext is None:
            # attempt to extract from constant_force or similar
            if "constant_force" in config:
                F_ext = config["constant_force"]
            else:
                # fallback to known values based on note
                force_field_type = config.get("force_field_type", "")
                if eid == "exp_02":
                    F_ext = 1.0
                elif eid == "exp_04":
                    F_ext = 2.0
                elif eid == "exp_05":
                    F_ext = 1.0
                elif eid == "exp_06":
                    F_ext = 1.0
                else:
                    raise ValueError(f"Cannot determine F_ext for experiment {eid}")
        else:
            F_ext = float(F_ext)

        # convert to numpy arrays
        t_arr = np.array(t, dtype=float)
        v_arr = np.array(v_poly, dtype=float)
        a_arr = np.array(a_poly, dtype=float)

        if len(t_arr) == 0:
            raise ValueError(f"Experiment {eid}: empty series")

        # Compute delta_a = a_poly - F_ext
        delta_a = a_arr - F_ext

        # Compute x = v_poly * abs(v_poly)
        x = v_arr * np.abs(v_arr)

        # Linear regression: delta_a = k * x + b
        X = x.reshape(-1, 1)
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, delta_a)
        k = reg.coef_[0]
        b = reg.intercept_
        delta_pred = reg.predict(X)
        r2 = r2_score(delta_a, delta_pred)

        # Check b close to 0? Use relative tolerance: if |b| < 0.01 maybe
        b_near_zero = np.abs(b) < 0.01  # simple threshold; could be more robust

        # Compute ratio delta_a / x, exclude |v| near zero
        mask = np.abs(v_arr) > 1e-6
        if np.sum(mask) > 0:
            ratio = delta_a[mask] / x[mask]
            ratio_mean = float(np.mean(ratio))
            ratio_std = float(np.std(ratio))
        else:
            ratio_mean = np.nan
            ratio_std = np.nan

        # Store metrics for this experiment
        metrics[f"{eid}_k"] = float(k)
        metrics[f"{eid}_b"] = float(b)
        metrics[f"{eid}_R2"] = float(r2)
        metrics[f"{eid}_b_near_zero"] = int(b_near_zero)
        metrics[f"{eid}_ratio_mean"] = ratio_mean
        metrics[f"{eid}_ratio_std"] = ratio_std
        metrics[f"{eid}_F_ext"] = float(F_ext)

        # Create derived series: delta_a
        derived_series.append({
            "experiment_id": eid,
            "name": "delta_a",
            "values": delta_a.tolist(),
            "source_name": "a_poly - F_ext",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "Acceleration deviation from expected constant external force"
        })

        # Plot scatter + fit line for this experiment
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, delta_a, s=10, alpha=0.7, label="data")
        # fit line range
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = k * x_fit + b
        ax.plot(x_fit, y_fit, 'r-', label=f"fit: k={k:.4f}, b={b:.4f}, R²={r2:.4f}")
        ax.set_xlabel(r"$v \cdot |v|$")
        ax.set_ylabel(r"$\Delta a = a_{poly} - F_{ext}$")
        ax.set_title(f"{eid} (F_ext={F_ext})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        figure_filename = os.path.join(output_dir, f"{eid}_delta_a_vs_vabsv.png")
        plt.tight_layout()
        plt.savefig(figure_filename, dpi=150)
        plt.close(fig)
        figure_paths.append(figure_filename)

    # Also create a combined figure with all experiments if more than one
    if len(experiment_ids) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        for idx, eid in enumerate(experiment_ids):
            if idx >= len(axes):
                break
            exp = experiments[eid]
            v_poly = np.array(exp["series"]["v_poly"])
            a_poly = np.array(exp["series"]["a_poly"])
            F_ext = metrics.get(f"{eid}_F_ext", 1.0)
            delta_a = a_poly - F_ext
            x = v_poly * np.abs(v_poly)
            k = metrics[f"{eid}_k"]
            b = metrics[f"{eid}_b"]
            r2 = metrics[f"{eid}_R2"]
            ax = axes[idx]
            ax.scatter(x, delta_a, s=8, alpha=0.6)
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = k * x_fit + b
            ax.plot(x_fit, y_fit, 'r-')
            ax.set_title(f"{eid} (F={F_ext})")
            ax.set_xlabel(r"$v|v|$")
            ax.set_ylabel(r"$\Delta a$")
            ax.text(0.05, 0.95, f"k={k:.3f}\nb={b:.3f}\nR²={r2:.3f}",
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        # Hide unused subplots
        for idx in range(len(experiment_ids), len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        combined_fig_path = os.path.join(output_dir, "all_experiments_delta_a_vs_vabsv.png")
        plt.savefig(combined_fig_path, dpi=150)
        plt.close(fig)
        figure_paths.append(combined_fig_path)

    # Build observation string
    obs_parts = [f"对 {len(experiment_ids)} 个恒力实验执行 delta_a = a_poly - F_ext 分析。"]
    for eid in experiment_ids:
        k = metrics[f"{eid}_k"]
        b = metrics[f"{eid}_b"]
        r2 = metrics[f"{eid}_R2"]
        b_near = metrics[f"{eid}_b_near_zero"]
        ratio_mean = metrics[f"{eid}_ratio_mean"]
        ratio_std = metrics[f"{eid}_ratio_std"]
        b_str = "接近0" if b_near else "不接近0"
        obs_parts.append(
            f"{eid}: 线性拟合 k={k:.4f}, b={b:.4f}, R²={r2:.4f} (b {b_str})；"
            f"排除 |v|<1e-6 后 ratio(Δa/(v|v|)) 均值={ratio_mean:.4f}, 标准差={ratio_std:.4f}"
        )
    obs_parts.append("散点图及拟合线已保存。")
    observation = "\n".join(obs_parts)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figure_paths,
        "metrics": metrics
    }
