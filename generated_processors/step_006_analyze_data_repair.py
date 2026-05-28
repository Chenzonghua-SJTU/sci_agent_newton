import numpy as np
from typing import Dict, Any, List, Tuple

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = payload["parameters"]
    experiments = payload["experiments"]
    experiment_ids = params.get("experiment_ids", [])
    hypothesis_id = params.get("hypothesis_id", "H001")

    # Validate required experiments exist
    missing = [eid for eid in experiment_ids if eid not in experiments]
    if missing:
        raise ValueError(f"Missing experiments: {missing}")

    per_exp_results = {}
    all_v = []
    all_y = []

    for eid in experiment_ids:
        exp = experiments[eid]
        config = exp["config"]
        F_ext = config.get("F_ext", 0.0)
        series = exp["series"]

        # Fix: extract the experiment number from eid (e.g., "exp_02" -> "02")
        num_part = eid.split("_")[1]
        v_name = f"v_gradient_ledger_exp_exp_{num_part}"
        a_name = f"a_gradient_ledger_exp_exp_{num_part}"
        if v_name not in series or a_name not in series:
            raise ValueError(f"Required series not found for {eid}: {v_name} or {a_name}")

        v = np.array(series[v_name], dtype=float)
        a = np.array(series[a_name], dtype=float)
        if len(v) != len(a):
            raise ValueError(f"v and a series length mismatch for {eid}")

        y = a - F_ext  # target: should be -gamma * v
        x = v

        # Check for zero variance in x
        if np.all(x == 0):
            # v constant: no variation, gamma is undefined
            gamma = 0.0
            residuals = y
            rmse = float(np.sqrt(np.mean(residuals**2)))
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum(y**2)
            r2 = 0.0 if ss_tot > 0 else 1.0
        else:
            # Least squares: minimize sum((y - (-gamma * x))^2)
            # Solve y = slope * x, slope = -gamma
            slope = np.dot(x, y) / np.dot(x, x)  # normal equation
            gamma = -slope
            y_pred = slope * x
            residuals = y - y_pred
            rmse = float(np.sqrt(np.mean(residuals**2)))
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum(y**2)
            if ss_tot > 1e-15:
                r2 = 1.0 - ss_res / ss_tot
            else:
                r2 = 1.0 if ss_res < 1e-15 else 0.0

        per_exp_results[eid] = {
            "gamma": float(gamma),
            "R2": float(r2),
            "RMSE": float(rmse),
            "n_points": len(v)
        }

        # Collect for global regression
        all_v.extend(v.tolist())
        all_y.extend(y.tolist())

    # Gamma summary across experiments
    gamma_values = [r["gamma"] for r in per_exp_results.values()]
    gamma_mean = float(np.mean(gamma_values))
    gamma_std = float(np.std(gamma_values, ddof=1)) if len(gamma_values) > 1 else 0.0

    # Global regression using all data points
    all_v_arr = np.array(all_v, dtype=float)
    all_y_arr = np.array(all_y, dtype=float)
    if np.all(all_v_arr == 0):
        global_gamma = 0.0
        global_residuals = all_y_arr
        global_rmse = float(np.sqrt(np.mean(global_residuals**2)))
        global_ss_res = np.sum(global_residuals**2)
        global_ss_tot = np.sum(all_y_arr**2)
        global_r2 = 0.0 if global_ss_tot > 0 else 1.0
    else:
        global_slope = np.dot(all_v_arr, all_y_arr) / np.dot(all_v_arr, all_v_arr)
        global_gamma = -global_slope
        global_y_pred = global_slope * all_v_arr
        global_residuals = all_y_arr - global_y_pred
        global_rmse = float(np.sqrt(np.mean(global_residuals**2)))
        global_ss_res = np.sum(global_residuals**2)
        global_ss_tot = np.sum(all_y_arr**2)
        if global_ss_tot > 1e-15:
            global_r2 = 1.0 - global_ss_res / global_ss_tot
        else:
            global_r2 = 1.0 if global_ss_res < 1e-15 else 0.0

    # Build validation
    metric_values = {
        "per_experiment": per_exp_results,
        "gamma_mean": gamma_mean,
        "gamma_std": gamma_std,
        "global_gamma": float(global_gamma),
        "global_R2": float(global_r2),
        "global_RMSE": float(global_rmse)
    }

    # Decide supports based on global R² (simple heuristic)
    supports = bool(global_r2 > 0.7)

    summary = (
        f"验证假说H001: a = F_ext - gamma * v。对{len(experiment_ids)}个实验进行逐实验线性回归（过原点: a-F_ext = -gamma*v）。"
        f"Gamma均值={gamma_mean:.4f}, 标准差={gamma_std:.4f}。"
        f"合并全部数据点回归: gamma={global_gamma:.4f}, R²={global_r2:.4f}, RMSE={global_rmse:.6f}。"
        f"逐实验指标详见metric_values。"
    )

    validation = {
        "hypothesis_id": hypothesis_id,
        "experiment_ids": experiment_ids,
        "supports": supports,
        "metric_name": "gamma_estimation_R2_RMSE",
        "metric_values": metric_values,
        "aggregate_score": float(global_r2),
        "summary": summary,
        "source_data_refs": [
            f"{eid}:v_gradient_ledger_exp_exp_{eid.split('_')[1]}" for eid in experiment_ids
        ] + [f"{eid}:a_gradient_ledger_exp_exp_{eid.split('_')[1]}" for eid in experiment_ids]
    }

    # Construct return
    result = {
        "observation": (
            f"完成{len(experiment_ids)}个实验的假说验证。逐实验gamma、R²、RMSE已计算。"
            f"Gamma均值={gamma_mean:.4f}, 标准差={gamma_std:.4f}。"
            f"全局回归: gamma={global_gamma:.4f}, R²={global_r2:.4f}, RMSE={global_rmse:.6f}。"
        ),
        "validations": [validation],
        "metrics": {
            "experiment_count": len(experiment_ids),
            "gamma_mean": gamma_mean,
            "gamma_std": gamma_std,
            "global_gamma": float(global_gamma),
            "global_R2": float(global_r2),
            "global_RMSE": float(global_rmse)
        },
        "derived_series": [],
        "observations": [],
        "figures": []
    }
    return result
