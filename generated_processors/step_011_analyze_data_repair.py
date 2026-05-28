import json
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

def process(payload: dict) -> dict:
    # --- extract parameters ---
    params = payload["parameters"]
    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        experiment_ids = list(payload["experiments"].keys())
    output_dir = payload["output_dir"]
    experiments = payload["experiments"]

    # --- validate required series ---
    required = ["q", "v", "residue_aF"]
    for eid in experiment_ids:
        exp = experiments[eid]
        avail = set(exp["available_series"])
        for s in required:
            if s not in avail:
                raise ValueError(f"Experiment {eid} missing required series '{s}'")

    # --- collect data across experiments ---
    per_exp_data = {}
    all_v, all_q, all_res = [], [], []
    for eid in experiment_ids:
        exp = experiments[eid]
        t = np.array(exp["series"]["t"])
        q = np.array(exp["series"]["q"])
        v = np.array(exp["series"]["v"])
        res = np.array(exp["series"]["residue_aF"])
        n = len(t)
        if not (len(q) == n and len(v) == n and len(res) == n):
            raise ValueError(f"Series length mismatch in {eid}")
        per_exp_data[eid] = {"v": v, "q": q, "res": res}
        all_v.append(v)
        all_q.append(q)
        all_res.append(res)
    V = np.concatenate(all_v)
    Q = np.concatenate(all_q)
    R = np.concatenate(all_res)
    n_total = len(V)

    observations_list = []
    figures_list = []

    # ========================================================================
    # TASK 1: Multiple linear regression: residue_aF ~ v + q
    # ========================================================================
    X_lin = np.column_stack([V, Q])
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(X_lin, R)
    r2_lin = r2_score(R, lin_reg.predict(X_lin))
    rmse_lin = math.sqrt(mean_squared_error(R, lin_reg.predict(X_lin)))
    coef_lin = lin_reg.coef_
    intercept_lin = lin_reg.intercept_

    n_lin = n_total
    p_lin = 2
    residuals_lin = R - lin_reg.predict(X_lin)
    mse_lin = np.sum(residuals_lin ** 2) / (n_lin - p_lin - 1)
    X_design_lin = np.column_stack([np.ones(n_lin), V, Q])
    cov_lin = mse_lin * np.linalg.inv(X_design_lin.T @ X_design_lin)
    se_lin = np.sqrt(np.diag(cov_lin))

    obs_lin = {
        "summary": f"跨实验多元线性回归 R²={r2_lin:.6f}, RMSE={rmse_lin:.6f}, "
                   f"intercept={intercept_lin:.6f}±{se_lin[0]:.6f}, "
                   f"coef_v={coef_lin[0]:.6f}±{se_lin[1]:.6f}, "
                   f"coef_q={coef_lin[1]:.6f}±{se_lin[2]:.6f}, "
                   f"n_points={n_lin}",
        "source_data_refs": [f"{e}:residue_aF" for e in experiment_ids] +
                            [f"{e}:v" for e in experiment_ids] +
                            [f"{e}:q" for e in experiment_ids],
        "metrics": {
            "linear_R2": r2_lin,
            "linear_RMSE": rmse_lin,
            "linear_intercept": intercept_lin,
            "linear_intercept_se": se_lin[0],
            "linear_coef_v": coef_lin[0],
            "linear_coef_v_se": se_lin[1],
            "linear_coef_q": coef_lin[1],
            "linear_coef_q_se": se_lin[2],
            "observation_count": n_lin
        }
    }
    observations_list.append(obs_lin)

    # ========================================================================
    # TASK 2: Multiple quadratic regression: residue_aF ~ v + q + v^2 + q^2 + v*q
    # ========================================================================
    V2 = V ** 2
    Q2 = Q ** 2
    VQ = V * Q
    X_quad = np.column_stack([V, Q, V2, Q2, VQ])
    quad_reg = LinearRegression(fit_intercept=True)
    quad_reg.fit(X_quad, R)
    r2_quad = r2_score(R, quad_reg.predict(X_quad))
    rmse_quad = math.sqrt(mean_squared_error(R, quad_reg.predict(X_quad)))
    coef_names = ["v", "q", "v^2", "q^2", "v*q"]
    coef_quad = quad_reg.coef_
    intercept_quad = quad_reg.intercept_
    n_quad = n_total  # <-- fix: define n_quad

    obs_quad = {
        "summary": f"跨实验多元二次回归 R²={r2_quad:.6f}, RMSE={rmse_quad:.6f}, "
                   f"intercept={intercept_quad:.6f}, coefficients: "
                   + ", ".join([f"{n}={c:.6f}" for n, c in zip(coef_names, coef_quad)]),
        "source_data_refs": [f"{e}:residue_aF" for e in experiment_ids] +
                            [f"{e}:v" for e in experiment_ids] +
                            [f"{e}:q" for e in experiment_ids],
        "metrics": {
            "quadratic_R2": r2_quad,
            "quadratic_RMSE": rmse_quad,
            "quadratic_intercept": intercept_quad,
            "quadratic_coef_v": coef_quad[0],
            "quadratic_coef_q": coef_quad[1],
            "quadratic_coef_v2": coef_quad[2],
            "quadratic_coef_q2": coef_quad[3],
            "quadratic_coef_vq": coef_quad[4],
            "observation_count": n_quad
        }
    }
    observations_list.append(obs_quad)

    # ========================================================================
    # TASK 3: Per-experiment linear regression: residue_aF ~ v + q
    # ========================================================================
    per_exp_results = {}
    for eid in experiment_ids:
        v = per_exp_data[eid]["v"]
        q = per_exp_data[eid]["q"]
        r = per_exp_data[eid]["res"]
        X = np.column_stack([v, q])
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, r)
        r2 = r2_score(r, reg.predict(X))
        rmse = math.sqrt(mean_squared_error(r, reg.predict(X)))
        per_exp_results[eid] = {
            "intercept": reg.intercept_,
            "coef_v": reg.coef_[0],
            "coef_q": reg.coef_[1],
            "R2": r2,
            "RMSE": rmse
        }

    intercepts = [v["intercept"] for v in per_exp_results.values()]
    coef_vs = [v["coef_v"] for v in per_exp_results.values()]
    coef_qs = [v["coef_q"] for v in per_exp_results.values()]
    summary_coef = (f"各实验线性回归系数范围: intercept [{min(intercepts):.6f}, {max(intercepts):.6f}], "
                    f"coef_v [{min(coef_vs):.6f}, {max(coef_vs):.6f}], "
                    f"coef_q [{min(coef_qs):.6f}, {max(coef_qs):.6f}]. "
                    "跨实验系数是否一致需进一步检验。")
    obs_per_exp = {
        "summary": f"12个实验分别进行residue_aF ~ v+q线性回归。{summary_coef}",
        "source_data_refs": [f"{e}:residue_aF" for e in experiment_ids] +
                            [f"{e}:v" for e in experiment_ids] +
                            [f"{e}:q" for e in experiment_ids],
        "metrics": {
            "experiment_count": len(experiment_ids),
            "intercept_mean": np.mean(intercepts),
            "intercept_std": np.std(intercepts),
            "coef_v_mean": np.mean(coef_vs),
            "coef_v_std": np.std(coef_vs),
            "coef_q_mean": np.mean(coef_qs),
            "coef_q_std": np.std(coef_qs),
            "R2_per_exp": {e: per_exp_results[e]["R2"] for e in experiment_ids}
        }
    }
    observations_list.append(obs_per_exp)

    # ========================================================================
    # TASK 4: Residual plots for the quadratic regression
    # ========================================================================
    residuals_quad = R - quad_reg.predict(X_quad)
    fitted_quad = quad_reg.predict(X_quad)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].scatter(fitted_quad, residuals_quad, alpha=0.5, s=10)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=0.8)
    axes[0, 0].set_xlabel("Fitted values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Fitted (quadratic model)")

    axes[0, 1].scatter(V, residuals_quad, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=0.8)
    axes[0, 1].set_xlabel("v")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residuals vs v")

    axes[1, 0].scatter(Q, residuals_quad, alpha=0.5, s=10)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=0.8)
    axes[1, 0].set_xlabel("q")
    axes[1, 0].set_ylabel("Residuals")
    axes[1, 0].set_title("Residuals vs q")

    axes[1, 1].hist(residuals_quad, bins=50, density=True, alpha=0.7)
    axes[1, 1].set_xlabel("Residuals")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Residual distribution")

    plt.tight_layout()
    fig_path = str(Path(output_dir) / "residual_diagnostics_quadratic.png")
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)
    figures_list.append(fig_path)

    # ========================================================================
    # Build final return
    # ========================================================================
    observation_text = (
        f"完成12个恒外力实验的多元回归分析。线性回归: R²={r2_lin:.4f}, RMSE={rmse_lin:.4f}；"
        f"二次回归: R²={r2_quad:.4f}, RMSE={rmse_quad:.4f}。"
        f"各实验单独线性回归系数范围见OBS。残差诊断图已保存。"
    )

    result = {
        "observation": observation_text,
        "derived_series": [],
        "observations": observations_list,
        "figures": figures_list,
        "metrics": {
            "linear_R2": r2_lin,
            "linear_RMSE": rmse_lin,
            "quadratic_R2": r2_quad,
            "quadratic_RMSE": rmse_quad,
            "n_experiments": len(experiment_ids),
            "n_points_total": n_total,
            "observation_count": len(observations_list)
        }
    }
    return result
