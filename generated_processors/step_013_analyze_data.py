import json
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import f as f_dist, ttest_1samp
from typing import Dict, List, Any

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    if action != "analyze_data":
        raise ValueError(f"Unsupported action: {action}")

    analysis_mode = params.get("analysis_mode", "")
    if analysis_mode != "maintain_ledger":
        raise ValueError("This routine only supports maintain_ledger mode")

    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        raise ValueError("experiment_ids is required")

    # Validate all experiments exist and have a, v series
    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
        exp = experiments[eid]
        if "a" not in exp["available_series"] or "v" not in exp["available_series"]:
            raise ValueError(f"Experiment {eid} lacks a or v series")

    # 1. Perform a-v linear regression for each experiment
    results = []  # list of dict: exp_id, F_ext, v0, slope, intercept
    for eid in exp_ids:
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        v = np.array(series["v"])
        a = np.array(series["a"])
        if len(v) != len(a):
            raise ValueError(f"Experiment {eid}: v and a series length mismatch")
        coeffs = np.polyfit(v, a, 1)  # returns [slope, intercept]
        slope = float(coeffs[0])
        intercept = float(coeffs[1])
        F_ext = float(config["F_ext"])
        v0 = float(config["initial_v"])
        results.append({
            "exp_id": eid,
            "F_ext": F_ext,
            "v0": v0,
            "slope": slope,
            "intercept": intercept
        })

    n = len(results)
    F_ext_arr = np.array([r["F_ext"] for r in results]).reshape(-1, 1)
    v0_arr = np.array([r["v0"] for r in results]).reshape(-1, 1)
    X = np.hstack((F_ext_arr, v0_arr))  # (n, 2)

    # ---- Intercept regression ----
    y_intercept = np.array([r["intercept"] for r in results])
    reg_intercept = LinearRegression(fit_intercept=True).fit(X, y_intercept)
    c0 = float(reg_intercept.intercept_)
    c1, c2 = reg_intercept.coef_
    y_pred_intercept = reg_intercept.predict(X)
    ss_res = np.sum((y_intercept - y_pred_intercept) ** 2)
    ss_tot = np.sum((y_intercept - np.mean(y_intercept)) ** 2)
    r2_intercept = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # F-test for overall significance
    df_model = 2
    df_res = n - df_model - 1
    if df_res > 0 and r2_intercept < 1.0:
        F_intercept = (r2_intercept / df_model) / ((1 - r2_intercept) / df_res)
        p_intercept = 1 - f_dist.cdf(F_intercept, df_model, df_res)
    else:
        F_intercept = float('inf')
        p_intercept = 0.0

    # ---- Slope regression ----
    y_slope = np.array([r["slope"] for r in results])
    reg_slope = LinearRegression(fit_intercept=True).fit(X, y_slope)
    d0 = float(reg_slope.intercept_)
    d1, d2 = reg_slope.coef_
    y_pred_slope = reg_slope.predict(X)
    ss_res_slope = np.sum((y_slope - y_pred_slope) ** 2)
    ss_tot_slope = np.sum((y_slope - np.mean(y_slope)) ** 2)
    r2_slope = 1 - ss_res_slope / ss_tot_slope if ss_tot_slope > 0 else 0.0
    if df_res > 0 and r2_slope < 1.0:
        F_slope = (r2_slope / df_model) / ((1 - r2_slope) / df_res)
        p_slope = 1 - f_dist.cdf(F_slope, df_model, df_res)
    else:
        F_slope = float('inf')
        p_slope = 0.0

    # ---- Intercept - F_ext residual ----
    diffs = np.array([r["intercept"] - r["F_ext"] for r in results])
    diff_mean = float(np.mean(diffs))
    diff_std = float(np.std(diffs, ddof=1))  # sample std

    # t-test for zero mean
    t_stat, p_ttest = ttest_1samp(diffs, 0)

    # ---- Build observations ----
    observations = []
    source_refs_all = [f"{eid}:a" for eid in exp_ids] + [f"{eid}:v" for eid in exp_ids]

    # Observation 1: Intercept regression
    obs1 = {
        "summary": "截距关于F_ext和v0的二元线性回归：截距= {:.6f}, 系数(F_ext)={:.6f}, 系数(v0)={:.6f}, R²={:.6f}, p_value={:.6e}".format(
            c0, c1, c2, r2_intercept, p_intercept),
        "source_data_refs": source_refs_all,
        "metrics": {
            "intercept_regression_coeff_F_ext": c1,
            "intercept_regression_coeff_v0": c2,
            "intercept_regression_intercept": c0,
            "intercept_R_squared": r2_intercept,
            "intercept_p_value": p_intercept
        }
    }
    observations.append(obs1)

    # Observation 2: Slope regression
    obs2 = {
        "summary": "斜率关于F_ext和v0的二元线性回归：截距= {:.6f}, 系数(F_ext)={:.6f}, 系数(v0)={:.6f}, R²={:.6f}, p_value={:.6e}".format(
            d0, d1, d2, r2_slope, p_slope),
        "source_data_refs": source_refs_all,
        "metrics": {
            "slope_regression_coeff_F_ext": d1,
            "slope_regression_coeff_v0": d2,
            "slope_regression_intercept": d0,
            "slope_R_squared": r2_slope,
            "slope_p_value": p_slope
        }
    }
    observations.append(obs2)

    # Observation 3: Diffs list and summary
    diffs_formatted = [float(x) for x in diffs]
    obs3 = {
        "summary": f"各实验截距减F_ext差值: {diffs_formatted}; 均值={diff_mean:.6f}, 标准差={diff_std:.6f}。单样本t检验(零均值) t={t_stat:.6f}, p={p_ttest:.6e}",
        "source_data_refs": source_refs_all,
        "metrics": {
            "intercept_minus_F_ext_mean": diff_mean,
            "intercept_minus_F_ext_std": diff_std,
            "intercept_minus_F_ext_ttest_p": p_ttest,
            "intercept_minus_F_ext_ttest_t": t_stat,
            "observation_count": n
        }
    }
    observations.append(obs3)

    # Observation 4: per-experiment a-v regression details (optional but helpful)
    for r in results:
        obs4 = {
            "summary": f"实验{r['exp_id']}: a-v线性回归 slope={r['slope']:.6f}, intercept={r['intercept']:.6f}, F_ext={r['F_ext']}, v0={r['v0']}",
            "source_data_refs": [f"{r['exp_id']}:a", f"{r['exp_id']}:v"],
            "metrics": {
                "exp_id": r['exp_id'],
                "F_ext": r['F_ext'],
                "v0": r['v0'],
                "slope": r['slope'],
                "intercept": r['intercept']
            }
        }
        observations.append(obs4)

    # Build main observation string
    main_obs = (
        f"对{len(exp_ids)}个constant场实验进行a-v线性回归，并对截距和斜率分别对F_ext和v0做二元线性回归。"
        f"截距回归R²={r2_intercept:.4f}, p={p_intercept:.2e}; "
        f"斜率回归R²={r2_slope:.4f}, p={p_slope:.2e}. "
        f"截距减F_ext均值为{diff_mean:.6f}, 标准差{diff_std:.6f}. "
        f"共计{len(observations)}条OBS记录。未宣布任何定律。"
    )

    return {
        "observation": main_obs,
        "observations": observations,
        "derived_series": [],
        "figures": [],
        "metrics": {
            "experiments_processed": len(exp_ids),
            "intercept_regression_R2": r2_intercept,
            "slope_regression_R2": r2_slope,
            "intercept_minus_F_ext_mean": diff_mean,
            "intercept_minus_F_ext_std": diff_std,
            "observation_count": len(observations)
        }
    }
