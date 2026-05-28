import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import List, Dict, Any, Tuple
import math

def _linear_regression_1d(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return slope, intercept, R² for 1D linear regression."""
    A = np.vstack([x, np.ones_like(x)]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = coeffs[0], coeffs[1]
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)

def _compute_statistics(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "count": int(len(values))
    }

def process(payload: dict) -> dict:
    # Extract experiments
    experiments = payload["experiments"]
    # Filter constant field experiments
    constant_exps = {}
    for eid, exp in experiments.items():
        if exp["config"].get("force_field_type") == "constant":
            constant_exps[eid] = exp
    if not constant_exps:
        raise ValueError("No constant field experiments found.")
    
    # Prepare lists
    exp_ids_sorted = sorted(constant_exps.keys())
    results = {}  # eid -> dict
    for eid in exp_ids_sorted:
        exp = constant_exps[eid]
        config = exp["config"]
        series = exp["series"]
        t = np.array(series["t"])
        a = np.array(series["a"])
        v = np.array(series["v"])
        q = np.array(series["q"])
        F_ext = config["F_ext"]
        v0 = config["initial_v"]
        
        # a-v regression
        slope_av, intercept_av, r2_av = _linear_regression_1d(v, a)
        # a-q regression
        slope_aq, _, r2_aq = _linear_regression_1d(q, a)
        # a0 and a_end
        a0 = a[0]
        a_end = a[-1]
        diff0 = a0 - F_ext
        diff_end = a_end - F_ext
        
        results[eid] = {
            "F_ext": F_ext,
            "v0": v0,
            "slope_av": slope_av,
            "intercept_av": intercept_av,
            "slope_aq": slope_aq,
            "a0": a0,
            "a_end": a_end,
            "diff0": diff0,
            "diff_end": diff_end,
            "r2_av": r2_av,
            "r2_aq": r2_aq,
            "t": t,
            "a": a,
            "v": v,
            "q": q
        }
    
    # Prepare data for cross-experiment regressions
    n_exp = len(results)
    F_exts = np.array([r["F_ext"] for r in results.values()])
    v0s = np.array([r["v0"] for r in results.values()])
    intercept_avs = np.array([r["intercept_av"] for r in results.values()])
    slope_avs = np.array([r["slope_av"] for r in results.values()])
    diff0s = np.array([r["diff0"] for r in results.values()])
    diff_ends = np.array([r["diff_end"] for r in results.values()])
    
    # 1. intercept_av vs F_ext simple linear regression
    slope_intercept_F, intercept_intercept_F, r2_intercept_vs_F = _linear_regression_1d(F_exts, intercept_avs)
    
    # 2. slope_av vs F_ext + v0 multiple linear regression
    X_multi = np.column_stack([F_exts, v0s])
    y_slope = slope_avs
    multi_reg = LinearRegression(fit_intercept=True)
    multi_reg.fit(X_multi, y_slope)
    multi_coefs = multi_reg.coef_
    multi_intercept = multi_reg.intercept_
    multi_r2 = multi_reg.score(X_multi, y_slope)
    multi_rmse = np.sqrt(np.mean((y_slope - multi_reg.predict(X_multi))**2))
    
    # 3. Statistics for diff0 and diff_end
    diff0_stats = _compute_statistics(diff0s)
    diff_end_stats = _compute_statistics(diff_ends)
    
    # Build observations
    observations = []
    source_refs_all = []
    for eid in exp_ids_sorted:
        r = results[eid]
        obs_entry = {
            "summary": (
                f"常数场实验 {eid}: "
                f"F_ext={r['F_ext']:.4f}, v0={r['v0']:.4f}, "
                f"slope_av={r['slope_av']:.6f}, intercept_av={r['intercept_av']:.6f}, "
                f"slope_aq={r['slope_aq']:.6f}, "
                f"a0={r['a0']:.6f}, a_end={r['a_end']:.6f}, "
                f"a0-F_ext={r['diff0']:.6f}, a_end-F_ext={r['diff_end']:.6f}"
            ),
            "source_data_refs": [f"{eid}:a", f"{eid}:v", f"{eid}:q"],
            "metrics": {
                "F_ext": r["F_ext"],
                "v0": r["v0"],
                "slope_av": r["slope_av"],
                "intercept_av": r["intercept_av"],
                "slope_aq": r["slope_aq"],
                "a0": r["a0"],
                "a_end": r["a_end"],
                "a0_minus_F_ext": r["diff0"],
                "a_end_minus_F_ext": r["diff_end"],
                "r2_av": r["r2_av"],
                "r2_aq": r["r2_aq"]
            }
        }
        observations.append(obs_entry)
        source_refs_all.append(eid)
    
    # Cross-experiment regression results
    # intercept_av vs F_ext
    observations.append({
        "summary": (
            f"跨实验 intercept_av 对 F_ext 线性回归: "
            f"斜率={slope_intercept_F:.6f}, 截距={intercept_intercept_F:.6f}, R²={r2_intercept_vs_F:.6f}"
        ),
        "source_data_refs": [f"{eid}:a,{eid}:v" for eid in exp_ids_sorted],
        "metrics": {
            "slope_intercept_vs_F": slope_intercept_F,
            "intercept_intercept_vs_F": intercept_intercept_F,
            "r2_intercept_vs_F": r2_intercept_vs_F
        }
    })
    
    # slope_av vs F_ext + v0
    observations.append({
        "summary": (
            f"跨实验 slope_av 对 [F_ext, v0] 多元线性回归: "
            f"系数(intercept)={multi_intercept:.6f}, "
            f"系数(F_ext)={multi_coefs[0]:.6f}, 系数(v0)={multi_coefs[1]:.6f}, "
            f"R²={multi_r2:.6f}, RMSE={multi_rmse:.6f}"
        ),
        "source_data_refs": [f"{eid}:a,{eid}:v" for eid in exp_ids_sorted],
        "metrics": {
            "multi_intercept": multi_intercept,
            "multi_coef_F_ext": multi_coefs[0],
            "multi_coef_v0": multi_coefs[1],
            "multi_r2": multi_r2,
            "multi_rmse": multi_rmse
        }
    })
    
    # diff0 and diff_end statistics
    observations.append({
        "summary": (
            f"a0 - F_ext 差值统计: "
            f"均值={diff0_stats['mean']:.6f}, std={diff0_stats['std']:.6f}, "
            f"min={diff0_stats['min']:.6f}, max={diff0_stats['max']:.6f}"
        ),
        "source_data_refs": [f"{eid}:a" for eid in exp_ids_sorted],
        "metrics": diff0_stats
    })
    
    observations.append({
        "summary": (
            f"a_end - F_ext 差值统计: "
            f"均值={diff_end_stats['mean']:.6f}, std={diff_end_stats['std']:.6f}, "
            f"min={diff_end_stats['min']:.6f}, max={diff_end_stats['max']:.6f}"
        ),
        "source_data_refs": [f"{eid}:a" for eid in exp_ids_sorted],
        "metrics": diff_end_stats
    })
    
    # Main observation summary
    observation_text = (
        f"处理了 {n_exp} 个常数场实验。"
        f"跨实验: intercept_av~F_ext R²={r2_intercept_vs_F:.4f}, "
        f"slope_av~F_ext+v0 R²={multi_r2:.4f}。"
        f"a0-F_ext 均值={diff0_stats['mean']:.4f}, "
        f"a_end-F_ext 均值={diff_end_stats['mean']:.4f}。"
        f"共生成 {len(observations)} 条 OBS。"
    )
    
    return {
        "observation": observation_text,
        "derived_series": [],  # no derived series needed
        "observations": observations,
        "validations": [],
        "figures": [],
        "metrics": {
            "experiment_count": n_exp,
            "observation_count": len(observations),
            "r2_intercept_av_vs_F_ext": r2_intercept_vs_F,
            "r2_slope_av_multi": multi_r2,
            "diff0_mean": diff0_stats["mean"],
            "diff_end_mean": diff_end_stats["mean"],
            "diff0_std": diff0_stats["std"],
            "diff_end_std": diff_end_stats["std"]
        }
    }
