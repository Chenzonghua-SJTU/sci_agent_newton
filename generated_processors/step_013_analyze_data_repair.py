import json
import math
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Any, List

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    parameters = payload["parameters"]
    experiment_ids = parameters.get("experiment_ids", [])
    hypothesis_id = parameters.get("hypothesis_id", "H002")

    constant_exp_ids = ["exp_02", "exp_03", "exp_05", "exp_06", "exp_07",
                        "exp_08", "exp_09", "exp_10", "exp_13", "exp_14",
                        "exp_16", "exp_18"]
    free_exp_ids = ["exp_01", "exp_04", "exp_11", "exp_12", "exp_15", "exp_17"]

    experiments = payload["experiments"]

    def get_series_values(exp: dict, series_name: str) -> list:
        series_dict = exp.get("series", {})
        if series_name not in series_dict:
            raise ValueError(f"Series '{series_name}' not found in experiment")
        raw = series_dict[series_name]
        if isinstance(raw, dict):
            if "values" in raw:
                return raw["values"]
            else:
                raise ValueError(f"Series '{series_name}' is a dict but missing 'values' key")
        return raw

    def get_a_v_series(exp_id: str) -> tuple:
        exp = experiments[exp_id]
        series = exp["series"]
        avail = exp["available_series"]

        a_name = None
        v_name = None
        for name in avail:
            if "a_gradient_ledger" in name and exp_id in name:
                a_name = name
            if "v_gradient_ledger" in name and exp_id in name:
                v_name = name

        if a_name is None or v_name is None:
            raise ValueError(f"Experiment {exp_id}: cannot find a_gradient_ledger or v_gradient_ledger series in available_series: {avail}")

        a_raw = get_series_values(exp, a_name)
        v_raw = get_series_values(exp, v_name)
        a = np.array(a_raw, dtype=float)
        v = np.array(v_raw, dtype=float)
        if len(a) == 0 or len(v) == 0:
            raise ValueError(f"Experiment {exp_id}: empty series")
        if len(a) != len(v):
            raise ValueError(f"Experiment {exp_id}: series length mismatch a={len(a)} v={len(v)}")
        return a, v

    free_a_means = {}
    for exp_id in free_exp_ids:
        exp = experiments[exp_id]
        if exp["config"]["force_field_type"] != "free":
            continue
        a, _ = get_a_v_series(exp_id)
        mean_a = np.mean(np.abs(a))
        free_a_means[exp_id] = float(mean_a)

    all_free_a = np.abs(np.concatenate([get_a_v_series(eid)[0] for eid in free_exp_ids]))
    free_mean_abs_a = float(np.mean(all_free_a))

    per_exp_results = {}
    all_x = []
    all_y = []

    for exp_id in constant_exp_ids:
        exp = experiments[exp_id]
        F_ext = float(exp["config"]["F_ext"])
        if abs(F_ext) < 1e-12:
            raise ValueError(f"Constant experiment {exp_id} has F_ext={F_ext}, expected non-zero")
        a, v = get_a_v_series(exp_id)
        sign_F = np.sign(F_ext)
        v_adj = v * sign_F
        ratio = a / F_ext
        mask = ratio > 1e-12
        if np.sum(mask) < 2:
            raise ValueError(f"Experiment {exp_id}: insufficient valid points (a/F_ext>0) for fitting")
        x = v_adj[mask]
        y = np.log(ratio[mask])

        gamma_est = -float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else 0.0
        a_pred = F_ext * np.exp(-gamma_est * v_adj)
        r2 = r2_score(a, a_pred)
        rmse = math.sqrt(mean_squared_error(a, a_pred))

        per_exp_results[exp_id] = {"gamma": gamma_est, "R2": r2, "RMSE": rmse}
        all_x.append(x)
        all_y.append(y)

    X_global = np.concatenate(all_x)
    Y_global = np.concatenate(all_y)
    if len(X_global) == 0:
        raise ValueError("No valid data points for global fit")
    gamma_global = -float(np.dot(X_global, Y_global) / np.dot(X_global, X_global))

    all_a_true = []
    all_a_pred = []
    for exp_id in constant_exp_ids:
        exp = experiments[exp_id]
        F_ext = float(exp["config"]["F_ext"])
        a, v = get_a_v_series(exp_id)
        v_adj = v * np.sign(F_ext)
        a_pred = F_ext * np.exp(-gamma_global * v_adj)
        all_a_true.append(a)
        all_a_pred.append(a_pred)
    a_true_global = np.concatenate(all_a_true)
    a_pred_global = np.concatenate(all_a_pred)
    global_R2 = r2_score(a_true_global, a_pred_global)
    global_RMSE = math.sqrt(mean_squared_error(a_true_global, a_pred_global))

    supports = (global_R2 > 0.9) and (free_mean_abs_a < 1e-10)

    metric_values = {}
    for exp_id in per_exp_results:
        res = per_exp_results[exp_id]
        metric_values[exp_id] = {"gamma": res["gamma"], "R2": res["R2"], "RMSE": res["RMSE"]}
    for exp_id in free_exp_ids:
        metric_values[exp_id] = {"abs_a_mean": free_a_means.get(exp_id, None)}

    aggregate_score = {
        "global_gamma": gamma_global,
        "global_R2": global_R2,
        "global_RMSE": global_RMSE,
        "free_mean_abs_a": free_mean_abs_a,
        "supports": supports
    }

    source_data_refs = []
    for exp_id in constant_exp_ids + free_exp_ids:
        exp = experiments[exp_id]
        for name in exp["available_series"]:
            if "a_gradient_ledger" in name or "v_gradient_ledger" in name:
                source_data_refs.append(f"{exp_id}:{name}")

    validation_entry = {
        "hypothesis_id": hypothesis_id,
        "experiment_ids": constant_exp_ids + free_exp_ids,
        "supports": supports,
        "metric_name": "H002_validation_details",
        "metric_values": metric_values,
        "aggregate_score": aggregate_score,
        "summary": (
            f"Validation of H002: constant field: each experiment fitted model a = F_ext * exp(-gamma * v_adj). "
            f"Global gamma = {gamma_global:.4f}, global R2 = {global_R2:.4f}, global RMSE = {global_RMSE:.4f}. "
            f"Free field: mean(|a|) = {free_mean_abs_a:.2e}. "
            f"Global R2 > 0.9 and free field mean(|a|) < 1e-10: {supports}."
        ),
        "source_data_refs": source_data_refs
    }

    observation_lines = []
    observation_lines.append(f"验证假说 H002 完成。")
    observation_lines.append(f"常量实验拟合结果:")
    for exp_id in sorted(per_exp_results.keys()):
        res = per_exp_results[exp_id]
        observation_lines.append(
            f"  {exp_id}: gamma={res['gamma']:.4f}, R2={res['R2']:.4f}, RMSE={res['RMSE']:.4f}"
        )
    observation_lines.append(f"全局拟合: gamma={gamma_global:.4f}, R2={global_R2:.4f}, RMSE={global_RMSE:.4f}")
    observation_lines.append(f"自由场实验: 各实验 |a| 均值:")
    for exp_id in free_exp_ids:
        mu = free_a_means.get(exp_id, None)
        if mu is not None:
            observation_lines.append(f"  {exp_id}: {mu:.2e}")
    observation_lines.append(f"自由场整体 |a| 均值 = {free_mean_abs_a:.2e}")
    observation_lines.append(f"判断: global_R2 > 0.9 且 free_mean_abs_a < 1e-10 => supports = {supports}")
    observation = "\n".join(observation_lines)

    result = {
        "observation": observation,
        "validations": [validation_entry],
        "metrics": {
            "constant_experiment_count": len(constant_exp_ids),
            "free_experiment_count": len(free_exp_ids),
            "global_gamma": gamma_global,
            "global_R2": global_R2,
            "global_RMSE": global_RMSE,
            "free_mean_abs_a": free_mean_abs_a,
            "supports": supports
        },
        "derived_series": [],
        "observations": [],
        "figures": []
    }
    return result
