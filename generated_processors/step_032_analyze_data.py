import json
import math
import statistics
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # Determine which experiments to process
    exp_ids = params.get("experiment_ids")
    if exp_ids is None:
        exp_ids = list(experiments.keys())
    elif isinstance(exp_ids, str):
        exp_ids = [exp_ids]

    free_exps = []
    const_exps = []
    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
        cfg = experiments[eid]["config"]
        if cfg.get("force_field_type") == "free":
            free_exps.append(eid)
        else:
            const_exps.append(eid)

    # 1) Free field acceleration check
    free_a_values = []
    free_t_values = []
    for eid in free_exps:
        exp = experiments[eid]
        # find acceleration series
        avail = exp.get("available_series", [])
        a_name = None
        for s in avail:
            if s.startswith("a_gradient"):
                a_name = s
                break
        if a_name is None:
            # fallback: compute from q
            q = np.array(exp["series"]["q"])
            t = np.array(exp["series"]["t"])
            if len(t) < 3:
                continue
            a = np.gradient(np.gradient(q, t, edge_order=2), t, edge_order=2)
        else:
            a = np.array(exp["series"][a_name])
        t = np.array(exp["series"]["t"])
        free_a_values.append(a)
        free_t_values.append(t)
    if len(free_a_values) == 0:
        free_mean_a = np.nan
        free_max_abs_a = np.nan
    else:
        all_free_a = np.concatenate(free_a_values)
        free_mean_a = float(np.mean(all_free_a))
        free_max_abs_a = float(np.max(np.abs(all_free_a)))

    # 2) Constant field exponential fit
    # Collect data: v, a, F_ext
    v_all = []
    a_all = []
    F_all = []
    per_exp_data = {}  # for generating per-experiment metrics
    for eid in const_exps:
        exp = experiments[eid]
        cfg = exp["config"]
        F_ext = float(cfg["F_ext"])
        avail = exp.get("available_series", [])
        # acceleration
        a_name = None
        for s in avail:
            if s.startswith("a_gradient"):
                a_name = s
                break
        if a_name is None:
            q = np.array(exp["series"]["q"])
            t = np.array(exp["series"]["t"])
            a_smooth = np.gradient(np.gradient(q, t, edge_order=2), t, edge_order=2)
        else:
            a_smooth = np.array(exp["series"][a_name])
        # velocity
        v_name = None
        for s in avail:
            if s.startswith("v_gradient"):
                v_name = s
                break
        if v_name is None:
            q = np.array(exp["series"]["q"])
            t = np.array(exp["series"]["t"])
            v_smooth = np.gradient(q, t, edge_order=2)
        else:
            v_smooth = np.array(exp["series"][v_name])
        # ensure same length
        t = np.array(exp["series"]["t"])
        if len(a_smooth) != len(t) or len(v_smooth) != len(t):
            raise ValueError(f"Length mismatch in {eid}: t={len(t)}, a={len(a_smooth)}, v={len(v_smooth)}")
        v_all.append(v_smooth)
        a_all.append(a_smooth)
        F_all.append(np.full(len(t), F_ext))
        per_exp_data[eid] = {
            "t": t,
            "v": v_smooth,
            "a": a_smooth,
            "F_ext": F_ext
        }
    if len(v_all) == 0:
        gamma_opt = np.nan
        R2 = np.nan
        RMSE = np.nan
    else:
        v_concat = np.concatenate(v_all)
        a_concat = np.concatenate(a_all)
        F_concat = np.concatenate(F_all)

        def model_func(x, gamma):
            v = x[0]
            F = x[1]
            return F * np.exp(-gamma * np.abs(v))

        xdata = np.vstack([v_concat, F_concat])
        p0 = [0.8]
        try:
            popt, _ = curve_fit(model_func, xdata, a_concat, p0=p0, bounds=(0, np.inf))
        except Exception as e:
            raise ValueError(f"Nonlinear fit failed: {e}")
        gamma_opt = float(popt[0])
        a_pred = F_concat * np.exp(-gamma_opt * np.abs(v_concat))
        RSS = np.sum((a_concat - a_pred) ** 2)
        TSS = np.sum((a_concat - np.mean(a_concat)) ** 2)
        R2 = float(1 - RSS / TSS)
        RMSE = float(np.sqrt(RSS / len(a_concat)))

    # 3) Determine support
    free_ok = free_mean_a < 1e-6
    fit_ok = R2 > 0.9  # simple threshold
    supports = bool(free_ok and fit_ok)

    # Prepare validation entry
    metric_values = {
        "free_field_mean_a": free_mean_a,
        "free_field_max_abs_a": free_max_abs_a,
        "global_gamma": gamma_opt,
        "global_R2": R2,
        "global_RMSE": RMSE
    }
    # per-experiment RMSE (optional)
    per_exp_rmse = {}
    for eid, d in per_exp_data.items():
        v_e = d["v"]
        a_e = d["a"]
        F_e = d["F_ext"]
        pred_e = F_e * np.exp(-gamma_opt * np.abs(v_e))
        rmse_e = float(np.sqrt(np.mean((a_e - pred_e) ** 2)))
        per_exp_rmse[eid] = rmse_e
    metric_values["per_experiment_RMSE"] = per_exp_rmse

    source_refs = []
    for eid in free_exps:
        exp = experiments[eid]
        avail = exp.get("available_series", [])
        a_name = next((s for s in avail if s.startswith("a_gradient")), None)
        if a_name:
            source_refs.append(f"{eid}:{a_name}")
    for eid in const_exps:
        exp = experiments[eid]
        avail = exp.get("available_series", [])
        a_name = next((s for s in avail if s.startswith("a_gradient")), None)
        v_name = next((s for s in avail if s.startswith("v_gradient")), None)
        if a_name:
            source_refs.append(f"{eid}:{a_name}")
        if v_name:
            source_refs.append(f"{eid}:{v_name}")

    validations = [{
        "hypothesis_id": "H005",
        "experiment_ids": sorted(free_exps + const_exps),
        "supports": supports,
        "metric_name": "free_field_acc_mean_max_abs_and_global_exponential_fit",
        "metric_values": metric_values,
        "aggregate_score": float(R2) if not np.isnan(R2) else 0.0,
        "summary": f"Free field acceleration mean={free_mean_a:.3e}, max_abs={free_max_abs_a:.3e}; "
                   f"Global fit: gamma={gamma_opt:.6f}, R2={R2:.6f}, RMSE={RMSE:.6f}. "
                   f"Support={supports}.",
        "source_data_refs": source_refs
    }]

    # Optionally generate residual sequences for constant fields
    derived_series = []
    for eid, d in per_exp_data.items():
        v_e = d["v"]
        F_e = d["F_ext"]
        pred_e = F_e * np.exp(-gamma_opt * np.abs(v_e))
        residual = (d["a"] - pred_e).tolist()
        derived_series.append({
            "experiment_id": eid,
            "name": f"residual_H005_{eid}",
            "values": residual,
            "source_name": f"a - F_ext*exp(-gamma*|v|) with gamma={gamma_opt:.6f}",
            "provenance": f"generated data processor: step_{payload.get('step_index', 'unknown')}_analyze_data",
            "description": "Residual to H005 exponential model"
        })

    # Observation summary
    observation = (
        f"验证假说H005: a = 0 if free else F_ext * exp(-gamma*|v|)。"
        f"自由场实验({', '.join(sorted(free_exps))})加速度均值={free_mean_a:.4e}, 最大绝对值={free_max_abs_a:.4e}; "
        f"恒定场实验({len(const_exps)}个)全局拟合gamma={gamma_opt:.6f}, R²={R2:.6f}, RMSE={RMSE:.6f}. "
        f"支持假说: {supports}。"
    )

    return {
        "observation": observation,
        "derived_series": derived_series,
        "observations": [{
            "summary": observation,
            "source_data_refs": source_refs,
            "metrics": {
                "free_field_mean_a": free_mean_a,
                "free_field_max_abs_a": free_max_abs_a,
                "global_gamma": gamma_opt,
                "global_R2": R2,
                "global_RMSE": RMSE,
                "supports": int(supports)
            }
        }],
        "validations": validations,
        "figures": [],
        "metrics": {
            "free_mean_a": free_mean_a,
            "free_max_abs_a": free_max_abs_a,
            "gamma": gamma_opt,
            "R2": R2,
            "RMSE": RMSE,
            "supports": int(supports)
        }
    }
