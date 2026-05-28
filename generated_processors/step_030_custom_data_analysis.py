import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import signal
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process(payload: dict) -> dict:
    params = payload["parameters"]
    exp_ids = params.get("experiment_ids", [])
    optional_series = params.get("optional_series", [])
    expected_outputs = params.get("expected_outputs", [])
    experiments = payload["experiments"]
    output_dir = Path(payload["output_dir"])

    # ---- collect per-experiment data ----
    data = {}
    for eid in exp_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        cfg = exp["config"]
        series = exp["series"]
        avail = exp["available_series"]

        # F_ext
        force_type = cfg.get("force_field_type", "")
        if force_type == "free":
            F_ext = 0.0
        elif "F_ext" in cfg and cfg["F_ext"] is not None:
            F_ext = cfg["F_ext"]
        elif "constant_force" in cfg:
            F_ext = cfg["constant_force"]
        else:
            F_ext = 0.0
        if F_ext == 0.0:
            continue   # skip free experiments

        # acceleration
        if "a_sg" in avail:
            a = np.array(series["a_sg"], dtype=float)
        elif "a_new" in avail:
            a = np.array(series["a_new"], dtype=float)
        elif "a_est_sg" in avail:
            a = np.array(series["a_est_sg"], dtype=float)
        elif "a_est" in avail:
            a = np.array(series["a_est"], dtype=float)
        else:
            raise ValueError(f"Experiment {eid}: no acceleration series available")

        # velocity
        if "v_sg" in avail:
            v = np.array(series["v_sg"], dtype=float)
        elif "v_new" in avail:
            v = np.array(series["v_new"], dtype=float)
        elif "v_est_sg" in avail:
            v = np.array(series["v_est_sg"], dtype=float)
        elif "v_est" in avail:
            v = np.array(series["v_est"], dtype=float)
        else:
            raise ValueError(f"Experiment {eid}: no velocity series available")

        # trim to common length
        n = min(len(a), len(v))
        a = a[:n]
        v = v[:n]

        t = np.array(series.get("t", np.arange(n)), dtype=float)
        if len(t) > n:
            t = t[:n]

        data[eid] = {
            "a": a,
            "v": v,
            "F_ext": float(F_ext),
            "a_norm": a / F_ext,
            "t": t
        }

    if not data:
        raise ValueError("No experiment with nonzero F_ext found")

    # ---- global arrays ----
    all_v = np.concatenate([d["v"] for d in data.values()])
    all_a = np.concatenate([d["a"] for d in data.values()])
    all_a_norm = np.concatenate([d["a_norm"] for d in data.values()])
    all_F = np.concatenate([np.full(len(d["a"]), d["F_ext"]) for d in data.values()])

    # ---- helper for robust correlation ----
    def robust_corr(x, y):
        if np.std(y) == 0 or np.std(x) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    # ---- model definitions (on a_norm) ----
    def lin_model(v, c):
        return 1 - c * v

    def exp_model(v, b):
        return np.exp(-b * v)

    def rat_model(v, d):
        return 1 / (1 + d * v)

    # ---- fitting ----
    results = {}
    y_mean = float(np.mean(all_a_norm))
    ss_tot = float(np.sum((all_a_norm - y_mean) ** 2))

    # ----- linear model -----
    try:
        popt_lin, pcov_lin = curve_fit(lin_model, all_v, all_a_norm, p0=[0.1])
        c_opt = popt_lin[0]
        c_se = float(np.sqrt(pcov_lin[0, 0])) if pcov_lin[0, 0] > 0 else None
        pred_lin = lin_model(all_v, c_opt)
        resid_lin = all_a_norm - pred_lin
        ss_res = float(np.sum(resid_lin ** 2))
        r2_lin = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse_lin = float(np.sqrt(np.mean(resid_lin ** 2)))

        idx = 0
        exp_res_lin = {}
        for eid, d in data.items():
            n = len(d["v"])
            r = resid_lin[idx:idx + n]
            exp_res_lin[eid] = {
                "mean": float(np.mean(r)),
                "std": float(np.std(r)),
                "min": float(np.min(r)),
                "max": float(np.max(r)),
                "corr_v_residual": robust_corr(d["v"], r)
            }
            idx += n
        results["linear"] = {
            "c": c_opt,
            "c_se": c_se,
            "R2": r2_lin,
            "RMSE": rmse_lin,
            "per_exp": exp_res_lin
        }
    except Exception as e:
        results["linear"] = None
        # print(f"Linear fit failed: {e}")

    # ----- exponential model -----
    try:
        popt_exp, pcov_exp = curve_fit(exp_model, all_v, all_a_norm, p0=[0.5])
        b_opt = popt_exp[0]
        b_se = float(np.sqrt(pcov_exp[0, 0])) if pcov_exp[0, 0] > 0 else None
        pred_exp = exp_model(all_v, b_opt)
        resid_exp = all_a_norm - pred_exp
        ss_res = float(np.sum(resid_exp ** 2))
        r2_exp = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse_exp = float(np.sqrt(np.mean(resid_exp ** 2)))

        idx = 0
        exp_res_exp = {}
        for eid, d in data.items():
            n = len(d["v"])
            r = resid_exp[idx:idx + n]
            exp_res_exp[eid] = {
                "mean": float(np.mean(r)),
                "std": float(np.std(r)),
                "min": float(np.min(r)),
                "max": float(np.max(r)),
                "corr_v_residual": robust_corr(d["v"], r)
            }
            idx += n
        results["exponential"] = {
            "b": b_opt,
            "b_se": b_se,
            "R2": r2_exp,
            "RMSE": rmse_exp,
            "per_exp": exp_res_exp
        }
    except Exception as e:
        results["exponential"] = None

    # ----- rational model -----
    try:
        popt_rat, pcov_rat = curve_fit(rat_model, all_v, all_a_norm, p0=[0.1])
        d_opt = popt_rat[0]
        d_se = float(np.sqrt(pcov_rat[0, 0])) if pcov_rat[0, 0] > 0 else None
        pred_rat = rat_model(all_v, d_opt)
        resid_rat = all_a_norm - pred_rat
        ss_res = float(np.sum(resid_rat ** 2))
        r2_rat = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse_rat = float(np.sqrt(np.mean(resid_rat ** 2)))

        idx = 0
        exp_res_rat = {}
        for eid, d in data.items():
            n = len(d["v"])
            r = resid_rat[idx:idx + n]
            exp_res_rat[eid] = {
                "mean": float(np.mean(r)),
                "std": float(np.std(r)),
                "min": float(np.min(r)),
                "max": float(np.max(r)),
                "corr_v_residual": robust_corr(d["v"], r)
            }
            idx += n
        results["rational"] = {
            "d": d_opt,
            "d_se": d_se,
            "R2": r2_rat,
            "RMSE": rmse_rat,
            "per_exp": exp_res_rat
        }
    except Exception as e:
        results["rational"] = None

    # ---- figures ----
    fig_paths = []

    # 1) per‑experiment a vs v with model curves
    n_exp = len(data)
    n_cols = min(3, n_exp)
    n_rows = (n_exp + n_cols - 1) // n_cols
    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_exp > 1 else [axes]
    for i, (eid, d) in enumerate(data.items()):
        ax = axes[i]
        v = d["v"]
        a = d["a"]
        F = d["F_ext"]
        ax.scatter(v, a, s=8, label=f"data (F={F})", color="black", alpha=0.6)
        v_sorted = np.sort(v)
        if results.get("linear"):
            c = results["linear"]["c"]
            ax.plot(v_sorted, F * (1 - c * v_sorted), label=f"linear c={c:.3f}", linestyle="--")
        if results.get("exponential"):
            b = results["exponential"]["b"]
            ax.plot(v_sorted, F * np.exp(-b * v_sorted), label=f"exp b={b:.3f}", linestyle=":")
        if results.get("rational"):
            d_param = results["rational"]["d"]
            ax.plot(v_sorted, F / (1 + d_param * v_sorted), label=f"rat d={d_param:.3f}", linestyle="-.")
        ax.set_xlabel("v")
        ax.set_ylabel("a")
        ax.set_title(f"{eid}")
        ax.legend(fontsize=8)
    for j in range(len(data), len(axes)):
        axes[j].axis("off")
    fig1.tight_layout()
    path1 = output_dir / "a_vs_v_with_fits.png"
    fig1.savefig(path1)
    plt.close(fig1)
    fig_paths.append(str(path1))

    # 2) a_norm vs v scatter + fits
    fig2, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n_exp))
    for i, (eid, d) in enumerate(data.items()):
        ax.scatter(d["v"], d["a_norm"], s=10, color=colors[i], label=f"{eid} (F={d['F_ext']})", alpha=0.7, edgecolors="none")
    v_grid = np.linspace(min(all_v), max(all_v), 500)
    if results.get("linear"):
        c = results["linear"]["c"]
        ax.plot(v_grid, 1 - c * v_grid, "r--", label=f"linear: 1 - {c:.4f}v")
    if results.get("exponential"):
        b = results["exponential"]["b"]
        ax.plot(v_grid, np.exp(-b * v_grid), "g--", label=f"exp: exp(-{b:.4f}v)")
    if results.get("rational"):
        d_param = results["rational"]["d"]
        ax.plot(v_grid, 1 / (1 + d_param * v_grid), "b--", label=f"rat: 1/(1+{d_param:.4f}v)")
    ax.set_xlabel("v")
    ax.set_ylabel("a / F_ext")
    ax.set_title("Normalized acceleration vs velocity")
    ax.legend(fontsize=9)
    fig2.tight_layout()
    path2 = output_dir / "anorm_vs_v_with_fits.png"
    fig2.savefig(path2)
    plt.close(fig2)
    fig_paths.append(str(path2))

    # ---- metrics ----
    metrics = {}
    for model_name, res in results.items():
        if res is None:
            metrics[f"{model_name}_status"] = "failed"
            continue
        prefix = model_name + "_"
        metrics[prefix + "status"] = "success"
        for key in ("c", "b", "d"):
            if key in res:
                metrics[prefix + key] = res[key]
                if res.get(key + "_se") is not None:
                    metrics[prefix + key + "_se"] = res[key + "_se"]
        metrics[prefix + "R2"] = res["R2"]
        metrics[prefix + "RMSE"] = res["RMSE"]
        for eid, rst in res["per_exp"].items():
            for stat_key, val in rst.items():
                metrics[prefix + f"{eid}_{stat_key}"] = val

    # ---- derived series (a_norm) ----
    derived_series = []
    for eid, d in data.items():
        derived_series.append({
            "experiment_id": eid,
            "name": "a_norm",
            "values": d["a_norm"].tolist(),
            "source_name": f"a / F_ext (F_ext={d['F_ext']})",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"normalized acceleration for experiment {eid}"
        })

    # ---- observation ----
    lines = [f"对实验 {exp_ids} 进行了三种全局模型拟合（a = F_ext * f(v)），所有实验数据合并，允许不同F_ext:"]
    for model_name in ("linear", "exponential", "rational"):
        res = results.get(model_name)
        if res is None:
            lines.append(f"  {model_name} 拟合失败")
            continue
        if model_name == "linear":
            param_str = f"c = {res['c']:.4f} ± {res.get('c_se', '?'):.4f}"
        elif model_name == "exponential":
            param_str = f"b = {res['b']:.4f} ± {res.get('b_se', '?'):.4f}"
        else:
            param_str = f"d = {res['d']:.4f} ± {res.get('d_se', '?'):.4f}"
        lines.append(f"  {model_name}: {param_str}, R²={res['R2']:.4f}, RMSE={res['RMSE']:.4f}")
    lines.append("各实验残差统计（模型预测与a_norm的差异）:")
    for eid, d in data.items():
        sub = []
        for model_name in ("linear", "exponential", "rational"):
            res = results.get(model_name)
            if res is None:
                continue
            rst = res["per_exp"].get(eid)
            if rst is None:
                continue
            sub.append(f"{model_name}: mean={rst['mean']:.4f}, std={rst['std']:.4f}, corr(v,res)={rst['corr_v_residual']:.4f}")
        if sub:
            lines.append(f"  {eid} (F_ext={d['F_ext']}): " + "; ".join(sub))
    lines.append(f"图像：a_vs_v_with_fits.png, anorm_vs_v_with_fits.png")
    lines.append("返回了派生序列 a_norm（每个实验的归一化加速度）。拟合结果可用于判断最佳全局模型形式。")

    return {
        "observation": "\n".join(lines),
        "derived_series": derived_series,
        "figures": fig_paths,
        "metrics": metrics
    }
