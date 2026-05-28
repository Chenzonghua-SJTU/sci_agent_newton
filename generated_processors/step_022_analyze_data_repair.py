import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    params = payload["parameters"]
    experiment_ids = params["experiment_ids"]
    experiments = payload["experiments"]
    output_dir = Path(payload["output_dir"])

    # ----- 自由场加速度统计 -----
    free_a_arrays = []
    free_exp_ids = []
    for eid, exp in experiments.items():
        cfg = exp["config"]
        if cfg["force_field_type"] != "free":
            continue
        a_name = None
        for sname in exp.get("available_series", []):
            if "a_gradient_ledger" in sname:
                a_name = sname
                break
        if a_name is None or a_name not in exp.get("series", {}):
            continue
        a_vals = np.array(exp["series"][a_name])
        if len(a_vals) == 0:
            continue
        free_a_arrays.append(a_vals)
        free_exp_ids.append(eid)

    if free_a_arrays:
        free_all_a = np.concatenate(free_a_arrays)
        free_mean_a = float(np.mean(free_all_a))
        free_max_abs_a = float(np.max(np.abs(free_all_a)))
    else:
        free_mean_a = 0.0
        free_max_abs_a = 0.0

    # ----- 收集恒定外力实验数据 -----
    x_all = []          # |v|
    y_all = []          # a / F_ext
    a_all = []          # 原始加速度
    fext_all = []       # F_ext
    per_exp_data = {}   # eid -> {v, a, F_ext}
    skipped = []

    for eid in experiment_ids:
        exp = experiments.get(eid)
        if exp is None:
            skipped.append(eid)
            continue
        cfg = exp["config"]
        F_ext = cfg["F_ext"]
        if F_ext == 0.0:
            skipped.append(eid)
            continue

        v_name = None
        a_name = None
        for sname in exp.get("available_series", []):
            if "v_gradient_ledger" in sname:
                v_name = sname
            if "a_gradient_ledger" in sname:
                a_name = sname
        if v_name is None or a_name is None:
            skipped.append(eid)
            continue
        v_vals = np.array(exp["series"].get(v_name, []))
        a_vals = np.array(exp["series"].get(a_name, []))
        if len(v_vals) == 0 or len(a_vals) == 0:
            skipped.append(eid)
            continue
        # 移除无效点
        mask = ~(np.isnan(v_vals) | np.isnan(a_vals))
        v_vals = v_vals[mask]
        a_vals = a_vals[mask]
        if len(v_vals) == 0:
            skipped.append(eid)
            continue

        x_all.append(np.abs(v_vals))
        y_all.append(a_vals / F_ext)
        a_all.append(a_vals)
        fext_all.append(np.full_like(v_vals, F_ext))
        per_exp_data[eid] = {"v": v_vals, "a": a_vals, "F_ext": F_ext}

    if len(x_all) == 0:
        return {
            "observation": "没有可用数据进行拟合",
            "metrics": {"data_points": 0, "task_under_specified": True}
        }

    X = np.concatenate(x_all)
    Y = np.concatenate(y_all)
    A_obs = np.concatenate(a_all)
    F_exts = np.concatenate(fext_all)

    # ----- 非线性拟合 y = exp(-gamma * x) -----
    def model(x, gamma):
        return np.exp(-gamma * x)

    try:
        popt, _ = curve_fit(model, X, Y, p0=[0.1])
    except Exception as e:
        return {
            "observation": f"非线性拟合失败: {str(e)}",
            "metrics": {"fit_error": str(e)}
        }

    gamma_fit = float(popt[0])
    Y_pred = model(X, gamma_fit)

    # ----- R² & RMSE -----
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    # 原始加速度 RMSE
    A_pred = F_exts * Y_pred
    rmse = float(np.sqrt(np.mean((A_obs - A_pred) ** 2)))

    # ----- 每个实验残差及 RMSE -----
    residuals = {}
    per_exp_rmse = {}
    for eid, d in per_exp_data.items():
        v = d["v"]
        a = d["a"]
        F = d["F_ext"]
        a_pred = F * np.exp(-gamma_fit * np.abs(v))
        resid = a - a_pred
        residuals[eid] = resid.tolist()
        per_exp_rmse[eid] = float(np.sqrt(np.mean(resid ** 2)))

    # ----- 支持性判定 -----
    supports = bool(r2 > 0.9 and abs(free_mean_a) < 1e-6)

    # ----- 构建 validations -----
    metric_values = {
        "gamma": gamma_fit,
        "R2": r2,
        "RMSE": rmse,
        "free_field_mean_a": free_mean_a,
        "free_field_max_abs_a": free_max_abs_a,
    }
    for eid, rmse_val in per_exp_rmse.items():
        metric_values[f"RMSE_{eid}"] = rmse_val

    source_refs = []
    for eid in experiment_ids:
        if eid not in skipped:
            source_refs.append(f"{eid}:v_gradient_ledger, a_gradient_ledger")
    if free_exp_ids:
        source_refs.append("free_field_experiments:" + ",".join(free_exp_ids))

    validations = [
        {
            "hypothesis_id": "H002",
            "experiment_ids": experiment_ids,
            "supports": supports,
            "metric_name": "global_exponential_fit_with_free_field_check",
            "metric_values": metric_values,
            "aggregate_score": r2,
            "summary": (
                f"验证假说 H002（a = 0 if free else F_ext * exp(-gamma*|v|)）。"
                f"全局拟合 gamma = {gamma_fit:.6f}, R² = {r2:.6f}, RMSE = {rmse:.6f}。"
                f"自由场加速度均值 = {free_mean_a:.2e}, 最大绝对值 = {free_max_abs_a:.2e}。"
                f"支持假说 = {supports}。"
            ),
            "source_data_refs": source_refs,
        }
    ]

    # ----- 派生序列：残差（使用新名称以避免与已有序列冲突） -----
    derived_series = []
    for eid, resid_vals in residuals.items():
        # 使用新命名规则：residual_H002_revalidate_<eid> 并在 provenance 中说明差异
        derived_series.append({
            "experiment_id": eid,
            "name": f"residual_H002_revalidate_{eid}",
            "values": resid_vals,
            "source_name": f"a - F_ext * exp(-{gamma_fit:.6f} * |v|)",
            "provenance": "generated data processor: validate_H002 revalidation; derived with global gamma, differs from earlier residual_H002_* sequences",
            "description": f"假说 H002 重新验证残差 (全局 gamma)"
        })

    # ----- observation 记录 -----
    obs_metrics = {
        "gamma": gamma_fit,
        "R2": r2,
        "RMSE": rmse,
        "free_field_mean_a": free_mean_a,
        "free_field_max_abs_a": free_max_abs_a,
        "supports": supports,
        "experiments_used": len(per_exp_data),
        "total_data_points": len(X),
    }
    observations = [
        {
            "summary": (
                f"假说 H002 重新验证: 对 {len(per_exp_data)} 个恒定外力实验拟合 "
                f"a = F_ext * exp(-gamma*|v|)，全局 gamma = {gamma_fit:.6f}, "
                f"R² = {r2:.6f}, RMSE = {rmse:.6f}。"
                f"自由场加速度均值 = {free_mean_a:.2e}, 绝对值最大值 = {free_max_abs_a:.2e}。"
                f"支持假说 = {supports}。"
            ),
            "source_data_refs": source_refs,
            "metrics": obs_metrics,
        }
    ]

    # ----- 绘图 -----
    fig_path = output_dir / "H002_exponential_fit_revalidate.png"
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, s=1, alpha=0.5, label='Data (a / F_ext)')
    x_plot = np.linspace(0, float(np.max(X)), 200)
    plt.plot(x_plot, model(x_plot, gamma_fit), 'r-', label=f'fit: gamma = {gamma_fit:.4f}')
    plt.xlabel('|v|')
    plt.ylabel('a / F_ext')
    plt.title(f'H002 exponential fit (global): R² = {r2:.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    figures = [str(fig_path)]

    return {
        "observation": observations[0]["summary"],
        "derived_series": derived_series,
        "observations": observations,
        "validations": validations,
        "figures": figures,
        "metrics": obs_metrics,
    }
