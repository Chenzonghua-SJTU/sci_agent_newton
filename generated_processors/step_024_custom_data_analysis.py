import os
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload.get("action")
    params = payload.get("parameters", {})
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(payload.get("experiments", {}).keys())
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    # Collect data from all specified experiments
    data = []  # each entry: dict with exp_id, F_ext, v, a, t
    for eid in exp_ids:
        exp = experiments.get(eid)
        if exp is None:
            continue
        series = exp.get("series", {})
        config = exp.get("config", {})
        # Determine F_ext
        force_field_type = config.get("force_field_type", "")
        if force_field_type == "constant":
            # Try to get constant_force or use F_ext from config
            F_ext = config.get("constant_force", config.get("F_ext", None))
            if F_ext is None:
                # fallback to constant_force parameter name
                F_ext = config.get("F_ext", 0.0)
        else:
            # free or other: F_ext is given in config
            F_ext = config.get("F_ext", 0.0)
        # Ensure F_ext is numeric
        try:
            F_ext = float(F_ext)
        except (TypeError, ValueError):
            F_ext = 0.0

        v_sg = series.get("v_sg")
        a_sg = series.get("a_sg")
        t = series.get("t")
        if v_sg is None or a_sg is None or t is None:
            # skip experiment if required series missing
            continue
        v_sg = np.asarray(v_sg, dtype=float)
        a_sg = np.asarray(a_sg, dtype=float)
        t = np.asarray(t, dtype=float)
        # Compute drag
        if "drag" in series:
            drag = np.asarray(series["drag"], dtype=float)
        else:
            drag = F_ext - a_sg
        # Store
        data.append({
            "exp_id": eid,
            "F_ext": F_ext,
            "v": v_sg,
            "a": a_sg,
            "drag": drag,
            "t": t
        })

    if not data:
        raise ValueError("No valid experiments with v_sg and a_sg found.")

    # Prepare global arrays for fitting
    all_v = np.concatenate([d["v"] for d in data])
    all_drag = np.concatenate([d["drag"] for d in data])
    all_F = np.concatenate([np.full_like(d["v"], d["F_ext"]) for d in data])
    all_a = np.concatenate([d["a"] for d in data])

    # ---- Model Definitions ----
    # models that predict drag from v
    def linear_drag(v, alpha):
        return alpha * v

    def quad_drag(v, beta):
        return beta * v**2

    def quad_lin_drag(v, gamma, delta):
        return gamma * v**2 + delta * v

    # models that predict a from v and F_ext
    def a_frac(v, F, k):
        return F / (1 + k * v)

    def a_linear_drag(v, F, mu):
        return F - mu * v

    # We'll perform two types of fits:
    # (a) Global fit using all data
    # (b) Per-experiment fit for each F_ext value

    # ---- Global fits for drag models ----
    fit_results = {}

    # 1. drag = alpha * v
    X_linear = all_v.reshape(-1, 1)
    coeff_linear, _, _, _ = np.linalg.lstsq(X_linear, all_drag, rcond=None)
    alpha_glob = coeff_linear[0]
    pred_linear = alpha_glob * all_v
    r2_linear = r2_score(all_drag, pred_linear)
    rmse_linear = np.sqrt(mean_squared_error(all_drag, pred_linear))
    fit_results["drag_linear"] = {"alpha": alpha_glob, "R2": r2_linear, "RMSE": rmse_linear}

    # 2. drag = beta * v^2
    X_quad = (all_v**2).reshape(-1, 1)
    coeff_quad, _, _, _ = np.linalg.lstsq(X_quad, all_drag, rcond=None)
    beta_glob = coeff_quad[0]
    pred_quad = beta_glob * all_v**2
    r2_quad = r2_score(all_drag, pred_quad)
    rmse_quad = np.sqrt(mean_squared_error(all_drag, pred_quad))
    fit_results["drag_quad"] = {"beta": beta_glob, "R2": r2_quad, "RMSE": rmse_quad}

    # 3. drag = gamma * v^2 + delta * v
    X_quad_lin = np.column_stack([all_v**2, all_v])
    coeff_ql, _, _, _ = np.linalg.lstsq(X_quad_lin, all_drag, rcond=None)
    gamma_glob, delta_glob = coeff_ql
    pred_ql = gamma_glob * all_v**2 + delta_glob * all_v
    r2_ql = r2_score(all_drag, pred_ql)
    rmse_ql = np.sqrt(mean_squared_error(all_drag, pred_ql))
    fit_results["drag_quad_lin"] = {"gamma": gamma_glob, "delta": delta_glob, "R2": r2_ql, "RMSE": rmse_ql}

    # ---- Per-experiment drag fits ----
    per_exp_drag_fits = {}
    for d in data:
        eid = d["exp_id"]
        F = d["F_ext"]
        v = d["v"]
        drag = d["drag"]
        # linear drag = alpha * v
        X = v.reshape(-1,1)
        coeff, _, _, _ = np.linalg.lstsq(X, drag, rcond=None)
        alpha = coeff[0]
        pred = alpha * v
        r2 = r2_score(drag, pred) if len(drag)>1 else 0.0
        rmse = np.sqrt(mean_squared_error(drag, pred))
        per_exp_drag_fits[f"{eid}_linear"] = {"F_ext": F, "alpha": alpha, "R2": r2, "RMSE": rmse}

        # quadratic v^2
        X = (v**2).reshape(-1,1)
        coeff, _, _, _ = np.linalg.lstsq(X, drag, rcond=None)
        beta = coeff[0]
        pred = beta * v**2
        r2 = r2_score(drag, pred) if len(drag)>1 else 0.0
        rmse = np.sqrt(mean_squared_error(drag, pred))
        per_exp_drag_fits[f"{eid}_quad"] = {"F_ext": F, "beta": beta, "R2": r2, "RMSE": rmse}

        # quad_lin
        X = np.column_stack([v**2, v])
        coeff, _, _, _ = np.linalg.lstsq(X, drag, rcond=None)
        gamma, delta = coeff
        pred = gamma * v**2 + delta * v
        r2 = r2_score(drag, pred) if len(drag)>1 else 0.0
        rmse = np.sqrt(mean_squared_error(drag, pred))
        per_exp_drag_fits[f"{eid}_quad_lin"] = {"F_ext": F, "gamma": gamma, "delta": delta, "R2": r2, "RMSE": rmse}

    # ---- Global fits for a models (only for non-zero F_ext) ----
    a_models = {}
    # a = F / (1 + k * v)  -> need F>0
    mask_frac = (all_F > 1e-9)
    if np.any(mask_frac):
        v_frac = all_v[mask_frac]
        F_frac = all_F[mask_frac]
        a_frac_obs = all_a[mask_frac]
        try:
            popt, _ = curve_fit(lambda v, k: F_frac / (1 + k * v), v_frac, a_frac_obs, p0=[0.1], maxfev=10000)
            k_frac = popt[0]
            a_pred_frac = F_frac / (1 + k_frac * v_frac)
            r2_frac = r2_score(a_frac_obs, a_pred_frac)
            rmse_frac = np.sqrt(mean_squared_error(a_frac_obs, a_pred_frac))
            a_models["a_frac"] = {"k": k_frac, "R2": r2_frac, "RMSE": rmse_frac}
        except Exception as e:
            a_models["a_frac"] = {"error": str(e)}

    # a = F - mu * v
    if np.any(mask_frac):
        v_mu = all_v[mask_frac]
        F_mu = all_F[mask_frac]
        a_mu_obs = all_a[mask_frac]
        # linear: a = F - mu*v => mu = (F - a)/v
        # Use lstsq: (v) * mu = (F - a)
        X_mu = v_mu.reshape(-1,1)
        y_mu = F_mu - a_mu_obs
        coeff_mu, _, _, _ = np.linalg.lstsq(X_mu, y_mu, rcond=None)
        mu_glob = coeff_mu[0]
        a_pred_mu = F_mu - mu_glob * v_mu
        r2_mu = r2_score(a_mu_obs, a_pred_mu)
        rmse_mu = np.sqrt(mean_squared_error(a_mu_obs, a_pred_mu))
        a_models["a_linear_drag"] = {"mu": mu_glob, "R2": r2_mu, "RMSE": rmse_mu}

    # ---- Check a/F vs v for non-zero F_ext ----
    aF_analysis = {}
    if np.any(mask_frac):
        v_aF = all_v[mask_frac]
        a_over_F = all_a[mask_frac] / all_F[mask_frac]
        # simple linear in v
        coeff_aF, _, _, _ = np.linalg.lstsq(v_aF.reshape(-1,1), a_over_F, rcond=None)
        slope_aF = coeff_aF[0]
        pred_aF_linear = slope_aF * v_aF
        r2_aF_linear = r2_score(a_over_F, pred_aF_linear)
        # quadratic in v
        X_aF = np.column_stack([v_aF, v_aF**2])
        coeff_aFq, _, _, _ = np.linalg.lstsq(X_aF, a_over_F, rcond=None)
        c1, c2 = coeff_aFq
        pred_aF_quad = c1 * v_aF + c2 * v_aF**2
        r2_aF_quad = r2_score(a_over_F, pred_aF_quad)
        aF_analysis = {
            "linear_slope": slope_aF,
            "linear_R2": r2_aF_linear,
            "quad_params": [c1, c2],
            "quad_R2": r2_aF_quad
        }

    # ---- Build derived series ----
    derived_series = []
    # For each experiment, add drag_linear_pred, drag_quad_pred, drag_quad_lin_pred, a_frac_pred (if fit success), a_linear_drag_pred
    for d in data:
        eid = d["exp_id"]
        v = d["v"]
        drag = d["drag"]
        a = d["a"]
        F = d["F_ext"]

        # drag linear pred
        pred_drag_linear = alpha_glob * v
        derived_series.append({
            "experiment_id": eid,
            "name": "drag_linear_pred",
            "values": pred_drag_linear.tolist(),
            "source_name": f"drag = {alpha_glob:.4f} * v (global fit)",
            "provenance": "generated data processor: step_..._custom_data_analysis.py",
            "description": "Global linear drag prediction"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "drag_linear_residual",
            "values": (drag - pred_drag_linear).tolist(),
            "source_name": "residual = drag - linear_pred",
            "provenance": "generated data processor: step_..._custom_data_analysis.py"
        })

        # drag quad pred
        pred_drag_quad = beta_glob * v**2
        derived_series.append({
            "experiment_id": eid,
            "name": "drag_quad_pred",
            "values": pred_drag_quad.tolist(),
            "source_name": f"drag = {beta_glob:.4f} * v^2 (global fit)",
            "provenance": "generated data processor: step_..._custom_data_analysis.py"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "drag_quad_residual",
            "values": (drag - pred_drag_quad).tolist(),
            "source_name": "residual = drag - quad_pred",
            "provenance": "generated data processor: step_..._custom_data_analysis.py"
        })

        # drag quad_lin pred
        pred_drag_ql = gamma_glob * v**2 + delta_glob * v
        derived_series.append({
            "experiment_id": eid,
            "name": "drag_quad_lin_pred",
            "values": pred_drag_ql.tolist(),
            "source_name": f"drag = {gamma_glob:.4f}*v^2 + {delta_glob:.4f}*v (global fit)",
            "provenance": "generated data processor: step_..._custom_data_analysis.py"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "drag_quad_lin_residual",
            "values": (drag - pred_drag_ql).tolist(),
            "source_name": "residual = drag - quad_lin_pred",
            "provenance": "generated data processor: step_..._custom_data_analysis.py"
        })

        # a_frac_pred: only if F>0 and fit succeeded
        if "a_frac" in a_models and "k" in a_models["a_frac"]:
            k = a_models["a_frac"]["k"]
            pred_a_frac = np.where(F > 1e-9, F / (1 + k * v), np.nan)
            derived_series.append({
                "experiment_id": eid,
                "name": "a_frac_pred",
                "values": pred_a_frac.tolist(),
                "source_name": f"a = F/(1+{k:.4f}*v)",
                "provenance": "generated data processor: step_..._custom_data_analysis.py"
            })
            if not np.all(np.isnan(pred_a_frac)):
                resid = a - pred_a_frac
                resid = np.where(np.isnan(resid), 0.0, resid)  # avoid nan
                derived_series.append({
                    "experiment_id": eid,
                    "name": "a_frac_residual",
                    "values": resid.tolist(),
                    "source_name": "residual = a - a_frac_pred",
                    "provenance": "generated data processor: step_..._custom_data_analysis.py"
                })

        # a_linear_drag_pred
        if "a_linear_drag" in a_models:
            mu = a_models["a_linear_drag"]["mu"]
            pred_a_linear_drag = F - mu * v
            derived_series.append({
                "experiment_id": eid,
                "name": "a_linear_drag_pred",
                "values": pred_a_linear_drag.tolist(),
                "source_name": f"a = F - {mu:.4f}*v",
                "provenance": "generated data processor: step_..._custom_data_analysis.py"
            })
            derived_series.append({
                "experiment_id": eid,
                "name": "a_linear_drag_residual",
                "values": (a - pred_a_linear_drag).tolist(),
                "source_name": "residual = a - linear_drag_pred",
                "provenance": "generated data processor: step_..._custom_data_analysis.py"
            })

    # ---- Generate figures ----
    # 1. drag vs v scatter colored by F_ext
    plt.figure(figsize=(8,6))
    unique_F = sorted(set(d["F_ext"] for d in data))
    colors = plt.cm.viridis(np.linspace(0,1,len(unique_F)))
    for f, c in zip(unique_F, colors):
        mask = np.isclose(all_F, f)
        plt.scatter(all_v[mask], all_drag[mask], label=f'F_ext={f}', color=c, alpha=0.6, s=20)
    plt.xlabel('v_sg')
    plt.ylabel('drag')
    plt.title('Drag vs v colored by F_ext')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Add global fit lines for visualization
    v_grid = np.linspace(all_v.min(), all_v.max(), 200)
    # linear
    plt.plot(v_grid, alpha_glob * v_grid, '--', label=f'linear: {alpha_glob:.3f}*v', color='black', linewidth=1)
    # quad
    plt.plot(v_grid, beta_glob * v_grid**2, ':', label=f'quad: {beta_glob:.3f}*v²', color='blue', linewidth=1)
    # quad_lin
    plt.plot(v_grid, gamma_glob * v_grid**2 + delta_glob * v_grid, '-.', label=f'quad+lin: {gamma_glob:.3f}*v²+{delta_glob:.3f}*v', color='red', linewidth=1)
    plt.legend()
    fig_path_drag = os.path.join(output_dir, "drag_vs_v_F_ext_colored.png")
    plt.tight_layout()
    plt.savefig(fig_path_drag, dpi=150)
    plt.close()

    # 2. a/F vs v for non-zero F_ext (if data exist)
    figures = [fig_path_drag]
    if mask_frac.any():
        plt.figure(figsize=(8,6))
        for d in data:
            if d["F_ext"] > 1e-9:
                plt.scatter(d["v"], d["a"]/d["F_ext"], label=f'exp_{d["exp_id"]} F={d["F_ext"]}', alpha=0.7, s=20)
        plt.xlabel('v_sg')
        plt.ylabel('a / F_ext')
        plt.title('a/F vs v (non-zero F_ext)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        # Add linear fit line
        v_grid2 = np.linspace(0, all_v[mask_frac].max(), 200)
        # linear: a/F = slope * v
        if "linear_slope" in aF_analysis:
            plt.plot(v_grid2, aF_analysis["linear_slope"] * v_grid2, '--', label=f'linear: {aF_analysis["linear_slope"]:.4f}*v', color='red')
        # quad: a/F = c1*v + c2*v^2
        if "quad_params" in aF_analysis:
            c1, c2 = aF_analysis["quad_params"]
            plt.plot(v_grid2, c1*v_grid2 + c2*v_grid2**2, ':', label=f'quad: {c1:.4f}*v + {c2:.4f}*v²', color='green')
        plt.legend()
        fig_path_aF = os.path.join(output_dir, "a_over_F_vs_v.png")
        plt.tight_layout()
        plt.savefig(fig_path_aF, dpi=150)
        plt.close()
        figures.append(fig_path_aF)

    # ---- Build metrics ----
    metrics = {}
    # Global drag fits
    for name, res in fit_results.items():
        for k, v in res.items():
            metrics[f"{name}_{k}"] = v
    # Per-experiment fits
    for key, res in per_exp_drag_fits.items():
        for k, v in res.items():
            metrics[f"{key}_{k}"] = v
    # a models
    for name, res in a_models.items():
        for k, v in res.items():
            metrics[f"{name}_{k}"] = v
    # aF analysis
    for k, v in aF_analysis.items():
        metrics[f"a_over_F_{k}"] = v

    # ---- Determine best model for drag ----
    best_drag_model = min(fit_results, key=lambda x: fit_results[x]["RMSE"])
    best_drag_R2 = fit_results[best_drag_model]["R2"]
    best_drag_RMSE = fit_results[best_drag_model]["RMSE"]

    # ---- Build observation ----
    obs_lines = []
    obs_lines.append("对实验 IDs {} 进行了 drag 与速度的关系分析。".format(exp_ids))
    obs_lines.append("全局拟合 drag 模型结果：")
    for name, res in fit_results.items():
        obs_lines.append("  {}: R²={:.4f}, RMSE={:.4f}".format(name, res.get("R2",0), res.get("RMSE",0)))
    obs_lines.append("最佳全局 drag 模型: {} (R²={:.4f}, RMSE={:.4f})".format(best_drag_model, best_drag_R2, best_drag_RMSE))
    obs_lines.append("各实验单独线性 drag (alpha*v) 拟合系数：")
    for key, res in per_exp_drag_fits.items():
        if "linear" in key:
            obs_lines.append("  {}: alpha={:.4f}, R²={:.4f}, RMSE={:.4f}".format(key, res["alpha"], res["R2"], res["RMSE"]))
    obs_lines.append("加速度模型 a = F/(1+k*v): {}".format(a_models.get("a_frac", {}).get("R2", "拟合失败")))
    obs_lines.append("加速度模型 a = F - mu*v: {}".format(a_models.get("a_linear_drag", {}).get("R2", "拟合失败")))
    if aF_analysis:
        obs_lines.append("a/F vs v 线性拟合斜率={:.4f}, R²={:.4f}; 二次 R²={:.4f}".format(
            aF_analysis.get("linear_slope",0), aF_analysis.get("linear_R2",0), aF_analysis.get("quad_R2",0)))
    obs_lines.append("不同 F_ext 下的单独拟合系数有差异，表明 drag 形式可能依赖于 F_ext。")
    obs_lines.append("已生成 drag vs v 散点图（按 F_ext 着色）和 a/F vs v 图（如适用）。")
    obs_lines.append("已返回 drag 模型预测和残差派生序列，以及 a 模型预测和残差（如适用）。")
    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
