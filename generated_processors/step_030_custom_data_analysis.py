import os
import math
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    output_dir = payload.get("output_dir", ".")
    experiments = payload.get("experiments", {})
    parameters = payload.get("parameters", {})
    requested_ids = parameters.get("experiment_ids")
    if requested_ids is None:
        requested_ids = list(experiments.keys())

    # validate all requested ids exist
    for eid in requested_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload experiments.")

    # helpers
    def get_F_ext(exp):
        config = exp.get("config", {})
        if config.get("force_field_type") == "constant":
            # try multiple possible keys
            return config.get("constant_force") or config.get("F_ext") or 0.0
        else:
            return 0.0

    # separate experiments
    constant_exps = {}
    free_exps = {}
    for eid in requested_ids:
        exp = experiments[eid]
        cfg = exp.get("config", {})
        if cfg.get("force_field_type") == "constant":
            constant_exps[eid] = exp
        else:
            free_exps[eid] = exp

    # ---- derived series containers ----
    derived_series = []
    metrics = {}
    figures = []

    # ---- Part 1: free experiments: check a_sg near zero ----
    free_checks = {}
    for eid, exp in free_exps.items():
        series = exp.get("series", {})
        if "a_sg" not in series:
            raise ValueError(f"Free experiment {eid} missing a_sg series.")
        a_sg = np.array(series["a_sg"])
        a_mean = np.mean(a_sg)
        a_std = np.std(a_sg)
        a_max = np.max(np.abs(a_sg))
        free_checks[eid] = {"mean": a_mean, "std": a_std, "max_abs": a_max}
        metrics[f"{eid}_a_sg_mean"] = a_mean
        metrics[f"{eid}_a_sg_std"] = a_std
        metrics[f"{eid}_a_sg_max_abs"] = a_max

    # ---- Part 2: model test: 1/a_sg vs v_sg^2 for each constant exp ----
    model_results = {}  # eid -> {slope, intercept, R2, resid_std, beta_est, n_points}
    for eid, exp in constant_exps.items():
        series = exp.get("series", {})
        if "a_sg" not in series or "v_sg" not in series:
            raise ValueError(f"Constant experiment {eid} missing a_sg or v_sg series.")
        a_sg = np.array(series["a_sg"])
        v_sg = np.array(series["v_sg"])
        # filter out very small a_sg to avoid division by zero
        mask = a_sg > 1e-8
        a_sg_f = a_sg[mask]
        v_sg_f = v_sg[mask]
        if len(a_sg_f) < 5:
            metrics[f"{eid}_model_test_skipped"] = 1
            continue
        inv_a = 1.0 / a_sg_f
        v_sq = v_sg_f ** 2
        # linear regression
        X = v_sq.reshape(-1, 1)
        reg = LinearRegression(fit_intercept=True).fit(X, inv_a)
        pred = reg.predict(X)
        resid = inv_a - pred
        n = len(inv_a)
        r2 = 1.0 - np.sum(resid**2) / np.sum((inv_a - np.mean(inv_a))**2)
        intercept = reg.intercept_
        slope = reg.coef_[0]
        resid_std = np.std(resid, ddof=2)
        # get F_ext
        F_ext = get_F_ext(exp)
        beta_est = slope * F_ext if F_ext != 0 else float('nan')
        intercept_theory = 1.0 / F_ext if F_ext != 0 else float('nan')
        model_results[eid] = {
            "slope": slope,
            "intercept": intercept,
            "R2": r2,
            "resid_std": resid_std,
            "beta_est": beta_est,
            "n_points": n,
            "F_ext": F_ext,
            "intercept_theory": intercept_theory,
            "intercept_rel_error": (intercept - intercept_theory) / intercept_theory if intercept_theory != 0 else float('nan')
        }
        # store derived series for this experiment (on full time grid)
        full_inv_a = 1.0 / np.array(series["a_sg"])
        full_v_sq = np.array(series["v_sg"]) ** 2
        full_pred = reg.intercept_ + reg.coef_[0] * full_v_sq
        # handle points where a_sg was near zero -> keep inv_a large but we keep them
        derived_series.append({
            "experiment_id": eid,
            "name": "inv_a_sg",
            "values": full_inv_a.tolist(),
            "source_name": "1 / a_sg",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "reciprocal of acceleration"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "v_sg_sq",
            "values": full_v_sq.tolist(),
            "source_name": "v_sg ** 2",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "square of sg velocity"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "pred_inv_a_sg_linear",
            "values": full_pred.tolist(),
            "source_name": f"intercept({intercept:.4f}) + slope({slope:.4f})*v_sg^2",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "linear fit prediction for 1/a_sg"
        })
        # metrics for this exp
        metrics[f"{eid}_inv_a_vs_vsq_slope"] = slope
        metrics[f"{eid}_inv_a_vs_vsq_intercept"] = intercept
        metrics[f"{eid}_inv_a_vs_vsq_R2"] = r2
        metrics[f"{eid}_inv_a_vs_vsq_resid_std"] = resid_std
        metrics[f"{eid}_beta_estimate"] = beta_est
        metrics[f"{eid}_intercept_theory"] = intercept_theory
        metrics[f"{eid}_intercept_rel_error"] = model_results[eid]["intercept_rel_error"]

    # compute cross-experiment beta statistics if at least 2 valid
    beta_values = [v["beta_est"] for v in model_results.values() if not math.isnan(v["beta_est"])]
    if len(beta_values) >= 2:
        beta_mean = np.mean(beta_values)
        beta_std = np.std(beta_values, ddof=1)
        beta_cv = beta_std / beta_mean if beta_mean != 0 else float('inf')
        metrics["beta_cross_exp_mean"] = beta_mean
        metrics["beta_cross_exp_std"] = beta_std
        metrics["beta_cross_exp_CV"] = beta_cv

    # ---- Part 3: exploration of alternative relationships ----
    # for each constant exp, fit a_sg vs v_sg (linear, quadratic) and drag vs v_sg^2 (linear)
    exploration_metrics = {}
    for eid, exp in constant_exps.items():
        series = exp.get("series", {})
        a_sg = np.array(series["a_sg"])
        v_sg = np.array(series["v_sg"])
        F_ext = get_F_ext(exp)
        # compute drag if not present, else use existing
        if "drag" in series:
            drag = np.array(series["drag"])
        else:
            drag = F_ext - a_sg
        # a_sg vs v_sg linear
        X1 = v_sg.reshape(-1, 1)
        reg1 = LinearRegression(fit_intercept=True).fit(X1, a_sg)
        pred1 = reg1.predict(X1)
        r2_1 = 1.0 - np.sum((a_sg - pred1)**2) / np.sum((a_sg - np.mean(a_sg))**2)
        resid_std1 = np.std(a_sg - pred1, ddof=2)
        # a_sg vs v_sg quadratic (add v^2)
        v_sq = v_sg ** 2
        X2 = np.column_stack([v_sg, v_sq])
        reg2 = LinearRegression(fit_intercept=True).fit(X2, a_sg)
        pred2 = reg2.predict(X2)
        r2_2 = 1.0 - np.sum((a_sg - pred2)**2) / np.sum((a_sg - np.mean(a_sg))**2)
        resid_std2 = np.std(a_sg - pred2, ddof=3)
        # drag vs v_sg^2 linear (no intercept)
        X3 = v_sq.reshape(-1, 1)
        reg3 = LinearRegression(fit_intercept=False).fit(X3, drag)
        pred3 = reg3.predict(X3)
        # For R2 with no intercept, use total sum of squares about zero, not mean
        ss_res = np.sum((drag - pred3)**2)
        ss_tot = np.sum(drag**2)
        r2_3 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        resid_std3 = np.sqrt(ss_res / (len(drag) - 1))  # no intercept, so df=n-1
        # store
        exp_key = f"{eid}"
        exploration_metrics[exp_key] = {
            "a_linear_slope": reg1.coef_[0],
            "a_linear_intercept": reg1.intercept_,
            "a_linear_R2": r2_1,
            "a_linear_resid_std": resid_std1,
            "a_quad_c1": reg2.coef_[0],
            "a_quad_c2": reg2.coef_[1],
            "a_quad_intercept": reg2.intercept_,
            "a_quad_R2": r2_2,
            "a_quad_resid_std": resid_std2,
            "drag_vs_vsq_slope": reg3.coef_[0],
            "drag_vs_vsq_R2": r2_3,
            "drag_vs_vsq_resid_std": resid_std3
        }
        # add to metrics
        for k, v in exploration_metrics[exp_key].items():
            metrics[f"{eid}_{k}"] = v

        # create optional derived series for predictions
        # a_linear_pred
        derived_series.append({
            "experiment_id": eid,
            "name": "pred_a_linear_alt",
            "values": pred1.tolist(),
            "source_name": f"linear fit: {reg1.intercept_:.4f} + {reg1.coef_[0]:.4f}*v_sg",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "alternative linear prediction for a_sg"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "pred_a_quad_alt",
            "values": pred2.tolist(),
            "source_name": f"quad fit: {reg2.intercept_:.4f} + {reg2.coef_[0]:.4f}*v_sg + {reg2.coef_[1]:.4f}*v_sg^2",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "alternative quadratic prediction for a_sg"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "pred_drag_vsq",
            "values": pred3.tolist(),
            "source_name": f"drag = {reg3.coef_[0]:.4f} * v_sg^2",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "drag proportional to v_sg^2 prediction"
        })

    # ---- Figures ----
    # Figure 1: 1/a_sg vs v_sg^2 for each constant exp with linear fit
    n_const = len(constant_exps)
    if n_const > 0:
        fig1, axes = plt.subplots(1, n_const, figsize=(5*n_const, 4))
        if n_const == 1:
            axes = [axes]
        for ax, (eid, exp) in zip(axes, constant_exps.items()):
            series = exp.get("series", {})
            a_sg = np.array(series["a_sg"])
            v_sg = np.array(series["v_sg"])
            mask = a_sg > 1e-8
            inv_a = 1.0 / a_sg[mask]
            v_sq = v_sg[mask]**2
            ax.scatter(v_sq, inv_a, s=10, label='data')
            # sort for line
            idx = np.argsort(v_sq)
            ax.plot(v_sq[idx], model_results[eid]["intercept"] + model_results[eid]["slope"]*v_sq[idx], 'r-', label='fit')
            ax.set_xlabel('v_sg^2')
            ax.set_ylabel('1/a_sg')
            ax.set_title(f'{eid} (F_ext={model_results[eid]["F_ext"]})')
            ax.legend()
            ax.grid(True)
        fig1.tight_layout()
        fpath1 = os.path.join(output_dir, "inv_a_vs_vsq_fits.png")
        fig1.savefig(fpath1)
        plt.close(fig1)
        figures.append(fpath1)

    # Figure 2: cross-experiment beta comparison
    if len(beta_values) >= 2:
        fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
        eids_list = list(model_results.keys())
        betas = [model_results[e]["beta_est"] for e in eids_list]
        intercepts = [model_results[e]["intercept"] for e in eids_list]
        F_vals = [model_results[e]["F_ext"] for e in eids_list]
        intercept_theory = [model_results[e]["intercept_theory"] for e in eids_list]
        # left: intercept vs 1/F_ext
        axes[0].scatter([1/f for f in F_vals if f>0], intercepts, c='blue', label='intercept')
        axes[0].plot([min(1/f for f in F_vals if f>0), max(1/f for f in F_vals if f>0)],
                     [min(intercept_theory), max(intercept_theory)], 'r--', label='theory: 1/F')
        axes[0].set_xlabel('1/F_ext')
        axes[0].set_ylabel('fitted intercept')
        axes[0].legend()
        axes[0].grid(True)
        # right: beta values per experiment
        axes[1].bar(range(len(eids_list)), betas, tick_label=eids_list)
        axes[1].axhline(y=beta_mean, color='r', linestyle='--', label=f'mean beta={beta_mean:.4f}')
        axes[1].set_ylabel('beta estimate')
        axes[1].legend()
        axes[1].grid(True, axis='y')
        fig2.tight_layout()
        fpath2 = os.path.join(output_dir, "model_consistency.png")
        fig2.savefig(fpath2)
        plt.close(fig2)
        figures.append(fpath2)

    # Figure 3: exploration comparison – a_sg vs v_sg scatter with linear and quadratic fits
    if n_const > 0:
        fig3, axes = plt.subplots(1, n_const, figsize=(5*n_const, 4))
        if n_const == 1:
            axes = [axes]
        for ax, (eid, exp) in zip(axes, constant_exps.items()):
            series = exp.get("series", {})
            a_sg = np.array(series["a_sg"])
            v_sg = np.array(series["v_sg"])
            ax.scatter(v_sg, a_sg, s=8, label='data')
            idx = np.argsort(v_sg)
            # linear fit
            em = exploration_metrics[eid]
            pred_lin = em["a_linear_intercept"] + em["a_linear_slope"] * v_sg
            ax.plot(v_sg[idx], pred_lin[idx], '-', label=f'linear R²={em["a_linear_R2"]:.3f}')
            # quadratic fit
            pred_quad = em["a_quad_intercept"] + em["a_quad_c1"] * v_sg + em["a_quad_c2"] * v_sg**2
            ax.plot(v_sg[idx], pred_quad[idx], '--', label=f'quad R²={em["a_quad_R2"]:.3f}')
            ax.set_xlabel('v_sg')
            ax.set_ylabel('a_sg')
            ax.set_title(f'{eid}')
            ax.legend()
            ax.grid(True)
        fig3.tight_layout()
        fpath3 = os.path.join(output_dir, "a_vs_v_exploration.png")
        fig3.savefig(fpath3)
        plt.close(fig3)
        figures.append(fpath3)

    # ---- Build observation ----
    obs_lines = []
    obs_lines.append("对指定实验进行`a_sg = F_ext / (1+beta*v_sg^2)`模型检验及替代关系探索。")
    # free experiments
    if free_exps:
        obs_lines.append("自由实验加速度检测：")
        for eid, info in free_checks.items():
            obs_lines.append(f"  {eid}: a_sg均值={info['mean']:.4e}, 标准差={info['std']:.4e}, 最大绝对偏差={info['max_abs']:.4e}")
    # model test results
    if model_results:
        obs_lines.append("模型检验（1/a_sg = intercept + slope * v_sg^2）：")
        for eid, res in model_results.items():
            obs_lines.append(f"  {eid}: F_ext={res['F_ext']}, intercept={res['intercept']:.4f} (理论={res['intercept_theory']:.4f}, 相对误差={res['intercept_rel_error']:.2%}), slope={res['slope']:.4f}, beta估计={res['beta_est']:.4f}, R²={res['R2']:.4f}")
        if len(beta_values) >= 2:
            obs_lines.append(f"跨实验beta统计: 均值={beta_mean:.4f}, 标准差={beta_std:.4f}, CV={beta_cv:.2%}")
        else:
            obs_lines.append("有效beta估计数不足，无法跨实验比较。")
    else:
        obs_lines.append("无有效恒定外力实验数据用于模型检验。")

    obs_lines.append("替代关系探索（每个恒定外力实验分别拟合）：")
    for eid, em in exploration_metrics.items():
        obs_lines.append(f"  {eid}: a_linear R²={em['a_linear_R2']:.4f}, a_quad R²={em['a_quad_R2']:.4f}, drag vs v_sq R²={em['drag_vs_vsq_R2']:.4f}")
    # indicate which derived series and figures were produced
    obs_lines.append(f"派生序列: {[ds['experiment_id']+':'+ds['name'] for ds in derived_series]}")
    obs_lines.append(f"图像: {figures}")

    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
