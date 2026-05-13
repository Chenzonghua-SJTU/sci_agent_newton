import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os, warnings
warnings.filterwarnings('ignore')

def process(payload: dict) -> dict:
    params = payload.get("parameters", {})
    analysis_goal = params.get("analysis_goal", "")
    experiment_ids = params.get("experiment_ids", [])
    output_dir = payload.get("output_dir", ".")

    experiments = payload.get("experiments", {})

    # Select experiments
    if experiment_ids:
        exp_dict = {eid: experiments[eid] for eid in experiment_ids if eid in experiments}
    else:
        exp_dict = experiments

    # Collect data
    all_F = []
    all_v = []
    all_v2 = []
    all_a = []
    exp_labels = []
    color_map = {}

    # Define markers for different experiments
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    derived_series_list = []
    counter = 0

    for eid, exp in exp_dict.items():
        config = exp.get("config", {})
        series = exp.get("series", {})
        avail = exp.get("available_series", [])

        t = series.get("t", None)
        if t is None:
            raise ValueError(f"Experiment {eid} missing t series.")

        # Get F_ext
        if "F_ext" in series:
            F_ext = np.array(series["F_ext"])
        else:
            # from config constant force
            F_ext_val = config.get("constant_force", 0.0)
            F_ext = np.full(len(t), F_ext_val)
        all_F.extend(F_ext.tolist())

        # Get v_sg
        if "v_sg" in series:
            v = np.array(series["v_sg"])
        else:
            # Estimate from q using sg filter if q is available
            q = series.get("q", None)
            if q is None:
                # fallback to q_smooth
                q = series.get("q_smooth", None)
            if q is None:
                raise ValueError(f"Experiment {eid} has neither v_sg nor q.")
            from scipy.signal import savgol_filter
            window = min(11, len(q))
            if window % 2 == 0:
                window += 1
            q_smooth = savgol_filter(q, window, 3)
            dt = config.get("dt", 0.1)
            v = np.gradient(q_smooth, dt)
        all_v.extend(v.tolist())
        all_v2.extend((v ** 2).tolist())

        # Get a_sg
        if "a_sg" in series:
            a = np.array(series["a_sg"])
        else:
            # For exp07 (free field, v constant, a=0)
            if eid == "exp_07" or (config.get("force_field_type") == "free" and config.get("constant_force", 0) == 0):
                a = np.zeros(len(t))
            else:
                # Estimate from v using sg filter
                window2 = min(11, len(v))
                if window2 % 2 == 0:
                    window2 += 1
                dt = config.get("dt", 0.1)
                a = savgol_filter(v, window2, 3, deriv=1, delta=dt)
        all_a.extend(a.tolist())

        # Store for derived series later
        exp_labels.append((eid, len(t)))
        color_map[eid] = counter
        counter += 1

    # Convert to arrays
    X_F = np.array(all_F).reshape(-1, 1)
    X_v = np.array(all_v).reshape(-1, 1)
    X_v2 = np.array(all_v2).reshape(-1, 1)
    y = np.array(all_a).reshape(-1, 1)

    # Build design matrices for different models
    # Model 1: a = c0 + c1*F_ext + c2*v
    X1 = np.hstack([np.ones_like(X_F), X_F, X_v])
    # Model 2: a = c0 + c1*F_ext + c2*v^2
    X2 = np.hstack([np.ones_like(X_F), X_F, X_v2])
    # Model 3: a = c1*F_ext + c2*v (no intercept)
    X3 = np.hstack([X_F, X_v])
    # Model 4: a = c1*F_ext + c2*v^2 (no intercept)
    X4 = np.hstack([X_F, X_v2])
    # Model 5: a = c0 + c1*F_ext + c2*v + c3*v^2
    X5 = np.hstack([np.ones_like(X_F), X_F, X_v, X_v2])

    models_info = [
        ("a = c0 + c1*F_ext + c2*v", X1, ["c0", "c1", "c2"]),
        ("a = c0 + c1*F_ext + c2*v^2", X2, ["c0", "c1", "c2"]),
        ("a = c1*F_ext + c2*v (no intercept)", X3, ["c1", "c2"]),
        ("a = c1*F_ext + c2*v^2 (no intercept)", X4, ["c1", "c2"]),
        ("a = c0 + c1*F_ext + c2*v + c3*v^2", X5, ["c0", "c1", "c2", "c3"])
    ]

    metrics = {}
    best_model_name = ""
    best_r2 = -1e9
    predictions_per_model = {}
    residuals_per_model = {}

    for formula, X_mat, coef_names in models_info:
        # Solve least squares
        if X_mat.shape[1] == 0:
            continue
        coeff, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        coeff = coeff.flatten()
        y_pred = X_mat.dot(coeff).flatten()
        y_actual = y.flatten()
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        resid_std = np.sqrt(ss_res / (len(y_actual) - X_mat.shape[1]))

        # Store metrics
        prefix = f"model_{formula[:3]}_{len(coeff)}_".replace(" ", "_").replace(",", "").replace("+", "p").replace("*", "x").replace("^", "pow")
        for i, cname in enumerate(coef_names):
            metrics[f"{prefix}{cname}"] = coeff[i]
        metrics[f"{prefix}R2_global"] = r2
        metrics[f"{prefix}residual_std_global"] = resid_std

        # Save predictions for derived series
        predictions_per_model[prefix] = y_pred
        residuals_per_model[prefix] = y_actual - y_pred

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = formula

    # Compute per-experiment residuals for best model
    best_prefix = None
    for prefix in predictions_per_model.keys():
        if best_model_name in prefix or f"model_{best_model_name[:3]}" in prefix:
            best_prefix = prefix
            break
    if best_prefix is None:
        best_prefix = list(predictions_per_model.keys())[0]

    # Build derived series: predictions and residuals for each experiment using global best model
    # We need to re-index the global predictions back to each experiment
    idx = 0
    for eid, length in exp_labels:
        exp = exp_dict[eid]
        # Get the slice of global arrays
        end = idx + length
        pred_slice = predictions_per_model[best_prefix][idx:end]
        resid_slice = residuals_per_model[best_prefix][idx:end]

        derived_series_list.append({
            "experiment_id": eid,
            "name": "joint_fit_pred_a",
            "values": pred_slice.tolist(),
            "source_name": f"global best model ({best_model_name})",
            "description": f"Predicted a from joint fit using {best_model_name}"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": "joint_fit_residual",
            "values": resid_slice.tolist(),
            "source_name": f"residual from best model",
            "description": f"Residual a_actual - a_pred from joint fit"
        })
        idx = end

    # Generate scatter plot a vs v colored by F_ext
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use points from all experiments
    colors = []
    for f in all_F:
        if f == 0:
            colors.append('black')
        elif f == 1:
            colors.append('blue')
        elif f == 5:
            colors.append('green')
        elif f == 10:
            colors.append('red')
        else:
            colors.append('gray')

    scatter = ax.scatter(all_v, all_a, c=colors, alpha=0.7, edgecolors='none', s=20)
    # Add legend for F_ext values
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='F_ext=0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='F_ext=1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='F_ext=5'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='F_ext=10')
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel('v_sg (velocity)')
    ax.set_ylabel('a_sg (acceleration)')
    ax.set_title('a_sg vs v_sg colored by F_ext')
    fig_path = os.path.join(output_dir, "a_vs_v_F_ext_colored.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # Prepare observation
    obs = f"对所有恒定外力实验以及自由场景exp_07进行了联合多变量拟合。\n"
    obs += f"尝试的模型及全局结果：\n"
    for formula, X_mat, coef_names in models_info:
        prefix = f"model_{formula[:3]}_{X_mat.shape[1]}_".replace(" ", "_").replace(",", "").replace("+", "p").replace("*", "x").replace("^", "pow")
        r2 = metrics.get(f"{prefix}R2_global", "N/A")
        rstd = metrics.get(f"{prefix}residual_std_global", "N/A")
        coef_vals = ", ".join([f"{c}={metrics.get(f'{prefix}{c}', '?')}" for c in coef_names])
        obs += f"  {formula}: {coef_vals}, R²={r2:.4f}, 残差标准差={rstd:.4f}\n"
    obs += f"全局最佳模型: {best_model_name} (R²={best_r2:.4f})\n"
    obs += f"注意: exp_07 (自由场)的a_sg恒为0, v_sg=5恒定, F_ext=0。它在拟合中提供了零截距参考点。\n"
    obs += f"已为每个实验生成预测序列 joint_fit_pred_a 和残差序列 joint_fit_residual。\n"

    return {
        "observation": obs,
        "derived_series": derived_series_list,
        "figures": [fig_path],
        "metrics": metrics
    }
