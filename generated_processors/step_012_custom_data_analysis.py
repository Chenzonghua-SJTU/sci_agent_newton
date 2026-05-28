import os
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def _get_sg_velocity_acceleration(t, q, window_length=11, polyorder=3):
    """Use Savitzky-Golay filter to estimate velocity and acceleration.
    Returns v_sg, a_sg (numpy arrays)."""
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    if len(q) < window_length:
        window_length = len(q) if len(q) % 2 == 1 else len(q) - 1
        if window_length < 3:
            raise ValueError("Too few data points for SG filter")
    v = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt)
    a = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)
    return v, a

def process(payload):
    output_dir = payload.get("output_dir", ".")
    experiments = payload["experiments"]
    exp_ids = payload["parameters"].get("experiment_ids", list(experiments.keys()))
    # Only process specified experiment ids that exist
    valid_ids = [eid for eid in exp_ids if eid in experiments]
    if not valid_ids:
        raise ValueError("No valid experiment ids found in parameters: " + str(exp_ids))

    # Force mapping: use F_ext from config; fall back to constant_force
    def get_force(exp):
        cfg = exp["config"]
        F = cfg.get("F_ext")
        if F is not None:
            return float(F)
        # fallback: constant_force field
        if cfg.get("force_field_type") == "constant":
            return float(cfg.get("constant_force", 0.0))
        return 0.0

    # 1. Extract or compute a_sg and v_sg for each experiment
    data = {}  # key: eid, value: dict with t, q, v, a, F, label
    for eid in valid_ids:
        exp = experiments[eid]
        series = exp["series"]
        available = exp.get("available_series", [])
        t = np.array(series["t"])
        q = np.array(series["q"])
        # Try to use existing a_sg / v_sg
        if "a_sg" in available and "v_sg" in available:
            v_sg = np.array(series["v_sg"])
            a_sg = np.array(series["a_sg"])
        else:
            v_sg, a_sg = _get_sg_velocity_acceleration(t, q)
        F = get_force(exp)
        data[eid] = {
            "t": t,
            "q": q,
            "v": v_sg,
            "a": a_sg,
            "F": F,
            "label": f"{eid} (F={F})"
        }
    # Determine time steps for interpolation
    # We will use a common time base (the t from first experiment)
    # but for comparison at specific v values we need to interpolate a(v)
    # We'll create a function a_of_v for each experiment using linear interpolation
    a_interp_funcs = {}
    for eid, d in data.items():
        # sort by v
        v_sorted = np.sort(d["v"])
        a_sorted = d["a"][np.argsort(d["v"])]
        # Remove duplicates if any
        unique_mask = np.diff(v_sorted) > 0
        v_sorted = np.concatenate(([v_sorted[0]], v_sorted[1:][unique_mask]))
        a_sorted = np.concatenate(([a_sorted[0]], a_sorted[1:][unique_mask]))
        if len(v_sorted) < 2:
            a_interp_funcs[eid] = lambda v, a0=a_sorted[0]: np.full_like(v, a0, dtype=float)
        else:
            a_interp_funcs[eid] = interp1d(v_sorted, a_sorted, kind='linear', bounds_error=False, fill_value=np.nan)

    # 2. Plot a vs v scatter for all experiments (different colors)
    plt.figure(figsize=(10, 7))
    colors = plt.cm.tab10(range(len(data)))
    for (eid, d), c in zip(data.items(), colors):
        plt.scatter(d["v"], d["a"], s=10, alpha=0.6, color=c, label=d["label"])
    plt.xlabel("Velocity v_sg")
    plt.ylabel("Acceleration a_sg")
    plt.title("a_sg vs v_sg (all experiments)")
    plt.legend()
    scatter_path = os.path.join(output_dir, "a_vs_v_all_experiments.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()

    # 3. Fit a = α + β*v and a = α + β*v^2 per experiment
    fits = {}  # eid -> dict with 'linear', 'quadratic'
    for eid, d in data.items():
        v = d["v"]
        a = d["a"]
        # Remove NaN or inf
        valid = np.isfinite(v) & np.isfinite(a)
        v_clean = v[valid]
        a_clean = a[valid]
        # Linear: a = alpha + beta*v
        A = np.column_stack([np.ones_like(v_clean), v_clean])
        coeff_lin, residuals_lin, rank, sv = np.linalg.lstsq(A, a_clean, rcond=None)
        alpha_lin, beta_lin = coeff_lin[0], coeff_lin[1]
        pred_lin = A @ coeff_lin
        ss_res_lin = np.sum((a_clean - pred_lin)**2)
        ss_tot_lin = np.sum((a_clean - np.mean(a_clean))**2)
        r2_lin = 1 - ss_res_lin/ss_tot_lin if ss_tot_lin > 0 else 0
        # Quadratic: a = alpha + beta*v^2
        A2 = np.column_stack([np.ones_like(v_clean), v_clean**2])
        coeff_quad, residuals_quad, rank2, sv2 = np.linalg.lstsq(A2, a_clean, rcond=None)
        alpha_quad, beta_quad = coeff_quad[0], coeff_quad[1]
        pred_quad = A2 @ coeff_quad
        ss_res_quad = np.sum((a_clean - pred_quad)**2)
        r2_quad = 1 - ss_res_quad/ss_tot_lin if ss_tot_lin > 0 else 0
        fits[eid] = {
            "linear": {"alpha": alpha_lin, "beta": beta_lin, "R2": r2_lin},
            "quadratic": {"alpha": alpha_quad, "beta": beta_quad, "R2": r2_quad}
        }

    # 4. Compare a at same v values (v_target = [1, 2, 3])
    v_targets = [1.0, 2.0, 3.0]
    comparison_table = {}
    for vt in v_targets:
        row = {}
        for eid in valid_ids:
            func = a_interp_funcs[eid]
            val = func(vt)
            row[eid] = float(val) if np.isfinite(val) else np.nan
        comparison_table[f"v={vt}"] = row

    # 5. Multi-experiment fit: a = β*F + γ*v^2
    all_a = []
    all_F = []
    all_v2 = []
    for eid in valid_ids:
        d = data[eid]
        F = d["F"]
        v = d["v"]
        a = d["a"]
        valid = np.isfinite(v) & np.isfinite(a) & np.isfinite(F)
        all_a.extend(a[valid])
        all_F.extend([F]*np.sum(valid))
        all_v2.extend((v[valid]**2))
    all_a = np.array(all_a)
    all_F = np.array(all_F)
    all_v2 = np.array(all_v2)
    # Design matrix: [F, v^2]
    X = np.column_stack([all_F, all_v2])
    # Add intercept? The model a = β*F + γ*v^2 has no intercept. We'll fit with intercept=False.
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, all_a)
    beta = reg.coef_[0]
    gamma = reg.coef_[1]
    pred = reg.predict(X)
    residuals = all_a - pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((all_a - np.mean(all_a))**2)
    r2_multi = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    multi_fit = {"beta": float(beta), "gamma": float(gamma), "R2": float(r2_multi),
                 "residual_mean": float(np.mean(residuals)),
                 "residual_std": float(np.std(residuals))}

    # 6. Generate new sequence: a - β*F - γ*v^2 for each experiment
    derived_series = []
    for eid in valid_ids:
        d = data[eid]
        a = d["a"]
        v = d["v"]
        F = d["F"]
        residual_seq = a - beta * F - gamma * v**2
        # Also generate a_minus_F (if relevant)
        # But analysis goal says "若发现好的组合，生成新序列如a_minus_F或a_plus_kv等"
        # We'll generate a_residual_F_v2 and a_minus_F
        a_minus_F = a - F
        # Save as derived series
        derived_series.append({
            "experiment_id": eid,
            "name": "a_residual_F_v2",
            "values": residual_seq.tolist(),
            "source_name": "a - beta*F - gamma*v^2, beta={:.4f}, gamma={:.4f}".format(beta, gamma),
            "provenance": "generated data processor: custom_data_analysis",
            "description": "Residual from multi-experiment fit a = beta*F + gamma*v^2"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "a_minus_F",
            "values": a_minus_F.tolist(),
            "source_name": "a - F_ext",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "Acceleration minus external force"
        })

    # Build observation string
    obs_lines = []
    obs_lines.append("自定义数据分析：对所有恒定外力实验(exp_03~exp_06)和无外力实验(exp_02)进行了统一分析。")
    obs_lines.append(f"处理实验: {', '.join(valid_ids)}")
    obs_lines.append("1. 提取了各实验的加速度a_sg和速度v_sg（exp_02通过SG滤波从位置估计）。")
    obs_lines.append("2. 绘制了所有实验的a vs v散点图（已保存）。")
    obs_lines.append("3. 每个实验独立拟合a = α + β*v 和 a = α + β*v^2 的结果：")
    for eid in valid_ids:
        f = fits[eid]
        lin = f["linear"]
        quad = f["quadratic"]
        obs_lines.append(f"   {eid}: 线性 α={lin['alpha']:.4f}, β={lin['beta']:.4f}, R²={lin['R2']:.4f}; 二次 α={quad['alpha']:.4f}, β={quad['beta']:.4f}, R²={quad['R2']:.4f}")
    obs_lines.append("4. 在相同速度(v=1,2,3)下比较不同外力的加速度值：")
    for vt, row in comparison_table.items():
        vals = ", ".join([f"{eid}: {row[eid]:.4f}" for eid in valid_ids])
        obs_lines.append(f"   {vt}: {vals}")
    obs_lines.append("5. 跨实验拟合模型 a = β*F + γ*v^2 (无截距)：")
    obs_lines.append(f"   β={multi_fit['beta']:.4f}, γ={multi_fit['gamma']:.4f}, R²={multi_fit['R2']:.4f}, 残差均值={multi_fit['residual_mean']:.6f}, 残差标准差={multi_fit['residual_std']:.6f}")
    obs_lines.append("6. 生成了新序列：a_residual_F_v2 (多实验拟合残差) 和 a_minus_F (加速度减外力)。")

    observation = "\n".join(obs_lines)

    # Build metrics
    metrics = {}
    for eid in valid_ids:
        f = fits[eid]
        metrics[f"{eid}_lin_alpha"] = f["linear"]["alpha"]
        metrics[f"{eid}_lin_beta"] = f["linear"]["beta"]
        metrics[f"{eid}_lin_R2"] = f["linear"]["R2"]
        metrics[f"{eid}_quad_alpha"] = f["quadratic"]["alpha"]
        metrics[f"{eid}_quad_beta"] = f["quadratic"]["beta"]
        metrics[f"{eid}_quad_R2"] = f["quadratic"]["R2"]
        metrics[f"{eid}_F_ext"] = data[eid]["F"]
    for vt, row in comparison_table.items():
        for eid in valid_ids:
            metrics[f"interp_{vt}_{eid}"] = row[eid]
    metrics["multi_fit_beta"] = multi_fit["beta"]
    metrics["multi_fit_gamma"] = multi_fit["gamma"]
    metrics["multi_fit_R2"] = multi_fit["R2"]
    metrics["multi_fit_residual_mean"] = multi_fit["residual_mean"]
    metrics["multi_fit_residual_std"] = multi_fit["residual_std"]

    figures = [scatter_path]

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
