import os
import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    # --- parse parameters ---
    parameters = payload["parameters"]
    exp_ids = parameters.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(payload["experiments"].keys())
    analysis_goal = parameters.get("analysis_goal", "")
    output_dir = payload["output_dir"]
    experiments = payload["experiments"]

    # --- Step 1: estimate kinematics for exp_04 using Savitzky-Golay (window=11, polyorder=3) ---
    exp04 = experiments["exp_04"]
    t04 = np.array(exp04["series"]["t"])
    q04 = np.array(exp04["series"]["q"])
    n04 = len(t04)
    window = 11
    polyorder = 3
    # Savitzky-Golay smoothing
    q_smooth04 = savgol_filter(q04, window_length=window, polyorder=polyorder)
    dt04 = t04[1] - t04[0] if len(t04) > 1 else 0.1
    # velocity from smooth position using gradient
    v_sg04 = np.gradient(q_smooth04, dt04)
    # acceleration from velocity
    a_sg04 = np.gradient(v_sg04, dt04)

    # --- collect F_ext for all experiments ---
    def get_f_ext(eid):
        cfg = experiments[eid]["config"]
        # try common field names
        f = cfg.get("F_ext")
        if f is not None:
            return float(f)
        f = cfg.get("constant_force")
        if f is not None:
            return float(f)
        # fallback: default to 0
        return 0.0

    f_ext = {eid: get_f_ext(eid) for eid in exp_ids}

    # --- Step 2: gather a_sg, v_sg for all four experiments ---
    # For exp_04 use just computed v_sg04, a_sg04; for others use existing series.
    a_series = {}
    v_series = {}
    for eid in exp_ids:
        if eid == "exp_04":
            a_series[eid] = a_sg04
            v_series[eid] = v_sg04
        else:
            series = experiments[eid]["series"]
            if "a_sg" not in series or "v_sg" not in series:
                raise ValueError(f"Experiment {eid} missing a_sg or v_sg series")
            a_series[eid] = np.array(series["a_sg"])
            v_series[eid] = np.array(series["v_sg"])

    # --- Step 2: plot a_sg vs v_sg for all experiments ---
    colors = {"exp_01": "blue", "exp_02": "red", "exp_03": "green", "exp_04": "orange"}
    fig1, ax1 = plt.subplots(figsize=(8,6))
    for eid in exp_ids:
        ax1.scatter(v_series[eid], a_series[eid], c=colors.get(eid, "gray"), label=eid, s=10, alpha=0.7)
    ax1.set_xlabel("v_sg")
    ax1.set_ylabel("a_sg")
    ax1.set_title("a_sg vs v_sg for all experiments")
    ax1.legend()
    fig1.tight_layout()
    fig1_path = os.path.join(output_dir, "a_vs_v_all_experiments.png")
    fig1.savefig(fig1_path)
    plt.close(fig1)

    # --- Step 3: fit unified model a = α * F_ext + β * v + γ * v^2 ---
    # collect all data points
    X_list = []
    y_list = []
    for eid in exp_ids:
        f = f_ext[eid]
        v = v_series[eid]
        a = a_series[eid]
        # ensure same length
        n = len(v)
        X_eid = np.column_stack([np.full(n, f), v, v**2])
        X_list.append(X_eid)
        y_list.append(a)
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    # linear regression without intercept (model: a = α*F_ext + β*v + γ*v^2)
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X_all, y_all)
    alpha, beta, gamma = reg.coef_
    y_pred = reg.predict(X_all)
    r2 = r2_score(y_all, y_pred)

    # --- Step 4: plot a_sg / F_ext vs v_sg for experiments with non-zero F_ext ---
    fig2, ax2 = plt.subplots(figsize=(8,6))
    non_zero_f = [eid for eid in exp_ids if abs(f_ext[eid]) > 1e-12]
    for eid in non_zero_f:
        f = f_ext[eid]
        a_div_f = a_series[eid] / f
        ax2.scatter(v_series[eid], a_div_f, c=colors.get(eid, "gray"), label=eid, s=10, alpha=0.7)
    ax2.set_xlabel("v_sg")
    ax2.set_ylabel("a_sg / F_ext")
    ax2.set_title("a_sg/F_ext vs v_sg (non-zero F_ext)")
    ax2.legend()
    fig2.tight_layout()
    fig2_path = os.path.join(output_dir, "a_over_F_vs_v.png")
    fig2.savefig(fig2_path)
    plt.close(fig2)

    # --- prepare derived series for exp_04 ---
    derived_series = [
        {"experiment_id": "exp_04", "name": "q_smooth", "values": q_smooth04.tolist(),
         "source_name": "Savitzky-Golay filter window=11 polyorder=3 on q",
         "provenance": "generated data processor: step_030_custom_data_analysis",
         "description": "smoothed position using Savitzky-Golay (window=11, polyorder=3)"},
        {"experiment_id": "exp_04", "name": "v_sg", "values": v_sg04.tolist(),
         "source_name": "gradient of q_smooth / dt",
         "provenance": "generated data processor: step_030_custom_data_analysis",
         "description": "velocity estimated from smoothed position gradient"},
        {"experiment_id": "exp_04", "name": "a_sg", "values": a_sg04.tolist(),
         "source_name": "gradient of v_sg / dt",
         "provenance": "generated data processor: step_030_custom_data_analysis",
         "description": "acceleration estimated from velocity gradient"}
    ]

    # --- metrics ---
    metrics = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "R2": r2,
        "n_sample": len(y_all),
        "exp_04_q_smooth_mean": float(np.mean(q_smooth04)),
        "exp_04_v_sg_mean": float(np.mean(v_sg04)),
        "exp_04_a_sg_mean": float(np.mean(a_sg04))
    }

    # --- observation ---
    obs = (
        f"对实验 {exp_ids} 执行自定义分析。\n"
        f"Step 1: 为 exp_04 估计运动学 (Savitzky-Golay 窗口11, 阶次3)：\n"
        f"  q_smooth 均值={metrics['exp_04_q_smooth_mean']:.6f}, "
        f"v_sg 均值={metrics['exp_04_v_sg_mean']:.6f}, "
        f"a_sg 均值={metrics['exp_04_a_sg_mean']:.6f}\n"
        f"Step 2: 绘制所有实验的 a_sg vs v_sg 散点图，已保存。\n"
        f"Step 3: 统一模型拟合 a = α*F_ext + β*v + γ*v^2, 结果：\n"
        f"  α={alpha:.6f}, β={beta:.6f}, γ={gamma:.6f}, R²={r2:.6f}\n"
        f"Step 4: 绘制 a_sg/F_ext vs v_sg (F_ext非零实验) 散点图，已保存。\n"
        f"返回 exp_04 的 q_smooth, v_sg, a_sg 派生序列。"
    )

    return {
        "observation": obs,
        "derived_series": derived_series,
        "figures": [fig1_path, fig2_path],
        "metrics": metrics
    }
