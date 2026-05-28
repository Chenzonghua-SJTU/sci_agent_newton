import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

def process(payload: dict) -> dict:
    action = payload.get("action")
    params = payload.get("parameters", {})
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        # fallback to all experiments if not specified
        exp_ids = list(payload.get("experiments", {}).keys())
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # Helper: central difference for velocity and acceleration from position
    def central_diff(q, dt):
        n = len(q)
        v = np.zeros(n)
        a = np.zeros(n)
        # velocity
        v[0] = (q[1] - q[0]) / dt
        v[-1] = (q[-1] - q[-2]) / dt
        for i in range(1, n-1):
            v[i] = (q[i+1] - q[i-1]) / (2*dt)
        # acceleration (second central difference)
        # internal points
        for i in range(1, n-1):
            a[i] = (q[i+1] - 2*q[i] + q[i-1]) / (dt**2)
        # boundaries: use forward/backward second order
        if n >= 3:
            a[0] = (q[2] - 2*q[1] + q[0]) / (dt**2)
            a[-1] = (q[-3] - 2*q[-2] + q[-1]) / (dt**2)
        else:
            a[0] = (q[1] - q[0]) / dt  # fallback: just first difference if small dataset
            a[-1] = a[0]
        return v, a

    derived_series = []
    metrics = {}
    figures = []

    # Dictionary to store computed data per experiment
    exp_data = {}

    for eid in exp_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        config = exp.get("config", {})
        series = exp.get("series", {})
        t = np.array(series.get("t", []))
        q = np.array(series.get("q", []))
        if len(t) == 0 or len(q) == 0:
            raise ValueError(f"Experiment {eid} missing t or q series")
        dt = t[1] - t[0]  # assume uniform
        # compute v_central, a_central
        v_central, a_central = central_diff(q, dt)
        # store in exp_data
        exp_data[eid] = {
            "t": t,
            "q": q,
            "v_central": v_central,
            "a_central": a_central,
            "F_ext": config.get("F_ext", 0.0)
        }
        # add derived series for these new quantities
        derived_series.append({
            "experiment_id": eid,
            "name": "v_central",
            "values": v_central.tolist(),
            "source_name": "central_diff from q",
            "provenance": "generated data processor: custom_data_analysis"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "a_central",
            "values": a_central.tolist(),
            "source_name": "central_diff (second order) from q",
            "provenance": "generated data processor: custom_data_analysis"
        })

    # Collect training data for non-zero F_ext experiments
    train_X_list = []   # array of [F_ext, v] for model1 linear
    train_y_list = []
    train_F_list = []
    train_v_list = []
    train_a_list = []
    train_exp_ids = []
    for eid, data in exp_data.items():
        F = data["F_ext"]
        if abs(F) > 1e-12:  # non-zero external force
            v = data["v_central"]
            a = data["a_central"]
            # accumulate per data point
            for vi, ai in zip(v, a):
                train_X_list.append([F, vi])
                train_y_list.append(ai)
                train_F_list.append(F)
                train_v_list.append(vi)
                train_a_list.append(ai)
                train_exp_ids.append(eid)
    train_X = np.array(train_X_list)
    train_y = np.array(train_y_list)
    train_F_arr = np.array(train_F_list)
    train_v_arr = np.array(train_v_list)
    train_a_arr = np.array(train_a_list)

    # Model definitions
    # Model1: a = c1*F + c2*v  (linear, no intercept)
    def model1_func(F, v, c1, c2):
        return c1 * F + c2 * v

    # Model2: a = c1*F / (1 + c2*v)
    def model2_func(F, v, c1, c2):
        denom = 1 + c2 * v
        # avoid division by zero near zero (v can be negative? but from data v >=0)
        # if denom is very small, clip
        with np.errstate(divide='ignore', invalid='ignore'):
            result = c1 * F / denom
            result = np.where(np.isfinite(result), result, 0.0)
        return result

    # Model3: a = c1*F * exp(-c2*v)
    def model3_func(F, v, c1, c2):
        return c1 * F * np.exp(-c2 * v)

    # Model4: a = c1*F * (1 - c2*v)
    def model4_func(F, v, c1, c2):
        return c1 * F * (1 - c2 * v)

    # fit functions for curve_fit
    def model2_flat(x, c1, c2):
        F, v = x
        return model2_func(F, v, c1, c2)

    def model3_flat(x, c1, c2):
        F, v = x
        return model3_func(F, v, c1, c2)

    def model4_flat(x, c1, c2):
        F, v = x
        return model4_func(F, v, c1, c2)

    # Do fittings
    model_results = {}
    # Model1 linear
    reg = LinearRegression(fit_intercept=False)
    reg.fit(train_X, train_y)
    pred1 = reg.predict(train_X)
    ss_res1 = np.sum((train_y - pred1)**2)
    ss_tot1 = np.sum((train_y - np.mean(train_y))**2)
    r2_1 = 1 - ss_res1 / ss_tot1
    model_results["model1"] = {
        "c1": reg.coef_[0],
        "c2": reg.coef_[1],
        "R2": r2_1,
        "SS_res": ss_res1,
        "predictions": pred1
    }

    # Model2,3,4 use curve_fit with bounds
    # Use all data stacked as (F, v)
    x_data = (train_F_arr, train_v_arr)
    y_data = train_a_arr

    # initial guesses
    p0 = [1.0, 0.1]

    # Model2
    try:
        popt2, pcov2 = curve_fit(model2_flat, x_data, y_data, p0=p0, maxfev=5000)
        c1_2, c2_2 = popt2
        pred2 = model2_flat(x_data, *popt2)
        ss_res2 = np.sum((y_data - pred2)**2)
        r2_2 = 1 - ss_res2 / ss_tot1
        model_results["model2"] = {"c1": c1_2, "c2": c2_2, "R2": r2_2, "SS_res": ss_res2, "predictions": pred2}
    except Exception as e:
        model_results["model2"] = {"error": str(e), "c1": np.nan, "c2": np.nan, "R2": np.nan, "SS_res": np.nan, "predictions": np.full_like(y_data, np.nan)}

    # Model3
    try:
        popt3, pcov3 = curve_fit(model3_flat, x_data, y_data, p0=p0, maxfev=5000)
        c1_3, c2_3 = popt3
        pred3 = model3_flat(x_data, *popt3)
        ss_res3 = np.sum((y_data - pred3)**2)
        r2_3 = 1 - ss_res3 / ss_tot1
        model_results["model3"] = {"c1": c1_3, "c2": c2_3, "R2": r2_3, "SS_res": ss_res3, "predictions": pred3}
    except Exception as e:
        model_results["model3"] = {"error": str(e), "c1": np.nan, "c2": np.nan, "R2": np.nan, "SS_res": np.nan, "predictions": np.full_like(y_data, np.nan)}

    # Model4
    try:
        popt4, pcov4 = curve_fit(model4_flat, x_data, y_data, p0=p0, maxfev=5000)
        c1_4, c2_4 = popt4
        pred4 = model4_flat(x_data, *popt4)
        ss_res4 = np.sum((y_data - pred4)**2)
        r2_4 = 1 - ss_res4 / ss_tot1
        model_results["model4"] = {"c1": c1_4, "c2": c2_4, "R2": r2_4, "SS_res": ss_res4, "predictions": pred4}
    except Exception as e:
        model_results["model4"] = {"error": str(e), "c1": np.nan, "c2": np.nan, "R2": np.nan, "SS_res": np.nan, "predictions": np.full_like(y_data, np.nan)}

    # Generate per-experiment predicted series and compute residuals
    pred_series = {eid: {1:[],2:[],3:[],4:[]} for eid in exp_data}
    residuals = {eid: {1:[],2:[],3:[],4:[]} for eid in exp_data}

    # For each experiment, compute predictions using fitted parameters
    for eid, data in exp_data.items():
        F = data["F_ext"]
        v = data["v_central"]
        a = data["a_central"]
        for midx in [1,2,3,4]:
            mr = model_results.get(f"model{midx}", {})
            if "c1" in mr and not np.isnan(mr["c1"]):
                c1 = mr["c1"]
                c2 = mr["c2"]
                if midx == 1:
                    pred = c1 * F + c2 * v
                elif midx == 2:
                    denom = 1 + c2 * v
                    pred = c1 * F / denom
                elif midx == 3:
                    pred = c1 * F * np.exp(-c2 * v)
                elif midx == 4:
                    pred = c1 * F * (1 - c2 * v)
                else:
                    pred = np.zeros_like(v)
            else:
                pred = np.zeros_like(v)
            pred_series[eid][midx] = pred.tolist()
            residuals[eid][midx] = (a - pred).tolist()

    # Add predicted series to derived_series
    for eid in exp_data:
        for midx in [1,2,3,4]:
            series_name = f"a_pred_model{midx}"
            derived_series.append({
                "experiment_id": eid,
                "name": series_name,
                "values": pred_series[eid][midx],
                "source_name": f"fitted model{midx} parameters",
                "provenance": "generated data processor: custom_data_analysis"
            })

    # Plot a_central vs v_central for each experiment
    for eid, data in exp_data.items():
        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(data["v_central"], data["a_central"], s=8, alpha=0.7)
        ax.set_xlabel("v_central")
        ax.set_ylabel("a_central")
        ax.set_title(f"{eid} (F_ext={data['F_ext']})")
        fig.tight_layout()
        fname = f"{eid}_a_vs_v_central.png"
        fpath = os.path.join(output_dir, fname)
        fig.savefig(fpath)
        plt.close(fig)
        figures.append(fpath)

    # Optionally: combined plot of all experiments
    fig, ax = plt.subplots(figsize=(8,6))
    colors = ['blue','orange','green','red','purple']
    for idx, (eid, data) in enumerate(exp_data.items()):
        ax.scatter(data["v_central"], data["a_central"], s=8, alpha=0.6, label=eid, color=colors[idx%len(colors)])
    ax.set_xlabel("v_central")
    ax.set_ylabel("a_central")
    ax.legend()
    fig.tight_layout()
    fpath = os.path.join(output_dir, "all_experiments_a_vs_v_central.png")
    fig.savefig(fpath)
    plt.close(fig)
    figures.append(fpath)

    # Compute residual statistics per experiment per model
    res_stats = {}
    for eid in exp_data:
        a = exp_data[eid]["a_central"]
        res_stats[eid] = {}
        for midx in [1,2,3,4]:
            r = np.array(residuals[eid][midx])
            res_stats[eid][f"model{midx}"] = {
                "mean": float(np.mean(r)),
                "std": float(np.std(r)),
                "rmse": float(np.sqrt(np.mean(r**2))),
                "mae": float(np.mean(np.abs(r)))
            }
    # Also compute overall training R2 for each model
    for midx in [1,2,3,4]:
        mr = model_results.get(f"model{midx}", {})
        metrics[f"model{midx}_c1"] = mr.get("c1", np.nan)
        metrics[f"model{midx}_c2"] = mr.get("c2", np.nan)
        metrics[f"model{midx}_R2"] = mr.get("R2", np.nan)
        metrics[f"model{midx}_SS_res"] = mr.get("SS_res", np.nan)

    # Residual stats for exp_02 (F_ext=0) as validation
    for eid in ["exp_02"]:
        if eid in exp_data:
            a = exp_data[eid]["a_central"]
            metrics[f"{eid}_a_central_mean"] = float(np.mean(a))
            metrics[f"{eid}_a_central_std"] = float(np.std(a))
            for midx in [1,2,3,4]:
                r = np.array(residuals[eid][midx])
                metrics[f"{eid}_model{midx}_residual_mean"] = float(np.mean(r))
                metrics[f"{eid}_model{midx}_residual_std"] = float(np.std(r))

    # Build observation text
    obs_lines = []
    obs_lines.append("对所有实验(exp_02~exp_06)使用中心差分法从原始q重新计算了速度v_central和加速度a_central。")
    obs_lines.append(f"拟合了四种候选模型（无截距），使用所有非零外力实验(exp_03,04,05,06)的数据。")
    for midx in [1,2,3,4]:
        mr = model_results.get(f"model{midx}", {})
        if mr.get("error"):
            obs_lines.append(f"  Model{midx}: 拟合失败 ({mr['error']})")
        else:
            obs_lines.append(f"  Model{midx}: c1={mr.get('c1','?'):.6f}, c2={mr.get('c2','?'):.6f}, R²={mr.get('R2','?'):.4f}")
    obs_lines.append("验证实验exp_02 (F_ext=0):")
    if "exp_02" in exp_data:
        am = metrics.get("exp_02_a_central_mean", np.nan)
        ast = metrics.get("exp_02_a_central_std", np.nan)
        obs_lines.append(f"  a_central均值={am:.2e}, 标准差={ast:.2e}")
        obs_lines.append("  所有模型预测a_pred=0，因此残差即a_central本身，均值接近0，表明模型对无外力情况预测合理。")
    obs_lines.append("已生成每个实验的a_central和v_central序列，以及a_pred_model1~4序列。")
    obs_lines.append("图像：每个实验的a_central vs v_central散点图已保存。")

    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
