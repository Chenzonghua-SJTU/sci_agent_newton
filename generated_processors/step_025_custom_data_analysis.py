import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import os


def process(payload):
    output_dir = payload.get("output_dir", ".")
    experiments = payload.get("experiments", {})
    parameters = payload.get("parameters", {})
    exp_ids = parameters.get("experiment_ids", list(experiments.keys()))

    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")

    results = []
    derived_series_list = []
    figures = []
    metrics = {}

    for eid in exp_ids:
        exp = experiments[eid]
        config = exp.get("config", {})
        series = exp.get("series", {})
        available_series = exp.get("available_series", [])

        t = np.array(series.get("t", []))
        if len(t) == 0:
            raise ValueError(f"Experiment {eid} has no time series 't'")
        q = np.array(series.get("q", []))
        if len(q) == 0:
            raise ValueError(f"Experiment {eid} has no position series 'q'")
        n = len(t)

        F_ext = config.get("F_ext", None)
        if F_ext is None:
            F_ext = config.get("constant_force", 0.0)

        # try to use existing velocity, acceleration, drag
        v_raw = None
        a_raw = None
        drag_raw = None
        if "v_est" in available_series and "v_est" in series:
            v_raw = np.array(series["v_est"])
        elif "velocity" in available_series and "velocity" in series:
            v_raw = np.array(series["velocity"])
        if "a_est" in available_series and "a_est" in series:
            a_raw = np.array(series["a_est"])
        elif "acceleration" in available_series and "acceleration" in series:
            a_raw = np.array(series["acceleration"])
        if "drag" in available_series and "drag" in series:
            drag_raw = np.array(series["drag"])

        if v_raw is None or a_raw is None:
            # need to estimate kinematics
            if n < 5:
                window = n if n % 2 == 1 else n - 1
                if window < 3:
                    raise ValueError(f"Experiment {eid}: too few points ({n}) for kinematics estimation")
            else:
                window = min(5, n if n % 2 == 1 else n - 1)
            if window % 2 == 0:
                window -= 1
            if window < 3:
                window = 3
            dt = t[1] - t[0] if len(t) > 1 else 0.1
            q_smooth = savgol_filter(q, window_length=window, polyorder=2)
            v_est = savgol_filter(q, window_length=window, polyorder=2, deriv=1, delta=dt)
            a_est = savgol_filter(q, window_length=window, polyorder=2, deriv=2, delta=dt)
            derived_series_list.append({
                "experiment_id": eid,
                "name": "v_est",
                "values": v_est.tolist(),
                "source_name": "savgol_filter derivative of q",
                "provenance": "generated data processor: custom_data_analysis",
                "description": "estimated velocity"
            })
            derived_series_list.append({
                "experiment_id": eid,
                "name": "a_est",
                "values": a_est.tolist(),
                "source_name": "savgol_filter second derivative of q",
                "provenance": "generated data processor: custom_data_analysis",
                "description": "estimated acceleration"
            })
            v = v_est
            a = a_est
            drag = F_ext - a
            derived_series_list.append({
                "experiment_id": eid,
                "name": "drag",
                "values": drag.tolist(),
                "source_name": f"F_ext - a_est (F_ext={F_ext})",
                "provenance": "generated data processor: custom_data_analysis",
                "description": "drag force"
            })
        else:
            v = v_raw
            a = a_raw
            if drag_raw is None:
                drag = F_ext - a
                derived_series_list.append({
                    "experiment_id": eid,
                    "name": "drag",
                    "values": drag.tolist(),
                    "source_name": f"F_ext - a (F_ext={F_ext})",
                    "provenance": "generated data processor: custom_data_analysis",
                    "description": "drag force"
                })
            else:
                drag = drag_raw

        if not (len(v) == len(drag) == n):
            raise ValueError(f"Experiment {eid}: length mismatch v={len(v)}, drag={len(drag)}, t={n}")

        # filter out non-positive values for fitting
        mask = (v > 0) & (drag > 0)
        v_pos = v[mask]
        drag_pos = drag[mask]
        if len(v_pos) < 5:
            raise ValueError(f"Experiment {eid}: insufficient valid points (v>0,drag>0): {len(v_pos)}")

        # power law: log(drag) = log(k) + b*log(v)
        log_v = np.log(v_pos).reshape(-1, 1)
        log_drag = np.log(drag_pos)
        reg_power = LinearRegression().fit(log_v, log_drag)
        b = reg_power.coef_[0]
        log_k = reg_power.intercept_
        k = np.exp(log_k)
        pred_power = k * (v_pos ** b)
        ss_res_power = np.sum((drag_pos - pred_power)**2)
        ss_tot_power = np.sum((drag_pos - np.mean(drag_pos))**2)
        R2_power = 1 - ss_res_power / ss_tot_power
        rmse_power = np.sqrt(np.mean((drag_pos - pred_power)**2))

        # sqrt model: drag = c + d*sqrt(v)
        sqrt_v = np.sqrt(v_pos).reshape(-1, 1)
        reg_sqrt = LinearRegression().fit(sqrt_v, drag_pos)
        d = reg_sqrt.coef_[0]
        c = reg_sqrt.intercept_
        pred_sqrt = c + d * np.sqrt(v_pos)
        ss_res_sqrt = np.sum((drag_pos - pred_sqrt)**2)
        ss_tot_sqrt = np.sum((drag_pos - np.mean(drag_pos))**2)
        R2_sqrt = 1 - ss_res_sqrt / ss_tot_sqrt
        rmse_sqrt = np.sqrt(np.mean((drag_pos - pred_sqrt)**2))

        # store metrics per experiment
        metrics[f"{eid}_b"] = b
        metrics[f"{eid}_k"] = k
        metrics[f"{eid}_c"] = c
        metrics[f"{eid}_d"] = d
        metrics[f"{eid}_R2_power"] = R2_power
        metrics[f"{eid}_rmse_power"] = rmse_power
        metrics[f"{eid}_R2_sqrt"] = R2_sqrt
        metrics[f"{eid}_rmse_sqrt"] = rmse_sqrt

        # plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(v, drag, s=10, label='data', alpha=0.7)
        v_sort = np.sort(v_pos)
        ax.plot(v_sort, k * (v_sort ** b), 'r-', label=f'power: drag={k:.4f}*v^{b:.4f}')
        ax.plot(v_sort, c + d * np.sqrt(v_sort), 'g--', label=f'sqrt: drag={c:.4f}+{d:.4f}*sqrt(v)')
        ax.set_xlabel('v')
        ax.set_ylabel('drag')
        ax.set_title(f'{eid} (F_ext={F_ext}): drag vs v')
        ax.legend()
        ax.grid(True)
        textstr = f'Power: R²={R2_power:.4f}, RMSE={rmse_power:.4f}\nSqrt: R²={R2_sqrt:.4f}, RMSE={rmse_sqrt:.4f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        fig_path = os.path.join(output_dir, f"drag_vs_v_{eid}.png")
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        figures.append(fig_path)

        results.append({
            "id": eid,
            "F_ext": F_ext,
            "b": b,
            "k": k,
            "c": c,
            "d": d,
            "R2_power": R2_power,
            "rmse_power": rmse_power,
            "R2_sqrt": R2_sqrt,
            "rmse_sqrt": rmse_sqrt,
            "n_valid": len(v_pos)
        })

    # cross‑experiment analysis
    F_exts = [r["F_ext"] for r in results]
    bs = [r["b"] for r in results]
    ks = [r["k"] for r in results]
    cs = [r["c"] for r in results]
    b_mean = float(np.mean(bs))
    b_std = float(np.std(bs, ddof=1)) if len(bs) > 1 else 0.0
    metrics["b_mean"] = b_mean
    metrics["b_std"] = b_std

    if len(F_exts) >= 3:
        X_F = np.array(F_exts).reshape(-1, 1)
        reg_k = LinearRegression().fit(X_F, ks)
        k_slope = reg_k.coef_[0]
        k_intercept = reg_k.intercept_
        R2_k = reg_k.score(X_F, ks)
        metrics["k_vs_F_slope"] = k_slope
        metrics["k_vs_F_intercept"] = k_intercept
        metrics["R2_k_vs_F"] = R2_k

        reg_c = LinearRegression().fit(X_F, cs)
        c_slope = reg_c.coef_[0]
        c_intercept = reg_c.intercept_
        R2_c = reg_c.score(X_F, cs)
        metrics["c_vs_F_slope"] = c_slope
        metrics["c_vs_F_intercept"] = c_intercept
        metrics["R2_c_vs_F"] = R2_c
    else:
        metrics["k_vs_F_slope"] = None
        metrics["k_vs_F_intercept"] = None
        metrics["R2_k_vs_F"] = None
        metrics["c_vs_F_slope"] = None
        metrics["c_vs_F_intercept"] = None
        metrics["R2_c_vs_F"] = None

    # build observation text
    lines = []
    lines.append("对各实验分别进行了drag vs v的拟合：幂律 drag = k * v^b 和线性 sqrt drag = c + d*sqrt(v)。")
    for r in results:
        lines.append(
            f"  {r['id']} (F_ext={r['F_ext']}): b={r['b']:.4f}, k={r['k']:.4f}, "
            f"R²_power={r['R2_power']:.4f}, RMSE_power={r['rmse_power']:.4f}; "
            f"c={r['c']:.4f}, d={r['d']:.4f}, R²_sqrt={r['R2_sqrt']:.4f}, "
            f"RMSE_sqrt={r['rmse_sqrt']:.4f} (有效点 {r['n_valid']})"
        )
    lines.append(f"所有实验b的均值={b_mean:.4f}，标准差={b_std:.4f}。")
    if len(F_exts) >= 3:
        lines.append(
            f"k与F_ext线性回归：斜率={k_slope:.4f}，截距={k_intercept:.4f}，R²={R2_k:.4f}"
        )
        lines.append(
            f"c与F_ext线性回归：斜率={c_slope:.4f}，截距={c_intercept:.4f}，R²={R2_c:.4f}"
        )
    observation = "\n".join(lines)

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics
    }
