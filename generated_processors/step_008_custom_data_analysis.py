import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def process(payload: dict) -> dict:
    parameters = payload.get("parameters", {})
    experiment_ids = parameters.get("experiment_ids", [])
    if not experiment_ids:
        raise ValueError("experiment_ids must be provided")
    experiments = payload.get("experiments", {})
    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
    output_dir = payload.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    window = 11
    polyorder = 2
    k_ref = 0.2685  # absolute slope from exp_03 given in analysis_goal

    results = {}
    derived_series = []
    figures = []
    metrics = {}

    for eid in experiment_ids:
        exp = experiments[eid]
        t = exp["series"]["t"]
        q = exp["series"]["q"]
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        n = len(t)

        if n < window:
            raise ValueError(f"Experiment {eid}: sequence length {n} < window size {window}")

        # Savitzky-Golay filtering for velocity and acceleration
        v = savgol_filter(q, window_length=window, polyorder=polyorder, deriv=1, delta=dt)
        a = savgol_filter(q, window_length=window, polyorder=polyorder, deriv=2, delta=dt)

        # Store derived series
        v_name = f"v_sg_{eid}"
        a_name = f"a_sg_{eid}"
        derived_series.append({
            "experiment_id": eid,
            "name": v_name,
            "values": v.tolist(),
            "source_name": f"Savitzky-Golay滤波(窗口{window}, polyorder{polyorder})一阶导数",
            "provenance": f"generated processor: custom_data_analysis for experiment {eid}",
            "description": f"速度序列（SG滤波，dt={dt}）"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": a_name,
            "values": a.tolist(),
            "source_name": f"Savitzky-Golay滤波(窗口{window}, polyorder{polyorder})二阶导数",
            "provenance": f"generated processor: custom_data_analysis for experiment {eid}",
            "description": f"加速度序列（SG滤波，dt={dt}）"
        })

        # a vs v linear fit
        coeffs = np.polyfit(v, a, 1)
        slope_av = coeffs[0]
        intercept_av = coeffs[1]
        a_pred = slope_av * v + intercept_av
        ss_res = np.sum((a - a_pred)**2)
        ss_tot = np.sum((a - np.mean(a))**2)
        r2_av = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # a vs t linear fit
        coeffs_t = np.polyfit(t, a, 1)
        slope_at = coeffs_t[0]

        # Mean acceleration
        mean_a = np.mean(a)

        # Store metrics for this experiment
        metrics[f"{eid}_v_mean"] = float(np.mean(v))
        metrics[f"{eid}_a_mean"] = float(mean_a)
        metrics[f"{eid}_av_slope"] = float(slope_av)
        metrics[f"{eid}_av_intercept"] = float(intercept_av)
        metrics[f"{eid}_av_R2"] = float(r2_av)
        metrics[f"{eid}_at_slope"] = float(slope_at)

        # Plot a vs v scatter and linear fit
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(v, a, s=8, alpha=0.7, label='data')
        v_line = np.linspace(v.min(), v.max(), 100)
        a_line = slope_av * v_line + intercept_av
        ax.plot(v_line, a_line, 'r-', label=f'fit: k={slope_av:.4f}, b={intercept_av:.4f}')
        ax.set_xlabel('v')
        ax.set_ylabel('a')
        ax.set_title(f'{eid}: a vs v')
        ax.legend()
        fig_path = os.path.join(output_dir, f'{eid}_a_vs_v.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        figures.append(fig_path)

    # ---- Exp_04 specific analysis: check a + k*v constant ----
    if "exp_04" in experiment_ids:
        exp04 = experiments["exp_04"]
        t04 = exp04["series"]["t"]
        q04 = exp04["series"]["q"]
        dt04 = t04[1] - t04[0] if len(t04) > 1 else 1.0
        v04 = savgol_filter(q04, window_length=window, polyorder=polyorder, deriv=1, delta=dt04)
        a04 = savgol_filter(q04, window_length=window, polyorder=polyorder, deriv=2, delta=dt04)
        combo = a04 + k_ref * v04
        combo_mean = float(np.mean(combo))
        combo_std = float(np.std(combo))
        metrics["exp_04_combo_mean"] = combo_mean
        metrics["exp_04_combo_std"] = combo_std

        # Slope difference if exp_03 is also analyzed
        if "exp_03" in experiment_ids:
            # exp_03 slope already computed above, retrieve from metrics
            k03 = metrics.get("exp_03_av_slope", None)
            k04_slope = metrics.get("exp_04_av_slope", None)
            if k03 is not None and k04_slope is not None:
                slope_diff = k04_slope - k03
                metrics["exp_03_vs_04_av_slope_difference"] = float(slope_diff)
            else:
                slope_diff = None
        else:
            slope_diff = None
    else:
        # If exp_04 not requested, still define key with nan to avoid missing
        metrics["exp_04_combo_mean"] = None
        metrics["exp_04_combo_std"] = None
        slope_diff = None

    # ---- Build observation ----
    obs_lines = []
    for eid in experiment_ids:
        m = metrics
        obs_lines.append(
            f"实验 {eid}: "
            f"加速度均值={m[f'{eid}_a_mean']:.6f}, "
            f"a-v线性拟合斜率={m[f'{eid}_av_slope']:.6f}, 截距={m[f'{eid}_av_intercept']:.6f}, R²={m[f'{eid}_av_R2']:.6f}; "
            f"a-t线性拟合斜率（加速度变化率）={m[f'{eid}_at_slope']:.6f}"
        )
    if "exp_04" in experiment_ids:
        obs_lines.append(
            f"exp_04 中组合量 a + {k_ref}*v 的均值={combo_mean:.6f}, 标准差={combo_std:.6f}"
        )
        if slope_diff is not None:
            obs_lines.append(f"exp_04与exp_03的a-v斜率差异 = {slope_diff:.6f}")
    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
