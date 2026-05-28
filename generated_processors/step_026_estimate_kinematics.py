import numpy as np
from scipy.signal import savgol_filter
import os

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    if action != "estimate_kinematics":
        raise ValueError(f"Unexpected action: {action}")

    experiment_ids = params.get("experiment_ids", list(experiments.keys()))
    source_series = params.get("source_series", "q")
    window_length = params.get("window_length", 5)
    polyorder = params.get("polyorder", 2)
    overwrite = params.get("overwrite", True)

    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}")
    if polyorder >= window_length:
        raise ValueError(f"polyorder ({polyorder}) must be less than window_length ({window_length})")

    derived_series = []
    metrics = {}
    figures = []

    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]

        if source_series not in series:
            raise ValueError(f"Experiment {eid}: series '{source_series}' not available")
        q = np.array(series[source_series], dtype=float)
        t = np.array(series["t"], dtype=float)

        N = len(q)
        if N < window_length:
            raise ValueError(f"Experiment {eid}: series length {N} < window_length {window_length}")

        dt = config.get("dt", None)
        if dt is None:
            # 从 t 序列推断 dt
            if N > 1:
                dt = np.median(np.diff(t))
            else:
                dt = 1.0

        # 平滑
        q_smooth = savgol_filter(q, window_length, polyorder)
        rmse = np.sqrt(np.mean((q - q_smooth) ** 2))

        # 速度
        v = savgol_filter(q_smooth, window_length, polyorder, deriv=1, delta=dt)
        # 加速度
        a = savgol_filter(q_smooth, window_length, polyorder, deriv=2, delta=dt)

        # 构建 derived_series 条目
        def make_series(name, values):
            return {
                "experiment_id": eid,
                "name": name,
                "values": values.tolist(),
                "source_name": f"Savitzky-Golay filter (window={window_length}, poly={polyorder}) on {source_series}",
                "provenance": f"generated data processor: {action}",
                "description": f"Derived from {source_series} using sgolay with dt={dt:.4f}"
            }

        derived_series.append(make_series(f"{source_series}_smooth", q_smooth))
        derived_series.append(make_series("v", v))
        derived_series.append(make_series("a", a))

        # metrics
        metrics[f"{eid}_smooth_rmse"] = rmse
        metrics[f"{eid}_window_length"] = window_length
        metrics[f"{eid}_polyorder"] = polyorder
        metrics[f"{eid}_dt"] = dt

        # 画图（可选）
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(8, 6))
        axes[0].plot(t, q, 'b-', label='original', alpha=0.6)
        axes[0].plot(t, q_smooth, 'r-', label='smoothed', linewidth=2)
        axes[0].set_title(f"Experiment {eid}: Q(t) smoothing")
        axes[0].set_xlabel("t"); axes[0].set_ylabel("q"); axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(t, v, 'g-', label='v (savgol 1st deriv)')
        axes[1].plot(t, a, 'm-', label='a (savgol 2nd deriv)')
        axes[1].set_title("Velocity and acceleration")
        axes[1].set_xlabel("t"); axes[1].set_ylabel("value"); axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        fname = f"kinematics_{eid}.png"
        fpath = os.path.join(output_dir, fname)
        plt.savefig(fpath, dpi=100)
        plt.close()
        figures.append(fpath)

    # 跨实验统计
    rmse_list = [metrics[f"{eid}_smooth_rmse"] for eid in experiment_ids]
    metrics["smooth_rmse_mean"] = float(np.mean(rmse_list))
    metrics["smooth_rmse_std"] = float(np.std(rmse_list))
    metrics["smooth_rmse_list"] = rmse_list

    # 构建 observation
    obs_lines = []
    for eid in experiment_ids:
        dt_val = metrics[f"{eid}_dt"]
        rmse_val = metrics[f"{eid}_smooth_rmse"]
        obs_lines.append(
            f"实验 {eid}: {source_series} 平滑后 {source_series}_smooth RMSE={rmse_val:.6f}；"
            f"使用参数 window_length={window_length}, polyorder={polyorder}, dt={dt_val:.4f}。"
            f"已生成 {source_series}_smooth, v, a。"
        )
    obs_lines.append(
        f"跨实验平均平滑RMSE={metrics['smooth_rmse_mean']:.6f}±{metrics['smooth_rmse_std']:.6f}"
    )

    observation = "对实验 {} 进行了运动学估计。\n{}".format(
        str(experiment_ids), "\n".join(obs_lines)
    )

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
