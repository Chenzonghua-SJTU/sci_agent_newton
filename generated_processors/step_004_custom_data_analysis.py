import json
import math
import statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Any, Dict, List

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # 只处理 exp_02 和 exp_03
    target_exps = ["exp_02", "exp_03"]
    for eid in target_exps:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")

    window = 7
    polyorder = 2
    dt = 0.1  # 根据上下文 dt 固定

    derived_series_list = []
    metrics_dict = {}
    figures_list = []

    # 准备画图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex='col')
    # 0行: v vs t; 1行: a vs t
    colors = {'exp_02': 'blue', 'exp_03': 'red'}

    for idx, eid in enumerate(target_exps):
        exp = experiments[eid]
        t = exp["series"]["t"]
        q = exp["series"]["q"]
        t_arr = np.array(t)
        q_arr = np.array(q)
        n = len(t_arr)
        if len(q_arr) != n:
            raise ValueError(f"{eid}: t and q length mismatch")

        # 使用 SG 滤波器直接计算速度和加速度
        if n < window:
            raise ValueError(f"{eid}: too few points ({n}) for SG window {window}")
        v_arr = savgol_filter(q_arr, window_length=window, polyorder=polyorder, deriv=1, delta=dt)
        a_arr = savgol_filter(q_arr, window_length=window, polyorder=polyorder, deriv=2, delta=dt)
        # 对加速度再做一次平滑？直接使用 SG 导数是合理的

        # 统计信息
        v_min = float(np.min(v_arr))
        v_max = float(np.max(v_arr))
        v_mean = float(np.mean(v_arr))
        v_std = float(np.std(v_arr, ddof=1))
        a_min = float(np.min(a_arr))
        a_max = float(np.max(a_arr))
        a_mean = float(np.mean(a_arr))
        a_std = float(np.std(a_arr, ddof=1))

        # 检查加速度是否恒定：计算 a 对 t 的线性回归斜率
        coeffs = np.polyfit(t_arr, a_arr, 1)
        a_slope = coeffs[0]  # 斜率
        # 恒定判据：std 相对于均值（如果均值接近0则看绝对 std）
        if abs(a_mean) > 1e-6:
            cv = a_std / abs(a_mean)
        else:
            cv = a_std  # 如果均值接近0，直接看 std
        is_constant = cv < 0.1  # 变异系数小于 0.1 视为恒定

        # 已知外力
        F_ext = exp["config"]["F_ext"]  # exp_02:0.0, exp_03:1.0
        a_diff = a_mean - F_ext

        # 记录指标
        prefix = f"{eid}"
        metrics_dict.update({
            f"{prefix}_v_min": v_min,
            f"{prefix}_v_max": v_max,
            f"{prefix}_v_mean": v_mean,
            f"{prefix}_v_std": v_std,
            f"{prefix}_a_min": a_min,
            f"{prefix}_a_max": a_max,
            f"{prefix}_a_mean": a_mean,
            f"{prefix}_a_std": a_std,
            f"{prefix}_a_slope": a_slope,
            f"{prefix}_is_constant": is_constant,
            f"{prefix}_F_ext": F_ext,
            f"{prefix}_a_diff_from_F_ext": a_diff,
        })

        # 注册派生序列
        derived_series_list.append({
            "experiment_id": eid,
            "name": "v_sg",
            "values": v_arr.tolist(),
            "source_name": f"Savitzky-Golay deriv=1 (window={window}, polyorder={polyorder}) on q",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "Velocity estimated via SG filter"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": "a_sg",
            "values": a_arr.tolist(),
            "source_name": f"Savitzky-Golay deriv=2 (window={window}, polyorder={polyorder}) on q",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "Acceleration estimated via SG filter"
        })

        # 画图
        col = idx  # 0 for exp_02, 1 for exp_03
        ax_v = axes[0, col]
        ax_a = axes[1, col]
        color = colors[eid]
        ax_v.plot(t_arr, v_arr, color=color, label=f"{eid} v")
        ax_v.set_title(f"{eid}: Velocity vs Time")
        ax_v.set_ylabel("v (m/s)")
        ax_v.grid(True)
        ax_v.legend()

        ax_a.plot(t_arr, a_arr, color=color, label=f"{eid} a")
        ax_a.axhline(y=F_ext, color='gray', linestyle='--', label=f"F_ext={F_ext}")
        ax_a.set_title(f"{eid}: Acceleration vs Time")
        ax_a.set_xlabel("t (s)")
        ax_a.set_ylabel("a (m/s^2)")
        ax_a.grid(True)
        ax_a.legend()

    # 保存图片
    fig.tight_layout()
    fig_path = Path(output_dir) / "kinematics_exp_02_03.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    figures_list.append(str(fig_path))

    # 构建 observation
    obs_lines = []
    for eid in target_exps:
        p = metrics_dict[f"{eid}_F_ext"]
        am = metrics_dict[f"{eid}_a_mean"]
        as_ = metrics_dict[f"{eid}_a_std"]
        ic = metrics_dict[f"{eid}_is_constant"]
        vm = metrics_dict[f"{eid}_v_mean"]
        vs = metrics_dict[f"{eid}_v_std"]
        ad = metrics_dict[f"{eid}_a_diff_from_F_ext"]
        obs_lines.append(
            f"{eid}: SG(win={window}, poly={polyorder}) derived v and a. "
            f"v mean={vm:.4f}, std={vs:.4f}. a mean={am:.4f}, std={as_:.4f}. "
            f"a differs from F_ext={p:.1f} by {ad:.4f}. "
            f"Acceleration {'appears constant' if ic else 'not constant'} (std/|mean|={as_/abs(am) if abs(am)>1e-6 else as_:.4f})."
        )
    obs_lines.append("Plots saved: v vs t and a vs t for each experiment.")
    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures_list,
        "metrics": metrics_dict
    }
