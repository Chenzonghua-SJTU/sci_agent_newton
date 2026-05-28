import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 参数解析
    experiment_ids = params.get("experiment_ids", list(experiments.keys()))
    expression = params.get("expression", "")
    output_name = params.get("output_name", expression + "_test")

    if not expression:
        raise ValueError("expression parameter is required for test_candidate_expression")
    if not experiment_ids:
        raise ValueError("experiment_ids parameter is required")

    # 检查每个实验是否有所需的序列
    for exp_id in experiment_ids:
        if exp_id not in experiments:
            raise ValueError(f"Experiment {exp_id} not found in payload")
        series = experiments[exp_id].get("series", {})
        if expression not in series:
            raise ValueError(
                f"Expression sequence '{expression}' not found in experiment {exp_id}. "
                f"Available series: {list(series.keys())}"
            )
        if "t" not in series:
            raise ValueError(f"Time series 't' not found in experiment {exp_id}")

    # 收集数据
    stats = {}
    derived_series_list = []
    t_data = {}  # 用于绘图
    candidate_data = {}

    for exp_id in experiment_ids:
        series = experiments[exp_id]["series"]
        t = np.array(series["t"])
        c = np.array(series[expression])

        # 计算统计
        c_mean = float(np.mean(c))
        c_std = float(np.std(c, ddof=0))
        c_min = float(np.min(c))
        c_max = float(np.max(c))
        abs_dev_from_1 = abs(c_mean - 1.0)
        max_dev_from_1 = float(np.max(np.abs(c - 1.0)))
        rms_dev_from_1 = float(np.sqrt(np.mean((c - 1.0) ** 2)))

        stats[exp_id] = {
            "mean": c_mean,
            "std": c_std,
            "min": c_min,
            "max": c_max,
            "abs_dev_from_1": abs_dev_from_1,
            "max_dev_from_1": max_dev_from_1,
            "rms_dev_from_1": rms_dev_from_1,
        }

        t_data[exp_id] = t
        candidate_data[exp_id] = c

        # 构造派生序列：按 output_name 返回原始值（或可考虑保留）
        derived_series_list.append({
            "experiment_id": exp_id,
            "name": output_name,
            "values": c.tolist(),
            "source_name": f"copy of '{expression}'",
            "provenance": "generated data processor: test_candidate_expression",
            "description": f"test of {expression} (copied values) for experiment {exp_id}"
        })

    # 构建观察文本
    lines = [f"测试候选表达式 '{expression}' 在指定实验中的表现。"]
    lines.append("统计摘要：")
    for exp_id in experiment_ids:
        s = stats[exp_id]
        lines.append(
            f"  {exp_id}: mean={s['mean']:.6f}, std={s['std']:.6f}, "
            f"|mean-1|={s['abs_dev_from_1']:.6f}, max|C-1|={s['max_dev_from_1']:.6f}, "
            f"RMS(C-1)={s['rms_dev_from_1']:.6f}"
        )
    # 计算所有实验的合并统计
    all_c = np.concatenate([candidate_data[exp_id] for exp_id in experiment_ids])
    global_mean = float(np.mean(all_c))
    global_std = float(np.std(all_c, ddof=0))
    global_abs_dev = abs(global_mean - 1.0)
    global_max_dev = float(np.max(np.abs(all_c - 1.0)))
    global_rms = float(np.sqrt(np.mean((all_c - 1.0) ** 2)))
    lines.append("全局统计（合并所有实验）：")
    lines.append(f"  mean={global_mean:.6f}, std={global_std:.6f}, "
                 f"|mean-1|={global_abs_dev:.6f}, max|C-1|={global_max_dev:.6f}, "
                 f"RMS(C-1)={global_rms:.6f}")

    observation = "\n".join(lines)

    # 绘制图形
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_ids)))
    for idx, exp_id in enumerate(experiment_ids):
        t = t_data[exp_id]
        c = candidate_data[exp_id]
        axes.plot(t, c, label=f"{exp_id} (mean={stats[exp_id]['mean']:.4f})",
                  color=colors[idx], linewidth=1.5)
    axes.axhline(y=1.0, color='k', linestyle='--', linewidth=1, label='y=1 (理想)')
    axes.set_xlabel("Time t")
    axes.set_ylabel(expression)
    axes.set_title(f"Test of candidate expression: {expression}")
    axes.legend(loc='best')
    axes.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_filename = f"{output_name}_vs_t.png"
    plot_path = os.path.join(output_dir, plot_filename)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    # 构造 metrics
    metrics = {}
    for exp_id in experiment_ids:
        s = stats[exp_id]
        prefix = f"{exp_id}_{output_name}"
        metrics[f"{prefix}_mean"] = s["mean"]
        metrics[f"{prefix}_std"] = s["std"]
        metrics[f"{prefix}_abs_dev_from_1"] = s["abs_dev_from_1"]
        metrics[f"{prefix}_max_dev_from_1"] = s["max_dev_from_1"]
        metrics[f"{prefix}_rms_dev_from_1"] = s["rms_dev_from_1"]
    # 全局
    metrics[f"global_{output_name}_mean"] = global_mean
    metrics[f"global_{output_name}_std"] = global_std
    metrics[f"global_{output_name}_abs_dev_from_1"] = global_abs_dev
    metrics[f"global_{output_name}_max_dev_from_1"] = global_max_dev
    metrics[f"global_{output_name}_rms_dev_from_1"] = global_rms

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": [plot_path],
        "metrics": metrics
    }
