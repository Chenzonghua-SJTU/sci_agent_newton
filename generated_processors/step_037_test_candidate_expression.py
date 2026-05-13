import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    parameters = payload["parameters"]
    experiment_ids = parameters.get("experiment_ids", [])
    expression = parameters.get("expression", "")
    output_name = parameters.get("output_name", "check_constancy_free")
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    if not experiment_ids:
        experiment_ids = list(experiments.keys())

    # 验证所有实验都有 check 序列
    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
        if "check" not in experiments[eid]["series"]:
            raise ValueError(f"Series 'check' not available in experiment {eid}")

    # 收集数据
    results = {}
    fig, axes = plt.subplots(len(experiment_ids), 1, figsize=(8, 4*len(experiment_ids)), squeeze=False)
    if len(experiment_ids) == 1:
        axes = [axes[0][0]]

    for idx, eid in enumerate(experiment_ids):
        exp = experiments[eid]
        config = exp["config"]
        f_ext = config.get("F_ext", 0.0)
        t = np.array(exp["series"]["t"])
        check = np.array(exp["series"]["check"])
        mean_val = np.mean(check)
        std_val = np.std(check)
        deviation = mean_val - f_ext
        abs_deviation = np.abs(deviation)
        results[eid] = {
            "mean": mean_val,
            "std": std_val,
            "max_abs": np.max(np.abs(check)),
            "F_ext": f_ext,
            "deviation": deviation,
            "abs_deviation": abs_deviation,
            "n": len(check)
        }
        # 画图
        ax = axes[idx]
        ax.plot(t, check, label=f"{eid}: check")
        ax.axhline(y=f_ext, color='r', linestyle='--', label=f"F_ext={f_ext}")
        ax.set_xlabel("t")
        ax.set_ylabel("check")
        ax.set_title(f"{eid} (F_ext={f_ext})")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    all_plot_path = os.path.join(output_dir, "check_constancy_free_all.png")
    plt.savefig(all_plot_path)
    plt.close()

    # 构建 observation 和 metrics
    lines = []
    metrics = {}
    for eid in experiment_ids:
        r = results[eid]
        lines.append(
            f"实验 {eid} (F_ext={r['F_ext']}): check均值={r['mean']:.6e}, "
            f"标准差={r['std']:.6e}, 最大绝对值={r['max_abs']:.6e}, "
            f"与F_ext偏差={r['deviation']:.6e} "
            f"(绝对偏差={r['abs_deviation']:.6e})"
        )
        metrics[f"{eid}_check_mean"] = r["mean"]
        metrics[f"{eid}_check_std"] = r["std"]
        metrics[f"{eid}_check_max_abs"] = r["max_abs"]
        metrics[f"{eid}_check_deviation"] = r["deviation"]
        metrics[f"{eid}_check_abs_deviation"] = r["abs_deviation"]
        metrics[f"{eid}_F_ext"] = r["F_ext"]

    observation = (
        f"对自由实验 {experiment_ids} 中的 check 序列 (a_sg*(1+v_sg^2)) 进行恒常性检验。\n"
        + "\n".join(lines)
    )

    # 判断是否接近 0
    all_near_zero = True
    for eid in experiment_ids:
        abs_dev = np.abs(results[eid]["deviation"])
        if abs_dev > 1e-6:  # 可容忍的阈值
            all_near_zero = False
    if all_near_zero:
        observation += "\n所有报告实验的 check 均值与 F_ext 的偏差均小于 1e-6，在数值精度内认为恒等于 F_ext。"
    else:
        observation += "\n部分实验的 check 均值与 F_ext 存在显著偏差（>1e-6），需要进一步分析。"

    figures = [all_plot_path]

    return {
        "observation": observation,
        "derived_series": [],  # 不生成新序列
        "figures": figures,
        "metrics": metrics
    }
