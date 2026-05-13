import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Any, Dict, List

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = payload.get("action", "custom_data_analysis")
    parameters = payload.get("parameters", {})
    experiment_ids = parameters.get("experiment_ids", [])
    if not experiment_ids:
        # 如果没有指定，从 payload["experiments"] 中选择所有 free 实验？但分析目标指定了free实验，而参数中给出了，所以不可能为空。
        raise ValueError("experiment_ids must be provided")
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    # 目标实验需要是 free 类型，但检查一下
    results = {}
    figs = []
    all_check_data = {}

    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
        exp = experiments[eid]
        config = exp.get("config", {})
        series = exp.get("series", {})
        available = exp.get("available_series", [])

        # 检查 check 序列是否存在；如果不存在则尝试用 a_sg 和 v_sg 计算
        if "check" in series:
            check = np.array(series["check"])
        elif "a_sg" in series and "v_sg" in series:
            v = np.array(series["v_sg"])
            a = np.array(series["a_sg"])
            check = a * (1.0 + v**2)
        else:
            raise ValueError(f"Experiment {eid}: neither 'check' series nor both 'a_sg' and 'v_sg' available")

        # 计算统计量
        mean = float(np.mean(check))
        std = float(np.std(check, ddof=1))  # 样本标准差
        max_abs = float(np.max(np.abs(check)))
        # 如果 std 为 0，t 检验无效，此时直接认为不显著偏离 0
        if std == 0:
            t_stat = 0.0
            p_value = 1.0
        else:
            n = len(check)
            t_stat = mean / (std / np.sqrt(n))
            p_value = float(2 * stats.t.sf(np.abs(t_stat), df=n-1))  # 双尾

        results[eid] = {
            "mean": mean,
            "std": std,
            "max_abs": max_abs,
            "t_stat": t_stat,
            "p_value": p_value,
            "n": len(check)
        }
        all_check_data[eid] = {"t": np.array(series.get("t", [])), "check": check}

    # 构建 observation
    obs_lines = [f"对free实验 {experiment_ids} 中的 check 序列 (a_sg*(1+v_sg^2)) 进行统计分析。"]
    for eid in experiment_ids:
        r = results[eid]
        line = (f"实验 {eid}: 均值={r['mean']:.6e}, 标准差={r['std']:.6e}, "
                f"最大绝对值={r['max_abs']:.6e}, 单样本t检验(vs 0): t={r['t_stat']:.3e}, "
                f"p={r['p_value']:.3e}, 样本数={r['n']}")
        obs_lines.append(line)
    # 判断是否显著偏离 0：如果 p < 0.05 则认为显著
    sig_info = []
    for eid in experiment_ids:
        if results[eid]["p_value"] < 0.05:
            sig_info.append(f"{eid} 的 check 均值显著偏离 0 (p<0.05)")
        else:
            sig_info.append(f"{eid} 的 check 均值在统计上不显著偏离 0 (p>=0.05)")
    obs_lines.append("显著性判断: " + "; ".join(sig_info))

    # 生成图像：三个实验的 check vs t 在同一张图上
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['b', 'g', 'r']
    for idx, eid in enumerate(experiment_ids):
        t = all_check_data[eid]["t"]
        check = all_check_data[eid]["check"]
        ax.plot(t, check, color=colors[idx], label=eid, alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('check = a_sg * (1 + v_sg^2)')
    ax.set_title('check vs time for free experiments')
    ax.legend()
    fig_path = os.path.join(output_dir, "free_check_vs_t.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    figs.append(fig_path)

    # 构建 metrics
    metrics = {}
    for eid in experiment_ids:
        r = results[eid]
        prefix = eid + "_check_"
        metrics[prefix + "mean"] = r["mean"]
        metrics[prefix + "std"] = r["std"]
        metrics[prefix + "max_abs"] = r["max_abs"]
        metrics[prefix + "t_stat"] = r["t_stat"]
        metrics[prefix + "p_value"] = r["p_value"]
        metrics[prefix + "n"] = r["n"]
    # 全局
    all_checks = np.concatenate([all_check_data[eid]["check"] for eid in experiment_ids])
    metrics["all_check_global_mean"] = float(np.mean(all_checks))
    metrics["all_check_global_std"] = float(np.std(all_checks, ddof=1))
    metrics["all_check_global_max_abs"] = float(np.max(np.abs(all_checks)))

    return {
        "observation": "\n".join(obs_lines),
        "derived_series": [],
        "figures": figs,
        "metrics": metrics
    }
