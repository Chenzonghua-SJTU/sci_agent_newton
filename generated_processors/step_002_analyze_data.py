import numpy as np
from typing import Any, Dict, List

def process(payload: dict) -> dict:
    experiments = payload["experiments"]
    target_ids = payload["parameters"].get("experiment_ids", list(experiments.keys()))
    output_dir = payload.get("output_dir", ".")

    derived_series = []
    observations = []
    all_mean_acc = []
    all_init_v = []
    all_final_v = []
    all_std_acc = []

    for eid in target_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload.")
        exp = experiments[eid]
        t = np.array(exp["series"]["t"], dtype=float)
        q = np.array(exp["series"]["q"], dtype=float)
        dt = t[1] - t[0]  # uniform spacing

        # 检查是否已有同名派生序列
        exist_series = exp.get("available_series", [])
        if "v" in exist_series:
            raise ValueError(f"Experiment {eid} already has a series named 'v'. Use a different name or skip.")
        if "a" in exist_series:
            raise ValueError(f"Experiment {eid} already has a series named 'a'. Use a different name or skip.")

        # 一阶中心差分速度
        v = np.gradient(q, t, edge_order=2)
        # 二阶差分加速度
        a = np.gradient(v, t, edge_order=2)

        # 派生序列
        derived_series.append({
            "experiment_id": eid,
            "name": "v",
            "values": v.tolist(),
            "source_name": "np.gradient(q, t, edge_order=2)",
            "provenance": "generated data processor: maintain_ledger (step v)",
            "description": "first central difference velocity"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "a",
            "values": a.tolist(),
            "source_name": "np.gradient(v, t, edge_order=2)",
            "provenance": "generated data processor: maintain_ledger (step a)",
            "description": "second central difference acceleration"
        })

        # 内部点均值加速度：去掉首尾各一个点以减小边界误差
        interior_a = a[1:-1]
        mean_acc = float(np.mean(interior_a))
        std_acc = float(np.std(interior_a))  # 加速度标准差（内部点）

        # 初始速度（前两点差分）
        init_v = float((q[1] - q[0]) / dt)
        # 最终速度（后两点差分）
        final_v = float((q[-1] - q[-2]) / dt)

        all_mean_acc.append(mean_acc)
        all_init_v.append(init_v)
        all_final_v.append(final_v)
        all_std_acc.append(std_acc)

        # 单条 observation
        obs = {
            "summary": f"实验 {eid}：平均加速度 = {mean_acc:.6f}, 初始速度 = {init_v:.6f}, 最终速度 = {final_v:.6f}, 加速度标准差（内部点）= {std_acc:.6f}",
            "source_data_refs": [f"{eid}:q", f"{eid}:t"],
            "metrics": {
                "experiment_id": eid,
                "mean_acceleration": mean_acc,
                "initial_velocity": init_v,
                "final_velocity": final_v,
                "acceleration_std": std_acc
            }
        }
        observations.append(obs)

    # 构造总的 observation 字符串
    summary_lines = []
    for eid, macc, iv, fv, sacc in zip(target_ids, all_mean_acc, all_init_v, all_final_v, all_std_acc):
        summary_lines.append(f"{eid}: mean_a={macc:.6f}, init_v={iv:.6f}, final_v={fv:.6f}, acc_std={sacc:.6f}")
    observation_text = "已完成6个实验的一阶中心差分速度v、二阶差分加速度a的定义，并计算各实验的平均加速度（内部点均值）、初始速度、最终速度及加速度标准差。\n" + "\n".join(summary_lines)

    # 全局 metrics
    metrics = {
        "experiments_processed": len(target_ids),
        "derived_series_count": len(derived_series),
        "observation_count": len(observations)
    }

    return {
        "observation": observation_text,
        "derived_series": derived_series,
        "observations": observations,
        "validations": [],
        "figures": [],
        "metrics": metrics
    }
