import json
import math
import numpy as np
from collections import OrderedDict

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    hypotheses = payload.get("hypotheses", {})
    observations_existing = payload.get("observations", [])
    validations_existing = payload.get("validations", [])
    output_dir = payload.get("output_dir", ".")

    # 参数检查
    if action != "analyze_data":
        raise ValueError(f"Unexpected action: {action}")
    if params.get("analysis_mode") != "validate_hypothesis":
        raise ValueError("analysis_mode must be 'validate_hypothesis'")
    if params.get("hypothesis_id") != "H006":
        raise ValueError("This script is for H006 only")

    hypothesis_id = "H006"
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        raise ValueError("No experiment_ids provided")

    gamma = 0.729934  # 全局最优，来自之前加速度拟合
    threshold = 0.1

    per_experiment_rmse = {}
    all_rmses = []
    free_rmses = []
    q_pred_series = {}  # experiment_id -> list of predicted q

    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found")
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        t = np.array(series["t"])
        q_meas = np.array(series["q"])
        dt = config["dt"]
        force_type = config["force_field_type"]
        F_ext = config["F_ext"]

        # 初始条件
        q0 = config.get("initial_q", q_meas[0])
        v0 = config.get("initial_v", 0.0)

        n = len(t)
        q_pred = np.zeros(n)
        v = v0
        q_pred[0] = q0

        if force_type == "free":
            # free field: a=0, constant velocity
            for i in range(1, n):
                q_pred[i] = q_pred[i-1] + dt * v
        elif force_type == "constant":
            for i in range(1, n):
                a = F_ext * math.exp(-gamma * abs(v))
                q_pred[i] = q_pred[i-1] + dt * v
                v = v + dt * a
        else:
            raise ValueError(f"Unknown force_field_type for {eid}: {force_type}")

        rmse = math.sqrt(np.mean((q_pred - q_meas)**2))
        per_experiment_rmse[eid] = round(rmse, 10)
        all_rmses.append(rmse)
        if force_type == "free":
            free_rmses.append(rmse)

        q_pred_series[eid] = q_pred.tolist()

    avg_rmse = sum(all_rmses) / len(all_rmses) if all_rmses else 0.0
    supports = avg_rmse < threshold

    # 计算自由场平均RMSE（可选，但任务要求同时计算free场预测误差）
    free_avg_rmse = sum(free_rmses) / len(free_rmses) if free_rmses else None

    # 构建 validations
    validation_entry = {
        "hypothesis_id": hypothesis_id,
        "experiment_ids": exp_ids,
        "supports": supports,
        "metric_name": "position_prediction_RMSE",
        "metric_values": per_experiment_rmse,
        "aggregate_score": avg_rmse,
        "summary": (
            f"验证假说H006: 用欧拉法积分 a = F_ext*exp(-gamma*|v|) (gamma={gamma}) 预测位置，"
            f"所有 {len(exp_ids)} 个实验的q预测RMSE均值为 {avg_rmse:.6f}。"
            f" {'支持' if supports else '不支持'}假说（阈值<{threshold}）。"
            f" 自由场实验({len(free_rmses)}个)平均RMSE = {free_avg_rmse:.6e}"
        ),
        "source_data_refs": [f"{eid}:q" for eid in exp_ids] + [f"{eid}:t" for eid in exp_ids]
    }

    # 构建 observation 文本
    obs_text = (
        f"假说H006验证完成: 对所有{len(exp_ids)}个实验进行位置预测积分（欧拉法, gamma={gamma}）。"
        f"平均RMSE={avg_rmse:.6f}，支持={supports}。"
        f"各实验RMSE: {json.dumps({k: round(v,6) for k,v in per_experiment_rmse.items()})}"
    )

    # 构建 derived_series (可选，便于后续复用)
    derived_series_list = []
    for eid, pred_vals in q_pred_series.items():
        derived_series_list.append({
            "experiment_id": eid,
            "name": f"q_pred_H006_{eid}",
            "values": pred_vals,
            "source_name": f"Euler integration: q_pred from H006 model, gamma={gamma}",
            "provenance": "generated data processor: step_H006_validate_position",
            "description": f"预测位置序列（假说H006）"
        })

    # 构建 metrics 用于 observation（可选的）
    metrics_info = {
        "average_rmse": avg_rmse,
        "support_threshold": threshold,
        "hypothesis_supported": supports,
        "free_field_avg_rmse": free_avg_rmse,
        "experiments_processed": len(exp_ids)
    }

    return {
        "observation": obs_text,
        "derived_series": derived_series_list,
        "observations": [
            {
                "summary": obs_text,
                "source_data_refs": [f"{eid}:q" for eid in exp_ids],
                "metrics": metrics_info
            }
        ],
        "validations": [validation_entry],
        "figures": [],
        "metrics": metrics_info
    }
