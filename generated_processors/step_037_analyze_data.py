import json
import math
import numpy as np
from typing import Any, Dict, List, Tuple

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    parameters = payload["parameters"]
    experiment_ids = parameters["experiment_ids"]
    gamma = 0.729934
    support_threshold = 0.1

    exp_dict = payload["experiments"]
    per_exp_rmse = {}
    all_residual_series = []
    all_q_pred_series = []

    for eid in experiment_ids:
        if eid not in exp_dict:
            raise ValueError(f"Experiment {eid} not found in payload.")
        exp = exp_dict[eid]
        config = exp["config"]
        series = exp["series"]

        F_ext = config["F_ext"]
        force_field_type = config["force_field_type"]
        dt = config["dt"]
        initial_q = config.get("initial_q", 0.0)
        initial_v = config.get("initial_v", 0.0)

        t = np.array(series["t"])
        q_exp = np.array(series["q"])
        n = len(t)

        # RK4 integration
        state = np.array([initial_q, initial_v])
        q_pred_list = [state[0]]

        def deriv(state_vec):
            q, v = state_vec
            if force_field_type == "free":
                a = 0.0
            else:
                a = F_ext * math.exp(-gamma * abs(v))
            return np.array([v, a])

        for i in range(n - 1):
            k1 = deriv(state)
            k2 = deriv(state + 0.5 * dt * k1)
            k3 = deriv(state + 0.5 * dt * k2)
            k4 = deriv(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            q_pred_list.append(state[0])

        q_pred = np.array(q_pred_list)
        residual = q_exp - q_pred
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        per_exp_rmse[eid] = rmse

        all_residual_series.append({
            "experiment_id": eid,
            "name": f"residual_H005_RK4_{eid}",
            "values": residual.tolist(),
            "source_name": "q_exp - q_pred (RK4, gamma=0.729934)",
            "provenance": "generated data processor: step_analyze_H005_RK4",
            "description": "Residual of H005 position prediction using RK4 integration"
        })
        all_q_pred_series.append({
            "experiment_id": eid,
            "name": f"q_pred_H005_RK4_{eid}",
            "values": q_pred.tolist(),
            "source_name": "RK4 integration, a = 0 if free else F_ext*exp(-gamma*|v|)",
            "provenance": "generated data processor: step_analyze_H005_RK4",
            "description": "Predicted position using H005 with RK4, gamma=0.729934"
        })

    avg_rmse = float(np.mean(list(per_exp_rmse.values())))
    supports = avg_rmse < support_threshold

    validation = {
        "hypothesis_id": "H005",
        "experiment_ids": experiment_ids,
        "supports": supports,
        "metric_name": "position_prediction_RMSE_RK4",
        "metric_values": per_exp_rmse,
        "aggregate_score": avg_rmse,
        "summary": (f"Validated H005 via RK4 position prediction. "
                    f"Average RMSE = {avg_rmse:.6f}, threshold = {support_threshold}. "
                    f"Hypothesis supported: {supports}."),
        "source_data_refs": [f"{eid}:q" for eid in experiment_ids] +
                            [f"{eid}:t" for eid in experiment_ids]
    }

    output = {
        "observation": (f"假说H005位置预测验证完成（RK4积分, gamma={gamma}）。"
                        f"全局平均RMSE = {avg_rmse:.6f}。"
                        f"根据阈值{support_threshold}，假说{'支持' if supports else '不支持'}。"
                        f"各实验RMSE：{json.dumps({k: round(v,8) for k,v in per_exp_rmse.items()})}。"),
        "derived_series": all_q_pred_series + all_residual_series,
        "metrics": {
            "average_rmse": avg_rmse,
            "support_threshold": support_threshold,
            "hypothesis_supported": supports,
            "experiments_processed": len(experiment_ids)
        },
        "validations": [validation],
        "observations": [],
        "figures": []
    }
    return output
