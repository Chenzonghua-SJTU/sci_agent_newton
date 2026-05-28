import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings

def process(payload: dict) -> dict:
    parameters = payload["parameters"]
    experiments = payload["experiments"]
    experiment_ids = parameters.get("experiment_ids", list(experiments.keys()))
    output_dir = Path(payload["output_dir"])

    # 验证所有实验存在
    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")

    # 定义运动方程加速度
    def acceleration(v: float, F_ext: float, gamma: float) -> float:
        return F_ext * math.exp(-gamma * abs(v))

    # RK4 积分函数，返回在实验时间点上的预测位置
    def integrate_rk4(t_exp: np.ndarray, q0: float, v0: float, F_ext: float, gamma: float, h: float = 0.01) -> np.ndarray:
        t_min, t_max = t_exp[0], t_exp[-1]
        n_steps = int((t_max - t_min) / h) + 1
        t_fine = np.linspace(t_min, t_max, n_steps)
        q_fine = np.empty(n_steps)
        q_fine[0] = q0
        v = v0
        dt = h
        for i in range(n_steps - 1):
            # RK4 for coupled system: dq/dt = v, dv/dt = a(v)
            t_curr = t_fine[i]
            # k1
            k1_q = v
            k1_v = acceleration(v, F_ext, gamma)
            # k2
            v_mid = v + 0.5 * dt * k1_v
            k2_q = v_mid
            k2_v = acceleration(v_mid, F_ext, gamma)
            # k3
            v_mid2 = v + 0.5 * dt * k2_v
            k3_q = v_mid2
            k3_v = acceleration(v_mid2, F_ext, gamma)
            # k4
            v_end = v + dt * k3_v
            k4_q = v_end
            k4_v = acceleration(v_end, F_ext, gamma)
            # update
            q_fine[i+1] = q_fine[i] + (dt / 6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
            v = v + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            # 处理自由场时v不变，但为了数值稳定不特殊处理
        # 插值到实验时间点
        interp_func = interp1d(t_fine, q_fine, kind='linear', bounds_error=False, fill_value='extrapolate')
        q_pred = interp_func(t_exp)
        return q_pred

    # 目标函数：所有实验的平均RMSE
    def objective(gamma: float) -> float:
        if gamma <= 0:
            return 1e12
        rmse_list = []
        for eid in experiment_ids:
            exp = experiments[eid]
            config = exp["config"]
            F_ext = config["F_ext"]
            q0 = config["initial_q"]
            v0 = config["initial_v"]
            t = np.array(exp["series"]["t"])
            q = np.array(exp["series"]["q"])
            # 自由场：F_ext==0，此时加速度始终为0，RK4应精确保持v不变
            q_pred = integrate_rk4(t, q0, v0, F_ext, gamma, h=0.01)
            rmse = math.sqrt(np.mean((q - q_pred)**2))
            rmse_list.append(rmse)
        return np.mean(rmse_list)

    # 优化 gamma
    result = optimize.minimize_scalar(objective, bounds=(0.001, 20.0), method='bounded')
    gamma_opt = result.x
    avg_rmse = result.fun

    # 计算每个实验的RMSE和预测序列
    per_experiment_rmse = {}
    derived_series_list = []
    for eid in experiment_ids:
        exp = experiments[eid]
        config = exp["config"]
        F_ext = config["F_ext"]
        q0 = config["initial_q"]
        v0 = config["initial_v"]
        t = np.array(exp["series"]["t"])
        q = np.array(exp["series"]["q"])
        q_pred = integrate_rk4(t, q0, v0, F_ext, gamma_opt, h=0.01)
        rmse = math.sqrt(np.mean((q - q_pred)**2))
        per_experiment_rmse[eid] = rmse
        # 生成预测序列（可选）
        derived_series_list.append({
            "experiment_id": eid,
            "name": f"q_pred_H004_RK4_{eid}",
            "values": q_pred.tolist(),
            "source_name": f"RK4 integration of a=F_ext*exp(-gamma*|v|), gamma={gamma_opt:.6f}, h=0.01",
            "provenance": "generated data processor: step_040_analyze_data",
            "description": f"Predicted position using optimized gamma={gamma_opt:.6f}"
        })

    # 构建 validations
    supports = avg_rmse < 0.1
    validations = [{
        "hypothesis_id": "H004",
        "experiment_ids": experiment_ids,
        "supports": supports,
        "metric_name": "position_prediction_RMSE_RK4",
        "metric_values": per_experiment_rmse,
        "aggregate_score": avg_rmse,
        "summary": f"验证假说H004位置预测（RK4积分, h=0.01）: 优化gamma={gamma_opt:.6f}, 平均RMSE={avg_rmse:.6f}。阈值0.1 -> {'支持' if supports else '不支持'}。",
        "source_data_refs": [f"{eid}:q" for eid in experiment_ids] + [f"{eid}:t" for eid in experiment_ids]
    }]

    # 生成图像（可选）
    fig_paths = []
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        # 选取几个代表性实验绘图
        sample_ids = ['exp_02', 'exp_06', 'exp_16', 'exp_24']
        for eid in sample_ids:
            if eid in experiment_ids:
                exp = experiments[eid]
                t = exp["series"]["t"]
                q = exp["series"]["q"]
                q_pred = None
                for ds in derived_series_list:
                    if ds["experiment_id"] == eid and ds["name"].startswith("q_pred_H004_RK4"):
                        q_pred = ds["values"]
                        break
                if q_pred:
                    ax.plot(t, q, label=f'{eid} true', linestyle='-')
                    ax.plot(t, q_pred, label=f'{eid} pred (gamma={gamma_opt:.3f})', linestyle='--')
        ax.set_xlabel('t')
        ax.set_ylabel('q')
        ax.set_title(f'H004 position prediction (RK4, gamma={gamma_opt:.4f})')
        ax.legend()
        fig_path = str(output_dir / "H004_RK4_position_validation.png")
        plt.savefig(fig_path, dpi=100)
        plt.close()
        fig_paths.append(fig_path)
    except Exception:
        pass

    observation = (
        f"假说H004位置预测验证（RK4积分, 步长0.01）：优化gamma={gamma_opt:.6f}，"
        f"所有{len(experiment_ids)}个实验的平均RMSE={avg_rmse:.6f}。"
        f"各实验RMSE：{json.dumps(per_experiment_rmse, indent=2, ensure_ascii=False)}。"
        f"平均RMSE<0.1？{'是，支持假说' if supports else '否，不支持假说'}"
    )

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "observations": [],
        "validations": validations,
        "figures": fig_paths,
        "metrics": {
            "gamma_optimized": gamma_opt,
            "average_RMSE": avg_rmse,
            "per_experiment_RMSE": per_experiment_rmse
        }
    }
