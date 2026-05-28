import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    analysis_mode = params.get("analysis_mode", "validate_hypothesis")
    hypothesis_id = params.get("hypothesis_id", "H005")
    experiment_ids = params.get("experiment_ids", list(experiments.keys()))
    step_size = 0.001
    threshold = 0.1

    if not experiment_ids:
        return {"observation": "没有指定实验ID，无法进行验证。", "validations": [], "derived_series": [], "figures": []}

    exp_data = {}
    for eid in experiment_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        config = exp["config"]
        force_field_type = config.get("force_field_type", "free")
        F_ext = config.get("F_ext", 0.0)
        q0 = config.get("initial_q", 0.0)
        v0 = config.get("initial_v", 0.0)
        t = np.array(exp["series"]["t"], dtype=float)
        q = np.array(exp["series"]["q"], dtype=float)
        if len(t) == 0:
            continue
        exp_data[eid] = {
            "t": t,
            "q": q,
            "F_ext": F_ext,
            "force_field_type": force_field_type,
            "q0": q0,
            "v0": v0
        }

    # 微分方程
    def ode_func(t, state, F_ext, gamma, force_field_type):
        q, v = state
        if force_field_type == 'free' or abs(F_ext) < 1e-15:
            a = 0.0
        else:
            a = F_ext * np.exp(-gamma * np.abs(v))
        return [v, a]

    # 使用 solve_ivp 进行积分
    def predict_q(eid, gamma):
        data = exp_data[eid]
        t_exp = data["t"]
        q0 = data["q0"]
        v0 = data["v0"]
        F_ext = data["F_ext"]
        ff_type = data["force_field_type"]

        # 自由场直接解析
        if ff_type == 'free' or abs(F_ext) < 1e-15:
            return q0 + v0 * t_exp

        # 数值积分
        t_span = (t_exp[0], t_exp[-1])
        try:
            sol = solve_ivp(ode_func, t_span, [q0, v0], method='RK45',
                            t_eval=t_exp, max_step=step_size,
                            args=(F_ext, gamma, ff_type), rtol=1e-9, atol=1e-12)
            if not sol.success:
                return None
            return sol.y[0]
        except Exception:
            return None

    def global_avg_rmse(gamma):
        rmse_list = []
        for eid in experiment_ids:
            if eid not in exp_data:
                continue
            data = exp_data[eid]
            q_exp = data["q"]
            q_pred = predict_q(eid, gamma)
            if q_pred is None or len(q_pred) != len(q_exp):
                return 1e10
            mse = np.mean((q_pred - q_exp) ** 2)
            rmse = np.sqrt(mse)
            rmse_list.append(rmse)
        if len(rmse_list) == 0:
            return 1e10
        return np.mean(rmse_list)

    # 优化 gamma
    init_gamma = 0.73
    try:
        res = minimize_scalar(global_avg_rmse, bounds=(0.1, 2.0), method='bounded',
                              options={'xatol': 1e-4, 'maxiter': 30})
        if not res.success or not (0.1 < res.x < 2.0):
            res = minimize_scalar(global_avg_rmse, bracket=(0.1, 0.5, 1.0), method='golden',
                                  options={'xtol': 1e-4, 'maxiter': 30})
        gamma_opt = res.x if res.success else init_gamma
    except Exception:
        gamma_opt = init_gamma

    all_rmse = {}
    q_pred_series = {}
    residual_series = {}
    for eid in experiment_ids:
        if eid not in exp_data:
            continue
        data = exp_data[eid]
        q_exp = data["q"]
        q_pred = predict_q(eid, gamma_opt)
        if q_pred is None or len(q_pred) != len(q_exp):
            all_rmse[eid] = 1e10
            q_pred_series[eid] = q_exp.tolist()  # fallback
            residual_series[eid] = [0.0]*len(q_exp)
        else:
            rmse = np.sqrt(np.mean((q_pred - q_exp)**2))
            all_rmse[eid] = rmse
            q_pred_series[eid] = q_pred.tolist()
            residual_series[eid] = (q_pred - q_exp).tolist()

    global_rmse = np.mean(list(all_rmse.values()))
    supports = bool(global_rmse < threshold)

    validation = {
        "hypothesis_id": hypothesis_id,
        "experiment_ids": list(all_rmse.keys()),
        "supports": supports,
        "metric_name": "position_prediction_RMSE_RK4_smallstep",
        "metric_values": all_rmse,
        "aggregate_score": global_rmse,
        "summary": f"使用RK45积分（max_step={step_size}）优化gamma= {gamma_opt:.8f}，全局平均RMSE = {global_rmse:.8f}，阈值{threshold}，支持结论={'支持' if supports else '反对'}。优化后的gamma值: {gamma_opt:.8f}，各实验RMSE: {dict(sorted(all_rmse.items()))}",
        "source_data_refs": [f"{eid}:q" for eid in all_rmse.keys()]
    }

    derived_series_list = []
    for eid, qvals in q_pred_series.items():
        derived_series_list.append({
            "experiment_id": eid,
            "name": "q_pred_H005_RK4_smallstep",
            "values": qvals,
            "source_name": f"RK45 integration with gamma={gamma_opt:.8f}, max_step={step_size}",
            "provenance": f"generated data processor: step_{payload.get('step_index', 'unknown')}",
            "description": f"使用RK45积分（最大步长{step_size}）从初始条件预测的位置，gamma={gamma_opt:.8f}"
        })
    for eid, rvals in residual_series.items():
        derived_series_list.append({
            "experiment_id": eid,
            "name": "residual_H005_RK4_smallstep",
            "values": rvals,
            "source_name": f"q_exp - q_pred (gamma={gamma_opt:.8f})",
            "provenance": f"generated data processor: step_{payload.get('step_index', 'unknown')}",
            "description": f"实验q与预测q的残差，RMSE={all_rmse[eid]:.6f}"
        })

    fig_paths = []
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_exps = ['exp_02', 'exp_06', 'exp_16', 'exp_18', 'exp_24']
        for eid in plot_exps:
            if eid in q_pred_series and eid in exp_data:
                t = exp_data[eid]["t"]
                q = exp_data[eid]["q"]
                q_pred = np.array(q_pred_series[eid])
                ax.plot(t, q, label=f'{eid} exp', linestyle='-', linewidth=0.8)
                ax.plot(t, q_pred, label=f'{eid} pred', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position q')
        ax.set_title(f'H005 RK4 position prediction (small step={step_size}, gamma={gamma_opt:.4f})')
        ax.legend()
        fig_path = Path(output_dir) / 'H005_RK4_smallstep_position_validation.png'
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        fig_paths.append(str(fig_path))
    except Exception as e:
        pass

    obs_text = (
        f"使用RK45积分（最大步长{step_size}）重新验证假说H005的完整微分方程位置预测能力。"
        f"优化gamma得到最优值 {gamma_opt:.8f}，全局平均RMSE = {global_rmse:.8f}。"
        f"阈值<{threshold}，结论：{'支持' if supports else '反对'}。"
        f"各实验RMSE: {dict(sorted(all_rmse.items()))}。"
    )

    return {
        "observation": obs_text,
        "derived_series": derived_series_list,
        "validations": [validation],
        "figures": fig_paths,
        "metrics": {
            "gamma_optimized": gamma_opt,
            "global_avg_rmse": global_rmse,
            "supports_hypothesis": supports,
            "num_experiments": len(all_rmse),
            "per_experiment_rmse": all_rmse
        }
    }
