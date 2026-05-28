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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # 提取参数
    analysis_mode = params.get("analysis_mode", "validate_hypothesis")
    hypothesis_id = params.get("hypothesis_id", "H005")
    experiment_ids = params.get("experiment_ids", list(experiments.keys()))
    step_size = 0.001  # 固定积分步长
    threshold = 0.1

    # 检查是否有实验
    if not experiment_ids:
        return {"observation": "没有指定实验ID，无法进行验证。", "validations": [], "derived_series": [], "figures": []}

    # 加载每个实验的数据
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
        dt = config.get("dt", 0.1)
        # 检查噪声：固定为0
        noise_std = config.get("noise_std", 0.0)
        if noise_std != 0.0:
            # 按指令噪声固定为0，忽略非零设置
            pass
        exp_data[eid] = {
            "t": t,
            "q": q,
            "F_ext": F_ext,
            "force_field_type": force_field_type,
            "q0": q0,
            "v0": v0,
            "dt": dt
        }

    # 定义微分方程
    def ode_func(state, F_ext, gamma, force_field_type):
        q, v = state
        if force_field_type == 'free' or abs(F_ext) < 1e-15:  # free场或F_ext≈0
            a = 0.0
        else:
            a = F_ext * np.exp(-gamma * np.abs(v))
        return np.array([v, a])

    # RK4步进函数
    def rk4_step(state, t, h, F_ext, gamma, force_field_type):
        k1 = ode_func(state, F_ext, gamma, force_field_type)
        k2 = ode_func(state + 0.5 * h * k1, F_ext, gamma, force_field_type)
        k3 = ode_func(state + 0.5 * h * k2, F_ext, gamma, force_field_type)
        k4 = ode_func(state + h * k3, F_ext, gamma, force_field_type)
        return state + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # 单个实验预测函数
    def predict_q(eid, gamma):
        data = exp_data[eid]
        t_exp = data["t"]
        q0 = data["q0"]
        v0 = data["v0"]
        F_ext = data["F_ext"]
        ff_type = data["force_field_type"]

        # 初始化
        t_curr = 0.0
        state = np.array([q0, v0], dtype=float)
        q_pred_list = []
        # 对于每个实验时间点，积分到该点
        for t_target in t_exp:
            # 逐步推进，最后一步自适应
            while t_curr + step_size < t_target - 1e-12:
                state = rk4_step(state, t_curr, step_size, F_ext, gamma, ff_type)
                t_curr += step_size
            # 最后一步自适应
            h_last = t_target - t_curr
            if h_last > 1e-14:
                state = rk4_step(state, t_curr, h_last, F_ext, gamma, ff_type)
                t_curr = t_target
            q_pred_list.append(state[0].item())
        return np.array(q_pred_list, dtype=float)

    # 计算全局平均RMSE
    def global_avg_rmse(gamma):
        rmse_list = []
        for eid in experiment_ids:
            if eid not in exp_data:
                continue
            data = exp_data[eid]
            q_exp = data["q"]
            try:
                q_pred = predict_q(eid, gamma)
            except Exception:
                return 1e10
            # 计算RMSE
            mse = np.mean((q_pred - q_exp) ** 2)
            rmse = np.sqrt(mse)
            rmse_list.append(rmse)
        if len(rmse_list) == 0:
            return 1e10
        return np.mean(rmse_list)

    # 优化gamma
    print("开始优化gamma...")
    # 先计算一个粗略的初始gamma（从已有的H005加速度拟合结果0.729934）
    init_gamma = 0.73
    # 使用有界优化
    res = minimize_scalar(global_avg_rmse, bounds=(0.01, 5.0), method='bounded', options={'xatol': 1e-6, 'maxiter': 200})
    if not res.success:
        # 尝试备选方法
        res = minimize_scalar(global_avg_rmse, bracket=(0.1, 0.5, 1.0), method='golden', options={'xtol': 1e-6, 'maxiter': 200})
    gamma_opt = res.x if res.success else init_gamma
    print(f"优化后gamma = {gamma_opt:.8f}")

    # 使用最优gamma计算每个实验的RMSE和q_pred序列
    all_rmse = {}
    q_pred_series = {}
    residual_series = {}
    for eid in experiment_ids:
        if eid not in exp_data:
            continue
        data = exp_data[eid]
        q_exp = data["q"]
        q_pred = predict_q(eid, gamma_opt)
        rmse = np.sqrt(np.mean((q_pred - q_exp)**2))
        all_rmse[eid] = rmse
        q_pred_series[eid] = q_pred.tolist()
        residual_series[eid] = (q_pred - q_exp).tolist()

    # 全局平均RMSE
    global_rmse = np.mean(list(all_rmse.values()))
    supports = bool(global_rmse < threshold)

    # 构建validations
    validation = {
        "hypothesis_id": hypothesis_id,
        "experiment_ids": list(all_rmse.keys()),
        "supports": supports,
        "metric_name": "position_prediction_RMSE_RK4_smallstep",
        "metric_values": all_rmse,
        "aggregate_score": global_rmse,
        "summary": f"使用RK4积分（步长{step_size}）优化gamma= {gamma_opt:.8f}，全局平均RMSE = {global_rmse:.8f}，阈值{threshold}，支持结论={'支持' if supports else '反对'}。优化后的gamma值: {gamma_opt:.8f}，各实验RMSE: {dict(sorted(all_rmse.items()))}",
        "source_data_refs": [f"{eid}:q" for eid in all_rmse.keys()]
    }

    # 构造derived_series
    derived_series_list = []
    # 根据optional_series参数，只生成"q_pred_H005_RK4_smallstep"
    for eid, qvals in q_pred_series.items():
        derived_series_list.append({
            "experiment_id": eid,
            "name": "q_pred_H005_RK4_smallstep",
            "values": qvals,
            "source_name": f"RK4 integration with gamma={gamma_opt:.8f}, step={step_size}",
            "provenance": f"generated data processor: step_{payload.get('step_index', 'unknown')}",
            "description": f"使用RK4积分（步长{step_size}）从初始条件预测的位置，gamma={gamma_opt:.8f}"
        })
    # 同时生成残差序列（可选，但有助于分析）
    for eid, rvals in residual_series.items():
        derived_series_list.append({
            "experiment_id": eid,
            "name": "residual_H005_RK4_smallstep",
            "values": rvals,
            "source_name": f"q_exp - q_pred (gamma={gamma_opt:.8f})",
            "provenance": f"generated data processor: step_{payload.get('step_index', 'unknown')}",
            "description": f"实验q与预测q的残差，RMSE={all_rmse[eid]:.6f}"
        })

    # 生成图像（可选）
    fig_paths = []
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        # 选取几个代表性实验画图
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
        print(f"绘图失败: {e}")

    # 构造observation
    obs_text = (
        f"使用RK4积分（步长{step_size}）重新验证假说H005的完整微分方程位置预测能力。"
        f"优化gamma得到最优值 {gamma_opt:.8f}，全局平均RMSE = {global_rmse:.8f}。"
        f"阈值<{threshold}，结论：{'支持' if supports else '反对'}。"
        f"各实验RMSE: {dict(sorted(all_rmse.items()))}。"
    )

    # 返回结果
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
