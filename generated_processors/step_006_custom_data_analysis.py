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
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 辅助函数：从实验获取 q, t 序列
# ------------------------------------------------------------
def _get_q_t(experiment: dict) -> Tuple[np.ndarray, np.ndarray]:
    series = experiment["series"]
    if "q" not in series:
        raise ValueError(f"Experiment {experiment.get('id','?')} missing required series 'q'")
    if "t" not in series:
        raise ValueError(f"Experiment {experiment.get('id','?')} missing required series 't'")
    q = np.array(series["q"], dtype=float)
    t = np.array(series["t"], dtype=float)
    if len(q) != len(t):
        raise ValueError("q and t length mismatch")
    return q, t

# ------------------------------------------------------------
# 辅助函数：Savitzky-Golay 求导
# ------------------------------------------------------------
def _savgol_derivative(q: np.ndarray, window_length: int, polyorder: int, deriv: int) -> np.ndarray:
    return savgol_filter(q, window_length=window_length, polyorder=polyorder, deriv=deriv, mode='interp')

# ------------------------------------------------------------
# 主处理函数
# ------------------------------------------------------------
def process(payload: dict) -> dict:
    action = payload["action"]
    if action != "custom_data_analysis":
        raise ValueError(f"Expected action 'custom_data_analysis', got '{action}'")
    
    parameters = payload["parameters"]
    experiment_ids = parameters.get("experiment_ids", [])
    if not experiment_ids:
        raise ValueError("Missing 'experiment_ids' in parameters")
    
    # 验证需要的实验存在
    experiments = payload["experiments"]
    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
    
    output_dir = Path(payload["output_dir"])
    derived_series_list = []
    metrics = {}
    figures = []
    observations = []
    
    # 固定 SG 参数
    window_length = 7
    polyorder = 2
    
    for eid in experiment_ids:
        exp = experiments[eid]
        q, t = _get_q_t(exp)
        config = exp.get("config", {})
        F_ext = config.get("F_ext", 0.0)
        
        # 计算 v_sg 和 a_sg
        v_sg = _savgol_derivative(q, window_length, polyorder, deriv=1)
        a_sg = _savgol_derivative(q, window_length, polyorder, deriv=2)
        
        # 线性拟合 a = intercept + slope * v
        v = v_sg
        a = a_sg
        # 使用 numpy polyfit
        coeffs = np.polyfit(v, a, 1)   # [slope, intercept]
        slope = coeffs[0]
        intercept = coeffs[1]
        a_pred = np.polyval(coeffs, v)
        residuals = a - a_pred
        rmse = np.sqrt(np.mean(residuals**2))
        r2 = 1.0 - np.sum(residuals**2) / np.sum((a - np.mean(a))**2)
        
        # 加速度统计
        a_mean = float(np.mean(a))
        a_std = float(np.std(a, ddof=0))
        
        # 计算 a + (-slope)*v = a - slope*v
        a_minus_slope_v = a - slope * v
        amv_mean = float(np.mean(a_minus_slope_v))
        amv_std = float(np.std(a_minus_slope_v, ddof=0))
        
        # 检查 a_minus_slope_v 是否接近常数（std 相对均值很小）并且接近 F_ext
        is_constant_flag = (amv_std < 1e-10) or (amv_std / (abs(amv_mean)+1e-20) < 0.01)
        diff_from_F_ext = abs(amv_mean - F_ext)
        
        # 构建实验 metrics 前缀
        obs_parts = []
        obs_parts.append(f"实验 {eid}:")
        obs_parts.append(f"SG滤波(window={window_length}, polyorder={polyorder}) 从 q 估计 v_sg 和 a_sg")
        obs_parts.append(f"v_sg: min={np.min(v_sg):.6f}, max={np.max(v_sg):.6f}, mean={np.mean(v_sg):.6f}, std={np.std(v_sg, ddof=0):.6f}")
        obs_parts.append(f"a_sg: min={np.min(a_sg):.6f}, max={np.max(a_sg):.6f}, mean={a_mean:.6f}, std={a_std:.6f}")
        obs_parts.append(f"线性拟合 a = {intercept:.6f} + ({slope:.6f})*v  | RMSE={rmse:.6f}, R²={r2:.6f}")
        obs_parts.append(f"a + (-{slope:.6f})*v 均值={amv_mean:.6f}, 标准差={amv_std:.6f}, 与 F_ext={F_ext} 的偏差={diff_from_F_ext:.6f}")
        if is_constant_flag:
            obs_parts.append("该残差序列接近常数")
        else:
            obs_parts.append("该残差序列不接近常数")
        
        observations.append("\n".join(obs_parts))
        
        # 保存 metrics
        prefix = f"{eid}_"
        metrics[prefix+"v_sg_min"] = float(np.min(v_sg))
        metrics[prefix+"v_sg_max"] = float(np.max(v_sg))
        metrics[prefix+"v_sg_mean"] = float(np.mean(v_sg))
        metrics[prefix+"v_sg_std"] = float(np.std(v_sg, ddof=0))
        metrics[prefix+"a_sg_min"] = float(np.min(a_sg))
        metrics[prefix+"a_sg_max"] = float(np.max(a_sg))
        metrics[prefix+"a_sg_mean"] = a_mean
        metrics[prefix+"a_sg_std"] = a_std
        metrics[prefix+"linear_slope"] = float(slope)
        metrics[prefix+"linear_intercept"] = float(intercept)
        metrics[prefix+"linear_rmse"] = float(rmse)
        metrics[prefix+"linear_r2"] = float(r2)
        metrics[prefix+"amv_mean"] = amv_mean
        metrics[prefix+"amv_std"] = amv_std
        metrics[prefix+"amv_diff_from_F_ext"] = diff_from_F_ext
        metrics[prefix+"amv_is_constant"] = is_constant_flag
        
        # 派生序列
        derived_series_list.append({
            "experiment_id": eid,
            "name": "v_sg",
            "values": v_sg.tolist(),
            "source_name": f"Savitzky-Golay deriv=1 (window={window_length}, polyorder={polyorder}) on q",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"速度估计，对 {eid} 的 q 做 SG 滤波一阶导数"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": "a_sg",
            "values": a_sg.tolist(),
            "source_name": f"Savitzky-Golay deriv=2 (window={window_length}, polyorder={polyorder}) on q",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"加速度估计，对 {eid} 的 q 做 SG 滤波二阶导数"
        })
        
        # 绘制 a vs v 散点图及拟合线
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(v, a, label="Data", alpha=0.7, s=20)
        # 绘制拟合线
        v_sorted = np.sort(v)
        a_fit_line = np.polyval(coeffs, v_sorted)
        ax.plot(v_sorted, a_fit_line, 'r-', label=f"Fit: a={intercept:.4f}+{slope:.4f}*v")
        ax.set_xlabel("v (velocity)")
        ax.set_ylabel("a (acceleration)")
        ax.set_title(f"Experiment {eid}: a vs v (SG smoothed)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # 添加统计文本
        textstr = f"RMSE={rmse:.4f}\nR²={r2:.4f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        fig.tight_layout()
        fig_path = output_dir / f"a_vs_v_fit_{eid}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        figures.append(str(fig_path))
        
        # 也可画 a_minus_slope_v 随时间变化？但未明确要求，省略
    
    observation = "\n\n".join(observations)
    result = {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics
    }
    return result
