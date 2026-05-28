import json
import math
import statistics
import itertools
import functools
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats, signal, optimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    # 解析参数
    params = payload.get("parameters", {})
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        raise ValueError("参数中没有指定 experiment_ids")
    
    experiments = payload.get("experiments", {})
    # 将整数 id 转为 exp_XX 格式
    exp_keys = [f"exp_{eid:02d}" for eid in exp_ids]
    
    # 检查所有实验是否都存在
    for ek in exp_keys:
        if ek not in experiments:
            raise ValueError(f"实验 {ek} 在 payload 中不存在")
    
    observations = []   # 将新观测记录放在这里
    
    for ek in exp_keys:
        exp = experiments[ek]
        config = exp.get("config", {})
        series = exp.get("series", {})
        available = exp.get("available_series", [])
        
        if "a" not in available or "v" not in available:
            raise ValueError(f"实验 {ek} 缺少序列 a 或 v")
        
        a = series.get("a")
        v = series.get("v")
        if a is None or v is None:
            raise ValueError(f"实验 {ek} 的序列值为空")
        
        a = np.array(a, dtype=float)
        v = np.array(v, dtype=float)
        n = len(a)
        if n != len(v):
            raise ValueError(f"实验 {ek} 中 a 和 v 长度不一致")
        
        force_type = config.get("force_field_type", "free")
        is_constant = (force_type == "constant")
        is_free = (force_type == "free")
        
        if is_constant:
            # 线性回归
            coeff_lin = np.polyfit(v, a, 1)
            a_pred_lin = np.polyval(coeff_lin, v)
            ss_res_lin = np.sum((a - a_pred_lin)**2)
            ss_tot_lin = np.sum((a - np.mean(a))**2)
            r2_lin = 1 - ss_res_lin / ss_tot_lin if ss_tot_lin > 1e-12 else np.nan
            rmse_lin = np.sqrt(np.mean((a - a_pred_lin)**2))
            
            # 二次回归
            coeff_quad = np.polyfit(v, a, 2)
            a_pred_quad = np.polyval(coeff_quad, v)
            ss_res_quad = np.sum((a - a_pred_quad)**2)
            ss_tot_quad = np.sum((a - np.mean(a))**2)
            r2_quad = 1 - ss_res_quad / ss_tot_quad if ss_tot_quad > 1e-12 else np.nan
            rmse_quad = np.sqrt(np.mean((a - a_pred_quad)**2))
            
            # 记录线性回归观察
            obs_lin = {
                "summary": f"{ek} 线性回归 a = {coeff_lin[1]:.6f} + {coeff_lin[0]:.6f} * v, R²={r2_lin:.4f}, RMSE={rmse_lin:.6f}",
                "source_data_refs": [f"{ek}:a", f"{ek}:v"],
                "metrics": {
                    "regression_type": "linear",
                    "experiment_id": ek,
                    "intercept": coeff_lin[1],
                    "slope": coeff_lin[0],
                    "R2": r2_lin,
                    "RMSE": rmse_lin
                }
            }
            observations.append(obs_lin)
            
            # 记录二次回归观察
            obs_quad = {
                "summary": f"{ek} 二次回归 a = {coeff_quad[2]:.6f} + {coeff_quad[1]:.6f} * v + {coeff_quad[0]:.6f} * v^2, R²={r2_quad:.4f}, RMSE={rmse_quad:.6f}",
                "source_data_refs": [f"{ek}:a", f"{ek}:v"],
                "metrics": {
                    "regression_type": "quadratic",
                    "experiment_id": ek,
                    "intercept": coeff_quad[2],
                    "slope1": coeff_quad[1],
                    "slope2": coeff_quad[0],
                    "R2": r2_quad,
                    "RMSE": rmse_quad
                }
            }
            observations.append(obs_quad)
            
        elif is_free:
            # 对照线性回归
            coeff_lin = np.polyfit(v, a, 1)
            a_pred_lin = np.polyval(coeff_lin, v)
            ss_res_lin = np.sum((a - a_pred_lin)**2)
            ss_tot_lin = np.sum((a - np.mean(a))**2)
            r2_lin = 1 - ss_res_lin / ss_tot_lin if ss_tot_lin > 1e-12 else np.nan
            rmse_lin = np.sqrt(np.mean((a - a_pred_lin)**2))
            
            obs_lin = {
                "summary": f"{ek} (free 对照) 线性回归 a = {coeff_lin[1]:.6f} + {coeff_lin[0]:.6f} * v, R²={r2_lin:.4f}, RMSE={rmse_lin:.6f}",
                "source_data_refs": [f"{ek}:a", f"{ek}:v"],
                "metrics": {
                    "regression_type": "linear_control",
                    "experiment_id": ek,
                    "intercept": coeff_lin[1],
                    "slope": coeff_lin[0],
                    "R2": r2_lin,
                    "RMSE": rmse_lin
                }
            }
            observations.append(obs_lin)
        else:
            raise ValueError(f"实验 {ek} 的 force_field_type 不是 constant 或 free，无法处理")
    
    # 构造总体 observation 文本
    obs_text = f"完成了对 {len(exp_keys)} 个实验的回归分析："
    for ob in observations:
        obs_text += f"\n- {ob['summary']}"
    obs_text += "\n所有回归系数、R²、RMSE 已记录为 OBS 条目。"
    
    # 返回结构
    result = {
        "observation": obs_text,
        "derived_series": [],
        "observations": observations,
        "figures": [],
        "metrics": {
            "total_experiments": len(exp_keys),
            "total_observations": len(observations)
        }
    }
    return result
