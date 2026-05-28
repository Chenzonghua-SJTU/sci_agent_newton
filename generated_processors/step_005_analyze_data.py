import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload.get("action", "analyze_data")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    observations = payload.get("observations", [])
    validations = payload.get("validations", [])
    hypotheses = payload.get("hypotheses", {})
    output_dir = Path(payload.get("output_dir", "."))

    # 从 parameters 获取实验 ID 列表
    exp_ids = params.get("experiment_ids", list(experiments.keys()))
    # 验证所有 exp_ids 都存在
    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")

    # 按 F_ext 分组
    group_F1 = []   # F_ext = 1.0
    group_Fn1 = []  # F_ext = -1.0
    group_F0 = []   # F_ext = 0.0

    for eid in exp_ids:
        exp = experiments[eid]
        config = exp["config"]
        F_ext = config["F_ext"]
        if F_ext == 1.0:
            group_F1.append(eid)
        elif F_ext == -1.0:
            group_Fn1.append(eid)
        elif F_ext == 0.0:
            group_F0.append(eid)
        else:
            # 忽略未知外力
            pass

    # 辅助函数：拼接多个实验的 a 和 v 序列
    def merge_av(group_ids):
        a_all = []
        v_all = []
        for eid in group_ids:
            series = experiments[eid]["series"]
            a_vals = np.array(series["a"])
            v_vals = np.array(series["v"])
            a_all.append(a_vals)
            v_all.append(v_vals)
        a_concat = np.concatenate(a_all)
        v_concat = np.concatenate(v_all)
        return a_concat, v_concat

    # 辅助函数：线性回归
    def linear_fit(a, v):
        X = v.reshape(-1, 1)
        model = LinearRegression(fit_intercept=True)
        model.fit(X, a)
        a_pred = model.predict(X)
        r2 = r2_score(a, a_pred)
        rmse = np.sqrt(mean_squared_error(a, a_pred))
        intercept = model.intercept_
        slope = model.coef_[0]
        return intercept, slope, r2, rmse, a_pred

    # 辅助函数：二次回归
    def quadratic_fit(a, v):
        # 构造特征 [v, v^2]
        X = np.column_stack([v, v**2])
        model = LinearRegression(fit_intercept=True)
        model.fit(X, a)
        a_pred = model.predict(X)
        r2 = r2_score(a, a_pred)
        rmse = np.sqrt(mean_squared_error(a, a_pred))
        intercept = model.intercept_
        coef_v = model.coef_[0]
        coef_v2 = model.coef_[1]
        return intercept, coef_v, coef_v2, r2, rmse, a_pred

    new_observations = []
    figures = []

    # ========== 处理 F_ext=1 合并组 (exp_02, exp_05) ==========
    if group_F1:
        a_comb, v_comb = merge_av(group_F1)
        # 线性回归
        c0_lin, c1_lin, r2_lin, rmse_lin, a_pred_lin = linear_fit(a_comb, v_comb)
        # 二次回归
        c0_quad, c1_quad, c2_quad, r2_quad, rmse_quad, a_pred_quad = quadratic_fit(a_comb, v_comb)

        # 残差（线性）
        residual_lin = a_comb - a_pred_lin
        residual_std = float(np.std(residual_lin))

        # 绘制残差图
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(v_comb, residual_lin, alpha=0.6, label='Linear residual')
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
        ax.set_xlabel('v')
        ax.set_ylabel('Residual (a - a_pred_linear)')
        ax.set_title(f'F_ext=1 merged ({",".join(group_F1)})\nLinear model residual vs v')
        ax.legend()
        fig.tight_layout()
        fig_path = output_dir / f"residual_F1_merged.png"
        fig.savefig(fig_path)
        plt.close(fig)
        figures.append(str(fig_path))

        # Observation 1: 线性回归结果
        obs_lin = {
            "summary": f"合并 {','.join(group_F1)} (F_ext=1) 线性回归: a = {c0_lin:.6f} + {c1_lin:.6f} * v, R²={r2_lin:.6f}, RMSE={rmse_lin:.6f}, 残差标准差={residual_std:.6f}",
            "source_data_refs": [f"{eid}:a" for eid in group_F1] + [f"{eid}:v" for eid in group_F1],
            "metrics": {
                "intercept": c0_lin,
                "slope": c1_lin,
                "r2_linear": r2_lin,
                "rmse_linear": rmse_lin,
                "residual_std": residual_std,
                "F_ext": 1.0,
                "group": f"Fext=1_{'_'.join(group_F1)}"
            }
        }
        new_observations.append(obs_lin)

        # Observation 2: 二次回归结果
        r2_improvement = r2_quad - r2_lin
        obs_quad = {
            "summary": f"合并 {','.join(group_F1)} (F_ext=1) 二次回归: a = {c0_quad:.6f} + {c1_quad:.6f} * v + {c2_quad:.6f} * v², R²={r2_quad:.6f}, RMSE={rmse_quad:.6f}, 相比线性 ΔR²={r2_improvement:.6f}",
            "source_data_refs": [f"{eid}:a" for eid in group_F1] + [f"{eid}:v" for eid in group_F1],
            "metrics": {
                "quad_intercept": c0_quad,
                "quad_coef_v": c1_quad,
                "quad_coef_v2": c2_quad,
                "r2_quadratic": r2_quad,
                "rmse_quadratic": rmse_quad,
                "r2_improvement": r2_improvement,
                "F_ext": 1.0,
                "group": f"Fext=1_{'_'.join(group_F1)}"
            }
        }
        new_observations.append(obs_quad)

        # Observation 3: 截距与F_ext比较
        intercept_diff = abs(c0_lin - 1.0)  # F_ext=1
        obs_intercept = {
            "summary": f"合并 {','.join(group_F1)} (F_ext=1) 线性截距 c0={c0_lin:.6f}, F_ext=1.0, 差值={intercept_diff:.6f}",
            "source_data_refs": [f"{eid}:a" for eid in group_F1] + [f"{eid}:v" for eid in group_F1],
            "metrics": {
                "intercept": c0_lin,
                "F_ext": 1.0,
                "intercept_diff": intercept_diff,
                "group": f"Fext=1_{'_'.join(group_F1)}"
            }
        }
        new_observations.append(obs_intercept)

        # Observation 4: 斜率符号与F_ext
        obs_slope = {
            "summary": f"合并 {','.join(group_F1)} (F_ext=1) 线性斜率 c1={c1_lin:.6f}, 符号与F_ext(正)相关情况: 斜率负值, 绝对值={abs(c1_lin):.6f}",
            "source_data_refs": [f"{eid}:a" for eid in group_F1] + [f"{eid}:v" for eid in group_F1],
            "metrics": {
                "slope": c1_lin,
                "F_ext": 1.0,
                "abs_slope": abs(c1_lin),
                "group": f"Fext=1_{'_'.join(group_F1)}"
            }
        }
        new_observations.append(obs_slope)

        # Observation 5: 二次系数比较（如果显著，此处只报告系数值）
        obs_quad_coef = {
            "summary": f"合并 {','.join(group_F1)} (F_ext=1) 二次项系数 c2={c2_quad:.6f}, 可与其他组比较",
            "source_data_refs": [f"{eid}:a" for eid in group_F1] + [f"{eid}:v" for eid in group_F1],
            "metrics": {
                "quad_coef_v2": c2_quad,
                "F_ext": 1.0,
                "group": f"Fext=1_{'_'.join(group_F1)}"
            }
        }
        new_observations.append(obs_quad_coef)

    # ========== 处理 F_ext=-1 合并组 (exp_03, exp_06) ==========
    if group_Fn1:
        a_comb, v_comb = merge_av(group_Fn1)
        c0_lin, c1_lin, r2_lin, rmse_lin, a_pred_lin = linear_fit(a_comb, v_comb)
        c0_quad, c1_quad, c2_quad, r2_quad, rmse_quad, a_pred_quad = quadratic_fit(a_comb, v_comb)
        residual_lin = a_comb - a_pred_lin
        residual_std = float(np.std(residual_lin))

        # 残差图
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(v_comb, residual_lin, alpha=0.6, label='Linear residual')
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
        ax.set_xlabel('v')
        ax.set_ylabel('Residual (a - a_pred_linear)')
        ax.set_title(f'F_ext=-1 merged ({",".join(group_Fn1)})\nLinear model residual vs v')
        ax.legend()
        fig.tight_layout()
        fig_path = output_dir / f"residual_Fn1_merged.png"
        fig.savefig(fig_path)
        plt.close(fig)
        figures.append(str(fig_path))

        obs_lin = {
            "summary": f"合并 {','.join(group_Fn1)} (F_ext=-1) 线性回归: a = {c0_lin:.6f} + {c1_lin:.6f} * v, R²={r2_lin:.6f}, RMSE={rmse_lin:.6f}, 残差标准差={residual_std:.6f}",
            "source_data_refs": [f"{eid}:a" for eid in group_Fn1] + [f"{eid}:v" for eid in group_Fn1],
            "metrics": {
                "intercept": c0_lin,
                "slope": c1_lin,
                "r2_linear": r2_lin,
                "rmse_linear": rmse_lin,
                "residual_std": residual_std,
                "F_ext": -1.0,
                "group": f"Fext=-1_{'_'.join(group_Fn1)}"
            }
        }
        new_observations.append(obs_lin)

        r2_improvement = r2_quad - r2_lin
        obs_quad = {
            "summary": f"合并 {','.join(group_Fn1)} (F_ext=-1) 二次回归: a = {c0_quad:.6f} + {c1_quad:.6f} * v + {c2_quad:.6f} * v², R²={r2_quad:.6f}, RMSE={rmse_quad:.6f}, 相比线性 ΔR²={r2_improvement:.6f}",
            "source_data_refs": [f"{eid}:a" for eid in group_Fn1] + [f"{eid}:v" for eid in group_Fn1],
            "metrics": {
                "quad_intercept": c0_quad,
                "quad_coef_v": c1_quad,
                "quad_coef_v2": c2_quad,
                "r2_quadratic": r2_quad,
                "rmse_quadratic": rmse_quad,
                "r2_improvement": r2_improvement,
                "F_ext": -1.0,
                "group": f"Fext=-1_{'_'.join(group_Fn1)}"
            }
        }
        new_observations.append(obs_quad)

        intercept_diff = abs(c0_lin - (-1.0))  # F_ext=-1
        obs_intercept = {
            "summary": f"合并 {','.join(group_Fn1)} (F_ext=-1) 线性截距 c0={c0_lin:.6f}, F_ext=-1.0, 差值={intercept_diff:.6f}",
            "source_data_refs": [f"{eid}:a" for eid in group_Fn1] + [f"{eid}:v" for eid in group_Fn1],
            "metrics": {
                "intercept": c0_lin,
                "F_ext": -1.0,
                "intercept_diff": intercept_diff,
                "group": f"Fext=-1_{'_'.join(group_Fn1)}"
            }
        }
        new_observations.append(obs_intercept)

        obs_slope = {
            "summary": f"合并 {','.join(group_Fn1)} (F_ext=-1) 线性斜率 c1={c1_lin:.6f}, 符号与F_ext(负)相关情况: 斜率负值, 绝对值={abs(c1_lin):.6f}",
            "source_data_refs": [f"{eid}:a" for eid in group_Fn1] + [f"{eid}:v" for eid in group_Fn1],
            "metrics": {
                "slope": c1_lin,
                "F_ext": -1.0,
                "abs_slope": abs(c1_lin),
                "group": f"Fext=-1_{'_'.join(group_Fn1)}"
            }
        }
        new_observations.append(obs_slope)

        obs_quad_coef = {
            "summary": f"合并 {','.join(group_Fn1)} (F_ext=-1) 二次项系数 c2={c2_quad:.6f}, 可与其他组比较",
            "source_data_refs": [f"{eid}:a" for eid in group_Fn1] + [f"{eid}:v" for eid in group_Fn1],
            "metrics": {
                "quad_coef_v2": c2_quad,
                "F_ext": -1.0,
                "group": f"Fext=-1_{'_'.join(group_Fn1)}"
            }
        }
        new_observations.append(obs_quad_coef)

    # ========== 处理 F_ext=0 组 (exp_01, exp_04) ==========
    if group_F0:
        a_comb, _ = merge_av(group_F0)
        mean_a = float(np.mean(a_comb))
        std_a = float(np.std(a_comb))
        obs_f0 = {
            "summary": f"合并 {','.join(group_F0)} (F_ext=0) 加速度统计: 均值={mean_a:.6e}, 标准差={std_a:.6e}",
            "source_data_refs": [f"{eid}:a" for eid in group_F0],
            "metrics": {
                "mean_a": mean_a,
                "std_a": std_a,
                "observation_count": len(a_comb),
                "F_ext": 0.0,
                "group": f"Fext=0_{'_'.join(group_F0)}"
            }
        }
        new_observations.append(obs_f0)

    # 构建最终返回
    # 汇总观察结果
    main_summary = (
        f"跨实验合并分析完成。共生成 {len(new_observations)} 条观察记录，"
        f"保存 {len(figures)} 张残差图。"
    )

    return {
        "observation": main_summary,
        "derived_series": [],  # 没有创建新的派生序列
        "observations": new_observations,
        "validations": [],     # 本任务不需要验证
        "figures": figures,
        "metrics": {
            "total_observations": len(new_observations),
            "figure_count": len(figures),
            "group_F1_experiments": group_F1,
            "group_Fn1_experiments": group_Fn1,
            "group_F0_experiments": group_F0
        }
    }
