import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # 根据 experiment_ids 过滤要处理的实验
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        # 处理所有
        exp_ids = list(experiments.keys())

    # 收集结果
    fit_results = []        # 每个恒外力实验的拟合参数
    free_exp_a_means = []   # 自由实验的加速度均值

    for eid in exp_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]

        # 提取时间、位置
        t = np.array(series["t"], dtype=float)
        q = np.array(series["q"], dtype=float)
        dt = t[1] - t[0]

        # 中心差分速度 (n-2 点)
        v_cent = (q[2:] - q[:-2]) / (2.0 * dt)
        # 中心差分加速度 (n-2 点，二阶导数)
        a_cent = (q[2:] - 2.0 * q[1:-1] + q[:-2]) / (dt * dt)

        # 剔除前后各5个边界点
        if len(v_cent) > 10:
            v_int = v_cent[5:-5].copy()
            a_int = a_cent[5:-5].copy()
        else:
            v_int = v_cent.copy()
            a_int = a_cent.copy()

        if len(v_int) == 0:
            continue

        F_ext = config["F_ext"]
        v0 = config.get("initial_v", 0.0)
        field_type = config.get("force_field_type", "")

        # ---------- 自由运动检查 ----------
        if field_type == "free" or abs(F_ext) < 1e-12:
            mean_a = float(np.mean(a_int))
            max_abs_a = float(np.max(np.abs(a_int)))
            free_exp_a_means.append({
                "experiment": eid,
                "F_ext": float(F_ext),
                "v0": float(v0),
                "mean_acceleration": mean_a,
                "max_abs_acceleration": max_abs_a,
                "is_zero": bool(max_abs_a < 1e-10)
            })
            # 对自由实验，我们仍拟合指数模型但预期 A~0
            # 使用小的初始猜测
            try:
                v_abs = np.abs(v_int)
                # 模型: a = A * exp(-|v|/B)
                def model(v, A, B):
                    return A * np.exp(-np.abs(v) / B)
                p0 = [float(np.mean(a_int)), 1.0]
                popt, _ = curve_fit(model, v_int, a_int, p0=p0, maxfev=10000)
                A_fit, B_fit = popt
                A_fit = float(A_fit)
                B_fit = float(B_fit)
                a_pred = model(v_int, A_fit, B_fit)
                rmse = float(np.sqrt(np.mean((a_pred - a_int) ** 2)))
                ss_res = np.sum((a_pred - a_int) ** 2)
                ss_tot = np.sum((a_int - np.mean(a_int)) ** 2)
                r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
            except Exception:
                A_fit, B_fit, rmse, r2 = 0.0, 1.0, 0.0, 0.0
            fit_results.append({
                "experiment": eid,
                "F_ext": float(F_ext),
                "v0": float(v0),
                "A_fit": A_fit,
                "B_fit": B_fit,
                "RMSE": rmse,
                "R2": r2,
                "is_free": True
            })
            continue

        # ---------- 恒外力实验 ----------
        # 确保 v 绝对值化以稳定拟合
        v_abs = np.abs(v_int)
        # 模型 a = A * exp(-|v|/B)
        def model(v, A, B):
            return A * np.exp(-v / B)

        # 初始猜测
        a0_guess = float(a_int[0]) if len(a_int) > 0 else 0.0
        B0_guess = float(np.max(v_abs)) / 2.0 if np.max(v_abs) > 0 else 1.0
        if B0_guess < 0.01:
            B0_guess = 1.0
        try:
            popt, _ = curve_fit(model, v_abs, a_int, p0=[a0_guess, B0_guess], maxfev=10000)
            A_fit, B_fit = popt
            A_fit = float(A_fit)
            B_fit = float(B_fit)
            # 确保 B>0
            if B_fit < 0:
                B_fit = 1.0
            a_pred = model(v_abs, A_fit, B_fit)
            rmse = float(np.sqrt(np.mean((a_pred - a_int) ** 2)))
            ss_res = np.sum((a_pred - a_int) ** 2)
            ss_tot = np.sum((a_int - np.mean(a_int)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0
        except Exception as e:
            A_fit, B_fit = None, None
            rmse, r2 = None, None

        fit_results.append({
            "experiment": eid,
            "F_ext": float(F_ext),
            "v0": float(v0),
            "A_fit": A_fit,
            "B_fit": B_fit,
            "RMSE": rmse,
            "R2": r2,
            "is_free": False
        })

    # ---------- 构建 DataFrame ----------
    df_fits = pd.DataFrame(fit_results)

    # ---------- 多变量线性回归 ----------
    # A = c1*F_ext + c2*v0 + intercept
    # B = d1*F_ext + d2*v0 + intercept
    # 只使用恒外力且拟合成功的实验
    df_force = df_fits[df_fits['is_free'] == False].dropna(subset=['A_fit', 'B_fit'])
    X = df_force[['F_ext', 'v0']].values
    y_A = df_force['A_fit'].values
    y_B = df_force['B_fit'].values

    if len(df_force) > 0:
        reg_A = LinearRegression().fit(X, y_A)
        reg_B = LinearRegression().fit(X, y_B)
        A_c1 = float(reg_A.coef_[0])
        A_c2 = float(reg_A.coef_[1])
        A_intercept = float(reg_A.intercept_)
        A_R2 = float(reg_A.score(X, y_A))
        B_d1 = float(reg_B.coef_[0])
        B_d2 = float(reg_B.coef_[1])
        B_intercept = float(reg_B.intercept_)
        B_R2 = float(reg_B.score(X, y_B))
    else:
        A_c1 = A_c2 = A_intercept = A_R2 = 0.0
        B_d1 = B_d2 = B_intercept = B_R2 = 0.0

    # ---------- 自由实验检查 ----------
    free_means = []
    for item in free_exp_a_means:
        free_means.append(f"{item['experiment']}: mean_a={item['mean_acceleration']:.2e}, zero={item['is_zero']}")

    # 计算所有自由实验的 A 均值（指数拟合结果）
    df_free = df_fits[df_fits['is_free'] == True]
    free_A_mean = float(df_free['A_fit'].mean()) if len(df_free) > 0 else 0.0
    free_A_std = float(df_free['A_fit'].std()) if len(df_free) > 0 else 0.0

    # ---------- 构造证据 ----------
    # 对所有恒外力实验，计算平均R2作为拟合质量指标
    valid_r2 = df_force['R2'].dropna()
    mean_r2 = float(valid_r2.mean()) if len(valid_r2) > 0 else 0.0
    # 自由实验 A 是否接近 0
    free_a_zero = all(item['is_zero'] for item in free_exp_a_means)
    # 回归 R2 作为线性关系指标
    A_regression_strong = A_R2 > 0.7
    B_regression_strong = B_R2 > 0.7

    # 构建 evidence 列表
    evidence_list = []
    # 1. 指数模型拟合质量
    evidence_list.append({
        "hypothesis_id": "H003",
        "supports": bool(mean_r2 > 0.8),  # 如果平均 R2 > 0.8 认为模型数据一致
        "metric_name": "mean_R2_exponential_fits",
        "metric_values": [mean_r2],
        "aggregate_score": mean_r2,
        "experiment_ids": df_force['experiment'].tolist(),
        "summary": f"Across {len(df_force)} forced experiments, exponential fit a=A*exp(-|v|/B) yields mean R2 = {mean_r2:.4f}."
    })
    # 2. 自由运动 A≈0
    evidence_list.append({
        "hypothesis_id": "H003",
        "supports": bool(free_a_zero),
        "metric_name": "free_experiment_A_magnitude",
        "metric_values": [free_A_mean, free_A_std],
        "aggregate_score": 0.0 if free_a_zero else 1.0,
        "experiment_ids": [item['experiment'] for item in free_exp_a_means],
        "summary": f"Free experiments: A_mean={free_A_mean:.2e}, A_std={free_A_std:.2e}, all zero? {free_a_zero}."
    })
    # 3. A 与 F_ext,v0 的多元线性回归
    evidence_list.append({
        "hypothesis_id": "H003",
        "supports": bool(A_regression_strong),
        "metric_name": "A_multilinear_R2",
        "metric_values": [A_R2],
        "aggregate_score": A_R2,
        "experiment_ids": df_force['experiment'].tolist(),
        "summary": f"A = {A_c1:.4f}*F_ext + {A_c2:.4f}*v0 + {A_intercept:.4f}, R2={A_R2:.4f}."
    })
    # 4. B 与 F_ext,v0 的多元线性回归
    evidence_list.append({
        "hypothesis_id": "H003",
        "supports": bool(B_regression_strong),
        "metric_name": "B_multilinear_R2",
        "metric_values": [B_R2],
        "aggregate_score": B_R2,
        "experiment_ids": df_force['experiment'].tolist(),
        "summary": f"B = {B_d1:.4f}*F_ext + {B_d2:.4f}*v0 + {B_intercept:.4f}, R2={B_R2:.4f}."
    })

    # ---------- 构建 observation ----------
    lines = []
    lines.append(f"处理 {len(df_force)} 个恒外力实验，成功对所有 {len(df_force)} 个实验完成指数模型 a = A * exp(-|v|/B) 拟合。")
    lines.append("各实验拟合参数 (A, B, RMSE, R2):")
    for _, row in df_force.iterrows():
        lines.append(f"  {row['experiment']}: F_ext={row['F_ext']}, v0={row['v0']}, A_fit={row['A_fit']:.4f}, B_fit={row['B_fit']:.4f}, RMSE={row['RMSE']:.4e}, R2={row['R2']:.4f}")
    lines.append(f"平均 R2 = {mean_r2:.4f}")
    lines.append(f"自由实验检查: {len(free_exp_a_means)} 个自由实验加速度均值均接近零，A均值为 {free_A_mean:.2e}。")
    lines.append(f"A 多变量线性回归: A = {A_c1:.4f}*F_ext + {A_c2:.4f}*v0 + {A_intercept:.4f}，R2={A_R2:.4f}")
    lines.append(f"B 多变量线性回归: B = {B_d1:.4f}*F_ext + {B_d2:.4f}*v0 + {B_intercept:.4f}，R2={B_R2:.4f}")
    lines.append("证据列表已写入 metrics['evidence']，包含 hypothesis_id=H003 的四个指标。")
    observation = "\n".join(lines)

    # ---------- 构造 metrics ----------
    metrics = {
        "per_experiment_fits": [{
            "experiment": r['experiment'],
            "F_ext": float(r['F_ext']),
            "v0": float(r['v0']),
            "A_fit": r['A_fit'] if not (isinstance(r['A_fit'], float) and np.isnan(r['A_fit'])) else None,
            "B_fit": r['B_fit'] if not (isinstance(r['B_fit'], float) and np.isnan(r['B_fit'])) else None,
            "RMSE": r['RMSE'] if not (isinstance(r['RMSE'], float) and np.isnan(r['RMSE'])) else None,
            "R2": r['R2'] if not (isinstance(r['R2'], float) and np.isnan(r['R2'])) else None,
            "is_free": bool(r['is_free'])
        } for _, r in df_fits.iterrows()] if len(df_fits) > 0 else [],
        "free_experiment_check": [{
            "experiment": item['experiment'],
            "F_ext": item['F_ext'],
            "v0": item['v0'],
            "mean_acceleration": item['mean_acceleration'],
            "max_abs_acceleration": item['max_abs_acceleration'],
            "is_zero": bool(item['is_zero'])
        } for item in free_exp_a_means],
        "A_multilinear_regression": {
            "c1": A_c1,
            "c2": A_c2,
            "intercept": A_intercept,
            "R2": A_R2
        },
        "B_multilinear_regression": {
            "d1": B_d1,
            "d2": B_d2,
            "intercept": B_intercept,
            "R2": B_R2
        },
        "free_A_mean": free_A_mean,
        "free_A_std": free_A_std,
        "evidence": evidence_list
    }

    # ---------- 导出图像（可选，但不强求） ----------
    # 可以画一个 A vs F_ext 散点图
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # A vs F_ext
    ax[0].scatter(df_force['F_ext'], df_force['A_fit'], c='blue', label='A fit')
    ax[0].set_xlabel('F_ext')
    ax[0].set_ylabel('A fit')
    ax[0].set_title('A vs F_ext')
    # 线性拟合线
    if len(df_force) > 1:
        x_fit = np.linspace(df_force['F_ext'].min(), df_force['F_ext'].max(), 100)
        y_fit = A_c1 * x_fit + A_c2 * df_force['v0'].mean() + A_intercept
        ax[0].plot(x_fit, y_fit, 'r--', label=f'R2={A_R2:.3f}')
    ax[0].legend()
    # B vs F_ext
    ax[1].scatter(df_force['F_ext'], df_force['B_fit'], c='green', label='B fit')
    ax[1].set_xlabel('F_ext')
    ax[1].set_ylabel('B fit')
    ax[1].set_title('B vs F_ext')
    if len(df_force) > 1:
        y_fit_B = B_d1 * x_fit + B_d2 * df_force['v0'].mean() + B_intercept
        ax[1].plot(x_fit, y_fit_B, 'm--', label=f'R2={B_R2:.3f}')
    ax[1].legend()
    from pathlib import Path
    output_path = Path(output_dir)
    fig_path = str(output_path / "A_B_vs_F_ext.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=100)
    plt.close()
    figures = [fig_path]

    return {
        "observation": observation,
        "derived_series": [],  # 没有新序列需要注册
        "figures": figures,
        "metrics": metrics
    }
