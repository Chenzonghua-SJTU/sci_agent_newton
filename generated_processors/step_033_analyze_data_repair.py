import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------

def sign_ext(F: float) -> float:
    """返回 F 的符号，0 时返回 1（自由实验不会用到）"""
    if F == 0.0:
        return 1.0
    return 1.0 if F > 0 else -1.0

def exp_decay(v: np.ndarray, A: float, B: float) -> np.ndarray:
    """a = A * exp(-|v|/B)"""
    return A * np.exp(-np.abs(v) / B)

def fit_exponential(
    v: np.ndarray,
    a: np.ndarray,
    sign_f: float
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    拟合 a = sign_f * A * exp(-|v|/B), 要求 A > 0.
    返回 (A, B, rmse, r2, residuals)
    """
    # 先去掉可能的 nan/inf
    mask = np.isfinite(v) & np.isfinite(a)
    vv = v[mask]
    aa = a[mask]
    if len(vv) < 3:
        raise ValueError("数据点过少，无法拟合")

    # 变换：a_sign = a * sign_f，则应满足 a_sign = A * exp(-|v|/B)
    a_sign = aa * sign_f

    # 初值估计
    A0 = max(np.max(a_sign), 1e-10)   # 确保正数
    B0 = np.median(np.abs(vv)) if np.median(np.abs(vv)) > 0 else 1.0

    # 尝试曲线拟合，增加多重初值尝试
    best_params = None
    best_rmse = np.inf
    for p0 in [(A0, B0), (A0*0.5, B0*1.5), (A0*2, B0*0.8)]:
        try:
            popt, _ = curve_fit(
                exp_decay, vv, a_sign,
                p0=p0,
                bounds=([0, 1e-6], [np.inf, np.inf]),
                maxfev=5000
            )
            pred = exp_decay(vv, *popt)
            rmse = np.sqrt(np.mean((pred - a_sign)**2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = popt
        except Exception:
            continue

    if best_params is None:
        # 后备：对数线性拟合
        # ln(a_sign) = ln(A) - |v|/B   (要求 a_sign > 0)
        pos = a_sign > 0
        if np.sum(pos) < 3:
            raise ValueError("无法拟合指数模型（数据非正）")
        vv_pos = vv[pos]
        a_pos = a_sign[pos]
        ln_a = np.log(a_pos)
        # 线性拟合 ln_a vs |v|
        coeff = np.polyfit(np.abs(vv_pos), ln_a, 1)
        lnA = coeff[1]
        invB = -coeff[0]
        if invB <= 0:
            invB = 1e-6
        A_est = np.exp(lnA)
        B_est = 1.0 / invB
        if A_est <= 0 or B_est <= 0:
            A_est = np.max(a_sign) * 0.9
            B_est = np.median(np.abs(vv)) if np.median(np.abs(vv))>0 else 1.0
        pred = exp_decay(vv, A_est, B_est)
        best_rmse = np.sqrt(np.mean((pred - a_sign)**2))
        best_params = (A_est, B_est)

    A_fit, B_fit = best_params
    pred = exp_decay(vv, A_fit, B_fit)
    residuals = a_sign - pred
    rmse = best_rmse
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((a_sign - np.mean(a_sign))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return A_fit, B_fit, rmse, r2, residuals

# ------------------------------------------------------------
# 主处理函数
# ------------------------------------------------------------

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", "/tmp/artifacts")

    # 使用实验列表
    exp_ids = params.get("experiment_ids", list(experiments.keys()))

    per_experiment_fits = []
    free_experiment_checks = []
    forced_data = []  # 用于回归的 (A, B, F_ext, v0)

    for exp_id in exp_ids:
        if exp_id not in experiments:
            continue
        exp = experiments[exp_id]
        config = exp["config"]
        series = exp["series"]
        available = exp.get("available_series", [])

        F_ext = config.get("F_ext", 0.0)
        v0 = config.get("initial_v", 0.0)
        force_type = config.get("force_field_type", "")

        # 选择加速度和速度序列 (优先中央差分)
        # 期望存在 acceleration_central_{exp_id} 和 velocity_central_{exp_id}
        acc_key = f"acceleration_central_{exp_id}"
        vel_key = f"velocity_central_{exp_id}"
        if acc_key in series and vel_key in series:
            acc_raw = np.array(series[acc_key])
            vel_raw = np.array(series[vel_key])
        else:
            # 尝试其他名称
            acc_candidates = [k for k in available if "acceleration" in k and exp_id in k]
            vel_candidates = [k for k in available if "velocity" in k and exp_id in k]
            if acc_candidates and vel_candidates:
                acc_raw = np.array(series[acc_candidates[0]])
                vel_raw = np.array(series[vel_candidates[0]])
            else:
                # 没有加速度/速度，跳过
                continue

        # 剔除前后5个边界点
        if len(acc_raw) <= 10:
            v_internal = vel_raw
            a_internal = acc_raw
        else:
            v_internal = vel_raw[5:-5]
            a_internal = acc_raw[5:-5]

        if len(v_internal) < 3:
            continue

        # 对于自由实验，验证加速度是否为零
        if abs(F_ext) < 1e-12:
            mean_a = np.mean(a_internal)
            max_abs_a = np.max(np.abs(a_internal))
            # 显式转换为 Python 原生类型，确保 JSON 可序列化
            is_zero = bool(max_abs_a < 1e-10)
            free_experiment_checks.append({
                "experiment": exp_id,
                "F_ext": float(F_ext),
                "v0": float(v0),
                "mean_acceleration": float(mean_a),
                "max_abs_acceleration": float(max_abs_a),
                "is_zero": is_zero
            })
            continue  # 不拟合指数模型

        # 恒外力实验，拟合指数模型 a = sign(F_ext) * A * exp(-|v|/B)
        sign_f = sign_ext(F_ext)
        try:
            A_fit, B_fit, rmse, r2, residuals = fit_exponential(
                v_internal, a_internal, sign_f
            )
        except Exception as e:
            # 拟合失败，记录错误
            per_experiment_fits.append({
                "experiment": exp_id,
                "F_ext": float(F_ext),
                "v0": float(v0),
                "A_fit": None,
                "B_fit": None,
                "RMSE": None,
                "R2": None,
                "error": str(e)
            })
            continue

        # 计算残差统计
        residual_mean = float(np.mean(residuals))
        residual_std = float(np.std(residuals, ddof=1) if len(residuals)>1 else 0.0)

        per_experiment_fits.append({
            "experiment": exp_id,
            "F_ext": float(F_ext),
            "v0": float(v0),
            "A_fit": float(A_fit),
            "B_fit": float(B_fit),
            "RMSE": float(rmse),
            "R2": float(r2),
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "n_points": len(v_internal)
        })

        forced_data.append({
            "A": A_fit,
            "B": B_fit,
            "F_ext": float(F_ext),
            "v0": float(v0)
        })

    # 多元线性回归：A = c1*F_ext + c2*v0 + intercept
    if len(forced_data) >= 3:
        X = np.array([[d["F_ext"], d["v0"]] for d in forced_data])
        y_A = np.array([d["A"] for d in forced_data])
        y_B = np.array([d["B"] for d in forced_data])

        reg_A = LinearRegression().fit(X, y_A)
        reg_B = LinearRegression().fit(X, y_B)

        A_multilinear = {
            "c1": float(reg_A.coef_[0]),
            "c2": float(reg_A.coef_[1]),
            "intercept": float(reg_A.intercept_),
            "R2": float(reg_A.score(X, y_A))
        }
        B_multilinear = {
            "d1": float(reg_B.coef_[0]),
            "d2": float(reg_B.coef_[1]),
            "intercept": float(reg_B.intercept_),
            "R2": float(reg_B.score(X, y_B))
        }
    else:
        A_multilinear = {"error": "insufficient data"}
        B_multilinear = {"error": "insufficient data"}

    # 自由实验总结
    free_experiment_check = free_experiment_checks

    # 绘图
    figures = []
    if len(forced_data) > 0:
        df = pd.DataFrame(forced_data)
        # A vs F_ext 按 v0 着色
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        scatter = ax1.scatter(df["F_ext"], df["A"], c=df["v0"], cmap="viridis", s=60, edgecolors='k')
        ax1.set_xlabel("F_ext")
        ax1.set_ylabel("A")
        ax1.set_title("A vs F_ext (colored by v0)")
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("v0")
        fig1.tight_layout()
        path1 = str(Path(output_dir) / "A_vs_F_ext.png")
        fig1.savefig(path1, dpi=150)
        plt.close(fig1)
        figures.append(path1)

        # B vs F_ext 按 v0 着色
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        scatter2 = ax2.scatter(df["F_ext"], df["B"], c=df["v0"], cmap="plasma", s=60, edgecolors='k')
        ax2.set_xlabel("F_ext")
        ax2.set_ylabel("B")
        ax2.set_title("B vs F_ext (colored by v0)")
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label("v0")
        fig2.tight_layout()
        path2 = str(Path(output_dir) / "B_vs_F_ext.png")
        fig2.savefig(path2, dpi=150)
        plt.close(fig2)
        figures.append(path2)

    # 构建 observation
    n_forced = len(per_experiment_fits)
    n_free = len(free_experiment_checks)
    obs_lines = [
        f"处理 {n_forced} 个恒外力实验和 {n_free} 个自由实验。",
        "恒外力实验指数模型 a = sign(F_ext)*A*exp(-|v|/B) 拟合结果："
    ]
    for f in per_experiment_fits:
        if f["A_fit"] is not None:
            obs_lines.append(
                f"  {f['experiment']}: F_ext={f['F_ext']:.1f}, v0={f['v0']:.1f}, "
                f"A={f['A_fit']:.4f}, B={f['B_fit']:.4f}, RMSE={f['RMSE']:.6f}, R2={f['R2']:.4f}"
            )
        else:
            obs_lines.append(f"  {f['experiment']}: 拟合失败 - {f.get('error','')}")
    if free_experiment_checks:
        obs_lines.append("自由实验加速度验证：")
        for c in free_experiment_checks:
            obs_lines.append(
                f"  {c['experiment']}: mean_acc={c['mean_acceleration']:.2e}, "
                f"max_abs={c['max_abs_acceleration']:.2e}, zero={c['is_zero']}"
            )
    if len(forced_data) >= 3:
        obs_lines.append(
            f"A多元回归: A = {A_multilinear['c1']:.4f}*F_ext + {A_multilinear['c2']:.4f}*v0 + "
            f"{A_multilinear['intercept']:.4f}, R2={A_multilinear['R2']:.4f}"
        )
        obs_lines.append(
            f"B多元回归: B = {B_multilinear['d1']:.4f}*F_ext + {B_multilinear['d2']:.4f}*v0 + "
            f"{B_multilinear['intercept']:.4f}, R2={B_multilinear['R2']:.4f}"
        )
    observation = "\n".join(obs_lines)

    metrics = {
        "per_experiment_fits": per_experiment_fits,
        "free_experiment_check": free_experiment_check,
        "A_multilinear_regression": A_multilinear,
        "B_multilinear_regression": B_multilinear
    }

    return {
        "observation": observation,
        "derived_series": [],
        "figures": figures,
        "metrics": metrics
    }
