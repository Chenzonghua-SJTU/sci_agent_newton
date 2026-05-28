import json
import math
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload["action"]
    parameters = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # ------ 解析参数 ------
    analysis_goal = parameters.get("analysis_goal", "")
    exp_ids = parameters.get("experiment_ids", list(experiments.keys()))
    optional_series = parameters.get("optional_series", [])
    expected_outputs = parameters.get("expected_outputs", [])

    # ------ 辅助函数：安全获取序列 ------
    def get_series(exp_id: str, name: str) -> Optional[List[float]]:
        exp = experiments[exp_id]
        series_dict = exp.get("series", {})
        if name in series_dict:
            return series_dict[name]
        # 尝试带后缀的模式
        alt_name = f"{name}_{exp_id}"
        if alt_name in series_dict:
            return series_dict[alt_name]
        return None

    def get_time(exp_id: str) -> List[float]:
        t = get_series(exp_id, "t")
        if t is not None:
            return t
        raise ValueError(f"Experiment {exp_id} has no time series 't'.")

    def get_position(exp_id: str) -> List[float]:
        q = get_series(exp_id, "q")
        if q is not None:
            return q
        raise ValueError(f"Experiment {exp_id} has no position series 'q'.")

    # ------ 提取加速度和速度序列（优先中心差分） ------
    def get_acc_and_vel(exp_id: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (t_internal, a_internal, v_internal) after trimming 5 points at each end.
        If sequences not available, compute from q using central difference.
        """
        t = np.array(get_time(exp_id), dtype=float)
        q = np.array(get_position(exp_id), dtype=float)
        dt = t[1] - t[0]  # uniform
        n = len(q)

        # Try to use existing computed sequences
        a = None
        v = None
        acc_names = ["acceleration_central", "acceleration"]
        vel_names = ["velocity_central", "velocity"]
        for name in acc_names:
            a_raw = get_series(exp_id, name)
            if a_raw is not None:
                a = np.array(a_raw, dtype=float)
                break
        for name in vel_names:
            v_raw = get_series(exp_id, name)
            if v_raw is not None:
                v = np.array(v_raw, dtype=float)
                break

        # If missing, compute central differences
        if a is None or v is None:
            # compute velocity via central difference
            v_central = np.empty(n)
            v_central[0] = (q[1] - q[0]) / dt
            for i in range(1, n-1):
                v_central[i] = (q[i+1] - q[i-1]) / (2*dt)
            v_central[-1] = (q[-1] - q[-2]) / dt
            v = v_central

            a_central = np.empty(n)
            a_central[0] = (v_central[1] - v_central[0]) / dt
            for i in range(1, n-1):
                a_central[i] = (v_central[i+1] - v_central[i-1]) / (2*dt)
            a_central[-1] = (v_central[-1] - v_central[-2]) / dt
            a = a_central

        # Trim 5 points at each end
        trim = 5
        if n > 2*trim:
            a = a[trim:n-trim]
            v = v[trim:n-trim]
            t = t[trim:n-trim]
        else:
            a = a.copy()
            v = v.copy()
            t = t.copy()

        return t, a, v

    # ------ 拟合函数 ------
    def fit_linear(v, a):
        """a = a0 + b*v"""
        def model(v, a0, b):
            return a0 + b*v
        try:
            popt, pcov = curve_fit(model, v, a, p0=[a[0], 0.0])
            a0, b = popt
            residuals = a - model(v, a0, b)
            n = len(v)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((a - np.mean(a))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
            rmse = np.sqrt(ss_res / n)
            return {"a0": a0, "b": b, "RMSE": rmse, "R2": r2}
        except Exception as e:
            return {"error": str(e)}

    def fit_quadratic(v, a):
        """a = a0 + b*v + c*v^2"""
        def model(v, a0, b, c):
            return a0 + b*v + c*v**2
        try:
            popt, pcov = curve_fit(model, v, a, p0=[a[0], 0.0, 0.0])
            a0, b, c = popt
            residuals = a - model(v, a0, b, c)
            n = len(v)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((a - np.mean(a))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
            rmse = np.sqrt(ss_res / n)
            return {"a0": a0, "b": b, "c": c, "RMSE": rmse, "R2": r2}
        except Exception as e:
            return {"error": str(e)}

    def fit_exponential(v, a):
        """a = A*exp(-v/B) + C  (with offset)"""
        def model(v, A, B, C):
            return A * np.exp(-v / B) + C
        # initial guess: A = max(a)-min(a), B = positive, C = min(a)
        try:
            A0 = np.max(a) - np.min(a) if np.max(a) != np.min(a) else 1.0
            B0 = 1.0
            C0 = np.min(a)
            popt, pcov = curve_fit(model, v, a, p0=[A0, B0, C0], maxfev=5000)
            A, B, C = popt
            residuals = a - model(v, A, B, C)
            n = len(v)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((a - np.mean(a))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
            rmse = np.sqrt(ss_res / n)
            return {"A": A, "B": B, "C": C, "RMSE": rmse, "R2": r2}
        except Exception as e:
            return {"error": str(e)}

    # ------ 全局联合拟合 a = p1 * F_ext * exp(-p2 * v) + p3 ------
    def fit_global(all_v, all_a, all_F):
        """joint fit over all forced experiments"""
        def model(params, v, F):
            p1, p2, p3 = params
            return p1 * F * np.exp(-p2 * v) + p3
        def loss(params, v, a, F):
            return np.sum((a - model(params, v, F))**2)
        from scipy.optimize import minimize
        # initial guess
        p0 = np.array([1.0, 0.5, 0.0])
        res = minimize(loss, p0, args=(all_v, all_a, all_F), method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success:
            p1, p2, p3 = res.x
            pred = model(res.x, all_v, all_F)
            residuals = all_a - pred
            n = len(all_v)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((all_a - np.mean(all_a))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
            rmse = np.sqrt(ss_res / n)
            return {"p1": p1, "p2": p2, "p3": p3, "RMSE": rmse, "R2": r2}
        else:
            return {"error": res.message}

    # ------ 主循环 ------
    forced_exps = []
    free_exps = []
    for eid in exp_ids:
        if eid not in experiments:
            continue
        config = experiments[eid].get("config", {})
        F_ext = config.get("F_ext", 0.0)
        if abs(F_ext) > 1e-12:
            forced_exps.append(eid)
        else:
            free_exps.append(eid)

    per_exp_results = []
    forced_data = []  # for global fit
    for eid in forced_exps:
        config = experiments[eid].get("config", {})
        F_ext = config.get("F_ext", 0.0)
        v0 = config.get("initial_v", 0.0)
        try:
            t, a, v = get_acc_and_vel(eid)
        except Exception as e:
            per_exp_results.append({
                "experiment": eid,
                "F_ext": F_ext,
                "v0": v0,
                "error": f"Failed to extract data: {str(e)}"
            })
            continue
        if len(a) < 3:
            per_exp_results.append({"experiment": eid, "F_ext": F_ext, "v0": v0, "error": "Insufficient data points"})
            continue

        # fit three models
        linear = fit_linear(v, a)
        quad = fit_quadratic(v, a)
        expo = fit_exponential(v, a)

        result = {
            "experiment": eid,
            "F_ext": F_ext,
            "v0": v0,
            "n_points": len(a),
            "linear_fit": linear,
            "quadratic_fit": quad,
            "exponential_fit": expo,
        }
        per_exp_results.append(result)

        # collect for global fit (use absolute v to avoid sign issues in exp)
        # but formula uses raw v; we follow the specification.
        forced_data.append({
            "v": v,
            "a": a,
            "F": np.full_like(v, F_ext)
        })

    # global joint fit
    global_result = None
    if len(forced_data) > 0:
        all_v = np.concatenate([d["v"] for d in forced_data])
        all_a = np.concatenate([d["a"] for d in forced_data])
        all_F = np.concatenate([d["F"] for d in forced_data])
        global_result = fit_global(all_v, all_a, all_F)
    else:
        global_result = {"error": "No forced experiments available"}

    # free experiments check
    free_checks = []
    for eid in free_exps:
        config = experiments[eid].get("config", {})
        F_ext = config.get("F_ext", 0.0)
        try:
            t, a, v = get_acc_and_vel(eid)
        except:
            free_checks.append({"experiment": eid, "error": "data extraction failed"})
            continue
        mean_a = float(np.mean(a))
        max_abs_a = float(np.max(np.abs(a)))
        is_zero = abs(mean_a) < 1e-10 and max_abs_a < 1e-9
        free_checks.append({
            "experiment": eid,
            "F_ext": F_ext,
            "v0": config.get("initial_v", 0.0),
            "mean_acceleration": mean_a,
            "max_abs_acceleration": max_abs_a,
            "is_zero": is_zero
        })

    # ------ 构建输出 metrics ------
    metrics = {
        "per_experiment_fits": per_exp_results,
        "global_joint_fit": global_result,
        "free_experiment_checks": free_checks
    }

    # ------ 构建 observation ------
    n_forced = len(forced_exps)
    n_free = len(free_exps)
    obs_lines = [
        f"处理了 {n_forced} 个恒外力实验和 {n_free} 个自由实验。",
        f"对每个恒外力实验排除边界各5点后，拟合线性、二次、指数（带偏移）模型。",
        f"全局联合拟合使用公式 a = p1 * F_ext * exp(-p2 * v) + p3。"
    ]
    # 报告平均 RMSE
    if global_result and "error" not in global_result:
        obs_lines.append(
            f"全局联合拟合: p1={global_result['p1']:.4f}, p2={global_result['p2']:.4f}, "
            f"p3={global_result['p3']:.4f}, RMSE={global_result['RMSE']:.4f}, R²={global_result['R2']:.4f}"
        )
    else:
        obs_lines.append(f"全局联合拟合失败: {global_result.get('error', '')}")
    # 自由实验检查
    zero_count = sum(1 for c in free_checks if c.get("is_zero"))
    obs_lines.append(f"自由实验检查: {zero_count}/{len(free_checks)} 实验加速度接近零。")
    observation = "\n".join(obs_lines)

    # ------ 生成图像 ------
    figures = []
    if len(forced_exps) > 0:
        # 1. 每个实验三个模型的 RMSE 对比
        fig, ax = plt.subplots(figsize=(10, 5))
        exp_names = [r["experiment"][:8] for r in per_exp_results]
        lin_rmse = [r.get("linear_fit", {}).get("RMSE") if "error" not in r.get("linear_fit", {}) else None for r in per_exp_results]
        quad_rmse = [r.get("quadratic_fit", {}).get("RMSE") if "error" not in r.get("quadratic_fit", {}) else None for r in per_exp_results]
        exp_rmse = [r.get("exponential_fit", {}).get("RMSE") if "error" not in r.get("exponential_fit", {}) else None for r in per_exp_results]
        x = np.arange(len(exp_names))
        width = 0.25
        ax.bar(x - width, lin_rmse, width, label='Linear', alpha=0.7)
        ax.bar(x, quad_rmse, width, label='Quadratic', alpha=0.7)
        ax.bar(x + width, exp_rmse, width, label='Exponential', alpha=0.7)
        ax.set_xlabel('Experiment')
        ax.set_ylabel('RMSE')
        ax.set_title('Model RMSE Comparison for Forced Experiments')
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45)
        ax.legend()
        plt.tight_layout()
        path1 = f"{output_dir}/model_comparison_rmse.png"
        fig.savefig(path1)
        plt.close(fig)
        figures.append(path1)

        # 2. 全局联合拟合的残差分布
        if global_result and "error" not in global_result:
            fig, ax = plt.subplots(figsize=(8, 5))
            all_v = np.concatenate([d["v"] for d in forced_data])
            all_a = np.concatenate([d["a"] for d in forced_data])
            all_F = np.concatenate([d["F"] for d in forced_data])
            pred = global_result["p1"] * all_F * np.exp(-global_result["p2"] * all_v) + global_result["p3"]
            resid = all_a - pred
            ax.scatter(all_v, resid, s=5, alpha=0.5)
            ax.axhline(0, color='k', ls='--')
            ax.set_xlabel('Velocity')
            ax.set_ylabel('Residual')
            ax.set_title('Global Joint Fit Residuals')
            plt.tight_layout()
            path2 = f"{output_dir}/global_fit_residuals.png"
            fig.savefig(path2)
            plt.close(fig)
            figures.append(path2)

        # 3. 自由实验加速度检查
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [c["experiment"][:6] for c in free_checks]
        means = [c.get("mean_acceleration", 0) for c in free_checks]
        ax.bar(labels, means, alpha=0.7)
        ax.axhline(0, color='k', ls='--')
        ax.set_ylabel('Mean Acceleration')
        ax.set_title('Free Experiments: Mean Acceleration')
        plt.tight_layout()
        path3 = f"{output_dir}/free_experiment_acceleration.png"
        fig.savefig(path3)
        plt.close(fig)
        figures.append(path3)

    return {
        "observation": observation,
        "figures": figures,
        "metrics": metrics,
        "derived_series": []
    }
