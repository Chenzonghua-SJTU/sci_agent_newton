import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def _compute_kinematics(q: np.ndarray, t: np.ndarray, window: int = 15, polyorder: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """使用 Savitzky-Golay 滤波器从位置序列估计速度和加速度。
    返回 (v, a) 数组，长度与 q 相同。
    """
    dt = t[1] - t[0]
    # 一阶导数
    v = savgol_filter(q, window, polyorder, deriv=1, delta=dt)
    # 二阶导数
    a = savgol_filter(q, window, polyorder, deriv=2, delta=dt)
    return v, a

def _safe_mean_std(arr: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(arr)), float(np.std(arr))

def _fit_linear_free(v: np.ndarray, a: np.ndarray) -> Dict:
    """拟合 a = intercept + slope * v"""
    coeffs = np.polyfit(v, a, 1)
    intercept, slope = coeffs[1], coeffs[0]
    pred = intercept + slope * v
    rmse = math.sqrt(mean_squared_error(a, pred))
    r2 = r2_score(a, pred)
    return {
        'alpha': float(intercept),
        'beta': float(-slope),  # 因为 a = alpha - beta*v => slope = -beta
        'intercept': float(intercept),
        'slope_free': float(slope),
        'rmse': rmse,
        'r2': r2
    }

def _fit_quad_free(vsq: np.ndarray, a: np.ndarray) -> Dict:
    """拟合 a = intercept + coeff * v²"""
    coeffs = np.polyfit(vsq, a, 1)
    intercept, coeff = coeffs[1], coeffs[0]
    pred = intercept + coeff * vsq
    rmse = math.sqrt(mean_squared_error(a, pred))
    r2 = r2_score(a, pred)
    return {
        'alpha': float(intercept),
        'gamma': float(-coeff),  # a = alpha - gamma*v²，所以 gamma = -coeff
        'intercept': float(intercept),
        'coeff_vsq': float(coeff),
        'rmse': rmse,
        'r2': r2
    }

def _fit_constrained_linear(v: np.ndarray, a: np.ndarray, F_ext: float) -> Dict:
    """拟合 a = F_ext - gamma * v, gamma >=0"""
    def model(v, gamma):
        return F_ext - gamma * v
    try:
        popt, _ = curve_fit(model, v, a, p0=[1.0], bounds=(0, np.inf))
        gamma = popt[0]
        pred = model(v, gamma)
        rmse = math.sqrt(mean_squared_error(a, pred))
        r2 = r2_score(a, pred)
        return {'gamma': float(gamma), 'rmse': rmse, 'r2': r2}
    except Exception:
        return {'gamma': np.nan, 'rmse': np.nan, 'r2': np.nan}

def _fit_constrained_quad(vsq: np.ndarray, a: np.ndarray, F_ext: float) -> Dict:
    """拟合 a = F_ext - gamma * v², gamma >=0"""
    def model(vsq, gamma):
        return F_ext - gamma * vsq
    try:
        popt, _ = curve_fit(model, vsq, a, p0=[1.0], bounds=(0, np.inf))
        gamma = popt[0]
        pred = model(vsq, gamma)
        rmse = math.sqrt(mean_squared_error(a, pred))
        r2 = r2_score(a, pred)
        return {'gamma': float(gamma), 'rmse': rmse, 'r2': r2}
    except Exception:
        return {'gamma': np.nan, 'rmse': np.nan, 'r2': np.nan}

def _interpolate_at_velocity(v: np.ndarray, a: np.ndarray, target_v: float) -> float:
    """在速度 v 处线性插值加速度 a。如果在范围外则返回 nan。"""
    if target_v < np.min(v) or target_v > np.max(v):
        return float('nan')
    return float(np.interp(target_v, v, a))

def process(payload: dict) -> dict:
    action = payload.get("action", "custom_data_analysis")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(experiments.keys())
    target_ids = [e for e in exp_ids if e in experiments]
    if not target_ids:
        raise ValueError(f"无有效实验ID: {exp_ids}")

    # 存储处理后的数据
    data: Dict[str, dict] = {}
    derived_series = []

    for eid in target_ids:
        exp = experiments[eid]
        config = exp.get("config", {})
        series = exp.get("series", {})
        available = exp.get("available_series", [])
        F_ext = config.get("F_ext", 0.0)
        t = np.array(series.get("t", []))
        q = np.array(series.get("q", []))
        if len(t) == 0 or len(q) == 0:
            raise ValueError(f"实验 {eid} 缺少 t 或 q 序列")

        # 尝试使用已有 v_sg_long / a_sg_long，否则计算
        if "v_sg_long" in available and "a_sg_long" in available:
            v = np.array(series["v_sg_long"])
            a = np.array(series["a_sg_long"])
            computed = False
        else:
            window = 15
            polyorder = 3
            v, a = _compute_kinematics(q, t, window, polyorder)
            computed = True
            # 记录派生序列
            derived_series.append({
                "experiment_id": eid,
                "name": "v_sg_long",
                "values": v.tolist(),
                "source_name": f"Savitzky-Golay filter (window={window}, polyorder={polyorder}, deriv=1) on q",
                "provenance": "custom_data_analysis: optimal window kinematics",
                "description": "速度（SG滤波，窗长15，3阶多项式）"
            })
            derived_series.append({
                "experiment_id": eid,
                "name": "a_sg_long",
                "values": a.tolist(),
                "source_name": f"Savitzky-Golay filter (window={window}, polyorder={polyorder}, deriv=2) on q",
                "provenance": "custom_data_analysis: optimal window kinematics",
                "description": "加速度（SG滤波，窗长15，3阶多项式）"
            })

        vsq = v ** 2
        data[eid] = {
            't': t,
            'q': q,
            'v': v,
            'a': a,
            'vsq': vsq,
            'F_ext': F_ext,
            'computed_kinematics': computed
        }

    # 2. 拟合分析
    fit_results = {}
    for eid in target_ids:
        d = data[eid]
        v = d['v']
        a = d['a']
        vsq = d['vsq']
        F_ext = d['F_ext']

        # 线性自由
        free_linear = _fit_linear_free(v, a)
        # 二次自由
        free_quad = _fit_quad_free(vsq, a)
        # 线性约束
        const_linear = _fit_constrained_linear(v, a, F_ext)
        # 二次约束
        const_quad = _fit_constrained_quad(vsq, a, F_ext)

        fit_results[eid] = {
            'free_linear': free_linear,
            'free_quad': free_quad,
            'constrained_linear': const_linear,
            'constrained_quad': const_quad,
            'F_ext': F_ext
        }

    # 3. 固定速度下分析 a vs F_ext
    common_v_test = [0.5, 1.0, 1.5, 2.0]  # 如果速度范围不足则跳过
    interp_results = {}
    for v_target in common_v_test:
        vals = []
        for eid in target_ids:
            d = data[eid]
            try:
                a_interp = _interpolate_at_velocity(d['v'], d['a'], v_target)
            except Exception:
                a_interp = float('nan')
            vals.append((eid, d['F_ext'], a_interp))
        interp_results[v_target] = vals

    # 检查线性比例：对于每个 v_target，如果非nan点>=2，计算相关性
    scale_metrics = {}
    for v_target, vals in interp_results.items():
        valid = [(F_ext, a_val) for _, F_ext, a_val in vals if not math.isnan(a_val)]
        if len(valid) >= 2:
            f_arr = np.array([x[0] for x in valid])
            a_arr = np.array([x[1] for x in valid])
            slope, intercept = np.polyfit(f_arr, a_arr, 1)
            corr = np.corrcoef(f_arr, a_arr)[0,1] if len(valid)>=3 else float('nan')
            scale_metrics[v_target] = {'slope': float(slope), 'intercept': float(intercept), 'corr': float(corr) if not math.isnan(corr) else None}
        else:
            scale_metrics[v_target] = None

    # 4. 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'exp_03': 'blue', 'exp_04': 'green', 'exp_05': 'red'}
    markers = {'exp_03': 'o', 'exp_04': 's', 'exp_05': '^'}

    # (a) a vs v
    ax1 = axes[0,0]
    for eid in target_ids:
        d = data[eid]
        ax1.scatter(d['v'], d['a'], label=f"{eid} (F_ext={d['F_ext']})", 
                    c=colors.get(eid, 'gray'), marker=markers.get(eid, 'o'), s=10, alpha=0.7)
        # 绘制自由线性拟合线
        fr = fit_results[eid]['free_linear']
        v_sort = np.sort(d['v'])
        pred = fr['intercept'] + fr['slope_free'] * v_sort
        ax1.plot(v_sort, pred, '--', c=colors.get(eid, 'gray'), label=f"{eid} linear fit")
    ax1.set_xlabel('v')
    ax1.set_ylabel('a')
    ax1.set_title('a vs v')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # (b) a vs v²
    ax2 = axes[0,1]
    for eid in target_ids:
        d = data[eid]
        ax2.scatter(d['vsq'], d['a'], label=f"{eid} (F_ext={d['F_ext']})", 
                    c=colors.get(eid, 'gray'), marker=markers.get(eid, 'o'), s=10, alpha=0.7)
        fr = fit_results[eid]['free_quad']
        vsq_sort = np.sort(d['vsq'])
        pred = fr['intercept'] + fr['coeff_vsq'] * vsq_sort
        ax2.plot(vsq_sort, pred, '--', c=colors.get(eid, 'gray'), label=f"{eid} quad fit")
    ax2.set_xlabel('v²')
    ax2.set_ylabel('a')
    ax2.set_title('a vs v²')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # (c) 固定速度下 a vs F_ext
    ax3 = axes[1,0]
    for v_target in common_v_test:
        vals = interp_results[v_target]
        valid = [(F_ext, a_val) for _, F_ext, a_val in vals if not math.isnan(a_val)]
        if not valid:
            continue
        f_vals = [x[0] for x in valid]
        a_vals = [x[1] for x in valid]
        ax3.scatter(f_vals, a_vals, label=f'v={v_target}', marker='o')
        # 拟合线
        if len(valid) >= 2:
            slope, intercept = np.polyfit(f_vals, a_vals, 1)
            f_line = np.linspace(min(f_vals), max(f_vals), 10)
            a_line = slope * f_line + intercept
            ax3.plot(f_line, a_line, '--')
    ax3.set_xlabel('F_ext')
    ax3.set_ylabel('a at fixed v')
    ax3.set_title('a vs F_ext at selected velocities')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # (d) 残差示例：线性自由拟合残差 vs v
    ax4 = axes[1,1]
    for eid in target_ids:
        d = data[eid]
        v = d['v']
        a = d['a']
        fr = fit_results[eid]['free_linear']
        pred = fr['intercept'] + fr['slope_free'] * v
        residual = a - pred
        ax4.scatter(v, residual, label=f"{eid}", c=colors.get(eid, 'gray'), marker='o', s=10, alpha=0.6)
    ax4.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax4.set_xlabel('v')
    ax4.set_ylabel('residual (a - linear fit)')
    ax4.set_title('Residuals of free linear fit')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "a_v_analysis_exp03_04_05.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)

    # 5. 构建 metrics
    metrics = {}
    for eid in target_ids:
        d = data[eid]
        fr = fit_results[eid]['free_linear']
        fq = fit_results[eid]['free_quad']
        cl = fit_results[eid]['constrained_linear']
        cq = fit_results[eid]['constrained_quad']
        metrics[f"{eid}_free_linear_alpha"] = fr['alpha']
        metrics[f"{eid}_free_linear_beta"] = fr['beta']
        metrics[f"{eid}_free_linear_rmse"] = fr['rmse']
        metrics[f"{eid}_free_linear_r2"] = fr['r2']
        metrics[f"{eid}_free_quad_alpha"] = fq['alpha']
        metrics[f"{eid}_free_quad_gamma"] = fq['gamma']
        metrics[f"{eid}_free_quad_rmse"] = fq['rmse']
        metrics[f"{eid}_free_quad_r2"] = fq['r2']
        metrics[f"{eid}_constrained_linear_gamma"] = cl['gamma']
        metrics[f"{eid}_constrained_linear_rmse"] = cl['rmse']
        metrics[f"{eid}_constrained_quad_gamma"] = cq['gamma']
        metrics[f"{eid}_constrained_quad_rmse"] = cq['rmse']
        # 基本统计
        v_mean, v_std = _safe_mean_std(d['v'])
        a_mean, a_std = _safe_mean_std(d['a'])
        metrics[f"{eid}_v_mean"] = v_mean
        metrics[f"{eid}_v_std"] = v_std
        metrics[f"{eid}_a_mean"] = a_mean
        metrics[f"{eid}_a_std"] = a_std
    for v_target, sm in scale_metrics.items():
        if sm is not None:
            metrics[f"v{v_target}_a_vs_F_slope"] = sm['slope']
            metrics[f"v{v_target}_a_vs_F_intercept"] = sm['intercept']
            metrics[f"v{v_target}_a_vs_F_corr"] = sm['corr'] if sm['corr'] is not None else 0.0

    # 跨实验比较 gamma 差异
    gamma_constrained_linear_list = [fit_results[eid]['constrained_linear']['gamma'] for eid in target_ids]
    gamma_constrained_quad_list = [fit_results[eid]['constrained_quad']['gamma'] for eid in target_ids]
    valid_linear = [g for g in gamma_constrained_linear_list if not math.isnan(g)]
    valid_quad = [g for g in gamma_constrained_quad_list if not math.isnan(g)]
    if len(valid_linear) >= 2:
        metrics['constrained_linear_gamma_range'] = max(valid_linear) - min(valid_linear)
        metrics['constrained_linear_gamma_mean'] = statistics.mean(valid_linear)
        metrics['constrained_linear_gamma_std'] = statistics.stdev(valid_linear) if len(valid_linear)>1 else 0.0
    if len(valid_quad) >= 2:
        metrics['constrained_quad_gamma_range'] = max(valid_quad) - min(valid_quad)
        metrics['constrained_quad_gamma_mean'] = statistics.mean(valid_quad)
        metrics['constrained_quad_gamma_std'] = statistics.stdev(valid_quad) if len(valid_quad)>1 else 0.0

    # 6. observation 字符串 (中文)
    obs_lines = ["恒外力实验 (exp_03, exp_04, exp_05) 的加速度-速度关系分析。"]
    for eid in target_ids:
        d = data[eid]
        obs_lines.append(f"{eid}: F_ext={d['F_ext']}, v均值={metrics[f'{eid}_v_mean']:.4f}, a均值={metrics[f'{eid}_a_mean']:.6f}.")
        fr = fit_results[eid]['free_linear']
        obs_lines.append(f"  自由线性 a = {fr['alpha']:.6f} - {fr['beta']:.6f}*v, R²={fr['r2']:.4f}, RMSE={fr['rmse']:.6f}.")
        fq = fit_results[eid]['free_quad']
        obs_lines.append(f"  自由二次 a = {fq['alpha']:.6f} - {fq['gamma']:.6f}*v², R²={fq['r2']:.4f}, RMSE={fq['rmse']:.6f}.")
        cl = fit_results[eid]['constrained_linear']
        obs_lines.append(f"  约束线性 a = F_ext - {cl['gamma']:.6f}*v, RMSE={cl['rmse']:.6f}.")
        cq = fit_results[eid]['constrained_quad']
        obs_lines.append(f"  约束二次 a = F_ext - {cq['gamma']:.6f}*v², RMSE={cq['rmse']:.6f}.")
    # 跨实验 gamma
    if 'constrained_linear_gamma_mean' in metrics:
        obs_lines.append(f"跨实验约束线性 gamma 均值={metrics['constrained_linear_gamma_mean']:.4f}, 范围={metrics['constrained_linear_gamma_range']:.4f}.")
    if 'constrained_quad_gamma_mean' in metrics:
        obs_lines.append(f"跨实验约束二次 gamma 均值={metrics['constrained_quad_gamma_mean']:.4f}, 范围={metrics['constrained_quad_gamma_range']:.4f}.")
    # 固定速度下 a vs F_ext
    for v_target in common_v_test:
        if f"v{v_target}_a_vs_F_corr" in metrics:
            obs_lines.append(f"v={v_target}时 a vs F_ext 相关系数={metrics[f'v{v_target}_a_vs_F_corr']:.4f}, 斜率={metrics[f'v{v_target}_a_vs_F_slope']:.4f}.")
    obs = "\n".join(obs_lines)

    return {
        "observation": obs,
        "derived_series": derived_series,
        "figures": [str(fig_path)],
        "metrics": metrics
    }
