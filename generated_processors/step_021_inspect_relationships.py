import json, math, statistics, itertools, functools, collections, pathlib, typing
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn import metrics as sk_metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiment_ids = params.get("experiment_ids", [])
    x_series = params["x_series"]
    y_series = params["y_series"]
    output_dir = payload["output_dir"]

    if not experiment_ids:
        raise ValueError("inspect_relationships requires at least one experiment_id")
    
    experiments = payload["experiments"]
    missing_series = []

    figures = []
    metrics = {}
    observations = []

    # 为所有实验绘制一张对比图
    fig_all, ax_all = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_ids)))

    for idx, eid in enumerate(experiment_ids):
        if eid not in experiments:
            continue
        exp = experiments[eid]
        series = exp["series"]
        config = exp["config"]
        F_ext = config.get("F_ext", 0.0)
        force_type = config.get("force_field_type", "constant")

        # 检查序列
        if x_series not in series:
            missing_series.append(f"{eid}: missing {x_series}")
            continue
        if y_series not in series:
            missing_series.append(f"{eid}: missing {y_series}")
            continue

        x = np.array(series[x_series], dtype=float)
        y = np.array(series[y_series], dtype=float)

        # 去除 NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        if len(x_clean) < 3:
            observations.append(f"{eid}: insufficient data points ({len(x_clean)}), skip fitting")
            continue

        # 相关系数
        corr, p_value = scipy_stats.pearsonr(x_clean, y_clean)

        # 线性拟合 y = slope * x + intercept
        A = np.vstack([x_clean, np.ones_like(x_clean)]).T
        m, c = np.linalg.lstsq(A, y_clean, rcond=None)[0]

        # 预测和残差
        y_pred = m * x_clean + c
        residuals = y_clean - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # 标准误差和置信区间 (95%)
        n = len(x_clean)
        dof = n - 2
        if dof > 0:
            mse = ss_res / dof
            var_m = mse / np.sum((x_clean - np.mean(x_clean))**2)
            var_c = mse * (1.0 / n + np.mean(x_clean)**2 / np.sum((x_clean - np.mean(x_clean))**2))
            se_m = math.sqrt(var_m)
            se_c = math.sqrt(var_c)
            t_val = scipy_stats.t.ppf(0.975, dof)
            ci_m = (m - t_val * se_m, m + t_val * se_m)
            ci_c = (c - t_val * se_c, c + t_val * se_c)
        else:
            ci_m = (m, m)
            ci_c = (c, c)

        # 保存指标
        prefix = f"{eid}"
        metrics[f"{prefix}_corr"] = corr
        metrics[f"{prefix}_corr_p"] = p_value
        metrics[f"{prefix}_slope"] = m
        metrics[f"{prefix}_intercept"] = c
        metrics[f"{prefix}_R2"] = r2
        metrics[f"{prefix}_slope_ci_low"] = ci_m[0]
        metrics[f"{prefix}_slope_ci_high"] = ci_m[1]
        metrics[f"{prefix}_intercept_ci_low"] = ci_c[0]
        metrics[f"{prefix}_intercept_ci_high"] = ci_c[1]

        # 观察文本
        obs = (f"{eid}: y={y_series} vs x={x_series}, "
               f"拟合: a = {m:.4f}*v + ({c:.4f}), "
               f"R²={r2:.4f}, 相关系数={corr:.4f} (p={p_value:.2e}), "
               f"slope CI95=[{ci_m[0]:.4f}, {ci_m[1]:.4f}], "
               f"intercept CI95=[{ci_c[0]:.4f}, {ci_c[1]:.4f}]")
        observations.append(obs)

        # 散点图 + 拟合线
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x_clean, y_clean, alpha=0.7, label='data')
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        y_line = m * x_line + c
        ax.plot(x_line, y_line, 'r-', label=f'fit (R²={r2:.3f})')
        ax.set_xlabel(x_series)
        ax.set_ylabel(y_series)
        ax.set_title(f"{eid} | {y_series} vs {x_series}\nF_ext={F_ext}, type={force_type}")
        ax.legend()
        fig.tight_layout()

        # 保存单个实验图
        img_path = f"{output_dir}/{eid}_inspect_relationships.png"
        fig.savefig(img_path, dpi=100)
        plt.close(fig)
        figures.append(img_path)

        # 合并图添加
        ax_all.scatter(x_clean, y_clean, color=colors[idx], label=eid, alpha=0.7)
        ax_all.plot(x_line, y_line, color=colors[idx], linestyle='--')

    # 合并图修饰
    ax_all.set_xlabel(x_series)
    ax_all.set_ylabel(y_series)
    ax_all.set_title(f"All experiments: {y_series} vs {x_series}")
    ax_all.legend()
    all_fig_path = f"{output_dir}/all_inspect_relationships.png"
    fig_all.tight_layout()
    fig_all.savefig(all_fig_path, dpi=100)
    plt.close(fig_all)
    figures.append(all_fig_path)

    # 合并观察
    full_obs = "inspect_relationships 结果：\n" + "\n".join(observations)
    if missing_series:
        full_obs += "\n警告：以下实验缺少所需序列：" + "; ".join(missing_series)

    return {
        "observation": full_obs,
        "derived_series": [],   # 不产生新序列
        "figures": figures,
        "metrics": metrics
    }
