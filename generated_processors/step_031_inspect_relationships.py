import json
import math
import statistics
import itertools
import functools
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn import linear_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process(payload: dict) -> dict:
    action = payload.get("action", "")
    parameters = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    experiment_ids = parameters.get("experiment_ids", [])
    if not experiment_ids:
        experiment_ids = list(experiments.keys())

    x_series = parameters.get("x_series", "v_sg")
    y_series = parameters.get("y_series", "a_sg")

    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True, exist_ok=True)

    # 结果收集
    observations_lines = []
    metrics = {}
    figure_paths = []

    # 对每个实验进行分析
    for eid in experiment_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        series = exp.get("series", {})
        config = exp.get("config", {})
        F_ext = config.get("F_ext", None)

        if x_series not in series:
            raise ValueError(f"Experiment {eid}: missing required series '{x_series}'")
        if y_series not in series:
            raise ValueError(f"Experiment {eid}: missing required series '{y_series}'")

        x = np.array(series[x_series], dtype=float)
        y = np.array(series[y_series], dtype=float)
        if len(x) != len(y):
            raise ValueError(f"Experiment {eid}: series length mismatch: x={len(x)}, y={len(y)}")

        # 线性回归
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
        r2 = r_value ** 2
        n = len(x)
        # 斜率置信区间（95%）
        t_val = scipy_stats.t.ppf(0.975, n - 2)
        slope_ci_low = slope - t_val * std_err
        slope_ci_high = slope + t_val * std_err
        # 截距置信区间
        # 计算截距标准误
        x_mean = np.mean(x)
        sxx = np.sum((x - x_mean) ** 2)
        if sxx == 0:
            intercept_se = 0.0
        else:
            intercept_se = std_err * np.sqrt(1.0 / n + x_mean ** 2 / sxx)
        intercept_ci_low = intercept - t_val * intercept_se
        intercept_ci_high = intercept + t_val * intercept_se

        # 存储 metrics
        prefix = eid + "_"
        metrics[prefix + "corr"] = r_value
        metrics[prefix + "corr_p"] = p_value
        metrics[prefix + "slope"] = slope
        metrics[prefix + "intercept"] = intercept
        metrics[prefix + "R2"] = r2
        metrics[prefix + "slope_ci_low"] = slope_ci_low
        metrics[prefix + "slope_ci_high"] = slope_ci_high
        metrics[prefix + "intercept_ci_low"] = intercept_ci_low
        metrics[prefix + "intercept_ci_high"] = intercept_ci_high

        # 构造观察字符串
        obs_line = (f"{eid}: y={y_series} vs x={x_series}, 拟合: {y_series} = {slope:.4f}*{x_series} + ({intercept:.4f}), "
                    f"R²={r2:.4f}, 相关系数={r_value:.4f} (p={p_value:.2e}), "
                    f"slope CI95=[{slope_ci_low:.4f}, {slope_ci_high:.4f}], "
                    f"intercept CI95=[{intercept_ci_low:.4f}, {intercept_ci_high:.4f}]")
        observations_lines.append(obs_line)

        # 绘制单个实验图
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, y, alpha=0.6, label='data')
        x_plot = np.linspace(x.min(), x.max(), 100)
        y_plot = slope * x_plot + intercept
        ax.plot(x_plot, y_plot, 'r-', label=f'linear fit (R²={r2:.3f})')
        ax.set_xlabel(x_series)
        ax.set_ylabel(y_series)
        ax.set_title(f"{eid}: {y_series} vs {x_series}")
        ax.legend()
        fig.tight_layout()
        fname = f"{eid}_inspect_relationships.png"
        fpath = output_dir_path / fname
        fig.savefig(str(fpath), dpi=100)
        plt.close(fig)
        figure_paths.append(str(fpath))

    # 绘制总览图（如果只有一个实验则跳过）
    if len(experiment_ids) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.tab10(range(len(experiment_ids)))
        for idx, eid in enumerate(experiment_ids):
            if eid not in experiments:
                continue
            exp = experiments[eid]
            series = exp.get("series", {})
            if x_series not in series or y_series not in series:
                continue
            x = np.array(series[x_series], dtype=float)
            y = np.array(series[y_series], dtype=float)
            slope = metrics.get(eid + "_slope")
            intercept = metrics.get(eid + "_intercept")
            r2 = metrics.get(eid + "_R2")
            label = f"{eid} (R²={r2:.3f})" if r2 is not None else eid
            ax.scatter(x, y, color=colors[idx], alpha=0.6, label=label)
            if slope is not None and intercept is not None:
                x_plot = np.linspace(x.min(), x.max(), 100)
                y_plot = slope * x_plot + intercept
                ax.plot(x_plot, y_plot, color=colors[idx], linestyle='--', lw=1)
        ax.set_xlabel(x_series)
        ax.set_ylabel(y_series)
        ax.set_title(f"All experiments: {y_series} vs {x_series}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fname_all = "all_inspect_relationships.png"
        fpath_all = output_dir_path / fname_all
        fig.savefig(str(fpath_all), dpi=100)
        plt.close(fig)
        figure_paths.append(str(fpath_all))

    observation = "inspect_relationships 结果：\n" + "\n".join(observations_lines)

    return {
        "observation": observation,
        "derived_series": [],  # 此 action 不产生派生序列
        "figures": figure_paths,
        "metrics": metrics
    }
