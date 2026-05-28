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
import scipy.signal
import scipy.stats
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = Path(payload["output_dir"])

    # ---------- 参数解析 ----------
    experiment_ids: List[str] = params.get("experiment_ids", [])
    source_series: str = params.get("source_series", "q")
    position_name: str = params.get("position_name", "q_new")
    velocity_name: str = params.get("velocity_name", "v_new")
    acceleration_name: str = params.get("acceleration_name", "a_new")
    window_length: int = params.get("window_length", 11)
    polyorder: int = params.get("polyorder", 3)
    overwrite: bool = params.get("overwrite", False)

    # 窗口长度必须是奇数，且不能超过数据长度
    if window_length % 2 == 0:
        window_length += 1  # 转换为奇数
    if window_length <= polyorder:
        raise ValueError(f"window_length ({window_length}) must be > polyorder ({polyorder})")

    # ---------- 数据处理 ----------
    derived_series_list = []
    figures_list = []
    metrics = {}

    for exp_id in experiment_ids:
        if exp_id not in experiments:
            raise ValueError(f"Experiment {exp_id} not found in payload")
        exp = experiments[exp_id]
        config = exp.get("config", {})
        series = exp.get("series", {})
        available = exp.get("available_series", list(series.keys()))

        # 检查目标序列是否已存在
        target_names = [position_name, velocity_name, acceleration_name]
        if not overwrite:
            existing = [n for n in target_names if n in available]
            if existing:
                # 跳过该实验，记录信息
                metrics[f"{exp_id}_skipped"] = True
                metrics[f"{exp_id}_existing_series"] = existing
                continue

        # 获取源序列和时间
        if source_series not in series:
            raise ValueError(f"Experiment {exp_id}: source series '{source_series}' not found")
        q = np.array(series[source_series], dtype=float)
        t = np.array(series.get("t", None), dtype=float)
        if t is None or len(t) != len(q):
            # 尝试从config推断时间轴
            dt = config.get("dt", None)
            if dt is None:
                # 如果都没有，抛出错误
                raise ValueError(f"Experiment {exp_id}: cannot determine time axis (missing 't' series and config['dt'])")
            t = np.arange(len(q)) * dt
        else:
            dt = np.median(np.diff(t)) if len(t) > 1 else config.get("dt", 1.0)

        # 数据长度检查
        if len(q) < window_length:
            # 如果数据太短，简单使用差分
            # 但为了统一，扩展窗口？
            raise ValueError(f"Experiment {exp_id}: data length ({len(q)}) < window_length ({window_length})")

        # 应用 Savitzky-Golay 滤波器
        try:
            q_smooth = scipy.signal.savgol_filter(q, window_length, polyorder, deriv=0)
            v = scipy.signal.savgol_filter(q, window_length, polyorder, deriv=1, delta=dt)
            a = scipy.signal.savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)
        except Exception as e:
            raise ValueError(f"Experiment {exp_id}: Savitzky-Golay filter failed: {e}")

        # 转换为列表
        q_smooth_list = q_smooth.tolist()
        v_list = v.tolist()
        a_list = a.tolist()

        # 添加派生序列
        derived_series_list.append({
            "experiment_id": exp_id,
            "name": position_name,
            "values": q_smooth_list,
            "source_name": f"savgol_filter(q, window={window_length}, polyorder={polyorder}, deriv=0)",
            "provenance": f"generated data processor: {action}",
            "description": f"Savitzky-Golay smoothed position (window={window_length}, polyorder={polyorder})"
        })
        derived_series_list.append({
            "experiment_id": exp_id,
            "name": velocity_name,
            "values": v_list,
            "source_name": f"savgol_filter(q, window={window_length}, polyorder={polyorder}, deriv=1, delta={dt})",
            "provenance": f"generated data processor: {action}",
            "description": f"Savitzky-Golay estimated velocity (deriv=1)"
        })
        derived_series_list.append({
            "experiment_id": exp_id,
            "name": acceleration_name,
            "values": a_list,
            "source_name": f"savgol_filter(q, window={window_length}, polyorder={polyorder}, deriv=2, delta={dt})",
            "provenance": f"generated data processor: {action}",
            "description": f"Savitzky-Golay estimated acceleration (deriv=2)"
        })

        # 简单的统计 metrics
        metrics[f"{exp_id}_{position_name}_min"] = float(np.min(q_smooth))
        metrics[f"{exp_id}_{position_name}_max"] = float(np.max(q_smooth))
        metrics[f"{exp_id}_{position_name}_mean"] = float(np.mean(q_smooth))
        metrics[f"{exp_id}_{velocity_name}_min"] = float(np.min(v))
        metrics[f"{exp_id}_{velocity_name}_max"] = float(np.max(v))
        metrics[f"{exp_id}_{velocity_name}_mean"] = float(np.mean(v))
        metrics[f"{exp_id}_{acceleration_name}_min"] = float(np.min(a))
        metrics[f"{exp_id}_{acceleration_name}_max"] = float(np.max(a))
        metrics[f"{exp_id}_{acceleration_name}_mean"] = float(np.mean(a))

        # 绘图：每个实验的 q_smooth, v, a vs t
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(t, q, 'k-', alpha=0.4, label='raw q')
        axes[0].plot(t, q_smooth, 'b-', linewidth=2, label=f'{position_name}')
        axes[0].set_ylabel('Position')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, v, 'r-', linewidth=2, label=f'{velocity_name}')
        axes[1].set_ylabel('Velocity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, a, 'g-', linewidth=2, label=f'{acceleration_name}')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Acceleration')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f'{exp_id}: Kinematics Estimation (window={window_length}, polyorder={polyorder})')
        fname = f"{exp_id}_kinematics_estimation.png"
        fig_path = output_dir / fname
        fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        figures_list.append(str(fig_path))

    # ---------- 组装返回 ----------
    observation = f"为实验 {experiment_ids} 使用 Savitzky-Golay 滤波器 (window={window_length}, polyorder={polyorder}, dt={dt:.4f}) 从 '{source_series}' 估计了平滑位置 '{position_name}'、速度 '{velocity_name}'、加速度 '{acceleration_name}'。"
    if not overwrite:
        skipped = [k for k in metrics if k.endswith('_skipped') and metrics[k]]
        if skipped:
            observation += f" 由于 overwrite=False，跳过了已存在目标序列的实验: {[k.split('_')[0] for k in skipped]}。"
    observation += f" 已绘制每个实验的运动学估计图并保存。关键统计量见 metrics。"
    observation += f" 示例: exp_02: q_new 均值={metrics.get('exp_02_q_new_mean', 'N/A'):.4f}, v_new 均值={metrics.get('exp_02_v_new_mean', 'N/A'):.4f}, a_new 均值={metrics.get('exp_02_a_new_mean', 'N/A'):.4f}。"

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures_list,
        "metrics": metrics
    }
