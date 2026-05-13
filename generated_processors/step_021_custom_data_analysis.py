import os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from collections import OrderedDict

def process(payload: dict) -> dict:
    """
    验证候选定律 a = F_ext / (1 + v^2)。对于所有实验，计算派生量 check = a_sg * (1 + v_sg^2)，
    并报告每个实验的 check 是否接近常数 F_ext。输出每个实验的 check 均值、标准差、与 F_ext 的偏差，
    并绘制 check 随时间的变化图。
    """
    action = payload.get("action", "custom_data_analysis")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")
    experiment_ids = params.get("experiment_ids", list(experiments.keys()))
    if not experiment_ids:
        # 如果没有指定，处理所有实验
        experiment_ids = list(experiments.keys())

    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 存储结果
    derived_series = []
    metrics = {}
    figures = []
    observation_lines = []

    # 辅助函数：从实验配置获取 F_ext
    def get_F_ext(exp):
        config = exp.get("config", {})
        # 尝试多种可能的字段
        F_ext = config.get("F_ext", None)
        if F_ext is None:
            F_ext = config.get("constant_force", None)
        if F_ext is None:
            # 根据 force_field_type 推断
            ftype = config.get("force_field_type", "")
            if ftype == "free":
                F_ext = 0.0
            else:
                # 默认当作未知，可能为 None
                F_ext = 0.0  # fallback
        return float(F_ext)

    # 辅助函数：获取或生成 a_sg 和 v_sg 序列
    def get_sg_series(exp, exp_id):
        series = exp.get("series", {})
        available = exp.get("available_series", [])
        t = np.array(series.get("t", []))
        q = np.array(series.get("q", []))
        # 优先使用已有序列
        a_sg = np.array(series.get("a_sg", []))
        v_sg = np.array(series.get("v_sg", []))
        if len(a_sg) == 0 or len(v_sg) == 0:
            # 尝试从 q 用 savgol 滤波生成
            if len(q) < 11:
                raise ValueError(f"实验 {exp_id}: q 序列长度不足 (len={len(q)})，无法使用 Savgol 滤波。")
            # 确保 q 长度足够且至少有 3 个点
            window_length = min(11, len(q) if len(q) % 2 == 1 else len(q) - 1)
            if window_length < 5:
                window_length = 5 if len(q) >= 5 else len(q)
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(3, window_length - 1)
            dt = t[1] - t[0] if len(t) > 1 else 0.1
            # 平滑位置
            q_smooth = savgol_filter(q, window_length, polyorder, mode='interp')
            # 速度 (一阶导数)
            v_sg = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt, mode='interp')
            # 加速度 (二阶导数)
            a_sg = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt, mode='interp')
            # 保存为新序列返回（注意不要覆盖原序列，而是返回派生序列）
            # 如果原实验没有 a_sg/v_sg，我们生成的应被返回
        return t, q, a_sg, v_sg

    # 收集每个实验的 check 用于整体图
    all_checks = {}
    all_times = {}

    for exp_id in experiment_ids:
        if exp_id not in experiments:
            observation_lines.append(f"实验 {exp_id} 不在数据中，跳过。")
            continue
        exp = experiments[exp_id]
        config = exp.get("config", {})
        F_ext = get_F_ext(exp)
        try:
            t, q, a_sg, v_sg = get_sg_series(exp, exp_id)
        except ValueError as e:
            observation_lines.append(f"实验 {exp_id}: {e}")
            continue

        # 计算 check
        check = np.array(a_sg) * (1 + np.array(v_sg)**2)

        # 统计
        check_mean = np.mean(check)
        check_std = np.std(check, ddof=1)  # 样本标准差
        if abs(F_ext) > 1e-12:
            dev = (check_mean - F_ext) / abs(F_ext)
            dev_abs = check_mean - F_ext
        else:
            dev = check_mean  # 因为 F_ext=0，偏差即为均值
            dev_abs = check_mean

        # 记录 metrics
        metrics[f"{exp_id}_check_mean"] = float(check_mean)
        metrics[f"{exp_id}_check_std"] = float(check_std)
        metrics[f"{exp_id}_check_deviation"] = float(dev)
        metrics[f"{exp_id}_check_abs_deviation"] = float(dev_abs)
        metrics[f"{exp_id}_F_ext"] = float(F_ext)

        # 收集 check 用于整体图
        all_checks[exp_id] = check
        all_times[exp_id] = t

        # 返回派生序列
        derived_series.append({
            "experiment_id": exp_id,
            "name": "check",
            "values": check.tolist(),
            "source_name": "check = a_sg * (1 + v_sg^2)",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "验证候选定律 a = F_ext/(1+v^2) 的派生量"
        })

        # 判断 check 是否接近常数 F_ext: 如果标准差/|F_ext| 小于某个阈值？或者 check 均值和 F_ext 很接近？
        # 这里只给出观察，不判定结论。
        obs_line = (f"实验 {exp_id} (F_ext={F_ext}): check 均值={check_mean:.6f}, 标准差={check_std:.6f}, "
                    f"偏差={dev:.4e} (绝对偏差={dev_abs:.4e})")
        observation_lines.append(obs_line)

    # 绘图：每个实验的 check 随时间变化
    for exp_id in experiment_ids:
        if exp_id not in all_checks:
            continue
        t = all_times[exp_id]
        check = all_checks[exp_id]
        F_ext = metrics.get(f"{exp_id}_F_ext", 0)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t, check, 'b-', label='check')
        ax.axhline(y=F_ext, color='r', linestyle='--', label=f'F_ext={F_ext}')
        ax.set_xlabel('Time')
        ax.set_ylabel('check = a_sg * (1+v_sg^2)')
        ax.set_title(f'Experiment {exp_id}: check vs time')
        ax.legend()
        ax.grid(True)
        fname = f"check_vs_time_{exp_id}.png"
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=100)
        plt.close(fig)
        figures.append(path)

    # 可选：将所有实验的 check 放在一个图中比较（叠加）
    if len(all_checks) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        for exp_id in experiment_ids:
            if exp_id not in all_checks:
                continue
            t = all_times[exp_id]
            check = all_checks[exp_id]
            F_ext = metrics.get(f"{exp_id}_F_ext", 0)
            ax.plot(t, check, label=f'{exp_id} (F_ext={F_ext})')
        ax.set_xlabel('Time')
        ax.set_ylabel('check')
        ax.set_title('check vs time for all experiments')
        ax.legend()
        ax.grid(True)
        fname = "check_vs_time_all.png"
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=100)
        plt.close(fig)
        figures.append(path)

    # 构建 observation
    obs_header = f"自定义数据分析：验证候选定律 a = F_ext / (1 + v^2)。\n对实验 {experiment_ids} 计算派生量 check = a_sg * (1 + v_sg^2) 并统计。\n"
    obs_body = "\n".join(observation_lines)
    obs_summary = f"全部实验的 check 均值、标准差、与 F_ext 的偏差已记录在 metrics 中。图像已保存至 {output_dir}。"
    observation = obs_header + obs_body + "\n" + obs_summary

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
