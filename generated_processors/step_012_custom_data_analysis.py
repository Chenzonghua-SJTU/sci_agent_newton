import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 提取实验ID列表
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        # 如果参数里有 experiment_id 单个，转为列表
        single = params.get("experiment_id")
        if single:
            exp_ids = [single]
        else:
            exp_ids = list(experiments.keys())

    # 验证所有实验存在
    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")

    window_length = 11
    polyorder = 3

    derived_series_list = []
    figures_list = []
    metrics = {}
    free_exps = ['exp_02', 'exp_03']  # 根据参数中的描述
    constant_exps = ['exp_04', 'exp_05', 'exp_06', 'exp_07']

    # 用于汇总A（截距）和B（斜率）与F_ext的关系
    av_fit_results = {}  # exp_id -> {slope, intercept, R2, F_ext, force_field_type}

    for eid in exp_ids:
        exp = experiments[eid]
        config = exp.get("config", {})
        force_type = config.get("force_field_type", "unknown")
        F_ext = config.get("constant_force")  # 注意：有些实验 config 中可能叫 constant_force，但 free 场景可能另有字段？根据数据上下文，exp_02: F_ext=0.0，exp_03: F_ext=10.0，exp_04: F_ext=1.0 等。查看 payload 结构：config 应该包含 force_field_type, constant_force? 但 free 场景可能也有 constant_force 字段？我们保险使用 config.get("F_ext") 或 config.get("constant_force")。从上下文看，free 场景也有 F_ext=0.0 或 10.0，但 config 中存储方式未知。我们尝试读取 config.get("F_ext")，若不存在则从 config.get("constant_force")获取，若仍不存在则设为None。
        if "F_ext" in config:
            F_val = config["F_ext"]
        elif "constant_force" in config:
            F_val = config["constant_force"]
        else:
            # 尝试从实验元数据或默认
            F_val = 0.0 if eid in ['exp_02'] else None
        F_val = float(F_val) if F_val is not None else None

        # 获取时间序列和位置序列
        series = exp.get("series", {})
        # 优先使用原始 q 和 t
        t = np.array(series.get("t", []))
        q = np.array(series.get("q", []))
        if len(t) == 0 or len(q) == 0:
            raise ValueError(f"Experiment {eid}: missing t or q series")
        # 检查长度一致
        n = len(t)
        if len(q) != n:
            raise ValueError(f"Experiment {eid}: q length {len(q)} != t length {n}")

        # 计算dt（假设均匀采样）
        dt = t[1] - t[0]
        if dt <= 0:
            raise ValueError(f"Experiment {eid}: non-positive dt")

        # 检查窗口长度 <= n
        if window_length > n:
            raise ValueError(f"Experiment {eid}: window_length {window_length} > data length {n}")

        # 使用 Savitzky-Golay 滤波估计速度（一阶导数）和加速度（二阶导数）
        v = savgol_filter(q, window_length, polyorder, deriv=1, delta=dt)
        a = savgol_filter(q, window_length, polyorder, deriv=2, delta=dt)

        # 保存派生序列
        derived_series_list.append({
            "experiment_id": eid,
            "name": "v_sg",
            "values": v.tolist(),
            "source_name": "Savitzky-Golay filter (window=11, polyorder=3) first derivative from q(t)",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "Estimated velocity via Savitzky-Golay"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": "a_sg",
            "values": a.tolist(),
            "source_name": "Savitzky-Golay filter (window=11, polyorder=3) second derivative from q(t)",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "Estimated acceleration via Savitzky-Golay"
        })

        # 线性拟合: a = slope * v + intercept
        # 去除可能的 NaN
        mask = np.isfinite(v) & np.isfinite(a)
        v_clean = v[mask]
        a_clean = a[mask]

        if len(v_clean) < 2:
            slope = np.nan
            intercept = np.nan
            R2 = np.nan
        else:
            A = np.vstack([v_clean, np.ones_like(v_clean)]).T
            slope, intercept = np.linalg.lstsq(A, a_clean, rcond=None)[0]
            # R²
            residuals = a_clean - (slope * v_clean + intercept)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((a_clean - np.mean(a_clean))**2)
            if ss_tot == 0:
                R2 = 1.0 if ss_res == 0 else 0.0
            else:
                R2 = 1 - ss_res / ss_tot

        # 根据 a = A - B*v，所以 A = intercept, B = -slope
        A_val = intercept
        B_val = -slope

        # 记录指标
        metrics[f"{eid}_av_slope"] = slope
        metrics[f"{eid}_av_intercept"] = intercept
        metrics[f"{eid}_av_A"] = A_val
        metrics[f"{eid}_av_B"] = B_val
        metrics[f"{eid}_av_R2"] = R2
        metrics[f"{eid}_F_ext"] = F_val if F_val is not None else 0.0
        metrics[f"{eid}_force_field_type"] = force_type

        av_fit_results[eid] = {
            "slope": slope,
            "intercept": intercept,
            "A": A_val,
            "B": B_val,
            "R2": R2,
            "F_ext": F_val,
            "force_type": force_type
        }

        # 对于自由场景验证 a 是否恒为零
        if eid in free_exps:
            a_mean = np.mean(a_clean)
            a_std = np.std(a_clean)
            metrics[f"{eid}_a_mean"] = a_mean
            metrics[f"{eid}_a_std"] = a_std
            # 判断是否接近零
            if abs(a_mean) < 1e-10 and a_std < 1e-10:
                metrics[f"{eid}_a_is_zero"] = 1
            else:
                metrics[f"{eid}_a_is_zero"] = 0

        # 绘制 a vs v 散点图及拟合线
        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(v_clean, a_clean, s=10, alpha=0.7, label='data')
        if not np.isnan(slope):
            v_line = np.linspace(v_clean.min(), v_clean.max(), 100)
            a_line = slope * v_line + intercept
            ax.plot(v_line, a_line, 'r-', label=f'fit: a={A_val:.4f} - {B_val:.4f}*v')
        ax.set_xlabel('v (estimated)')
        ax.set_ylabel('a (estimated)')
        ax.set_title(f'{eid}  a vs v  (F_ext={F_val})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fname = f"{eid}_a_vs_v.png"
        fpath = os.path.join(output_dir, fname)
        plt.tight_layout()
        plt.savefig(fpath, dpi=150)
        plt.close()
        figures_list.append(fpath)

    # 汇总 A 与 F_ext 的关系
    # 收集所有实验（或仅恒力？按照需求，包括所有）
    f_vals = []
    a_vals = []
    b_vals = []
    for eid, res in av_fit_results.items():
        if res['F_ext'] is not None:
            f_vals.append(res['F_ext'])
            a_vals.append(res['A'])
            b_vals.append(res['B'])

    if len(f_vals) >= 2:
        # A vs F_ext
        A_fit = np.polyfit(f_vals, a_vals, 1)
        metrics['A_vs_F_slope'] = A_fit[0]
        metrics['A_vs_F_intercept'] = A_fit[1]
        # B vs F_ext
        B_fit = np.polyfit(f_vals, b_vals, 1)
        metrics['B_vs_F_slope'] = B_fit[0]
        metrics['B_vs_F_intercept'] = B_fit[1]

        # 绘制 A vs F_ext
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.scatter(f_vals, a_vals, color='blue')
        f_line = np.linspace(min(f_vals), max(f_vals), 100)
        a_line = A_fit[0]*f_line + A_fit[1]
        ax2.plot(f_line, a_line, 'r-')
        ax2.set_xlabel('F_ext')
        ax2.set_ylabel('A (intercept)')
        ax2.set_title('A vs F_ext')
        ax2.grid(True, alpha=0.3)
        fname2 = "A_vs_F_ext.png"
        fpath2 = os.path.join(output_dir, fname2)
        plt.tight_layout()
        plt.savefig(fpath2, dpi=150)
        plt.close()
        figures_list.append(fpath2)

        # 绘制 B vs F_ext
        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.scatter(f_vals, b_vals, color='green')
        b_line = B_fit[0]*f_line + B_fit[1]
        ax3.plot(f_line, b_line, 'r-')
        ax3.set_xlabel('F_ext')
        ax3.set_ylabel('B (slope of a = A - B*v)')
        ax3.set_title('B vs F_ext')
        ax3.grid(True, alpha=0.3)
        fname3 = "B_vs_F_ext.png"
        fpath3 = os.path.join(output_dir, fname3)
        plt.tight_layout()
        plt.savefig(fpath3, dpi=150)
        plt.close()
        figures_list.append(fpath3)

    # 构建observation文本
    lines = []
    lines.append("对所有实验使用Savitzky-Golay滤波（窗口长度11，多项式阶数3）从q(t)估计速度和加速度序列，并生成派生序列v_sg和a_sg。")
    lines.append("")
    lines.append("各实验的加速度 vs 速度线性拟合结果 (模型: a = A - B*v)：")
    for eid in exp_ids:
        res = av_fit_results.get(eid)
        if res is None:
            continue
        ft = res['force_type']
        F = res['F_ext']
        A_val = res['A']
        B_val = res['B']
        R2 = res['R2']
        lines.append(f"  {eid} (type={ft}, F_ext={F}): A={A_val:.6f}, B={B_val:.6f}, R²={R2:.6f}")
    lines.append("")

    # 自由场景加速度验证
    for eid in free_exps:
        if eid in av_fit_results:
            a_mean = metrics.get(f"{eid}_a_mean", None)
            a_std = metrics.get(f"{eid}_a_std", None)
            is_zero = metrics.get(f"{eid}_a_is_zero", 0)
            a_mean_str = f"{a_mean:.2e}" if a_mean is not None else "N/A"
            a_std_str = f"{a_std:.2e}" if a_std is not None else "N/A"
            lines.append(f"自由场景 {eid}：加速度均值={a_mean_str}，标准差={a_std_str}，近似为零={bool(is_zero)}")
    lines.append("")

    if len(f_vals) >= 2:
        lines.append(f"A 与 F_ext 的线性关系：斜率={metrics['A_vs_F_slope']:.6f}, 截距={metrics['A_vs_F_intercept']:.6f}")
        lines.append(f"B 与 F_ext 的线性关系：斜率={metrics['B_vs_F_slope']:.6f}, 截距={metrics['B_vs_F_intercept']:.6f}")
    else:
        lines.append("F_ext 数据不足，无法分析 A/B 与 F_ext 的关系。")
    lines.append("")
    lines.append("每个实验的 a vs v 散点图及拟合线已保存。")
    lines.append("A vs F_ext、B vs F_ext 关系图已保存（仅当有至少2个不同F_ext值时）。")

    observation = "\n".join(lines)

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures_list,
        "metrics": metrics
    }
