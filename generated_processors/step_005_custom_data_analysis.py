import os
import math
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    # 解析参数
    params = payload.get("parameters", {})
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(payload.get("experiments", {}).keys())
    
    output_dir = payload.get("output_dir", ".")
    
    derived_series = []
    figures = []
    metrics = {}
    observations = []
    
    for eid in exp_ids:
        exp = payload["experiments"].get(eid)
        if exp is None:
            raise ValueError(f"Experiment {eid} not found in payload")
        
        config = exp.get("config", {})
        series = exp.get("series", {})
        
        # 只处理指定的实验 ID 和名称匹配（exp_03）
        if eid != "exp_03":
            continue
        
        # 提取 t 和 q
        t = np.array(series.get("t"))
        q = np.array(series.get("q"))
        if t is None or q is None:
            raise ValueError(f"Experiment {eid} missing t or q series")
        
        # 检查长度一致性
        n = len(t)
        if len(q) != n:
            raise ValueError(f"Experiment {eid} t and q length mismatch")
        
        # 采样间隔
        dt = np.mean(np.diff(t)) if n > 1 else 1.0
        
        # 使用 Savitzky-Golay 滤波估计速度和加速度
        window_length = 9
        polyorder = 2
        # 窗口长度不能超过数据长度
        if n < window_length:
            raise ValueError(f"Experiment {eid} has insufficient points ({n}) for window {window_length}")
        
        v_sg = savgol_filter(q, window_length=window_length, polyorder=polyorder, deriv=1, delta=dt)
        a_sg = savgol_filter(q, window_length=window_length, polyorder=polyorder, deriv=2, delta=dt)
        
        # 计算加速度统计
        a_mean = float(np.mean(a_sg))
        a_std = float(np.std(a_sg))
        v_mean = float(np.mean(v_sg))
        v_std = float(np.std(v_sg))
        
        # 加速度是否近似常数：标准偏差相对于均值很小？这里只输出值，由 LLM 判断
        # 记录到 metrics
        
        # 二次多项式拟合：q = a*t^2 + b*t + c
        coeffs = np.polyfit(t, q, 2)  # [a, b, c] 从高次到低次
        a_fit = coeffs[0]
        b_fit = coeffs[1]
        c_fit = coeffs[2]
        
        # 拟合值和残差
        q_fit = np.polyval(coeffs, t)
        residuals = q - q_fit
        rmse = math.sqrt(np.mean(residuals**2))
        
        # 检查 b 和 c 是否接近 0，a 是否接近 0.5（数值误差）
        # 记录值，由 LLM 判断
        
        # 外部力
        F_ext = config.get("constant_force", config.get("F_ext", None))
        if F_ext is None:
            # 尝试从 config 中查找其他字段
            F_ext = config.get("force", None)
        if F_ext is None:
            F_ext = 1.0  # 默认值，但一般会有
        
        # 加速度均值与 F_ext 比较
        acc_compared_to_F = a_mean - float(F_ext)
        
        # 构建 observation 用字符串
        obs_lines = []
        obs_lines.append(f"实验 {eid} (力场类型={config.get('force_field_type','?')}, F_ext={F_ext}) 分析结果：")
        obs_lines.append(f"  位置 q(t) 范围: [{float(np.min(q)):.6f}, {float(np.max(q)):.6f}]")
        obs_lines.append(f"  速度 v_sg 均值={v_mean:.15f}, 标准差={v_std:.15e}")
        obs_lines.append(f"  加速度 a_sg 均值={a_mean:.15f}, 标准差={a_std:.15e}")
        obs_lines.append(f"  加速度标准差较小: {'是' if a_std < 1e-10 else '否'}")
        obs_lines.append(f"  二次多项式拟合 q = a*t^2 + b*t + c:")
        obs_lines.append(f"    a={a_fit:.15f}, b={b_fit:.15f}, c={c_fit:.15f}")
        obs_lines.append(f"    残差 RMSE = {rmse:.15e}")
        obs_lines.append(f"  加速度均值与外力 F_ext 的差异 Δa = a_mean - F_ext = {acc_compared_to_F:.15f}")
        obs_lines.append(f"  若加速度接近常数，则 Δa 反映质量倒数（若 F=ma）")
        observations.append("\n".join(obs_lines))
        
        # 记录 metrics
        metrics[f"{eid}_t_len"] = n
        metrics[f"{eid}_v_mean"] = v_mean
        metrics[f"{eid}_v_std"] = v_std
        metrics[f"{eid}_a_mean"] = a_mean
        metrics[f"{eid}_a_std"] = a_std
        metrics[f"{eid}_a_approx_constant"] = 1 if a_std < 1e-10 else 0
        metrics[f"{eid}_quadratic_a"] = float(a_fit)
        metrics[f"{eid}_quadratic_b"] = float(b_fit)
        metrics[f"{eid}_quadratic_c"] = float(c_fit)
        metrics[f"{eid}_quadratic_rmse"] = rmse
        metrics[f"{eid}_F_ext"] = float(F_ext)
        metrics[f"{eid}_delta_a_F"] = acc_compared_to_F
        
        # 创建加速度派生序列
        derived_series.append({
            "experiment_id": eid,
            "name": "acceleration",
            "values": a_sg.tolist(),
            "source_name": "Savitzky-Golay二阶导数 (窗口9, 阶2)",
            "provenance": "generated data processor: custom_data_analysis for exp_03",
            "description": "从 q 通过 Savitzky-Golay 滤波估计的加速度序列"
        })
        
        # 绘制图
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t, q, 'b-', label='q(t)')
        axes[0].plot(t, q_fit, 'r--', label=f'二次拟合: a={a_fit:.4f}, b={b_fit:.4f}, c={c_fit:.4f}')
        axes[0].set_ylabel('q')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(t, v_sg, 'g-', label='v_sg')
        axes[1].set_ylabel('v')
        axes[1].legend()
        axes[1].grid(True)
        
        axes[2].plot(t, a_sg, 'm-', label='a_sg')
        axes[2].axhline(y=a_mean, color='k', linestyle='--', alpha=0.5, label=f'a_mean={a_mean:.4f}')
        axes[2].set_xlabel('t')
        axes[2].set_ylabel('a')
        axes[2].legend()
        axes[2].grid(True)
        
        fig.suptitle(f'Kinematics Analysis for {eid} (F_ext={F_ext})')
        fig.tight_layout()
        
        fig_path = os.path.join(output_dir, f"{eid}_kinematics_quadratic.png")
        fig.savefig(fig_path, dpi=100)
        plt.close(fig)
        figures.append(fig_path)
        
        # 图也保存为另一张备用（可选）
    
    # 将多个实验的观察合并
    if not observations:
        observations.append("没有匹配的实验进行处理。")
    
    # 如果只处理了一个实验，observation 可以精简一些
    observation_str = "\n".join(observations)
    
    return {
        "observation": observation_str,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
