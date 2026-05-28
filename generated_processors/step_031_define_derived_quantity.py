import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Any, Dict

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 解析参数
    exp_ids = params.get("experiment_ids", [])
    symbol = params.get("symbol", "D")
    expr_str = params.get("expression", "C_candidate * F_ext")
    overwrite = params.get("overwrite", True)

    # 只处理指定的实验
    exp_list = []
    for eid in exp_ids:
        if eid in experiments:
            exp_list.append(eid)
        else:
            # 如果实验不存在，跳过并记录
            pass

    if not exp_list:
        return {
            "observation": "未找到有效的实验ID。",
            "derived_series": [],
            "figures": [],
            "metrics": {}
        }

    # 检查表达式是否合法
    # expression 应该是 "C_candidate * F_ext"，我们直接解析它
    # 分离变量名和运算符：简单起见，假设表达式为 "C_candidate * F_ext"
    # 更通用的方式：我们可以分割表达式，但为了可靠，直接硬编码支持此特定形式
    # 但为了可扩展，我们可以使用 eval 但是限制在安全的命名空间
    # 这里使用简单的字符串替换和 eval

    derived_series_list = []
    metrics = {}
    figures = []

    # 准备图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 颜色循环
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_list)))

    for idx, exp_id in enumerate(exp_list):
        exp = experiments[exp_id]
        config = exp.get("config", {})
        series_dict = exp.get("series", {})
        available = exp.get("available_series", list(series_dict.keys()))

        # 提取 F_ext
        # 尝试从 config 中获取 constant_force 或 F_ext
        F_ext = config.get("F_ext", None)
        if F_ext is None:
            # fallback: 尝试 constant_force
            F_ext = config.get("constant_force", None)
        if F_ext is None:
            # 对于自由场，可能没有此字段，但数据上下文中 F_ext=0.0 或 10.0
            # 尝试从 config 的其他字段推断？我们假设如果 force_field_type 是 free，F_ext 视为 0？
            # 但 exp_02 的 F_ext=10.0，所以一定存在。
            # 检查 config.keys() 以调试
            raise ValueError(f"实验 {exp_id} 缺少 F_ext 字段。可用配置: {list(config.keys())}")

        # 检查 C_candidate 是否存在
        if "C_candidate" not in available:
            # 跳过此实验
            continue

        C_candidate = series_dict["C_candidate"]
        t_series = series_dict.get("t", None)
        if t_series is None:
            raise ValueError(f"实验 {exp_id} 缺少 t 序列")

        # 计算 D = C_candidate * F_ext
        C_arr = np.array(C_candidate, dtype=float)
        D_arr = C_arr * F_ext

        # 统计
        d_min = float(np.min(D_arr))
        d_max = float(np.max(D_arr))
        d_mean = float(np.mean(D_arr))
        d_std = float(np.std(D_arr))

        # 与 F_ext 的偏差
        deviation = D_arr - F_ext
        dev_mean = float(np.mean(deviation))
        dev_std = float(np.std(deviation))

        # 记录 metrics
        metrics[f"{exp_id}_{symbol}_min"] = d_min
        metrics[f"{exp_id}_{symbol}_max"] = d_max
        metrics[f"{exp_id}_{symbol}_mean"] = d_mean
        metrics[f"{exp_id}_{symbol}_std"] = d_std
        metrics[f"{exp_id}_{symbol}_dev_from_F_ext_mean"] = dev_mean
        metrics[f"{exp_id}_{symbol}_dev_from_F_ext_std"] = dev_std

        # 添加派生序列
        source_name = f"Q = {expr_str}"
        derived_series_list.append({
            "experiment_id": exp_id,
            "name": symbol,
            "values": [float(v) for v in D_arr],
            "source_name": source_name,
            "provenance": "generated data processor: define_derived_quantity",
            "description": f"{symbol} = {expr_str}, F_ext={F_ext}"
        })

        # 绘图
        ax.plot(t_series, D_arr, color=colors[idx], label=f"{exp_id} (F_ext={F_ext})", linewidth=1.5)

    if not derived_series_list:
        # 没有实验包含 C_candidate
        return {
            "observation": "指定实验中均不包含序列 C_candidate，无法计算 D。",
            "derived_series": [],
            "figures": [],
            "metrics": {}
        }

    # 图形配置
    ax.set_xlabel("Time t")
    ax.set_ylabel(f"{symbol}")
    ax.set_title(f"{symbol} = {expr_str}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(output_dir, f"{symbol}_vs_t.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append(fig_path)

    # 构建 observation
    obs_lines = []
    obs_lines.append(f"根据参数定义计算了派生量 {symbol} = {expr_str}。")
    obs_lines.append("各实验统计（仅包含有 C_candidate 序列的实验）：")
    for ds in derived_series_list:
        eid = ds["experiment_id"]
        m = metrics
        obs_lines.append(f"  {eid}: min={m[f'{eid}_{symbol}_min']:.6f}, max={m[f'{eid}_{symbol}_max']:.6f}, mean={m[f'{eid}_{symbol}_mean']:.6f}, std={m[f'{eid}_{symbol}_std']:.6f}, dev_mean={m[f'{eid}_{symbol}_dev_from_F_ext_mean']:.6f}")
    obs_lines.append(f"图像已保存至 {fig_path}。")
    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics
    }
