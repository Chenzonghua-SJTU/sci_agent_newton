import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any, List

def process(payload: dict) -> dict:
    action = payload.get("action", "define_derived_quantity")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    exp_ids = params.get("experiment_ids", list(experiments.keys()))
    symbol = params.get("symbol", "check")
    expression = params.get("expression", "")
    overwrite = params.get("overwrite", False)
    description = params.get("description", "")

    # ---------- 准备安全的数学环境 ----------
    safe_math = {
        "square": np.square,
        "cube": lambda x: np.power(x, 3),
        "sqrt": np.sqrt,
        "log": np.log,
        "exp": np.exp,
        "sin": np.sin,
        "cos": np.cos,
        "abs": np.abs,
        "pi": math.pi,
        "e": math.e
    }
    # 替换 ^ 为 **
    expression_safe = expression.replace("^", "**")

    derived_series = []
    metrics = {}
    figures = []

    # 用于综合图
    all_t = []
    all_check = []
    all_labels = []

    for eid in exp_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        series = exp.get("series", {})
        available = exp.get("available_series", list(series.keys()))

        # 检查必要的序列是否存在
        needed_vars = set()
        # 从表达式解析变量（粗略：只取合法标识符，排除函数名和数字）
        import re
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expression)
        for token in tokens:
            if token not in safe_math and token not in ['F_ext']:
                needed_vars.add(token)
        # 实际需要 a_sg 和 v_sg
        if 'a_sg' not in available or 'v_sg' not in available:
            raise ValueError(f"实验 {eid} 缺少 a_sg 或 v_sg，无法计算 check")

        t = np.array(series["t"])
        a_sg = np.array(series["a_sg"])
        v_sg = np.array(series["v_sg"])
        if not (len(t) == len(a_sg) == len(v_sg)):
            raise ValueError(f"实验 {eid} 的时间、速度、加速度长度不一致")

        # 构造 eval 局部环境
        local_vars = {"a_sg": a_sg, "v_sg": v_sg}
        # 如果有 F_ext，也加入（用于检查）
        config = exp.get("config", {})
        if "F_ext" in config:
            local_vars["F_ext"] = config["F_ext"]

        # 添加数学函数
        local_vars.update(safe_math)

        try:
            check_values = eval(expression_safe, {"__builtins__": {}}, local_vars)
        except Exception as e:
            raise ValueError(f"实验 {eid} 计算表达式 '{expression_safe}' 失败: {e}")

        # 确保是 numpy array
        check_values = np.asarray(check_values, dtype=float)
        if len(check_values) != len(t):
            raise ValueError(f"实验 {eid} 计算结果长度 {len(check_values)} 与时间长度 {len(t)} 不一致")

        # 度量统计
        mean_val = float(np.mean(check_values))
        std_val = float(np.std(check_values))
        min_val = float(np.min(check_values))
        max_val = float(np.max(check_values))
        start_val = float(check_values[0])
        end_val = float(check_values[-1])

        # 与 F_ext 对比
        F_ext = config.get("F_ext", None)
        if F_ext is not None:
            dev = mean_val - F_ext
            abs_dev = abs(mean_val - F_ext)
        else:
            dev = None
            abs_dev = None

        # 记录 metrics
        metrics[f"{eid}_{symbol}_mean"] = mean_val
        metrics[f"{eid}_{symbol}_std"] = std_val
        metrics[f"{eid}_{symbol}_min"] = min_val
        metrics[f"{eid}_{symbol}_max"] = max_val
        metrics[f"{eid}_{symbol}_start"] = start_val
        metrics[f"{eid}_{symbol}_end"] = end_val
        if dev is not None:
            metrics[f"{eid}_{symbol}_deviation"] = dev
            metrics[f"{eid}_{symbol}_abs_deviation"] = abs_dev

        # 生成图片
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t, check_values, 'b-', linewidth=1.5)
        if F_ext is not None:
            ax.axhline(y=F_ext, color='r', linestyle='--', label=f'F_ext={F_ext}')
        ax.set_xlabel("t")
        ax.set_ylabel(f"{symbol}")
        ax.set_title(f"Experiment {eid}: {symbol} = {expression_safe}")
        ax.legend()
        ax.grid(True)
        fig_path = os.path.join(output_dir, f"{eid}_{symbol}_vs_t.png")
        fig.savefig(fig_path, dpi=100)
        plt.close(fig)
        figures.append(fig_path)

        # 准备综合图数据
        all_t.append(t)
        all_check.append(check_values)
        all_labels.append(eid)

        # 记录 derived_series
        derived_series.append({
            "experiment_id": eid,
            "name": symbol,
            "values": check_values.tolist(),
            "source_name": f"defined as {expression_safe}",
            "provenance": "generated data processor: define_derived_quantity",
            "description": description
        })

    # 综合图
    if len(all_t) > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for i, eid in enumerate(all_labels):
            ax2.plot(all_t[i], all_check[i], label=eid, linewidth=1.2)
        ax2.set_xlabel("t")
        ax2.set_ylabel(f"{symbol}")
        ax2.set_title(f"All experiments: {symbol} vs t")
        ax2.legend()
        ax2.grid(True)
        comb_path = os.path.join(output_dir, f"{symbol}_vs_time_all.png")
        fig2.savefig(comb_path, dpi=100)
        plt.close(fig2)
        figures.append(comb_path)

    # 构造观察文本
    obs_lines = []
    obs_lines.append(f"定义派生量 {symbol} = {expression_safe}，应用于 {len(exp_ids)} 个实验。")
    for eid in exp_ids:
        if eid not in metrics:
            continue
        mean_val = metrics.get(f"{eid}_{symbol}_mean", None)
        std_val = metrics.get(f"{eid}_{symbol}_std", None)
        dev_val = metrics.get(f"{eid}_{symbol}_deviation", None)
        abs_dev_val = metrics.get(f"{eid}_{symbol}_abs_deviation", None)
        F_ext = experiments[eid]["config"].get("F_ext", None)
        if mean_val is not None:
            line = f"实验 {eid} (F_ext={F_ext}): {symbol} 均值={mean_val:.6e}, 标准差={std_val:.6e}"
            if dev_val is not None:
                line += f", 与F_ext偏差={dev_val:.6e} (绝对偏差={abs_dev_val:.6e})"
            obs_lines.append(line)
    observation = "\n".join(obs_lines)

    result = {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
    return result
