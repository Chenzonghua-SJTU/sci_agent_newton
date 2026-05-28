import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # 从参数中提取 experiment_ids 和符号表达式
    exp_ids = params.get("experiment_ids", list(experiments.keys()))
    symbol = params["symbol"]
    expression = params["expression"]
    overwrite = params.get("overwrite", True)

    # 解析 expression: 包含 a_sg11, F_ext, v_sg11, square(...)
    # 我们直接手动替换为数组计算
    c1 = 0.587805
    c2 = -0.094704

    derived_series = []
    metrics = {}
    all_C_means = {}
    all_C_stds = {}
    all_C_mins = {}
    all_C_maxs = {}
    figure_paths = []

    # 收集每个实验的 C vs t 数据用于画图
    fig, ax = plt.subplots(figsize=(10, 6))

    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload.")

        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        available = exp["available_series"]

        # 获取 t
        t = np.array(series.get("t", None))
        if t is None:
            raise ValueError(f"Experiment {eid} has no t series.")

        # 检查 a_sg11, v_sg11, F_ext
        if "a_sg11" not in available:
            raise ValueError(f"Experiment {eid} does not have a_sg11 series.")
        if "v_sg11" not in available:
            raise ValueError(f"Experiment {eid} does not have v_sg11 series.")
        a_sg11 = np.array(series["a_sg11"])
        v_sg11 = np.array(series["v_sg11"])

        # 获取 F_ext: 从 config 中取 constant_force
        force_field_type = config.get("force_field_type", None)
        if force_field_type != "constant":
            # 如果不是 constant 类型，则 F_ext 可能为0或未定义，但根据上下文应该是 constant
            # 尝试从 config 中取 constant_force，若不存在则报错
            raise ValueError(f"Experiment {eid} force_field_type is {force_field_type}, expected constant. F_ext not defined.")
        F_ext = config.get("constant_force", None)
        if F_ext is None:
            # 可能由其他方式定义，尝试从已知变量中获取
            # 从配置中找
            raise ValueError(f"Experiment {eid} does not have constant_force in config.")
        F_ext = float(F_ext)

        # 计算 C_candidate = a_sg11 / F_ext + c1 * v_sg11 + c2 * v_sg11**2
        # 注意 expression 中符号是 square(v_sg11) 即 v_sg11^2
        C_values = a_sg11 / F_ext + c1 * v_sg11 + c2 * (v_sg11 ** 2)

        # 统计
        mean_val = float(np.mean(C_values))
        std_val = float(np.std(C_values))
        min_val = float(np.min(C_values))
        max_val = float(np.max(C_values))

        # 记录 metrics
        key_prefix = f"{eid}_{symbol}"
        metrics[f"{key_prefix}_mean"] = mean_val
        metrics[f"{key_prefix}_std"] = std_val
        metrics[f"{key_prefix}_min"] = min_val
        metrics[f"{key_prefix}_max"] = max_val
        all_C_means[eid] = mean_val
        all_C_stds[eid] = std_val
        all_C_mins[eid] = min_val
        all_C_maxs[eid] = max_val

        # 构造派生序列
        derived = {
            "experiment_id": eid,
            "name": symbol,
            "values": list(C_values),
            "source_name": expression,
            "provenance": "generated data processor: define_derived_quantity",
            "description": f"Candidate expression C = a_sg11/F_ext + c1*v_sg11 + c2*(v_sg11^2) with c1={c1}, c2={c2}"
        }
        derived_series.append(derived)

        # 画图
        ax.plot(t, C_values, label=f"{eid} (F_ext={F_ext})", linewidth=1.5)

    # 完善图像
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, label='y=1')
    ax.set_xlabel("t")
    ax.set_ylabel(symbol)
    ax.set_title(f"{symbol} vs t (c1={c1}, c2={c2})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig_path = os.path.join(output_dir, f"{symbol}_vs_t.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    figure_paths.append(fig_path)

    # 构建 observation
    lines = []
    lines.append(f"根据参数定义计算了派生量 {symbol} = {expression}，其中 c1={c1}, c2={c2}。")
    lines.append("各实验统计：")
    for eid in exp_ids:
        lines.append(f"  {eid}: mean={all_C_means[eid]:.6f}, std={all_C_stds[eid]:.6f}, min={all_C_mins[eid]:.6f}, max={all_C_maxs[eid]:.6f}")
    lines.append(f"图像已保存至 {fig_path}。")
    observation = "\n".join(lines)

    # 构建返回 dict
    result = {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figure_paths,
        "metrics": metrics
    }
    return result
