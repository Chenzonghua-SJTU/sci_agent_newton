import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

def process(payload):
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # 规范化实验ID: 如果带下划线则直接使用，否则尝试添加下划线
    def normalize_eid(eid):
        if eid in experiments:
            return eid
        # 尝试带下划线格式：exp_02
        if eid.startswith("exp") and len(eid) > 3:
            new_id = "exp_" + eid[3:]
            if new_id in experiments:
                return new_id
        raise ValueError(f"Experiment {eid} not found in payload. Available: {list(experiments.keys())}")

    exp_ids = [normalize_eid(eid) for eid in params["experiment_ids"]]
    x_series_name = params["x_series"]
    y_series_name = params["y_series"]

    # 存储每个实验的数据和拟合结果
    exp_data = []  # 每个元素: (exp_id, x, y, config)
    for eid in exp_ids:
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        # 检查序列是否存在
        if x_series_name not in series:
            raise ValueError(f"Series '{x_series_name}' not found in experiment {eid}. Available: {exp['available_series']}")
        if y_series_name not in series:
            raise ValueError(f"Series '{y_series_name}' not found in experiment {eid}. Available: {exp['available_series']}")
        x = np.array(series[x_series_name])
        y = np.array(series[y_series_name])
        # 确保长度一致
        if len(x) != len(y):
            raise ValueError(f"Series lengths mismatch in {eid}: {x_series_name} len={len(x)}, {y_series_name} len={len(y)}")
        exp_data.append((eid, x, y, config))

    # 执行线性回归并收集统计
    metrics = {}
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_data)))

    observation_lines = []
    observation_lines.append(f"对 {len(exp_data)} 个实验的 {x_series_name} 和 {y_series_name} 关系进行探索。")

    for idx, (eid, x, y, config) in enumerate(exp_data):
        # 计算皮尔逊相关系数
        r, p_value = stats.pearsonr(x, y)
        # 线性回归
        slope, intercept, r_value, p_value_slope, std_err = stats.linregress(x, y)
        r2 = r_value ** 2
        # 记录 metrics
        metrics[f"{eid}_r"] = r
        metrics[f"{eid}_slope"] = slope
        metrics[f"{eid}_intercept"] = intercept
        metrics[f"{eid}_R2"] = r2
        metrics[f"{eid}_p_value"] = p_value
        # 异常检测：如果数据点极少或零方差，跳过拟合线
        if x.std() < 1e-12 or y.std() < 1e-12:
            # 常数序列，不画拟合线
            label = f"{eid} (const)"
            ax.scatter(x, y, color=colors[idx], label=label, alpha=0.7, s=10)
            line_str = "常数序列，未拟合"
        else:
            # 画散点和拟合线
            label = f"{eid} (R²={r2:.4f})"
            ax.scatter(x, y, color=colors[idx], label=label, alpha=0.7, s=10)
            # 拟合线: 从x最小到最大
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=colors[idx], linestyle='--', alpha=0.8)
            line_str = f"斜率={slope:.4f}, 截距={intercept:.4f}, R²={r2:.4f}"
        # 添加实验信息
        F_ext = config.get("constant_force", config.get("F_ext", "N/A"))
        v0 = config.get("initial_v", "N/A")
        observation_lines.append(f"  {eid}: F_ext={F_ext}, v0={v0}, 序列长度={len(x)}; 拟合: {line_str}")

    ax.set_xlabel(x_series_name)
    ax.set_ylabel(y_series_name)
    ax.set_title(f"Relationship between {y_series_name} vs {x_series_name}")
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    # 保存图像
    figure_filename = f"relationship_{x_series_name}_vs_{y_series_name}.png"
    figure_path = os.path.join(output_dir, figure_filename)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close(fig)

    observation = "\n".join(observation_lines)

    return {
        "observation": observation,
        "derived_series": [],
        "figures": [figure_path],
        "metrics": metrics
    }
