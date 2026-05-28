import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

def process(payload: dict) -> dict:
    action = payload.get("action")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 提取参数
    exp_id = params.get("experiment_id")
    x_name = params.get("x_series")
    y_name = params.get("y_series")

    if exp_id not in experiments:
        raise ValueError(f"Experiment {exp_id} not found in payload.")
    exp = experiments[exp_id]
    series_dict = exp.get("series", {})
    available = exp.get("available_series", [])

    # 检查序列是否存在
    if x_name not in available:
        raise ValueError(f"x_series '{x_name}' not available for experiment {exp_id}. Available: {available}")
    if y_name not in available:
        raise ValueError(f"y_series '{y_name}' not available for experiment {exp_id}. Available: {available}")

    x = np.array(series_dict[x_name], dtype=float)
    y = np.array(series_dict[y_name], dtype=float)

    # 检查长度和有效性
    if len(x) != len(y):
        raise ValueError(f"Length mismatch: x ({len(x)}) vs y ({len(y)})")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        raise ValueError(f"Too few valid points ({len(x)}) after removing NaNs/Infs.")

    # 统计描述
    n = len(x)
    corr_coef, p_value = stats.pearsonr(x, y)
    # 线性拟合
    slope, intercept, r_value, p_value_slope, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2

    # 度量
    metrics = {
        "n_points": n,
        "pearson_r": float(corr_coef),
        "pearson_p": float(p_value),
        "linear_slope": float(slope),
        "linear_intercept": float(intercept),
        "R_squared": float(r_squared),
        "p_value_slope": float(p_value_slope)
    }

    # 绘制图像
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=20, alpha=0.7, label=f"{y_name} vs {x_name}")
    # 拟合线
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r-', label=f"Fit: y={slope:.4f}x+{intercept:.4f}, R²={r_squared:.4f}")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"Relationship: {y_name} vs {x_name} (exp_{exp_id})")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # 保存图像
    figure_filename = f"inspect_{exp_id}_{x_name}_vs_{y_name}.png"
    figure_path = os.path.join(output_dir, figure_filename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 构造 observation
    obs = (
        f"检查实验 {exp_id} 中 {y_name} 与 {x_name} 的关系。\n"
        f"有效数据点 {n} 个。皮尔逊相关系数 r = {corr_coef:.4f} (p = {p_value:.6e})。\n"
        f"线性拟合: {y_name} = {slope:.6f} * {x_name} + {intercept:.6f}, R² = {r_squared:.6f}, 斜率p值 = {p_value_slope:.6e}。\n"
        f"散点图及拟合线已保存至 {figure_filename}。"
    )

    return {
        "observation": obs,
        "derived_series": [],
        "figures": [figure_path],
        "metrics": metrics
    }
