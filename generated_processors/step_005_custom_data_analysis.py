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

    # 只处理 exp_03
    exp_ids = params.get("experiment_ids", ["exp_03"])
    # 但参数已明确只包含 exp_03，所以我们只取第一个，或者迭代处理其实只有一个
    target_exp = "exp_03"
    if target_exp not in experiments:
        raise ValueError(f"Experiment {target_exp} not found in payload experiments.")
    
    exp = experiments[target_exp]
    config = exp.get("config", {})
    series = exp.get("series", {})
    available = exp.get("available_series", [])

    # 检查必须的序列
    required_series = ["a_exp_03", "v_exp_03", "t"]
    for s in required_series:
        if s not in series:
            raise ValueError(f"Required series '{s}' not available in experiment {target_exp}. Available: {available}")

    t = np.array(series["t"], dtype=float)
    a = np.array(series["a_exp_03"], dtype=float)
    v = np.array(series["v_exp_03"], dtype=float)

    # ----------------- 1. a vs v 散点图 + 线性拟合 -----------------
    slope, intercept, r_value, p_value, std_err = stats.linregress(v, a)
    r_squared = r_value ** 2
    a_pred = intercept + slope * v
    residuals = a - a_pred

    # 图1: 散点+拟合线
    fig1, ax1 = plt.subplots(figsize=(6,5))
    ax1.scatter(v, a, label='Data', alpha=0.7)
    v_sort = np.sort(v)
    ax1.plot(v_sort, intercept + slope * v_sort, 'r-', label=f'Linear fit: a = {intercept:.4f} + {slope:.4f}*v')
    ax1.set_xlabel('v_exp_03')
    ax1.set_ylabel('a_exp_03')
    ax1.set_title('a vs v with linear fit (exp_03)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    fig1_path = os.path.join(output_dir, "a_vs_v_linear_fit_exp_03.png")
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # ----------------- 2. a + k*v 常数性检验 (k=0.2) -----------------
    k = 0.2
    q = a + k * v  # 表达式 a + k*v
    q_mean = np.mean(q)
    q_std = np.std(q, ddof=1)
    # 对时间做线性回归看趋势
    slope_q, intercept_q, r_q, p_q, std_err_q = stats.linregress(t, q)
    r_squared_q = r_q ** 2  # 拟合适度
    # 检查斜率是否接近零（趋势小）
    trend_significant = abs(slope_q) > 0.01 * (q_std if q_std > 0 else 1e-9)

    # 图2: a + k*v 随时间变化
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(t, q, 'b.-', label=f'a + {k}*v')
    ax2.axhline(y=q_mean, color='gray', linestyle='--', label=f'mean={q_mean:.4f}')
    ax2.set_xlabel('t')
    ax2.set_ylabel('a + k*v')
    ax2.set_title(f'a + {k}*v vs time (exp_03)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    fig2_path = os.path.join(output_dir, "a_plus_kv_vs_time_exp_03.png")
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # ----------------- 构造返回 -----------------
    observation = (
        f"实验 {target_exp} 分析完成。\n"
        f"线性拟合: a = {intercept:.4f} + {slope:.4f} * v, R² = {r_squared:.4f}, 斜率 p-value = {p_value:.2e}\n"
        f"残差统计: 均值={np.mean(residuals):.4f}, 标准差={np.std(residuals, ddof=1):.4f}\n"
        f"常数性检验 (k={k}): a + {k}*v 均值={q_mean:.4f}, 标准差={q_std:.4f}\n"
        f"  a+{k}*v 对时间的线性趋势: 斜率={slope_q:.6f}, R²={r_squared_q:.4f}, p={p_q:.2e}\n"
        f"  趋势明显性判断: {'可能存在趋势' if trend_significant else '无明显趋势（近似常数）'}\n"
        f"图像已保存: {os.path.basename(fig1_path)}, {os.path.basename(fig2_path)}。\n"
        f"残差序列 'residual_linear_a_vs_v' 和派生序列 'a_plus_{k}_v' 已返回。"
    )

    derived_series = [
        {
            "experiment_id": target_exp,
            "name": "residual_linear_a_vs_v",
            "values": residuals.tolist(),
            "source_name": f"residual = a_exp_03 - ({intercept:.4f} + {slope:.4f} * v_exp_03)",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "线性拟合残差 a_exp_03 - (intercept + slope*v_exp_03)"
        },
        {
            "experiment_id": target_exp,
            "name": f"a_plus_{k}_v",
            "values": q.tolist(),
            "source_name": f"a_exp_03 + {k} * v_exp_03",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"常数性检验所用序列 a + {k}*v"
        }
    ]

    metrics = {
        "linear_slope": slope,
        "linear_intercept": intercept,
        "linear_R2": r_squared,
        "linear_pvalue": p_value,
        "linear_residual_mean": float(np.mean(residuals)),
        "linear_residual_std": float(np.std(residuals, ddof=1)),
        "a_plus_kv_mean": q_mean,
        "a_plus_kv_std": q_std,
        "a_plus_kv_trend_slope": slope_q,
        "a_plus_kv_trend_R2": r_squared_q,
        "a_plus_kv_trend_pvalue": p_q,
        "k_used": k
    }

    figures = [fig1_path, fig2_path]

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
