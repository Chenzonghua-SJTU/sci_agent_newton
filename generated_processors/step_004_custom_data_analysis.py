import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def process(payload: dict) -> dict:
    # 提取参数
    params = payload.get("parameters", {})
    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        # 如果没有指定，尝试从 experiments 中获取所有实验
        experiment_ids = list(payload.get("experiments", {}).keys())
    
    # 只处理 exp_02（根据 action 要求）
    if "exp_02" not in experiment_ids:
        raise ValueError("当前任务仅指定分析 exp_02，但 experiment_ids 中未包含 exp_02")
    
    exp_id = "exp_02"
    exp_data = payload["experiments"].get(exp_id)
    if exp_data is None:
        raise ValueError(f"实验 {exp_id} 不存在于 payload 中")
    
    series = exp_data.get("series", {})
    required = ["a_estimated", "v_estimated", "q", "t"]
    for s in required:
        if s not in series:
            raise ValueError(f"实验 {exp_id} 缺少必要序列 {s}")
    
    t = np.array(series["t"])
    a = np.array(series["a_estimated"])
    v = np.array(series["v_estimated"])
    q = np.array(series["q"])
    
    output_dir = payload.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 计算 a 与 v 的相关系数
    corr_a_v = np.corrcoef(a, v)[0, 1]
    
    # 2. 绘制 a vs v 散点图
    fig1, ax1 = plt.subplots()
    ax1.scatter(v, a, s=10, alpha=0.7)
    ax1.set_xlabel("v_estimated")
    ax1.set_ylabel("a_estimated")
    ax1.set_title(f"exp_{exp_id}: a vs v (corr={corr_a_v:.4f})")
    fig1.tight_layout()
    fig1_path = os.path.join(output_dir, f"{exp_id}_a_vs_v_scatter.png")
    fig1.savefig(fig1_path, dpi=150)
    plt.close(fig1)
    
    # 3. 线性拟合 a vs v
    slope_v, intercept_v, r_value_v, p_value_v, std_err_v = stats.linregress(v, a)
    a_pred_v = slope_v * v + intercept_v
    residuals_v = a - a_pred_v
    resid_std_v = np.std(residuals_v)
    
    # 4. 线性拟合 a vs q
    slope_q, intercept_q, r_value_q, p_value_q, std_err_q = stats.linregress(q, a)
    a_pred_q = slope_q * q + intercept_q
    residuals_q = a - a_pred_q
    resid_std_q = np.std(residuals_q)
    
    # 5. 计算组合量 a/v, a/q, a*v
    # 注意处理除零，用 np.divide 设置 where
    a_over_v = np.divide(a, v, out=np.full_like(a, np.nan), where=(v != 0))
    a_over_q = np.divide(a, q, out=np.full_like(a, np.nan), where=(q != 0))
    a_times_v = a * v
    
    # 计算各组合的统计量（忽略 NaN）
    def safe_mean_std_rsd(arr):
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan, np.nan, np.nan
        mean_val = np.mean(valid)
        std_val = np.std(valid, ddof=1)  # 样本标准差
        rsd = std_val / abs(mean_val) * 100 if mean_val != 0 else np.nan
        return mean_val, std_val, rsd
    
    mean_a_over_v, std_a_over_v, rsd_a_over_v = safe_mean_std_rsd(a_over_v)
    mean_a_over_q, std_a_over_q, rsd_a_over_q = safe_mean_std_rsd(a_over_q)
    mean_a_times_v, std_a_times_v, rsd_a_times_v = safe_mean_std_rsd(a_times_v)
    
    # 构建返回 metrics
    metrics = {
        "corr_a_v": corr_a_v,
        "a_vs_v_slope": slope_v,
        "a_vs_v_intercept": intercept_v,
        "a_vs_v_r2": r_value_v**2,
        "a_vs_v_resid_std": resid_std_v,
        "a_vs_pvalue": p_value_v,
        "a_vs_q_slope": slope_q,
        "a_vs_q_intercept": intercept_q,
        "a_vs_q_r2": r_value_q**2,
        "a_vs_q_resid_std": resid_std_q,
        "a_vs_q_pvalue": p_value_q,
        "a_over_v_mean": mean_a_over_v,
        "a_over_v_std": std_a_over_v,
        "a_over_v_rsd_percent": rsd_a_over_v,
        "a_over_q_mean": mean_a_over_q,
        "a_over_q_std": std_a_over_q,
        "a_over_q_rsd_percent": rsd_a_over_q,
        "a_times_v_mean": mean_a_times_v,
        "a_times_v_std": std_a_times_v,
        "a_times_v_rsd_percent": rsd_a_times_v
    }
    
    # 构造 derived_series（长度与 t 一致）
    derived_series = [
        {
            "experiment_id": exp_id,
            "name": "a_over_v",
            "values": a_over_v.tolist(),
            "source_name": "a_estimated / v_estimated",
            "provenance": "custom_data_analysis",
            "description": "加速度与速度的比值"
        },
        {
            "experiment_id": exp_id,
            "name": "a_over_q",
            "values": a_over_q.tolist(),
            "source_name": "a_estimated / q",
            "provenance": "custom_data_analysis",
            "description": "加速度与位置的比值"
        },
        {
            "experiment_id": exp_id,
            "name": "a_times_v",
            "values": a_times_v.tolist(),
            "source_name": "a_estimated * v_estimated",
            "provenance": "custom_data_analysis",
            "description": "加速度与速度的乘积"
        }
    ]
    
    # 构建 observation 字符串
    observation = (f"分析实验 {exp_id} 中 a_estimated 与 v_estimated 和 q 的关系。\n"
                   f"a 与 v 的 Pearson 相关系数：{corr_a_v:.4f}\n"
                   f"a vs v 线性拟合：a = {intercept_v:.4f} + {slope_v:.4f} * v，R²={r_value_v**2:.4f}，残差标准差={resid_std_v:.4f}\n"
                   f"a vs q 线性拟合：a = {intercept_q:.4f} + {slope_q:.4f} * q，R²={r_value_q**2:.4f}，残差标准差={resid_std_q:.4f}\n"
                   f"组合量统计（均值 ± 标准差，相对标准差%）：\n"
                   f"  a/v = {mean_a_over_v:.4f} ± {std_a_over_v:.4f} (RSD={rsd_a_over_v:.2f}%)\n"
                   f"  a/q = {mean_a_over_q:.4f} ± {std_a_over_q:.4f} (RSD={rsd_a_over_q:.2f}%)\n"
                   f"  a*v = {mean_a_times_v:.4f} ± {std_a_times_v:.4f} (RSD={rsd_a_times_v:.2f}%)\n"
                   f"已生成 a vs v 散点图，并返回 a_over_v, a_over_q, a_times_v 派生序列。")
    
    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [fig1_path],
        "metrics": metrics
    }
