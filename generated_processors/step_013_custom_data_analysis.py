import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
import os, warnings

def process(payload: dict) -> dict:
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 确定要处理的实验列表
    exp_ids = params.get("experiment_ids", list(experiments.keys()))
    # 确保只处理存在的实验
    exp_ids = [eid for eid in exp_ids if eid in experiments]

    # 存储每个实验的拟合结果
    fit_results = {}
    free_check = {}

    for eid in exp_ids:
        exp_data = experiments[eid]
        config = exp_data["config"]
        series = exp_data["series"]
        t = np.array(series["t"])
        available = exp_data.get("available_series", list(series.keys()))

        # 优先使用现有 a_sg, v_sg；若不存在则从 q 生成
        if "a_sg" in available and "v_sg" in available:
            a = np.array(series["a_sg"])
            v = np.array(series["v_sg"])
        else:
            q = np.array(series["q"])
            # 使用 Savitzky-Golay 滤波（窗口11, polyorder3）估计速度和加速度
            dt = config.get("dt", 0.1)
            t2 = t
            if len(t) < 11:
                raise ValueError(f"Experiment {eid}: 数据点数 {len(t)} 小于窗口长度 11")
            q_sg = savgol_filter(q, window_length=11, polyorder=3)
            v = savgol_filter(q, window_length=11, polyorder=3, deriv=1, delta=dt)
            a = savgol_filter(q, window_length=11, polyorder=3, deriv=2, delta=dt)
            # 可选：将新序列注册到 derived_series 中
            # 这里不返回，因为 action 没有要求创建新序列，但需要时可在派生中返回
            # 为了简洁，我们仅使用本地变量，不在派生中返回

        # 计算 v^2
        v2 = v ** 2

        # 线性回归 a = intercept + slope * v^2
        reg = LinearRegression()
        reg.fit(v2.reshape(-1, 1), a)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        # 计算 R²
        a_pred = reg.predict(v2.reshape(-1, 1))
        ss_res = np.sum((a - a_pred)**2)
        ss_tot = np.sum((a - np.mean(a))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

        # 模型：a = α - β*v^2，所以 α = intercept, β = -slope
        alpha = intercept
        beta = -slope

        fit_results[eid] = {
            "alpha": alpha,
            "beta": beta,
            "R2": r2,
            "F_ext": config.get("constant_force", None) or config.get("F_ext", None),
            "force_field_type": config.get("force_field_type", "unknown")
        }

        # 检查自由场景 a 是否接近零
        if config.get("force_field_type") == "free" or eid in ("exp_02", "exp_03"):
            a_mean = np.mean(a)
            a_std = np.std(a)
            is_zero = bool(np.abs(a_mean) < 1e-12)
            free_check[eid] = {
                "a_mean": a_mean,
                "a_std": a_std,
                "is_zero": is_zero
            }

        # 绘制 a vs v^2 散点图
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(v2, a, s=10, alpha=0.7, label="data")
        v2_sorted = np.sort(v2)
        ax.plot(v2_sorted, intercept + slope * v2_sorted, 'r-', label=f"fit: α={alpha:.4f}, β={beta:.4f}\nR²={r2:.4f}")
        ax.set_xlabel("v²")
        ax.set_ylabel("a")
        ax.set_title(f"{eid} (F={fit_results[eid]['F_ext']})")
        ax.legend()
        fig.tight_layout()
        fig_path = os.path.join(output_dir, f"{eid}_a_vs_v2.png")
        fig.savefig(fig_path)
        plt.close(fig)

    # 自由场检查
    free_obs = ""
    for eid in free_check:
        d = free_check[eid]
        free_obs += f"实验 {eid}: 加速度均值={d['a_mean']:.2e}, 标准差={d['a_std']:.2e}, 是否接近零={d['is_zero']}; "

    # 分析 α vs F_ext, β 是否常数（仅限 constant 场景，且 R² 较高 >0.8）
    const_exps = [eid for eid in exp_ids if fit_results[eid]["force_field_type"] == "constant" and eid not in ("exp_02","exp_03")]
    # 进一步筛选 R² > 0.8 的实验
    high_r2_exps = [eid for eid in const_exps if fit_results[eid]["R2"] > 0.8]

    alpha_vs_F_plot = None
    beta_vs_F_plot = None
    alpha_F_slope = None
    alpha_F_intercept = None
    beta_F_slope = None
    beta_F_intercept = None
    alpha_F_r2 = None
    beta_F_r2 = None
    beta_constant_analysis = ""

    if len(high_r2_exps) >= 2:
        # α vs F_ext
        F_vals = np.array([fit_results[eid]["F_ext"] for eid in high_r2_exps])
        alpha_vals = np.array([fit_results[eid]["alpha"] for eid in high_r2_exps])
        beta_vals = np.array([fit_results[eid]["beta"] for eid in high_r2_exps])

        # α 线性拟合
        reg_alpha = LinearRegression()
        reg_alpha.fit(F_vals.reshape(-1,1), alpha_vals)
        alpha_F_slope = reg_alpha.coef_[0]
        alpha_F_intercept = reg_alpha.intercept_
        a_pred = reg_alpha.predict(F_vals.reshape(-1,1))
        ss_res = np.sum((alpha_vals - a_pred)**2)
        ss_tot = np.sum((alpha_vals - np.mean(alpha_vals))**2)
        alpha_F_r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0

        # β 线性拟合
        reg_beta = LinearRegression()
        reg_beta.fit(F_vals.reshape(-1,1), beta_vals)
        beta_F_slope = reg_beta.coef_[0]
        beta_F_intercept = reg_beta.intercept_
        b_pred = reg_beta.predict(F_vals.reshape(-1,1))
        ss_res = np.sum((beta_vals - b_pred)**2)
        ss_tot = np.sum((beta_vals - np.mean(beta_vals))**2)
        beta_F_r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0

        # 绘制 α vs F_ext
        fig, ax = plt.subplots(figsize=(5,4))
        ax.scatter(F_vals, alpha_vals, s=50)
        F_sorted = np.sort(F_vals)
        ax.plot(F_sorted, alpha_F_intercept + alpha_F_slope*F_sorted, 'r-')
        ax.set_xlabel("F_ext")
        ax.set_ylabel("α")
        ax.set_title(f"α vs F_ext (R²={alpha_F_r2:.4f})")
        fig.tight_layout()
        alpha_vs_F_plot = os.path.join(output_dir, "alpha_vs_F_ext.png")
        fig.savefig(alpha_vs_F_plot)
        plt.close(fig)

        # 绘制 β vs F_ext
        fig, ax = plt.subplots(figsize=(5,4))
        ax.scatter(F_vals, beta_vals, s=50)
        ax.plot(F_sorted, beta_F_intercept + beta_F_slope*F_sorted, 'r-')
        ax.set_xlabel("F_ext")
        ax.set_ylabel("β")
        ax.set_title(f"β vs F_ext (R²={beta_F_r2:.4f})")
        fig.tight_layout()
        beta_vs_F_plot = os.path.join(output_dir, "beta_vs_F_ext.png")
        fig.savefig(beta_vs_F_plot)
        plt.close(fig)

        # 分析 β 是否大致常数：若 β 的变异系数（std/mean）小于 0.2 或斜率接近零
        beta_std = np.std(beta_vals)
        beta_mean = np.mean(beta_vals)
        cv = beta_std / abs(beta_mean) if beta_mean != 0 else np.inf
        if cv < 0.2:
            beta_constant_analysis = f"β 相对稳定（CV={cv:.3f}），平均 β = {beta_mean:.4f}±{beta_std:.4f}"
        else:
            beta_constant_analysis = f"β 变化明显（CV={cv:.3f}），平均 β = {beta_mean:.4f}±{beta_std:.4f}"
    else:
        beta_constant_analysis = "R²>0.8 的实验少于2个，无法进行 α vs F_ext 和 β 常数性分析"

    # 构造 observation
    obs_lines = ["使用 a_sg 和 v_sg 序列拟合模型 a = α - β*v²（加速度与速度平方线性关系）:", ""]
    for eid in exp_ids:
        d = fit_results[eid]
        ft = d["force_field_type"]
        F = d["F_ext"]
        obs_lines.append(f"  {eid} ({ft}, F={F}): α={d['alpha']:.6f}, β={d['beta']:.6f}, R²={d['R2']:.6f}")
    obs_lines.append("")
    if free_check:
        obs_lines.append("自由场加速度检查:")
        for eid, d in free_check.items():
            obs_lines.append(f"  {eid}: a_mean={d['a_mean']:.3e}, a_std={d['a_std']:.3e}, 接近零={d['is_zero']}")
    obs_lines.append("")
    obs_lines.append(f"α vs F_ext 线性拟合 (仅R²>0.8的恒力实验):")
    if alpha_F_slope is not None:
        obs_lines.append(f"  slope={alpha_F_slope:.6f}, intercept={alpha_F_intercept:.6f}, R²={alpha_F_r2:.6f}")
    obs_lines.append(f"β 常数性分析: {beta_constant_analysis}")
    observation = "\n".join(obs_lines)

    # 构建 metrics
    metrics = {}
    for eid in exp_ids:
        d = fit_results[eid]
        prefix = eid
        metrics[f"{prefix}_alpha"] = d["alpha"]
        metrics[f"{prefix}_beta"] = d["beta"]
        metrics[f"{prefix}_R2"] = d["R2"]
        metrics[f"{prefix}_F_ext"] = d["F_ext"]
    for eid, d in free_check.items():
        metrics[f"{eid}_a_mean"] = d["a_mean"]
        metrics[f"{eid}_a_is_zero"] = int(d["is_zero"])
    if alpha_F_slope is not None:
        metrics["alpha_vs_F_slope"] = alpha_F_slope
        metrics["alpha_vs_F_intercept"] = alpha_F_intercept
        metrics["alpha_vs_F_R2"] = alpha_F_r2
    metrics["beta_constant_cv"] = np.std([fit_results[eid]["beta"] for eid in high_r2_exps]) / abs(np.mean([fit_results[eid]["beta"] for eid in high_r2_exps])) if len(high_r2_exps)>=2 else None

    # 收集图像路径
    fig_list = [os.path.join(output_dir, f"{eid}_a_vs_v2.png") for eid in exp_ids]
    if alpha_vs_F_plot:
        fig_list.append(alpha_vs_F_plot)
    if beta_vs_F_plot:
        fig_list.append(beta_vs_F_plot)

    # 不再返回派生序列（未创建新序列）
    return {
        "observation": observation,
        "derived_series": [],  # 没有新派生序列
        "figures": fig_list,
        "metrics": metrics
    }
