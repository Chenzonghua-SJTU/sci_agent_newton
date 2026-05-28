import numpy as np
from pathlib import Path
from typing import Dict, Any
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # 只处理指定实验
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(experiments.keys())

    # 检查所有实验都存在
    missing = [e for e in exp_ids if e not in experiments]
    if missing:
        raise ValueError(f"Missing experiments: {missing}")

    F_ext = 1.0  # 根据 analysis_goal 明确给出

    derived_series = []
    figures = []
    metrics = {}

    # 用于存储跨实验的比较数据
    gamma_linear = {}
    gamma_quad = {}

    # 绘制子图：每个实验两张子图（a vs v, a vs v^2），放在同一个 figure 里
    n_exp = len(exp_ids)
    fig, axes = plt.subplots(n_exp, 2, figsize=(12, 4 * n_exp), squeeze=False)

    for idx, exp_id in enumerate(exp_ids):
        exp = experiments[exp_id]
        config = exp["config"]
        series_dict = exp["series"]

        # 获取时间序列长度，确保与 t 一致
        t = np.array(series_dict["t"])
        n = len(t)

        # 获取或计算 v_sg 和 a_sg
        if "v_sg" in series_dict and "a_sg" in series_dict:
            v = np.array(series_dict["v_sg"])
            a = np.array(series_dict["a_sg"])
            if len(v) != n or len(a) != n:
                raise ValueError(f"v_sg/a_sg length mismatch in {exp_id}")
            # 检查 SG 参数是否匹配（窗口=7, polyorder=2），由于之前可能不同，我们重新计算以确保一致
            # 但为了效率，直接使用现有序列，同时记录
            source_desc = "existing v_sg/a_sg (assumed SG(7,2))"
        else:
            # 从 q 重新计算
            q = np.array(series_dict["q"])
            v = savgol_filter(q, window_length=7, polyorder=2, deriv=1, delta=config.get("dt", 0.1))
            a = savgol_filter(q, window_length=7, polyorder=2, deriv=2, delta=config.get("dt", 0.1))
            source_desc = "SG(window=7, polyorder=2) from q"
            # 记录新序列
            derived_series.append({
                "experiment_id": exp_id,
                "name": "v_sg",
                "values": v.tolist(),
                "source_name": source_desc,
                "provenance": "generated data processor: custom_data_analysis",
                "description": "velocity from SG filter"
            })
            derived_series.append({
                "experiment_id": exp_id,
                "name": "a_sg",
                "values": a.tolist(),
                "source_name": source_desc,
                "provenance": "generated data processor: custom_data_analysis",
                "description": "acceleration from SG filter"
            })

        # 确保长度一致
        if len(v) != n or len(a) != n:
            raise ValueError(f"Velocity/acceleration length mismatch in {exp_id}")

        # 拟合模型
        # 模型1: a = alpha + beta * v  (自由截距线性)
        coeff1 = np.polyfit(v, a, 1)  # [beta, alpha]
        beta1 = coeff1[0]
        alpha1 = coeff1[1]
        a_pred1 = np.polyval(coeff1, v)
        rmse1 = np.sqrt(mean_squared_error(a, a_pred1))
        r2_1 = r2_score(a, a_pred1)

        # 模型2: a = alpha + beta * v^2 (自由截距二次)
        coeff2 = np.polyfit(v**2, a, 1)  # [beta, alpha]
        beta2 = coeff2[0]
        alpha2 = coeff2[1]
        a_pred2 = np.polyval(coeff2, v**2)
        rmse2 = np.sqrt(mean_squared_error(a, a_pred2))
        r2_2 = r2_score(a, a_pred2)

        # 模型3: a = F_ext - gamma * v  (约束截距为 F_ext)
        # y_trans = a - F_ext = -gamma * v
        # 无截距线性回归
        X3 = v.reshape(-1, 1)
        y3 = a - F_ext
        reg3 = LinearRegression(fit_intercept=False).fit(X3, y3)
        gamma3 = -reg3.coef_[0]  # 因为 y3 = -gamma * v
        a_pred3 = F_ext - gamma3 * v
        rmse3 = np.sqrt(mean_squared_error(a, a_pred3))
        # R2 with respect to mean (有常数模型)
        ss_res = np.sum((a - a_pred3)**2)
        ss_tot = np.sum((a - np.mean(a))**2)
        r2_3 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

        # 模型4: a = F_ext - gamma * v^2
        X4 = (v**2).reshape(-1, 1)
        y4 = a - F_ext
        reg4 = LinearRegression(fit_intercept=False).fit(X4, y4)
        gamma4 = -reg4.coef_[0]
        a_pred4 = F_ext - gamma4 * (v**2)
        rmse4 = np.sqrt(mean_squared_error(a, a_pred4))
        ss_res4 = np.sum((a - a_pred4)**2)
        r2_4 = 1 - ss_res4 / ss_tot if ss_tot != 0 else 0.0

        # 常数检验: a + gamma3 * v (使用模型3的gamma)
        amv = a + gamma3 * v
        amv_mean = np.mean(amv)
        amv_std = np.std(amv, ddof=0)
        # 类似地，a + gamma4 * v^2
        amv2 = a + gamma4 * (v**2)
        amv2_mean = np.mean(amv2)
        amv2_std = np.std(amv2, ddof=0)

        # 记录 gamma 用于跨实验比较
        gamma_linear[exp_id] = gamma3
        gamma_quad[exp_id] = gamma4

        # 存储 metrics
        exp_metrics = {
            f"{exp_id}_beta1": beta1,
            f"{exp_id}_alpha1": alpha1,
            f"{exp_id}_rmse1": rmse1,
            f"{exp_id}_r2_1": r2_1,
            f"{exp_id}_beta2": beta2,
            f"{exp_id}_alpha2": alpha2,
            f"{exp_id}_rmse2": rmse2,
            f"{exp_id}_r2_2": r2_2,
            f"{exp_id}_gamma3": gamma3,
            f"{exp_id}_rmse3": rmse3,
            f"{exp_id}_r2_3": r2_3,
            f"{exp_id}_gamma4": gamma4,
            f"{exp_id}_rmse4": rmse4,
            f"{exp_id}_r2_4": r2_4,
            f"{exp_id}_amv_mean": amv_mean,
            f"{exp_id}_amv_std": amv_std,
            f"{exp_id}_amv2_mean": amv2_mean,
            f"{exp_id}_amv2_std": amv2_std,
        }
        metrics.update(exp_metrics)

        # 返回派生序列：模型预测和残差（可选）
        derived_series.extend([
            {"experiment_id": exp_id, "name": f"a_pred_linear_free", "values": a_pred1.tolist(),
             "source_name": "a = alpha + beta*v fit", "provenance": "custom_data_analysis",
             "description": "prediction from free-intercept linear model"},
            {"experiment_id": exp_id, "name": f"a_pred_quad_free", "values": a_pred2.tolist(),
             "source_name": "a = alpha + beta*v^2 fit", "provenance": "custom_data_analysis",
             "description": "prediction from free-intercept quadratic model"},
            {"experiment_id": exp_id, "name": f"a_pred_linear_constrained", "values": a_pred3.tolist(),
             "source_name": f"a = F_ext - gamma*v fit, F_ext={F_ext}", "provenance": "custom_data_analysis",
             "description": "prediction from constrained linear model"},
            {"experiment_id": exp_id, "name": f"a_pred_quad_constrained", "values": a_pred4.tolist(),
             "source_name": f"a = F_ext - gamma*v^2 fit, F_ext={F_ext}", "provenance": "custom_data_analysis",
             "description": "prediction from constrained quadratic model"},
            {"experiment_id": exp_id, "name": f"a_res_linear_free", "values": (a - a_pred1).tolist(),
             "source_name": "residual of free linear model", "provenance": "custom_data_analysis",
             "description": "residual from a = alpha + beta*v"},
            {"experiment_id": exp_id, "name": f"a_res_quad_free", "values": (a - a_pred2).tolist(),
             "source_name": "residual of free quadratic model", "provenance": "custom_data_analysis",
             "description": "residual from a = alpha + beta*v^2"},
            {"experiment_id": exp_id, "name": f"a_plus_gamma_v", "values": amv.tolist(),
             "source_name": f"a + gamma*v (gamma={gamma3:.6f})", "provenance": "custom_data_analysis",
             "description": "check for constancy"},
            {"experiment_id": exp_id, "name": f"a_plus_gamma_v2", "values": amv2.tolist(),
             "source_name": f"a + gamma*v^2 (gamma={gamma4:.6f})", "provenance": "custom_data_analysis",
             "description": "check for constancy"},
        ])

        # 绘图：左图 a vs v，右图 a vs v^2
        ax_left = axes[idx, 0]
        ax_right = axes[idx, 1]

        # 左图: a vs v
        ax_left.scatter(v, a, s=10, alpha=0.7, label='data')
        v_sorted = np.sort(v)
        # 自由线性
        ax_left.plot(v_sorted, np.polyval(coeff1, v_sorted), '--', label=f'free: a={alpha1:.4f}{beta1:+.4f}v')
        # 约束线性
        ax_left.plot(v_sorted, F_ext - gamma3 * v_sorted, '-', label=f'constrained: a={F_ext:.1f}{-gamma3:.4f}v')
        ax_left.set_xlabel('v')
        ax_left.set_ylabel('a')
        ax_left.set_title(f'{exp_id}: a vs v')
        ax_left.legend(fontsize=8)
        ax_left.grid(True, alpha=0.3)

        # 右图: a vs v^2
        v2 = v**2
        ax_right.scatter(v2, a, s=10, alpha=0.7, label='data')
        v2_sorted = np.sort(v2)
        # 自由二次
        ax_right.plot(v2_sorted, np.polyval(coeff2, v2_sorted), '--', label=f'free: a={alpha2:.4f}{beta2:+.4f}v²')
        # 约束二次
        ax_right.plot(v2_sorted, F_ext - gamma4 * v2_sorted, '-', label=f'constrained: a={F_ext:.1f}{-gamma4:.4f}v²')
        ax_right.set_xlabel('v²')
        ax_right.set_ylabel('a')
        ax_right.set_title(f'{exp_id}: a vs v²')
        ax_right.legend(fontsize=8)
        ax_right.grid(True, alpha=0.3)

    # 调整布局并保存
    plt.tight_layout()
    fig_path = Path(output_dir) / "a_vs_v_and_v2_fits.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    figures.append(str(fig_path))

    # 跨实验 gamma 比较
    if len(exp_ids) > 1:
        gamma_linear_values = list(gamma_linear.values())
        gamma_quad_values = list(gamma_quad.values())
        metrics["gamma_linear_range"] = max(gamma_linear_values) - min(gamma_linear_values)
        metrics["gamma_quad_range"] = max(gamma_quad_values) - min(gamma_quad_values)
    else:
        metrics["gamma_linear_range"] = 0.0
        metrics["gamma_quad_range"] = 0.0

    # 构建 observation
    lines = []
    for exp_id in exp_ids:
        m = {k: v for k, v in metrics.items() if k.startswith(exp_id)}
        lines.append(f"实验 {exp_id}:")
        lines.append(f"  自由线性: a = {m[f'{exp_id}_alpha1']:.6f} + {m[f'{exp_id}_beta1']:.6f}*v  | RMSE={m[f'{exp_id}_rmse1']:.6f}, R²={m[f'{exp_id}_r2_1']:.6f}")
        lines.append(f"  自由二次: a = {m[f'{exp_id}_alpha2']:.6f} + {m[f'{exp_id}_beta2']:.6f}*v² | RMSE={m[f'{exp_id}_rmse2']:.6f}, R²={m[f'{exp_id}_r2_2']:.6f}")
        lines.append(f"  约束线性 (a=F_ext-gamma*v): gamma={m[f'{exp_id}_gamma3']:.6f}, RMSE={m[f'{exp_id}_rmse3']:.6f}, R²={m[f'{exp_id}_r2_3']:.6f}")
        lines.append(f"  约束二次 (a=F_ext-gamma*v²): gamma={m[f'{exp_id}_gamma4']:.6f}, RMSE={m[f'{exp_id}_rmse4']:.6f}, R²={m[f'{exp_id}_r2_4']:.6f}")
        lines.append(f"  a+gamma*v (线性约束) 均值={m[f'{exp_id}_amv_mean']:.6f}, 标准差={m[f'{exp_id}_amv_std']:.6f}")
        lines.append(f"  a+gamma*v² (二次约束) 均值={m[f'{exp_id}_amv2_mean']:.6f}, 标准差={m[f'{exp_id}_amv2_std']:.6f}")
    if len(exp_ids) > 1:
        lines.append(f"跨实验 gamma_linear 极差 = {metrics['gamma_linear_range']:.6f}")
        lines.append(f"跨实验 gamma_quad 极差 = {metrics['gamma_quad_range']:.6f}")
    observation = "\n".join(lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
