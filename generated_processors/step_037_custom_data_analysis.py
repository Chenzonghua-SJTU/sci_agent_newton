import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = payload["parameters"]
    experiment_ids = params["experiment_ids"]
    experiments = payload["experiments"]
    output_dir = Path(payload["output_dir"])

    results = {}
    derived_series_all = []
    figures = []
    metrics = {}

    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload.")
        exp = experiments[eid]
        config = exp["config"]
        F_ext = config.get("F_ext", 0.0)
        if F_ext is None:
            F_ext = 0.0

        # 选择加速度序列：exp_08 用 a_new，否则用 a_sg
        if eid == "exp_08":
            a_series_name = "a_new"
        else:
            a_series_name = "a_sg"

        if a_series_name not in exp["series"]:
            raise ValueError(f"Experiment {eid}: required series '{a_series_name}' not found.")
        if "v_sg" not in exp["series"]:
            raise ValueError(f"Experiment {eid}: required series 'v_sg' not found.")
        # v_sq 可能不存在，则计算
        v_sq_available = "v_sq" in exp["series"]

        a = np.array(exp["series"][a_series_name])
        v = np.array(exp["series"]["v_sg"])
        t = np.array(exp["series"]["t"])
        n = len(t)
        if len(a) != n or len(v) != n:
            raise ValueError(f"Experiment {eid}: series length mismatch.")
        if v_sq_available:
            v_sq = np.array(exp["series"]["v_sq"])
        else:
            v_sq = v ** 2

        # ---------- 模型1: a = beta0 + beta1*v + beta2*v^2 ----------
        coeffs = np.polyfit(v, a, 2)  # [beta2, beta1, beta0]
        beta2, beta1, beta0 = coeffs[0], coeffs[1], coeffs[2]
        pred1 = np.polyval(coeffs, v)
        residual1 = a - pred1
        ss_res1 = np.sum(residual1 ** 2)
        ss_tot1 = np.sum((a - np.mean(a)) ** 2)
        r2_1 = 1 - ss_res1 / ss_tot1 if ss_tot1 != 0 else 0.0
        rmse_1 = np.sqrt(ss_res1 / n)
        resid_std1 = np.std(residual1, ddof=1)

        # ---------- 模型2: a = F_ext - gamma*v ----------
        # gamma = sum((F_ext - a)*v) / sum(v^2)
        numerator2 = np.sum((F_ext - a) * v)
        denominator2 = np.sum(v ** 2)
        if abs(denominator2) < 1e-15:
            gamma = 0.0
        else:
            gamma = numerator2 / denominator2
        pred2 = F_ext - gamma * v
        residual2 = a - pred2
        ss_res2 = np.sum(residual2 ** 2)
        ss_tot2 = np.sum((a - np.mean(a)) ** 2)
        r2_2 = 1 - ss_res2 / ss_tot2 if ss_tot2 != 0 else 0.0
        rmse_2 = np.sqrt(ss_res2 / n)
        resid_std2 = np.std(residual2, ddof=1)

        # ---------- 模型3: a = F_ext * (1 - v/v_max) => v_max = F_ext/gamma (gamma != 0) ----------
        if abs(gamma) > 1e-15:
            v_max = F_ext / gamma
        else:
            v_max = float('inf')  # 没有定义
        # 模型3的拟合值与模型2相同，R²相同
        pred3 = pred2
        residual3 = residual2
        r2_3 = r2_2
        rmse_3 = rmse_2

        # ---------- 检查 beta0 与 F_ext ----------
        beta0_diff = beta0 - F_ext
        if abs(F_ext) > 1e-15:
            beta0_diff_rel = beta0_diff / F_ext
        else:
            beta0_diff_rel = float('inf')

        # ---------- 保存结果 ----------
        exp_metrics = {
            f"{eid}_beta0": beta0,
            f"{eid}_beta1": beta1,
            f"{eid}_beta2": beta2,
            f"{eid}_model1_R2": r2_1,
            f"{eid}_model1_RMSE": rmse_1,
            f"{eid}_model1_residual_std": resid_std1,
            f"{eid}_beta0_diff": beta0_diff,
            f"{eid}_beta0_diff_rel": beta0_diff_rel,
            f"{eid}_gamma": gamma,
            f"{eid}_model2_R2": r2_2,
            f"{eid}_model2_RMSE": rmse_2,
            f"{eid}_model2_residual_std": resid_std2,
            f"{eid}_v_max": v_max,
        }
        metrics.update(exp_metrics)

        # ---------- 派生序列 ----------
        derived_series_all.append({
            "experiment_id": eid,
            "name": "residual_model1",
            "values": residual1.tolist(),
            "source_name": f"polyfit(a_sg/{a_series_name}, v_sg, deg=2)",
            "provenance": "generated data processor: custom_data_analysis"
        })
        derived_series_all.append({
            "experiment_id": eid,
            "name": "residual_model2",
            "values": residual2.tolist(),
            "source_name": f"linear model a = F_ext - gamma*v",
            "provenance": "generated data processor: custom_data_analysis"
        })

        # ---------- 绘图 ----------
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(v, a, s=10, alpha=0.7, label='data')
        v_sort = np.sort(v)
        ax.plot(v_sort, np.polyval(coeffs, v_sort), 'r-', label='Model1 (quadratic)', lw=2)
        ax.plot(v_sort, F_ext - gamma * v_sort, 'g--', label='Model2/3 (linear)', lw=2)
        ax.set_xlabel('v_sg')
        ax.set_ylabel(a_series_name)
        ax.set_title(f"{eid}: a vs v, F_ext={F_ext}")
        ax.legend()
        fig_path = output_dir / f"{eid}_multi_model_fit.png"
        fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        figures.append(str(fig_path))

    # ---------- 构建观察 (中文) ----------
    obs_lines = ["对所有恒外力实验进行联合拟合分析。使用序列：a_sg（exp_08用a_new）、v_sg、v_sq。"]
    for eid in experiment_ids:
        obs_lines.append(f"{eid}:")
        m = metrics
        obs_lines.append(f"  模型1 (a = β0 + β1*v + β2*v²): β0={m[f'{eid}_beta0']:.5f}, β1={m[f'{eid}_beta1']:.5f}, β2={m[f'{eid}_beta2']:.5f}, R²={m[f'{eid}_model1_R2']:.4f}, RMSE={m[f'{eid}_model1_RMSE']:.5f}")
        obs_lines.append(f"    β0 与 F_ext 的差值: {m[f'{eid}_beta0_diff']:.5f}, 相对差值: {m[f'{eid}_beta0_diff_rel']:.3%}")
        obs_lines.append(f"  模型2 (a = F_ext - γ*v): γ={m[f'{eid}_gamma']:.5f}, R²={m[f'{eid}_model2_R2']:.4f}, RMSE={m[f'{eid}_model2_RMSE']:.5f}")
        if m[f'{eid}_v_max'] != float('inf'):
            obs_lines.append(f"  模型3 v_max = F_ext/γ = {m[f'{eid}_v_max']:.5f}")
        else:
            obs_lines.append(f"  模型3: γ=0, v_max 无定义")

    obs_lines.append("残差序列已返回为派生序列 'residual_model1' 和 'residual_model2'。")
    obs_lines.append("每个实验的散点图与拟合曲线已保存。")

    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series_all,
        "figures": figures,
        "metrics": metrics
    }
