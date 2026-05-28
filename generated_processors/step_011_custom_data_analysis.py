import json
import math
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiment_ids = params.get("experiment_ids", params.get("experiment_id", []))
    if isinstance(experiment_ids, str):
        experiment_ids = [experiment_ids]
    if not experiment_ids:
        experiment_ids = list(payload.get("experiments", {}).keys())

    output_dir = payload.get("output_dir", ".")
    experiments = payload.get("experiments", {})
    # 检查必要实验
    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
        exp = experiments[eid]
        for s_name in ['q', 'v_sg_long', 'a_sg_long', 't']:
            if s_name not in exp.get("series", {}):
                raise ValueError(f"Series '{s_name}' missing in experiment {eid}")

    results = {}
    metrics = OrderedDict()
    fig_paths = []

    # 准备数据
    data = {}
    for eid in experiment_ids:
        exp = experiments[eid]
        t = np.array(exp["series"]["t"])
        q = np.array(exp["series"]["q"])
        v = np.array(exp["series"]["v_sg_long"])
        a = np.array(exp["series"]["a_sg_long"])
        if not (len(t) == len(q) == len(v) == len(a)):
            raise ValueError(f"Series length mismatch in {eid}")
        data[eid] = {"t": t, "q": q, "v": v, "a": a}

    # 拟合两个多变量模型: a = b0 + b1*v + b2*q  和 a = b0 + b1*v^2 + b2*q
    model_results = {}
    for eid, d in data.items():
        v = d["v"]
        a = d["a"]
        q = d["q"]
        n = len(v)

        # 模型1: [1, v, q]
        X1 = np.column_stack([np.ones(n), v, q])
        coeff1, res1, rank1, s1 = np.linalg.lstsq(X1, a, rcond=None)
        a_pred1 = X1 @ coeff1
        resid1 = a - a_pred1
        rss1 = np.sum(resid1**2)
        tss1 = np.sum((a - np.mean(a))**2)
        r2_1 = 1 - rss1 / tss1 if tss1 != 0 else 0.0
        rmse1 = np.sqrt(rss1 / n)

        # 标准误估计
        sigma1 = np.sqrt(rss1 / (n - 3)) if n > 3 else 0.0
        try:
            cov1 = sigma1**2 * np.linalg.inv(X1.T @ X1)
            se1 = np.sqrt(np.diag(cov1))
        except np.linalg.LinAlgError:
            se1 = np.full(3, np.nan)

        # 模型2: [1, v^2, q]
        X2 = np.column_stack([np.ones(n), v**2, q])
        coeff2, res2, rank2, s2 = np.linalg.lstsq(X2, a, rcond=None)
        a_pred2 = X2 @ coeff2
        resid2 = a - a_pred2
        rss2 = np.sum(resid2**2)
        r2_2 = 1 - rss2 / tss1 if tss1 != 0 else 0.0
        rmse2 = np.sqrt(rss2 / n)
        sigma2 = np.sqrt(rss2 / (n - 3)) if n > 3 else 0.0
        try:
            cov2 = sigma2**2 * np.linalg.inv(X2.T @ X2)
            se2 = np.sqrt(np.diag(cov2))
        except np.linalg.LinAlgError:
            se2 = np.full(3, np.nan)

        model_results[eid] = {
            "model1": {
                "b0": coeff1[0],
                "b1": coeff1[1],
                "b2": coeff1[2],
                "se_b0": se1[0],
                "se_b1": se1[1],
                "se_b2": se1[2],
                "R2": r2_1,
                "RMSE": rmse1,
                "residuals": resid1.tolist(),
                "predictions": a_pred1.tolist()
            },
            "model2": {
                "b0": coeff2[0],
                "b1": coeff2[1],
                "b2": coeff2[2],
                "se_b0": se2[0],
                "se_b1": se2[1],
                "se_b2": se2[2],
                "R2": r2_2,
                "RMSE": rmse2,
                "residuals": resid2.tolist(),
                "predictions": a_pred2.tolist()
            }
        }

        # 记录metrics
        prefix = eid
        for mod, tag in [("model1", "linear_q"), ("model2", "quad_q")]:
            m = model_results[eid][mod]
            metrics[f"{prefix}_{tag}_b0"] = m["b0"]
            metrics[f"{prefix}_{tag}_b1"] = m["b1"]
            metrics[f"{prefix}_{tag}_b2"] = m["b2"]
            metrics[f"{prefix}_{tag}_R2"] = m["R2"]
            metrics[f"{prefix}_{tag}_RMSE"] = m["RMSE"]

    # 跨实验一致性：比较系数（以模型1为例，看b0,b1,b2的差异范围）
    for tag, mod in [("linear_q", "model1"), ("quad_q", "model2")]:
        b0_vals = [model_results[e][mod]["b0"] for e in experiment_ids]
        b1_vals = [model_results[e][mod]["b1"] for e in experiment_ids]
        b2_vals = [model_results[e][mod]["b2"] for e in experiment_ids]
        metrics[f"cross_{tag}_b0_range"] = max(b0_vals) - min(b0_vals)
        metrics[f"cross_{tag}_b1_range"] = max(b1_vals) - min(b1_vals)
        metrics[f"cross_{tag}_b2_range"] = max(b2_vals) - min(b2_vals)
        # 报告系数均值
        metrics[f"cross_{tag}_b0_mean"] = np.mean(b0_vals)
        metrics[f"cross_{tag}_b1_mean"] = np.mean(b1_vals)
        metrics[f"cross_{tag}_b2_mean"] = np.mean(b2_vals)

    # 跨实验比较相同速度区间内的a值
    # 找出三个实验速度的公共区间
    v_mins = [np.min(data[e]["v"]) for e in experiment_ids]
    v_maxs = [np.max(data[e]["v"]) for e in experiment_ids]
    v_low = max(v_mins)
    v_high = min(v_maxs)
    metrics["common_v_low"] = v_low
    metrics["common_v_high"] = v_high
    if v_low < v_high:
        a_means_common = {}
        for eid in experiment_ids:
            v = data[eid]["v"]
            a = data[eid]["a"]
            mask = (v >= v_low) & (v <= v_high)
            if np.sum(mask) > 0:
                a_means_common[eid] = np.mean(a[mask])
            else:
                a_means_common[eid] = np.nan
        metrics["common_v_interval_a_means"] = a_means_common
        if any(np.isfinite(v) for v in a_means_common.values()):
            valid_means = [v for v in a_means_common.values() if np.isfinite(v)]
            metrics["common_v_interval_a_max_diff"] = max(valid_means) - min(valid_means) if len(valid_means) > 1 else 0.0
        else:
            metrics["common_v_interval_a_max_diff"] = np.nan
    else:
        metrics["common_v_interval_a_means"] = {}
        metrics["common_v_interval_a_max_diff"] = np.nan

    # 生成图像
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # 左图: a vs v 散点 + 公共区间均值线
    ax1 = axes[0]
    colors = {'exp_03': 'blue', 'exp_04': 'green', 'exp_05': 'red'}
    for eid in experiment_ids:
        v = data[eid]["v"]
        a = data[eid]["a"]
        ax1.scatter(v, a, c=colors.get(eid, 'gray'), s=5, alpha=0.6, label=eid)
    # 公共区间均值
    if v_low < v_high and any(np.isfinite(v) for v in metrics.get("common_v_interval_a_means", {}).values()):
        for eid in experiment_ids:
            if eid in a_means_common and np.isfinite(a_means_common[eid]):
                ax1.axhline(y=a_means_common[eid], color=colors.get(eid, 'gray'), linestyle='--', linewidth=1, alpha=0.7, label=f'{eid} mean in common v')
    ax1.set_xlabel('v_sg_long')
    ax1.set_ylabel('a_sg_long')
    ax1.set_title('a vs v (three experiments)')
    ax1.legend(fontsize=7)

    # 右图: 模型1 (a ~ v + q) 的预测值与实际值对比
    ax2 = axes[1]
    for eid in experiment_ids:
        a_actual = data[eid]["a"]
        a_pred = np.array(model_results[eid]["model1"]["predictions"])
        ax2.scatter(a_actual, a_pred, c=colors.get(eid, 'gray'), s=5, alpha=0.6, label=eid)
    ax2.plot([a.min(), a.max()], [a.min(), a.max()], 'k--', linewidth=0.8)
    ax2.set_xlabel('Actual a')
    ax2.set_ylabel('Predicted a (model1)')
    ax2.set_title('Model1: a = b0 + b1*v + b2*q')
    ax2.legend(fontsize=7)
    plt.tight_layout()
    fig_path = f"{output_dir}/multi_model_fit_{'_'.join(experiment_ids)}.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    fig_paths.append(fig_path)

    # 为每个实验生成单独的残差图（可选），但为避免过多，只生成一个综合图
    # 可以额外生成一个残差直方图
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
    for i, tag in enumerate([("Model1 residuals", "model1"), ("Model2 residuals", "model2")]):
        ax = axes2[i]
        for eid in experiment_ids:
            res = np.array(model_results[eid][tag[1]]["residuals"])
            ax.hist(res, bins=20, alpha=0.5, label=eid, color=colors.get(eid, 'gray'))
        ax.set_xlabel('Residual')
        ax.set_ylabel('Count')
        ax.set_title(tag[0])
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig_path2 = f"{output_dir}/residuals_hist_{'_'.join(experiment_ids)}.png"
    plt.savefig(fig_path2, dpi=150)
    plt.close()
    fig_paths.append(fig_path2)

    # 构造 observation 中文描述
    obs_lines = []
    obs_lines.append("恒力实验(exp03,exp04,exp05)联合多变量模型分析结果：")
    for eid in experiment_ids:
        obs_lines.append(f"  实验 {eid}:")
        m1 = model_results[eid]["model1"]
        m2 = model_results[eid]["model2"]
        obs_lines.append(f"    模型1 (a = b0 + b1*v + b2*q): b0={m1['b0']:.4f}(se={m1['se_b0']:.4f}), b1={m1['b1']:.4f}(se={m1['se_b1']:.4f}), b2={m1['b2']:.4f}(se={m1['se_b2']:.4f}), R2={m1['R2']:.4f}, RMSE={m1['RMSE']:.4f}")
        obs_lines.append(f"    模型2 (a = b0 + b1*v^2 + b2*q): b0={m2['b0']:.4f}(se={m2['se_b0']:.4f}), b1={m2['b1']:.4f}(se={m2['se_b1']:.4f}), b2={m2['b2']:.4f}(se={m2['se_b2']:.4f}), R2={m2['R2']:.4f}, RMSE={m2['RMSE']:.4f}")

    obs_lines.append(f"跨实验系数一致性（模型1）: b0极差={metrics['cross_linear_q_b0_range']:.4f}, b1极差={metrics['cross_linear_q_b1_range']:.4f}, b2极差={metrics['cross_linear_q_b2_range']:.4f}")
    obs_lines.append(f"跨实验系数一致性（模型2）: b0极差={metrics['cross_quad_q_b0_range']:.4f}, b1极差={metrics['cross_quad_q_b1_range']:.4f}, b2极差={metrics['cross_quad_q_b2_range']:.4f}")

    if v_low < v_high:
        obs_lines.append(f"三个实验速度公共区间 [{v_low:.3f}, {v_high:.3f}] 内加速度均值：")
        for eid in experiment_ids:
            if eid in a_means_common and np.isfinite(a_means_common[eid]):
                obs_lines.append(f"  {eid}: {a_means_common[eid]:.4f}")
        if len(valid_means) > 1:
            obs_lines.append(f"  最大差异={metrics['common_v_interval_a_max_diff']:.4f}")
        else:
            obs_lines.append(f"  仅一个实验有数据点，无法比较")
    else:
        obs_lines.append("三个实验速度范围无明显公共重叠区间，无法进行相同速度区间比较。")

    obs = "\n".join(obs_lines)

    # 构建 derived_series：每个实验的预测值和残差（可选，但有助于后续分析）
    derived_series = []
    for eid in experiment_ids:
        for tag, mod in [("linear_q", "model1"), ("quad_q", "model2")]:
            pred_name = f"a_pred_{tag}"
            res_name = f"a_res_{tag}"
            derived_series.append({
                "experiment_id": eid,
                "name": pred_name,
                "values": model_results[eid][mod]["predictions"],
                "source_name": f"multi-variable regression (b0+b1*v+b2*q or b0+b1*v^2+b2*q)",
                "provenance": "generated data processor: custom_data_analysis",
                "description": f"Predicted acceleration from model {tag}"
            })
            derived_series.append({
                "experiment_id": eid,
                "name": res_name,
                "values": model_results[eid][mod]["residuals"],
                "source_name": f"residual = a - a_pred_{tag}",
                "provenance": "generated data processor: custom_data_analysis",
                "description": f"Residual from model {tag}"
            })

    return {
        "observation": obs,
        "derived_series": derived_series,
        "figures": fig_paths,
        "metrics": dict(metrics)
    }
