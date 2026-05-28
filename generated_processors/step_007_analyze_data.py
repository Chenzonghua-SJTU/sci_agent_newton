import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import scipy
from sklearn import linear_model, metrics, preprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_v_a(t: np.ndarray, q: np.ndarray) -> tuple:
    """使用中心差分计算速度和加速度序列。
    优先使用 np.gradient，edge_order=2。
    """
    if len(t) < 2:
        raise ValueError("时间序列长度不足2，无法差分")
    v = np.gradient(q, t, edge_order=2)
    a = np.gradient(v, t, edge_order=2)
    return v, a

def process(payload: dict) -> dict:
    # 提取参数
    params = payload.get("parameters", {})
    analysis_mode = params.get("analysis_mode", "maintain_ledger")
    if analysis_mode != "maintain_ledger":
        raise ValueError(f"仅支持 maintain_ledger 模式，收到: {analysis_mode}")
    
    experiment_ids = params.get("experiment_ids", None)
    if experiment_ids is None:
        experiment_ids = list(payload.get("experiments", {}).keys())
    else:
        # 确保所有实验都存在
        available = payload.get("experiments", {})
        for eid in experiment_ids:
            if eid not in available:
                raise ValueError(f"实验 {eid} 不在 payload.experiments 中")
    
    output_dir = Path(payload.get("output_dir", "."))
    
    derived_series = []
    observations = []
    figures = []
    
    # 用于收集跨实验数据绘图
    plot_data = []  # 每个元素: (v, a, F_ext, v0, experiment_id)
    
    # 逐实验处理
    for eid in experiment_ids:
        exp = payload["experiments"][eid]
        config = exp.get("config", {})
        series = exp.get("series", {})
        available_series = exp.get("available_series", [])
        
        t = np.array(series.get("t", []))
        q = np.array(series.get("q", []))
        if len(t) == 0:
            raise ValueError(f"实验 {eid} 缺少 t 序列")
        if len(q) == 0:
            raise ValueError(f"实验 {eid} 缺少 q 序列")
        if len(t) != len(q):
            raise ValueError(f"实验 {eid} 的 t 和 q 长度不一致")
        
        # 获取现有 v, a 序列
        v_existing = series.get("v", None)
        a_existing = series.get("a", None)
        
        # 如果已有 v 和 a，直接使用；否则计算
        if v_existing is not None and a_existing is not None:
            v = np.array(v_existing)
            a = np.array(a_existing)
            # 验证长度一致
            if len(v) != len(t) or len(a) != len(t):
                raise ValueError(f"实验 {eid} 中已有 v/a 序列长度与 t 不一致")
        else:
            v, a = compute_v_a(t, q)
            # 注册新派生序列
            derived_series.append({
                "experiment_id": eid,
                "name": "v",
                "values": v.tolist(),
                "source_name": "np.gradient(q, t, edge_order=2)",
                "provenance": f"generated data processor: maintain_ledger for {eid}",
                "description": "中心差分速度 (从q)"
            })
            derived_series.append({
                "experiment_id": eid,
                "name": "a",
                "values": a.tolist(),
                "source_name": "np.gradient(v, t, edge_order=2)",
                "provenance": f"generated data processor: maintain_ledger for {eid}",
                "description": "中心差分加速度 (从v)"
            })
        
        # 提取 F_ext 和 v0
        F_ext = config.get("F_ext", None)
        if F_ext is None:
            F_ext = 0.0  # 默认自由场
        v0 = config.get("initial_v", 0.0)
        
        # 记录初始和最终的 v, a
        init_v = float(v[0])
        final_v = float(v[-1])
        init_a = float(a[0])
        final_a = float(a[-1])
        
        obs_entry = {
            "summary": f"实验 {eid}: F_ext={F_ext}, v0={v0}, 初始 v={init_v:.6f}, 最终 v={final_v:.6f}, 初始 a={init_a:.6f}, 最终 a={final_a:.6f}",
            "source_data_refs": [f"{eid}:t", f"{eid}:q"],
            "metrics": {
                "F_ext": F_ext,
                "initial_v": init_v,
                "final_v": final_v,
                "initial_a": init_a,
                "final_a": final_a,
                "v0": v0
            }
        }
        observations.append(obs_entry)
        
        # 将点加入跨实验数据 (每序列的每个点都包含? 数据量不大, 可以加入)
        for vi, ai in zip(v, a):
            plot_data.append((vi, ai, F_ext, v0, eid))
    
    # 生成跨实验散点图
    if plot_data:
        df_plot = pd.DataFrame(plot_data, columns=["v","a","F_ext","v0","experiment_id"])
        # 按 F_ext 和 v0 组合分组 (使用字符串组合)
        df_plot["group"] = df_plot.apply(lambda row: f"F={row['F_ext']},v0={row['v0']}", axis=1)
        unique_groups = df_plot["group"].unique()
        colors = plt.cm.tab20(np.linspace(0,1,len(unique_groups)))
        color_map = {g: c for g,c in zip(unique_groups, colors)}
        
        fig, ax = plt.subplots(figsize=(10, 8))
        for g, gdf in df_plot.groupby("group"):
            ax.scatter(gdf["v"], gdf["a"], label=g, color=color_map[g], s=10, alpha=0.7)
        ax.set_xlabel("速度 v")
        ax.set_ylabel("加速度 a")
        ax.set_title("跨实验 a-v 散点图 (按 F_ext & v0 分组)")
        ax.legend(loc='best', fontsize=8)
        fig_path = output_dir / "cross_experiment_a_v_scatter.png"
        fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        figures.append(str(fig_path))
    
    # 生成汇总数值事实 OBS
    # 1. 每个不同 F_ext 下, v接近0附近的a (截距)
    # 我们直接从 plot_data 中过滤 |v| < 0.01 的点, 计算平均a
    fact_observations = []
    all_F_ext = sorted(set(p[2] for p in plot_data))
    for fext in all_F_ext:
        pts = [(v,a) for v,a,F,v0,eid in plot_data if F==fext and abs(v)<0.01]
        if len(pts) > 0:
            avg_a = np.mean([p[1] for p in pts])
            facts = {
                "summary": f"F_ext={fext} 时 |v|<0.01 的平均加速度 a ≈ {avg_a:.6f}",
                "source_data_refs": [f"cross_experiment:v,a for F_ext={fext}"],
                "metrics": {
                    "F_ext": fext,
                    "v_near_zero_samples": len(pts),
                    "mean_acceleration_near_zero": avg_a
                }
            }
            fact_observations.append(facts)
    
    # 2. 每个 v0 分组下的直线趋势 (只输出数值事实，不拟合公式)
    # 可以报告每个 F_ext 下 a 与 v 的线性相关程度 (Pearson r)
    for fext in all_F_ext:
        pts = [(v,a) for v,a,F,v0,eid in plot_data if F==fext]
        if len(pts) >= 3:
            v_vals = [p[0] for p in pts]
            a_vals = [p[1] for p in pts]
            corr = np.corrcoef(v_vals, a_vals)[0,1]
            fact_observations.append({
                "summary": f"F_ext={fext} 时 a 与 v 的 Pearson 相关系数 r ≈ {corr:.6f}, 点数={len(pts)}",
                "source_data_refs": [f"cross_experiment:v,a for F_ext={fext}"],
                "metrics": {
                    "F_ext": fext,
                    "pearson_r": corr,
                    "sample_count": len(pts)
                }
            })
    
    # 3. 不同 F_ext 下截距 (v=0 附近 a) 与 F_ext 的关系
    # 我们已经有了 fact_observations[0..n] 中的截距信息
    # 再汇总一个
    intercepts = []
    for obs in fact_observations:
        if "mean_acceleration_near_zero" in obs["metrics"]:
            intercepts.append((obs["metrics"]["F_ext"], obs["metrics"]["mean_acceleration_near_zero"]))
    if len(intercepts) >= 2:
        F_intercepts = [x[0] for x in intercepts]
        a_intercepts = [x[1] for x in intercepts]
        summary = "不同 F_ext 下 v≈0 时的平均加速度："
        for i, (F, a_int) in enumerate(intercepts):
            summary += f" F={F} -> a≈{a_int:.4f};"
        fact_observations.append({
            "summary": summary,
            "source_data_refs": [f"cross_experiment:v,a near zero"],
            "metrics": {
                "intercept_list": {f"F_{x[0]}": x[1] for x in intercepts},
                "observation_count": len(intercepts)
            }
        })
    
    observations.extend(fact_observations)
    
    # 4. 记录整体观测统计
    observations.append({
        "summary": f"共处理 {len(experiment_ids)} 个实验，生成 {len(derived_series)} 个新派生序列，{len(observations)} 条观测",
        "source_data_refs": [e+":t" for e in experiment_ids],
        "metrics": {
            "experiments_processed": len(experiment_ids),
            "derived_series_count": len(derived_series),
            "observation_count": len(observations),
            "figure_count": len(figures)
        }
    })
    
    return {
        "observation": f"维护实验数据记录表完成。处理 {len(experiment_ids)} 个实验，"
                       f"计算了速度、加速度派生序列（已存在的直接复用），"
                       f"记录各实验初始/最终速度加速度值，"
                       f"生成跨实验 a-v 散点图（{len(figures)} 张），"
                       f"并记录 F_ext 分组下的截距与线性相关数值事实。"
                       f"未提出任何物理定律，仅包含可核验数值。",
        "derived_series": derived_series,
        "observations": observations,
        "figures": figures,
        "metrics": {
            "experiments_processed": len(experiment_ids),
            "derived_series_count": len(derived_series),
            "observation_count": len(observations),
            "figure_count": len(figures)
        }
    }
