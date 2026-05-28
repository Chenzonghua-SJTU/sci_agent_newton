import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

def process(payload: dict) -> dict:
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]
    
    # 收集每个实验的 a-v 线性回归结果
    av_results = {}  # exp_id -> {intercept, slope, r2, F_ext, field_type}
    for eid in params.get("experiment_ids", experiments.keys()):
        if eid not in experiments:
            # 跳过不存在的实验
            continue
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        available = exp.get("available_series", [])
        # 确保有 a 和 v 序列
        if "a" not in series or "v" not in series:
            continue
        a = np.array(series["a"])
        v = np.array(series["v"])
        # 检查方差是否非零
        if np.var(v) < 1e-12:
            # v 完全恒定，回归无意义，人工设定斜率=0，截距=a均值，R2=0
            intercept = float(np.mean(a))
            slope = 0.0
            r2 = 0.0
        else:
            # 使用 numpy 进行线性拟合
            coeffs = np.polyfit(v, a, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            # 计算 R²
            a_pred = np.polyval(coeffs, v)
            ss_res = np.sum((a - a_pred) ** 2)
            ss_tot = np.sum((a - np.mean(a)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        # 获取外力
        F_ext = float(config["F_ext"])
        field_type = config["force_field_type"]
        av_results[eid] = {
            "intercept": intercept,
            "slope": slope,
            "r2": r2,
            "F_ext": F_ext,
            "field_type": field_type
        }
    
    # 分离 constant 和 free 实验
    const_exps = {eid: r for eid, r in av_results.items() if r["field_type"] == "constant"}
    free_exps = {eid: r for eid, r in av_results.items() if r["field_type"] == "free"}
    
    observations = []
    derived_series = []
    figures = []
    
    # 1. 记录每个实验的回归结果作为 OBS
    for eid, r in av_results.items():
        obs = {
            "summary": f"实验{eid} a-v线性回归: 截距={r['intercept']:.6f}, 斜率={r['slope']:.6f}, R²={r['r2']:.6f}",
            "source_data_refs": [f"{eid}:a", f"{eid}:v"],
            "metrics": {
                "intercept": r['intercept'],
                "slope": r['slope'],
                "r2": r['r2'],
                "F_ext": r['F_ext'],
                "field_type": r['field_type']
            }
        }
        observations.append(obs)
    
    # 2. 对 constant 场实验，拟合截距 vs F_ext 和 斜率 vs F_ext
    const_F = np.array([r["F_ext"] for r in const_exps.values()])
    const_intercepts = np.array([r["intercept"] for r in const_exps.values()])
    const_slopes = np.array([r["slope"] for r in const_exps.values()])
    
    if len(const_F) >= 2:
        # 截距 vs F_ext 线性回归
        coeffs_int = np.polyfit(const_F, const_intercepts, 1)
        slope_int_vs_F = coeffs_int[0]
        intercept_int_vs_F = coeffs_int[1]
        pred_int = np.polyval(coeffs_int, const_F)
        ss_res_int = np.sum((const_intercepts - pred_int) ** 2)
        ss_tot_int = np.sum((const_intercepts - np.mean(const_intercepts)) ** 2)
        r2_int = 1.0 - ss_res_int / ss_tot_int if ss_tot_int > 0 else 0.0
        
        # 斜率 vs F_ext 线性回归
        coeffs_slp = np.polyfit(const_F, const_slopes, 1)
        slope_slp_vs_F = coeffs_slp[0]
        intercept_slp_vs_F = coeffs_slp[1]
        pred_slp = np.polyval(coeffs_slp, const_F)
        ss_res_slp = np.sum((const_slopes - pred_slp) ** 2)
        ss_tot_slp = np.sum((const_slopes - np.mean(const_slopes)) ** 2)
        r2_slp = 1.0 - ss_res_slp / ss_tot_slp if ss_tot_slp > 0 else 0.0
        
        # 记录截距 vs F_ext 结果
        obs_int = {
            "summary": f"常数场实验: 截距 vs F_ext 线性回归: 斜率={slope_int_vs_F:.6f}, 截距={intercept_int_vs_F:.6f}, R²={r2_int:.6f}",
            "source_data_refs": [f"exp_02:a", f"exp_02:v", f"exp_03:a", f"exp_03:v", f"exp_05:a", f"exp_05:v", f"exp_06:a", f"exp_06:v", f"exp_08:a", f"exp_08:v", f"exp_09:a", f"exp_09:v", f"exp_10:a", f"exp_10:v", f"exp_11:a", f"exp_11:v", f"exp_12:a", f"exp_12:v", f"exp_13:a", f"exp_13:v", f"exp_14:a", f"exp_14:v", f"exp_15:a", f"exp_15:v"],
            "metrics": {
                "regression_type": "intercept_vs_Fext",
                "slope": slope_int_vs_F,
                "intercept": intercept_int_vs_F,
                "r2": r2_int,
                "n_points": len(const_F)
            }
        }
        observations.append(obs_int)
        
        # 记录斜率 vs F_ext 结果
        obs_slp = {
            "summary": f"常数场实验: 斜率 vs F_ext 线性回归: 斜率={slope_slp_vs_F:.6f}, 截距={intercept_slp_vs_F:.6f}, R²={r2_slp:.6f}",
            "source_data_refs": [f"exp_02:a", f"exp_02:v", f"exp_03:a", f"exp_03:v", f"exp_05:a", f"exp_05:v", f"exp_06:a", f"exp_06:v", f"exp_08:a", f"exp_08:v", f"exp_09:a", f"exp_09:v", f"exp_10:a", f"exp_10:v", f"exp_11:a", f"exp_11:v", f"exp_12:a", f"exp_12:v", f"exp_13:a", f"exp_13:v", f"exp_14:a", f"exp_14:v", f"exp_15:a", f"exp_15:v"],
            "metrics": {
                "regression_type": "slope_vs_Fext",
                "slope": slope_slp_vs_F,
                "intercept": intercept_slp_vs_F,
                "r2": r2_slp,
                "n_points": len(const_F)
            }
        }
        observations.append(obs_slp)
    else:
        # 常数场实验不足2个，无法拟合
        obs_warn = {
            "summary": "常数场实验数量不足，无法进行截距/slope vs F_ext 回归",
            "source_data_refs": [],
            "metrics": {"constant_experiment_count": len(const_F)}
        }
        observations.append(obs_warn)
    
    # 3. 自由场实验的截距和斜率统计
    if free_exps:
        free_intercepts = [r["intercept"] for r in free_exps.values()]
        free_slopes = [r["slope"] for r in free_exps.values()]
        intercept_mean = np.mean(free_intercepts)
        intercept_std = np.std(free_intercepts, ddof=0)
        slope_mean = np.mean(free_slopes)
        slope_std = np.std(free_slopes, ddof=0)
        obs_free = {
            "summary": f"自由场实验({len(free_exps)}个): 截距均值={intercept_mean:.6f}, 截距标准差={intercept_std:.6f}; 斜率均值={slope_mean:.6f}, 斜率标准差={slope_std:.6f}",
            "source_data_refs": [f"{eid}:a" for eid in free_exps.keys()] + [f"{eid}:v" for eid in free_exps.keys()],
            "metrics": {
                "free_experiment_count": len(free_exps),
                "intercept_mean": intercept_mean,
                "intercept_std": intercept_std,
                "slope_mean": slope_mean,
                "slope_std": slope_std
            }
        }
        observations.append(obs_free)
    else:
        obs_free = {
            "summary": "没有自由场实验数据",
            "source_data_refs": [],
            "metrics": {"free_experiment_count": 0}
        }
        observations.append(obs_free)
    
    # 4. 生成散点图: 截距 vs F_ext 和 斜率 vs F_ext
    if len(const_F) >= 2:
        # 截距 vs F_ext 散点图
        fig1, ax1 = plt.subplots(figsize=(6,5))
        ax1.scatter(const_F, const_intercepts, color='blue', label='Constant field experiments')
        # 添加回归线
        f_range = np.linspace(min(const_F), max(const_F), 100)
        pred_line = slope_int_vs_F * f_range + intercept_int_vs_F
        ax1.plot(f_range, pred_line, 'r--', label=f'Linear fit (R²={r2_int:.3f})')
        ax1.set_xlabel('F_ext')
        ax1.set_ylabel('Intercept (a-v regression)')
        ax1.set_title('Intercept vs External Force')
        ax1.legend()
        plt.tight_layout()
        fig_path1 = Path(output_dir) / "intercept_vs_F_ext.png"
        fig1.savefig(fig_path1, dpi=150)
        plt.close(fig1)
        figures.append(str(fig_path1))
        
        # 斜率 vs F_ext 散点图
        fig2, ax2 = plt.subplots(figsize=(6,5))
        ax2.scatter(const_F, const_slopes, color='green', label='Constant field experiments')
        pred_line2 = slope_slp_vs_F * f_range + intercept_slp_vs_F
        ax2.plot(f_range, pred_line2, 'r--', label=f'Linear fit (R²={r2_slp:.3f})')
        ax2.set_xlabel('F_ext')
        ax2.set_ylabel('Slope (a-v regression)')
        ax2.set_title('Slope vs External Force')
        ax2.legend()
        plt.tight_layout()
        fig_path2 = Path(output_dir) / "slope_vs_F_ext.png"
        fig2.savefig(fig_path2, dpi=150)
        plt.close(fig2)
        figures.append(str(fig_path2))
    
    # 构建总 observation 字符串（给决策 LLM 看的简洁中文）
    obs_summary_parts = []
    obs_summary_parts.append(f"处理了{len(av_results)}个实验的a-v线性回归。")
    if const_exps:
        obs_summary_parts.append(f"常数场实验{len(const_exps)}个: 截距范围[{min(const_intercepts):.3f}, {max(const_intercepts):.3f}], 斜率范围[{min(const_slopes):.3f}, {max(const_slopes):.3f}]。")
    if free_exps:
        obs_summary_parts.append(f"自由场实验{len(free_exps)}个: 截距均值={intercept_mean:.6f}, 斜率均值={slope_mean:.6f}。")
    if len(const_F) >= 2:
        obs_summary_parts.append(f"截距vsF_ext回归: 斜率={slope_int_vs_F:.6f}, R²={r2_int:.6f}; 斜率vsF_ext回归: 斜率={slope_slp_vs_F:.6f}, R²={r2_slp:.6f}。")
    obs_summary_parts.append(f"共生成{len(observations)}条OBS记录, 图像{len(figures)}张。")
    observation_text = "".join(obs_summary_parts)
    
    # metrics 汇总
    metrics = {
        "experiments_analyzed": len(av_results),
        "constant_experiment_count": len(const_exps),
        "free_experiment_count": len(free_exps),
        "observation_count": len(observations),
        "figure_count": len(figures)
    }
    if len(const_F) >= 2:
        metrics["intercept_vs_F_r2"] = r2_int
        metrics["slope_vs_F_r2"] = r2_slp
    
    return {
        "observation": observation_text,
        "observations": observations,
        "figures": figures,
        "derived_series": derived_series,
        "metrics": metrics
    }
