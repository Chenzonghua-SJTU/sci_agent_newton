import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload.get("parameters", {})
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]
    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        experiment_ids = list(experiments.keys())
    target_ids = [eid for eid in experiment_ids if eid in experiments]
    if not target_ids:
        return {"observation": "指定的 experiment_ids 在数据中不存在", "derived_series": [], "figures": [], "metrics": {}}
    # 只处理 F_ext=1 的实验（exp_04, exp_06, exp_08）
    F1_ids = ["exp_04", "exp_06", "exp_08"]
    F1_ids = [eid for eid in F1_ids if eid in target_ids and eid in experiments]
    if len(F1_ids) < 2:
        return {"observation": "相同外力F=1的实验不足2个", "derived_series": [], "figures": [], "metrics": {}}
    # 获取序列，若缺失则自动估计
    series_data = {}
    derived_new = []
    missing_any = False
    for eid in F1_ids:
        exp = experiments[eid]
        config = exp.get("config", {})
        ds = {}
        # 检查 a_sg
        if "a_sg" in exp.get("series", {}):
            ds["a"] = np.array(exp["series"]["a_sg"])
        else:
            # 尝试从 q 生成
            if "q" in exp.get("series", {}):
                q = np.array(exp["series"]["q"])
                t = np.array(exp["series"]["t"])
                dt = t[1] - t[0] if len(t) > 1 else 0.1
                # 检查是否已有 v_sg
                if "v_sg" in exp.get("series", {}):
                    v = np.array(exp["series"]["v_sg"])
                else:
                    # 估计 v
                    v_est = savgol_filter(q, window_length=11, polyorder=3, deriv=1, delta=dt)
                    v = v_est
                    derived_new.append({
                        "experiment_id": eid,
                        "name": "v_sg",
                        "values": v.tolist(),
                        "source_name": "savgol_filter(q, window=11, polyorder=3, deriv=1, delta=dt)",
                        "provenance": "generated data processor: custom_data_analysis",
                        "description": "自动从 q 估计的速度"
                    })
                a_est = savgol_filter(v, window_length=11, polyorder=3, deriv=1, delta=dt) if len(v) >= 11 else np.gradient(v, dt)
                ds["a"] = a_est
                derived_new.append({
                    "experiment_id": eid,
                    "name": "a_sg",
                    "values": a_est.tolist(),
                    "source_name": "savgol_filter(v, window=11, polyorder=3, deriv=1, delta=dt) 或 np.gradient",
                    "provenance": "generated data processor: custom_data_analysis",
                    "description": "自动从 v 估计的加速度"
                })
                missing_any = True
            else:
                raise ValueError(f"实验 {eid} 缺少 q 序列，无法估计 a_sg")
        # 检查 v_sg
        if "v_sg" in exp.get("series", {}):
            ds["v"] = np.array(exp["series"]["v_sg"])
        else:
            # 从 q 估计
            if "q" in exp.get("series", {}):
                q = np.array(exp["series"]["q"])
                t = np.array(exp["series"]["t"])
                dt = t[1] - t[0] if len(t) > 1 else 0.1
                v_est = savgol_filter(q, window_length=11, polyorder=3, deriv=1, delta=dt)
                ds["v"] = v_est
                derived_new.append({
                    "experiment_id": eid,
                    "name": "v_sg",
                    "values": v_est.tolist(),
                    "source_name": "savgol_filter(q, window=11, polyorder=3, deriv=1, delta=dt)",
                    "provenance": "generated data processor: custom_data_analysis",
                    "description": "自动从 q 估计的速度"
                })
                missing_any = True
            else:
                raise ValueError(f"实验 {eid} 缺少 q 序列，无法估计 v_sg")
        series_data[eid] = ds
    # 颜色和标签
    colors = {'exp_04': 'blue', 'exp_06': 'green', 'exp_08': 'red'}
    labels = {'exp_04': 'exp_04 (v0=0)', 'exp_06': 'exp_06 (v0=10)', 'exp_08': 'exp_08 (v0=5)'}
    # 拟合二次多项式 a = c0 + c1*v + c2*v^2
    fit_results = {}
    all_v = []
    all_a = []
    for eid in F1_ids:
        v = series_data[eid]["v"]
        a = series_data[eid]["a"]
        # 去除可能的 NaN
        mask = ~(np.isnan(v) | np.isnan(a))
        v_clean = v[mask]
        a_clean = a[mask]
        if len(v_clean) < 5:
            fit_results[eid] = {"coeffs": None, "r2": None, "n": 0}
            continue
        # 二次拟合
        coeffs = np.polyfit(v_clean, a_clean, 2)
        a_pred = np.polyval(coeffs, v_clean)
        r2 = r2_score(a_clean, a_pred)
        fit_results[eid] = {
            "coeffs": coeffs.tolist(),
            "r2": r2,
            "n": len(v_clean)
        }
        all_v.extend(v_clean.tolist())
        all_a.extend(a_clean.tolist())
    # 合并拟合
    all_v = np.array(all_v)
    all_a = np.array(all_a)
    if len(all_v) >= 5:
        merged_coeffs = np.polyfit(all_v, all_a, 2)
        merged_pred = np.polyval(merged_coeffs, all_v)
        merged_r2 = r2_score(all_a, merged_pred)
    else:
        merged_coeffs = [None, None, None]
        merged_r2 = None
    # 绘制
    plt.figure(figsize=(8, 6))
    for eid in F1_ids:
        v = series_data[eid]["v"]
        a = series_data[eid]["a"]
        mask = ~(np.isnan(v) | np.isnan(a))
        plt.scatter(v[mask], a[mask], s=10, alpha=0.7, color=colors[eid], label=labels[eid])
        # 绘制拟合曲线
        coeffs = fit_results[eid]["coeffs"]
        if coeffs is not None:
            v_sort = np.sort(v[mask])
            a_fit = np.polyval(coeffs, v_sort)
            plt.plot(v_sort, a_fit, color=colors[eid], linestyle='--', linewidth=1.5, alpha=0.8,
                     label=f"{eid} 二次拟合 R²={fit_results[eid]['r2']:.4f}")
    # 合并拟合曲线
    if merged_coeffs[0] is not None and merged_r2 is not None:
        v_plot = np.linspace(np.min(all_v), np.max(all_v), 200)
        a_plot = np.polyval(merged_coeffs, v_plot)
        plt.plot(v_plot, a_plot, color='black', linestyle='-', linewidth=2, alpha=0.6,
                 label=f"合并拟合 R²={merged_r2:.4f}")
    plt.xlabel("v_sg (m/s)")
    plt.ylabel("a_sg (m/s²)")
    plt.title("相同外力 F=1 不同初速度实验的 a vs v 对比")
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)
    fig_path = f"{output_dir}/F1_av_comparison_with_fits.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    # 构造 observation
    obs_lines = []
    obs_lines.append(f"对相同外力 F=1 的三个实验 {F1_ids} 进行了 a_sg vs v_sg 的散点图和二次拟合比较。")
    for eid in F1_ids:
        res = fit_results[eid]
        if res["coeffs"]:
            obs_lines.append(
                f"  {eid}: 二次拟合系数 [c2={res['coeffs'][0]:.6f}, c1={res['coeffs'][1]:.6f}, c0={res['coeffs'][2]:.6f}], R²={res['r2']:.4f}, 数据点 {res['n']}"
            )
        else:
            obs_lines.append(f"  {eid}: 有效数据不足，无法拟合")
    obs_lines.append(f" 合并拟合(三个实验所有数据): 系数 [{merged_coeffs[0]:.6f}, {merged_coeffs[1]:.6f}, {merged_coeffs[2]:.6f}], R²={merged_r2:.4f}" if merged_r2 else "  合并拟合: 数据不足")
    if merged_r2 and merged_r2 > 0.9:
        obs_lines.append("合并 R² 较高 (>0.9)，散点大致落在同一二次曲线上，阻尼可能仅与速度有关。")
    else:
        obs_lines.append(f"合并 R²={merged_r2}，散点未明显重合，可能存在初速度或时间依赖性。")
    observation = "\n".join(obs_lines)
    # 构造 metrics
    metrics = {}
    for eid in F1_ids:
        res = fit_results[eid]
        if res["coeffs"]:
            metrics[f"{eid}_c2"] = res["coeffs"][0]
            metrics[f"{eid}_c1"] = res["coeffs"][1]
            metrics[f"{eid}_c0"] = res["coeffs"][2]
            metrics[f"{eid}_r2"] = res["r2"]
            metrics[f"{eid}_n"] = res["n"]
        else:
            metrics[f"{eid}_r2"] = None
    metrics["merged_c2"] = merged_coeffs[0] if merged_coeffs[0] is not None else None
    metrics["merged_c1"] = merged_coeffs[1] if merged_coeffs[1] is not None else None
    metrics["merged_c0"] = merged_coeffs[2] if merged_coeffs[2] is not None else None
    metrics["merged_r2"] = merged_r2
    # 如果生成了新序列则返回 derived_series
    return {
        "observation": observation,
        "derived_series": derived_new,
        "figures": [fig_path],
        "metrics": metrics
    }
