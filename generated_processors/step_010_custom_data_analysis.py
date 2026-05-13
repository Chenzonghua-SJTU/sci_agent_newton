import os
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    params = payload.get("parameters", {})
    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        experiment_ids = list(experiments.keys())
    
    # 映射实验ID（支持带下划线和不带下划线）
    eid_map = {}
    for eid in experiment_ids:
        if eid in experiments:
            eid_map[eid] = eid
        else:
            # 尝试加下划线
            candidate = f"exp_{eid[3:]}"
            if candidate in experiments:
                eid_map[eid] = candidate
            else:
                raise ValueError(f"Experiment {eid} not found in payload. Available keys: {list(experiments.keys())}")
    
    # 外力对应表
    F_ext_map = {
        "exp03": 1.0,
        "exp04": 2.0,
        "exp05": 0.5,
        "exp_03": 1.0,
        "exp_04": 2.0,
        "exp_05": 0.5
    }
    
    results = {}  # 存储每个实验的结果
    figures = []
    derived_series_list = []
    
    for eid_original, eid in eid_map.items():
        exp = experiments[eid]
        series = exp["series"]
        config = exp.get("config", {})
        
        t = np.array(series["t"])
        q = np.array(series["q"])
        if len(t) == 0:
            raise ValueError(f"Experiment {eid} has empty t series")
        
        # 1. Savitzky-Golay滤波，窗口长度15，polyorder 3
        window_length = 15
        polyorder = 3
        # 平滑位置
        q_sg = savgol_filter(q, window_length, polyorder, mode='interp')
        # 一阶导数（速度）
        v_sg = savgol_filter(q, window_length, polyorder, deriv=1, delta=t[1]-t[0], mode='interp')
        # 二阶导数（加速度）
        a_sg = savgol_filter(q, window_length, polyorder, deriv=2, delta=t[1]-t[0], mode='interp')
        
        # 注册派生序列
        derived_series_list.append({
            "experiment_id": eid,
            "name": "q_sg",
            "values": q_sg.tolist(),
            "source_name": "Savitzky-Golay滤波(窗口15, polyorder3)平滑位置",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "通过savgol滤波从q(t)得到的光滑位置"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": "v_sg",
            "values": v_sg.tolist(),
            "source_name": "Savitzky-Golay滤波(窗口15, polyorder3)一阶导数",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "通过savgol滤波从q(t)得到的速度估计"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": "a_sg",
            "values": a_sg.tolist(),
            "source_name": "Savitzky-Golay滤波(窗口15, polyorder3)二阶导数",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "通过savgol滤波从q(t)得到的加速度估计"
        })
        
        # 2. a vs v 线性拟合
        # 去除可能的NaN或inf
        finite_mask = np.isfinite(v_sg) & np.isfinite(a_sg)
        v_finite = v_sg[finite_mask]
        a_finite = a_sg[finite_mask]
        if len(v_finite) < 2:
            raise ValueError(f"Not enough finite points for a-v fit in {eid}")
        
        res_av = linregress(v_finite, a_finite)
        k = res_av.slope
        b_av = res_av.intercept
        r2_av = res_av.rvalue ** 2
        
        # 3. a vs t 线性拟合
        res_at = linregress(t[finite_mask], a_finite)
        a_t_slope = res_at.slope
        a_t_intercept = res_at.intercept
        r2_at = res_at.rvalue ** 2
        
        # 4. 组合量 a_sg + k * v_sg
        combo = a_sg + k * v_sg
        combo_mean = np.mean(combo[finite_mask])
        combo_std = np.std(combo[finite_mask])
        
        # 获取外力
        F_ext = None
        if eid in F_ext_map:
            F_ext = F_ext_map[eid]
        elif eid_original in F_ext_map:
            F_ext = F_ext_map[eid_original]
        # 也可以从 config 读取
        if F_ext is None:
            F_ext = config.get("F_ext", config.get("constant_force", np.nan))
        
        results[eid_original] = {
            "k": k,
            "b_av": b_av,
            "r2_av": r2_av,
            "a_t_slope": a_t_slope,
            "a_t_intercept": a_t_intercept,
            "r2_at": r2_at,
            "combo_mean": combo_mean,
            "combo_std": combo_std,
            "F_ext": F_ext
        }
        
        # 保存图：a vs v
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(v_finite, a_finite, s=10, alpha=0.7, label='data')
        v_fit = np.linspace(v_finite.min(), v_finite.max(), 100)
        a_fit = k * v_fit + b_av
        ax.plot(v_fit, a_fit, 'r-', label=f'fit: a={k:.4f}v+{b_av:.4f}')
        ax.set_xlabel('v_sg')
        ax.set_ylabel('a_sg')
        ax.set_title(f'{eid_original}: a vs v (R²={r2_av:.4f})')
        ax.legend()
        fname = f"{eid_original}_a_vs_v.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=100)
        plt.close(fig)
        figures.append(os.path.join(output_dir, fname))
        
        # 保存图：a vs t
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(t[finite_mask], a_finite, s=10, alpha=0.7, label='data')
        t_fit = np.linspace(t.min(), t.max(), 100)
        a_fit_t = a_t_slope * t_fit + a_t_intercept
        ax.plot(t_fit, a_fit_t, 'r-', label=f'fit: a={a_t_slope:.4f}t+{a_t_intercept:.4f}')
        ax.set_xlabel('t')
        ax.set_ylabel('a_sg')
        ax.set_title(f'{eid_original}: a vs t (R²={r2_at:.4f})')
        ax.legend()
        fname = f"{eid_original}_a_vs_t.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=100)
        plt.close(fig)
        figures.append(os.path.join(output_dir, fname))
    
    # 汇总分析
    # 构建表格字符串
    lines = []
    lines.append(f"{'Experiment':>12} {'F_ext':>8} {'k(av)':>12} {'b(av)':>12} {'R²(av)':>10} {'a-t slope':>12} {'R²(at)':>10} {'combo_mean':>12} {'combo_std':>12}")
    lines.append("-"*110)
    for eid_original in experiment_ids:
        r = results[eid_original]
        lines.append(f"{eid_original:>12} {r['F_ext']:>8.2f} {r['k']:>12.4f} {r['b_av']:>12.4f} {r['r2_av']:>10.4f} {r['a_t_slope']:>12.4f} {r['r2_at']:>10.4f} {r['combo_mean']:>12.4f} {r['combo_std']:>12.4f}")
    
    # 关系检查: k vs F, b vs F, a-t slope vs F
    F_vals = np.array([results[eid]['F_ext'] for eid in experiment_ids])
    k_vals = np.array([results[eid]['k'] for eid in experiment_ids])
    b_vals = np.array([results[eid]['b_av'] for eid in experiment_ids])
    at_slope_vals = np.array([results[eid]['a_t_slope'] for eid in experiment_ids])
    
    # 仅当至少3个点才做线性拟合
    def check_linearity(x, y, label):
        if len(x) >= 2:
            res = linregress(x, y)
            slope = res.slope
            intercept = res.intercept
            r2 = res.rvalue**2
            return f"{label} vs F: slope={slope:.4f}, intercept={intercept:.4f}, R²={r2:.4f}"
        else:
            return f"{label} vs F: insufficient data points"
    
    lines.append("")
    lines.append("--- 参数与外力F的关系 ---")
    lines.append(check_linearity(F_vals, k_vals, "k"))
    lines.append(check_linearity(F_vals, b_vals, "b"))
    lines.append(check_linearity(F_vals, at_slope_vals, "a-t slope"))
    
    # 也检查b与F是否线性（可能是质量倒数*F + 其他）
    observation = "已完成对exp03, exp04, exp05的Savitzky-Golay滤波（窗口15, polyorder3）估计q_sg, v_sg, a_sg。\n"
    observation += "各实验拟合参数如下：\n"
    observation += "\n".join(lines)
    
    # 构建metrics字典
    metrics = {}
    for eid_original in experiment_ids:
        r = results[eid_original]
        prefix = eid_original.replace("-", "_")
        metrics[f"{prefix}_k"] = r['k']
        metrics[f"{prefix}_b_av"] = r['b_av']
        metrics[f"{prefix}_r2_av"] = r['r2_av']
        metrics[f"{prefix}_a_t_slope"] = r['a_t_slope']
        metrics[f"{prefix}_r2_at"] = r['r2_at']
        metrics[f"{prefix}_combo_mean"] = r['combo_mean']
        metrics[f"{prefix}_combo_std"] = r['combo_std']
        metrics[f"{prefix}_F_ext"] = r['F_ext']
    
    # 添加相关性检查结果
    if len(experiment_ids) >= 2:
        res_k = linregress(F_vals, k_vals)
        metrics["k_vs_F_slope"] = res_k.slope
        metrics["k_vs_F_intercept"] = res_k.intercept
        metrics["k_vs_F_r2"] = res_k.rvalue**2
        res_b = linregress(F_vals, b_vals)
        metrics["b_vs_F_slope"] = res_b.slope
        metrics["b_vs_F_intercept"] = res_b.intercept
        metrics["b_vs_F_r2"] = res_b.rvalue**2
        res_at = linregress(F_vals, at_slope_vals)
        metrics["at_slope_vs_F_slope"] = res_at.slope
        metrics["at_slope_vs_F_intercept"] = res_at.intercept
        metrics["at_slope_vs_F_r2"] = res_at.rvalue**2
    
    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics
    }
