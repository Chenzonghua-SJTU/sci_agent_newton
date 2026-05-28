import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

def process(payload: dict) -> dict:
    params = payload["parameters"]
    experiment_ids = params.get("experiment_ids", [])
    output_dir = payload["output_dir"]
    experiments = payload["experiments"]
    
    # 辅助函数：构建实验键
    def exp_key(eid):
        return f"exp_{eid:02d}"
    
    # 检查实验都存在
    for eid in experiment_ids:
        key = exp_key(eid)
        if key not in experiments:
            raise ValueError(f"Experiment {key} not found in payload")
    
    # 存储结果
    derived_series_list = []
    metrics = {}
    figures = []
    
    # 准备画图：normalized_drag随时间变化
    fig1, axs1 = plt.subplots(len(experiment_ids), 1, figsize=(8, 4*len(experiment_ids)))
    if len(experiment_ids) == 1:
        axs1 = [axs1]
    
    # drag vs sqrt(v)拟合图
    fig2, axs2 = plt.subplots(len(experiment_ids), 1, figsize=(8, 4*len(experiment_ids)))
    if len(experiment_ids) == 1:
        axs2 = [axs2]
    
    for idx, eid in enumerate(experiment_ids):
        key = exp_key(eid)
        exp = experiments[key]
        config = exp["config"]
        series = exp["series"]
        
        # 获取F_ext（优先使用F_ext字段，若不存在尝试constant_force）
        F_ext = config.get("F_ext", None)
        if F_ext is None:
            F_ext = config.get("constant_force", 0.0)
        
        # 获取必要序列
        if "drag" not in series:
            raise ValueError(f"Experiment {key} does not have drag series")
        if "v_est" not in series:
            # 尝试velocity
            if "velocity" in series:
                v_est = np.array(series["velocity"])
            else:
                raise ValueError(f"Experiment {key} does not have v_est or velocity series")
        else:
            v_est = np.array(series["v_est"])
        
        drag = np.array(series["drag"])
        t = np.array(series["t"])
        
        # 确保长度一致
        n = len(t)
        if len(drag) != n or len(v_est) != n:
            raise ValueError(f"Series length mismatch in {key}: t={n}, drag={len(drag)}, v_est={len(v_est)}")
        
        # 计算normalized_drag
        sqrt_v = np.sqrt(np.maximum(v_est, 1e-12))  # 防止负数或零，但假设v_est>=0
        denominator = F_ext * sqrt_v
        # 避免除零（分母不可能为零除非F_ext=0，但F_ext=1或2）
        if np.any(denominator == 0):
            # 对于极少情况v_est=0且F_ext>0，标记为非正常
            normalized_drag = np.divide(drag, denominator, out=np.full_like(drag, np.nan), where=denominator>0)
        else:
            normalized_drag = drag / denominator
        
        # 统计量
        nd_mean = np.nanmean(normalized_drag)
        nd_std = np.nanstd(normalized_drag)
        nd_min = np.nanmin(normalized_drag)
        nd_max = np.nanmax(normalized_drag)
        nd_range = nd_max - nd_min
        nd_std_ratio = nd_std / nd_mean if nd_mean != 0 else np.inf
        
        # 判断是否近似常数（相对标准差<0.1）
        is_constant = nd_std_ratio < 0.1
        
        # 记录metrics
        pref = f"exp{eid:02d}"
        metrics[f"{pref}_normalized_drag_mean"] = round(nd_mean, 6)
        metrics[f"{pref}_normalized_drag_std"] = round(nd_std, 6)
        metrics[f"{pref}_normalized_drag_min"] = round(nd_min, 6)
        metrics[f"{pref}_normalized_drag_max"] = round(nd_max, 6)
        metrics[f"{pref}_normalized_drag_range"] = round(nd_range, 6)
        metrics[f"{pref}_normalized_drag_std_ratio"] = round(nd_std_ratio, 6)
        metrics[f"{pref}_normalized_drag_is_constant"] = is_constant
        
        # 添加派生序列
        derived_series_list.append({
            "experiment_id": key,
            "name": "normalized_drag",
            "values": normalized_drag.tolist(),
            "source_name": f"drag / (F_ext * sqrt(v_est)), F_ext={F_ext}",
            "provenance": "custom_data_analysis: drag / (F_ext * sqrt(v_est))",
            "description": "Normalized drag by external force and sqrt(velocity)"
        })
        
        # 绘制normalized_drag vs time
        ax = axs1[idx]
        ax.plot(t, normalized_drag, 'b-', linewidth=1)
        ax.set_xlabel('Time')
        ax.set_ylabel('normalized_drag')
        ax.set_title(f'{key}: F_ext={F_ext}')
        ax.grid(True)
        
        # drag vs sqrt(v)拟合（过原点）
        x = sqrt_v
        y = drag
        # 过滤无效值
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x_ok = x[mask]
        y_ok = y[mask]
        
        if len(x_ok) < 2:
            slope = np.nan
            r2 = np.nan
        else:
            # 过原点线性回归：y = k * x
            # 最小二乘解：k = (x^T y) / (x^T x)
            k = np.dot(x_ok, y_ok) / np.dot(x_ok, x_ok)
            y_pred = k * x_ok
            ss_res = np.sum((y_ok - y_pred) ** 2)
            ss_tot = np.sum(y_ok ** 2)  # 过原点总平方和是y^2和
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            slope = k
        
        metrics[f"{pref}_drag_sqrtv_slope"] = round(slope, 6) if np.isfinite(slope) else None
        metrics[f"{pref}_drag_sqrtv_R2"] = round(r2, 6) if np.isfinite(r2) else None
        
        # 绘制drag vs sqrt(v)散点图和拟合线
        ax2 = axs2[idx]
        ax2.scatter(sqrt_v, drag, s=10, alpha=0.6, label='data')
        if np.isfinite(slope):
            x_line = np.linspace(sqrt_v.min(), sqrt_v.max(), 100)
            ax2.plot(x_line, slope * x_line, 'r-', label=f'fit: k={slope:.4f}, R²={r2:.4f}')
        ax2.set_xlabel('sqrt(v_est)')
        ax2.set_ylabel('drag')
        ax2.set_title(f'{key}: F_ext={F_ext}')
        ax2.legend()
        ax2.grid(True)
    
    # 保存图像
    fig1.tight_layout()
    path1 = os.path.join(output_dir, "normalized_drag_time.png")
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)
    figures.append(path1)
    
    fig2.tight_layout()
    path2 = os.path.join(output_dir, "drag_vs_sqrtv_fit.png")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    figures.append(path2)
    
    # 构建observation
    lines = []
    for eid in experiment_ids:
        pref = f"exp{eid:02d}"
        mean = metrics[f"{pref}_normalized_drag_mean"]
        std = metrics[f"{pref}_normalized_drag_std"]
        rng = metrics[f"{pref}_normalized_drag_range"]
        const = metrics[f"{pref}_normalized_drag_is_constant"]
        slope = metrics[f"{pref}_drag_sqrtv_slope"]
        r2 = metrics[f"{pref}_drag_sqrtv_R2"]
        lines.append(f"{pref}: normalized_drag mean={mean}, std={std}, range={rng}, constant? {const}; drag vs sqrt(v) slope={slope}, R²={r2}")
    
    observation = "分析了实验 " + ", ".join([f"exp{eid:02d}" for eid in experiment_ids]) + "。\n" + "\n".join(lines)
    observation += "\n图像已保存至 " + ", ".join(figures)
    
    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics
    }
