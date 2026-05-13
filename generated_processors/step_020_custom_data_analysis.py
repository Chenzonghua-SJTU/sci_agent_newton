import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiment_ids = params.get("experiment_ids", [])
    output_dir = payload["output_dir"]
    
    # Collect all data
    all_v = []
    all_m = []
    derived_series_list = []
    exp_colors = {}
    color_cycle = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_idx = 0
    
    for eid in experiment_ids:
        if eid not in payload["experiments"]:
            raise ValueError(f"Experiment {eid} not found in payload")
        exp = payload["experiments"][eid]
        config = exp["config"]
        series = exp["series"]
        available = exp["available_series"]
        
        # Extract external force
        F_ext = config.get("F_ext", config.get("constant_force", None))
        if F_ext is None:
            # try to get from force_field_type and constant_force
            if config.get("force_field_type") == "constant":
                F_ext = config.get("constant_force", None)
        if F_ext is None:
            raise ValueError(f"Cannot determine F_ext for experiment {eid}, config keys: {list(config.keys())}")
        
        t = np.array(series["t"])
        q = np.array(series["q"])
        n = len(t)
        
        # Determine acceleration series
        accel_series = None
        if "a_sg" in available and series.get("a_sg") is not None:
            accel_series = np.array(series["a_sg"])
        elif "a_central_diff" in available and series.get("a_central_diff") is not None:
            accel_series = np.array(series["a_central_diff"])
        elif "a_smooth" in available and series.get("a_smooth") is not None:
            accel_series = np.array(series["a_smooth"])
        else:
            # Compute from q using Savitzky-Golay
            if len(q) < 11:
                raise ValueError(f"Experiment {eid} has too few points ({len(q)}) for Savgol filter")
            q_smooth = savgol_filter(q, 11, 3)
            v_est = savgol_filter(q, 11, 3, deriv=1, delta=t[1]-t[0])
            a_est = savgol_filter(q, 11, 3, deriv=2, delta=t[1]-t[0])
            accel_series = a_est
            # also provide v_sg
            v_series = v_est
            # register derived series for q_smooth? not required by analysis but we return only m
            # We'll add v_sg if not present already (for later experiments)
        # If we didn't compute v from Savgol, check v_sg or v_smooth
        if 'v_series' not in locals():
            if "v_sg" in available and series.get("v_sg") is not None:
                v_series = np.array(series["v_sg"])
            elif "v_smooth" in available and series.get("v_smooth") is not None:
                v_series = np.array(series["v_smooth"])
            elif "v_central_diff" in available and series.get("v_central_diff") is not None:
                v_series = np.array(series["v_central_diff"])
            else:
                # compute from q
                if len(q) < 11:
                    raise ValueError(f"Experiment {eid} has too few points to compute velocity")
                q_smooth = savgol_filter(q, 11, 3)
                v_series = savgol_filter(q, 11, 3, deriv=1, delta=t[1]-t[0])
                # register as derived series? not required but can be useful
                # But payload prohibits modification; we just compute locally
        
        # Check lengths
        if len(accel_series) != n or len(v_series) != n:
            raise ValueError(f"Series length mismatch in {eid}: t={n}, accel={len(accel_series)}, v={len(v_series)}")
        
        # Compute m = F_ext / a, avoid division by zero
        a_safe = np.where(np.abs(accel_series) > 1e-12, accel_series, np.nan)
        m_values = F_ext / a_safe
        
        # Append to global for joint fitting
        valid = ~np.isnan(m_values) & ~np.isnan(v_series)
        all_v.extend(v_series[valid])
        all_m.extend(m_values[valid])
        
        # Record m as derived series
        derived_series_list.append({
            "experiment_id": eid,
            "name": "m",
            "values": m_values.tolist(),
            "source_name": f"m = F_ext / a_sg (with a from available or Savgol), F_ext={F_ext}",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"Computed effective mass m = {F_ext}/a for constant-force experiment"
        })
        
        # Assign color
        exp_colors[eid] = color_cycle[color_idx % len(color_cycle)]
        color_idx += 1
    
    all_v = np.array(all_v)
    all_m = np.array(all_m)
    
    if len(all_v) < 4:
        raise ValueError("Too few valid data points for fitting")
    
    # Fit quadratic: m = c0 + c1*v + c2*v^2
    coeffs = np.polyfit(all_v, all_m, 2)  # returns highest to lowest: c2, c1, c0
    c2, c1, c0 = coeffs
    m_pred = np.polyval(coeffs, all_v)
    r2 = r2_score(all_m, m_pred)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8,6))
    for eid in experiment_ids:
        # Extract this experiment's v and m from derived series (we have them in all_v/all_m but need per exp)
        # Simpler: use the stored series from payload? But we already filtered NaN. Re-extract from derived_series_list.
        pass
    # Better: iterate experiments again but plot per experiment
    # We can re-retrieve from payload experiments (recompute locally for plot)
    # To avoid recomputation, we saved derived_series_list but not the sorted data.
    # Let's re-iterate over experiment_ids and retrieve computed m and v from payload? payload unchanged.
    # Easiest: rebuild per-experiment data inside plotting loop.
    for eid in experiment_ids:
        exp = payload["experiments"][eid]
        config = exp["config"]
        F_ext = config.get("F_ext", config.get("constant_force", None))
        if F_ext is None:
            if config.get("force_field_type") == "constant":
                F_ext = config.get("constant_force", None)
        series = exp["series"]
        available = exp["available_series"]
        t = np.array(series["t"])
        q = np.array(series["q"])
        # get acceleration
        a = None
        if "a_sg" in available:
            a = np.array(series["a_sg"])
        elif "a_central_diff" in available:
            a = np.array(series["a_central_diff"])
        elif "a_smooth" in available:
            a = np.array(series["a_smooth"])
        else:
            if len(q) >= 11:
                a = savgol_filter(q, 11, 3, deriv=2, delta=t[1]-t[0])
        if a is None:
            continue
        # get velocity
        v = None
        if "v_sg" in available:
            v = np.array(series["v_sg"])
        elif "v_smooth" in available:
            v = np.array(series["v_smooth"])
        elif "v_central_diff" in available:
            v = np.array(series["v_central_diff"])
        else:
            if len(q) >= 11:
                v = savgol_filter(q, 11, 3, deriv=1, delta=t[1]-t[0])
        if v is None:
            continue
        # compute m
        a_safe = np.where(np.abs(a) > 1e-12, a, np.nan)
        m = F_ext / a_safe
        valid = ~np.isnan(m) & ~np.isnan(v)
        ax.scatter(v[valid], m[valid], label=eid, color=exp_colors.get(eid, 'gray'), s=8, alpha=0.7)
    
    # Plot fit curve
    v_sort = np.sort(all_v)
    m_fit = np.polyval(coeffs, v_sort)
    ax.plot(v_sort, m_fit, 'k-', linewidth=2, label=f'Quadratic fit (R²={r2:.4f})')
    ax.set_xlabel('v_sg (m/s)')
    ax.set_ylabel('m = F_ext / a_sg (kg)')
    ax.set_title('Effective mass m vs velocity, with joint quadratic fit')
    ax.legend()
    fig_path = os.path.join(output_dir, 'm_vs_v_joint_fit.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    observation = (
        f"对恒力实验 {experiment_ids} 计算了派生序列 m = F_ext / a_sg。"
        f"合并所有实验的数据点进行二次拟合 m = c0 + c1*v + c2*v^2，得到系数: c0={c0:.6f}, c1={c1:.6f}, c2={c2:.6f}，R²={r2:.6f}。"
        f"散点图已保存。若R²接近1说明不同实验的m-v关系沿单一曲线坍塌；当前R²={r2:.4f}，坍塌程度一般。"
    )
    
    metrics = {
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "R_squared": r2,
        "num_points": len(all_v)
    }
    
    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": [fig_path],
        "metrics": metrics
    }
