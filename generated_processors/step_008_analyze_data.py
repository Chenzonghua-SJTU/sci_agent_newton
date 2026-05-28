import numpy as np
from scipy.stats import linregress
from typing import Dict, List, Any, Union

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    对所有15个实验（constant和free场）进行 a-v 线性回归，
    报告斜率、截距、R²，并跨实验对比斜率。
    """
    parameters = payload["parameters"]
    experiment_ids = parameters.get("experiment_ids", [])
    experiments = payload["experiments"]

    # 确定需要处理的实验列表
    if not experiment_ids:
        experiment_ids = list(experiments.keys())

    observations = []
    derived_series = []

    # 按F_ext分组存储回归结果，用于跨实验对比（仅constant场）
    constant_reg = {}  # key: F_ext, value: list of (slope, intercept, r2)
    free_reg = []      # list of (exp_id, slope, intercept, r2, F_ext)

    for eid in experiment_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        available = exp["available_series"]

        F_ext = config["F_ext"]
        field_type = config.get("force_field_type", "constant")

        # 检查a和v序列是否存在
        if "a" not in series or "v" not in series:
            observations.append({
                "summary": f"实验 {eid} 缺少 a 或 v 序列，跳过回归",
                "source_data_refs": [f"{eid}:a", f"{eid}:v"],
                "metrics": {"available": available}
            })
            continue

        a = np.array(series["a"])
        v = np.array(series["v"])
        n = len(a)

        if len(v) != n:
            observations.append({
                "summary": f"实验 {eid} a和v长度不匹配，跳过",
                "source_data_refs": [f"{eid}:a", f"{eid}:v"],
                "metrics": {"len_a": n, "len_v": len(v)}
            })
            continue

        # 检查v是否变化（方差是否接近0）
        v_std = np.std(v)
        if v_std < 1e-10:
            # free场且v无变化，跳过回归
            obs = {
                "summary": f"实验 {eid} (field_type={field_type}, F_ext={F_ext}) v为常数({v[0]:.6f})，不做回归",
                "source_data_refs": [f"{eid}:a", f"{eid}:v"],
                "metrics": {
                    "v_mean": float(np.mean(v)),
                    "v_std": float(v_std),
                    "a_mean": float(np.mean(a))
                }
            }
            observations.append(obs)
            if field_type == "free":
                free_reg.append((eid, None, None, None, F_ext))
            continue

        # 执行线性回归 a = intercept + slope * v
        if n < 2:
            obs = {
                "summary": f"实验 {eid} 数据点不足2个，无法回归",
                "source_data_refs": [f"{eid}:a", f"{eid}:v"],
                "metrics": {"n": n}
            }
            observations.append(obs)
            continue

        slope, intercept, r_value, p_value, std_err = linregress(v, a)
        r2 = r_value ** 2

        # 截距与F_ext的偏差
        intercept_deviation = intercept - F_ext

        # 构建observation
        obs = {
            "summary": (
                f"实验 {eid} (field_type={field_type}, F_ext={F_ext}) "
                f"线性回归 a = {intercept:.6f} + {slope:.6f} * v, "
                f"R² = {r2:.6f}, 截距偏差 = {intercept_deviation:.6f}, 点数 = {n}"
            ),
            "source_data_refs": [f"{eid}:a", f"{eid}:v"],
            "metrics": {
                "slope": float(slope),
                "intercept": float(intercept),
                "R2": float(r2),
                "p_value": float(p_value),
                "std_err": float(std_err),
                "intercept_deviation": float(intercept_deviation),
                "data_points": n,
                "v_std": float(v_std),
                "a_std": float(np.std(a))
            }
        }
        observations.append(obs)

        # 收集按场类型
        if field_type == "constant":
            if F_ext not in constant_reg:
                constant_reg[F_ext] = []
            constant_reg[F_ext].append((float(slope), float(intercept), float(r2)))
        else:  # free
            free_reg.append((eid, float(slope), float(intercept), float(r2), F_ext))

    # ---- 生成跨实验斜率对比 OBS ----
    # constant 场：按F_ext显示平均斜率
    for F_ext, reg_list in sorted(constant_reg.items()):
        slopes = [r[0] for r in reg_list]
        intercepts = [r[1] for r in reg_list]
        r2s = [r[2] for r in reg_list]
        n_exp = len(reg_list)
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        mean_intercept = np.mean(intercepts)
        std_intercept = np.std(intercepts)
        mean_r2 = np.mean(r2s)

        obs = {
            "summary": (
                f"跨实验斜率对比 (constant场, F_ext={F_ext}): "
                f"{n_exp}个实验, 平均斜率 = {mean_slope:.6f} ± {std_slope:.6f}, "
                f"平均截距 = {mean_intercept:.6f} ± {std_intercept:.6f}, "
                f"平均R² = {mean_r2:.6f}"
            ),
            "source_data_refs": [f"exp_{eid}:a" for eid in experiment_ids if experiments.get(eid, {}).get("config", {}).get("force_field_type") == "constant" and experiments[eid]["config"]["F_ext"] == F_ext],
            "metrics": {
                "F_ext": float(F_ext),
                "experiment_count": n_exp,
                "mean_slope": float(mean_slope),
                "std_slope": float(std_slope),
                "mean_intercept": float(mean_intercept),
                "std_intercept": float(std_intercept),
                "mean_R2": float(mean_r2),
                "individual_slopes": slopes,
                "individual_R2": r2s
            }
        }
        observations.append(obs)

    # free 场：列出所有回归结果（如果有v变化）
    for eid, slope, intercept, r2, F_ext in free_reg:
        if slope is not None:
            obs = {
                "summary": (
                    f"自由场实验 {eid} (F_ext=0): "
                    f"a-v回归 a = {intercept:.6f} + {slope:.6f} * v, R² = {r2:.6f}, "
                    f"截距偏差 = {intercept - F_ext:.6f}"
                ),
                "source_data_refs": [f"{eid}:a", f"{eid}:v"],
                "metrics": {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "R2": float(r2),
                    "intercept_deviation": float(intercept - F_ext)
                }
            }
            observations.append(obs)

    # 全局统计
    obs_count = len(observations)
    constant_experiments = [eid for eid in experiment_ids if experiments.get(eid, {}).get("config", {}).get("force_field_type") == "constant"]
    free_experiments = [eid for eid in experiment_ids if experiments.get(eid, {}).get("config", {}).get("force_field_type") == "free"]
    metrics = {
        "total_experiments": len(experiment_ids),
        "constant_experiments": len(constant_experiments),
        "free_experiments": len(free_experiments),
        "observations_count": obs_count,
        "constant_regression_groups": len(constant_reg),
        "free_regression_count": sum(1 for x in free_reg if x[1] is not None)
    }

    # 汇总观察
    summary_obs = {
        "summary": (
            f"对所有 {len(experiment_ids)} 个实验执行 a-v 线性回归。"
            f"常数场实验数: {len(constant_experiments)}, 自由场实验数: {len(free_experiments)}。"
            f"共生成 {obs_count} 条观察。"
            f"详见各实验回归参数及跨实验斜率对比。"
        ),
        "source_data_refs": [f"{eid}:a" for eid in experiment_ids] + [f"{eid}:v" for eid in experiment_ids],
        "metrics": metrics
    }
    observations.append(summary_obs)

    return {
        "observation": summary_obs["summary"],
        "derived_series": derived_series,
        "observations": observations,
        "figures": [],
        "metrics": metrics
    }
