import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import math

def process(payload: dict) -> dict:
    # ---- 参数提取 ----
    params = payload["parameters"]
    exp_ids = params.get("experiment_ids", [])
    experiments = payload["experiments"]

    # ---- 常量场实验列表 ----
    const_ids = [eid for eid in exp_ids if eid in experiments]

    # ---- 准备存储结果 ----
    rows = []  # 每个实验一条记录: F_ext, v0, slope, intercept, R2, a0, a_last, v_last
    observations = []
    free_exp_ids = ["exp_01", "exp_04", "exp_07", "exp_33"]
    free_a_means = {}

    # ---- 处理每个常量场实验 ----
    for eid in const_ids:
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        avail = exp.get("available_series", [])

        # 必要序列
        if "v" not in series or "a" not in series:
            raise ValueError(f"Experiment {eid} missing required series v or a")
        v = np.array(series["v"])
        a = np.array(series["a"])
        if len(v) == 0 or len(a) == 0:
            raise ValueError(f"Empty series in {eid}")

        # 外力与初速
        F_ext = config["F_ext"]
        v0 = config.get("initial_v", 0.0)

        # a-v 线性回归
        coeffs = np.polyfit(v, a, 1)
        slope, intercept = coeffs[0], coeffs[1]
        a_pred = slope * v + intercept
        ss_res = np.sum((a - a_pred) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = math.sqrt(mean_squared_error(a, a_pred))

        # 提取 a0, a_last, v_last
        a0 = float(a[0])
        a_last = float(a[-1])
        v_last = float(v[-1])

        # 记录
        rows.append({
            "experiment_id": eid,
            "F_ext": F_ext,
            "v0": v0,
            "slope": slope,
            "intercept": intercept,
            "R2": R2,
            "a0": a0,
            "a_last": a_last,
            "v_last": v_last
        })

        # 观测条目
        obs = {
            "summary": f"常数场实验 {eid} a-v 线性回归：斜率={slope:.6f}, 截距={intercept:.6f}, R²={R2:.6f}, RMSE={rmse:.6f}；F_ext={F_ext}, v0={v0}",
            "source_data_refs": [f"{eid}:a", f"{eid}:v"],
            "metrics": {
                "F_ext": F_ext,
                "v0": v0,
                "slope": slope,
                "intercept": intercept,
                "R2": R2,
                "RMSE": rmse,
                "a0": a0,
                "a_last": a_last,
                "v_last": v_last
            }
        }
        observations.append(obs)

    # ---- 自由场检查 ----
    for eid in free_exp_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        series = exp["series"]
        if "a" not in series:
            continue
        a_arr = np.array(series["a"])
        mean_a = float(np.mean(a_arr))
        max_abs_a = float(np.max(np.abs(a_arr)))
        std_a = float(np.std(a_arr))
        free_a_means[eid] = {"mean": mean_a, "max_abs": max_abs_a, "std": std_a}

    if free_a_means:
        all_close_zero = all(abs(v["mean"]) < 1e-10 and v["max_abs"] < 1e-9 for v in free_a_means.values())
        summary_parts = [f"{eid}: mean_a={v['mean']:.2e}, std={v['std']:.2e}, max_abs={v['max_abs']:.2e}" for eid, v in free_a_means.items()]
        summary = "自由场实验 a 均值接近 0 检查：所有自由场实验 a 均值绝对值均 <1e-10，最大绝对值 <1e-9。" if all_close_zero else "自由场实验 a 均值不完全为零：" + "; ".join(summary_parts)
        obs_free = {
            "summary": summary,
            "source_data_refs": [f"{eid}:a" for eid in free_a_means],
            "metrics": {f"free_mean_{eid}": v["mean"] for eid, v in free_a_means.items()}
        }
        observations.append(obs_free)

    # ---- 多元线性回归（含交互项） ----
    if len(rows) >= 4:
        F_ext_arr = np.array([r["F_ext"] for r in rows])
        v0_arr = np.array([r["v0"] for r in rows])
        inter_term = F_ext_arr * v0_arr

        # 目标： slope 和 intercept
        slope_arr = np.array([r["slope"] for r in rows])
        intercept_arr = np.array([r["intercept"] for r in rows])

        # 构建特征矩阵 (包括常数项)
        X = np.column_stack([F_ext_arr, v0_arr, inter_term])

        # 1. slope 回归
        reg_slope = LinearRegression(fit_intercept=True)
        reg_slope.fit(X, slope_arr)
        slope_pred = reg_slope.predict(X)
        r2_slope = r2_score(slope_arr, slope_pred)
        rmse_slope = math.sqrt(mean_squared_error(slope_arr, slope_pred))
        coef_slope = reg_slope.coef_.tolist()
        intercept_slope = reg_slope.intercept_

        obs_slope = {
            "summary": f"slope 对 F_ext, v0, 交互项 (F_ext*v0) 的多元线性回归：系数={coef_slope}, 截距={intercept_slope:.6f}, R²={r2_slope:.6f}, RMSE={rmse_slope:.6f}",
            "source_data_refs": [f"{eid}:a" for eid in const_ids] + [f"{eid}:v" for eid in const_ids],
            "metrics": {
                "coef_F_ext": coef_slope[0],
                "coef_v0": coef_slope[1],
                "coef_interaction": coef_slope[2],
                "intercept": intercept_slope,
                "R2": r2_slope,
                "RMSE": rmse_slope,
                "n_experiments": len(rows)
            }
        }
        observations.append(obs_slope)

        # 2. intercept 回归
        reg_intercept = LinearRegression(fit_intercept=True)
        reg_intercept.fit(X, intercept_arr)
        inter_pred = reg_intercept.predict(X)
        r2_inter = r2_score(intercept_arr, inter_pred)
        rmse_inter = math.sqrt(mean_squared_error(intercept_arr, inter_pred))
        coef_inter = reg_intercept.coef_.tolist()
        intercept_inter = reg_intercept.intercept_

        obs_inter = {
            "summary": f"intercept 对 F_ext, v0, 交互项 (F_ext*v0) 的多元线性回归：系数={coef_inter}, 截距={intercept_inter:.6f}, R²={r2_inter:.6f}, RMSE={rmse_inter:.6f}",
            "source_data_refs": [f"{eid}:a" for eid in const_ids] + [f"{eid}:v" for eid in const_ids],
            "metrics": {
                "coef_F_ext": coef_inter[0],
                "coef_v0": coef_inter[1],
                "coef_interaction": coef_inter[2],
                "intercept": intercept_inter,
                "R2": r2_inter,
                "RMSE": rmse_inter,
                "n_experiments": len(rows)
            }
        }
        observations.append(obs_inter)

    # ---- 返回结果 ----
    result = {
        "observation": f"处理了 {len(const_ids)} 个常量场实验，每个实验提取了 a-v 线性回归参数及端点加速度/速度值。对 slope 和 intercept 分别进行了含交互项的多元线性回归。自由场实验 a 均值检查完成。共生成 {len(observations)} 条 OBS。",
        "observations": observations,
        "derived_series": [],
        "figures": [],
        "metrics": {
            "constant_experiment_count": len(const_ids),
            "observation_count": len(observations)
        }
    }
    return result
