import json
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def process(payload: dict) -> dict:
    # --- 参数提取 ---
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    if action != "analyze_data":
        raise ValueError(f"Unexpected action: {action}")

    analysis_mode = params.get("analysis_mode", "")
    if analysis_mode != "maintain_ledger":
        raise ValueError(f"Expected analysis_mode 'maintain_ledger', got '{analysis_mode}'")

    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        raise ValueError("No experiment_ids provided")

    # --- 数据收集 ---
    # 过滤出 constant 场实验（但参数列表已保证）
    X0 = []  # [F_ext, v0]
    y0 = []  # a0
    X2 = []  # [F_ext, v2]
    y2 = []  # a2
    exp_data = {}  # exp_id -> dict of a0, a2, v0, v2, F_ext
    for eid in exp_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        # 验证 force_field_type
        if config.get("force_field_type") != "constant":
            # 如果出现非 constant 场，跳过（但参数列表已保证）
            continue
        F_ext = config["F_ext"]
        v_arr = series.get("v")
        a_arr = series.get("a")
        t_arr = series.get("t")
        if v_arr is None or a_arr is None or t_arr is None:
            raise ValueError(f"Experiment {eid}: missing 'v', 'a' or 't' series")
        # t 应等间隔从0到2，201点，简单取首尾
        if len(t_arr) != 201 or abs(t_arr[0] - 0.0) > 1e-8 or abs(t_arr[-1] - 2.0) > 1e-8:
            raise ValueError(f"Experiment {eid}: t series length or range mismatch")
        v0 = v_arr[0]
        v2 = v_arr[-1]
        a0 = a_arr[0]
        a2 = a_arr[-1]
        exp_data[eid] = {
            "a0": a0,
            "a2": a2,
            "v0": v0,
            "v2": v2,
            "F_ext": float(F_ext)
        }
        X0.append([float(F_ext), v0])
        y0.append(a0)
        X2.append([float(F_ext), v2])
        y2.append(a2)

    if len(exp_data) == 0:
        raise ValueError("No constant-field experiments found with the given IDs")

    X0 = np.array(X0)
    y0 = np.array(y0)
    X2 = np.array(X2)
    y2 = np.array(y2)

    # --- 线性回归 a0 ~ F_ext + v0 ---
    reg0 = LinearRegression(fit_intercept=True)
    reg0.fit(X0, y0)
    y0_pred = reg0.predict(X0)
    r2_0 = r2_score(y0, y0_pred)
    rmse_0 = math.sqrt(mean_squared_error(y0, y0_pred))
    mae_0 = mean_absolute_error(y0, y0_pred)
    coef0 = reg0.coef_.tolist()
    intercept0 = reg0.intercept_
    resid0 = y0 - y0_pred

    # --- 线性回归 a2 ~ F_ext + v2 ---
    reg2 = LinearRegression(fit_intercept=True)
    reg2.fit(X2, y2)
    y2_pred = reg2.predict(X2)
    r2_2 = r2_score(y2, y2_pred)
    rmse_2 = math.sqrt(mean_squared_error(y2, y2_pred))
    mae_2 = mean_absolute_error(y2, y2_pred)
    coef2 = reg2.coef_.tolist()
    intercept2 = reg2.intercept_
    resid2 = y2 - y2_pred

    # --- 偏差检查 a0 vs F_ext ---
    deviations = []
    for idx, eid in enumerate(exp_data):
        dev = exp_data[eid]["a0"] - exp_data[eid]["F_ext"]
        deviations.append(dev)
    dev_arr = np.array(deviations)
    dev_mean = float(np.mean(dev_arr))
    dev_std = float(np.std(dev_arr, ddof=1))
    dev_max_abs = float(np.max(np.abs(dev_arr)))
    dev_rmse = math.sqrt(np.mean(dev_arr**2))

    # --- 构造 observations ---
    observations = []

    # 1. 每个实验一条 observation，包含基本数值和残差
    for idx, eid in enumerate(exp_data):
        d = exp_data[eid]
        obs = {
            "summary": f"Constant-field experiment {eid}: a0={d['a0']:.6f}, a2={d['a2']:.6f}, v0={d['v0']:.6f}, v2={d['v2']:.6f}, F_ext={d['F_ext']}, residual_a0={resid0[idx]:.8f}, residual_a2={resid2[idx]:.8f}",
            "source_data_refs": [f"{eid}:a", f"{eid}:v", f"{eid}:t"],
            "metrics": {
                "a0": d["a0"],
                "a2": d["a2"],
                "v0": d["v0"],
                "v2": d["v2"],
                "F_ext": d["F_ext"],
                "residual_a0": resid0[idx],
                "residual_a2": resid2[idx]
            }
        }
        observations.append(obs)

    # 2. a0回归结果
    obs_reg0 = {
        "summary": f"Linear regression a0 ~ F_ext + v0: R2={r2_0:.6f}, RMSE={rmse_0:.6e}, MAE={mae_0:.6e}, coefficients=[F_ext={coef0[0]:.6f}, v0={coef0[1]:.6f}], intercept={intercept0:.6f}",
        "source_data_refs": [f"{eid}:a" for eid in exp_data] + [f"{eid}:v" for eid in exp_data],
        "metrics": {
            "R2": r2_0,
            "RMSE": rmse_0,
            "MAE": mae_0,
            "coef_F_ext": coef0[0],
            "coef_v0": coef0[1],
            "intercept": intercept0,
            "n_experiments": len(exp_data)
        }
    }
    observations.append(obs_reg0)

    # 3. a2回归结果
    obs_reg2 = {
        "summary": f"Linear regression a2 ~ F_ext + v2: R2={r2_2:.6f}, RMSE={rmse_2:.6e}, MAE={mae_2:.6e}, coefficients=[F_ext={coef2[0]:.6f}, v2={coef2[1]:.6f}], intercept={intercept2:.6f}",
        "source_data_refs": [f"{eid}:a" for eid in exp_data] + [f"{eid}:v" for eid in exp_data],
        "metrics": {
            "R2": r2_2,
            "RMSE": rmse_2,
            "MAE": mae_2,
            "coef_F_ext": coef2[0],
            "coef_v2": coef2[1],
            "intercept": intercept2,
            "n_experiments": len(exp_data)
        }
    }
    observations.append(obs_reg2)

    # 4. a0 vs F_ext 偏差检查
    obs_dev = {
        "summary": f"Deviation analysis: a0 - F_ext. Mean={dev_mean:.6e}, Std={dev_std:.6e}, MaxAbs={dev_max_abs:.6e}, RMSE={dev_rmse:.6e}. Ideal equality (slope=1, intercept=0) not fulfilled.",
        "source_data_refs": [f"{eid}:a" for eid in exp_data],
        "metrics": {
            "deviation_mean": dev_mean,
            "deviation_std": dev_std,
            "deviation_max_abs": dev_max_abs,
            "deviation_rmse": dev_rmse,
            "n_experiments": len(exp_data)
        }
    }
    observations.append(obs_dev)

    # --- 构造 derived_series (残差序列不是时间序列，但作为数值事实已包含在 observations 中; 不返回 derived_series) ---
    derived_series = []

    # --- 构造顶层 observation (中文摘要) ---
    obs_summary = (
        f"对 {len(exp_data)} 个常数场实验提取了 t=0 和 t=2 处的加速度和速度，并进行了两次多元线性回归。"
        f" a0~F_ext+v0: R²={r2_0:.4f}, RMSE={rmse_0:.2e}; a2~F_ext+v2: R²={r2_2:.4f}, RMSE={rmse_2:.2e}。"
        f" a0 与 F_ext 的偏差均值为 {dev_mean:.4f}，RMSE 为 {dev_rmse:.4f}。"
        f" 共产生 {len(observations)} 条 OBS，包含每个实验的数值及回归残差。未宣布任何定律。"
    )

    # 返回
    result = {
        "observation": obs_summary,
        "derived_series": derived_series,
        "observations": observations,
        "validations": [],
        "figures": [],
        "metrics": {
            "experiments_processed": len(exp_data),
            "observation_count": len(observations),
            "r2_a0_reg": r2_0,
            "r2_a2_reg": r2_2,
            "deviation_rmse": dev_rmse
        }
    }
    return result
