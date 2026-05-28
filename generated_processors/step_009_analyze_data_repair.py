import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path

def process(payload: dict) -> dict:
    # ---------- 参数解析 ----------
    params = payload["parameters"]
    analysis_mode = params.get("analysis_mode", "maintain_ledger")
    experiment_ids = params.get("experiment_ids", list(payload["experiments"].keys()))
    analysis_goal = params.get("analysis_goal", "")

    # 提取实验数据
    experiments = {eid: payload["experiments"][eid] for eid in experiment_ids}

    # 分离恒外力实验和自由实验
    constant_experiments = {}
    free_experiments = {}
    for eid, exp in experiments.items():
        F_ext = exp["config"]["F_ext"]
        if abs(F_ext) > 1e-12:
            constant_experiments[eid] = exp
        else:
            free_experiments[eid] = exp

    # 收集恒外力实验数据
    X_data = []  # 特征：F_ext, v (或 v^2)
    y_data = []
    exp_labels = []
    for eid, exp in constant_experiments.items():
        series = exp["series"]
        avail = exp.get("available_series", [])
        if "a_approx" not in avail or "v_approx" not in avail:
            raise ValueError(f"Experiment {eid} missing required series a_approx or v_approx")
        a = np.array(series["a_approx"])
        v = np.array(series["v_approx"])
        t = np.array(series["t"])
        F_ext = exp["config"]["F_ext"]
        # 每个点一条记录
        for i in range(len(t)):
            X_data.append([F_ext, v[i], v[i]**2])  # 包含 v 和 v^2 以备后用
            y_data.append(a[i])
            exp_labels.append(eid)

    X_data = np.array(X_data)  # shape (N, 3): [F_ext, v, v^2]
    y_data = np.array(y_data)
    N = len(y_data)

    if N == 0:
        raise ValueError("No constant force experiments found with required series.")

    # ---------- 多元线性回归（三个模型） ----------
    models = {
        "model1": {
            "formula": "a = b0 + b1*F_ext + b2*v",
            "col_indices": [0, 1]  # [F_ext, v]
        },
        "model2": {
            "formula": "a = b0 + b1*F_ext + b2*v^2",
            "col_indices": [0, 2]  # [F_ext, v^2]
        },
        "model3": {
            "formula": "a = b0 + b1*F_ext + b2*v + b3*v^2",
            "col_indices": [0, 1, 2]  # [F_ext, v, v^2]
        }
    }

    results = {}
    for mname, minfo in models.items():
        cols = minfo["col_indices"]
        # 设计矩阵：截距列 + 选中特征
        X = np.column_stack([np.ones(N), X_data[:, cols]])
        y = y_data
        # 用最小二乘
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        beta = reg.coef_
        y_pred = reg.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))  # 修复：无squared参数

        # 置信区间（alpha=0.05）
        n, p = X.shape
        MSE = np.sum((y - y_pred)**2) / (n - p)
        cov = MSE * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        t_val = stats.t.ppf(0.975, df=n - p)
        ci_low = beta - t_val * se
        ci_high = beta + t_val * se

        # 记录
        results[mname] = {
            "formula": minfo["formula"],
            "coefficients": beta.tolist(),
            "R2": r2,
            "RMSE": rmse,
            "conf_intervals": [[ci_low[i], ci_high[i]] for i in range(p)],
            "residuals": (y - y_pred).tolist(),
            "y_pred": y_pred.tolist()  # 存储预测值用于绘图
        }

    # ---------- 自由实验统计 ----------
    free_stats = {}
    for eid, exp in free_experiments.items():
        series = exp["series"]
        avail = exp.get("available_series", [])
        if "a_approx" not in avail:
            raise ValueError(f"Free experiment {eid} missing a_approx")
        a = np.array(series["a_approx"])
        mean = float(np.mean(a))
        std = float(np.std(a, ddof=1) if len(a) > 1 else 0.0)
        is_zero = abs(mean) < 1e-10 and std < 1e-10
        free_stats[eid] = {
            "a_approx_mean": mean,
            "a_approx_std": std,
            "is_zero": is_zero
        }

    # ---------- 生成图像 ----------
    output_dir = Path(payload["output_dir"])
    figures = []

    # 1. 跨实验散点图：a vs v, 颜色按 F_ext
    plt.figure(figsize=(8, 6))
    # 为了颜色区分，按 F_ext 分组
    unique_F = sorted(set(X_data[:, 0]))
    colors = plt.cm.autumn(np.linspace(0, 1, len(unique_F)))
    for fv, col in zip(unique_F, colors):
        mask = X_data[:, 0] == fv
        plt.scatter(X_data[mask, 1], y_data[mask], c=[col], label=f"F_ext={fv:.1f}", alpha=0.7, edgecolors='k')
    plt.xlabel("v_approx")
    plt.ylabel("a_approx")
    plt.title("Scatter: a_approx vs v_approx (colored by F_ext)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    scatter_path = output_dir / "scatter_a_v.png"
    plt.savefig(str(scatter_path), dpi=150, bbox_inches='tight')
    plt.close()
    figures.append(str(scatter_path))

    # 2. 残差图：三个模型子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    model_names = list(results.keys())
    for idx, (mname, res) in enumerate(results.items()):
        ax = axes[idx]
        y_pred = np.array(res["y_pred"])
        residuals = np.array(res["residuals"])
        ax.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
        ax.set_xlabel("Predicted a")
        ax.set_ylabel("Residual")
        ax.set_title(res["formula"])
        ax.grid(True, linestyle=':', alpha=0.5)
        # 加上 RMSE 文本
        ax.text(0.05, 0.95, f"RMSE={res['RMSE']:.4f}", transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    residual_path = output_dir / "residual_plots.png"
    plt.savefig(str(residual_path), dpi=150, bbox_inches='tight')
    plt.close()
    figures.append(str(residual_path))

    # ---------- 构建 OBS ----------
    # OBS1: 多元回归结果
    metrics_reg = {}
    for mname, res in results.items():
        key_prefix = mname + "_"
        metrics_reg[key_prefix + "R2"] = res["R2"]
        metrics_reg[key_prefix + "RMSE"] = res["RMSE"]
        metrics_reg[key_prefix + "coefficients"] = json.dumps(res["coefficients"])
        metrics_reg[key_prefix + "conf_intervals"] = json.dumps(res["conf_intervals"])
    # 添加样本数
    metrics_reg["observation_count"] = N
    # 来源数据引用
    source_refs = []
    for eid in constant_experiments.keys():
        source_refs.append(f"{eid}:a_approx")
        source_refs.append(f"{eid}:v_approx")
        source_refs.append(f"{eid}:config.F_ext")

    obs_reg = {
        "summary": f"Cross-experiment multiple linear regression on constant-force experiments (N={N}). Three models fitted: {', '.join([res['formula'] for res in results.values()])}. R2 values range from {min([res['R2'] for res in results.values()]):.4f} to {max([res['R2'] for res in results.values()]):.4f}. Coefficients and confidence intervals are in metrics.",
        "source_data_refs": source_refs,
        "metrics": metrics_reg
    }

    # OBS2: 自由实验统计
    obs_free_list = []
    for eid, fstat in free_stats.items():
        obs_free = {
            "summary": f"Free experiment {eid} a_approx statistics: mean={fstat['a_approx_mean']:.4e}, std={fstat['a_approx_std']:.4e}, consistent with zero? {fstat['is_zero']}.",
            "source_data_refs": [f"{eid}:a_approx"],
            "metrics": fstat
        }
        obs_free_list.append(obs_free)

    observations = [obs_reg] + obs_free_list

    # ---------- 总体说明 Observation 字符串 ----------
    summary_lines = []
    summary_lines.append("对恒外力实验(exp_02,03,05,06)进行了a_approx与v_approx的多元线性回归，共3个模型。")
    for mname, res in results.items():
        summary_lines.append(f"模型{mname}: {res['formula']}, R²={res['R2']:.4f}, RMSE={res['RMSE']:.4f}, 系数={[f'{c:.4f}' for c in res['coefficients']]}.")
    summary_lines.append("自由实验(exp_01,04): a_approx均值与标准差均接近零。")
    summary_lines.append(f"图像已保存至输出目录：{', '.join(figures)}")

    observation = "\n".join(summary_lines)

    # ---------- 返回 ----------
    output = {
        "observation": observation,
        "derived_series": [],  # 不需要新派生序列
        "observations": observations,
        "validations": [],
        "figures": figures,
        "metrics": {
            "total_experiments_used": len(experiments),
            "constant_experiments_count": len(constant_experiments),
            "free_experiments_count": len(free_experiments),
            "observation_count": N,
            "models_tested": 3
        }
    }
    return output
