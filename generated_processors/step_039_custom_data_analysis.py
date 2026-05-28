import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os


def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiment_ids = params.get("experiment_ids", [])
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # 验证实验存在且包含必要序列
    X_all = []
    y_all = []
    exp_data = {}
    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
        exp = experiments[eid]
        config = exp["config"]
        # 获取外力
        if "constant_force" in config:
            F_ext = config["constant_force"]
        elif "F_ext" in config:
            F_ext = config["F_ext"]
        else:
            raise ValueError(
                f"Experiment {eid} config missing constant_force or F_ext"
            )
        series = exp["series"]
        if "v_sg11" not in series or "a_sg11" not in series:
            raise ValueError(
                f"Experiment {eid} missing required series v_sg11 or a_sg11"
            )
        v = np.array(series["v_sg11"])
        a = np.array(series["a_sg11"])
        t = np.array(series.get("t", np.arange(len(v))))
        # 构建特征: F_ext, F_ext*v, F_ext*v^2
        X = np.column_stack([
            np.full_like(v, F_ext),
            F_ext * v,
            F_ext * v ** 2
        ])
        X_all.append(X)
        y_all.append(a)
        exp_data[eid] = {
            "F_ext": F_ext,
            "v": v,
            "a": a,
            "t": t,
            "X": X
        }

    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)

    # 线性回归 (无截距)
    reg = LinearRegression(fit_intercept=False).fit(X_all, y_all)
    c0, c1, c2 = reg.coef_
    r2 = reg.score(X_all, y_all)

    # 计算残差并绘制
    derived_series = []
    residuals_by_exp = {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_ids)))

    for idx, eid in enumerate(experiment_ids):
        ed = exp_data[eid]
        pred = ed["X"] @ reg.coef_
        residual = ed["a"] - pred
        residuals_by_exp[eid] = residual

        # 残差统计
        res_mean = float(np.mean(residual))
        res_std = float(np.std(residual, ddof=1))
        res_max_abs = float(np.max(np.abs(residual)))

        # 派生序列
        coeff_str = f"{c0:.6f} + {c1:.6f}*v_sg11 + {c2:.6f}*v_sg11^2"
        derived_series.append({
            "experiment_id": eid,
            "name": "residual_fit",
            "values": residual.tolist(),
            "source_name": f"a_sg11 - F_ext*({coeff_str})",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"Global linear regression residual for experiment {eid}"
        })

        # 绘图
        color = colors[idx]
        axes[0].scatter(ed["t"], residual, s=5, color=color, label=eid, alpha=0.7)
        axes[1].scatter(pred, residual, s=5, color=color, label=eid, alpha=0.7)

    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual vs Time")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Predicted a_sg11")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual vs Predicted")
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "residual_plots_global_fit.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # 组装 metrics
    metrics = {
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "R2_global": r2,
    }
    for eid in experiment_ids:
        res = residuals_by_exp[eid]
        metrics[f"{eid}_residual_mean"] = float(np.mean(res))
        metrics[f"{eid}_residual_std"] = float(np.std(res, ddof=1))
        metrics[f"{eid}_residual_max_abs"] = float(np.max(np.abs(res)))

    # 生成 observation
    stats_strs = []
    for eid in experiment_ids:
        s = f"{eid}: mean={metrics[eid+'_residual_mean']:.6f}, std={metrics[eid+'_residual_std']:.6f}, max|res|={metrics[eid+'_residual_max_abs']:.6f}"
        stats_strs.append(s)
    observation = (
        f"合并 {len(experiment_ids)} 个 constant 实验的全局线性回归："
        f"a_sg11 = F_ext * ({c0:.6f} + {c1:.6f}*v_sg11 + {c2:.6f}*v_sg11²)，"
        f"全局 R² = {r2:.6f}。\n各实验残差统计：\n" + "\n".join(stats_strs) +
        "\n已返回各实验残差序列及残差图（残差 vs 时间、残差 vs 预测值）。"
    )

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": [fig_path],
        "metrics": metrics,
    }
