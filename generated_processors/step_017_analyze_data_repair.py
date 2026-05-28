import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path


def normalize_experiment_id(eid: str, experiments: dict) -> str:
    """尝试将参数中的实验ID转换为payload中存在的键。"""
    if eid in experiments:
        return eid
    # 尝试添加下划线：exp03 -> exp_03
    if eid.startswith("exp") and eid[3:].isdigit():
        candidate = "exp_" + eid[3:]
        if candidate in experiments:
            return candidate
    raise ValueError(f"Experiment {eid} not found in payload.")


def process(payload: dict) -> dict:
    parameters = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = Path(payload["output_dir"])

    raw_exp_ids = parameters.get("experiment_ids")
    if not raw_exp_ids:
        raw_exp_ids = list(experiments.keys())

    # 转换所有实验ID为标准格式
    exp_ids = []
    for eid in raw_exp_ids:
        normalized = normalize_experiment_id(eid, experiments)
        exp_ids.append(normalized)

    # 验证系列可用性
    required_series = ["residue_aF", "v"]
    for eid in exp_ids:
        exp = experiments[eid]
        available = exp.get("available_series", [])
        missing = [s for s in required_series if s not in available]
        if missing:
            raise ValueError(f"Experiment {eid} missing required series: {missing}")

    results = {}
    for eid in exp_ids:
        exp = experiments[eid]
        t = np.array(exp["series"]["t"])
        v = np.array(exp["series"]["v"])
        y = np.array(exp["series"]["residue_aF"])

        if len(v) != len(y):
            raise ValueError(f"Experiment {eid}: v and residue_aF length mismatch.")

        # Design matrix: [v, v^2]
        X = np.column_stack([v, v**2])
        reg = LinearRegression(fit_intercept=True)
        reg.fit(X, y)
        y_pred = reg.predict(X)

        intercept = float(reg.intercept_)
        coef_v, coef_v2 = map(float, reg.coef_)
        r2 = float(r2_score(y, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

        config = exp["config"]
        F_ext = config["F_ext"]
        v0 = config.get("initial_v", 0.0)
        q0 = config.get("initial_q", 0.0)

        results[eid] = {
            "intercept": intercept,
            "coef_v": coef_v,
            "coef_v2": coef_v2,
            "R2": r2,
            "RMSE": rmse,
            "F_ext": F_ext,
            "v0": v0,
            "q0": q0,
        }

    # Cross-experiment statistics
    intercepts = np.array([r["intercept"] for r in results.values()])
    coef_vs = np.array([r["coef_v"] for r in results.values()])
    coef_v2s = np.array([r["coef_v2"] for r in results.values()])
    F_exts = np.array([r["F_ext"] for r in results.values()])
    v0s = np.array([r["v0"] for r in results.values()])
    q0s = np.array([r["q0"] for r in results.values()])

    def mean_std(arr):
        return float(np.mean(arr)), float(np.std(arr, ddof=1))

    intercept_mean, intercept_std = mean_std(intercepts)
    coef_v_mean, coef_v_std = mean_std(coef_vs)
    coef_v2_mean, coef_v2_std = mean_std(coef_v2s)

    def pearson_corr(x, y):
        r, p = pearsonr(x, y)
        return float(r), float(p)

    corr_intercept_F = pearson_corr(intercepts, F_exts)
    corr_intercept_v0 = pearson_corr(intercepts, v0s)
    corr_intercept_q0 = pearson_corr(intercepts, q0s)
    corr_coef_v_F = pearson_corr(coef_vs, F_exts)
    corr_coef_v_v0 = pearson_corr(coef_vs, v0s)
    corr_coef_v_q0 = pearson_corr(coef_vs, q0s)
    corr_coef_v2_F = pearson_corr(coef_v2s, F_exts)
    corr_coef_v2_v0 = pearson_corr(coef_v2s, v0s)
    corr_coef_v2_q0 = pearson_corr(coef_v2s, q0s)

    # Observation 1: per-experiment regression details
    obs1_metrics = {}
    for eid in exp_ids:
        r = results[eid]
        prefix = eid + "_"
        obs1_metrics[prefix + "intercept"] = r["intercept"]
        obs1_metrics[prefix + "coef_v"] = r["coef_v"]
        obs1_metrics[prefix + "coef_v2"] = r["coef_v2"]
        obs1_metrics[prefix + "R2"] = r["R2"]
        obs1_metrics[prefix + "RMSE"] = r["RMSE"]
    obs1_metrics["experiment_count"] = len(exp_ids)
    obs1_metrics["observation_count"] = len(exp_ids)

    r2_vals = [results[eid]["R2"] for eid in exp_ids]
    intercept_vals = [results[eid]["intercept"] for eid in exp_ids]
    coef_v_vals = [results[eid]["coef_v"] for eid in exp_ids]
    coef_v2_vals = [results[eid]["coef_v2"] for eid in exp_ids]

    obs1_summary = (
        f"对{len(exp_ids)}个恒外力实验('{', '.join(exp_ids)}')进行了residue_aF ~ v + v^2线性回归(带截距)。"
        f"R²范围: {min(r2_vals):.4f} – {max(r2_vals):.4f}; "
        f"截距范围: {min(intercept_vals):.4f} – {max(intercept_vals):.4f}; "
        f"v系数范围: {min(coef_v_vals):.4f} – {max(coef_v_vals):.4f}; "
        f"v²系数范围: {min(coef_v2_vals):.4f} – {max(coef_v2_vals):.4f}. "
        "详细指标见本条目metrics。"
    )
    obs1_source_refs = []
    for eid in exp_ids:
        obs1_source_refs.extend([f"{eid}:residue_aF", f"{eid}:v"])

    observation1 = {
        "summary": obs1_summary,
        "source_data_refs": obs1_source_refs,
        "metrics": obs1_metrics,
    }

    # Observation 2: cross-experiment statistics and correlations
    obs2_metrics = {
        "intercept_mean": intercept_mean,
        "intercept_std": intercept_std,
        "coef_v_mean": coef_v_mean,
        "coef_v_std": coef_v_std,
        "coef_v2_mean": coef_v2_mean,
        "coef_v2_std": coef_v2_std,
        "corr_intercept_F_ext_r": corr_intercept_F[0],
        "corr_intercept_F_ext_p": corr_intercept_F[1],
        "corr_intercept_v0_r": corr_intercept_v0[0],
        "corr_intercept_v0_p": corr_intercept_v0[1],
        "corr_intercept_q0_r": corr_intercept_q0[0],
        "corr_intercept_q0_p": corr_intercept_q0[1],
        "corr_coef_v_F_ext_r": corr_coef_v_F[0],
        "corr_coef_v_F_ext_p": corr_coef_v_F[1],
        "corr_coef_v_v0_r": corr_coef_v_v0[0],
        "corr_coef_v_v0_p": corr_coef_v_v0[1],
        "corr_coef_v_q0_r": corr_coef_v_q0[0],
        "corr_coef_v_q0_p": corr_coef_v_q0[1],
        "corr_coef_v2_F_ext_r": corr_coef_v2_F[0],
        "corr_coef_v2_F_ext_p": corr_coef_v2_F[1],
        "corr_coef_v2_v0_r": corr_coef_v2_v0[0],
        "corr_coef_v2_v0_p": corr_coef_v2_v0[1],
        "corr_coef_v2_q0_r": corr_coef_v2_q0[0],
        "corr_coef_v2_q0_p": corr_coef_v2_q0[1],
        "observation_count": len(exp_ids),
    }
    obs2_summary = (
        f"跨实验系数统计: 截距均值={intercept_mean:.4f}±{intercept_std:.4f}, "
        f"v系数均值={coef_v_mean:.4f}±{coef_v_std:.4f}, "
        f"v²系数均值={coef_v2_mean:.4f}±{coef_v2_std:.4f}. "
        f"Pearson相关系数: 截距 vs F_ext r={corr_intercept_F[0]:.3f}(p={corr_intercept_F[1]:.3f}), "
        f"vs v0 r={corr_intercept_v0[0]:.3f}(p={corr_intercept_v0[1]:.3f}), "
        f"vs q0 r={corr_intercept_q0[0]:.3f}(p={corr_intercept_q0[1]:.3f}); "
        f"v系数 vs F_ext r={corr_coef_v_F[0]:.3f}(p={corr_coef_v_F[1]:.3f}), "
        f"vs v0 r={corr_coef_v_v0[0]:.3f}(p={corr_coef_v_v0[1]:.3f}), "
        f"vs q0 r={corr_coef_v_q0[0]:.3f}(p={corr_coef_v_q0[1]:.3f}); "
        f"v²系数 vs F_ext r={corr_coef_v2_F[0]:.3f}(p={corr_coef_v2_F[1]:.3f}), "
        f"vs v0 r={corr_coef_v2_v0[0]:.3f}(p={corr_coef_v2_v0[1]:.3f}), "
        f"vs q0 r={corr_coef_v2_q0[0]:.3f}(p={corr_coef_v2_q0[1]:.3f})."
    )
    obs2_source_refs = []
    for eid in exp_ids:
        obs2_source_refs.extend([f"{eid}:residue_aF", f"{eid}:v"])

    observation2 = {
        "summary": obs2_summary,
        "source_data_refs": obs2_source_refs,
        "metrics": obs2_metrics,
    }

    observations = [observation1, observation2]

    # Figure: coefficients vs parameters (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    param_names = ["F_ext", "v0", "q0"]
    coef_names = ["intercept", "coef_v", "coef_v2"]
    for i, pname in enumerate(param_names):
        for j, cname in enumerate(coef_names):
            ax = axes[i, j]
            x_vals = [results[eid][pname] for eid in exp_ids]
            y_vals = [results[eid][cname] for eid in exp_ids]
            ax.scatter(x_vals, y_vals, c="blue")
            ax.set_xlabel(pname)
            ax.set_ylabel(cname)
            ax.set_title(f"{cname} vs {pname}")
    plt.tight_layout()
    fig_path = output_dir / "coefficients_vs_parameters.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    figures = [str(fig_path)]

    return {
        "observation": f"完成{len(exp_ids)}个实验的residue_aF ~ v+v²线性回归，产生2条观察记录和1张图像。",
        "derived_series": [],
        "observations": observations,
        "validations": [],
        "figures": figures,
        "metrics": {
            "experiment_count": len(exp_ids),
            "figure_count": len(figures),
            "observation_count": len(observations),
        },
    }
