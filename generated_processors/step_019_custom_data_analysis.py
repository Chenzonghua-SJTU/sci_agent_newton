import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any

def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 R² = 1 - SS_res/SS_tot"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot

def _residual_std(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算残差标准差（有偏，除以n）"""
    n = len(y_true)
    if n <= 2:
        return 0.0
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / (n - 2))  # 与polyfit一致（使用n-2自由度）

def process(payload: dict) -> dict:
    # ---- 提取参数 ----
    params = payload.get("parameters", {})
    exp_ids: List[str] = params.get("experiment_ids", [])
    if not exp_ids:
        # 如果没有指定，默认处理所有恒定外力实验（根据上下文，避免错误）
        exp_ids = ["exp_02", "exp_03", "exp_04", "exp_05"]

    experiments: Dict[str, Any] = payload.get("experiments", {})
    output_dir: str = payload.get("output_dir", ".")

    # ---- 存储结果 ----
    derived_series: List[Dict] = []
    figures: List[str] = []
    metrics: Dict[str, Any] = {}

    # ---- 遍历每个实验，计算序列和拟合 ----
    for eid in exp_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload.")
        exp = experiments[eid]
        config = exp.get("config", {})
        series = exp.get("series", {})
        available = exp.get("available_series", [])

        # 获取外力 F_ext
        if "F_ext" in config:
            F_ext = float(config["F_ext"])
        elif "constant_force" in config:
            F_ext = float(config["constant_force"])
        else:
            # 尝试从 force_field_type 推断
            if config.get("force_field_type") == "constant":
                # 从可用的序列中无法知道，但根据历史应已明确
                raise ValueError(f"Experiment {eid}: cannot determine F_ext from config. "
                                 "Expect 'F_ext' or 'constant_force' field.")
            else:
                raise ValueError(f"Experiment {eid}: force field is not constant, could not normalize.")

        if F_ext == 0.0:
            raise ValueError(f"Experiment {eid}: F_ext is zero, cannot normalize.")

        # 检查必需序列
        for sname in ["a_sg", "v_sg"]:
            if sname not in series:
                raise ValueError(f"Experiment {eid}: required series '{sname}' not available.")
        t = np.array(series.get("t", []))
        a_sg = np.array(series["a_sg"])
        v_sg = np.array(series["v_sg"])
        n = len(t)
        if len(a_sg) != n or len(v_sg) != n:
            raise ValueError(f"Experiment {eid}: length mismatch among series.")

        # 计算归一化序列
        a_over_F = a_sg / F_ext
        v_over_F = v_sg / F_ext
        v2_over_F = v_sg ** 2 / F_ext   # 注意平方除以F，与需求一致

        # ---- 线性模型 a/F = c0 + c1 * (v^2/F) ----
        coeffs_lin = np.polyfit(v2_over_F, a_over_F, 1)  # [c1, c0]
        c1, c0 = coeffs_lin
        pred_lin = c0 + c1 * v2_over_F
        resid_lin = a_over_F - pred_lin
        r2_lin = _compute_r2(a_over_F, pred_lin)
        resid_std_lin = _residual_std(a_over_F, pred_lin)

        # ---- 二次模型 a/F = c0 + c1*(v/F) + c2*(v/F)^2 ----
        coeffs_quad = np.polyfit(v_over_F, a_over_F, 2)  # [c2, c1, c0]
        c2, c1_q, c0_q = coeffs_quad
        pred_quad = c0_q + c1_q * v_over_F + c2 * v_over_F ** 2
        resid_quad = a_over_F - pred_quad
        r2_quad = _compute_r2(a_over_F, pred_quad)
        resid_std_quad = _residual_std(a_over_F, pred_quad)

        # ---- 存储指标 ----
        prefix = eid
        metrics[f"{prefix}_linear_c0"] = c0
        metrics[f"{prefix}_linear_c1"] = c1
        metrics[f"{prefix}_linear_R2"] = r2_lin
        metrics[f"{prefix}_linear_residual_std"] = resid_std_lin
        metrics[f"{prefix}_quad_c0"] = c0_q
        metrics[f"{prefix}_quad_c1"] = c1_q
        metrics[f"{prefix}_quad_c2"] = c2
        metrics[f"{prefix}_quad_R2"] = r2_quad
        metrics[f"{prefix}_quad_residual_std"] = resid_std_quad

        # ---- 派生序列（残差） ----
        derived_series.append({
            "experiment_id": eid,
            "name": "linear_residual_a_over_F",
            "values": resid_lin.tolist(),
            "source_name": f"a_sg/F_ext - (c0 + c1*(v_sg^2/F_ext)), linear fit",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"线性模型残差 (a/F - (c0 + c1*v^2/F))"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "quad_residual_a_over_F",
            "values": resid_quad.tolist(),
            "source_name": f"a_sg/F_ext - (c0 + c1*(v_sg/F_ext) + c2*(v_sg/F_ext)^2), quadratic fit",
            "provenance": "generated data processor: custom_data_analysis",
            "description": f"二次模型残差 (a/F - (c0 + c1*v/F + c2*(v/F)^2))"
        })

    # ---- 绘图 ----
    # 1. 线性模型图: a/F vs v^2/F，每个实验一个子图
    fig_lin, axes_lin = plt.subplots(2, 2, figsize=(12, 10))
    axes_lin = axes_lin.flatten()
    for idx, eid in enumerate(exp_ids):
        ax = axes_lin[idx]
        exp = experiments[eid]
        series = exp["series"]
        config = exp["config"]
        F_ext = float(config.get("F_ext", config.get("constant_force", 1)))
        a_over_F = np.array(series["a_sg"]) / F_ext
        v2_over_F = np.array(series["v_sg"]) ** 2 / F_ext
        # 从已计算的指标中获取系数
        c0 = metrics[f"{eid}_linear_c0"]
        c1 = metrics[f"{eid}_linear_c1"]
        r2 = metrics[f"{eid}_linear_R2"]
        resid_std = metrics[f"{eid}_linear_residual_std"]
        # 散点
        ax.scatter(v2_over_F, a_over_F, s=20, alpha=0.6, label="data")
        # 拟合线（排序点）
        x_sort = np.sort(v2_over_F)
        y_fit = c0 + c1 * x_sort
        ax.plot(x_sort, y_fit, 'r-', lw=2, label=f"fit: c0={c0:.4f}, c1={c1:.4f}")
        ax.set_title(f"Experiment {eid} (F_ext={F_ext})")
        ax.set_xlabel(r"$v_{sg}^2 / F_{\mathrm{ext}}$")
        ax.set_ylabel(r"$a_{sg} / F_{\mathrm{ext}}$")
        ax.legend(fontsize=8)
        ax.text(0.05, 0.95, f"R²={r2:.4f}\nσ_res={resid_std:.4f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    path_lin = os.path.join(output_dir, "a_over_F_vs_v2_over_F_linear.png")
    fig_lin.savefig(path_lin, dpi=150)
    plt.close(fig_lin)
    figures.append(path_lin)

    # 2. 二次模型图: a/F vs v/F，每个实验一个子图
    fig_quad, axes_quad = plt.subplots(2, 2, figsize=(12, 10))
    axes_quad = axes_quad.flatten()
    for idx, eid in enumerate(exp_ids):
        ax = axes_quad[idx]
        exp = experiments[eid]
        series = exp["series"]
        config = exp["config"]
        F_ext = float(config.get("F_ext", config.get("constant_force", 1)))
        a_over_F = np.array(series["a_sg"]) / F_ext
        v_over_F = np.array(series["v_sg"]) / F_ext
        c0 = metrics[f"{eid}_quad_c0"]
        c1 = metrics[f"{eid}_quad_c1"]
        c2 = metrics[f"{eid}_quad_c2"]
        r2 = metrics[f"{eid}_quad_R2"]
        resid_std = metrics[f"{eid}_quad_residual_std"]
        ax.scatter(v_over_F, a_over_F, s=20, alpha=0.6, label="data")
        x_sort = np.sort(v_over_F)
        y_fit = c0 + c1 * x_sort + c2 * x_sort ** 2
        ax.plot(x_sort, y_fit, 'g-', lw=2, label=f"fit: c0={c0:.4f}, c1={c1:.4f}, c2={c2:.4f}")
        ax.set_title(f"Experiment {eid} (F_ext={F_ext})")
        ax.set_xlabel(r"$v_{sg} / F_{\mathrm{ext}}$")
        ax.set_ylabel(r"$a_{sg} / F_{\mathrm{ext}}$")
        ax.legend(fontsize=7)
        ax.text(0.05, 0.95, f"R²={r2:.4f}\nσ_res={resid_std:.4f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    plt.tight_layout()
    path_quad = os.path.join(output_dir, "a_over_F_vs_v_over_F_quad.png")
    fig_quad.savefig(path_quad, dpi=150)
    plt.close(fig_quad)
    figures.append(path_quad)

    # 3. 残差图: 每个实验的线性残差和二次残差，合并在一张大图上
    fig_res, axes_res = plt.subplots(2, 2, figsize=(12, 10))
    axes_res = axes_res.flatten()
    for idx, eid in enumerate(exp_ids):
        ax = axes_res[idx]
        exp = experiments[eid]
        series = exp["series"]
        t = np.array(series["t"])
        # 从已注册的派生序列中获取残差（我们刚刚生成，但也可以通过metrics重新计算）
        # 为避免重复计算，我们直接从之前存储的指标？但残差序列已生成，但我们有数据
        # 直接使用已存储的派生序列，但可能还没加入payload；为了简便，重新计算一次
        config = exp["config"]
        F_ext = float(config.get("F_ext", config.get("constant_force", 1)))
        a_over_F = np.array(series["a_sg"]) / F_ext
        v_over_F = np.array(series["v_sg"]) / F_ext
        v2_over_F = np.array(series["v_sg"]) ** 2 / F_ext
        c0_lin = metrics[f"{eid}_linear_c0"]
        c1_lin = metrics[f"{eid}_linear_c1"]
        pred_lin = c0_lin + c1_lin * v2_over_F
        resid_lin = a_over_F - pred_lin
        c0_q = metrics[f"{eid}_quad_c0"]
        c1_q = metrics[f"{eid}_quad_c1"]
        c2_q = metrics[f"{eid}_quad_c2"]
        pred_quad = c0_q + c1_q * v_over_F + c2_q * v_over_F ** 2
        resid_quad = a_over_F - pred_quad
        ax.plot(t, resid_lin, 'r-', lw=1.5, label="linear residual")
        ax.plot(t, resid_quad, 'b--', lw=1.5, label="quad residual")
        ax.axhline(0, color='gray', linestyle=':')
        ax.set_title(f"Experiment {eid} residuals")
        ax.set_xlabel("t")
        ax.set_ylabel("Residual")
        ax.legend(fontsize=8)
    plt.tight_layout()
    path_res = os.path.join(output_dir, "residual_linear_and_quad.png")
    fig_res.savefig(path_res, dpi=150)
    plt.close(fig_res)
    figures.append(path_res)

    # ---- 构造 observation ----
    obs_lines = []
    obs_lines.append(f"对实验 {exp_ids} 进行了归一化加速度 a_sg/F_ext 与归一化速度平方 v_sg^2/F_ext 的线性模型拟合，及与 v_sg/F_ext 的二次模型拟合。")
    for eid in exp_ids:
        obs_lines.append(f"\n实验 {eid}:")
        obs_lines.append(f"  线性模型 a/F = c0 + c1*(v^2/F): c0={metrics[f'{eid}_linear_c0']:.4f}, c1={metrics[f'{eid}_linear_c1']:.4f}, R²={metrics[f'{eid}_linear_R2']:.4f}, 残差标准差={metrics[f'{eid}_linear_residual_std']:.4f}")
        obs_lines.append(f"  二次模型 a/F = c0 + c1*(v/F) + c2*(v/F)^2: c0={metrics[f'{eid}_quad_c0']:.4f}, c1={metrics[f'{eid}_quad_c1']:.4f}, c2={metrics[f'{eid}_quad_c2']:.4f}, R²={metrics[f'{eid}_quad_R2']:.4f}, 残差标准差={metrics[f'{eid}_quad_residual_std']:.4f}")
    obs_lines.append("\n已生成图像：线性拟合图、二次拟合图、残差组合图。")
    obs_lines.append("\n已返回每个实验的线性残差和二次残差派生序列。")
    observation = "".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
