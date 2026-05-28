import numpy as np
from scipy.stats import linregress
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def process(payload: dict) -> dict:
    """
    对每个constant外力实验（exp_04, exp_05, exp_06, exp_09），
    计算log(drag)和log(v_est)，线性拟合得到斜率b（阻力关于速度的幂指数），
    报告每个实验的b、截距logC、R²以及所有实验b的均值和标准差。
    绘制drag vs v的散点图和拟合曲线（原始尺度和log-log尺度）。
    """
    params = payload.get("parameters", {})
    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        raise ValueError("参数中必须提供 experiment_ids 列表")

    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 检查实验是否存在
    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"实验 {eid} 不在有效实验列表中")

    results = []  # 每个元素为dict: exp_id, b, intercept_logC, R2
    fig_paths = []

    for eid in experiment_ids:
        exp = experiments[eid]
        series = exp.get("series", {})
        if "drag" not in series or "v_est" not in series:
            raise ValueError(f"实验 {eid} 缺少 drag 或 v_est 序列")

        drag_arr = np.array(series["drag"], dtype=float)
        v_arr = np.array(series["v_est"], dtype=float)
        t_arr = np.array(series["t"], dtype=float)

        # 只使用正值（避免log(0)或负数）
        valid_mask = (drag_arr > 0) & (v_arr > 0)
        drag_valid = drag_arr[valid_mask]
        v_valid = v_arr[valid_mask]

        if len(drag_valid) < 3:
            raise ValueError(f"实验 {eid} 有效数据点不足（{len(drag_valid)}），无法进行拟合")

        log_drag = np.log(drag_valid)
        log_v = np.log(v_valid)

        # 线性回归
        slope, intercept, r_value, p_value, std_err = linregress(log_v, log_drag)
        r2 = r_value ** 2
        b = slope
        logC = intercept   # log(C) 其中 drag = C * v^b

        results.append({
            "exp_id": eid,
            "b": b,
            "logC": logC,
            "R2": r2,
            "n_valid": len(drag_valid)
        })

        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax1, ax2 = axes

        # 左图：原始尺度 drag vs v
        ax1.scatter(v_valid, drag_valid, s=8, alpha=0.6, label='data')
        v_grid = np.linspace(v_valid.min(), v_valid.max(), 200)
        drag_fit = np.exp(logC) * v_grid ** b
        ax1.plot(v_grid, drag_fit, 'r-', label=f'fit: C={np.exp(logC):.4f}, b={b:.4f}')
        ax1.set_xlabel('v_est')
        ax1.set_ylabel('drag')
        ax1.set_title(f'{eid}: drag vs v')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右图：log-log尺度
        ax2.scatter(log_v, log_drag, s=8, alpha=0.6, label='data (log)')
        log_v_grid = np.linspace(log_v.min(), log_v.max(), 200)
        log_drag_fit = b * log_v_grid + logC
        ax2.plot(log_v_grid, log_drag_fit, 'r-', label=f'fit: b={b:.4f}, R²={r2:.4f}')
        ax2.set_xlabel('log(v_est)')
        ax2.set_ylabel('log(drag)')
        ax2.set_title(f'{eid}: log-drag vs log-v')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"drag_vs_v_fit_{eid}.png"
        fpath = os.path.join(output_dir, fname)
        fig.savefig(fpath, dpi=100)
        plt.close(fig)
        fig_paths.append(fpath)

    # 汇总所有b
    b_values = [r["b"] for r in results]
    b_mean = float(np.mean(b_values))
    b_std = float(np.std(b_values, ddof=1)) if len(b_values) > 1 else 0.0

    # 构建observation
    lines = [f"对实验 {', '.join(experiment_ids)} 进行了 log(drag) vs log(v_est) 线性拟合（幂律模型 drag = C * v^b）。"]
    for r in results:
        lines.append(
            f"  {r['exp_id']}: b={r['b']:.4f}, logC={r['logC']:.4f} (C={np.exp(r['logC']):.4f}), R²={r['R2']:.4f}, 有效点数={r['n_valid']}"
        )
    lines.append(f"所有实验 b 的均值 = {b_mean:.4f}, 样本标准差 = {b_std:.4f}")
    observation = "\n".join(lines)

    # 构建派生序列：返回每个实验的拟合drag值（基于原始v_est，但只对有效点有意义，无效点设为NaN）
    derived_series = []
    for eid in experiment_ids:
        exp = experiments[eid]
        series = exp["series"]
        drag_arr = np.array(series["drag"], dtype=float)
        v_arr = np.array(series["v_est"], dtype=float)
        # 找出本次拟合对应的参数（按顺序匹配）
        r = next(item for item in results if item["exp_id"] == eid)
        C = np.exp(r["logC"])
        b = r["b"]
        # 计算预测，对非正v设NaN
        drag_pred = np.full_like(drag_arr, np.nan, dtype=float)
        valid_mask = (v_arr > 0) & (drag_arr > 0)  # 实际只用正v
        drag_pred[valid_mask] = C * v_arr[valid_mask] ** b
        derived_series.append({
            "experiment_id": eid,
            "name": "drag_fit",
            "values": drag_pred.tolist(),
            "source_name": f"drag_fit = C * v_est^b, C={C:.4f}, b={b:.4f}",
            "provenance": "generated data processor: step_custom_data_analysis",
            "description": f"基于幂律拟合 drag = C * v^b 得到的预测值"
        })

    metrics = {}
    for r in results:
        metrics[f"{r['exp_id']}_b"] = r["b"]
        metrics[f"{r['exp_id']}_logC"] = r["logC"]
        metrics[f"{r['exp_id']}_R2"] = r["R2"]
    metrics["b_mean"] = b_mean
    metrics["b_std"] = b_std

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": fig_paths,
        "metrics": metrics
    }
