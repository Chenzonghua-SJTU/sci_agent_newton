import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # 目标实验列表
    target_ids = ["exp_04", "exp_05", "exp_06", "exp_07", "exp_08"]
    # 验证所有目标存在
    for eid in target_ids:
        if eid not in experiments:
            raise ValueError(f"缺少实验 {eid}")

    # F_ext 映射
    F_ext_map = {
        "exp_04": 1.0,
        "exp_05": 2.0,
        "exp_06": 1.0,
        "exp_07": 4.0,
        "exp_08": 1.0,
    }

    # 存储数据
    all_v = []
    all_damp = []
    all_labels = []
    all_F = []
    v_by_exp = {}
    damp_by_exp = {}
    F_by_exp = {}

    derived_series_list = []

    for eid in target_ids:
        exp = experiments[eid]
        series = exp["series"]
        t = np.array(series["t"])
        # 检查必要序列
        if "a_sg" not in series or "v_sg" not in series:
            raise ValueError(f"实验 {eid} 缺少 a_sg 或 v_sg")
        a_sg = np.array(series["a_sg"])
        v_sg = np.array(series["v_sg"])
        if len(t) != len(a_sg) or len(t) != len(v_sg):
            raise ValueError(f"实验 {eid} 序列长度不一致")
        F_ext = F_ext_map[eid]
        damp_neg = a_sg - F_ext  # 阻尼项（负值）
        # 保存派生序列
        derived_series_list.append({
            "experiment_id": eid,
            "name": f"damp_neg",
            "values": damp_neg.tolist(),
            "source_name": "a_sg - F_ext",
            "provenance": "generated data processor: custom_data_analysis",
            "description": "阻尼力（负值）：平滑加速度减去外力"
        })
        # 收集散点数据
        all_v.extend(v_sg.tolist())
        all_damp.extend(damp_neg.tolist())
        all_labels.extend([f"${eid}$ (F={F_ext})"] * len(v_sg))
        all_F.extend([F_ext] * len(v_sg))
        v_by_exp[eid] = v_sg
        damp_by_exp[eid] = damp_neg
        F_by_exp[eid] = F_ext

    all_v = np.array(all_v)
    all_damp = np.array(all_damp)

    # ------------------- 判断是否坍塌 -------------------
    # 尝试通用二次拟合
    coeffs_uni = np.polyfit(all_v, all_damp, 2)
    poly_uni = np.poly1d(coeffs_uni)
    pred_uni = poly_uni(all_v)
    r2_uni = r2_score(all_damp, pred_uni)
    # 对每个F值分组拟合二次
    unique_F = sorted(set(all_F))
    grouped_r2 = []
    grouped_params = {}
    for F_val in unique_F:
        mask = np.array(all_F) == F_val
        if np.sum(mask) < 3:
            continue
        v_sub = all_v[mask]
        d_sub = all_damp[mask]
        coeffs = np.polyfit(v_sub, d_sub, 2)
        pred = np.poly1d(coeffs)(v_sub)
        r2 = r2_score(d_sub, pred)
        grouped_r2.append(r2)
        grouped_params[F_val] = (coeffs, r2)
    # 简单判定：如果联合R² > 0.8 且 各分组R² 都 > 0.7，认为坍塌
    threshold_ensemble = 0.8
    collapse = False
    if r2_uni > threshold_ensemble:
        if len(grouped_r2) > 0 and all(r2 > 0.7 for r2 in grouped_r2):
            collapse = True
    # 但为了展示，我们将两种结果都计算并报告

    # ------------------- 绘图 -------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']
    for idx, eid in enumerate(target_ids):
        c = colors[idx % len(colors)]
        ax.scatter(v_by_exp[eid], damp_by_exp[eid], s=10, c=c, label=f"{eid} (F={F_by_exp[eid]:.0f})", alpha=0.7)
    # 如果认为坍塌，绘制通用拟合；否则不画全局曲线
    if collapse:
        v_sorted = np.sort(all_v)
        pred_sorted = poly_uni(v_sorted)
        ax.plot(v_sorted, pred_sorted, 'k--', linewidth=2, label=f'Universal fit (R²={r2_uni:.4f})')
    else:
        # 绘制各F值的分组拟合曲线
        for F_val in unique_F:
            if F_val not in grouped_params:
                continue
            coeffs, r2 = grouped_params[F_val]
            mask = np.array(all_F) == F_val
            v_sub = all_v[mask]
            v_sorted = np.sort(v_sub)
            pred_sorted = np.poly1d(coeffs)(v_sorted)
            ax.plot(v_sorted, pred_sorted, '--', linewidth=1.5, label=f'F={F_val:.0f} fit (R²={r2:.4f})')
    ax.set_xlabel("v (m/s)", fontsize=12)
    ax.set_ylabel("damp_neg = a - F_ext (m/s²)", fontsize=12)
    ax.set_title("Damping force vs velocity (all constant-force experiments)")
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = f"{output_dir}/damp_neg_vs_v_all.png"
    fig.savefig(fig_path)
    plt.close(fig)

    # ------------------- 拟合结果记录 -------------------
    metrics = {}
    # 通用拟合
    metrics["universal_r2"] = r2_uni
    metrics["universal_coeffs"] = coeffs_uni.tolist()
    # 分组拟合
    for F_val in unique_F:
        if F_val not in grouped_params:
            continue
        coeffs, r2 = grouped_params[F_val]
        metrics[f"F_{F_val:.0f}_coeffs"] = coeffs.tolist()
        metrics[f"F_{F_val:.0f}_r2"] = r2

    # 再加一些统计描述
    for eid in target_ids:
        v = v_by_exp[eid]
        d = damp_by_exp[eid]
        metrics[f"{eid}_v_mean"] = float(np.mean(v))
        metrics[f"{eid}_damp_neg_mean"] = float(np.mean(d))
        # 对每个实验也做简单线性拟合作为参考
        if len(v) >= 2:
            coeff_lin = np.polyfit(v, d, 1)
            pred_lin = np.poly1d(coeff_lin)(v)
            r2_lin = r2_score(d, pred_lin)
            metrics[f"{eid}_linear_slope"] = coeff_lin[0]
            metrics[f"{eid}_linear_intercept"] = coeff_lin[1]
            metrics[f"{eid}_linear_r2"] = r2_lin

    # ------------------- observation -------------------
    obs = f"对 5 个恒力实验 (exp_04-08) 计算 damp_neg = a_sg - F_ext，并与 v_sg 作图。\n"
    obs += f"全局二次拟合 R² = {r2_uni:.4f}。"
    if collapse:
        obs += f" 认为数据沿一条通用曲线坍塌，全局拟合系数为 {np.round(coeffs_uni, 4)}。\n"
    else:
        obs += " 数据未明显沿单一曲线坍塌，将对每个 F_ext 分别拟合。\n"
    for F_val in unique_F:
        if F_val not in grouped_params:
            continue
        coeffs, r2 = grouped_params[F_val]
        obs += f"  F={F_val:.0f}: 二次拟合系数 {np.round(coeffs,4)}，R²={r2:.4f}\n"
    obs += "已为每个实验生成派生序列 damp_neg。图像保存至 damp_neg_vs_v_all.png。\n"
    obs += "注意：exp_06 的 v 接近常数 (≈10 m/s)，其 damp_neg 集中在约 -0.99，导致该组数据点非常紧凑，几乎垂直。联合拟合受到这部分数据影响较小。"
    obs += "\n下一步可由决策 LLM 判断是否采用通用模型，或进一步分析各分组的阻尼规律。"

    return {
        "observation": obs,
        "derived_series": derived_series_list,
        "figures": [fig_path],
        "metrics": metrics
    }
