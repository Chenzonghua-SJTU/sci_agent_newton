import os
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    # 确定要处理的实验
    exp_ids = params.get("experiment_ids", [])
    # 如果没提供，则处理所有恒力实验（但参数里明确给了）
    # 这里按给定的exp_04-08处理
    if not exp_ids:
        exp_ids = [eid for eid in experiments if experiments[eid]["config"].get("force_field_type") == "constant"]
        # 进一步限定到04-08
        exp_ids = [eid for eid in exp_ids if eid in ["exp_04","exp_05","exp_06","exp_07","exp_08"]]
    # 确保指定列表存在
    exp_ids = [eid for eid in exp_ids if eid in experiments]

    # 存储每个实验的数据和拟合结果
    exp_data = {}
    for eid in exp_ids:
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        available = exp["available_series"]

        t = np.array(series["t"])
        q = np.array(series["q"])

        # 检查是否有 v_sg 和 a_sg
        has_v = "v_sg" in available and len(series["v_sg"]) == len(t)
        has_a = "a_sg" in available and len(series["a_sg"]) == len(t)
        if has_v and has_a:
            v = np.array(series["v_sg"])
            a = np.array(series["a_sg"])
        else:
            # 用 SG 滤波估计
            window = 11
            polyorder = 3
            q_sg = savgol_filter(q, window, polyorder)
            v_sg = savgol_filter(q, window, polyorder, deriv=1, delta=config["dt"])
            a_sg = savgol_filter(q, window, polyorder, deriv=2, delta=config["dt"])
            # trim edges due to SG boundary effects (可选，保留全长度)
            # 但这里直接保留全长度，边界可能噪声大，但整体可用
            v = v_sg
            a = a_sg
            # 如果需要，可以trim掉前几个点，但分析目的可以保留
        exp_data[eid] = {
            "t": t,
            "q": q,
            "v": v,
            "a": a,
            "F_ext": config.get("constant_force", config.get("F_ext", 0.0)),
            "force_field_type": config.get("force_field_type", ""),
            "v0": config.get("initial_v", 0.0)
        }

    # 准备派生序列列表
    derived_series = []
    for eid in exp_ids:
        # 如果原来的没有v_sg和a_sg，则添加
        exp = experiments[eid]
        available = exp["available_series"]
        if "v_sg" not in available or len(exp["series"].get("v_sg", [])) != len(exp["series"]["t"]):
            derived_series.append({
                "experiment_id": eid,
                "name": "v_sg",
                "values": exp_data[eid]["v"].tolist(),
                "source_name": "Savitzky-Golay (window=11, polyorder=3) first derivative of q(t)",
                "provenance": "generated data processor: custom_data_analysis",
                "description": "velocity estimated via Savitzky-Golay filter"
            })
        if "a_sg" not in available or len(exp["series"].get("a_sg", [])) != len(exp["series"]["t"]):
            derived_series.append({
                "experiment_id": eid,
                "name": "a_sg",
                "values": exp_data[eid]["a"].tolist(),
                "source_name": "Savitzky-Golay (window=11, polyorder=3) second derivative of q(t)",
                "provenance": "generated data processor: custom_data_analysis",
                "description": "acceleration estimated via Savitzky-Golay filter"
            })

    # 拟合函数定义
    def linear_model(v, A, B):
        return A - B * v

    def quad_model(v, C, D):
        return C - D * v**2

    def exp_model(v, E, F, G):
        return E - F * np.exp(G * v)

    # 存储所有metrics
    metrics = {}
    figures = []

    # 对每个实验进行拟合
    for eid in exp_ids:
        data = exp_data[eid]
        v = data["v"]
        a = data["a"]
        F_ext = data["F_ext"]
        # 移除可能的nan
        mask = np.isfinite(v) & np.isfinite(a)
        if np.sum(mask) < 5:
            continue
        v_clean = v[mask]
        a_clean = a[mask]

        # 1. 线性模型 (a = A - B*v)
        # 使用线性回归拟合 a 对 v 的线性关系，得到斜率和截距
        reg_lin = LinearRegression(fit_intercept=True)
        X_lin = v_clean.reshape(-1, 1)
        reg_lin.fit(X_lin, a_clean)
        A = reg_lin.intercept_
        B = -reg_lin.coef_[0]   # 因为模型是 a = A - B*v，所以斜率是 -B
        # 重新计算R²用原始公式
        pred_a_lin = A - B * v_clean
        R2_lin = r2_score(a_clean, pred_a_lin)

        # 2. 二次模型 (a = C - D*v^2)
        v2 = v_clean**2
        reg_quad = LinearRegression(fit_intercept=True)
        reg_quad.fit(v2.reshape(-1, 1), a_clean)
        C = reg_quad.intercept_
        D = -reg_quad.coef_[0]
        pred_a_quad = C - D * v2
        R2_quad = r2_score(a_clean, pred_a_quad)

        # 3. 指数模型 (a = E - F*exp(G*v))
        # 初始猜测：E ~ max(a), F ~ max(a)-min(a), G ~ -0.1
        # 但需要确保exp不会太大，如果v很大，可能溢出。用边界检查。
        # 先尝试稳健初始猜测
        try:
            # 限制v范围，避免exp爆炸
            v_clip = np.clip(v_clean, -10, 10)  # v_phys不会超过10左右
            # 初始猜测
            E0 = np.max(a_clean)
            F0 = np.max(a_clean) - np.min(a_clean)
            G0 = -0.1
            popt, pcov = curve_fit(
                lambda v, E, F, G: E - F * np.exp(G * v),
                v_clip, a_clean,
                p0=[E0, F0, G0],
                maxfev=5000,
                bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, 0])  # F>=0, G<=0 (阻尼)
            )
            E, F, G = popt
            pred_a_exp = E - F * np.exp(G * v_clip)
            R2_exp = r2_score(a_clean, pred_a_exp)
        except Exception as e:
            # 拟合失败
            E, F, G = np.nan, np.nan, np.nan
            R2_exp = -np.inf

        # 平衡速度计算
        # 线性: v_eq_lin = A / B  (B>0)
        v_eq_lin = A / B if abs(B) > 1e-12 else np.nan
        # 二次: v_eq_quad = sqrt(C / D)  (D>0, C>0)
        v_eq_quad = np.sqrt(C / D) if (D > 0 and C > 0) else np.nan
        # 指数: v_eq_exp = (1/G) * log(E/F)  (G<0, E/F>0)
        v_eq_exp = (1/G) * np.log(E/F) if (G < 0 and E/F > 0) else np.nan

        # 存储metrics
        prefix = f"{eid}_"
        metrics[prefix + "A"] = A
        metrics[prefix + "B"] = B
        metrics[prefix + "R2_linear"] = R2_lin
        metrics[prefix + "C"] = C
        metrics[prefix + "D"] = D
        metrics[prefix + "R2_quad"] = R2_quad
        metrics[prefix + "E"] = E
        metrics[prefix + "F"] = F
        metrics[prefix + "G"] = G
        metrics[prefix + "R2_exp"] = R2_exp if np.isfinite(R2_exp) else -999
        metrics[prefix + "v_eq_lin"] = v_eq_lin if np.isfinite(v_eq_lin) else -999
        metrics[prefix + "v_eq_quad"] = v_eq_quad if np.isfinite(v_eq_quad) else -999
        metrics[prefix + "v_eq_exp"] = v_eq_exp if np.isfinite(v_eq_exp) else -999
        metrics[prefix + "F_ext"] = F_ext

        # 保存拟合数据用于后续绘图
        data["A"] = A; data["B"] = B; data["R2_lin"] = R2_lin
        data["C"] = C; data["D"] = D; data["R2_quad"] = R2_quad
        data["E"] = E; data["F"] = F; data["G"] = G; data["R2_exp"] = R2_exp
        data["v_eq_lin"] = v_eq_lin; data["v_eq_quad"] = v_eq_quad; data["v_eq_exp"] = v_eq_exp

        # 绘制每个实验的a vs v
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(v_clean, a_clean, s=8, label="data", color="gray", alpha=0.6)
        # 排序v用于画拟合曲线
        v_sort = np.sort(v_clean)
        # 线性拟合
        ax.plot(v_sort, A - B*v_sort, 'r-', label=f"linear: A={A:.4f}, B={B:.4f}, R²={R2_lin:.4f}")
        # 二次拟合
        ax.plot(v_sort, C - D*v_sort**2, 'g--', label=f"quad: C={C:.4f}, D={D:.4f}, R²={R2_quad:.4f}")
        # 指数拟合（如果有有效参数）
        if np.isfinite(E) and np.isfinite(F) and np.isfinite(G) and R2_exp > -np.inf:
            # 限制指数范围防止溢出
            exp_curve = E - F * np.exp(G * v_sort)
            if np.all(np.isfinite(exp_curve)):
                ax.plot(v_sort, exp_curve, 'b-.', label=f"exp: E={E:.4f}, F={F:.4f}, G={G:.4f}, R²={R2_exp:.4f}")
        ax.set_xlabel("v (m/s)")
        ax.set_ylabel("a (m/s²)")
        ax.set_title(f"{eid} (F_ext={F_ext}) a vs v")
        ax.legend(fontsize=7, loc='best')
        fig.tight_layout()
        fname = os.path.join(output_dir, f"{eid}_a_vs_v_models.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        figures.append(fname)

    # 跨实验比较：相同F_ext=1的实验（exp_04, exp_06, exp_08）
    f1_exps = [eid for eid in exp_ids if abs(exp_data[eid]["F_ext"] - 1.0) < 1e-6]
    if len(f1_exps) >= 2:
        fig_comp, ax_comp = plt.subplots(figsize=(9,7))
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        for i, eid in enumerate(f1_exps):
            data = exp_data[eid]
            v = data["v"]
            a = data["a"]
            mask = np.isfinite(v) & np.isfinite(a)
            ax_comp.scatter(v[mask], a[mask], s=6, color=colors[i % len(colors)],
                            label=f"{eid} (v0={data['v0']})", alpha=0.7)
        ax_comp.set_xlabel("v (m/s)")
        ax_comp.set_ylabel("a (m/s²)")
        ax_comp.set_title("Comparison of a vs v for experiments with F_ext=1")
        ax_comp.legend()
        fig_comp.tight_layout()
        fname_comp = os.path.join(output_dir, "F1_comparison_av.png")
        fig_comp.savefig(fname_comp, dpi=150)
        plt.close(fig_comp)
        figures.append(fname_comp)

    # 平衡速度 vs F_ext
    valid_eq = []
    for eid in exp_ids:
        data = exp_data[eid]
        F_ext = data["F_ext"]
        for name, key in [("lin", "v_eq_lin"), ("quad", "v_eq_quad")]:
            val = data.get(key, np.nan)
            if np.isfinite(val):
                valid_eq.append((eid, F_ext, val, name))
    if valid_eq:
        fig_eq, ax_eq = plt.subplots(figsize=(8,5))
        # 区分线性与二次
        fvals = {}
        for eid, F_ext, val, name in valid_eq:
            lbl = f"linear eq (v=A/B)" if name == "lin" else f"quad eq (v=sqrt(C/D))"
            ax_eq.scatter(F_ext, val, label=f"{eid} {name}", s=30)
        ax_eq.set_xlabel("F_ext")
        ax_eq.set_ylabel("Equilibrium velocity (m/s)")
        ax_eq.set_title("Equilibrium velocity (a=0) vs External force")
        ax_eq.legend(fontsize=7)
        fig_eq.tight_layout()
        fname_eq = os.path.join(output_dir, "v_eq_vs_F.png")
        fig_eq.savefig(fname_eq, dpi=150)
        plt.close(fig_eq)
        figures.append(fname_eq)

    # 参数 vs F_ext 图（A, C, E 和 B, D）
    params_names = [("A", "linear intercept A"), ("C", "quad intercept C"), ("E", "exp intercept E")]
    damp_names = [("B", "linear coeff B"), ("D", "quad coeff D"), ("F", "exp coeff F")]
    fig_params, axes = plt.subplots(2, 3, figsize=(15,8))
    axes = axes.flatten()
    for idx, (pname, plabel) in enumerate(params_names + damp_names):
        ax = axes[idx]
        xs = []
        ys = []
        for eid in exp_ids:
            data = exp_data[eid]
            F_ext = data["F_ext"]
            val = data.get(pname, np.nan)
            if np.isfinite(val):
                xs.append(F_ext)
                ys.append(val)
                ax.annotate(eid[-2:], (F_ext, val), fontsize=6)
        if xs:
            ax.scatter(xs, ys, s=20)
            # 尝试线性拟合
            if len(xs) >= 2:
                reg = LinearRegression().fit(np.array(xs).reshape(-1,1), ys)
                slope = reg.coef_[0]
                intercept = reg.intercept_
                ax.plot(xs, reg.predict(np.array(xs).reshape(-1,1)), 'r--', lw=1)
                ax.set_title(f"{plabel} vs F_ext\nslope={slope:.4f}, inter={intercept:.4f}", fontsize=9)
            else:
                ax.set_title(plabel, fontsize=9)
        else:
            ax.set_title(f"{plabel} (no data)", fontsize=9)
        ax.set_xlabel("F_ext")
    fig_params.tight_layout()
    fname_params = os.path.join(output_dir, "params_vs_F.png")
    fig_params.savefig(fname_params, dpi=150)
    plt.close(fig_params)
    figures.append(fname_params)

    # 构造 observation
    summary_lines = []
    summary_lines.append(f"对 {len(exp_ids)} 个恒力实验进行了 a vs v 建模分析。")
    for eid in exp_ids:
        d = exp_data[eid]
        F_ext = d["F_ext"]
        v0 = d["v0"]
        summary_lines.append(f"  {eid} (F_ext={F_ext}, v0={v0}):")
        summary_lines.append(f"    Linear: A={d['A']:.4f}, B={d['B']:.4f}, R²={d['R2_lin']:.4f}, v_eq={d['v_eq_lin'] if np.isfinite(d['v_eq_lin']) else 'nan'}")
        summary_lines.append(f"    Quad:  C={d['C']:.4f}, D={d['D']:.4f}, R²={d['R2_quad']:.4f}, v_eq={d['v_eq_quad'] if np.isfinite(d['v_eq_quad']) else 'nan'}")
        exp_metrics = metrics.copy()
        # 清理无穷大
        clean_metrics = {}
        for k, v in metrics.items():
            if np.isfinite(v):
                clean_metrics[k] = v
            else:
                clean_metrics[k] = -999.0
    summary_lines.append("跨实验比较 (F_ext=1): 绘制了 exp_04,06,08 的 a vs v 散点图，可观察轨迹是否重合。")
    summary_lines.append("参数与外力关系图已保存。")

    observation = "\n".join(summary_lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": clean_metrics
    }
