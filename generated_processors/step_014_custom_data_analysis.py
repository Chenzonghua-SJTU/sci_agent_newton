import os, json, itertools, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    output_dir = payload["output_dir"]
    experiments = payload["experiments"]
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(experiments.keys())
    # 只处理恒力且具有 a_sg, v_sg 的实验
    target_ids = [eid for eid in exp_ids if eid in experiments]
    if not target_ids:
        raise ValueError(f"没有可处理的实验ID: {params.get('experiment_ids')}")

    derived_series = []
    figures = []
    metrics = {}

    # 存储每个实验的拟合结果，用于汇总
    results = {}

    for eid in target_ids:
        exp = experiments[eid]
        config = exp["config"]
        ser = exp["series"]
        # 获取必要序列
        t = np.array(ser.get("t"))
        q = np.array(ser.get("q"))
        v_sg = np.array(ser.get("v_sg"))
        a_sg = np.array(ser.get("a_sg"))
        if t is None or q is None:
            continue
        F_ext = config.get("constant_force", config.get("F_ext", None))
        if F_ext is None:
            continue  # 只处理有明确外力的实验

        # 构造阻尼力 F_damp = F_ext - a_sg
        F_damp = F_ext - a_sg

        # 存储结果
        res = {"F_ext": F_ext}

        # ---- 线性模型 F_damp = b * v ----
        X_lin = v_sg.reshape(-1,1)
        y = F_damp
        reg_lin = LinearRegression(fit_intercept=False).fit(X_lin, y)
        b = reg_lin.coef_[0]
        y_pred_lin = reg_lin.predict(X_lin)
        r2_lin = 1 - np.sum((y - y_pred_lin)**2) / np.sum((y - np.mean(y))**2)
        res["lin_b"] = b
        res["lin_R2"] = r2_lin

        # ---- 平方模型 F_damp = c * v^2 ----
        X_sq = (v_sg**2).reshape(-1,1)
        reg_sq = LinearRegression(fit_intercept=False).fit(X_sq, y)
        c = reg_sq.coef_[0]
        y_pred_sq = reg_sq.predict(X_sq)
        r2_sq = 1 - np.sum((y - y_pred_sq)**2) / np.sum((y - np.mean(y))**2)
        res["sq_c"] = c
        res["sq_R2"] = r2_sq

        # ---- 幂律模型 F_damp = d * v^p ----
        # 需要正的速度，滤除非正点
        mask = v_sg > 0
        if np.sum(mask) > 5:
            v_pos = v_sg[mask]
            y_pos = F_damp[mask]
            def power_law(v, d, p):
                return d * v**p
            try:
                popt, _ = curve_fit(power_law, v_pos, y_pos, p0=[1.0, 1.0], maxfev=10000)
                d_fit, p_fit = popt
                y_pred_pow = power_law(v_pos, d_fit, p_fit)
                ss_res = np.sum((y_pos - y_pred_pow)**2)
                ss_tot = np.sum((y_pos - np.mean(y_pos))**2)
                r2_pow = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
                res["pow_d"] = d_fit
                res["pow_p"] = p_fit
                res["pow_R2"] = r2_pow
            except Exception as e:
                res["pow_d"] = None
                res["pow_p"] = None
                res["pow_R2"] = None
        else:
            res["pow_d"] = None
            res["pow_p"] = None
            res["pow_R2"] = None

        # ---- 针对exp_06的额外加速度估计 ----
        a_alt = None
        label_alt = None
        if eid == "exp_06":
            # 方法1: 直接对全局q(t)拟合二次函数 a = 2*coeff[0]
            coeffs = np.polyfit(t, q, 2)
            a_global = 2*coeffs[0]
            # 构造基于全局恒定加速度的阻尼力
            F_damp_alt = F_ext - a_global
            # 拟合线性模型等
            X_alt = v_sg.reshape(-1,1)
            reg_alt = LinearRegression(fit_intercept=False).fit(X_alt, F_damp_alt)
            b_alt = reg_alt.coef_[0]
            y_pred_alt = reg_alt.predict(X_alt)
            r2_alt = 1 - np.sum((F_damp_alt - y_pred_alt)**2) / np.sum((F_damp_alt - np.mean(F_damp_alt))**2)
            res["alt_method"] = "global_quadratic"
            res["alt_a_const"] = a_global
            res["alt_lin_b"] = b_alt
            res["alt_lin_R2"] = r2_alt

            # 方法2: 局部SG不同参数（窗口11但可能已用），不再重复
            # 为了完整，也可从q重新用savgol计算加速度
            # 使用窗口5, polyorder 3 尝试
            try:
                a_alt2 = savgol_filter(q, window_length=5, polyorder=3, deriv=2, delta=t[1]-t[0])
                F_damp_alt2 = F_ext - a_alt2
                X_alt2 = v_sg.reshape(-1,1)
                reg_alt2 = LinearRegression(fit_intercept=False).fit(X_alt2, F_damp_alt2)
                b_alt2 = reg_alt2.coef_[0]
                y_pred_alt2 = reg_alt2.predict(X_alt2)
                r2_alt2 = 1 - np.sum((F_damp_alt2 - y_pred_alt2)**2) / np.sum((F_damp_alt2 - np.mean(F_damp_alt2))**2)
                res["alt2_method"] = "savgol_w5"
                res["alt2_lin_b"] = b_alt2
                res["alt2_lin_R2"] = r2_alt2
            except Exception:
                pass

        # ---- 生成散点图及拟合曲线 ----
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # 线性
        ax = axes[0]
        ax.scatter(v_sg, F_damp, s=10, alpha=0.6, label=f'data (exp={eid})')
        v_range = np.linspace(v_sg.min(), v_sg.max(), 100)
        ax.plot(v_range, b * v_range, 'r-', label=f'linear: b={b:.5f}, R²={r2_lin:.4f}')
        ax.set_xlabel('v_sg')
        ax.set_ylabel('F_damp = F_ext - a_sg')
        ax.set_title(f'{eid}: Linear fit (F_ext={F_ext})')
        ax.legend()
        # 平方
        ax = axes[1]
        ax.scatter(v_sg**2, F_damp, s=10, alpha=0.6)
        v2_range = np.linspace((v_sg**2).min(), (v_sg**2).max(), 100)
        ax.plot(v2_range, c * v2_range, 'g-', label=f'square: c={c:.5f}, R²={r2_sq:.4f}')
        ax.set_xlabel('v_sg^2')
        ax.set_ylabel('F_damp')
        ax.set_title(f'{eid}: Square fit')
        ax.legend()
        # 幂律
        ax = axes[2]
        ax.scatter(v_sg, F_damp, s=10, alpha=0.6)
        if res.get("pow_p") is not None:
            v_pow = np.linspace(v_sg[v_sg>0].min(), v_sg[v_sg>0].max(), 100)
            ax.plot(v_pow, res["pow_d"] * v_pow**res["pow_p"], 'm-',
                    label=f'power: d={res["pow_d"]:.5f}, p={res["pow_p"]:.5f}, R²={res["pow_R2"]:.4f}')
        else:
            ax.text(0.5, 0.5, 'Power fit failed', transform=ax.transAxes, ha='center')
        ax.set_xlabel('v_sg')
        ax.set_ylabel('F_damp')
        ax.set_title(f'{eid}: Power-law fit')
        ax.legend()
        plt.tight_layout()
        fname = f"{eid}_Fdamp_vs_v_fits.png"
        fpath = os.path.join(output_dir, fname)
        fig.savefig(fpath, dpi=100)
        plt.close(fig)
        figures.append(fpath)

        # 如果有alt分析，额外画图
        if eid == "exp_06" and res.get("alt_lin_b") is not None:
            fig, ax = plt.subplots(1,1,figsize=(6,5))
            ax.scatter(v_sg, F_damp_alt, s=10, alpha=0.6, label='F_damp (global quadratic a)')
            ax.plot(v_range, b_alt * v_range, 'r-', label=f'linear: b={b_alt:.5f}, R²={r2_alt:.4f}')
            ax.set_xlabel('v_sg')
            ax.set_ylabel('F_damp_alt')
            ax.set_title(f'{eid}: Alternative acceleration from global quadratic')
            ax.legend()
            plt.tight_layout()
            fname2 = f"{eid}_Fdamp_alt_vs_v.png"
            fpath2 = os.path.join(output_dir, fname2)
            fig.savefig(fpath2, dpi=100)
            plt.close(fig)
            figures.append(fpath2)

        results[eid] = res

    # ---- 联合分析b vs F_ext (除exp_06) ----
    b_vs_F = []
    F_list = []
    for eid in target_ids:
        if eid == "exp_06":
            continue
        if eid in results and results[eid].get("lin_b") is not None:
            b_vs_F.append(results[eid]["lin_b"])
            F_list.append(results[eid]["F_ext"])
    if len(b_vs_F) >= 3:
        F_arr = np.array(F_list).reshape(-1,1)
        b_arr = np.array(b_vs_F)
        reg_bF = LinearRegression().fit(F_arr, b_arr)
        bF_slope = reg_bF.coef_[0]
        bF_intercept = reg_bF.intercept_
        bF_pred = reg_bF.predict(F_arr)
        ss_res = np.sum((b_arr - bF_pred)**2)
        ss_tot = np.sum((b_arr - np.mean(b_arr))**2)
        bF_R2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
        metrics["b_vs_F_slope"] = float(bF_slope)
        metrics["b_vs_F_intercept"] = float(bF_intercept)
        metrics["b_vs_F_R2"] = float(bF_R2)

        # 绘图
        fig, ax = plt.subplots(1,1,figsize=(6,5))
        ax.scatter(F_list, b_vs_F, s=50, label='experiments (excl. exp_06)')
        F_plot = np.linspace(min(F_list), max(F_list), 100)
        ax.plot(F_plot, bF_slope*F_plot + bF_intercept, 'r-',
                label=f'linear fit: slope={bF_slope:.5f}, R²={bF_R2:.4f}')
        ax.set_xlabel('F_ext')
        ax.set_ylabel('b (linear model coefficient)')
        ax.set_title('b vs F_ext (linear model F_damp = b*v)')
        ax.legend()
        plt.tight_layout()
        fname = "b_vs_F_ext.png"
        fpath = os.path.join(output_dir, fname)
        fig.savefig(fpath, dpi=100)
        plt.close(fig)
        figures.append(fpath)

    # ---- 构建观察报告 ----
    obs_lines = []
    obs_lines.append(f"对实验 {target_ids} 执行阻尼力分析。")
    for eid, res in results.items():
        obs_lines.append(f"\n--- {eid} (F_ext={res['F_ext']}) ---")
        obs_lines.append(f"线性模型: b={res['lin_b']:.6f}, R²={res['lin_R2']:.6f}")
        obs_lines.append(f"平方模型: c={res['sq_c']:.6f}, R²={res['sq_R2']:.6f}")
        if res.get("pow_R2") is not None:
            obs_lines.append(f"幂律模型: d={res['pow_d']:.6f}, p={res['pow_p']:.6f}, R²={res['pow_R2']:.6f}")
        else:
            obs_lines.append("幂律模型拟合失败")
        if eid == "exp_06" and "alt_lin_b" in res:
            obs_lines.append(f"替代方法(全局二次拟合): a_const={res['alt_a_const']:.6f}, 线性b={res['alt_lin_b']:.6f}, R²={res['alt_lin_R2']:.6f}")
            if "alt2_lin_b" in res:
                obs_lines.append(f"替代方法(SG窗5): 线性b={res['alt2_lin_b']:.6f}, R²={res['alt2_lin_R2']:.6f}")
    if len(b_vs_F) >= 3:
        obs_lines.append(f"\n--- b vs F_ext 分析 (排除exp_06) ---")
        obs_lines.append(f"所用实验点: F={F_list}, b={b_vs_F}")
        obs_lines.append(f"线性拟合: slope={bF_slope:.6f}, intercept={bF_intercept:.6f}, R²={bF_R2:.6f}")
    else:
        obs_lines.append("\nb vs F_ext 分析：有效实验点不足3个，未分析。")
    observation = "\n".join(obs_lines)

    # 打包 metrics
    # 将每个实验的拟合参数也加入 metrics
    for eid, res in results.items():
        for k, v in res.items():
            if v is not None:
                metrics[f"{eid}_{k}"] = v

    return {
        "observation": observation,
        "derived_series": [],  # 未创建新序列
        "figures": figures,
        "metrics": metrics
    }
