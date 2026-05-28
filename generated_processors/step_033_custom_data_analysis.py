import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    exp_ids = params.get("experiment_ids", [])
    if not exp_ids:
        exp_ids = list(payload.get("experiments", {}).keys())
    experiments_full = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 准备输出容器
    derived_series = []
    figures = []
    metrics = {}
    models_info = {}

    # 对每个实验进行分析
    for eid in exp_ids:
        if eid not in experiments_full:
            raise ValueError(f"Experiment {eid} not found in payload")
        exp_data = experiments_full[eid]
        config = exp_data.get("config", {})
        force_field_type = config.get("force_field_type", "")
        if force_field_type != "constant":
            raise ValueError(f"Experiment {eid} is not constant force field: {force_field_type}")
        F_ext = config.get("F_ext")
        if F_ext is None:
            raise ValueError(f"Experiment {eid} has no 'F_ext' in config")

        series = exp_data.get("series", {})
        t = np.array(series.get("t", []))
        v = np.array(series.get("v", []))
        a = np.array(series.get("a", []))

        if len(v) == 0 or len(a) == 0:
            raise ValueError(f"Experiment {eid} missing 'v' or 'a' series")

        # 过滤掉无效点
        valid = np.isfinite(v) & np.isfinite(a)
        v_clean = v[valid]
        a_clean = a[valid]
        t_clean = t[valid] if len(t) == len(v) else None

        if len(v_clean) < 3:
            raise ValueError(f"Experiment {eid} has fewer than 3 valid points")

        # 计算派生量
        v_sq = v_clean ** 2
        a_diff = a_clean - F_ext  # a - F_ext

        # 定义拟合模型
        def linear(x, c):
            return F_ext - c * x

        def square(x, d):
            return F_ext - d * x ** 2

        def rational(x, k):
            return F_ext / (1 + k * x)

        def exp_decay(x, gamma):
            return F_ext * np.exp(-gamma * x)

        # 拟合参数
        fit_results = {}
        # 模型1: a = F_ext - c*v
        try:
            popt, _ = curve_fit(linear, v_clean, a_clean, p0=[0.1])
            c_val = popt[0]
            pred = linear(v_clean, c_val)
            mse = np.mean((a_clean - pred) ** 2)
            ss_res = np.sum((a_clean - pred) ** 2)
            ss_tot = np.sum((a_clean - np.mean(a_clean)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
            fit_results['model1_linear'] = {'c': c_val, 'MSE': mse, 'R2': r2}
        except Exception as e:
            fit_results['model1_linear'] = {'error': str(e)}
            c_val, mse, r2 = None, None, None

        # 模型2: a = F_ext - d*v^2
        try:
            popt2, _ = curve_fit(square, v_clean, a_clean, p0=[0.1])
            d_val = popt2[0]
            pred2 = square(v_clean, d_val)
            mse2 = np.mean((a_clean - pred2) ** 2)
            ss_res2 = np.sum((a_clean - pred2) ** 2)
            ss_tot2 = np.sum((a_clean - np.mean(a_clean)) ** 2)
            r22 = 1 - ss_res2 / ss_tot2 if ss_tot2 != 0 else 0
            fit_results['model2_square'] = {'d': d_val, 'MSE': mse2, 'R2': r22}
        except Exception as e:
            fit_results['model2_square'] = {'error': str(e)}
            d_val, mse2, r22 = None, None, None

        # 模型3: a = F_ext / (1 + k*v)
        try:
            # 避免分母接近0导致无穷大
            def rational_safe(x, k):
                denom = 1 + k * x
                # 限制分母绝对值不小于1e-10
                denom = np.where(np.abs(denom) < 1e-10, np.sign(denom) * 1e-10, denom)
                return F_ext / denom
            popt3, _ = curve_fit(rational_safe, v_clean, a_clean, p0=[0.1],
                                 bounds=(-np.inf, np.inf))  # no bound
            k_val = popt3[0]
            pred3 = rational_safe(v_clean, k_val)
            mse3 = np.mean((a_clean - pred3) ** 2)
            ss_res3 = np.sum((a_clean - pred3) ** 2)
            ss_tot3 = np.sum((a_clean - np.mean(a_clean)) ** 2)
            r23 = 1 - ss_res3 / ss_tot3 if ss_tot3 != 0 else 0
            fit_results['model3_rational'] = {'k': k_val, 'MSE': mse3, 'R2': r23}
        except Exception as e:
            fit_results['model3_rational'] = {'error': str(e)}
            k_val, mse3, r23 = None, None, None

        # 模型4: a = F_ext * exp(-gamma*v)
        try:
            def exp_safe(x, gamma):
                return F_ext * np.exp(-gamma * x)
            popt4, _ = curve_fit(exp_safe, v_clean, a_clean, p0=[0.1])
            gamma_val = popt4[0]
            pred4 = exp_safe(v_clean, gamma_val)
            mse4 = np.mean((a_clean - pred4) ** 2)
            ss_res4 = np.sum((a_clean - pred4) ** 2)
            ss_tot4 = np.sum((a_clean - np.mean(a_clean)) ** 2)
            r24 = 1 - ss_res4 / ss_tot4 if ss_tot4 != 0 else 0
            fit_results['model4_exp'] = {'gamma': gamma_val, 'MSE': mse4, 'R2': r24}
        except Exception as e:
            fit_results['model4_exp'] = {'error': str(e)}
            gamma_val, mse4, r24 = None, None, None

        models_info[eid] = fit_results

        # 绘制单个实验的a vs v及所有拟合曲线
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(v_clean, a_clean, s=10, alpha=0.6, label='Data')
        # 排序x用于画曲线
        v_sort = np.linspace(v_clean.min(), v_clean.max(), 200)
        if fit_results.get('model1_linear', {}).get('c') is not None:
            ax.plot(v_sort, linear(v_sort, fit_results['model1_linear']['c']),
                    label='linear (c={:.4f})'.format(fit_results['model1_linear']['c']), linewidth=1.5)
        if fit_results.get('model2_square', {}).get('d') is not None:
            ax.plot(v_sort, square(v_sort, fit_results['model2_square']['d']),
                    label='square (d={:.4f})'.format(fit_results['model2_square']['d']), linewidth=1.5)
        if fit_results.get('model3_rational', {}).get('k') is not None:
            ax.plot(v_sort, rational_safe(v_sort, fit_results['model3_rational']['k']),
                    label='rational (k={:.4f})'.format(fit_results['model3_rational']['k']), linewidth=1.5)
        if fit_results.get('model4_exp', {}).get('gamma') is not None:
            ax.plot(v_sort, exp_safe(v_sort, fit_results['model4_exp']['gamma']),
                    label='exp (γ={:.4f})'.format(fit_results['model4_exp']['gamma']), linewidth=1.5)
        ax.set_xlabel('v')
        ax.set_ylabel('a')
        ax.set_title(f'{eid} (F_ext={F_ext})')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fname = f'fit_{eid}.png'
        fpath = os.path.join(output_dir, fname)
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        figures.append(fpath)

        # 记录指标
        for model_key, res in fit_results.items():
            if 'error' not in res:
                prefix = f'{eid}_{model_key}'
                for k, v in res.items():
                    metrics[f'{prefix}_{k}'] = v

        # 生成派生序列：a_ratio
        a_ratio = a_clean / F_ext if F_ext != 0 else np.zeros_like(a_clean)
        derived_series.append({
            "experiment_id": eid,
            "name": "a_over_F_ext",
            "values": a_ratio.tolist(),
            "source_name": "a / F_ext",
            "provenance": "custom_data_analysis: analysis of constant force experiments",
            "description": "归一化加速度 (a/F_ext)"
        })

    # 跨实验绘制 a/F_ext vs v 散点图
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    all_ratios = []
    all_vs = []
    for eid in exp_ids:
        exp_data = experiments_full[eid]
        config = exp_data.get("config", {})
        F_ext = config.get("F_ext")
        if F_ext is None or F_ext == 0:
            continue
        series = exp_data.get("series", {})
        v = np.array(series.get("v", []))
        a = np.array(series.get("a", []))
        valid = np.isfinite(v) & np.isfinite(a)
        v_c = v[valid]
        a_c = a[valid]
        a_ratio = a_c / F_ext
        ax2.scatter(v_c, a_ratio, s=5, alpha=0.5, label=eid)
        all_ratios.append(a_ratio)
        all_vs.append(v_c)

    ax2.set_xlabel('v')
    ax2.set_ylabel('a / F_ext')
    ax2.set_title('Normalized acceleration vs velocity (all constant experiments)')
    ax2.legend(loc='best', fontsize=6)
    fig2.tight_layout()
    fname2 = 'a_over_F_ext_vs_v.png'
    fpath2 = os.path.join(output_dir, fname2)
    fig2.savefig(fpath2, dpi=150)
    plt.close(fig2)
    figures.append(fpath2)

    # 构建观察报告
    obs_lines = []
    obs_lines.append(f"对 {len(exp_ids)} 个恒外力实验 (F_ext 分别为 {[experiments_full[e]['config']['F_ext'] for e in exp_ids]}) 进行了四种阻尼模型的拟合。")
    obs_lines.append("")
    for eid in exp_ids:
        obs_lines.append(f"--- {eid} ---")
        fm = models_info.get(eid, {})
        for mname, res in fm.items():
            if 'error' in res:
                obs_lines.append(f"  {mname}: 拟合失败 - {res['error']}")
            else:
                param_str = ", ".join(f"{pk}={pv:.4f}" for pk, pv in res.items() if pk not in ['MSE','R2'])
                obs_lines.append(f"  {mname}: {param_str}, MSE={res['MSE']:.6f}, R²={res['R2']:.4f}")
        obs_lines.append("")

    obs_lines.append(f"已生成各实验 a vs v 拟合图，以及所有实验 a/F_ext vs v 散点图。")
    obs_lines.append(f"各模型参数及拟合优度已记录到 metrics。")

    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
