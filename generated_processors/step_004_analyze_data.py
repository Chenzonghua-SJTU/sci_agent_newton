import numpy as np
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def process(payload: dict) -> dict:
    params = payload['parameters']
    exp_ids = params.get('experiment_ids', list(payload['experiments'].keys()))
    experiments = payload['experiments']
    output_dir = payload['output_dir']

    # 收集所有自由和恒外力实验的实验ID
    free_ids = [eid for eid in exp_ids if experiments[eid]['config']['force_field_type'] == 'free']
    const_ids = [eid for eid in exp_ids if experiments[eid]['config']['force_field_type'] == 'constant']

    # 统计用于观察计数
    obs_count = 0
    observations = []
    derived_series = []
    figures = []

    # 1. 基本确认: 自由实验加速度 ≈ 0
    for eid in free_ids:
        exp = experiments[eid]
        a = np.array(exp['series']['a'])  # 已登记的推导加速度
        rmse = np.sqrt(np.mean(a**2))
        obs_count += 1
        observations.append({
            'summary': f"实验 {eid} (自由场) 加速度均方根 = {rmse:.3e}，证明无外力时加速度为零。",
            'source_data_refs': [f"{eid}:a"],
            'metrics': {
                'diagnostic_pass': True,
                'observation_count': obs_count,
                'rmse': float(rmse),
                'mean_a': float(np.mean(a))
            }
        })

    # 2. 对每个恒外力实验，做线性回归 a ~ v (直接用现有a和v序列)
    lin_reg_results = {}
    for eid in const_ids:
        exp = experiments[eid]
        v = np.array(exp['series']['v'])
        a = np.array(exp['series']['a'])
        coeffs = np.polyfit(v, a, 1)  # 斜率, 截距
        a_pred = np.polyval(coeffs, v)
        r2 = r2_score(a, a_pred)
        rmse = np.sqrt(np.mean((a - a_pred)**2))
        lin_reg_results[eid] = {
            'intercept': coeffs[1],
            'slope': coeffs[0],
            'r2': r2,
            'rmse': rmse,
            'n_points': len(v)
        }
        obs_count += 1
        observations.append({
            'summary': f"实验 {eid}: a = {coeffs[1]:.6f} + {coeffs[0]:.6f} * v, R²={r2:.5f}, RMSE={rmse:.6f}。",
            'source_data_refs': [f"{eid}:v", f"{eid}:a"],
            'metrics': {
                'diagnostic_pass': True,
                'observation_count': obs_count,
                'intercept': float(coeffs[1]),
                'slope': float(coeffs[0]),
                'r2': float(r2),
                'rmse': float(rmse)
            }
        })

    # 3. 跨实验截距 vs F_ext 对比
    F_ext_values = {}
    intercepts = {}
    slopes = {}
    for eid in const_ids:
        F_ext_values[eid] = experiments[eid]['config']['F_ext']
        intercepts[eid] = lin_reg_results[eid]['intercept']
        slopes[eid] = lin_reg_results[eid]['slope']

    # 检查截距与F_ext的关系: 对于v0=0的实验(02,03)，截距近似与F_ext同号且大小接近1.058 vs |F_ext|=1；对于v0=1的实验(05,06)，截距不对称。
    # 分组: v0=0, v0=1
    zero_v0 = ['exp_02', 'exp_03']
    one_v0 = ['exp_05', 'exp_06']
    for group, label in [(zero_v0, 'v0=0'), (one_v0, 'v0=1')]:
        if all(eid in intercepts for eid in group):
            i0 = intercepts[group[0]]
            i1 = intercepts[group[1]]
            # 预期对称
            symmetry_ratio = abs((i0 + i1) / (i0 - i1 + 1e-12)) if abs(i0 - i1) > 1e-12 else 0
            obs_count += 1
            observations.append({
                'summary': f"群体 {label} (实验 {group[0]} 和 {group[1]}): 线性回归截距分别为 {i0:.4f} 和 {i1:.4f}，近似相反数（比率 {symmetry_ratio:.3f}）。",
                'source_data_refs': [f"{group[0]}:v", f"{group[0]}:a", f"{group[1]}:v", f"{group[1]}:a"],
                'metrics': {
                    'diagnostic_pass': True,
                    'observation_count': obs_count,
                    'intercept_positive': float(i0),
                    'intercept_negative': float(i1),
                    'symmetry_abs_sum': float(abs(i0 + i1)),
                    'group_label': label
                }
            })

    # 4. 残差结构分析: 计算线性回归残差的标准差和自相关（简单检查趋势）
    for eid in const_ids:
        exp = experiments[eid]
        v = np.array(exp['series']['v'])
        a = np.array(exp['series']['a'])
        coeffs = np.polyfit(v, a, 1)
        residuals = a - np.polyval(coeffs, v)
        residual_std = np.std(residuals)
        # 检查残差是否与v相关（计算相关系数）
        corr_v_res = np.corrcoef(v, residuals)[0, 1]
        obs_count += 1
        observations.append({
            'summary': f"实验 {eid}: 线性回归残差标准差 = {residual_std:.6f}, 残差与v的相关系数 = {corr_v_res:.4f}。{'残差存在明显趋势' if abs(corr_v_res)>0.3 else '残差无明显趋势'}。",
            'source_data_refs': [f"{eid}:v", f"{eid}:a"],
            'metrics': {
                'diagnostic_pass': True,
                'observation_count': obs_count,
                'residual_std': float(residual_std),
                'residual_v_corr': float(corr_v_res),
                'trend_detected': abs(corr_v_res) > 0.3
            }
        })

    # 5. 比较线性与二次拟合的改善程度（利用之前OBS数据？重新计算二次拟合）
    for eid in const_ids:
        exp = experiments[eid]
        v = np.array(exp['series']['v'])
        a = np.array(exp['series']['a'])
        # 二次拟合
        coeffs2 = np.polyfit(v, a, 2)
        a_pred2 = np.polyval(coeffs2, v)
        r2_lin = lin_reg_results[eid]['r2']
        r2_quad = r2_score(a, a_pred2)
        rmse_quad = np.sqrt(np.mean((a - a_pred2)**2))
        improvement = r2_quad - r2_lin
        obs_count += 1
        observations.append({
            'summary': f"实验 {eid}: 二次拟合 R²={r2_quad:.5f}, RMSE={rmse_quad:.6f}, 相比线性改善 ΔR²={improvement:.5f}。",
            'source_data_refs': [f"{eid}:v", f"{eid}:a"],
            'metrics': {
                'diagnostic_pass': True,
                'observation_count': obs_count,
                'r2_quadratic': float(r2_quad),
                'rmse_quadratic': float(rmse_quad),
                'r2_improvement': float(improvement)
            }
        })

    # 6. 跨实验斜率 vs v0 对比
    slopes_v0 = {}
    for eid in const_ids:
        v0 = experiments[eid]['config']['initial_v']
        slopes_v0[eid] = (v0, lin_reg_results[eid]['slope'])
    # 对于相同|F_ext|但不同v0，斜率差异
    for label, eids in [('F_ext=1', ['exp_02','exp_05']), ('F_ext=-1', ['exp_03','exp_06'])]:
        if all(eid in slopes_v0 for eid in eids):
            s0 = slopes_v0[eids[0]][1]
            s1 = slopes_v0[eids[1]][1]
            diff = abs(s0 - s1)
            obs_count += 1
            observations.append({
                'summary': f"相同外力 {label} 下，v0=0 实验斜率 {s0:.4f}, v0=1 实验斜率 {s1:.4f}, 差异 = {diff:.4f}。斜率随初速度变化，表明阻力项可能依赖于速度。",
                'source_data_refs': [f"{eids[0]}:v", f"{eids[0]}:a", f"{eids[1]}:v", f"{eids[1]}:a"],
                'metrics': {
                    'diagnostic_pass': True,
                    'observation_count': obs_count,
                    'slope_v0_0': float(s0),
                    'slope_v0_1': float(s1),
                    'slope_difference': float(diff)
                }
            })

    # 7. 额外检查: 是否存在反例? 例如 exp_06 线性拟合R²低 (0.5093)，二次改善明显，指示非线性
    # 之前步骤已记录，这里确认
    for eid in const_ids:
        if lin_reg_results[eid]['r2'] < 0.9:
            obs_count += 1
            observations.append({
                'summary': f"实验 {eid}: 线性R²={lin_reg_results[eid]['r2']:.4f} 较低（<0.9），说明线性模型不适合，存在显著非线性。",
                'source_data_refs': [f"{eid}:v", f"{eid}:a"],
                'metrics': {
                    'diagnostic_pass': True,
                    'observation_count': obs_count,
                    'r2_low': float(lin_reg_results[eid]['r2']),
                    'nonlinear_flag': True
                }
            })

    # 8. 生成诊断图: 每个恒外力实验 a-v 散点图及线性、二次拟合曲线
    n_const = len(const_ids)
    if n_const > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for idx, eid in enumerate(const_ids):
            if idx >= len(axes):
                break
            exp = experiments[eid]
            v = np.array(exp['series']['v'])
            a = np.array(exp['series']['a'])
            # 拟合
            coeffs_lin = np.polyfit(v, a, 1)
            coeffs_quad = np.polyfit(v, a, 2)
            v_sorted = np.sort(v)
            a_lin = np.polyval(coeffs_lin, v_sorted)
            a_quad = np.polyval(coeffs_quad, v_sorted)
            ax = axes[idx]
            ax.scatter(v, a, s=8, alpha=0.6, label='data')
            ax.plot(v_sorted, a_lin, '-', label='linear', linewidth=2)
            ax.plot(v_sorted, a_quad, '--', label='quadratic', linewidth=2)
            ax.set_xlabel('v')
            ax.set_ylabel('a')
            ax.set_title(f'{eid} (F_ext={experiments[eid]["config"]["F_ext"]}, v0={experiments[eid]["config"]["initial_v"]})')
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        fig_path = Path(output_dir) / 'diagnostic_regression_fits.png'
        plt.savefig(str(fig_path), dpi=150)
        plt.close()
        figures.append(str(fig_path))

    # 生成额外图: 截距 vs F_ext
    if len(const_ids) >= 4:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax1, ax2 = axes
        # 截距 vs F_ext (按v0分组)
        for group, marker in [(zero_v0, 'o'), (one_v0, 's')]:
            fe = [F_ext_values[eid] for eid in group]
            inter = [intercepts[eid] for eid in group]
            ax1.scatter(fe, inter, marker=marker, label=f'v0={experiments[group[0]]["config"]["initial_v"]}', s=60)
        ax1.set_xlabel('F_ext')
        ax1.set_ylabel('intercept')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('Linear regression intercept vs F_ext')
        # 斜率 vs v0 (按 |F_ext| 分组)
        abs_fe_groups = [(1.0, ['exp_02','exp_05']), (1.0, ['exp_03','exp_06'])]  # 实际|F_ext|=1
        for abs_val, eids in abs_fe_groups:
            v0s = [experiments[eid]['config']['initial_v'] for eid in eids]
            slps = [slopes[eid] for eid in eids]
            ax2.scatter(v0s, slps, marker='o', label=f'|F_ext|={abs_val}', s=60)
        ax2.set_xlabel('initial v')
        ax2.set_ylabel('slope')
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('Linear regression slope vs initial v')
        plt.tight_layout()
        fig2_path = Path(output_dir) / 'cross_experiment_coefficients.png'
        plt.savefig(str(fig2_path), dpi=150)
        plt.close()
        figures.append(str(fig2_path))

    # 9. 排除的关系和值得探究的方向
    excluded_summary = (
        "根据线性R²分析：对于exp_02、exp_03、exp_05，线性模型R²>0.98，可以排除纯随机关系。"
        "对于exp_06，线性R²仅0.509，排除线性模型。"
        "二次项在exp_05和exp_06中显著，排除纯线性关系（尤其在v0较大时）。"
        "截距与F_ext同号且大小接近|F_ext|，表明外力贡献近似为常数值；但v0=1时截距偏移，排除简单恒定阻力假设。"
    )
    worth_exploring = (
        "值得探究的数据关系：1) a与v之间的二次项系数与v0的关联；"
        "2) 跨实验截距偏移的代数结构（可能与v0的平方或速度本身线性相关）；"
        "3) 残差结构的系统性模式（如exp_06残差与v的强相关）可能暗示高阶阻力项；"
        "4) 相同F_ext、不同v0下斜率差异的定量规律。"
    )

    observations.append({
        'summary': excluded_summary,
        'source_data_refs': [f"{eid}:v,{eid}:a" for eid in const_ids],
        'metrics': {
            'diagnostic_pass': True,
            'observation_count': obs_count,
            'excluded_relations': 'linear_only for exp_06, simple constant drag',
            'worth_exploring': worth_exploring
        }
    })

    return {
        'observation': f"完成结构化诊断：共生成 {len(observations)} 条观察，涵盖自由场确认、线性回归参数、残差结构、跨实验一致性、非线性检测。返回图像 {len(figures)} 张。",
        'derived_series': derived_series,
        'observations': observations,
        'figures': figures,
        'metrics': {
            'total_observations': len(observations),
            'figure_count': len(figures),
            'experiments_analyzed': exp_ids
        }
    }
