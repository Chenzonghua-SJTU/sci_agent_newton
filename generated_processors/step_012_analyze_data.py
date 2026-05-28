import numpy as np
from scipy.stats import linregress

def process(payload: dict) -> dict:
    experiments = payload.get('experiments', {})
    params = payload.get('parameters', {})
    experiment_ids = params.get('experiment_ids', list(experiments.keys()))

    derived_series = []
    observations = []
    constant_results = []
    free_results = []

    # Step 1: 确保所有实验都有可用的 v 和 a 序列（计算或引用已有）
    for eid in experiment_ids:
        exp = experiments.get(eid)
        if exp is None:
            continue
        config = exp['config']
        series = exp['series']
        available = exp.get('available_series', [])

        t = np.array(series['t'])
        q = np.array(series['q'])

        # 如果已有 v 和 a，直接使用；否则计算并注册派生序列
        if 'v' in available and 'a' in available:
            v = np.array(series['v'])
            a = np.array(series['a'])
        else:
            v = np.gradient(q, t, edge_order=2)
            a = np.gradient(v, t, edge_order=2)
            derived_series.append({
                'experiment_id': eid,
                'name': 'v',
                'values': v.tolist(),
                'source_name': f'np.gradient(q, t, edge_order=2) for {eid}',
                'provenance': 'generate_data_processor: maintain_ledger_new_experiments',
                'description': 'velocity computed via np.gradient'
            })
            derived_series.append({
                'experiment_id': eid,
                'name': 'a',
                'values': a.tolist(),
                'source_name': f'np.gradient(v, t, edge_order=2) for {eid}',
                'provenance': 'generate_data_processor: maintain_ledger_new_experiments',
                'description': 'acceleration computed via np.gradient of v'
            })

        # 临时存储，以便后续回归使用
        exp['_v'] = v
        exp['_a'] = a

    # Step 2: 对所有实验执行 a-v 线性回归
    for eid in experiment_ids:
        exp = experiments.get(eid)
        if exp is None:
            continue
        config = exp['config']
        v = exp['_v']
        a = exp['_a']
        F_ext = config['F_ext']
        v0 = config['initial_v']
        field_type = config.get('force_field_type', 'constant')

        # 处理极端情况（速度或加速度标准差接近 0）
        std_v = np.std(v)
        std_a = np.std(a)
        if std_v < 1e-12 or std_a < 1e-12:
            slope = 0.0
            intercept = np.mean(a)
            r_value = 0.0
            p_value = 1.0
            r2 = 0.0
        else:
            slope, intercept, r_value, p_value, _ = linregress(v, a)
            r2 = r_value ** 2

        res = {
            'experiment_id': eid,
            'F_ext': F_ext,
            'v0': v0,
            'slope': slope,
            'intercept': intercept,
            'r2': r2,
            'p_value': p_value
        }

        if field_type == 'constant':
            constant_results.append(res)
        elif field_type == 'free':
            free_results.append(res)

        # 每个实验的 OBS
        obs_summary = (
            f"Experiment {eid}: a-v linear regression | "
            f"slope={slope:.6f}, intercept={intercept:.6f}, R²={r2:.6f}, p_value={p_value:.6e}, "
            f"F_ext={F_ext}, v0={v0}"
        )
        observations.append({
            'summary': obs_summary,
            'source_data_refs': [f"{eid}:a", f"{eid}:v"],
            'metrics': {
                'slope': slope,
                'intercept': intercept,
                'r2': r2,
                'p_value': p_value,
                'F_ext': F_ext,
                'v0': v0
            }
        })

    # Step 3: 跨实验回归（常数场）
    if constant_results:
        F_ext_arr = np.array([r['F_ext'] for r in constant_results])
        v0_arr = np.array([r['v0'] for r in constant_results])
        intercepts_arr = np.array([r['intercept'] for r in constant_results])
        slopes_arr = np.array([r['slope'] for r in constant_results])

        # 辅助函数：执行回归，处理单一值的情况
        def safe_linregress(x, y):
            if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
                return np.nan, np.nan, np.nan, np.nan, np.nan
            return linregress(x, y)

        # 截距 vs F_ext
        si, ii, ri, pi, _ = safe_linregress(F_ext_arr, intercepts_arr)
        r2i = ri ** 2 if not np.isnan(ri) else np.nan
        # 斜率 vs F_ext
        sf, iff, rf, pf, _ = safe_linregress(F_ext_arr, slopes_arr)
        r2f = rf ** 2 if not np.isnan(rf) else np.nan
        # 截距 vs v0
        siv, iiv, riv, piv, _ = safe_linregress(v0_arr, intercepts_arr)
        r2iv = riv ** 2 if not np.isnan(riv) else np.nan
        # 斜率 vs v0
        sfv, ifv, rfv, pfv, _ = safe_linregress(v0_arr, slopes_arr)
        r2fv = rfv ** 2 if not np.isnan(rfv) else np.nan

        cross_summary = (
            f"Cross-experiment regressions using {len(constant_results)} constant-field experiments:\n"
            f"  Intercept vs F_ext: slope={si:.4f}, R²={r2i:.4f}, p={pi:.4e}\n"
            f"  Slope vs F_ext: slope={sf:.4f}, R²={r2f:.4f}, p={pf:.4e}\n"
            f"  Intercept vs v0: slope={siv:.4f}, R²={r2iv:.4f}, p={piv:.4e}\n"
            f"  Slope vs v0: slope={sfv:.4f}, R²={r2fv:.4f}, p={pfv:.4e}"
        )
        observations.append({
            'summary': cross_summary,
            'source_data_refs': [f"{r['experiment_id']}:a,{r['experiment_id']}:v" for r in constant_results],
            'metrics': {
                'intercept_vs_Fext_slope': si,
                'intercept_vs_Fext_R2': r2i,
                'intercept_vs_Fext_p': pi,
                'slope_vs_Fext_slope': sf,
                'slope_vs_Fext_R2': r2f,
                'slope_vs_Fext_p': pf,
                'intercept_vs_v0_slope': siv,
                'intercept_vs_v0_R2': r2iv,
                'intercept_vs_v0_p': piv,
                'slope_vs_v0_slope': sfv,
                'slope_vs_v0_R2': r2fv,
                'slope_vs_v0_p': pfv,
                'n_experiments': len(constant_results)
            }
        })

        # 常数场回归参数表格 OBS
        table_lines = [
            f"{'exp_id':>6} {'F_ext':>6} {'v0':>4} {'slope':>10} {'intercept':>10} {'R²':>8} {'p_value':>12}"
        ]
        table_lines.append('-' * len(table_lines[0]))
        for r in constant_results:
            table_lines.append(
                f"{r['experiment_id']:>6} {r['F_ext']:>6.2f} {r['v0']:>4.1f} "
                f"{r['slope']:>10.6f} {r['intercept']:>10.6f} {r['r2']:>8.6f} {r['p_value']:>12.4e}"
            )
        table_str = '\n'.join(table_lines)
        observations.append({
            'summary': f"Constant-field a-v regression parameters:\n{table_str}",
            'source_data_refs': [f"{r['experiment_id']}:a,{r['experiment_id']}:v" for r in constant_results],
            'metrics': {'constant_experiment_count': len(constant_results)}
        })

    # Step 4: 自由场回归确认
    if free_results:
        slopes_free = np.array([r['slope'] for r in free_results])
        ints_free = np.array([r['intercept'] for r in free_results])
        slopes_near_zero = bool(np.all(np.abs(slopes_free) < 1e-12))
        ints_near_zero = bool(np.all(np.abs(ints_free) < 1e-12))

        free_summary = (
            f"Free-field experiments (exp_01,04,07): {len(free_results)} experiments.\n"
            f"  Slopes near zero: {slopes_near_zero} (max abs={np.max(np.abs(slopes_free)):.2e})\n"
            f"  Intercepts near zero: {ints_near_zero} (max abs={np.max(np.abs(ints_free)):.2e})"
        )
        free_table_lines = [
            f"{'exp_id':>6} {'slope':>10} {'intercept':>10} {'R²':>8} {'p_value':>12}"
        ]
        free_table_lines.append('-' * len(free_table_lines[0]))
        for r in free_results:
            free_table_lines.append(
                f"{r['experiment_id']:>6} {r['slope']:>10.6e} {r['intercept']:>10.6e} "
                f"{r['r2']:>8.6f} {r['p_value']:>12.4e}"
            )
        free_table_str = '\n'.join(free_table_lines)
        observations.append({
            'summary': f"Free-field regression details:\n{free_table_str}\n{free_summary}",
            'source_data_refs': [f"{r['experiment_id']}:a,{r['experiment_id']}:v" for r in free_results],
            'metrics': {
                'free_experiment_count': len(free_results),
                'slopes_near_zero': slopes_near_zero,
                'intercepts_near_zero': ints_near_zero,
                'max_abs_slope': float(np.max(np.abs(slopes_free))),
                'max_abs_intercept': float(np.max(np.abs(ints_free)))
            }
        })

    # 总观察
    total_obs = (
        f"维护实验数据记录表完成。处理了 {len(experiment_ids)} 个实验。"
        f"为 exp_16~19 新定义了 v 和 a 序列。"
        f"对 {len(constant_results)} 个 constant 场实验和 {len(free_results)} 个 free 场实验执行了 a-v 线性回归。"
        f"跨实验回归已计算。共生成 {len(observations)} 条 OBS。未宣布任何定律。"
    )

    return {
        'observation': total_obs,
        'derived_series': derived_series,
        'observations': observations,
        'figures': [],
        'metrics': {
            'experiments_processed': len(experiment_ids),
            'derived_series_count': len(derived_series),
            'observation_count': len(observations),
            'constant_experiment_regressions': len(constant_results),
            'free_experiment_regressions': len(free_results)
        }
    }
