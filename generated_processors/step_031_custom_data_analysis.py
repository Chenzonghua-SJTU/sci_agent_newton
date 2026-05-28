import os
import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict


def _estimate_velocity_and_drag(t, q, F_ext):
    """从 q(t) 估计速度和加速度，并计算 drag = F_ext - a。
    使用中心差分（两端用前后差分），返回 v 和 drag 数组。
    """
    n = len(t)
    dt = t[1] - t[0] if n > 1 else 1.0
    v = np.zeros(n)
    v[0] = (q[1] - q[0]) / dt
    v[-1] = (q[-1] - q[-2]) / dt
    for i in range(1, n - 1):
        v[i] = (q[i+1] - q[i-1]) / (2 * dt)
    # 加速度
    a = np.zeros(n)
    a[0] = (v[1] - v[0]) / dt
    a[-1] = (v[-1] - v[-2]) / dt
    for i in range(1, n - 1):
        a[i] = (v[i+1] - v[i-1]) / (2 * dt)
    drag = F_ext - a
    return v.tolist(), drag.tolist()


def _get_velocity_and_drag(exp_id, exp_data):
    """从实验数据中提取速度 v 和 drag 序列，优先级：
    - 首先使用已存在的 velocity 或 v_est 作为 v，drag 序列作为 drag
    - 如果没有，则从 q,t 和 F_ext 估计
    返回 (v_array, drag_array, F_ext)
    """
    config = exp_data['config']
    series = exp_data['series']
    available = exp_data['available_series']

    # 获取 F_ext
    if 'F_ext' in config:
        F_ext = float(config['F_ext'])
    elif 'constant_force' in config:
        F_ext = float(config['constant_force'])
    elif config.get('force_field_type') == 'free':
        F_ext = 0.0
    else:
        raise ValueError(f"Experiment {exp_id}: cannot determine F_ext")

    # 检查已有序列
    v_key = None
    for key in ['velocity', 'v_est']:
        if key in series:
            v_key = key
            break
    drag_key = 'drag' if 'drag' in series else None

    if v_key is not None and drag_key is not None:
        v = np.array(series[v_key], dtype=float)
        drag = np.array(series[drag_key], dtype=float)
        # 检查长度一致
        t_len = len(series['t'])
        if len(v) != t_len or len(drag) != t_len:
            raise ValueError(f"Experiment {exp_id}: velocity/drag length mismatch")
    else:
        # 需要从 q,t 估计
        if 'q' not in series or 't' not in series:
            raise ValueError(f"Experiment {exp_id}: no q/t series for estimation")
        q = np.array(series['q'])
        t = np.array(series['t'])
        v_list, drag_list = _estimate_velocity_and_drag(t, q, F_ext)
        v = np.array(v_list)
        drag = np.array(drag_list)
        # 返回时通过 derived_series 注册，此处仅内部使用

    return v, drag, F_ext


def _fit_power(v, drag, initial_b=0.5, initial_A=0.5):
    """拟合 drag = A * v^b，返回 (b, A, cov, R2, rmse)"""
    # 过滤 v>0
    mask = v > 0
    if np.sum(mask) < 5:
        return None, None, None, None, None
    v_pos = v[mask]
    drag_pos = drag[mask]

    def model(v, b, A):
        return A * np.power(v, b)

    # 初始猜测：对数值回归
    logv = np.log(v_pos)
    logdrag = np.log(drag_pos)
    coeffs = np.polyfit(logv, logdrag, 1)
    b0 = coeffs[0]
    A0 = np.exp(coeffs[1])
    # 使用对数拟合作为初始值
    try:
        popt, pcov = curve_fit(model, v_pos, drag_pos, p0=[b0, A0],
                               bounds=([0.0, 0.0], [5.0, 10.0]),
                               maxfev=10000)
    except RuntimeError:
        # 尝试更简单的初始值
        try:
            popt, pcov = curve_fit(model, v_pos, drag_pos, p0=[initial_b, initial_A],
                                   bounds=([0.0, 0.0], [5.0, 10.0]),
                                   maxfev=10000)
        except RuntimeError:
            return None, None, None, None, None

    b_fit, A_fit = popt
    perr = np.sqrt(np.diag(pcov))
    pred = model(v_pos, *popt)
    resid = drag_pos - pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((drag_pos - np.mean(drag_pos))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(resid**2))
    return b_fit, A_fit, perr, R2, rmse


def _fit_power_ratio(v, ratio, initial_b=0.5, initial_k=0.5):
    """拟合 ratio = k * v^b"""
    return _fit_power(v, ratio, initial_b, initial_k)


def _fit_multi_variable(all_v, all_F, all_drag):
    """多变量拟合 drag = A * v^b * F^c，返回参数和协方差"""
    mask = (all_v > 0) & (all_F > 0)
    if np.sum(mask) < 10:
        return None, None, None, None, None
    v = all_v[mask]
    F = all_F[mask]
    drag = all_drag[mask]

    def model(vF, b, c, A):
        v = vF[0]
        F = vF[1]
        return A * (v**b) * (F**c)

    # 对数线性初始估计
    logv = np.log(v)
    logF = np.log(F)
    logdrag = np.log(drag)
    X = np.column_stack([logv, logF, np.ones_like(logv)])
    coeffs, _, _, _ = np.linalg.lstsq(X, logdrag, rcond=None)
    b0 = coeffs[0]
    c0 = coeffs[1]
    A0 = np.exp(coeffs[2])

    try:
        popt, pcov = curve_fit(lambda vF, b, c, A: model(vF, b, c, A),
                               (v, F), drag, p0=[b0, c0, A0],
                               bounds=([0.0, 0.0, 0.0], [5.0, 5.0, 10.0]),
                               maxfev=20000)
    except RuntimeError:
        return None, None, None, None, None

    b_fit, c_fit, A_fit = popt
    perr = np.sqrt(np.diag(pcov))
    pred = model((v, F), *popt)
    resid = drag - pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((drag - np.mean(drag))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(resid**2))
    return (b_fit, c_fit, A_fit), perr, R2, rmse


def _fit_lin_sqrt(v, y):
    """拟合 y = p*v + q*sqrt(v)，使用线性最小二乘"""
    mask = v > 0
    if np.sum(mask) < 5:
        return None, None, None, None
    v_pos = v[mask]
    y_pos = y[mask]
    A = np.column_stack([v_pos, np.sqrt(v_pos)])
    popt, _, _, _ = np.linalg.lstsq(A, y_pos, rcond=None)
    p_fit, q_fit = popt
    pred = A @ popt
    resid = y_pos - pred
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_pos - np.mean(y_pos))**2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean(resid**2))
    return (p_fit, q_fit), R2, rmse


def process(payload: dict) -> dict:
    params = payload['parameters']
    experiments = payload['experiments']
    output_dir = payload['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    analysis_goal = params.get('analysis_goal', '')
    exp_ids = params.get('experiment_ids', [])
    if not exp_ids:
        exp_ids = list(experiments.keys())

    # 收集所有实验的数据
    all_v = []      # 速度
    all_F = []      # 外力
    all_drag = []   # drag
    per_exp_data = {}  # exp_id -> (v, drag, F_ext, ratio)
    derived_series = []
    figures = []
    metrics = {}

    # 先为每个实验准备数据，必要时估计 v 和 drag
    for eid in exp_ids:
        if eid not in experiments:
            continue
        exp_data = experiments[eid]
        try:
            v, drag, F = _get_velocity_and_drag(eid, exp_data)
        except Exception as exc:
            # 如果无法获取，跳过后续分析
            metrics[f'{eid}_error'] = str(exc)
            continue

        # 检查速度序列是否已经存在于实验中，若不存在则注册为新派生序列
        series = exp_data['series']
        t_len = len(series['t'])
        if 'velocity' not in series and 'v_est' not in series:
            # 注册速度序列，命名为 v_est
            derived_series.append({
                'experiment_id': eid,
                'name': 'v_est',
                'values': v.tolist(),
                'source_name': 'estimated from q using central difference',
                'provenance': 'generated data processor: custom_data_analysis',
                'description': 'estimated velocity'
            })
        if 'drag' not in series:
            # 注册 drag 序列
            derived_series.append({
                'experiment_id': eid,
                'name': 'drag',
                'values': drag.tolist(),
                'source_name': 'drag = F_ext - a (a estimated from q)',
                'provenance': 'generated data processor: custom_data_analysis',
                'description': 'drag force'
            })

        # 过滤 v>0 的有效点
        mask = v > 0
        if np.sum(mask) < 5:
            metrics[f'{eid}_skipped'] = 'too few positive v points'
            continue

        v_valid = v[mask]
        drag_valid = drag[mask]
        ratio_valid = drag_valid / F if F != 0 else np.zeros_like(v_valid)

        # 检查 ratio_drag_over_F 是否已存在，若不存在则注册
        if 'ratio_drag_over_F' not in series:
            derived_series.append({
                'experiment_id': eid,
                'name': 'ratio_drag_over_F',
                'values': (drag / F).tolist() if F != 0 else [0.0]*t_len,
                'source_name': 'drag / F_ext',
                'provenance': 'generated data processor: custom_data_analysis',
                'description': 'normalized drag'
            })

        per_exp_data[eid] = {
            'v': v_valid,
            'drag': drag_valid,
            'ratio': ratio_valid,
            'F_ext': F
        }
        all_v.extend(v_valid.tolist())
        all_F.extend([F]*len(v_valid))
        all_drag.extend(drag_valid.tolist())

    all_v = np.array(all_v)
    all_F = np.array(all_F)
    all_drag = np.array(all_drag)

    # 1. 多变量拟合 drag = A * v^b * F_ext^c
    if len(all_v) >= 10:
        multi_res = _fit_multi_variable(all_v, all_F, all_drag)
        if multi_res[0] is not None:
            (b_m, c_m, A_m), perr_m, R2_m, rmse_m = multi_res
            metrics['multi_b'] = b_m
            metrics['multi_c'] = c_m
            metrics['multi_A'] = A_m
            metrics['multi_b_std'] = perr_m[0]
            metrics['multi_c_std'] = perr_m[1]
            metrics['multi_A_std'] = perr_m[2]
            metrics['multi_R2'] = R2_m
            metrics['multi_RMSE'] = rmse_m
        else:
            metrics['multi_fit_error'] = 'multi variable fit failed'
    else:
        metrics['multi_fit_skip'] = 'insufficient data'

    # 2. 单独对每个实验拟合 ratio = k * v^b
    fit_params = {}
    for eid, data in per_exp_data.items():
        b, k, perr, R2, rmse = _fit_power_ratio(data['v'], data['ratio'])
        if b is not None:
            metrics[f'{eid}_ratio_power_b'] = b
            metrics[f'{eid}_ratio_power_k'] = k
            metrics[f'{eid}_ratio_power_R2'] = R2
            metrics[f'{eid}_ratio_power_RMSE'] = rmse
            metrics[f'{eid}_F_ext'] = data['F_ext']
            fit_params[eid] = {'b': b, 'k': k, 'F': data['F_ext'], 'R2': R2}
        else:
            metrics[f'{eid}_ratio_power_fail'] = True

    # 3. drag vs v 散点图，按 F_ext 着色
    fig1, ax1 = plt.subplots(figsize=(8,6))
    colors = plt.cm.jet(np.linspace(0,1,len(per_exp_data)))
    for idx, (eid, data) in enumerate(per_exp_data.items()):
        ax1.scatter(data['v'], data['drag'], c=[colors[idx]], label=f'{eid} (F={data["F_ext"]:.2f})', s=10, alpha=0.7)
    ax1.set_xlabel('velocity v')
    ax1.set_ylabel('drag force')
    ax1.set_title('Drag vs Velocity by experiment (F_ext)')
    ax1.legend()
    fig1.tight_layout()
    fname1 = os.path.join(output_dir, 'drag_vs_v_by_F_ext.png')
    fig1.savefig(fname1, dpi=150)
    plt.close(fig1)
    figures.append(fname1)

    # 4. 合并所有实验，拟合 drag/F_ext 与 v 的关系：线性+平方根和幂律
    # 构造合并 ratio 和 v
    all_ratio = all_drag / all_F
    # 过滤无效
    mask_global = (all_v > 0) & (all_F > 0)
    if np.sum(mask_global) >= 10:
        v_global = all_v[mask_global]
        ratio_global = all_ratio[mask_global]
        # 线性+平方根
        lin_sqrt_res = _fit_lin_sqrt(v_global, ratio_global)
        if lin_sqrt_res[0] is not None:
            (p_lin, q_lin), R2_lin, rmse_lin = lin_sqrt_res
            metrics['global_lin_sqrt_p'] = p_lin
            metrics['global_lin_sqrt_q'] = q_lin
            metrics['global_lin_sqrt_R2'] = R2_lin
            metrics['global_lin_sqrt_RMSE'] = rmse_lin
        # 幂律
        b_gl, k_gl, perr_gl, R2_gl, rmse_gl = _fit_power_ratio(v_global, ratio_global)
        if b_gl is not None:
            metrics['global_ratio_power_b'] = b_gl
            metrics['global_ratio_power_k'] = k_gl
            metrics['global_ratio_power_R2'] = R2_gl
            metrics['global_ratio_power_RMSE'] = rmse_gl

        # 绘制合并数据及拟合曲线
        fig2, ax2 = plt.subplots(figsize=(8,6))
        # 原始散点
        for eid, data in per_exp_data.items():
            ax2.scatter(data['v'], data['ratio'], s=8, alpha=0.6, label=f'{eid} (F={data["F_ext"]:.2f})')
        # 拟合曲线
        v_sort = np.sort(v_global)
        if lin_sqrt_res[0] is not None:
            pred_lin = p_lin * v_sort + q_lin * np.sqrt(v_sort)
            ax2.plot(v_sort, pred_lin, 'r-', lw=2, label=f'lin+sqrt (R²={R2_lin:.4f})')
        if b_gl is not None:
            pred_pow = k_gl * (v_sort**b_gl)
            ax2.plot(v_sort, pred_pow, 'g--', lw=2, label=f'power law (R²={R2_gl:.4f})')
        ax2.set_xlabel('velocity v')
        ax2.set_ylabel('drag / F_ext')
        ax2.set_title('Global fit of drag/F_ext vs v')
        ax2.legend()
        fig2.tight_layout()
        fname2 = os.path.join(output_dir, 'global_drag_over_F_vs_v_fit.png')
        fig2.savefig(fname2, dpi=150)
        plt.close(fig2)
        figures.append(fname2)
    else:
        metrics['global_fit_skip'] = 'insufficient global data'

    # 此外，为每个实验绘制单独的 drag/F_ext vs v 拟合图（可选，但可帮助判断）
    # 但 unexpected_outputs 没有要求每个实验单独图，所以跳过避免 clutter

    # 构建 observation
    obs_lines = []
    obs_lines.append(f"分析覆盖 {len(per_exp_data)} 个恒外力实验：{', '.join(per_exp_data.keys())}")
    if 'multi_b' in metrics:
        obs_lines.append(
            f"多变量拟合 drag = A * v^b * F_ext^c: "
            f"A={metrics['multi_A']:.4f}±{metrics['multi_A_std']:.4f}, "
            f"b={metrics['multi_b']:.4f}±{metrics['multi_b_std']:.4f}, "
            f"c={metrics['multi_c']:.4f}±{metrics['multi_c_std']:.4f}, "
            f"R²={metrics['multi_R2']:.4f}, RMSE={metrics['multi_RMSE']:.4f}"
        )
    obs_lines.append("各实验 drag/F_ext = k * v^b 拟合 (ratio):")
    for eid in per_exp_data:
        if f'{eid}_ratio_power_b' in metrics:
            obs_lines.append(
                f"  {eid} (F={metrics[f'{eid}_F_ext']:.2f}): "
                f"b={metrics[f'{eid}_ratio_power_b']:.4f}, "
                f"k={metrics[f'{eid}_ratio_power_k']:.4f}, "
                f"R²={metrics[f'{eid}_ratio_power_R2']:.4f}"
            )
    if 'global_lin_sqrt_R2' in metrics:
        obs_lines.append(
            f"全局合并拟合 drag/F_ext = p*v + q*sqrt(v): "
            f"p={metrics['global_lin_sqrt_p']:.4f}, "
            f"q={metrics['global_lin_sqrt_q']:.4f}, "
            f"R²={metrics['global_lin_sqrt_R2']:.4f}"
        )
    if 'global_ratio_power_R2' in metrics:
        obs_lines.append(
            f"全局合并拟合 drag/F_ext = k * v^b: "
            f"k={metrics['global_ratio_power_k']:.4f}, "
            f"b={metrics['global_ratio_power_b']:.4f}, "
            f"R²={metrics['global_ratio_power_R2']:.4f}"
        )
    obs_lines.append(f"已保存散点图 {len(figures)} 张。")
    observation = "\n".join(obs_lines)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
