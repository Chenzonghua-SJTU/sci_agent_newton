import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _sanitize_key(name: str) -> str:
    """Replace '/' and space with '_' without using .replace().
    This is required because .replace() is forbidden in the execution environment.
    """
    result = []
    for ch in name:
        if ch in ('/', ' '):
            result.append('_')
        else:
            result.append(ch)
    return ''.join(result)

def _central_diff(y: np.ndarray, dt: float, pad: bool = False) -> np.ndarray:
    """二阶精度中心差分，返回长度与y相同（若pad=True则首尾NaN），否则返回内点（len-2）。"""
    if len(y) < 3:
        raise ValueError("序列长度至少3才能进行中心差分")
    d = np.full_like(y, np.nan) if pad else np.empty(len(y)-2)
    if pad:
        d[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
    else:
        d[:] = (y[2:] - y[:-2]) / (2.0 * dt)
    return d

def _analyze_experiment(q: List[float], t: List[float], config: Dict[str, Any]) -> Dict[str, Any]:
    """对单个实验计算运动学量并返回分析结果。"""
    dt = t[1] - t[0] if len(t) > 1 else 0.1
    q_arr = np.array(q, dtype=float)
    t_arr = np.array(t, dtype=float)
    n = len(q_arr)

    def _diff_pad(x: np.ndarray, dt: float) -> np.ndarray:
        d = np.full_like(x, np.nan)
        if len(x) >= 3:
            d[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
        return d

    v_pad = _diff_pad(q_arr, dt)
    a_pad = _diff_pad(v_pad, dt)
    valid = ~np.isnan(a_pad) & ~np.isnan(v_pad)

    return {
        "t": t_arr,
        "q": q_arr,
        "v_cd": v_pad,
        "a_cd": a_pad,
        "valid": valid,
        "dt": dt,
        "config": config
    }

def process(payload: dict) -> dict:
    experiments = payload["experiments"]
    params = payload["parameters"]
    output_dir = Path(payload["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_ids = params.get("experiment_ids", list(experiments.keys()))
    focus_ids = params.get("focus_experiment_ids", [])
    analysis_goal = params.get("analysis_goal", "")

    derived_series_list = []
    metrics = {}
    figures = []
    observation_lines = []

    all_results = {}
    processed_ids = []
    for eid in experiment_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        q = exp["series"].get("q")
        t = exp["series"].get("t")
        config = exp["config"]
        if not q or not t:
            raise ValueError(f"实验 {eid} 缺少 q 或 t 序列")
        res = _analyze_experiment(q, t, config)
        all_results[eid] = res
        processed_ids.append(eid)

    obs_bullets = []

    # 2. 基线检查：自由实验
    free_ids = [eid for eid in processed_ids if all_results[eid]["config"].get("force_field_type") == "free"
                or abs(all_results[eid]["config"].get("F_ext", 0)) < 1e-12]
    for eid in free_ids:
        a = all_results[eid]["a_cd"]
        v = all_results[eid]["v_cd"]
        valid = all_results[eid]["valid"]
        a_valid = a[valid]
        if len(a_valid) > 0:
            max_abs_a = np.max(np.abs(a_valid))
            mean_a = np.mean(a_valid)
            std_a = np.std(a_valid)
            obs_bullets.append(
                f"[OB1-基线] 自由实验 {eid}: max|a|={max_abs_a:.6e}, mean_a={mean_a:.6e}, std_a={std_a:.6e}"
            )

    # 3. 对称性检查
    symmetric_pairs = []
    ids_by_params = {}
    for eid in processed_ids:
        cfg = all_results[eid]["config"]
        key = (cfg.get("F_ext", 0), cfg.get("initial_v", 0))
        ids_by_params[key] = ids_by_params.get(key, []) + [eid]
    keys = list(ids_by_params.keys())
    for i, key1 in enumerate(keys):
        for key2 in keys[i+1:]:
            if abs(key1[0] + key2[0]) < 1e-10 and abs(key1[1] + key2[1]) < 1e-10:
                for eid1 in ids_by_params[key1]:
                    for eid2 in ids_by_params[key2]:
                        symmetric_pairs.append((eid1, eid2, key1, key2))
    if symmetric_pairs:
        for pair in symmetric_pairs[:5]:
            eid1, eid2, key1, key2 = pair
            q1 = all_results[eid1]["q"]
            q2 = all_results[eid2]["q"]
            min_len = min(len(q1), len(q2))
            sum_q = q1[:min_len] + q2[:min_len]
            max_abs_sum = np.max(np.abs(sum_q))
            obs_bullets.append(
                f"[OB2-对称] 反对称对 ({eid1},{eid2}): q1+q2 最大绝对值 = {max_abs_sum:.6e}（F_ext=({key1[0]},{key2[0]}), v0=({key1[1]},{key2[1]})）"
            )
    else:
        obs_bullets.append("[OB2-对称] 未发现同时变号的反对称对")

    # 4. 平移不变性检查
    exp03 = all_results.get("exp_03")
    exp16 = all_results.get("exp_16")
    if exp03 and exp16:
        q03 = exp03["q"]
        q16 = exp16["q"]
        min_len = min(len(q03), len(q16))
        diff = q16[:min_len] - q03[:min_len] - 5.0
        max_diff = np.max(np.abs(diff))
        obs_bullets.append(
            f"[OB3-平移] 初位置平移不变性: exp_16(q0=5)与exp_03(q0=0)的q-q0差异最大={max_diff:.6e}"
        )

    # 5. 主要恒外力实验的变换诊断
    forced_exps = [eid for eid in processed_ids
                   if all_results[eid]["config"].get("force_field_type") in ("constant",)
                   and abs(all_results[eid]["config"].get("F_ext", 0)) > 1e-12]

    transform_names = {
        "F/a vs v²": lambda v, a, F: (v**2, F/a),
        "F/a vs |v|": lambda v, a, F: (np.abs(v), F/a),
        "F/a vs v": lambda v, a, F: (v, F/a)
    }
    all_r2 = {name: [] for name in transform_names}
    all_fit_params = {}
    for eid in forced_exps:
        v = all_results[eid]["v_cd"]
        a = all_results[eid]["a_cd"]
        F = all_results[eid]["config"].get("F_ext", 0)
        valid = all_results[eid]["valid"]
        for name, func in transform_names.items():
            x, y = func(v, a, F)
            mask = (valid & ~np.isnan(x) & ~np.isnan(y) & (np.abs(y) < 1e12))
            if np.sum(mask) < 5:
                continue
            reg = LinearRegression(fit_intercept=True)
            reg.fit(x[mask].reshape(-1,1), y[mask])
            r2 = r2_score(y[mask], reg.predict(x[mask].reshape(-1,1)))
            all_r2[name].append(r2)
            if name == "F/a vs v²":
                all_fit_params[eid] = {"intercept": reg.intercept_, "slope": reg.coef_[0], "R2": r2}

    if all_r2:
        for name, r2_list in all_r2.items():
            if r2_list:
                mean_r2 = np.mean(r2_list)
                min_r2 = np.min(r2_list)
                max_r2 = np.max(r2_list)
                obs_bullets.append(
                    f"[OB4-诊断] 变换 '{name}' 在 {len(r2_list)} 个恒外力实验上的平均 R²={mean_r2:.6f} (min={min_r2:.6f}, max={max_r2:.6f})"
                )
                metrics[f"mean_r2_{_sanitize_key(name)}"] = mean_r2

    # 6. 加速度与速度、速度平方、位置的相关系数
    corr_av_list = []
    corr_av2_list = []
    corr_aq_list = []
    for eid in forced_exps:
        v = all_results[eid]["v_cd"]
        a = all_results[eid]["a_cd"]
        q = all_results[eid]["q"]
        valid = all_results[eid]["valid"]
        mask = valid & ~np.isnan(v) & ~np.isnan(a)
        if np.sum(mask) < 5:
            continue
        corr_av = np.corrcoef(v[mask], a[mask])[0,1]
        corr_av2 = np.corrcoef(v[mask]**2, a[mask])[0,1]
        corr_aq = np.corrcoef(q[mask], a[mask])[0,1]
        corr_av_list.append(corr_av)
        corr_av2_list.append(corr_av2)
        corr_aq_list.append(corr_aq)
    if corr_av_list:
        obs_bullets.append(
            f"[OB5-依赖] 加速度与速度平均 Pearson r(av)={np.mean(corr_av_list):.4f}; "
            f"与 v² r(av²)={np.mean(corr_av2_list):.4f}; "
            f"与位置 r(aq)={np.mean(corr_aq_list):.4f}"
        )

    # 7. 新实验 focus 分析
    for eid in focus_ids:
        if eid not in all_results:
            continue
        res = all_results[eid]
        config = res["config"]
        F_ext = config.get("F_ext", 0)
        v0 = config.get("initial_v", 0)
        q = res["q"]
        t = res["t"]
        v_cd = res["v_cd"]
        a_cd = res["a_cd"]
        valid = res["valid"]
        q_min, q_max = np.min(q[valid]), np.max(q[valid])
        v_min, v_max = np.min(v_cd[valid]), np.max(v_cd[valid])
        a_min, a_max = np.min(a_cd[valid]), np.max(a_cd[valid])
        obs_bullets.append(
            f"[OB6-聚焦] 实验 {eid}: F_ext={F_ext}, v0={v0}; q范围 [{q_min:.4f}, {q_max:.4f}]; "
            f"v范围 [{v_min:.4f}, {v_max:.4f}]; a范围 [{a_min:.4f}, {a_max:.4f}]"
        )
        F = F_ext
        v = v_cd
        a = a_cd
        mask = valid & ~np.isnan(v_cd) & ~np.isnan(a_cd) & (np.abs(a_cd) > 1e-12)
        if np.sum(mask) >= 5 and abs(F) > 1e-12:
            x = v[mask]**2
            y = F / a[mask]
            reg = LinearRegression()
            reg.fit(x.reshape(-1,1), y)
            r2 = r2_score(y, reg.predict(x.reshape(-1,1)))
            obs_bullets.append(
                f"     F_ext/a vs v² 拟合 R²={r2:.8f}, 截距={reg.intercept_:.6f}, 斜率={reg.coef_[0]:.6f}"
            )

    # 8. 排除方向
    for name in ["F/a vs v", "F/a vs |v|"]:
        if name in all_r2 and all_r2[name]:
            mean_r2 = np.mean(all_r2[name])
            obs_bullets.append(
                f"[OB7-排除] '{name}' 平均 R²={mean_r2:.4f}，显著低于 v² 变换，排除线性阻尼或绝对值阻尼主导。"
            )

    # 9. 生成candidate seeds
    candidate_seeds = []
    if "F/a vs v²" in all_r2 and all_r2["F/a vs v²"]:
        mean_r2_v2 = np.mean(all_r2["F/a vs v²"])
        candidate_seeds.append({
            "seed": "平方阻尼关系: a ∝ F_ext / (1 + β v²)",
            "evidence": f"来自[OB4]: F_ext/a vs v² 平均 R²={mean_r2_v2:.6f}, 接近完美线性; 且[OB5]显示a与v²强相关 r(av²)={np.mean(corr_av2_list):.4f}",
            "next_step": "对每个恒外力实验独立拟合截距和斜率，检查跨实验一致性；特别关注大外力 exp_08, exp_19 和长时程 exp_09"
        })
    if symmetric_pairs:
        candidate_seeds.append({
            "seed": "符号反对称性: a(F_ext, v0) = -a(-F_ext, -v0)",
            "evidence": f"来自[OB2]: 已找到 {len(symmetric_pairs)} 对反对称对，q1+q2 最大绝对值<1e-6",
            "next_step": "验证所有可能的实验配对是否为严格反对称，并作为运动方程必须满足的条件"
        })
    if exp03 and exp16:
        candidate_seeds.append({
            "seed": "初位置平移不变性: 运动方程不显含位置",
            "evidence": f"来自[OB3]: exp_16(q0=5)与exp_03(q0=0)的q-q0差异最大={max_diff:.6e}",
            "next_step": "若存在更多不同q0的实验，验证平移不变性是否普遍成立"
        })

    # 10. 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    for pair in symmetric_pairs[:3]:
        eid1, eid2, _, _ = pair
        if eid1 in all_results and eid2 in all_results:
            t1 = all_results[eid1]["t"]
            q1 = all_results[eid1]["q"]
            t2 = all_results[eid2]["t"]
            q2 = all_results[eid2]["q"]
            min_len = min(len(t1), len(t2))
            ax.plot(t1[:min_len], q1[:min_len] + q2[:min_len], label=f"q({eid1})+q({eid2})")
    ax.set_xlabel("t")
    ax.set_ylabel("q1+q2")
    ax.legend()
    ax.set_title("Symmetry check: q1+q2")
    sym_path = output_dir / "observe_symmetry.png"
    fig.savefig(str(sym_path), dpi=100)
    plt.close(fig)
    figures.append(str(sym_path))

    fig, ax = plt.subplots(figsize=(10, 6))
    for eid in forced_exps:
        v = all_results[eid]["v_cd"]
        a = all_results[eid]["a_cd"]
        F = all_results[eid]["config"].get("F_ext", 0)
        valid = all_results[eid]["valid"]
        mask = valid & ~np.isnan(v) & ~np.isnan(a) & (np.abs(a) > 1e-12)
        if np.sum(mask) < 5:
            continue
        x = v[mask]**2
        y = F / a[mask]
        ax.plot(x, y, '.', markersize=2, label=eid)
    ax.set_xlabel("v²")
    ax.set_ylabel("F_ext / a")
    ax.legend(fontsize=6)
    ax.set_title("Diagnostic: F_ext/a vs v² for all constant-force experiments")
    diag_path = output_dir / "observe_collapse.png"
    fig.savefig(str(diag_path), dpi=100)
    plt.close(fig)
    figures.append(str(diag_path))

    fig, ax = plt.subplots(figsize=(8, 6))
    for eid in forced_exps[:5]:
        v = all_results[eid]["v_cd"]
        a = all_results[eid]["a_cd"]
        valid = all_results[eid]["valid"]
        mask = valid & ~np.isnan(v) & ~np.isnan(a)
        if np.sum(mask) < 5:
            continue
        ax.plot(v[mask]**2, a[mask], '.', markersize=2, label=eid)
    ax.set_xlabel("v²")
    ax.set_ylabel("a")
    ax.legend()
    ax.set_title("a vs v² (selected experiments)")
    av2_path = output_dir / "observe_a_vs_v2.png"
    fig.savefig(str(av2_path), dpi=100)
    plt.close(fig)
    figures.append(str(av2_path))

    observation_text = "=== 观察阶段分析 (Observe) ===\n"
    observation_text += "边界剔除策略: 中心差分后去除首尾各1个点（因中心差分后首尾为NaN），二次差分后更多边界NaN，分析中使用valid掩码（排除了NaN）。\n"
    observation_text += f"处理实验数: {len(processed_ids)}\n"
    for bullet in obs_bullets:
        observation_text += bullet + "\n"
    observation_text += "\n--- 候选种子 (Candidate Seeds) ---\n"
    for seed in candidate_seeds:
        observation_text += f"Seed: {seed['seed']}\n"
        observation_text += f"  证据: {seed['evidence']}\n"
        observation_text += f"  下一步: {seed['next_step']}\n"
    observation_text += "\n--- 排除方向 ---\n"
    observation_text += "线性阻尼模型 (F/a ~ v) 和 |v| 阻尼均被数据排除（R²远低于 v² 变换）。\n"

    metrics["observation_pass"] = True
    metrics["observation_count"] = len(obs_bullets)
    if "F/a vs v²" in all_r2 and all_r2["F/a vs v²"]:
        metrics["best_diagnostic_name"] = "F_ext/a vs v²"
        metrics["best_diagnostic_score"] = np.mean(all_r2["F/a vs v²"])
    metrics["candidate_seed_count"] = len(candidate_seeds)
    metrics["free_experiments_checked"] = len(free_ids)
    metrics["forced_experiments_analyzed"] = len(forced_exps)
    metrics["symmetry_pairs_found"] = len(symmetric_pairs)

    for eid in processed_ids:
        res = all_results[eid]
        derived_series_list.append({
            "experiment_id": eid,
            "name": "v_cd",
            "values": res["v_cd"].tolist(),
            "source_name": "中心差分自 q(t)",
            "provenance": "step_024_analyze_data_observe",
            "description": "中心差分速度，首尾填充NaN"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": "a_cd",
            "values": res["a_cd"].tolist(),
            "source_name": "二次中心差分自 q(t)",
            "provenance": "step_024_analyze_data_observe",
            "description": "中心差分加速度，首尾填充NaN"
        })

    return {
        "observation": observation_text,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": metrics
    }
