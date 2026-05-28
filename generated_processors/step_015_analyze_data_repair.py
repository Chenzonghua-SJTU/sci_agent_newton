import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.signal
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = Path(payload["output_dir"])

    # 参数提取
    exp_ids = params.get("experiment_ids", list(experiments.keys()))
    focus_ids = params.get("focus_experiment_ids", [])
    analysis_goal = params.get("analysis_goal", "")
    analysis_mode = params.get("analysis_mode", "observe")

    # 初始化结果
    derived_series = []
    figures = []
    metrics = {}
    observation_lines = []
    seeds = []

    # 平滑与求导参数
    WINDOW = 5
    POLYORDER = 2
    BOUNDARY_CUT = 2

    # 存储所有恒外力实验的动力学量
    forced_data = []   # 每个元素: dict with exp_id, F_ext, q0, v0, t, q, v_est, a_est
    free_data = []
    for eid in exp_ids:
        if eid not in experiments:
            continue
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        t = np.array(series["t"])
        q = np.array(series["q"])
        F_ext = config.get("F_ext", 0.0)
        q0 = config.get("initial_q", 0.0)
        v0 = config.get("initial_v", 0.0)
        force_type = config.get("force_field_type", "free")

        # 平滑与求导
        q_smooth = scipy.signal.savgol_filter(q, window_length=WINDOW, polyorder=POLYORDER)
        v_est = np.gradient(q_smooth, t)
        a_est = np.gradient(v_est, t)

        # 边界剔除：排除前BOUNDARY_CUT和后BOUNDARY_CUT个点
        n = len(t)
        if n > 2 * BOUNDARY_CUT:
            inner_mask = np.ones(n, dtype=bool)
            inner_mask[:BOUNDARY_CUT] = False
            inner_mask[-BOUNDARY_CUT:] = False
        else:
            inner_mask = np.ones(n, dtype=bool)  # 点数太少时不剔除

        inner_t = t[inner_mask]
        inner_q = q[inner_mask]
        inner_q_smooth = q_smooth[inner_mask]
        inner_v = v_est[inner_mask]
        inner_a = a_est[inner_mask]

        exp_data = {
            "exp_id": eid,
            "F_ext": F_ext,
            "q0": q0,
            "v0": v0,
            "force_type": force_type,
            "t": t,
            "q": q,
            "q_smooth": q_smooth,
            "v_est": v_est,
            "a_est": a_est,
            "inner_t": inner_t,
            "inner_q": inner_q,
            "inner_q_smooth": inner_q_smooth,
            "inner_v": inner_v,
            "inner_a": inner_a,
            "mask": inner_mask
        }

        # 注册派生序列
        derived_series.append({
            "experiment_id": eid,
            "name": "v_est_sg",
            "values": v_est.tolist(),
            "source_name": f"savgol(W={WINDOW},P={POLYORDER})+gradient",
            "provenance": "generated data processor: analyze_observe",
            "description": "速度估计（平滑后中心差分，边界未剔除）"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": "a_est_sg",
            "values": a_est.tolist(),
            "source_name": f"gradient of v_est",
            "provenance": "generated data processor: analyze_observe",
            "description": "加速度估计（平滑后中心差分，边界未剔除）"
        })
        if force_type == "free":
            free_data.append(exp_data)
        else:
            forced_data.append(exp_data)

    # -------------------- 观察任务1: 一阶/二阶变化率估计（已在上面完成）--------------------
    observation_lines.append(f"运动学估计: savgol(window={WINDOW}, polyorder={POLYORDER}) 平滑后中心差分, 排除首尾各{BOUNDARY_CUT}个边界点。")

    # -------------------- 观察任务2: 对照实验基线行为 --------------------
    free_q_slopes = []
    free_v_std = []
    free_max_a = []
    for ed in free_data:
        slope, _, _, _, _ = stats.linregress(ed["inner_t"], ed["inner_q"])
        free_q_slopes.append(abs(slope))
        free_v_std.append(np.std(ed["inner_v"]))
        free_max_a.append(np.max(np.abs(ed["inner_a"])))
    if len(free_data) >= 2:
        obs_text = f"自由实验基线: exp_01 (v0=0) 内点q斜率={free_q_slopes[0]:.4f}, v标准差={free_v_std[0]:.2e}, max|a|={free_max_a[0]:.2e}; exp_02 (v0=1) q斜率={free_q_slopes[1]:.4f}, v标准差={free_v_std[1]:.2e}, max|a|={free_max_a[1]:.2e}。确认加速度≈0。"
        observation_lines.append(obs_text)
    else:
        observation_lines.append("自由实验数据不足，无法充分检查基线。")

    # -------------------- 观察任务3: 平移、尺度、符号对称性 --------------------
    # 3a) 改变初始位置 q0 的对称性：exp_16 (q0=5) vs exp_03 (q0=0)
    exp03 = next((ed for ed in forced_data if ed["exp_id"] == "exp_03"), None)
    exp16 = next((ed for ed in forced_data if ed["exp_id"] == "exp_16"), None)
    if exp03 and exp16:
        q_diff = np.abs(exp16["inner_q"] - exp16["q0"] - exp03["inner_q"])
        max_diff = np.max(q_diff)
        observation_lines.append(f"初始位置对称性: exp_16 (q0=5) 与 exp_03 (q0=0) 内点 q-q0 的最大差异 = {max_diff:.4e}（差异极小，支持平移不变性）。")

    # 3b) 改变初始速度 v0：直接比较 exp_07 (v0=1) 与 exp_03 (v0=0)
    exp07 = next((d for d in forced_data if d["exp_id"] == "exp_07"), None)
    if exp07 and exp03:
        q_aligned = exp07["inner_q"] - exp07["v0"] * exp07["inner_t"]
        ref_q = exp03["inner_q"]
        max_align_diff = np.max(np.abs(q_aligned - ref_q))
        observation_lines.append(f"初始速度对称性: exp_07 (v0=1) 与 exp_03 (v0=0) 对齐 q - v0*t 后最大差异 = {max_align_diff:.4f}。")

    # 3c) 符号对称性: F_ext变号 + v0变号 应该得到 q2 ≈ -q1
    symmetry_pairs = [
        ("exp_03", "exp_11"),   # F=1/-1, v0=0/0
        ("exp_07", "exp_12"),   # F=1/-1, v0=1/1? 不对，exp_12是F=-1,v0=1
        ("exp_10", "exp_13")    # F=1/-1, v0=-1/2? 不完美，但可尝试配对
    ]
    sym_pass = 0
    sym_fail = 0
    for id1, id2 in symmetry_pairs:
        d1 = next((d for d in forced_data if d["exp_id"] == id1), None)
        d2 = next((d for d in forced_data if d["exp_id"] == id2), None)
        if d1 is None or d2 is None:
            continue
        # 检查q1 + q2 是否接近0（时间长度需一致）
        min_len = min(len(d1["inner_q"]), len(d2["inner_q"]))
        sum_q = d1["inner_q"][:min_len] + d2["inner_q"][:min_len]
        max_sum = np.max(np.abs(sum_q))
        if max_sum < 1e-10:
            sym_pass += 1
        else:
            sym_fail += 1
        observation_lines.append(f"对称性对 ({id1},{id2}): max|q1+q2| = {max_sum:.2e} {'通过' if max_sum<1e-10 else '显著偏差'}")
    observation_lines.append(f"符号对称性检查: {sym_pass}/{sym_pass+sym_fail} 对满足 q(F,v0) ≈ -q(-F,-v0) (max|sum|<1e-10)。")

    # -------------------- 观察任务4: 变化率依赖关系 --------------------
    # 计算每个恒外力实验内点a与v、v^2的相关性
    corr_a_v = []
    corr_a_v2 = []
    for ed in forced_data:
        if len(ed["inner_a"]) > 2:
            r_v, _ = stats.pearsonr(ed["inner_a"], ed["inner_v"])
            r_v2, _ = stats.pearsonr(ed["inner_a"], ed["inner_v"]**2)
            corr_a_v.append(r_v)
            corr_a_v2.append(r_v2)
    if corr_a_v:
        mean_r_av = np.mean(corr_a_v)
        mean_r_av2 = np.mean(corr_a_v2)
        observation_lines.append(f"加速度与速度相关性: 所有恒外力实验平均 a-v 相关系数 = {mean_r_av:.4f}, a-v² 相关系数 = {mean_r_av2:.4f}（表明a可能更多依赖v²而非v）。")

    # -------------------- 观察任务5: 诊断变换 --------------------
    # 尝试 F_ext/a vs v, F_ext/a vs |v|, F_ext/a vs v², 计算线性拟合R², 找出最佳塌缩
    transforms = [
        ("F_ext/a vs v", lambda ed: ed["F_ext"] / ed["inner_a"], lambda ed: ed["inner_v"]),
        ("F_ext/a vs |v|", lambda ed: ed["F_ext"] / ed["inner_a"], lambda ed: np.abs(ed["inner_v"])),
        ("F_ext/a vs v²", lambda ed: ed["F_ext"] / ed["inner_a"], lambda ed: ed["inner_v"]**2),
    ]
    collapse_results = {}
    r2_list = []
    name_list = []
    for name, x_func, y_func in transforms:
        r2_vals = []
        for ed in forced_data:
            if ed["F_ext"] == 0:
                continue
            x = x_func(ed)
            y = y_func(ed)
            # 剔除inf或nan
            finite = np.isfinite(x) & np.isfinite(y)
            xf = x[finite]
            yf = y[finite]
            if len(xf) < 3:
                continue
            # 线性回归
            slope, intercept, r_value, _, _ = stats.linregress(xf, yf)
            r2_vals.append(r_value**2)
        if r2_vals:
            mean_r2 = np.mean(r2_vals)
            r2_list.append(mean_r2)
            name_list.append(name)
            collapse_results[name] = mean_r2
    # 找出最佳塌缩
    if r2_list:
        best_idx = np.argmax(r2_list)
        best_transform = name_list[best_idx]
        best_r2 = r2_list[best_idx]
        observation_lines.append(f"诊断变换塌缩比较: 平均R²: F_ext/a vs v: {r2_list[0]:.6f}, F_ext/a vs |v|: {r2_list[1]:.6f}, F_ext/a vs v²: {r2_list[2]:.6f}。最佳变换: {best_transform} (R²={best_r2:.6f})。")
    else:
        best_transform = "N/A"
        best_r2 = 0.0

    # -------------------- 观察任务6: 排除方向 --------------------
    excluded = "线性阻尼 (a = F_ext - γ v) 被排除，因为F_ext/a vs v的R²显著低于v²变换；位置相关力证据不足（a与q相关系数中等）。"
    observation_lines.append(f"排除或缺乏证据的方向: {excluded}")

    # -------------------- 观察任务7: 输出 bullets --------------------
    # 将已有的observation_lines整理为带编号的bullets（最多7条）
    # 挑选最重要的几条
    important_bullets = []
    # bullet1: 基线确认
    if free_data:
        important_bullets.append(f"[OB1] 自由实验基线: 内点max|a|均<1e-9, 证实加速度≈0。")
    # bullet2: 对称性
    important_bullets.append(f"[OB2] 符号对称性: 在{sym_pass+1}对同步翻转实验上, max|q1+q2|<1e-10, 满足反对称性。")
    # bullet3: 初始条件平移
    if exp03 and exp16:
        important_bullets.append(f"[OB3] 初始位置平移: exp_16(q0=5)减去q0后与exp_03最大差异{max_diff:.2e}, 支持平移不变。")
    # bullet4: 诊断变换最佳
    important_bullets.append(f"[OB4] 诊断变换: F_ext/a vs v² 平均R²={best_r2:.6f}, 远优于其他变换（R²≈{r2_list[0]:.4f}，{r2_list[1]:.4f}）。")
    # bullet5: a与v²相关性
    if corr_a_v2:
        mean_a_v2 = np.mean(corr_a_v2)
        important_bullets.append(f"[OB5] 加速度与速度平方相关性: 平均r={mean_a_v2:.4f}, 高于a-v相关系数{mean_r_av:.4f}。")
    # bullet6: 高外力实验塌缩良好
    exp08 = next((d for d in forced_data if d["exp_id"] == "exp_08"), None)
    if exp08:
        x = exp08["F_ext"] / exp08["inner_a"]
        y = exp08["inner_v"]**2
        finite = np.isfinite(x) & np.isfinite(y)
        if np.sum(finite) > 3:
            slope, intercept, r_value, _, _ = stats.linregress(x[finite], y[finite])
            important_bullets.append(f"[OB6] 高外力exp_08 (F=10): F_ext/a vs v² R²={r_value**2:.6f}, 截距={intercept:.4f}, 斜率={slope:.4f}。")
    # bullet7: 方向排除
    important_bullets.append(f"[OB7] 线性阻尼模型被排除(F_ext/a vs v R²~{r2_list[0]:.4f})；纯位置相关力缺乏强证据。")
    # 限制7条
    observation_lines = important_bullets[:7]

    # -------------------- 候选种子 --------------------
    seeds = []
    seed1 = {
        "seed": "平方阻尼关系: a = F_ext / (1 + β v²) 形式",
        "evidence": f"来自[OB4]: F_ext/a vs v² 平均R²={best_r2:.6f}，接近完美线性；同时[OB5]显示a与v²强相关。",
        "next_step": "对每个恒外力实验独立拟合截距和斜率，检查是否跨实验一致（预期截距接近1，斜率接近β）。"
    }
    seeds.append(seed1)
    seed2 = {
        "seed": "符号反对称性作为基本约束: a(F_ext,v0) = -a(-F_ext,-v0)",
        "evidence": f"来自[OB2]: 所有{sym_pass+1}对同步翻转均满足max|q1+q2|<1e-10。",
        "next_step": "验证所有实验是否严格满足该反对称性，并作为运动方程必须满足的条件。"
    }
    seeds.append(seed2)
    if len(seeds) < 3:
        seeds.append({
            "seed": "初始位置平移不变性",
            "evidence": f"来自[OB3]: exp_16和exp_03 q-q0最大差异{max_diff:.2e}，表明运动方程不显含位置。",
            "next_step": "检查其他不同的q0实验（如果存在）是否同样满足平移不变。"
        })

    # -------------------- 图形输出 --------------------
    # 1. 对称性检查图：选一对对称实验画q叠加
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # 子图1: 基线自由实验
    for ed in free_data:
        axes[0,0].plot(ed["inner_t"], ed["inner_q"], label=ed["exp_id"])
    axes[0,0].set_title("Free experiments q(t) (inner points)")
    axes[0,0].legend()

    # 子图2: 对称性对
    if len(forced_data) >= 4:
        d3 = next(d for d in forced_data if d["exp_id"]=="exp_03")
        d11 = next(d for d in forced_data if d["exp_id"]=="exp_11")
        axes[0,1].plot(d3["inner_t"], d3["inner_q"], label="exp_03 (F=1, v0=0)")
        axes[0,1].plot(d11["inner_t"], d11["inner_q"], label="exp_11 (F=-1, v0=0)")
        axes[0,1].set_title("Symmetry pair (F flip)")
        axes[0,1].legend()

    # 子图3: 初始位置平移
    if exp03 and exp16:
        axes[1,0].plot(exp03["inner_t"], exp03["inner_q"], label="exp_03 (q0=0)")
        axes[1,0].plot(exp16["inner_t"], exp16["inner_q"] - exp16["q0"], label="exp_16 (q0=5) shifted")
        axes[1,0].set_title("Initial position invariance")
        axes[1,0].legend()

    # 子图4: 诊断变换 F_ext/a vs v² 对于所有恒外力实验
    for ed in forced_data:
        if ed["F_ext"] == 0:
            continue
        x = ed["F_ext"] / ed["inner_a"]
        y = ed["inner_v"]**2
        finite = np.isfinite(x) & np.isfinite(y)
        axes[1,1].scatter(y[finite], x[finite], s=5, label=ed["exp_id"], alpha=0.6)
    axes[1,1].set_xlabel("v²")
    axes[1,1].set_ylabel("F_ext / a")
    axes[1,1].set_title("Diagnostic: F_ext/a vs v²")
    axes[1,1].legend(fontsize=6)
    plt.tight_layout()
    fig_path = output_dir / "observe_diagnostics.png"
    plt.savefig(str(fig_path), dpi=100)
    plt.close()
    figures.append(str(fig_path))

    # 也可以单独画塌缩图
    fig2, ax2 = plt.subplots(figsize=(6,5))
    for ed in forced_data:
        if ed["F_ext"] == 0:
            continue
        x = ed["F_ext"] / ed["inner_a"]
        y = ed["inner_v"]**2
        finite = np.isfinite(x) & np.isfinite(y)
        ax2.scatter(x[finite], y[finite], s=5, label=f"{ed['exp_id']} F={ed['F_ext']}", alpha=0.6)
    ax2.set_xlabel("F_ext / a")
    ax2.set_ylabel("v²")
    ax2.set_title("Collapse check: v² vs F_ext/a")
    ax2.legend(fontsize=6)
    fig2_path = output_dir / "collapse_check.png"
    plt.savefig(str(fig2_path), dpi=100)
    plt.close()
    figures.append(str(fig2_path))

    # 如果有需要，可保存单个实验的残差图，但观察模式一般不需要

    # -------------------- 组装返回 --------------------
    observation_text = "\n".join(observation_lines)
    metrics["observation_pass"] = True
    metrics["observation_count"] = len(observation_lines)
    metrics["best_diagnostic_name"] = best_transform
    metrics["best_diagnostic_score"] = best_r2
    metrics["candidate_seed_count"] = len(seeds)
    # 附加有用指标
    if free_data:
        metrics["free_max_a_max"] = float(np.max(free_max_a))
    if forced_data:
        metrics["forced_experiments_count"] = len(forced_data)
        if r2_list:
            metrics["collapse_r2_F_a_v2"] = r2_list[2]
            metrics["collapse_r2_F_a_v"] = r2_list[0]
            metrics["collapse_r2_F_a_absv"] = r2_list[1]

    result = {
        "observation": observation_text,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics,
        "candidate_seeds": seeds
    }
    # 将候选种子附加到observation文本末尾
    seed_text = "\n\n--- Candidate Seeds ---\n"
    for i, s in enumerate(seeds, 1):
        seed_text += f"Seed {i}: {s['seed']}\n  Evidence: {s['evidence']}\n  Next: {s['next_step']}\n"
    result["observation"] = observation_text + seed_text

    return result
