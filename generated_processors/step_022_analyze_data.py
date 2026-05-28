import json
import statistics
import numpy as np
from scipy import stats as sp_stats
from collections import defaultdict

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    parameters = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    experiment_ids = parameters.get("experiment_ids", list(experiments.keys()))
    output_dir = payload.get("output_dir", "/tmp")

    # Validate mode
    if parameters.get("analysis_mode") != "maintain_ledger":
        return {
            "observation": "错误：期望 analysis_mode=maintain_ledger",
            "observations": [],
            "metrics": {"error": "invalid_mode"}
        }

    # Container for per-experiment results
    records = []
    missing = []

    for eid in experiment_ids:
        if eid not in experiments:
            missing.append(eid)
            continue

        exp = experiments[eid]
        config = exp.get("config", {})
        series = exp.get("series", {})

        # Check required series
        if "a" not in series or "v" not in series:
            missing.append(eid)
            continue

        a = np.array(series["a"])
        v = np.array(series["v"])
        t = np.array(series.get("t", []))

        if len(a) < 2 or len(v) < 2:
            continue

        # Extract endpoints
        a0 = float(a[0])
        a2 = float(a[-1])
        v2 = float(v[-1])

        # v0 from config
        v0 = config.get("initial_v", 0.0)

        # F_ext
        F_ext = config.get("F_ext", 0.0)

        # a-v linear regression
        slope_av, intercept_av, r_value, p_value, std_err = sp_stats.linregress(v, a)

        # Differences
        diff0 = a0 - F_ext
        diff2 = a2 - F_ext
        intercept_diff = intercept_av - F_ext

        record = {
            "experiment_id": eid,
            "F_ext": F_ext,
            "v0": v0,
            "a0": a0,
            "a2": a2,
            "v2": v2,
            "slope_av": slope_av,
            "intercept_av": intercept_av,
            "diff0": diff0,
            "diff2": diff2,
            "intercept_diff": intercept_diff
        }
        records.append(record)

    if missing:
        return {
            "observation": f"缺少实验数据: {missing}, 无法完成分析",
            "observations": [],
            "metrics": {"missing_experiments": missing}
        }

    # Build observations
    observations = []

    # 1. Per-experiment OBS
    for rec in records:
        obs = {
            "summary": f"实验 {rec['experiment_id']}: F_ext={rec['F_ext']}, v0={rec['v0']}, a0={rec['a0']:.6f}, a2={rec['a2']:.6f}, v2={rec['v2']:.6f}, slope_av={rec['slope_av']:.6f}, intercept_av={rec['intercept_av']:.6f}, diff0={rec['diff0']:.6f}, diff2={rec['diff2']:.6f}, intercept_diff={rec['intercept_diff']:.6f}",
            "source_data_refs": [
                f"{rec['experiment_id']}:a",
                f"{rec['experiment_id']}:v",
                f"{rec['experiment_id']}:config"
            ],
            "metrics": {
                "F_ext": rec["F_ext"],
                "v0": rec["v0"],
                "a0": rec["a0"],
                "a2": rec["a2"],
                "v2": rec["v2"],
                "slope_av": rec["slope_av"],
                "intercept_av": rec["intercept_av"],
                "diff0": rec["diff0"],
                "diff2": rec["diff2"],
                "intercept_diff": rec["intercept_diff"]
            }
        }
        observations.append(obs)

    # 2. Overall statistics of differences
    diff0_all = [r["diff0"] for r in records]
    diff2_all = [r["diff2"] for r in records]
    intercept_diff_all = [r["intercept_diff"] for r in records]

    overall_stats = {
        "diff0_mean": statistics.mean(diff0_all),
        "diff0_std": statistics.stdev(diff0_all) if len(diff0_all) > 1 else 0.0,
        "diff2_mean": statistics.mean(diff2_all),
        "diff2_std": statistics.stdev(diff2_all) if len(diff2_all) > 1 else 0.0,
        "intercept_diff_mean": statistics.mean(intercept_diff_all),
        "intercept_diff_std": statistics.stdev(intercept_diff_all) if len(intercept_diff_all) > 1 else 0.0
    }

    obs_overall = {
        "summary": f"整体统计: diff0 均值={overall_stats['diff0_mean']:.6f}, 标准差={overall_stats['diff0_std']:.6f}, diff2 均值={overall_stats['diff2_mean']:.6f}, 标准差={overall_stats['diff2_std']:.6f}, intercept_diff 均值={overall_stats['intercept_diff_mean']:.6f}, 标准差={overall_stats['intercept_diff_std']:.6f}",
        "source_data_refs": [f"{r['experiment_id']}:a" for r in records] + [f"{r['experiment_id']}:v" for r in records],
        "metrics": overall_stats
    }
    observations.append(obs_overall)

    # 3. Group by F_ext
    groups_F_ext = defaultdict(list)
    for r in records:
        groups_F_ext[r["F_ext"]].append(r)

    for F_val, group in groups_F_ext.items():
        diff0_vals = [r["diff0"] for r in group]
        diff2_vals = [r["diff2"] for r in group]
        intercept_diff_vals = [r["intercept_diff"] for r in group]

        stats_F = {
            "F_ext": F_val,
            "n": len(group),
            "diff0_mean": statistics.mean(diff0_vals),
            "diff0_std": statistics.stdev(diff0_vals) if len(diff0_vals) > 1 else 0.0,
            "diff2_mean": statistics.mean(diff2_vals),
            "diff2_std": statistics.stdev(diff2_vals) if len(diff2_vals) > 1 else 0.0,
            "intercept_diff_mean": statistics.mean(intercept_diff_vals),
            "intercept_diff_std": statistics.stdev(intercept_diff_vals) if len(intercept_diff_vals) > 1 else 0.0
        }

        obs_group = {
            "summary": f"按F_ext分组: F_ext={F_val}, n={len(group)}, diff0_mean={stats_F['diff0_mean']:.6f}±{stats_F['diff0_std']:.6f}, diff2_mean={stats_F['diff2_mean']:.6f}±{stats_F['diff2_std']:.6f}, intercept_diff_mean={stats_F['intercept_diff_mean']:.6f}±{stats_F['intercept_diff_std']:.6f}",
            "source_data_refs": [f"{r['experiment_id']}:a" for r in group] + [f"{r['experiment_id']}:v" for r in group],
            "metrics": stats_F
        }
        observations.append(obs_group)

    # 4. Group by v0
    groups_v0 = defaultdict(list)
    for r in records:
        groups_v0[r["v0"]].append(r)

    for v0_val, group in groups_v0.items():
        diff0_vals = [r["diff0"] for r in group]
        diff2_vals = [r["diff2"] for r in group]
        intercept_diff_vals = [r["intercept_diff"] for r in group]

        stats_v0 = {
            "v0": v0_val,
            "n": len(group),
            "diff0_mean": statistics.mean(diff0_vals),
            "diff0_std": statistics.stdev(diff0_vals) if len(diff0_vals) > 1 else 0.0,
            "diff2_mean": statistics.mean(diff2_vals),
            "diff2_std": statistics.stdev(diff2_vals) if len(diff2_vals) > 1 else 0.0,
            "intercept_diff_mean": statistics.mean(intercept_diff_vals),
            "intercept_diff_std": statistics.stdev(intercept_diff_vals) if len(intercept_diff_vals) > 1 else 0.0
        }

        obs_group = {
            "summary": f"按v0分组: v0={v0_val}, n={len(group)}, diff0_mean={stats_v0['diff0_mean']:.6f}±{stats_v0['diff0_std']:.6f}, diff2_mean={stats_v0['diff2_mean']:.6f}±{stats_v0['diff2_std']:.6f}, intercept_diff_mean={stats_v0['intercept_diff_mean']:.6f}±{stats_v0['intercept_diff_std']:.6f}",
            "source_data_refs": [f"{r['experiment_id']}:a" for r in group] + [f"{r['experiment_id']}:v" for r in group],
            "metrics": stats_v0
        }
        observations.append(obs_group)

    # Summarize observation string
    obs_text = (
        f"处理了 {len(records)} 个常数场实验。提取了每个实验的 a0, a2, v0, v2, slope_av, intercept_av，并计算了 diff0=a0-F_ext, diff2=a2-F_ext, intercept_diff=intercept_av-F_ext。"
        f"整体 diff0 均值={overall_stats['diff0_mean']:.6f}，diff2 均值={overall_stats['diff2_mean']:.6f}，intercept_diff 均值={overall_stats['intercept_diff_mean']:.6f}。"
        f"已按 F_ext 和 v0 分组统计。共生成 {len(observations)} 条 OBS。"
    )

    return {
        "observation": obs_text,
        "derived_series": [],
        "observations": observations,
        "validations": [],
        "figures": [],
        "metrics": {
            "experiment_count": len(records),
            "observation_count": len(observations),
            "overall_diff0_mean": overall_stats["diff0_mean"],
            "overall_diff0_std": overall_stats["diff0_std"],
            "overall_diff2_mean": overall_stats["diff2_mean"],
            "overall_diff2_std": overall_stats["diff2_std"],
            "overall_intercept_diff_mean": overall_stats["intercept_diff_mean"],
            "overall_intercept_diff_std": overall_stats["intercept_diff_std"]
        }
    }
