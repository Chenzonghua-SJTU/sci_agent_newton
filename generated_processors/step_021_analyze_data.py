import json
import math
import statistics
import itertools
import functools
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process(payload: dict) -> dict:
    # ------------------------------------------------------------------
    # extract experiment data
    # ------------------------------------------------------------------
    experiments = payload["experiments"]
    params = payload["parameters"]
    output_dir = Path(payload["output_dir"])

    # lists of experiment ids that should be processed
    exp_ids = params.get("experiment_ids") or list(experiments.keys())

    # results containers
    per_exp_table = []   # each entry: dict with exp_id, F_ext, v0, slope, intercept, R2
    constant_experiments = []  # only constant field experiments

    # ------------------------------------------------------------------
    # step 1: compute a-v linear regression for every requested experiment
    # ------------------------------------------------------------------
    for eid in exp_ids:
        exp = experiments[eid]
        config = exp["config"]
        series = exp["series"]
        avail = exp["available_series"]

        # required series
        if "a" not in avail or "v" not in avail:
            raise ValueError(f"Experiment {eid} missing required series a or v")

        t = np.array(series["t"])
        a = np.array(series["a"])
        v = np.array(series["v"])

        # regression
        # use scipy.stats.linregress, robust to degenerate cases
        if np.var(v) < 1e-12:   # v constant -> slope undefined
            slope_av = 0.0
            intercept_av = np.mean(a)
            rvalue_av = 0.0
            pvalue_av = 1.0
            stderr_av = 0.0
            r2_av = 0.0
        elif np.var(a) < 1e-12:  # a constant
            slope_av = 0.0
            intercept_av = a[0]
            rvalue_av = 0.0
            pvalue_av = 1.0
            stderr_av = 0.0
            r2_av = 0.0
        else:
            reg = stats.linregress(v, a)
            slope_av = reg.slope
            intercept_av = reg.intercept
            rvalue_av = reg.rvalue
            pvalue_av = reg.pvalue
            stderr_av = reg.stderr
            r2_av = rvalue_av ** 2

        # record
        entry = {
            "experiment_id": eid,
            "F_ext": config["F_ext"],
            "v0": config["initial_v"],
            "force_field_type": config["force_field_type"],
            "slope_av": slope_av,
            "intercept_av": intercept_av,
            "r2_av": r2_av,
            "pvalue_av": pvalue_av,
            "stderr_av": stderr_av
        }
        per_exp_table.append(entry)

        if config["force_field_type"] == "constant":
            constant_experiments.append(entry)

    # convert to DataFrame for easy filtering
    df_all = pd.DataFrame(per_exp_table)
    df_const = pd.DataFrame(constant_experiments)

    observations = []
    metrics = {}

    # ------------------------------------------------------------------
    # observation: per-experiment a-v regression table
    # ------------------------------------------------------------------
    summary_rows = []
    for row in per_exp_table:
        summary_rows.append(
            f"{row['experiment_id']}: F_ext={row['F_ext']}, v0={row['v0']}, "
            f"slope={row['slope_av']:.6f}, intercept={row['intercept_av']:.6f}, "
            f"R2={row['r2_av']:.6f}"
        )
    obs_per_exp = {
        "summary": "a-v linear regression for each experiment (all requested experiments).\n" + "\n".join(summary_rows),
        "source_data_refs": [f"{eid}:a" for eid in exp_ids] + [f"{eid}:v" for eid in exp_ids],
        "metrics": {
            "experiment_count": len(per_exp_table),
            "constant_count": len(constant_experiments),
            "free_count": len(df_all[df_all["force_field_type"] == "free"]),
            "r2_min": float(df_all["r2_av"].min()),
            "r2_max": float(df_all["r2_av"].max()),
            "slope_min": float(df_all["slope_av"].min()),
            "slope_max": float(df_all["slope_av"].max())
        }
    }
    observations.append(obs_per_exp)
    metrics["experiment_count"] = len(per_exp_table)
    metrics["constant_count"] = len(constant_experiments)

    # ------------------------------------------------------------------
    # step 2: multiple regression on intercept_av ~ F_ext + v0 + F_ext:v0
    #         (only constant field experiments)
    # ------------------------------------------------------------------
    if len(df_const) >= 4:
        X_intercept = df_const[["F_ext", "v0"]].copy()
        X_intercept["F_ext:v0"] = X_intercept["F_ext"] * X_intercept["v0"]
        y_intercept = df_const["intercept_av"]

        reg_intercept = LinearRegression().fit(X_intercept.values, y_intercept)
        y_pred_intercept = reg_intercept.predict(X_intercept.values)
        r2_intercept = r2_score(y_intercept, y_pred_intercept)
        rmse_intercept = math.sqrt(mean_squared_error(y_intercept, y_pred_intercept))

        intercept_coefs = {
            "intercept": float(reg_intercept.intercept_),
            "coef_F_ext": float(reg_intercept.coef_[0]),
            "coef_v0": float(reg_intercept.coef_[1]),
            "coef_F_ext:v0": float(reg_intercept.coef_[2])
        }
        obs_intercept = {
            "summary": (
                f"Multiple linear regression: intercept_av ~ F_ext + v0 + F_ext:v0, "
                f"on {len(df_const)} constant field experiments. "
                f"R2={r2_intercept:.6f}, RMSE={rmse_intercept:.6f}. "
                f"Coefficients: intercept={intercept_coefs['intercept']:.6f}, "
                f"F_ext={intercept_coefs['coef_F_ext']:.6f}, "
                f"v0={intercept_coefs['coef_v0']:.6f}, "
                f"F_ext:v0={intercept_coefs['coef_F_ext:v0']:.6f}"
            ),
            "source_data_refs": [f"{eid}:a" for eid in df_const["experiment_id"].tolist()],
            "metrics": {
                "r2_intercept_av_reg": float(r2_intercept),
                "rmse_intercept_av_reg": float(rmse_intercept),
                "intercept_av_reg_coef_intercept": intercept_coefs["intercept"],
                "intercept_av_reg_coef_F_ext": intercept_coefs["coef_F_ext"],
                "intercept_av_reg_coef_v0": intercept_coefs["coef_v0"],
                "intercept_av_reg_coef_F_ext:v0": intercept_coefs["coef_F_ext:v0"]
            }
        }
        observations.append(obs_intercept)
        metrics.update(obs_intercept["metrics"])
    else:
        obs_intercept = {
            "summary": f"Too few constant experiments ({len(df_const)}) for regression on intercept_av.",
            "source_data_refs": [],
            "metrics": {"task_could_not_complete": True}
        }
        observations.append(obs_intercept)

    # ------------------------------------------------------------------
    # step 3: multiple regression on slope_av ~ F_ext + v0 + F_ext:v0
    # ------------------------------------------------------------------
    if len(df_const) >= 4:
        X_slope = df_const[["F_ext", "v0"]].copy()
        X_slope["F_ext:v0"] = X_slope["F_ext"] * X_slope["v0"]
        y_slope = df_const["slope_av"]

        reg_slope = LinearRegression().fit(X_slope.values, y_slope)
        y_pred_slope = reg_slope.predict(X_slope.values)
        r2_slope = r2_score(y_slope, y_pred_slope)
        rmse_slope = math.sqrt(mean_squared_error(y_slope, y_pred_slope))

        slope_coefs = {
            "intercept": float(reg_slope.intercept_),
            "coef_F_ext": float(reg_slope.coef_[0]),
            "coef_v0": float(reg_slope.coef_[1]),
            "coef_F_ext:v0": float(reg_slope.coef_[2])
        }
        obs_slope = {
            "summary": (
                f"Multiple linear regression: slope_av ~ F_ext + v0 + F_ext:v0, "
                f"on {len(df_const)} constant field experiments. "
                f"R2={r2_slope:.6f}, RMSE={rmse_slope:.6f}. "
                f"Coefficients: intercept={slope_coefs['intercept']:.6f}, "
                f"F_ext={slope_coefs['coef_F_ext']:.6f}, "
                f"v0={slope_coefs['coef_v0']:.6f}, "
                f"F_ext:v0={slope_coefs['coef_F_ext:v0']:.6f}"
            ),
            "source_data_refs": [f"{eid}:a" for eid in df_const["experiment_id"].tolist()],
            "metrics": {
                "r2_slope_av_reg": float(r2_slope),
                "rmse_slope_av_reg": float(rmse_slope),
                "slope_av_reg_coef_intercept": slope_coefs["intercept"],
                "slope_av_reg_coef_F_ext": slope_coefs["coef_F_ext"],
                "slope_av_reg_coef_v0": slope_coefs["coef_v0"],
                "slope_av_reg_coef_F_ext:v0": slope_coefs["coef_F_ext:v0"]
            }
        }
        observations.append(obs_slope)
        metrics.update(obs_slope["metrics"])
    else:
        obs_slope = {
            "summary": f"Too few constant experiments ({len(df_const)}) for regression on slope_av.",
            "source_data_refs": [],
            "metrics": {"task_could_not_complete": True}
        }
        observations.append(obs_slope)

    # ------------------------------------------------------------------
    # step 4: grouped linear regression: slope_av ~ F_ext for v0 = 0, 1, 2 subsets
    # ------------------------------------------------------------------
    v0_values = sorted(df_const["v0"].unique())
    group_regressions = {}
    for v0_val in [0.0, 1.0, 2.0]:
        subset = df_const[df_const["v0"] == v0_val]
        n = len(subset)
        if n >= 3:
            F_vals = subset["F_ext"].values
            slope_vals = subset["slope_av"].values
            # check variance in F_ext
            if np.var(F_vals) < 1e-12:
                group_r2 = 0.0
                group_slope = 0.0
                group_intercept = float(np.mean(slope_vals))
                group_rmse = math.sqrt(np.var(slope_vals))
                group_slope = 0.0
                group_intercept = float(np.mean(slope_vals))
                group_r2 = 0.0
                group_rmse = float(np.std(slope_vals, ddof=0))
            else:
                reg_g = stats.linregress(F_vals, slope_vals)
                group_slope = reg_g.slope
                group_intercept = reg_g.intercept
                group_r2 = reg_g.rvalue ** 2
                residuals = slope_vals - (group_slope * F_vals + group_intercept)
                group_rmse = math.sqrt(np.mean(residuals ** 2))
            group_regressions[f"v0={v0_val}"] = {
                "n_experiments": n,
                "slope": float(group_slope),
                "intercept": float(group_intercept),
                "r2": float(group_r2),
                "rmse": float(group_rmse)
            }
        else:
            group_regressions[f"v0={v0_val}"] = {
                "n_experiments": n,
                "message": "insufficient data points (need >=3)"
            }

    # build observation for group regressions
    group_lines = []
    for k, v in group_regressions.items():
        if v.get("message"):
            group_lines.append(f"{k}: {v['message']} (n={v['n_experiments']})")
        else:
            group_lines.append(
                f"{k}: n={v['n_experiments']}, "
                f"slope={v['slope']:.6f}, intercept={v['intercept']:.6f}, "
                f"R2={v['r2']:.6f}, RMSE={v['rmse']:.6f}"
            )
    obs_group = {
        "summary": "Group-wise linear regression: slope_av ~ F_ext for each v0 subset (constant field experiments).\n" + "\n".join(group_lines),
        "source_data_refs": [f"{eid}:a" for eid in df_const["experiment_id"].tolist()],
        "metrics": {
            "group_regression_v0_0_r2": group_regressions.get("v0=0.0", {}).get("r2", None),
            "group_regression_v0_0_rmse": group_regressions.get("v0=0.0", {}).get("rmse", None),
            "group_regression_v0_1_r2": group_regressions.get("v0=1.0", {}).get("r2", None),
            "group_regression_v0_1_rmse": group_regressions.get("v0=1.0", {}).get("rmse", None),
            "group_regression_v0_2_r2": group_regressions.get("v0=2.0", {}).get("r2", None),
            "group_regression_v0_2_rmse": group_regressions.get("v0=2.0", {}).get("rmse", None)
        }
    }
    observations.append(obs_group)
    metrics["group_regression_available"] = True

    # ------------------------------------------------------------------
    # optional: generate figures for cross-experiment regressions
    # ------------------------------------------------------------------
    figures = []
    if len(df_const) >= 4:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # intercept_av scatter vs F_ext, colored by v0
        v0_vals = df_const["v0"].unique()
        for v0_val in v0_vals:
            mask = df_const["v0"] == v0_val
            axs[0].scatter(df_const.loc[mask, "F_ext"], df_const.loc[mask, "intercept_av"],
                           label=f"v0={v0_val}", alpha=0.7, edgecolors="k")
        axs[0].set_xlabel("F_ext")
        axs[0].set_ylabel("intercept_av")
        axs[0].set_title("intercept_av vs F_ext (color: v0)")
        axs[0].legend()

        # slope_av scatter vs F_ext, colored by v0
        for v0_val in v0_vals:
            mask = df_const["v0"] == v0_val
            axs[1].scatter(df_const.loc[mask, "F_ext"], df_const.loc[mask, "slope_av"],
                           label=f"v0={v0_val}", alpha=0.7, edgecolors="k")
        axs[1].set_xlabel("F_ext")
        axs[1].set_ylabel("slope_av")
        axs[1].set_title("slope_av vs F_ext (color: v0)")
        axs[1].legend()

        fig.tight_layout()
        fig_path = output_dir / "av_regression_scatter.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        figures.append(str(fig_path))

        # optional: regression lines for groups
        for v0_val in [0.0, 1.0, 2.0]:
            subset = df_const[df_const["v0"] == v0_val]
            if len(subset) >= 3 and np.var(subset["F_ext"]) > 1e-12:
                fig2, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(subset["F_ext"], subset["slope_av"], color="blue", edgecolors="k")
                # fit
                reg_g = stats.linregress(subset["F_ext"].values, subset["slope_av"].values)
                F_fit = np.linspace(subset["F_ext"].min(), subset["F_ext"].max(), 100)
                slope_fit = reg_g.slope * F_fit + reg_g.intercept
                ax.plot(F_fit, slope_fit, "r-", label=f"slope={reg_g.slope:.4f}, R²={reg_g.rvalue**2:.4f}")
                ax.set_xlabel("F_ext")
                ax.set_ylabel("slope_av")
                ax.set_title(f"v0={v0_val}: slope_av ~ F_ext")
                ax.legend()
                fig2.tight_layout()
                fp2 = output_dir / f"slope_vs_F_v0_{v0_val}.png"
                fig2.savefig(fp2, dpi=150)
                plt.close(fig2)
                figures.append(str(fp2))

    # ------------------------------------------------------------------
    # final metrics
    # ------------------------------------------------------------------
    metrics["observation_count"] = len(observations)
    metrics["figure_count"] = len(figures)

    return {
        "observation": f"Processed {len(per_exp_table)} experiments (constant: {len(constant_experiments)}, free: {len(df_all[df_all['force_field_type']=='free'])}). "
                       f"Per-experiment a-v regression parameters collected. "
                       f"Multiple regressions for intercept_av (R2={obs_intercept['metrics'].get('r2_intercept_av_reg', 'N/A')}) "
                       f"and slope_av (R2={obs_slope['metrics'].get('r2_slope_av_reg', 'N/A')}) performed. "
                       f"Group regressions for v0=0,1,2 completed. See observations for details.",
        "derived_series": [],   # no new series needed
        "observations": observations,
        "validations": [],      # maintain_ledger mode, no validation
        "figures": figures,
        "metrics": metrics
    }
