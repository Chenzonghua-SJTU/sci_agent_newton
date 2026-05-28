import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def process(payload: dict) -> dict:
    # Validate required keys
    if "parameters" not in payload or "analysis_mode" not in payload.get("parameters", {}):
        raise ValueError("Missing required parameters['analysis_mode']")
    if payload["parameters"]["analysis_mode"] != "maintain_ledger":
        raise ValueError("This script only supports analysis_mode='maintain_ledger'")
    
    output_dir = Path(payload["output_dir"])
    experiments = payload.get("experiments", {})
    if not experiments:
        raise ValueError("No experiments in payload")
    
    param_ids = payload["parameters"].get("experiment_ids")
    if param_ids:
        experiment_ids = list(param_ids)
    else:
        experiment_ids = list(experiments.keys())
    
    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
    
    observations = []
    figures = []
    summary_data = []  # For cross‑experiment comparison
    
    # --- Per‑experiment quadratic fit ---
    for eid in experiment_ids:
        exp = experiments[eid]
        t = np.array(exp["series"]["t"])
        q = np.array(exp["series"]["q"])
        
        if len(t) < 3:
            raise ValueError(f"Experiment {eid} has only {len(t)} data points, need ≥3 for quadratic fit")
        
        # Polyfit: highest degree first → coeffs[0]=c2, coeffs[1]=c1, coeffs[2]=c0
        coeffs = np.polyfit(t, q, 2)
        c2, c1, c0 = coeffs
        
        # Predicted values and error metrics
        q_pred = np.polyval(coeffs, t)
        q_mean = np.mean(q)
        SStot = np.sum((q - q_mean) ** 2)
        SSres = np.sum((q - q_pred) ** 2)
        if SStot < 1e-12:
            r2 = 1.0  # Perfect fit for constant data
        else:
            r2 = 1.0 - SSres / SStot
        rmse = np.sqrt(mean_squared_error(q, q_pred))
        
        # Exact acceleration and initial velocity from quadratic form
        a_exact = 2.0 * c2       # d^2 q / dt^2 = 2*c2
        v_exact = c1             # dq/dt at t=0 = c1
        
        # Experiment config
        config = exp["config"]
        F_ext = config["F_ext"]
        v0 = config["initial_v"]
        force_type = config.get("force_field_type", "unknown")
        
        summary_data.append({
            "exp_id": eid,
            "F_ext": F_ext,
            "v0": v0,
            "force_type": force_type,
            "c0": c0,
            "c1": c1,
            "c2": c2,
            "a_exact": a_exact,
            "v_exact": v_exact,
            "r2": r2,
            "rmse": rmse
        })
        
        obs_entry = {
            "summary": f"Exp {eid}: quadratic fit q(t) = {c0:.8f} + {c1:.8f}·t + {c2:.8f}·t², R²={r2:.6f}, RMSE={rmse:.2e}, a_exact={a_exact:.6f}, v_exact={v_exact:.6f}",
            "source_data_refs": [f"{eid}:t", f"{eid}:q"],
            "metrics": {
                "c0": float(c0),
                "c1": float(c1),
                "c2": float(c2),
                "R2": float(r2),
                "RMSE": float(rmse),
                "a_exact": float(a_exact),
                "v_exact": float(v_exact),
                "F_ext": float(F_ext),
                "v0": float(v0),
                "force_type": force_type
            }
        }
        observations.append(obs_entry)
    
    # --- Figure: 2×3 subplots ---
    cols = 3
    rows = (len(experiment_ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes_flat = axes.flatten() if rows*cols > 1 else [axes]
    
    for idx, eid in enumerate(experiment_ids):
        ax = axes_flat[idx]
        exp = experiments[eid]
        t = np.array(exp["series"]["t"])
        q = np.array(exp["series"]["q"])
        coeffs = np.polyfit(t, q, 2)
        t_fit = np.linspace(t.min(), t.max(), 200)
        q_fit = np.polyval(coeffs, t_fit)
        ax.scatter(t, q, s=20, label='Raw data')
        ax.plot(t_fit, q_fit, 'r-', linewidth=1.5, label='Quadratic fit')
        ax.set_title(eid)
        ax.set_xlabel('t')
        ax.set_ylabel('q')
        ax.legend(frameon=False)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Hide any unused subplots
    for idx in range(len(experiment_ids), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    fig_path = output_dir / "quadratic_fit_all.png"
    fig.savefig(str(fig_path), dpi=150)
    plt.close(fig)
    figures.append(str(fig_path))
    
    # --- Cross‑experiment comparison observation ---
    comp_lines = []
    for d in summary_data:
        comp_lines.append(f"{d['exp_id']}: F_ext={d['F_ext']:+g}, v0={d['v0']:+g}, a_exact={d['a_exact']:.6f}")
    comp_summary = "Comparison of exact acceleration a_exact with applied force F_ext and initial velocity v0:\n" + "\n".join(comp_lines)
    
    # Additional numeric facts for constant‑force experiments
    const_facts = {}
    for d in summary_data:
        if d['force_type'] == 'constant':
            key = f"F_ext={d['F_ext']:+g}"
            const_facts.setdefault(key, []).append((d['exp_id'], d['v0'], d['a_exact']))
    if const_facts:
        comp_summary += "\n\nFor constant‑force groups:"
        for key, entries in const_facts.items():
            entries_str = ", ".join([f"{eid}(v0={v:+g}, a_exact={a:.6f})" for eid, v, a in entries])
            comp_summary += f"\n  {key}: {entries_str}"
    
    comp_obs = {
        "summary": comp_summary,
        "source_data_refs": [f"{eid}:q" for eid in experiment_ids],
        "metrics": {
            "experiments_compared": len(summary_data),
            "F_ext_values": [d["F_ext"] for d in summary_data],
            "v0_values": [d["v0"] for d in summary_data],
            "a_exact_values": [d["a_exact"] for d in summary_data],
            "force_types": [d["force_type"] for d in summary_data]
        }
    }
    observations.append(comp_obs)
    
    # --- Build return dict ---
    result = {
        "observation": (
            f"Processed {len(experiment_ids)} experiments: performed quadratic fit q(t)=c0+c1·t+c2·t², "
            f"reported coefficients, R², RMSE, a_exact=2·c2, v_exact=c1. "
            f"Generated {len(observations)-1} per‑experiment OBS entries plus one cross‑experiment comparison OBS. "
            f"Saved comparison figure to {fig_path}."
        ),
        "derived_series": [],
        "observations": observations,
        "validations": [],
        "figures": figures,
        "metrics": {
            "experiments_processed": len(experiment_ids),
            "observation_count": len(observations)
        }
    }
    return result
