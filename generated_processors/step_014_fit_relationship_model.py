import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def process(payload: dict) -> dict:
    params = payload['parameters']
    experiment_ids = params['experiment_ids']
    target_series = params['target_series']
    basis_expressions = params['basis_expressions']
    include_intercept = params['include_intercept']
    prediction_name = params['prediction_name']
    residual_name = params['residual_name']
    output_dir = payload['output_dir']

    def eval_basis(expr, series):
        """Evaluate a basis expression from series."""
        if expr.startswith('square('):
            inner = expr[7:-1]
            v = np.array(series[inner])
            return v ** 2
        elif expr.startswith('cube('):
            inner = expr[5:-1]
            v = np.array(series[inner])
            return v ** 3
        elif expr.startswith('sqrt('):
            inner = expr[5:-1]
            v = np.array(series[inner])
            return np.sqrt(v)
        elif expr.startswith('log('):
            inner = expr[4:-1]
            v = np.array(series[inner])
            return np.log(v)
        elif expr.startswith('exp('):
            inner = expr[4:-1]
            v = np.array(series[inner])
            return np.exp(v)
        elif expr.startswith('sin('):
            inner = expr[4:-1]
            v = np.array(series[inner])
            return np.sin(v)
        elif expr.startswith('cos('):
            inner = expr[4:-1]
            v = np.array(series[inner])
            return np.cos(v)
        elif expr.startswith('abs('):
            inner = expr[4:-1]
            v = np.array(series[inner])
            return np.abs(v)
        else:
            # direct series name
            return np.array(series[expr])

    derived_series = []
    figures = []
    metrics = {}
    observation_parts = []

    for eid in experiment_ids:
        exp = payload['experiments'].get(eid)
        if exp is None:
            raise ValueError(f"Experiment {eid} not found in payload")
        series = exp['series']
        available = exp.get('available_series', list(series.keys()))
        for name in [target_series] + basis_expressions:
            # Extract base variable name for square/cube etc.
            base = name
            if name.startswith('square('):
                base = name[7:-1]
            elif name.startswith('cube('):
                base = name[5:-1]
            elif name.startswith('sqrt('):
                base = name[5:-1]
            elif name.startswith('log('):
                base = name[4:-1]
            elif name.startswith('exp('):
                base = name[4:-1]
            elif name.startswith('sin('):
                base = name[4:-1]
            elif name.startswith('cos('):
                base = name[4:-1]
            elif name.startswith('abs('):
                base = name[4:-1]
            if base not in series:
                raise ValueError(f"Series '{base}' (required for '{name}') not available in experiment {eid}. Available: {available}")
        t = np.array(series['t'])
        y = np.array(series[target_series])
        # Build design matrix
        X_terms = [eval_basis(expr, series) for expr in basis_expressions]
        if include_intercept:
            X = np.column_stack([np.ones(len(t))] + X_terms)
        else:
            X = np.column_stack(X_terms)

        # Fit using ordinary least squares (np.linalg.lstsq)
        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ coeffs
        resid = y - y_pred

        # Statistics
        n = len(y)
        p = X.shape[1]
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
        rmse = np.sqrt(np.mean(resid ** 2))
        mae = np.mean(np.abs(resid))

        # Store derived series
        derived_series.append({
            "experiment_id": eid,
            "name": prediction_name,
            "values": y_pred.tolist(),
            "source_name": f"Fit: {target_series} ~ {' + '.join(basis_expressions)} (intercept={include_intercept})",
            "provenance": "generated data processor: fit_relationship_model",
            "description": f"Predicted {target_series} from linear combination of {basis_expressions}"
        })
        derived_series.append({
            "experiment_id": eid,
            "name": residual_name,
            "values": resid.tolist(),
            "source_name": f"Residual: {target_series} - {prediction_name}",
            "provenance": "generated data processor: fit_relationship_model",
            "description": f"Residual of {target_series} regression"
        })

        # Build equation string for observation
        if include_intercept:
            intercept_val = coeffs[0]
            var_coeffs = coeffs[1:]
            eq_parts = [f"{intercept_val:.4f}"]
            for i, expr in enumerate(basis_expressions):
                eq_parts.append(f"{var_coeffs[i]:.4f} * {expr}")
            eq_str = f"{target_series} = " + " + ".join(eq_parts)
        else:
            var_coeffs = coeffs
            eq_parts = [f"{var_coeffs[i]:.4f} * {expr}" for i, expr in enumerate(basis_expressions)]
            eq_str = f"{target_series} = " + " + ".join(eq_parts)

        obs = (
            f"实验 {eid}: 拟合模型 {eq_str}，"
            f"R²={r2:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}。"
        )
        observation_parts.append(obs)

        # Save metrics
        prefix = eid
        if include_intercept:
            metrics[f"{prefix}_intercept"] = intercept_val
        for i, expr in enumerate(basis_expressions):
            # Clean expression for key (replace parentheses, spaces)
            key_expr = expr.replace('(', '_').replace(')', '').replace(' ', '')
            metrics[f"{prefix}_coef_{key_expr}"] = float(var_coeffs[i])
        metrics[f"{prefix}_R2"] = r2
        metrics[f"{prefix}_RMSE"] = rmse
        metrics[f"{prefix}_MAE"] = mae

        # Plot: scatter of target vs first basis variable (v_sg) with fitted curve
        # Use sorted v_sg values to draw smooth curve
        v_vals = np.array(series['v_sg'])
        sort_idx = np.argsort(v_vals)
        v_sorted = v_vals[sort_idx]
        # Recompute predictions for sorted X
        X_sorted = np.column_stack([np.ones(len(v_sorted))] + [eval_basis(expr, series)[sort_idx] for expr in basis_expressions]) if include_intercept else np.column_stack([eval_basis(expr, series)[sort_idx] for expr in basis_expressions])
        y_pred_sorted = X_sorted @ coeffs

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(v_vals, y, s=10, alpha=0.6, label='Actual')
        ax.plot(v_sorted, y_pred_sorted, 'r-', label='Fitted model')
        ax.set_xlabel('v_sg')
        ax.set_ylabel(target_series)
        ax.set_title(f'{eid}: {target_series} vs v_sg\nFitted: {eq_str}')
        ax.legend()
        fig_path = os.path.join(output_dir, f"fit_{prediction_name}_{eid}.png")
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        figures.append(fig_path)

    observation = "；".join(observation_parts)
    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
