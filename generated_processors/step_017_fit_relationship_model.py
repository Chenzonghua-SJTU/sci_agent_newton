import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    experiment_ids = params.get("experiment_ids", None)
    if experiment_ids is None:
        experiment_ids = list(experiments.keys())

    target_series = params.get("target_series")
    basis_expressions = params.get("basis_expressions", [])
    include_intercept = params.get("include_intercept", True)
    prediction_name = params.get("prediction_name", None)
    residual_name = params.get("residual_name", None)

    if not target_series:
        raise ValueError("target_series must be specified")
    if not basis_expressions:
        raise ValueError("basis_expressions must be a non‑empty list")

    # Validate that all experiments exist
    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")

    metrics = {}
    figures = []
    derived_series = []

    for eid in experiment_ids:
        exp = experiments[eid]
        series = exp.get("series", {})
        available = exp.get("available_series", [])

        # Check needed series
        if target_series not in series:
            raise ValueError(f"Experiment {eid}: target series '{target_series}' not available")
        for base in basis_expressions:
            if base not in series:
                raise ValueError(f"Experiment {eid}: basis expression '{base}' not available")

        # Extract data
        t = np.array(series["t"])
        y = np.array(series[target_series])
        X = np.column_stack([np.array(series[base]) for base in basis_expressions])

        if len(y) != len(t):
            raise ValueError(f"Experiment {eid}: target series length mismatch")
        if len(X) != len(t):
            raise ValueError(f"Experiment {eid}: basis series length mismatch")

        # Fit model
        if include_intercept:
            X_design = np.column_stack([np.ones(len(t)), X])
        else:
            X_design = X.copy()

        # Use sklearn LinearRegression for convenience
        reg = LinearRegression(fit_intercept=False)  # we already added constant
        reg.fit(X_design, y)
        coefs = reg.coef_
        intercept = 0.0
        if include_intercept:
            intercept = coefs[0]
            coefs = coefs[1:]  # remove intercept from list of basis coefs
        y_pred = reg.predict(X_design)
        residuals = y - y_pred

        # Compute metrics
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        mae = float(mean_absolute_error(y, y_pred))
        r2 = float(r2_score(y, y_pred))

        # Store per experiment
        metrics[f"{eid}_intercept"] = intercept
        for idx, base in enumerate(basis_expressions):
            metrics[f"{eid}_coef_{base}"] = float(coefs[idx])
        metrics[f"{eid}_r2"] = r2
        metrics[f"{eid}_rmse"] = rmse
        metrics[f"{eid}_mae"] = mae

        # Optionally return prediction / residual series
        if prediction_name is not None:
            derived_series.append({
                "experiment_id": eid,
                "name": prediction_name,
                "values": y_pred.tolist(),
                "source_name": f"Linear regression of {target_series} on {basis_expressions}",
                "provenance": "generated data processor: ...",
                "description": "Predicted values from fitted model"
            })
        if residual_name is not None:
            derived_series.append({
                "experiment_id": eid,
                "name": residual_name,
                "values": residuals.tolist(),
                "source_name": f"Residuals of {target_series} regression",
                "provenance": "generated data processor: ...",
                "description": "Residuals (observed - predicted)"
            })

        # Plot a vs v (if only one basis expression) or a vs predicted
        fig, ax = plt.subplots(figsize=(6, 4))
        if len(basis_expressions) == 1:
            base_vals = X[:, 0]
            ax.scatter(base_vals, y, s=10, alpha=0.6, label=f"{target_series} vs {basis_expressions[0]}")
            # plot fit line
            x_sort = np.sort(base_vals)
            # Predict on sorted x
            X_sort = np.column_stack([np.ones(len(x_sort)), x_sort]) if include_intercept else x_sort.reshape(-1,1)
            y_sort = reg.predict(X_sort)
            ax.plot(x_sort, y_sort, 'r-', label="Fit line")
            ax.set_xlabel(basis_expressions[0])
        else:
            ax.scatter(y_pred, y, s=10, alpha=0.6, label=f"{target_series} vs predicted")
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Ideal")
            ax.set_xlabel("Predicted")
        ax.set_ylabel(target_series)
        ax.set_title(f"Experiment {eid}: {target_series} = f({basis_expressions})")
        ax.legend()
        fig_path = os.path.join(output_dir, f"fit_{eid}.png")
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        figures.append(fig_path)

    # Compute aggregate metrics across experiments
    r2_list = [metrics[f"{eid}_r2"] for eid in experiment_ids]
    rmse_list = [metrics[f"{eid}_rmse"] for eid in experiment_ids]
    mae_list = [metrics[f"{eid}_mae"] for eid in experiment_ids]
    metrics["mean_r2"] = float(np.mean(r2_list))
    metrics["std_r2"] = float(np.std(r2_list))
    metrics["mean_rmse"] = float(np.mean(rmse_list))
    metrics["std_rmse"] = float(np.std(rmse_list))
    metrics["mean_mae"] = float(np.mean(mae_list))
    metrics["std_mae"] = float(np.std(mae_list))

    # Build observation
    obs_lines = []
    obs_lines.append(f"对实验 {experiment_ids} 进行了线性回归拟合：目标序列 = {target_series}，基表达式 = {basis_expressions}，包含截距 = {include_intercept}。")
    for eid in experiment_ids:
        intercept = metrics[f"{eid}_intercept"]
        coef_str = " + ".join([f"{metrics[f'{eid}_coef_{base}']:.6f}*{base}" for base in basis_expressions])
        r2 = metrics[f"{eid}_r2"]
        rmse = metrics[f"{eid}_rmse"]
        mae = metrics[f"{eid}_mae"]
        obs_lines.append(f"  {eid}: {target_series} = {intercept:.6f} + {coef_str}, R²={r2:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}")
    obs_lines.append(f"跨实验平均R²={metrics['mean_r2']:.6f}±{metrics['std_r2']:.6f}，平均RMSE={metrics['mean_rmse']:.6f}±{metrics['std_rmse']:.6f}。")
    observation = "\n".join(obs_lines)

    result = {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
    return result
