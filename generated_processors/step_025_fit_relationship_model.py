import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def process(payload: dict) -> dict:
    action = payload["action"]
    params = payload["parameters"]
    experiments = payload["experiments"]
    output_dir = payload["output_dir"]

    experiment_ids = params.get("experiment_ids", [])
    if not experiment_ids:
        experiment_ids = list(experiments.keys())

    target_series = params["target_series"]
    basis_expressions = params["basis_expressions"]
    prediction_name = params.get("prediction_name", "prediction")
    residual_name = params.get("residual_name", "residual")
    include_intercept = params.get("include_intercept", True)

    derived_series_list = []
    metrics = {}
    observations = []

    for eid in experiment_ids:
        if eid not in experiments:
            raise ValueError(f"Experiment {eid} not found in payload")
        exp = experiments[eid]
        series = exp["series"]
        available = exp["available_series"]

        # 检查 target_series
        if target_series not in series:
            raise ValueError(f"Target series '{target_series}' not available in experiment {eid}. Available: {available}")
        y = np.array(series[target_series])

        # 构建特征矩阵
        X_list = []
        for bexpr in basis_expressions:
            if bexpr in series:
                X_list.append(np.array(series[bexpr]))
            else:
                # 尝试计算简单表达式（仅支持直接序列名）
                raise ValueError(f"Basis expression '{bexpr}' not found in experiment {eid} series. Available: {available}")
        if not X_list:
            raise ValueError("No basis expressions provided")
        X = np.column_stack(X_list)

        # 拟合
        lr = LinearRegression(fit_intercept=include_intercept)
        lr.fit(X, y)
        y_pred = lr.predict(X)
        residual = y - y_pred

        # 计算指标
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        # 记录系数
        coef_names = basis_expressions.copy()
        intercept_val = lr.intercept_ if include_intercept else 0.0
        coef_values = {} if include_intercept else {}
        if include_intercept:
            metrics[f"{eid}_intercept"] = intercept_val
        for i, name in enumerate(coef_names):
            metrics[f"{eid}_coef_{name}"] = lr.coef_[i]
        metrics[f"{eid}_r2"] = r2
        metrics[f"{eid}_rmse"] = rmse
        metrics[f"{eid}_mae"] = mae

        # 添加派生序列
        derived_series_list.append({
            "experiment_id": eid,
            "name": prediction_name,
            "values": y_pred.tolist(),
            "source_name": f"LinearRegression({target_series} ~ {'+'.join(basis_expressions)})",
            "provenance": "generated data processor: fit_relationship_model",
            "description": f"Predicted values from fitted model (R²={r2:.4f})"
        })
        derived_series_list.append({
            "experiment_id": eid,
            "name": residual_name,
            "values": residual.tolist(),
            "source_name": f"{target_series} - {prediction_name}",
            "provenance": "generated data processor: fit_relationship_model",
            "description": "Residuals of the linear regression"
        })

        # 构建观察字符串
        coef_str_parts = []
        if include_intercept:
            coef_str_parts.append(f"intercept={intercept_val:.6f}")
        for i, name in enumerate(coef_names):
            coef_str_parts.append(f"{name}={lr.coef_[i]:.6f}")
        coef_str = ", ".join(coef_str_parts)
        obs = f"{eid}: {target_series} = {' + '.join([f'{lr.coef_[i]:.6f}*{coef_names[i]}' for i in range(len(coef_names))])}"
        if include_intercept:
            obs += f" + {intercept_val:.6f}"
        obs += f", R²={r2:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}"
        observations.append(obs)

    # 整体统计
    if len(experiment_ids) > 1:
        r2_list = [metrics[f"{eid}_r2"] for eid in experiment_ids]
        rmse_list = [metrics[f"{eid}_rmse"] for eid in experiment_ids]
        metrics["mean_r2"] = np.mean(r2_list)
        metrics["std_r2"] = np.std(r2_list)
        metrics["mean_rmse"] = np.mean(rmse_list)
        metrics["std_rmse"] = np.std(rmse_list)

    observation_str = "对实验 " + str(experiment_ids) + " 进行了线性回归拟合：目标序列 = " + target_series + "，基表达式 = " + str(basis_expressions) + "，包含截距 = " + str(include_intercept) + "。\n" + "\n".join(observations)
    if len(experiment_ids) > 1:
        observation_str += f"\n跨实验平均R²={metrics.get('mean_r2', 0):.6f}±{metrics.get('std_r2', 0):.6f}，平均RMSE={metrics.get('mean_rmse', 0):.6f}±{metrics.get('std_rmse', 0):.6f}。"

    return {
        "observation": observation_str,
        "derived_series": derived_series_list,
        "figures": [],
        "metrics": metrics
    }
