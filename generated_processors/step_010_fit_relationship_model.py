import numpy as np
import sklearn.linear_model
import sklearn.metrics
import os
import json
from typing import Dict, List, Any

def process(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    experiment_ids = params.get("experiment_ids", [])
    target_series = params.get("target_series")
    basis_expressions = params.get("basis_expressions", [])
    include_intercept = params.get("include_intercept", True)
    prediction_name = params.get("prediction_name", "prediction")
    residual_name = params.get("residual_name", "residual")

    if not experiment_ids:
        raise ValueError("experiment_ids list is empty, cannot fit relationship model")
    if not target_series:
        raise ValueError("target_series not specified")
    if not basis_expressions:
        raise ValueError("basis_expressions list is empty, need at least one predictor")

    observation_lines = []
    all_metrics = {}
    derived_series_list = []
    figures = []

    for exp_id in experiment_ids:
        exp_data = experiments.get(exp_id)
        if exp_data is None:
            raise ValueError(f"Experiment {exp_id} not found in payload")

        series = exp_data.get("series", {})
        available = exp_data.get("available_series", [])

        # 检查目标序列
        if target_series not in series:
            raise ValueError(f"Target series '{target_series}' not available in experiment {exp_id}")
        y = np.array(series[target_series], dtype=float)
        n = len(y)

        # 构建设计矩阵
        X_list = []
        feature_names = []
        for expr in basis_expressions:
            if expr not in series:
                raise ValueError(f"Basis expression '{expr}' not available in experiment {exp_id}")
            x_val = np.array(series[expr], dtype=float)
            if len(x_val) != n:
                raise ValueError(f"Length mismatch: target {n} vs predictor {expr} {len(x_val)}")
            X_list.append(x_val)
            feature_names.append(expr)

        X = np.column_stack(X_list)
        if include_intercept:
            X = np.column_stack((np.ones(n), X))
            # 特征名：intercept + 原始特征
            feats = ["intercept"] + feature_names
        else:
            feats = feature_names[:]

        # 线性回归
        model = sklearn.linear_model.LinearRegression(fit_intercept=False)  # 我们已经手动加截距列
        model.fit(X, y)
        coefs = model.coef_
        y_pred = model.predict(X)
        residuals = y - y_pred

        # 计算指标
        r2 = sklearn.metrics.r2_score(y, y_pred)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y, y_pred))
        mae = sklearn.metrics.mean_absolute_error(y, y_pred)
        n_points = n
        n_features = len(feature_names)
        adj_r2 = 1 - (1 - r2) * (n_points - 1) / (n_points - n_features - 1) if (n_points - n_features - 1) > 0 else float('nan')

        # 记录 metrics
        metrics_dict = {
            "r2": r2,
            "adj_r2": adj_r2,
            "rmse": rmse,
            "mae": mae,
            "n_points": n_points,
            "n_features": n_features,
            "include_intercept": include_intercept,
        }
        # 截距和系数
        if include_intercept:
            metrics_dict["intercept"] = coefs[0]
            for i, fname in enumerate(feature_names):
                metrics_dict[f"coeff_{fname}"] = coefs[i+1]
        else:
            metrics_dict["intercept"] = 0.0
            for i, fname in enumerate(feature_names):
                metrics_dict[f"coeff_{fname}"] = coefs[i]

        # 添加实验前缀避免冲突
        for k, v in metrics_dict.items():
            all_metrics[f"{exp_id}_{k}"] = v

        # 生成派生序列
        pred_values = y_pred.tolist()
        resid_values = residuals.tolist()
        derived_series_list.append({
            "experiment_id": exp_id,
            "name": prediction_name,
            "values": pred_values,
            "source_name": f"Linear model: {target_series} ~ {' + '.join(basis_expressions)} {'+ intercept' if include_intercept else ''}",
            "provenance": "generated data processor: fit_relationship_model"
        })
        derived_series_list.append({
            "experiment_id": exp_id,
            "name": residual_name,
            "values": resid_values,
            "source_name": f"{target_series} - prediction from linear model",
            "provenance": "generated data processor: fit_relationship_model"
        })

        # 构造观察文本
        coef_str = []
        if include_intercept:
            coef_str.append(f"intercept={metrics_dict['intercept']:.6f}")
        for fname in feature_names:
            coef_str.append(f"{fname}={metrics_dict[f'coeff_{fname}']:.6f}")
        obs = (f"实验 {exp_id}: 拟合 {target_series} = {' + '.join(coef_str)}  "
               f"R²={r2:.6f}, 调整R²={adj_r2:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}")
        observation_lines.append(obs)

    observation = "执行线性回归拟合：\n" + "\n".join(observation_lines)

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": figures,
        "metrics": all_metrics
    }
