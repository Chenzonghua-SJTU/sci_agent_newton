import numpy as np
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

def process(payload: dict) -> dict:
    # 提取参数
    action = payload.get("action", "")
    if action != "fit_relationship_model":
        raise ValueError(f"Unsupported action: {action}")
    params = payload.get("parameters", {})
    experiment_id = params.get("experiment_id")
    target_series = params.get("target_series", "")
    basis_expressions = params.get("basis_expressions", [])
    prediction_name = params.get("prediction_name", "")
    residual_name = params.get("residual_name", "")
    include_intercept = params.get("include_intercept", True)

    if not experiment_id:
        raise ValueError("Missing experiment_id")
    if not target_series:
        raise ValueError("Missing target_series")
    if not basis_expressions:
        raise ValueError("Missing basis_expressions")
    if not prediction_name or not residual_name:
        raise ValueError("prediction_name and residual_name must be provided")

    experiments = payload.get("experiments", {})
    if experiment_id not in experiments:
        raise ValueError(f"Experiment {experiment_id} not found in payload")

    exp = experiments[experiment_id]
    series = exp.get("series", {})

    # 检查目标序列和基序列是否存在
    required_series = [target_series] + basis_expressions
    for s in required_series:
        if s not in series:
            raise ValueError(f"Series '{s}' not available in experiment {experiment_id}. Available series: {list(series.keys())}")

    t = series.get("t")
    y = np.array(series[target_series])
    # 构建特征矩阵 X
    X_list = [np.array(series[basis]) for basis in basis_expressions]
    X = np.column_stack(X_list)

    # 拟合
    model = LinearRegression(fit_intercept=include_intercept)
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred

    # 计算指标
    n = len(y)
    p = X.shape[1] + (1 if include_intercept else 0)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    # 调整R2
    if n > p:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p)
    else:
        adj_r2 = r2

    # 构建系数信息
    coeff_info = {}
    if include_intercept:
        coeff_info["intercept"] = model.intercept_
    for i, basis in enumerate(basis_expressions):
        coeff_info[f"coeff_{basis}"] = model.coef_[i]

    # 构造派生序列
    derived_series = [
        {
            "experiment_id": experiment_id,
            "name": prediction_name,
            "values": y_pred.tolist(),
            "source_name": f"Linear model: {target_series} ~ {' + '.join(basis_expressions)}" + (" + intercept" if include_intercept else ""),
            "provenance": "generated data processor: fit_relationship_model",
            "description": f"Predicted {target_series} from linear regression using {basis_expressions}"
        },
        {
            "experiment_id": experiment_id,
            "name": residual_name,
            "values": residuals.tolist(),
            "source_name": f"{target_series} - prediction from linear model",
            "provenance": "generated data processor: fit_relationship_model",
            "description": f"Residuals of {target_series} linear regression using {basis_expressions}"
        }
    ]

    # 构造 metrics
    metrics = {
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "n_points": int(n),
        "n_features": int(X.shape[1]),
        "include_intercept": bool(include_intercept)
    }
    # 添加系数
    if include_intercept:
        metrics["intercept"] = float(model.intercept_)
    for i, basis in enumerate(basis_expressions):
        metrics[f"coeff_{basis}"] = float(model.coef_[i])

    # 生成系数描述的字符串
    coeff_str_parts = []
    if include_intercept:
        coeff_str_parts.append(f"intercept={model.intercept_:.6g}")
    for i, basis in enumerate(basis_expressions):
        coeff_str_parts.append(f"{basis}={model.coef_[i]:.6g}")
    coeff_str = " + ".join(coeff_str_parts)

    observation = (
        f"对实验 {experiment_id} 执行线性回归：{target_series} ~ {' + '.join(basis_expressions)}"
        + (" + intercept" if include_intercept else "")
        + f"。\n"
        f"拟合系数：{coeff_str}\n"
        f"R²={r2:.6g}, 调整R²={adj_r2:.6g}, RMSE={rmse:.6g}, MAE={mae:.6g}\n"
        f"生成了预测序列 '{prediction_name}' 和残差序列 '{residual_name}'。"
    )

    # 可选：保存图像（散点图+拟合线）
    output_dir = payload.get("output_dir", "/tmp")
    figures = []
    # 这里可以保存图像，但参数没有要求，我们为了简洁暂不添加，如果需要可以添加：
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.scatter(X[:,0], y, label='data', alpha=0.5)
    # ax.plot(X[:,0], y_pred, 'r-', label='fit')
    # ax.set_xlabel(basis_expressions[0])
    # ax.set_ylabel(target_series)
    # ax.set_title(f"{experiment_id}: {target_series} vs {basis_expressions[0]}")
    # ax.legend()
    # fig_path = os.path.join(output_dir, f"fit_{experiment_id}_{prediction_name}.png")
    # plt.savefig(fig_path, dpi=100)
    # plt.close()
    # figures.append(fig_path)

    return {
        "observation": observation,
        "derived_series": derived_series,
        "figures": figures,
        "metrics": metrics
    }
