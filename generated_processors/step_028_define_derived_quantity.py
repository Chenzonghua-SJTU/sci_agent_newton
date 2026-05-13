import numpy as np
import json
import os
from typing import Any, Dict, List, Union

def safe_eval_expression(expr: str, local_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    安全计算字符串表达式，支持四则运算、幂、函数调用。
    允许使用序列名和常量如 F_ext, pi, e 等。
    """
    # 内置函数映射
    allowed_funcs = {
        "abs": np.abs,
        "sqrt": np.sqrt,
        "exp": np.exp,
        "log": np.log,
        "sin": np.sin,
        "cos": np.cos,
        "square": lambda x: x**2,
        "cube": lambda x: x**3,
    }
    # 允许的常量
    allowed_consts = {
        "pi": np.pi,
        "e": np.e,
    }
    # 构建安全命名空间
    safe_namespace = {}
    safe_namespace.update(allowed_funcs)
    safe_namespace.update(allowed_consts)
    safe_namespace.update(local_dict)
    try:
        result = eval(expr, {"__builtins__": {}}, safe_namespace)
        if not isinstance(result, np.ndarray):
            raise ValueError(f"表达式 '{expr}' 计算结果不是数组，得到 {type(result)}")
        return result
    except Exception as e:
        raise ValueError(f"无法计算表达式 '{expr}': {e}")

def _apply_expression_to_experiment(
    experiment: Dict[str, Any],
    expression: str,
    symbol: str,
    description: str
) -> Dict[str, Any]:
    """
    对单个实验计算表达式并返回派生序列字典。
    """
    series = experiment["series"]
    available_series = experiment.get("available_series", list(series.keys()))
    # 检查表达式需要的所有序列都存在
    # 先将所有序列转为 numpy 数组
    local_dict = {name: np.array(series[name]) for name in series if isinstance(series[name], (list, np.ndarray))}
    # 添加常量
    config = experiment["config"]
    local_dict["F_ext"] = config.get("constant_force", config.get("F_ext", 0.0))
    local_dict["q0"] = config.get("initial_q", 0.0)
    local_dict["v0"] = config.get("initial_v", 0.0)
    local_dict["dt"] = config.get("dt", 0.01)
    # 计算
    values = safe_eval_expression(expression, local_dict)
    # 确保长度与 t 一致
    if "t" in series:
        expected_len = len(series["t"])
        if len(values) != expected_len:
            raise ValueError(
                f"计算结果长度 {len(values)} 与实验 t 序列长度 {expected_len} 不匹配"
            )
    return {
        "name": symbol,
        "values": values.tolist(),
        "source_name": f"表达式: {expression}",
        "provenance": "generated data processor: define_derived_quantity",
        "description": description
    }

def process(payload: dict) -> dict:
    action = payload.get("action", "")
    params = payload.get("parameters", {})
    experiments = payload.get("experiments", {})
    output_dir = payload.get("output_dir", ".")

    # 解析参数
    experiment_id = params.get("experiment_id")
    symbol = params.get("symbol")
    expression = params.get("expression")
    overwrite = params.get("overwrite", False)
    description = params.get("description", "")

    if not experiment_id or not symbol or not expression:
        raise ValueError("参数不完整: 需要 experiment_id, symbol, expression")

    if experiment_id not in experiments:
        raise ValueError(f"实验 {experiment_id} 不存在于 payload 中")

    experiment = experiments[experiment_id]
    derived_series_list = []

    # 如果表达式是纯序列名（在 available_series 中），直接复制，否则计算
    series = experiment["series"]
    available_series = experiment.get("available_series", list(series.keys()))
    if expression in series:
        # 直接复制序列
        values = series[expression]
        if not isinstance(values, list):
            values = list(values)
        # 验证长度
        if "t" in series and len(values) != len(series["t"]):
            raise ValueError(f"序列 {expression} 长度 {len(values)} 与 t 序列长度 {len(series['t'])} 不匹配")
        derived_series_list.append({
            "name": symbol,
            "values": values,
            "source_name": f"直接复制自 {expression}",
            "provenance": "generated data processor: define_derived_quantity",
            "description": description
        })
    else:
        # 使用通用表达式计算
        derived = _apply_expression_to_experiment(experiment, expression, symbol, description)
        derived_series_list.append(derived)

    # 构造统计信息
    new_values = np.array(derived_series_list[0]["values"])
    metrics = {
        "new_series_mean": float(np.mean(new_values)),
        "new_series_std": float(np.std(new_values)),
        "new_series_min": float(np.min(new_values)),
        "new_series_max": float(np.max(new_values)),
        "overwrite": overwrite
    }

    observation = (
        f"为实验 {experiment_id} 定义派生量 {symbol}，"
        f"表达式为 '{expression}'。"
        f"新序列长度 {len(new_values)}，"
        f"均值={metrics['new_series_mean']:.4f}，"
        f"标准差={metrics['new_series_std']:.4f}，"
        f"范围=[{metrics['new_series_min']:.4f}, {metrics['new_series_max']:.4f}]。"
        f"overwrite={overwrite}。"
    )
    if description:
        observation += f" 描述: {description}"

    return {
        "observation": observation,
        "derived_series": derived_series_list,
        "figures": [],
        "metrics": metrics
    }
