from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version

PySRRegressor: Any | None = None
_PYSR_IMPORT_ERROR: Exception | None = None


@dataclass(slots=True)
class VerificationResult:
    """符号验证结果。"""

    equation: str
    loss: float
    complexity: float | None
    model_selection: str
    raw_equations: pd.DataFrame | None = None


@dataclass(slots=True)
class InvariantSearchResult:
    """单实验内候选不变量搜索结果。"""

    equation: str
    loss: float
    complexity: float | None
    residual_std: float
    predicted_mean: float
    score: float
    raw_equations: pd.DataFrame | None = None


class VerificationEngine:
    """对 PySR 的面向对象封装。"""

    def __init__(
        self,
        niterations: int = 80,
        population_size: int = 33,
        maxsize: int = 20,
        binary_operators: list[str] | None = None,
        unary_operators: list[str] | None = None,
        model_selection: str = "best",
        random_state: int = 0,
        extra_pysr_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.niterations = niterations
        self.population_size = population_size
        self.maxsize = maxsize
        self.binary_operators = binary_operators or ["+", "-", "*", "/"]
        self.unary_operators = unary_operators or ["square", "cube"]
        self.model_selection = model_selection
        self.random_state = random_state
        self.extra_pysr_kwargs = extra_pysr_kwargs or {}
        self._valid_binary_operators = {"+", "-", "*", "/"}
        self._valid_unary_operators = {"square", "cube"}

    def fit_and_select(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        variable_names: list[str] | None = None,
    ) -> VerificationResult:
        """通用符号回归接口。"""
        equations = self._fit_equations(X, y, variable_names)
        best_row = self._select_best_equation(equations)
        return VerificationResult(
            equation=self._extract_equation_string(best_row),
            loss=float(best_row["loss"]),
            complexity=float(best_row["complexity"]) if "complexity" in best_row.index else None,
            model_selection=self.model_selection,
            raw_equations=equations.copy(),
        )

    def search_invariant(
        self,
        X: pd.DataFrame,
        variable_names: list[str] | None = None,
    ) -> InvariantSearchResult:
        """搜索在单实验内近似恒定的候选表达式。

        这里不再拟合随机目标。我们枚举由当前可观测/派生序列组成的
        简单候选表达式，直接评价它们在时间轴上的相对波动。
        """
        if X.empty:
            raise ValueError("X 不能为空。")

        feature_names = variable_names or list(X.columns)
        if len(feature_names) != X.shape[1]:
            raise ValueError("variable_names 的长度必须与 X 的列数一致。")

        candidate_rows = self._enumerate_invariant_candidates(X, feature_names)
        if candidate_rows.empty:
            raise RuntimeError("未能生成任何候选不变量表达式。")

        ranked = candidate_rows.sort_values(
            by=["score", "complexity", "residual_std"],
            ascending=True,
        )
        best_row = ranked.iloc[0]

        return InvariantSearchResult(
            equation=str(best_row["equation"]),
            loss=float(best_row["score"]),
            complexity=float(best_row["complexity"]),
            residual_std=float(best_row["residual_std"]),
            predicted_mean=float(best_row["predicted_mean"]),
            score=float(best_row["score"]),
            raw_equations=ranked.copy(),
        )

    def _enumerate_invariant_candidates(
        self,
        X: pd.DataFrame,
        feature_names: list[str],
    ) -> pd.DataFrame:
        """枚举低复杂度候选表达式，并按时间稳定性评分。"""
        arrays = {name: X[name].to_numpy(dtype=float) for name in feature_names}
        terms: list[tuple[str, np.ndarray, int]] = []

        for name, values in arrays.items():
            terms.append((name, values, 1))

            if "square" in self._sanitize_unary_operators(self.unary_operators):
                terms.append((f"square({name})", values * values, 2))

            if "cube" in self._sanitize_unary_operators(self.unary_operators):
                terms.append((f"cube({name})", values * values * values, 3))

        candidate_rows: list[dict[str, float | str]] = []
        seen: set[str] = set()

        def add_candidate(equation: str, values: np.ndarray, complexity: int) -> None:
            if equation in seen:
                return
            seen.add(equation)

            values = np.asarray(values, dtype=float)
            finite_mask = np.isfinite(values)
            if finite_mask.sum() < max(5, int(0.8 * len(values))):
                return

            finite_values = values[finite_mask]
            residual_std = float(np.std(finite_values))
            predicted_mean = float(np.mean(finite_values))
            mean_abs = float(np.mean(np.abs(finite_values)))
            dynamic_range = float(np.max(finite_values) - np.min(finite_values))
            scale = mean_abs + 1e-8
            relative_variation = residual_std / scale
            nontriviality_penalty = 1.0 if mean_abs < 1e-6 and dynamic_range < 1e-6 else 0.0
            score = relative_variation + 1e-3 * complexity + nontriviality_penalty

            candidate_rows.append(
                {
                    "equation": equation,
                    "loss": score,
                    "score": score,
                    "complexity": float(complexity),
                    "residual_std": residual_std,
                    "predicted_mean": predicted_mean,
                    "nontriviality_penalty": nontriviality_penalty,
                }
            )

        for equation, values, complexity in terms:
            add_candidate(equation, values, complexity)

        binary_ops = self._sanitize_binary_operators(self.binary_operators)
        product_terms: list[tuple[str, np.ndarray, int]] = []
        for left_idx, (left_name, left_values, left_complexity) in enumerate(terms):
            for right_name, right_values, right_complexity in terms[left_idx + 1:]:
                base_complexity = left_complexity + right_complexity + 1

                if "+" in binary_ops:
                    add_candidate(
                        f"({left_name} + {right_name})",
                        left_values + right_values,
                        base_complexity,
                    )
                if "-" in binary_ops:
                    add_candidate(
                        f"({left_name} - {right_name})",
                        left_values - right_values,
                        base_complexity,
                    )
                    add_candidate(
                        f"({right_name} - {left_name})",
                        right_values - left_values,
                        base_complexity,
                    )
                if "*" in binary_ops:
                    product_name = f"({left_name} * {right_name})"
                    product_values = left_values * right_values
                    product_terms.append((product_name, product_values, base_complexity))
                    add_candidate(
                        product_name,
                        product_values,
                        base_complexity,
                    )
                if "/" in binary_ops:
                    add_candidate(
                        f"({left_name} / {right_name})",
                        self._safe_divide(left_values, right_values),
                        base_complexity,
                    )
                    add_candidate(
                        f"({right_name} / {left_name})",
                        self._safe_divide(right_values, left_values),
                        base_complexity,
                    )

        if "*" in binary_ops and ("+" in binary_ops or "-" in binary_ops):
            for term_name, term_values, term_complexity in terms:
                for product_name, product_values, product_complexity in product_terms:
                    # 允许发现类似 a + square(v) * a 这样的速度相关惯性结构。
                    # 仍然保持低阶枚举，避免无限制组合导致表达式爆炸。
                    combined_complexity = term_complexity + product_complexity + 1
                    if combined_complexity > self.maxsize:
                        continue

                    if "+" in binary_ops:
                        add_candidate(
                            f"({term_name} + {product_name})",
                            term_values + product_values,
                            combined_complexity,
                        )
                    if "-" in binary_ops:
                        add_candidate(
                            f"({term_name} - {product_name})",
                            term_values - product_values,
                            combined_complexity,
                        )
                        add_candidate(
                            f"({product_name} - {term_name})",
                            product_values - term_values,
                            combined_complexity,
                        )

        return pd.DataFrame(candidate_rows)

    def _safe_divide(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=float),
            where=np.abs(denominator) > 1e-8,
        )

    def _fit_equations(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        variable_names: list[str] | None = None,
    ) -> pd.DataFrame:
        pysr_regressor = self._load_pysr_regressor()
        if X.empty:
            raise ValueError("X 不能为空。")

        y_array = np.asarray(y, dtype=float).reshape(-1)
        if len(X) != len(y_array):
            raise ValueError("X 和 y 的样本数必须一致。")

        feature_names = variable_names or list(X.columns)
        if len(feature_names) != X.shape[1]:
            raise ValueError("variable_names 的长度必须与 X 的列数一致。")

        model = pysr_regressor(
            niterations=self.niterations,
            populations=self.population_size,
            maxsize=self.maxsize,
            binary_operators=self._sanitize_binary_operators(self.binary_operators),
            unary_operators=self._sanitize_unary_operators(self.unary_operators),
            model_selection=self.model_selection,
            elementwise_loss="loss(prediction, target) = (prediction - target)^2",
            progress=False,
            procs=0,
            multithreading=False,
            random_state=self.random_state,
            deterministic=True,
            **self.extra_pysr_kwargs,
        )

        try:
            model.fit(X.values, y_array, variable_names=feature_names)
        except AttributeError as exc:
            if "_validate_data" in str(exc):
                raise RuntimeError(
                    "检测到 PySR 与 scikit-learn 版本不兼容。"
                    f"当前 scikit-learn 版本为 {sklearn_version}。"
                    "建议使用 Python 3.11，并将 scikit-learn 固定到 <1.7 "
                    "后重新安装依赖，例如 `pip install \"scikit-learn<1.7\"`。"
                ) from exc
            raise

        equations = model.equations_
        if equations is None or len(equations) == 0:
            raise RuntimeError("PySR 未返回任何候选方程。")
        return equations

    def _load_pysr_regressor(self) -> Any:
        """延迟导入 PySR，避免仅做不变量枚举时也启动 Julia 后端。"""
        global PySRRegressor, _PYSR_IMPORT_ERROR

        if PySRRegressor is not None:
            return PySRRegressor

        try:
            from pysr import PySRRegressor as imported_regressor
        except Exception as exc:  # pragma: no cover - 依赖和 Julia 环境因机器而异
            _PYSR_IMPORT_ERROR = exc
            raise ImportError(
                "无法加载 pysr 或其 Julia 后端。若你需要运行 PySR 符号回归，"
                "请确认已安装 pysr，并在当前 conda 环境中执行过 `import pysr; pysr.install()`。"
                f" 原始错误: {exc}"
            ) from exc

        PySRRegressor = imported_regressor
        return PySRRegressor

    def _sanitize_binary_operators(self, operators: list[str]) -> list[str]:
        sanitized = [op for op in operators if op in self._valid_binary_operators]
        return sanitized or ["+", "-", "*", "/"]

    def _sanitize_unary_operators(self, operators: list[str]) -> list[str]:
        sanitized = [op for op in operators if op in self._valid_unary_operators]
        return sanitized or ["square", "cube"]

    def _select_best_equation(self, equations: pd.DataFrame) -> pd.Series:
        ranked = equations.copy()
        sort_columns: list[str] = ["loss"]
        if "complexity" in ranked.columns:
            sort_columns.append("complexity")
        ranked = ranked.sort_values(by=sort_columns, ascending=True)
        return ranked.iloc[0]

    def _extract_equation_string(self, row: pd.Series) -> str:
        candidate_columns = ["equation", "sympy_format", "lambda_format"]
        for column in candidate_columns:
            if column in row.index:
                return str(row[column])
        raise KeyError(f"无法在 PySR 输出中找到表达式列，当前可用列为: {list(row.index)!r}")
