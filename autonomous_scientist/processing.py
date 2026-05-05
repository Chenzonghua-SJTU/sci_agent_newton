from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


@dataclass(slots=True)
class PhaseSpaceData:
    """兼容旧接口的相空间特征数据容器。"""

    t: np.ndarray
    q_raw: np.ndarray
    q_smooth: np.ndarray
    v_smooth: np.ndarray
    a_smooth: np.ndarray
    frame: pd.DataFrame


@dataclass(slots=True)
class SeriesSummary:
    """单个时间序列的统计摘要。"""

    name: str
    minimum: float
    maximum: float
    mean: float
    std: float
    first_value: float
    last_value: float
    trend_slope: float

    def to_text(self) -> str:
        return (
            f"{self.name}: min={self.minimum:.6f}, max={self.maximum:.6f}, "
            f"mean={self.mean:.6f}, std={self.std:.6f}, "
            f"start={self.first_value:.6f}, end={self.last_value:.6f}, "
            f"slope={self.trend_slope:.6f}"
        )


class DataProcessingTool:
    """面向 Agent 的通用时间序列处理工具。

    与上一版最大的不同在于：
    - 不再默认“必须生成 v 和 a”
    - 改为提供一组按需调用的数学工具
    - LLM 可以先观察 q(t)，然后自己决定是否要平滑、求一阶差分、求二阶差分
    """

    def __init__(
        self,
        window_length: int = 9,
        polyorder: int = 3,
        derivative_smoothing_window: int | None = None,
    ) -> None:
        if window_length < 5:
            raise ValueError("window_length 至少应为 5。")
        if window_length % 2 == 0:
            raise ValueError("window_length 必须为奇数。")
        if polyorder >= window_length:
            raise ValueError("polyorder 必须严格小于 window_length。")

        self.window_length = window_length
        self.polyorder = polyorder
        self.derivative_smoothing_window = derivative_smoothing_window

    def transform(self, t: np.ndarray, q: np.ndarray) -> PhaseSpaceData:
        """保留旧接口，便于兼容已有代码。"""
        t_array = np.asarray(t, dtype=float)
        q_array = np.asarray(q, dtype=float)

        q_smooth = self.smooth_series(t_array, q_array)
        v_estimate = self.differentiate_series(t_array, q_smooth, order=1, smooth_after=True)
        a_estimate = self.differentiate_series(t_array, q_smooth, order=2, smooth_after=True)

        frame = pd.DataFrame(
            {
                "t": t_array,
                "q": q_array,
                "q_smooth": q_smooth,
                "v": v_estimate,
                "a": a_estimate,
            }
        )
        return PhaseSpaceData(
            t=t_array,
            q_raw=q_array,
            q_smooth=q_smooth,
            v_smooth=v_estimate,
            a_smooth=a_estimate,
            frame=frame,
        )

    def smooth_series(
        self,
        t: np.ndarray,
        values: np.ndarray,
        window_length: int | None = None,
        polyorder: int | None = None,
    ) -> np.ndarray:
        """对任意时间序列做 Savitzky-Golay 平滑。"""
        t_array = np.asarray(t, dtype=float)
        values_array = np.asarray(values, dtype=float)
        self._validate_inputs(t=t_array, values=values_array)

        effective_window = self._resolve_window_length(
            target_length=len(values_array),
            preferred_window=window_length or self.window_length,
        )
        effective_polyorder = min(polyorder or self.polyorder, effective_window - 1)

        return savgol_filter(
            values_array,
            window_length=effective_window,
            polyorder=effective_polyorder,
            mode="interp",
        )

    def differentiate_series(
        self,
        t: np.ndarray,
        values: np.ndarray,
        order: int = 1,
        smooth_before: bool = False,
        smooth_after: bool = True,
    ) -> np.ndarray:
        """对任意时间序列做 1 阶或 2 阶导数估计。"""
        if order not in {1, 2}:
            raise ValueError("当前只支持 1 阶或 2 阶导数。")

        t_array = np.asarray(t, dtype=float)
        values_array = np.asarray(values, dtype=float)
        self._validate_inputs(t=t_array, values=values_array)
        dt = self._infer_uniform_dt(t_array)

        working_values = values_array
        if smooth_before:
            working_values = self.smooth_series(t_array, working_values)

        differentiated = working_values.copy()
        for _ in range(order):
            differentiated = np.gradient(differentiated, dt, edge_order=2)

        if smooth_after:
            derivative_window = self._resolve_window_length(
                target_length=len(values_array),
                preferred_window=(
                    self.derivative_smoothing_window
                    if self.derivative_smoothing_window is not None
                    else self.window_length
                ),
            )
            differentiated = savgol_filter(
                differentiated,
                window_length=derivative_window,
                polyorder=min(self.polyorder, derivative_window - 1),
                mode="interp",
            )

        return differentiated

    def summarize_series(self, t: np.ndarray, values: np.ndarray, name: str) -> SeriesSummary:
        """为单个序列生成简洁统计摘要。"""
        t_array = np.asarray(t, dtype=float)
        values_array = np.asarray(values, dtype=float)
        self._validate_inputs(t=t_array, values=values_array)

        finite_mask = np.isfinite(values_array)
        if finite_mask.sum() < 3:
            raise ValueError(f"序列 `{name}` 的有限数值点少于 3 个，无法生成可靠统计摘要。")

        finite_t = t_array[finite_mask]
        finite_values = values_array[finite_mask]
        trend_slope = float(np.polyfit(finite_t, finite_values, deg=1)[0]) if len(finite_t) >= 3 else 0.0
        return SeriesSummary(
            name=name,
            minimum=float(np.min(finite_values)),
            maximum=float(np.max(finite_values)),
            mean=float(np.mean(finite_values)),
            std=float(np.std(finite_values)),
            first_value=float(finite_values[0]),
            last_value=float(finite_values[-1]),
            trend_slope=trend_slope,
        )

    def compute_relationship_score(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> dict[str, float]:
        """计算两个序列之间的简单关系指标。"""
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        if len(left_array) != len(right_array):
            raise ValueError("两个序列长度必须一致。")

        finite_mask = np.isfinite(left_array) & np.isfinite(right_array)
        if finite_mask.sum() < 3:
            raise ValueError("两个序列的共同有限数值点少于 3 个，无法计算关系指标。")

        left_array = left_array[finite_mask]
        right_array = right_array[finite_mask]

        corr = 0.0
        if np.std(left_array) > 1e-12 and np.std(right_array) > 1e-12:
            corr = float(np.corrcoef(left_array, right_array)[0, 1])

        mse = float(np.mean((left_array - right_array) ** 2))
        return {
            "correlation": corr,
            "mse": mse,
        }

    def _validate_inputs(self, t: np.ndarray, values: np.ndarray) -> None:
        if t.ndim != 1 or values.ndim != 1:
            raise ValueError("t 和 values 都必须是一维数组。")
        if len(t) != len(values):
            raise ValueError("t 和 values 的长度必须一致。")
        if len(t) < 5:
            raise ValueError("至少需要 5 个采样点。")
        time_diff = np.diff(t)
        if np.any(time_diff <= 0):
            raise ValueError("t 必须严格递增。")

    def _infer_uniform_dt(self, t: np.ndarray) -> float:
        time_diff = np.diff(t)
        dt = float(np.mean(time_diff))
        if not np.allclose(time_diff, dt, rtol=1e-4, atol=1e-8):
            raise ValueError("当前工具假设时间采样近似均匀。")
        return dt

    def _resolve_window_length(self, target_length: int, preferred_window: int) -> int:
        window = min(preferred_window, target_length)
        if window % 2 == 0:
            window -= 1
        if window < 5:
            window = 5
        if window > target_length:
            window = target_length if target_length % 2 == 1 else target_length - 1
        if window <= self.polyorder:
            raise ValueError("样本点过少，无法构造合法窗口。")
        return window
