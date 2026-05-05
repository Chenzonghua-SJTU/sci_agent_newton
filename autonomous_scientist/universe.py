from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


class ForceFieldType(str, Enum):
    """当前支持的受力场类型。"""

    FREE = "free"
    CONSTANT = "constant"


@dataclass(slots=True)
class ExperimentConfig:
    """一次实验的输入配置。

    Attributes:
        initial_q: 初始位置 q(0)。
        initial_v: 初始速度 v(0)。
        force_field_type: 势场/受力场类型。
        t_span: 时间区间，格式为 (t_start, t_end)。
        dt: 采样时间间隔。
        noise_std: 观测噪声标准差，仅加在位置 q 上。
        constant_force: 当场景为恒定外力时使用的常力大小。
    """

    initial_q: float
    initial_v: float
    force_field_type: ForceFieldType
    t_span: tuple[float, float]
    dt: float
    noise_std: float = 0.0
    constant_force: float = 1.0


@dataclass(slots=True)
class ExperimentResult:
    """一次实验的输出结果。

    Attributes:
        t: 时间采样点。
        q: 位置序列。
        v: 速度序列。
        metadata: 实验元信息，便于 Agent 后续做反思与追踪。
    """

    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    metadata: dict[str, float | str]


class VirtualUniverse:
    """黑盒虚拟物理宇宙。

    这个宇宙的真实动力学不是经典牛顿第二定律，而是一个
    “速度相关惯性”模型：

        F(q) = (m0 + alpha * v^2) * a

    其中：
        q: 位置
        v: 速度
        a: 加速度 = dv/dt
        m0: 低速极限下的基础惯性
        alpha: 控制速度相关惯性的强度
        F(q): 外力，等于 -V'(q)

    对应到一阶常微分方程组可以写为：

        dq/dt = v
        dv/dt = F(q) / (m0 + alpha * v^2)

    这样既保留了“惯性随速度变化”的反常识结构，又避免了旧版
    v -> 0 时有效惯性趋近 0、加速度发散的奇点问题。
    """

    def __init__(
        self,
        alpha: float = 1.0,
        base_mass: float = 1.0,
        solver_rtol: float = 1e-7,
        solver_atol: float = 1e-9,
        random_seed: int | None = 42,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha 必须为正数。")
        if base_mass <= 0:
            raise ValueError("base_mass 必须为正数。")

        self.alpha = float(alpha)
        self.base_mass = float(base_mass)
        self.solver_rtol = float(solver_rtol)
        self.solver_atol = float(solver_atol)
        self._rng = np.random.default_rng(random_seed)

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """运行一次黑盒物理实验并返回轨迹数据。

        Args:
            config: 实验配置对象。

        Returns:
            包含时间、位置、速度和元信息的实验结果。
        """
        t_start, t_end = config.t_span
        if t_end <= t_start:
            raise ValueError("t_span 必须满足 t_end > t_start。")
        if config.dt <= 0:
            raise ValueError("dt 必须为正数。")

        t_eval = self._build_time_grid(t_start=t_start, t_end=t_end, dt=config.dt)
        force_fn = self._build_force_function(config)

        initial_state = np.array([config.initial_q, config.initial_v], dtype=float)

        solution = solve_ivp(
            fun=lambda t, y: self._dynamics(t=t, state=y, force_fn=force_fn),
            t_span=config.t_span,
            y0=initial_state,
            t_eval=t_eval,
            method="RK45",
            rtol=self.solver_rtol,
            atol=self.solver_atol,
        )

        if not solution.success:
            raise RuntimeError(f"数值积分失败: {solution.message}")

        q = solution.y[0].copy()
        v = solution.y[1].copy()

        if config.noise_std > 0:
            q += self._rng.normal(loc=0.0, scale=config.noise_std, size=q.shape)

        metadata: dict[str, float | str] = {
            "force_field_type": config.force_field_type.value,
            "constant_force": config.constant_force,
            "noise_std": config.noise_std,
        }

        return ExperimentResult(t=t_eval, q=q, v=v, metadata=metadata)

    def _build_time_grid(self, t_start: float, t_end: float, dt: float) -> np.ndarray:
        """构造稳定的采样时间网格。

        这里显式把终点拼回去，避免浮点误差导致最后一个采样点丢失。
        """
        steps = int(np.floor((t_end - t_start) / dt))
        t_eval = t_start + np.arange(steps + 1, dtype=float) * dt

        if t_eval[-1] < t_end:
            t_eval = np.append(t_eval, t_end)

        return t_eval

    def _build_force_function(self, config: ExperimentConfig) -> Callable[[float], float]:
        """根据实验配置构造受力函数 F(q)。"""
        if config.force_field_type is ForceFieldType.FREE:
            return lambda q: 0.0

        if config.force_field_type is ForceFieldType.CONSTANT:
            constant_force = float(config.constant_force)
            return lambda q: constant_force

        raise NotImplementedError(
            f"暂不支持的 force_field_type: {config.force_field_type!r}"
        )

    def _dynamics(
        self,
        t: float,
        state: np.ndarray,
        force_fn: Callable[[float], float],
    ) -> np.ndarray:
        """定义一阶状态方程。

        Args:
            t: 时间变量。当前动力学与 t 无显式关系，但 solve_ivp 需要这个参数。
            state: 当前状态向量 [q, v]。
            force_fn: 外力函数 F(q)。

        Returns:
            状态导数 [dq/dt, dv/dt]。
        """
        del t

        q, v = float(state[0]), float(state[1])
        force = force_fn(q)

        # 速度相关惯性：低速时惯性约为 base_mass，高速时惯性增大。
        # 因为分母始终 >= base_mass，所以 v=0 时不会出现加速度奇点。
        effective_inertia = self.base_mass + self.alpha * v * v

        dq_dt = v
        dv_dt = force / effective_inertia

        return np.array([dq_dt, dv_dt], dtype=float)
