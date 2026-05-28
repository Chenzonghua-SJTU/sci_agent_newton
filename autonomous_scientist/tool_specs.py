from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Planning and execution metadata for one scientist action."""

    name: str
    description: str
    parameters: str
    planning_notes: tuple[str, ...] = field(default_factory=tuple)
    category: str = "general"
    generated_code_allowed: bool = False
    requires_generated_code: bool = False
    mutates_notebook: bool = True
    risk_level: str = "low"

    def render_for_prompt(self) -> str:
        text = f"- {self.name}: {self.description}"
        if self.parameters:
            text = f"{text} 参数包含 {self.parameters}。"
        if self.planning_notes:
            text = f"{text} " + " ".join(self.planning_notes)
        return text


DEFAULT_TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="run_experiments",
        description="批量运行一个或多个黑盒物理实验",
        parameters=(
            "experiments: list[dict]，每个 dict 可包含 initial_q/q0, initial_v/v0, "
            "constant_force/F_ext/F, t_end, dt；noise_std 由宿主固定为 0，F_ext=0 表示无外力。"
            "如果只做一个实验，也仍然放在 experiments 列表里"
        ),
        planning_notes=(
            "一次尽量提交一组高信息密度实验，例如不同控制参数、不同初始条件或反例条件。",
            "观测噪声全局固定为 0，不要把 noise_std 当成可探索变量。",
        ),
        category="experiment",
        risk_level="medium",
    ),
    ToolSpec(
        name="analyze_data",
        description="调用数据处理 LLM 编写并执行 Python 数据处理代码",
        parameters=(
            "analysis_mode: maintain_ledger|validate_hypothesis, analysis_goal, "
            "experiment_id 或 experiment_ids, optional_series, expected_outputs；"
            "validate_hypothesis 还应包含 hypothesis_id 或 candidate_expression"
        ),
        planning_notes=(
            "决策 LLM 必须在 analysis_goal 中说明要维护什么表项或用什么口径验证假说；数据处理 LLM 只写代码执行。",
            "数据处理 LLM 是写代码和计算指标的工具，不负责提出最终物理公式。",
            "maintain_ledger 用于维护实验数据记录表：追加决策 LLM 指定的派生序列、中间物理量和 OBS 观察条目。",
            "validate_hypothesis 用于验证假说：按决策 LLM 指定的残差定义、实验范围和指标写入 VAL。",
            "数据处理 LLM 会收到完整 t,q 与已有派生序列；不要在 prompt 中暗示任何特定公式结构。",
        ),
        category="data",
        generated_code_allowed=True,
        requires_generated_code=True,
        risk_level="medium",
    ),
    ToolSpec(
        name="manage_hypotheses",
        description="提出、接受、拒绝、列出候选规律",
        parameters=(
            "operation: propose|accept|reject|list；"
            "可包含 hypothesis_id, expression, readable_summary, variables, assumptions, "
            "next_tests, evidence_type, experiment_ids, metric_name, metric_values, "
            "aggregate_score, summary, note；也兼容 hypothesis/evidence/metrics 嵌套对象"
        ),
        planning_notes=(
            "如果新想法与已有假设相近，不要重新提出；引用 hypothesis_id 后直接 accept 或 reject。",
            "propose 是决策 LLM 的职责：先综合完整原始 t,q、全部派生序列、OBS、VAL 和工具反馈，再提出公式。",
            "propose 只能提交单一、明确、可证伪的公式；不要提交“或/待定/某种关系”的猜想。",
            "propose 必须用 observation_ids 和 source_data_refs 说明公式来自哪些原始数据、派生数据和数据处理结果。",
            "accept/reject 必须引用 VAL 验证条目；没有 supported/testing/weakened 等中间状态。",
            "别名: refuted/failed/deny 会视为 reject，confirm/validated/support 会视为 accept。",
        ),
        category="hypothesis",
        risk_level="medium",
    ),
    ToolSpec(
        name="finish",
        description="当存在 accepted 假说后结束科研循环",
        parameters="可为空；宿主会检查假说表中是否已有 accepted 状态",
        planning_notes=(
            "只有 hypothesis_registry 中已经有 accepted 假说时才能结束。",
            "如果还没有 accepted 假说，请继续实验、维护实验数据记录表或验证假说。",
        ),
        category="control",
        mutates_notebook=False,
        risk_level="low",
    ),
)

TOOL_SPECS_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in DEFAULT_TOOL_SPECS}
PLANNING_TOOL_NAMES: tuple[str, ...] = tuple(spec.name for spec in DEFAULT_TOOL_SPECS)


def render_tool_name_list(specs: tuple[ToolSpec, ...] = DEFAULT_TOOL_SPECS) -> str:
    return "\n".join(f"{index}. {spec.name}" for index, spec in enumerate(specs, start=1))


def render_tool_descriptions(specs: tuple[ToolSpec, ...] = DEFAULT_TOOL_SPECS) -> str:
    return "\n".join(spec.render_for_prompt() for spec in specs)
