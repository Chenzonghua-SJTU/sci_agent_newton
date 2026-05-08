from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from .processing import DataProcessingTool
from .reporting import ScientificReporter
from .universe import (
    ExperimentConfig,
    ExperimentResult,
    ForceFieldType,
    VirtualUniverse,
)
from .verification import InvariantSearchResult, VerificationEngine


@dataclass(slots=True)
class ExperimentRecord:
    """记录一次原始实验。"""

    experiment_id: str
    config: ExperimentConfig
    result: ExperimentResult
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DerivedSeries:
    """Notebook 中的一条派生序列。"""

    experiment_id: str
    name: str
    values: np.ndarray
    source_name: str
    provenance: str
    summary_text: str


@dataclass(slots=True)
class ActionDecision:
    """LLM 输出的下一步动作。"""

    thought: str
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionRecord:
    """已执行动作及其反馈。"""

    step_index: int
    decision: ActionDecision
    observation: str


@dataclass(slots=True)
class LawHypothesis:
    """最终规律总结。"""

    summary: str
    proposed_law: str
    evidence: str
    confidence: str
    next_steps: str
    raw_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GeneralizationCheck:
    """跨实验泛化验证结果。"""

    expression: str
    experiment_ids: list[str]
    metric_name: str
    metric_values: dict[str, float]
    aggregate_score: float
    summary_text: str


@dataclass(slots=True)
class CandidateLaw:
    """候选规律条目。"""

    expression: str
    source_experiment_id: str
    score: float
    origin: str
    notes: str


@dataclass(slots=True)
class ScientificNotebook:
    """科学家智能体的实验记录本。"""

    experiments: dict[str, ExperimentRecord] = field(default_factory=dict)
    derived_series: dict[str, dict[str, DerivedSeries]] = field(default_factory=dict)
    action_history: list[ActionRecord] = field(default_factory=list)
    invariant_history: list[InvariantSearchResult] = field(default_factory=list)
    generalization_checks: list[GeneralizationCheck] = field(default_factory=list)
    candidate_laws: list[CandidateLaw] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def register_experiment(self, record: ExperimentRecord) -> None:
        self.experiments[record.experiment_id] = record
        q_summary = (
            f"{record.experiment_id}: 获取到原始轨迹 q(t)，"
            f"时间点数 {len(record.result.t)}，"
            f"q 范围 [{np.min(record.result.q):.6f}, {np.max(record.result.q):.6f}]。"
        )
        t_summary = (
            f"{record.experiment_id}: 时间范围 [{np.min(record.result.t):.6f}, {np.max(record.result.t):.6f}]。"
        )
        self.notes.append(q_summary)
        self.notes.append(t_summary)

    def register_series(self, series: DerivedSeries) -> None:
        self.derived_series.setdefault(series.experiment_id, {})[series.name] = series
        self.notes.append(
            f"{series.experiment_id}: 新增派生序列 `{series.name}`，来源 `{series.source_name}`，"
            f"方法 `{series.provenance}`。{series.summary_text}"
        )

    def add_action_record(self, action_record: ActionRecord) -> None:
        self.action_history.append(action_record)

    def add_invariant(self, invariant: InvariantSearchResult, experiment_id: str, feature_series: list[str]) -> None:
        self.invariant_history.append(invariant)
        self.notes.append(
            f"实验 {experiment_id}: 不变量搜索得到候选表达式 `{invariant.equation}`，"
            f"features={feature_series}，score={invariant.score:.12f}。"
        )
        self.candidate_laws.append(
            CandidateLaw(
                expression=invariant.equation,
                source_experiment_id=experiment_id,
                score=invariant.score,
                origin="search_invariants",
                notes=f"features={feature_series}",
            )
        )

    def add_candidate_law(self, candidate: CandidateLaw) -> None:
        self.candidate_laws.append(candidate)
        self.notes.append(
            f"实验 {candidate.source_experiment_id}: 新增候选规律 `{candidate.expression}`，"
            f"origin={candidate.origin}，score={candidate.score:.12f}。{candidate.notes}"
        )

    def add_generalization_check(self, check: GeneralizationCheck) -> None:
        self.generalization_checks.append(check)
        self.notes.append(check.summary_text)

    def get_series_values(self, experiment_id: str, series_name: str) -> np.ndarray:
        if series_name == "q":
            return self.experiments[experiment_id].result.q
        if series_name == "t":
            return self.experiments[experiment_id].result.t
        return self.derived_series[experiment_id][series_name].values

    def available_series(self, experiment_id: str) -> list[str]:
        names = ["q", "t"]
        names.extend(sorted(self.derived_series.get(experiment_id, {}).keys()))
        return names

    def latest_experiment_id(self) -> str | None:
        if not self.experiments:
            return None
        return sorted(self.experiments.keys())[-1]


@dataclass(slots=True)
class ScientificCycleResult:
    """一次多轮科学发现循环的输出。"""

    notebook: ScientificNotebook
    final_law: LawHypothesis
    report_markdown: str | None = None
    report_path: str | None = None
    figure_paths: list[str] = field(default_factory=list)


class HypothesisBrain:
    """LLM 驱动的科学发现大脑。

    这版不再假设系统天生知道 v 和 a。
    它只能看到：
    - 实验记录本中的文字摘要
    - 可用工具列表
    然后自己规划下一步动作。
    """

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        if OpenAI is None:
            raise ImportError("未检测到 openai 官方库。")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def summarize_notebook(self, notebook: ScientificNotebook, goal: str, max_steps: int) -> str:
        """将实验记录本压缩成适合 LLM 规划的文本。"""
        lines = [
            f"研究目标: {goal}",
            f"当前已执行步骤数: {len(notebook.action_history)} / {max_steps}",
            f"实验数量: {len(notebook.experiments)}",
            "世界观提醒: 这是一个人工构造的虚拟物理世界，不保证服从地球上的经典力学；"
            "任何熟悉的物理规律都只能作为待检验假设，不能作为默认真理。",
            "方法论约束: 不能因为某个低阶多项式拟合效果看起来不错，就直接宣布找到了底层定律；"
            "任何结论都应经过更多变量构造、更多实验条件和跨实验复验。",
        ]

        if not notebook.experiments:
            lines.append("当前还没有任何实验数据。")
        else:
            for experiment_id, record in sorted(notebook.experiments.items()):
                q = record.result.q
                t = record.result.t
                force_text = (
                    str(record.config.constant_force)
                    if record.config.force_field_type is ForceFieldType.CONSTANT
                    else "N/A（free 场景无外力，constant_force 参数被忽略）"
                )
                linear_coeff = np.polyfit(t, q, deg=1)
                linear_pred = np.polyval(linear_coeff, t)
                linear_mse = float(np.mean((q - linear_pred) ** 2))
                quad_coeff = np.polyfit(t, q, deg=2)
                quad_pred = np.polyval(quad_coeff, t)
                quad_mse = float(np.mean((q - quad_pred) ** 2))
                cubic_coeff = np.polyfit(t, q, deg=3)
                cubic_pred = np.polyval(cubic_coeff, t)
                cubic_mse = float(np.mean((q - cubic_pred) ** 2))
                lines.extend(
                    [
                        f"实验 {experiment_id}: 场景={record.config.force_field_type.value}, "
                        f"F_ext={force_text}, q0={record.config.initial_q}, v0={record.config.initial_v}",
                        f"实验 {experiment_id}: q 范围 [{np.min(q):.6f}, {np.max(q):.6f}]，"
                        f"线性拟合 MSE={linear_mse:.6e}，二次拟合 MSE={quad_mse:.6e}，三次拟合 MSE={cubic_mse:.6e}",
                        f"实验 {experiment_id}: 可用序列 {notebook.available_series(experiment_id)}",
                    ]
                )
                for series_name, series in sorted(notebook.derived_series.get(experiment_id, {}).items()):
                    lines.append(f"实验 {experiment_id} 派生序列 {series_name}: {series.summary_text}")

        if notebook.invariant_history:
            latest_invariant = notebook.invariant_history[-1]
            lines.append(
                f"最近一次不变量搜索: equation={latest_invariant.equation}, "
                f"score={latest_invariant.score:.12f}, residual_std={latest_invariant.residual_std:.12f}"
            )

        if notebook.generalization_checks:
            latest_check = notebook.generalization_checks[-1]
            lines.append(
                f"最近一次跨实验验证: expression={latest_check.expression}, "
                f"aggregate_score={latest_check.aggregate_score:.6f}, "
                f"metric={latest_check.metric_name}"
            )

        if notebook.action_history:
            latest_actions = notebook.action_history[-3:]
            lines.append("最近动作记录:")
            for item in latest_actions:
                lines.append(
                    f"- step {item.step_index}: action={item.decision.action}, "
                    f"thought={item.decision.thought}, observation={item.observation}"
                )

        return "\n".join(lines)

    def plan_next_action(
        self,
        notebook_summary: str,
        max_steps: int,
    ) -> ActionDecision:
        """根据实验记录本决定下一步动作。"""
        system_prompt = (
            "你是一位被派往人工虚拟物理世界的科学家。"
            "这个世界可能不服从地球上的经典牛顿力学，经典公式只能作为待检验假设。"
            "你不知道这个世界的底层方程。"
            "你不能预设速度和加速度一定重要；你只能从位置-时间数据和实验反馈出发，"
            "逐步决定下一步应该做什么。"
            "你必须主动寻找反常识规律，并避免过早套用熟悉的经典理论。"
            "你有一组工具可以调用，请输出严格 JSON。"
        )

        user_prompt = f"""
下面是当前实验记录本摘要：

{notebook_summary}

你可用的动作只有：
1. run_experiment
2. smooth_series
3. estimate_kinematics
4. differentiate_series
5. inspect_series
6. inspect_relationships
7. define_derived_quantity
8. fit_relationship_model
9. propose_candidate_expression
10. test_candidate_expression
11. cross_experiment_check
12. rank_candidate_laws
13. finalize_law

动作说明：
- run_experiment: 做一个新实验。参数可包含 initial_q, initial_v, force_field_type, constant_force, t_end, dt, noise_std。force_field_type 可用 free/none/no_force 或 constant/constant_force。
- smooth_series: 对某个已有序列平滑。参数包含 experiment_id, source_series, output_name, overwrite。
- estimate_kinematics: 从位置序列一次性估计平滑位置、速度和加速度。参数包含 experiment_id, source_series, position_name, velocity_name, acceleration_name, window_length, polyorder, overwrite。它使用同一个局部多项式窗口估计导数，适合从 q(t) 构造 v/a。
- differentiate_series: 对某个已有序列做一阶或二阶差分。参数包含 experiment_id, source_series, order, output_name, smooth_before, smooth_after, overwrite。
- inspect_series: 查看一个或多个序列的统计。参数包含 experiment_id, series_names。
- inspect_relationships: 只查看两个序列之间的关系。参数包含 experiment_id, x_series, y_series。它会生成一张时间轨迹/散点关系图，并输出中性的观察摘要；不会使用外力、平方项拟合或多变量搜索。
- define_derived_quantity: 定义一个新的可复用物理量。参数包含 experiment_id, symbol, expression, description, overwrite。symbol 是新物理量名称，expression 是构造公式，可使用已有序列名、实验控制量 F_ext 以及 square(x), cube(x), sqrt(x), log(x), exp(x), sin(x), cos(x), abs(x)。
- fit_relationship_model: 对一个目标序列做由你指定基函数的最小二乘拟合。参数包含 experiment_id, target_series, basis_expressions, prediction_name, residual_name, include_intercept。basis_expressions 由你给出，例如已有序列、新定义物理量或由它们组成的表达式；工具只返回拟合系数和残差，不自动搜索公式。
- propose_candidate_expression: 让 LLM 基于可用特征量主动提出一个候选表达式，然后由程序立即求值和评分。参数包含 experiment_id, feature_series, output_name, acceptance_threshold。表达式只能使用已有序列名、实验控制量 F_ext、数字常数、括号、+、-、*、/、square(x)、cube(x)、sqrt(x)、log(x)、exp(x)、sin(x)、cos(x)、abs(x)。
- test_candidate_expression: 测试一个由你提出的表达式是否近似常数。参数包含 experiment_id, expression, output_name。表达式可使用已有序列名以及 square(x), cube(x), sqrt(x), log(x), exp(x), sin(x), cos(x), abs(x)。
- cross_experiment_check: 在多个实验之间检查同一表达式的稳定性。参数包含 expression, experiment_ids, metric_name。metric_name 当前支持 relative_std、mean_value、force_residual。若要验证表达式是否等于外力，优先使用 force_residual。
- rank_candidate_laws: 对已有候选规律做排序和比较。参数可为空。
- finalize_law: 当你认为证据已经足够时，结束探索并进入定律总结。

规划要求：
1. 如果还没有实验，先做一个简单基准实验，只观察位置-时间轨迹。
2. 不要一开始就默认使用任何派生物理量；需要通过观察轨迹后，再决定是否构造变化率或其他序列。
3. 若你需要速度/加速度，优先用 estimate_kinematics 从 q(t) 同时估计 q_smooth/v/a；只有在特殊情况下才直接对已有序列差分。
4. 在提出公式前，优先用 inspect_series 和 inspect_relationships 主动观察变量关系。inspect_relationships 每次只能比较两个序列，例如先看 v 与 a，再另起一步看新定义量与 a。
5. 如果观察到某种组合可能有物理意义，可以先用 define_derived_quantity 给它命名，再用 inspect_relationships 观察这个新量和其他量的关系。
6. 如果你想检验“某个目标是否可由若干自定义特征解释”，使用 fit_relationship_model，但基函数必须由你基于观察主动指定。
7. 当你已经看到足够关系线索时，使用 propose_candidate_expression 自己提出公式；不要调用不变量搜索、符号回归、枚举搜索或 PySR。
8. 如果某个解释只在单个实验里成立，不应立刻接受；优先尝试新的实验条件来复验。
9. 在设计新实验时，尽量改变初始条件、控制参数或对照场景，以增加辨识力。
10. 不要把“拟合某个控制参数的常数”当作发现定律；更好的做法是先由你基于证据提出候选量，然后再做跨实验验证。
11. 若你认为已有证据支持某条规律，可选择 finalize_law，但前提是至少存在一条候选规律和一次跨实验验证。
12. 如果 propose_candidate_expression 返回的候选被评价为波动大或残差大，应继续处理数据、定义新量或重新设计实验，不要把它当作规律。
13. 总步数上限为 {max_steps}，因此动作要尽量高信息密度。

请只返回 JSON：
{{
  "thought": "...",
  "action": "...",
  "parameters": {{...}}
}}

请不要在 JSON 中插入额外的说明文字。你可以在返回的 thought 字段里表达你的理由."""
        payload = self._request_json(system_prompt=system_prompt, user_prompt=user_prompt)
        return ActionDecision(
            thought=str(payload.get("thought", "继续收集证据。")),
            action=str(payload.get("action", "run_experiment")),
            parameters=dict(payload.get("parameters", {})),
        )

    def synthesize_final_law(
        self,
        notebook_summary: str,
    ) -> LawHypothesis:
        """根据完整实验记录本输出最终规律。"""
        system_prompt = (
            "你是一位理论物理学家，正在总结一个人工虚拟物理世界中的动力学规律。"
            "这个世界不保证服从经典牛顿力学；如果证据显示反常识规律，请直接指出。"
            "不要假装拥有不存在的证据；如果证据不够，请明确说出局限。"
            "请只输出 JSON。"
        )
        user_prompt = f"""
请根据以下实验记录本总结当前发现：

{notebook_summary}

返回 JSON：
{{
  "summary": "...",
  "proposed_law": "...",
  "evidence": "...",
  "confidence": "...",
  "next_steps": "..."
}}
"""
        payload = self._request_json(system_prompt=system_prompt, user_prompt=user_prompt)
        return LawHypothesis(
            summary=str(payload.get("summary", "当前已观察到明显偏离经典牛顿运动的轨迹行为。")),
            proposed_law=str(payload.get("proposed_law", "尚未得到足够严格的最终动力学方程。")),
            evidence=str(payload.get("evidence", "证据主要来自轨迹形状、导数量和符号回归结果。")),
            confidence=str(payload.get("confidence", "medium")),
            next_steps=str(payload.get("next_steps", "建议继续做更多外力与初速度条件下的实验。")),
            raw_payload=payload,
        )

    def propose_candidate_expression(
        self,
        notebook_summary: str,
        experiment_context: str,
        feature_summaries: list[str],
        feature_series: list[str],
    ) -> dict[str, Any]:
        """让 LLM 基于已构造的特征量提出一个待验证表达式。"""
        allowed_variables = ", ".join(f"`{name}`" for name in feature_series)
        system_prompt = (
            "你是一个物理公式提案器。你的任务不是验证公式，而是基于已有时间序列特征"
            "提出一个简洁、可计算、可能近似守恒或可能等于外部控制量的候选表达式。"
            "你需要像科学家一样解释为什么这个组合值得检验，尤其要利用变量趋势、"
            "相关性、有效惯性或跨实验对照中出现的线索。"
            "验证将由程序完成，所以不要声称公式已经成立。"
            "请只输出严格 JSON。"
        )
        user_prompt = f"""
当前实验上下文：
{experiment_context}

可用特征量只能来自以下变量：
{allowed_variables}

这些特征量的统计摘要：
{chr(10).join(feature_summaries)}

当前实验记录本摘要：
{notebook_summary}

请提出一个候选表达式。要求：
1. 表达式只能使用上面列出的变量名、数字常数、括号、+、-、*、/、square(x)、cube(x)、sqrt(x)、log(x)、exp(x)、sin(x)、cos(x)、abs(x)。如果 F_ext 出现在可用变量中，它表示本实验已知的外部控制量。
2. 不要使用未列出的变量名，不要使用等号，不要输出 Python 赋值语句。
3. 优先提出低复杂度、可解释的表达式，例如变量乘积、平方项与导数量的组合。
4. 如果实验是恒定外力场景，可以提出一个可能等于 F_ext 的表达式；否则提出可能近似不随时间变化的表达式。
5. 只提出一个最值得验证的表达式。
6. 不要要求调用符号回归、不变量搜索、枚举器或 PySR；这里必须由你直接提出公式。
7. 不要把明显随时间单调变化的单个原始变量当作候选规律；如果候选评价很差，需要继续定义新量或重新分析。

返回 JSON：
{{
  "expression": "...",
  "output_name": "llm_candidate",
  "rationale": "...",
  "expected_relationship": "constant 或 external_force"
}}
"""
        payload = self._request_json(system_prompt=system_prompt, user_prompt=user_prompt)
        return {
            "expression": str(payload.get("expression", "")).strip(),
            "output_name": str(payload.get("output_name", "llm_candidate")).strip() or "llm_candidate",
            "rationale": str(payload.get("rationale", "LLM 基于当前特征量提出候选表达式。")),
            "expected_relationship": str(payload.get("expected_relationship", "constant")),
            "raw_payload": payload,
        }

    def _request_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = None
        try:
            content = response.choices[0].message.content
        except Exception:
            pass

        # DeepSeek 等模型可能把内容放在 reasoning_content 中
        if not content:
            try:
                reasoning = response.choices[0].message.reasoning_content
                if reasoning:
                    # 提取 JSON 部分：从 ```json 开始到 ``` 结束
                    import re
                    json_match = re.search(r'```json\s*(.*?)\s*```', reasoning, re.DOTALL)
                    if json_match:
                        content = json_match.group(1).strip()
                    else:
                        # 如果没有 ```json 标记，尝试直接解析整个 reasoning
                        content = reasoning.strip()
            except Exception:
                pass

        if not content:
            try:
                content = response.choices[0].text
            except Exception:
                pass

        if not content:
            raise RuntimeError(
                "LLM 返回空响应。请检查 API 配置和模型响应格式。"
                f" 当前 response: {response!r}"
            )

        if isinstance(content, bytes):
            content = content.decode("utf-8")

        if isinstance(content, str):
            content = content.strip()

        if not content:
            raise RuntimeError("LLM 返回空响应。JSON 内容为空。")

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            extracted_payload = self._extract_json_object(content)
            if extracted_payload is not None:
                return extracted_payload
            raise RuntimeError(
                "LLM 返回的文本无法解析为 JSON。请检查模型提示和 response_format。"
                f" 内容: {content!r}"
            ) from exc

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        """从混入解释文字的 LLM 响应中提取第一个完整 JSON 对象。

        DeepSeek 等模型有时即使设置了 JSON mode，也会返回：
        "一些说明文字 ... response{...}"。这里用括号配平而不是简单
        正则，避免 JSON 内部字符串里的花括号干扰解析。
        """
        fenced_candidates = self._extract_fenced_json_candidates(text)
        for candidate in fenced_candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

        for start_index, char in enumerate(text):
            if char != "{":
                continue

            depth = 0
            in_string = False
            escape_next = False

            for end_index in range(start_index, len(text)):
                current = text[end_index]

                if escape_next:
                    escape_next = False
                    continue

                if current == "\\":
                    escape_next = True
                    continue

                if current == '"':
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if current == "{":
                    depth += 1
                elif current == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start_index : end_index + 1]
                        try:
                            payload = json.loads(candidate)
                        except json.JSONDecodeError:
                            break
                        if isinstance(payload, dict):
                            return payload
                        break

        return None

    def _extract_fenced_json_candidates(self, text: str) -> list[str]:
        import re

        return [
            match.group(1).strip()
            for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        ]


class ScientistAgent:
    """多轮 ReAct 科学发现 Agent。"""

    def __init__(
        self,
        universe: VirtualUniverse,
        data_tool: DataProcessingTool | None = None,
        verification_engine: VerificationEngine | None = None,
        brain: HypothesisBrain | None = None,
        reporter: ScientificReporter | None = None,
    ) -> None:
        self.universe = universe
        self.data_tool = data_tool or DataProcessingTool()
        # VerificationEngine 仍保留给手动对照实验使用；自主 Agent 的默认动作菜单
        # 不再暴露不变量枚举，而是让 LLM 基于关系分析主动提出公式。
        self.verification_engine = verification_engine or VerificationEngine(
            niterations=150,
            population_size=50,
            maxsize=30,
        )
        self.brain = brain or HypothesisBrain()
        self.reporter = reporter or ScientificReporter()
        self._experiment_counter = 0

    def run_scientific_cycle(
        self,
        report_dir: str | Path | None = None,
        max_steps: int = 20,
        goal: str | None = None,
        progress_callback: Callable[[ActionRecord], None] | None = None,
    ) -> ScientificCycleResult:
        """执行一次真正由 LLM 规划的多轮科学发现循环。"""
        notebook = ScientificNotebook()
        research_goal = goal or (
            "你只能从时间-位置观测出发，逐步探索这个虚拟宇宙中的运动规律。"
            "这个宇宙是人工构造的，可能不服从经典牛顿力学。"
            "不要预设某一物理量一定重要，但如果你认为需要，可以自行构造。"
            "当你发现一个候选规律后，应尽量在不同实验条件下复验它。"
            "不要仅凭某个熟悉的轨迹形状就套用已有理论名称。"
        )

        for step_index in range(1, max_steps + 1):
            notebook_summary = self.brain.summarize_notebook(
                notebook=notebook,
                goal=research_goal,
                max_steps=max_steps,
            )
            decision = self.brain.plan_next_action(
                notebook_summary=notebook_summary,
                max_steps=max_steps,
            )

            try:
                observation, should_finish = self._execute_action(
                    notebook=notebook,
                    step_index=step_index,
                    decision=decision,
                )
            except Exception as exc:
                observation = (
                    f"动作执行失败: {exc}. "
                    "请基于这个失败反馈重新规划，必要时先生成缺失序列或选择已有变量。"
                )
                should_finish = False

            action_record = ActionRecord(
                step_index=step_index,
                decision=decision,
                observation=observation,
            )
            notebook.add_action_record(action_record)
            if progress_callback is not None:
                progress_callback(action_record)
            if should_finish:
                break

        final_summary = self.brain.summarize_notebook(
            notebook=notebook,
            goal=research_goal,
            max_steps=max_steps,
        )
        final_law = self.brain.synthesize_final_law(final_summary)
        cycle_result = ScientificCycleResult(
            notebook=notebook,
            final_law=final_law,
        )

        report_markdown = self.reporter.generate_markdown(cycle_result)
        cycle_result.report_markdown = report_markdown

        if report_dir is not None:
            report_path = self.reporter.save_report(
                markdown_text=report_markdown,
                output_dir=report_dir,
            )
            cycle_result.report_path = str(report_path)

        return cycle_result

    def _execute_action(
        self,
        notebook: ScientificNotebook,
        step_index: int,
        decision: ActionDecision,
    ) -> tuple[str, bool]:
        action = decision.action
        params = decision.parameters

        if action == "run_experiment":
            return self._action_run_experiment(notebook, params), False
        if action == "smooth_series":
            return self._action_smooth_series(notebook, params), False
        if action == "estimate_kinematics":
            return self._action_estimate_kinematics(notebook, params), False
        if action == "differentiate_series":
            return self._action_differentiate_series(notebook, params), False
        if action == "inspect_series":
            return self._action_inspect_series(notebook, params), False
        if action == "inspect_relationships":
            return self._action_inspect_relationships(notebook, params), False
        if action == "define_derived_quantity":
            return self._action_define_derived_quantity(notebook, params), False
        if action == "fit_relationship_model":
            return self._action_fit_relationship_model(notebook, params), False
        if action == "test_candidate_expression":
            return self._action_test_candidate_expression(notebook, params), False
        if action == "propose_candidate_expression":
            return self._action_propose_candidate_expression(notebook, step_index, params), False
        if action == "search_invariants":
            return (
                "search_invariants 已从自主研究流程中禁用。请像科学家一样先用 "
                "inspect_relationships 分析变量关系，必要时用 define_derived_quantity "
                "定义新物理量，再用 propose_candidate_expression 自己提出候选公式。",
                False,
            )
        if action == "cross_experiment_check":
            return self._action_cross_experiment_check(notebook, params), False
        if action == "rank_candidate_laws":
            return self._action_rank_candidate_laws(notebook), False
        if action == "finalize_law":
            finalize_observation, can_finish = self._action_finalize_law_guard(notebook)
            return finalize_observation, can_finish

        return f"未知动作 `{action}`，已忽略。", False

    def _action_run_experiment(
        self,
        notebook: ScientificNotebook,
        params: dict[str, Any],
    ) -> str:
        self._experiment_counter += 1
        experiment_id = f"exp_{self._experiment_counter:02d}"

        force_field_type = self._resolve_force_field_type(params.get("force_field_type", "constant"))

        config = ExperimentConfig(
            initial_q=self._coerce_float(params.get("initial_q"), 0.0),
            initial_v=self._coerce_float(params.get("initial_v"), 1.0),
            force_field_type=force_field_type,
            t_span=(0.0, self._coerce_float(params.get("t_end"), 5.0)),
            dt=self._coerce_float(params.get("dt"), 0.05),
            constant_force=self._coerce_float(params.get("constant_force"), 10.0),
            noise_std=self._coerce_float(params.get("noise_std"), 0.0),
        )
        result = self.universe.run_experiment(config)
        record = ExperimentRecord(
            experiment_id=experiment_id,
            config=config,
            result=result,
            summary=self._summarize_trajectory(result),
        )
        notebook.register_experiment(record)
        force_text = (
            f"F_ext={config.constant_force}"
            if config.force_field_type is ForceFieldType.CONSTANT
            else "F_ext=无外力场，constant_force 参数已忽略"
        )
        return (
            f"完成实验 {experiment_id}。场景={config.force_field_type.value}，"
            f"{force_text}，q 范围 [{np.min(result.q):.6f}, {np.max(result.q):.6f}]。"
        )

    def _action_smooth_series(
        self,
        notebook: ScientificNotebook,
        params: dict[str, Any],
    ) -> str:
        experiment_id = self._resolve_experiment_id(notebook, params.get("experiment_id"))
        source_series = str(params["source_series"])
        output_name = self._normalize_output_name(
            params.get("output_name", f"{source_series}_smooth"),
            f"{source_series}_smooth",
        )
        if not self._coerce_bool(params.get("overwrite"), False):
            output_name = self._make_unique_series_name(notebook, experiment_id, output_name)

        t = notebook.get_series_values(experiment_id, "t")
        values = notebook.get_series_values(experiment_id, source_series)
        smoothed = self.data_tool.smooth_series(t=t, values=values)
        summary = self.data_tool.summarize_series(t=t, values=smoothed, name=output_name).to_text()
        notebook.register_series(
            DerivedSeries(
                experiment_id=experiment_id,
                name=output_name,
                values=smoothed,
                source_name=source_series,
                provenance="Savitzky-Golay smoothing",
                summary_text=summary,
            )
        )
        return f"已对 `{source_series}` 平滑，生成 `{output_name}`。{summary}"

    def _action_estimate_kinematics(
        self,
        notebook: ScientificNotebook,
        params: dict[str, Any],
    ) -> str:
        experiment_id = self._resolve_experiment_id(notebook, params.get("experiment_id"))
        source_series = str(params.get("source_series", "q"))
        overwrite = self._coerce_bool(params.get("overwrite"), False)

        position_name = self._normalize_output_name(
            params.get("position_name", "q_smooth"),
            default="q_smooth",
        )
        velocity_name = self._normalize_output_name(
            params.get("velocity_name", "v"),
            default="v",
        )
        acceleration_name = self._normalize_output_name(
            params.get("acceleration_name", "a"),
            default="a",
        )
        if not overwrite:
            position_name = self._make_unique_series_name(notebook, experiment_id, position_name)
            velocity_name = self._make_unique_series_name(notebook, experiment_id, velocity_name)
            acceleration_name = self._make_unique_series_name(notebook, experiment_id, acceleration_name)

        window_length = params.get("window_length")
        polyorder = params.get("polyorder")
        t = notebook.get_series_values(experiment_id, "t")
        q_values = notebook.get_series_values(experiment_id, source_series)
        estimates = self.data_tool.estimate_kinematics(
            t=t,
            q=q_values,
            window_length=self._coerce_int(window_length, self.data_tool.window_length) if window_length else None,
            polyorder=self._coerce_int(polyorder, self.data_tool.polyorder) if polyorder else None,
        )

        output_map = {
            position_name: ("smoothed position", estimates["q_smooth"]),
            velocity_name: ("first derivative", estimates["v"]),
            acceleration_name: ("second derivative", estimates["a"]),
        }
        summaries: list[str] = []
        for output_name, (quantity_text, values) in output_map.items():
            summary = self.data_tool.summarize_series(t=t, values=values, name=output_name)
            notebook.register_series(
                DerivedSeries(
                    experiment_id=experiment_id,
                    name=output_name,
                    values=values,
                    source_name=source_series,
                    provenance=f"Savitzky-Golay kinematics estimation ({quantity_text})",
                    summary_text=summary.to_text(),
                )
            )
            summaries.append(summary.to_text())

        return (
            f"已从 `{source_series}` 同时估计运动学序列："
            f"`{position_name}`, `{velocity_name}`, `{acceleration_name}`。"
            + " | ".join(summaries)
        )

    def _action_differentiate_series(
        self,
        notebook: ScientificNotebook,
        params: dict[str, Any],
    ) -> str:
        experiment_id = self._resolve_experiment_id(notebook, params.get("experiment_id"))
        source_series = str(params["source_series"])
        order = self._coerce_int(params.get("order"), 1)
        output_name = self._normalize_output_name(
            params.get("output_name", f"d{source_series}_order_{order}"),
            f"d{source_series}_order_{order}",
        )
        if not self._coerce_bool(params.get("overwrite"), False):
            output_name = self._make_unique_series_name(notebook, experiment_id, output_name)
        smooth_before = self._coerce_bool(params.get("smooth_before"), True)
        smooth_after = self._coerce_bool(params.get("smooth_after"), True)

        t = notebook.get_series_values(experiment_id, "t")
        values = notebook.get_series_values(experiment_id, source_series)
        differentiated = self.data_tool.differentiate_series(
            t=t,
            values=values,
            order=order,
            smooth_before=smooth_before,
            smooth_after=smooth_after,
        )
        summary = self.data_tool.summarize_series(t=t, values=differentiated, name=output_name).to_text()
        notebook.register_series(
            DerivedSeries(
                experiment_id=experiment_id,
                name=output_name,
                values=differentiated,
                source_name=source_series,
                provenance=f"{order} order differentiation",
                summary_text=summary,
            )
        )
        return f"已对 `{source_series}` 做 {order} 阶差分，生成 `{output_name}`。{summary}"

    def _action_inspect_series(
        self,
        notebook: ScientificNotebook,
        params: dict[str, Any],
    ) -> str:
        experiment_id = self._resolve_experiment_id(notebook, params.get("experiment_id"))
        series_names = [str(name) for name in params.get("series_names", notebook.available_series(experiment_id))]
        t = notebook.get_series_values(experiment_id, "t")
        parts: list[str] = []
        for series_name in series_names:
            values = notebook.get_series_values(experiment_id, series_name)
            parts.append(self.data_tool.summarize_series(t=t, values=values, name=series_name).to_text())

        if len(series_names) >= 2:
            left = notebook.get_series_values(experiment_id, series_names[0])
            right = notebook.get_series_values(experiment_id, series_names[1])
            score = self.data_tool.compute_relationship_score(left, right)
            parts.append(
                f"{series_names[0]} vs {series_names[1]}: "
                f"correlation={score['correlation']:.6f}, mse={score['mse']:.6f}"
            )

        return " | ".join(parts)

    def _action_inspect_relationships(
        self,
        notebook: ScientificNotebook,
        params: dict[str, Any],
    ) -> str:
        experiment_id = self._resolve_experiment_id(notebook, params.get("experiment_id"))
        x_name = params.get("x_series")
        y_name = params.get("y_series")

        # Backward-compatible parsing for older LLM outputs, but the public prompt
        # now asks for x_series/y_series only.
        if x_name is None or y_name is None:
            feature_series = [str(name) for name in params.get("feature_series", [])]
            target_series = params.get("target_series")
            if x_name is None and feature_series:
                x_name = feature_series[0]
            if y_name is None and target_series is not None:
                y_name = target_series
            elif y_name is None and len(feature_series) >= 2:
                y_name = feature_series[1]

        if x_name is None or y_name is None:
            raise ValueError("inspect_relationships 需要 x_series 和 y_series 两个序列名。")

        x_name = str(x_name)
        y_name = str(y_name)
        if x_name == y_name:
            raise ValueError("inspect_relationships 需要两个不同序列。")

        t = notebook.get_series_values(experiment_id, "t")
        x_values = notebook.get_series_values(experiment_id, x_name)
        y_values = notebook.get_series_values(experiment_id, y_name)

        figure_path = self._generate_relationship_figure(
            experiment_id=experiment_id,
            t=t,
            x_name=x_name,
            x_values=x_values,
            y_name=y_name,
            y_values=y_values,
        )
        x_summary = self.data_tool.summarize_series(t=t, values=x_values, name=x_name).to_text()
        y_summary = self.data_tool.summarize_series(t=t, values=y_values, name=y_name).to_text()
        corr = self._safe_correlation(x_values, y_values)
        observation = self._basic_relationship_observation(
            x_name=x_name,
            x_values=x_values,
            y_name=y_name,
            y_values=y_values,
            correlation=corr,
        )
        summary_text = (
            f"关系观察 {experiment_id}: 只比较 `{x_name}` 与 `{y_name}`。"
            f"关系图={figure_path.resolve()}。"
            f"{x_summary} | {y_summary} | "
            f"Pearson correlation={corr:.6f}。"
            f"中性观察: {observation}"
        )
        notebook.notes.append(summary_text)
        return summary_text

    def _action_define_derived_quantity(
        self,
        notebook: ScientificNotebook,
        params: dict[str, Any],
    ) -> str:
        experiment_id = self._resolve_experiment_id(notebook, params.get("experiment_id"))
        raw_symbol = params.get("symbol") or params.get("output_name") or params.get("name")
        if raw_symbol is None:
            raise ValueError("define_derived_quantity 需要 symbol 参数。")

        symbol = self._normalize_output_name(raw_symbol, default="derived_quantity")
        if symbol in {"q", "t"}:
            raise ValueError("define_derived_quantity 不能覆盖原始序列 `q` 或 `t`。")

        overwrite = self._coerce_bool(params.get("overwrite"), False)
        if symbol in notebook.derived_series.get(experiment_id, {}) and not overwrite:
            raise ValueError(
                f"实验 {experiment_id} 中已存在派生物理量 `{symbol}`。"
                "如需替换，请设置 overwrite=true。"
            )

        expression = str(params.get("expression", "")).strip()
        if not expression:
            raise ValueError("define_derived_quantity 需要 expression 参数。")

        description = str(params.get("description", "LLM 定义的新物理量。"))
        values = self._evaluate_expression(notebook, experiment_id, expression)
        t = notebook.get_series_values(experiment_id, "t")
        summary = self.data_tool.summarize_series(t=t, values=values, name=symbol)
        notebook.register_series(
            DerivedSeries(
                experiment_id=experiment_id,
                name=symbol,
                values=values,
                source_name=expression,
                provenance=f"LLM-defined derived physical quantity: {description}",
                summary_text=summary.to_text(),
            )
        )
        return (
            f"已定义新物理量 `{symbol}` = `{expression}`。"
            f"{summary.to_text()}。说明：{description}。"
            f"后续可在 inspect_relationships、test_candidate_expression、"
            f"propose_candidate_expression 中直接引用 `{symbol}`。"
        )

    def _action_fit_relationship_model(
        self,
        notebook: ScientificNotebook,
        params: dict[str, Any],
    ) -> str:
        experiment_id = self._resolve_experiment_id(notebook, params.get("experiment_id"))
        target_series = str(params["target_series"])
        raw_basis_expressions = [
            str(item).strip()
            for item in params.get("basis_expressions", [])
            if str(item).strip()
        ]
        include_intercept = self._coerce_bool(params.get("include_intercept"), True)
        basis_expressions: list[str] = []
        skipped_terms: list[str] = []
        for expression in raw_basis_expressions:
            normalized_expression = expression.replace(" ", "")
            if include_intercept and normalized_expression in {"1", "1.0"}:
                skipped_terms.append(expression)
                continue
            if expression not in basis_expressions:
                basis_expressions.append(expression)
        if not basis_expressions:
            if include_intercept:
                raise ValueError("fit_relationship_model 除截距外还需要至少一个非平凡 basis_expressions。")
            raise ValueError("fit_relationship_model 需要至少一个 basis_expressions。")

        prediction_name = self._make_unique_series_name(
            notebook,
            experiment_id,
            self._normalize_output_name(params.get("prediction_name", "fit_prediction"), "fit_prediction"),
        )
        residual_name = self._make_unique_series_name(
            notebook,
            experiment_id,
            self._normalize_output_name(params.get("residual_name", "fit_residual"), "fit_residual"),
        )

        target_values = notebook.get_series_values(experiment_id, target_series)
        basis_values = [
            self._evaluate_expression(notebook, experiment_id, expression)
            for expression in basis_expressions
        ]
        columns: list[np.ndarray] = []
        term_names: list[str] = []
        if include_intercept:
            columns.append(np.ones_like(target_values, dtype=float))
            term_names.append("1")
        columns.extend(basis_values)
        term_names.extend(basis_expressions)

        design = np.column_stack(columns)
        finite_mask = np.isfinite(target_values) & np.all(np.isfinite(design), axis=1)
        if finite_mask.sum() < max(5, design.shape[1] + 2):
            raise ValueError("有限样本点过少，无法稳定拟合关系模型。")

        coeffs, *_ = np.linalg.lstsq(design[finite_mask], target_values[finite_mask], rcond=None)
        prediction = design @ coeffs
        residual = target_values - prediction

        finite_target = target_values[finite_mask]
        finite_prediction = prediction[finite_mask]
        finite_residual = residual[finite_mask]
        sst = float(np.sum((finite_target - np.mean(finite_target)) ** 2))
        r2 = 0.0 if sst <= 1e-12 else 1.0 - float(np.sum(finite_residual * finite_residual)) / sst
        rmse = float(np.sqrt(np.mean(finite_residual * finite_residual)))
        mae = float(np.mean(np.abs(finite_residual)))

        t = notebook.get_series_values(experiment_id, "t")
        prediction_summary = self.data_tool.summarize_series(
            t=t,
            values=prediction,
            name=prediction_name,
        )
        residual_summary = self.data_tool.summarize_series(
            t=t,
            values=residual,
            name=residual_name,
        )
        equation_terms = [
            f"{coefficient:.8g}*{term_name}"
            for coefficient, term_name in zip(coeffs, term_names)
        ]
        equation_text = f"{target_series} ≈ " + " + ".join(equation_terms)

        notebook.register_series(
            DerivedSeries(
                experiment_id=experiment_id,
                name=prediction_name,
                values=prediction,
                source_name=equation_text,
                provenance="least-squares relationship model prediction",
                summary_text=prediction_summary.to_text(),
            )
        )
        notebook.register_series(
            DerivedSeries(
                experiment_id=experiment_id,
                name=residual_name,
                values=residual,
                source_name=f"{target_series} - ({equation_text})",
                provenance="least-squares relationship model residual",
                summary_text=residual_summary.to_text(),
            )
        )
        notebook.notes.append(
            f"实验 {experiment_id}: 拟合关系模型 `{equation_text}`，R2={r2:.6f}, "
            f"RMSE={rmse:.6f}, MAE={mae:.6f}。"
        )
        skipped_text = ""
        if skipped_terms:
            skipped_text = f" 已忽略与截距重复的常数基函数: {skipped_terms}。"
        return (
            f"关系模型拟合完成：`{equation_text}`。"
            f"R2={r2:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}。"
            f"已生成 `{prediction_name}` 和 `{residual_name}`。{skipped_text}"
            f"{prediction_summary.to_text()} | {residual_summary.to_text()}"
        )

    def _action_test_candidate_expression(
        self,
        notebook: ScientificNotebook,
        params: dict[str, Any],
        ) -> str:
        experiment_id = self._resolve_experiment_id(notebook, params.get("experiment_id"))
        expression = str(params["expression"])
        output_name = self._make_unique_series_name(
            notebook,
            experiment_id,
            self._normalize_output_name(
                params.get("output_name", "candidate_expression"),
                "candidate_expression",
            ),
        )

        evaluated = self._evaluate_expression(notebook, experiment_id, expression)
        t = notebook.get_series_values(experiment_id, "t")
        summary = self.data_tool.summarize_series(t=t, values=evaluated, name=output_name)
        notebook.register_series(
            DerivedSeries(
                experiment_id=experiment_id,
                name=output_name,
                values=evaluated,
                source_name=expression,
                provenance="candidate expression evaluation",
                summary_text=summary.to_text(),
            )
        )
        constancy_score = float(summary.std / (abs(summary.mean) + 1e-8))
        return (
            f"表达式 `{expression}` 已求值为 `{output_name}`。{summary.to_text()}。"
            f"相对波动系数约为 {constancy_score:.6f}，越小表示越接近常数。"
        )

    def _action_propose_candidate_expression(
        self,
        notebook: ScientificNotebook,
        step_index: int,
        params: dict[str, Any],
    ) -> str:
        experiment_id = self._resolve_experiment_id(notebook, params.get("experiment_id"))
        requested_series = params.get("feature_series")
        if requested_series:
            feature_series = [str(name) for name in requested_series]
        else:
            feature_series = notebook.available_series(experiment_id)

        if not feature_series:
            raise ValueError("propose_candidate_expression 需要至少一个 feature_series。")

        record = notebook.experiments[experiment_id]
        t = notebook.get_series_values(experiment_id, "t")
        feature_summaries: list[str] = []
        for series_name in feature_series:
            values = notebook.get_series_values(experiment_id, series_name)
            summary = self.data_tool.summarize_series(t=t, values=values, name=series_name)
            feature_summaries.append(f"- {summary.to_text()}")
        proposal_feature_series = list(feature_series)
        if record.config.force_field_type is ForceFieldType.CONSTANT:
            proposal_feature_series.append("F_ext")
            feature_summaries.append(
                f"- F_ext: known constant experimental control = {float(record.config.constant_force):.6f}"
            )

        force_text = (
            str(record.config.constant_force)
            if record.config.force_field_type is ForceFieldType.CONSTANT
            else "N/A（free 场景无外力，constant_force 参数被忽略）"
        )
        experiment_context = (
            f"实验 {experiment_id}: 场景={record.config.force_field_type.value}, "
            f"F_ext={force_text}, q0={record.config.initial_q}, v0={record.config.initial_v}, "
            f"t_span={record.config.t_span}, dt={record.config.dt}。"
        )
        notebook_summary = self.brain.summarize_notebook(
            notebook=notebook,
            goal="基于当前实验特征量提出一个候选动力学表达式，并由程序验证。",
            max_steps=max(step_index, len(notebook.action_history) + 1),
        )

        proposal = self.brain.propose_candidate_expression(
            notebook_summary=notebook_summary,
            experiment_context=experiment_context,
            feature_summaries=feature_summaries,
            feature_series=proposal_feature_series,
        )
        expression = str(proposal["expression"]).strip()
        if not expression:
            raise ValueError("LLM 未返回可验证的 expression。")

        output_name = self._make_unique_series_name(
            notebook,
            experiment_id,
            self._normalize_output_name(
                params.get("output_name") or proposal.get("output_name"),
                default="llm_candidate",
            ),
        )
        evaluated = self._evaluate_expression(notebook, experiment_id, expression)
        summary = self.data_tool.summarize_series(t=t, values=evaluated, name=output_name)
        notebook.register_series(
            DerivedSeries(
                experiment_id=experiment_id,
                name=output_name,
                values=evaluated,
                source_name=expression,
                provenance="LLM candidate expression proposal",
                summary_text=summary.to_text(),
            )
        )

        constancy_score = float(summary.std / (abs(summary.mean) + 1e-8))
        score = constancy_score
        expected_relationship = str(proposal.get("expected_relationship", "constant"))
        force_residual_text = ""
        if record.config.force_field_type is ForceFieldType.CONSTANT:
            force_residual = float(summary.mean - float(record.config.constant_force))
            force_residual_text = f"，与 F_ext 的均值残差为 {force_residual:.6f}"
            if expected_relationship.strip().lower() in {"external_force", "force", "f_ext"}:
                score = abs(force_residual) + constancy_score

        rationale = str(proposal.get("rationale", "LLM 基于当前特征量提出候选表达式。"))
        acceptance_threshold = self._coerce_float(params.get("acceptance_threshold"), 0.08)
        if score <= acceptance_threshold:
            notebook.add_candidate_law(
                CandidateLaw(
                    expression=expression,
                    source_experiment_id=experiment_id,
                    score=score,
                    origin="propose_candidate_expression",
                    notes=(
                        f"features={feature_series}; output_name={output_name}; "
                        f"expected_relationship={expected_relationship}; rationale={rationale}"
                    ),
                )
            )
            candidate_text = "已登记为候选规律"
        else:
            notebook.notes.append(
                f"实验 {experiment_id}: LLM 提出的 `{expression}` 未登记为候选规律，"
                f"score={score:.6f} 高于阈值 {acceptance_threshold:.6f}。"
            )
            candidate_text = (
                f"未登记为候选规律，因为 score={score:.6f} 高于阈值 "
                f"{acceptance_threshold:.6f}"
            )
        return (
            f"LLM 提出候选表达式 `{expression}`，并已求值为 `{output_name}`。"
            f"{summary.to_text()}。相对波动系数约为 {constancy_score:.6f}"
            f"{force_residual_text}。{candidate_text}。LLM 理由：{rationale}"
        )

    def _action_search_invariants(
        self,
        notebook: ScientificNotebook,
        params: dict[str, Any],
    ) -> str:
        experiment_id = self._resolve_experiment_id(notebook, params.get("experiment_id"))
        feature_series = [str(name) for name in params.get("feature_series", [])]

        if not feature_series:
            raise ValueError("search_invariants 需要至少一个 feature_series。")

        X = pd.DataFrame(
            {feature_name: notebook.get_series_values(experiment_id, feature_name) for feature_name in feature_series}
        )

        binary_operators = params.get("binary_operators")
        unary_operators = params.get("unary_operators")
        if binary_operators:
            self.verification_engine.binary_operators = [str(op) for op in binary_operators]
        if unary_operators:
            self.verification_engine.unary_operators = [str(op) for op in unary_operators]

        invariant = self.verification_engine.search_invariant(
            X=X,
            variable_names=feature_series,
        )
        notebook.add_invariant(invariant, experiment_id=experiment_id, feature_series=feature_series)
        return (
            f"不变量搜索完成。候选表达式 `{invariant.equation}`，"
            f"score={invariant.score:.12f}，residual_std={invariant.residual_std:.12f}，"
            f"complexity={invariant.complexity}。"
        )

    def _action_cross_experiment_check(
        self,
        notebook: ScientificNotebook,
        params: dict[str, Any],
    ) -> str:
        expression = str(params["expression"])
        metric_name = str(params.get("metric_name", "relative_std"))
        requested_ids = params.get("experiment_ids")

        if requested_ids:
            experiment_ids = [self._resolve_experiment_id(notebook, item) for item in requested_ids]
        else:
            experiment_ids = sorted(notebook.experiments.keys())

        if len(experiment_ids) < 2:
            raise ValueError("cross_experiment_check 至少需要两个实验。")

        metric_values: dict[str, float] = {}
        for experiment_id in experiment_ids:
            values = self._evaluate_expression(notebook, experiment_id, expression)
            finite_values = values[np.isfinite(values)]
            if len(finite_values) < 3:
                raise ValueError(
                    f"表达式 `{expression}` 在 {experiment_id} 中有限数值点少于 3 个，无法跨实验验证。"
                )
            mean_value = float(np.mean(finite_values))
            std_value = float(np.std(finite_values))

            if metric_name == "relative_std":
                metric = std_value / (abs(mean_value) + 1e-8)
            elif metric_name == "mean_value":
                metric = mean_value
            elif metric_name == "force_residual":
                record = notebook.experiments[experiment_id]
                if record.config.force_field_type is not ForceFieldType.CONSTANT:
                    raise ValueError(
                        f"force_residual 只适用于 constant 场景，{experiment_id} 当前为 "
                        f"{record.config.force_field_type.value}。"
                    )
                metric = mean_value - float(record.config.constant_force)
            else:
                raise ValueError(f"不支持的 metric_name: {metric_name}")

            metric_values[experiment_id] = float(metric)

        if metric_name == "relative_std":
            aggregate_score = float(np.mean(list(metric_values.values())))
            summary_text = (
                f"跨实验验证表达式 `{expression}` 的相对波动："
                + ", ".join(f"{exp}={value:.6f}" for exp, value in metric_values.items())
                + f"。平均相对波动={aggregate_score:.6f}，越小越稳定。"
            )
        elif metric_name == "force_residual":
            aggregate_score = float(np.mean([abs(value) for value in metric_values.values()]))
            summary_text = (
                f"跨实验验证表达式 `{expression}` 与外力 F_ext 的残差："
                + ", ".join(f"{exp}={value:.6f}" for exp, value in metric_values.items())
                + f"。平均绝对残差={aggregate_score:.6f}，越小越接近动力学方程。"
            )
        else:
            aggregate_score = float(np.std(list(metric_values.values())))
            summary_text = (
                f"跨实验验证表达式 `{expression}` 的均值："
                + ", ".join(f"{exp}={value:.6f}" for exp, value in metric_values.items())
                + f"。不同实验之间均值标准差={aggregate_score:.6f}，越小越一致。"
            )

        notebook.add_generalization_check(
            GeneralizationCheck(
                expression=expression,
                experiment_ids=experiment_ids,
                metric_name=metric_name,
                metric_values=metric_values,
                aggregate_score=aggregate_score,
                summary_text=summary_text,
            )
        )
        return summary_text

    def _action_rank_candidate_laws(self, notebook: ScientificNotebook) -> str:
        """对当前候选规律进行粗略排序。"""
        ranking_items: list[tuple[str, float, str]] = []

        for candidate in notebook.candidate_laws:
            ranking_items.append((candidate.expression, candidate.score, candidate.origin))

        for check in notebook.generalization_checks:
            ranking_items.append((check.expression, check.aggregate_score, "cross_experiment_check"))

        if not ranking_items:
            return "当前还没有任何可排序的候选规律。"

        ranking_items.sort(key=lambda item: item[1])
        summary = "候选规律排序（分数越小越好）: " + "; ".join(
            f"{expression} | score={score:.6f} | source={origin}"
            for expression, score, origin in ranking_items[:5]
        )
        notebook.notes.append(summary)
        return summary

    def _action_finalize_law_guard(
        self,
        notebook: ScientificNotebook,
    ) -> tuple[str, bool]:
        """防止 LLM 在证据不足时过早结束。"""
        if not notebook.candidate_laws:
            return (
                "当前禁止结束：尚未形成任何候选规律。请先用 inspect_relationships "
                "分析变量关系，必要时通过 define_derived_quantity 定义新物理量，"
                "再通过 propose_candidate_expression 让 LLM 自己提出候选公式，而不是直接总结定律。",
                False,
            )
        if not notebook.generalization_checks:
            return (
                "当前禁止结束：尚未执行任何跨实验验证。请先使用 cross_experiment_check "
                "检查候选表达式在不同实验条件下是否稳定。",
                False,
            )

        return "LLM 认为当前证据已足够进入规律总结阶段。", True

    def _safe_correlation(self, left: np.ndarray, right: np.ndarray) -> float:
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        finite_mask = np.isfinite(left_array) & np.isfinite(right_array)
        if finite_mask.sum() < 3:
            return 0.0
        left_finite = left_array[finite_mask]
        right_finite = right_array[finite_mask]
        if np.std(left_finite) <= 1e-12 or np.std(right_finite) <= 1e-12:
            return 0.0
        return float(np.corrcoef(left_finite, right_finite)[0, 1])

    def _generate_relationship_figure(
        self,
        experiment_id: str,
        t: np.ndarray,
        x_name: str,
        x_values: np.ndarray,
        y_name: str,
        y_values: np.ndarray,
    ) -> Path:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path.cwd() / "relationship_assets"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{self._sanitize_filename_part(experiment_id)}_"
            f"{self._sanitize_filename_part(x_name)}_vs_"
            f"{self._sanitize_filename_part(y_name)}.png"
        )
        figure_path = output_dir / filename

        t_array = np.asarray(t, dtype=float)
        x_array = np.asarray(x_values, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
        finite_mask = np.isfinite(t_array) & np.isfinite(x_array) & np.isfinite(y_array)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        axes[0].plot(t_array, self._normalize_for_plot(x_array), label=f"{x_name} (normalized)")
        axes[0].plot(t_array, self._normalize_for_plot(y_array), label=f"{y_name} (normalized)")
        axes[0].set_title(f"{experiment_id}: time traces")
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("normalized value")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        if finite_mask.sum() > 0:
            scatter = axes[1].scatter(
                x_array[finite_mask],
                y_array[finite_mask],
                c=t_array[finite_mask],
                cmap="viridis",
                s=18,
                alpha=0.85,
            )
            fig.colorbar(scatter, ax=axes[1], label="t")
        axes[1].set_title(f"{x_name} vs {y_name}")
        axes[1].set_xlabel(x_name)
        axes[1].set_ylabel(y_name)
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return figure_path

    def _basic_relationship_observation(
        self,
        x_name: str,
        x_values: np.ndarray,
        y_name: str,
        y_values: np.ndarray,
        correlation: float,
    ) -> str:
        x_trend = self._overall_trend_text(x_values)
        y_trend = self._overall_trend_text(y_values)
        if abs(correlation) >= 0.8:
            relation = "明显同向变化" if correlation > 0 else "明显反向变化"
        elif abs(correlation) >= 0.4:
            relation = "有一定同向变化" if correlation > 0 else "有一定反向变化"
        else:
            relation = "线性同/反向关系不明显"

        return (
            f"`{x_name}` 整体趋势为{x_trend}，`{y_name}` 整体趋势为{y_trend}；"
            f"散点层面表现为{relation}。这一步只提供两序列观察，不构造公式。"
        )

    def _overall_trend_text(self, values: np.ndarray) -> str:
        values_array = np.asarray(values, dtype=float)
        finite_values = values_array[np.isfinite(values_array)]
        if len(finite_values) < 3:
            return "样本不足"

        index = np.arange(len(finite_values), dtype=float)
        slope = float(np.polyfit(index, finite_values, deg=1)[0])
        scale = float(np.std(finite_values) + 1e-8)
        normalized_slope = slope * len(finite_values) / scale
        if normalized_slope > 0.5:
            return "上升"
        if normalized_slope < -0.5:
            return "下降"
        return "近似平稳或非单调"

    def _normalize_for_plot(self, values: np.ndarray) -> np.ndarray:
        values_array = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(values_array)
        normalized = np.full_like(values_array, np.nan, dtype=float)
        if finite_mask.sum() == 0:
            return normalized
        finite_values = values_array[finite_mask]
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        if abs(maximum - minimum) <= 1e-12:
            normalized[finite_mask] = 0.0
        else:
            normalized[finite_mask] = (finite_values - minimum) / (maximum - minimum)
        return normalized

    def _sanitize_filename_part(self, value: str) -> str:
        normalized = re.sub(r"[^0-9A-Za-z_]+", "_", str(value)).strip("_")
        return normalized or "series"

    def _evaluate_expression(
        self,
        notebook: ScientificNotebook,
        experiment_id: str,
        expression: str,
    ) -> np.ndarray:
        """安全地求值表达式，支持 q, v, t 等序列和 square/cube 函数。"""
        available_names: dict[str, np.ndarray] = {}
        try:
            for series_name in notebook.available_series(experiment_id):
                try:
                    available_names[series_name] = notebook.get_series_values(experiment_id, series_name)
                except Exception as e:
                    raise ValueError(f"无法获取序列 '{series_name}' 的值: {e}") from e

            reference_shape = available_names[list(available_names.keys())[0]].shape
            record = notebook.experiments[experiment_id]
            force_value = (
                float(record.config.constant_force)
                if record.config.force_field_type is ForceFieldType.CONSTANT
                else 0.0
            )
            available_names["F_ext"] = np.full(reference_shape, force_value, dtype=float)

            def square(x: np.ndarray) -> np.ndarray:
                return np.asarray(x, dtype=float) * np.asarray(x, dtype=float)

            def cube(x: np.ndarray) -> np.ndarray:
                x_arr = np.asarray(x, dtype=float)
                return x_arr * x_arr * x_arr

            def exp(x: np.ndarray) -> np.ndarray:
                return np.exp(np.clip(np.asarray(x, dtype=float), -60.0, 60.0))

            safe_globals = {"__builtins__": {}}
            safe_locals: dict[str, Any] = {
                **available_names,
                "square": square,
                "cube": cube,
                "sqrt": np.sqrt,
                "log": np.log,
                "exp": exp,
                "sin": np.sin,
                "cos": np.cos,
                "abs": np.abs,
                "np": np,
            }

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                evaluated = eval(expression, safe_globals, safe_locals)
            result = np.asarray(evaluated, dtype=float)

            if result.shape == ():
                result = np.full(reference_shape, float(result), dtype=float)

            if result.shape != reference_shape:
                raise ValueError(
                    f"表达式 '{expression}' 的结果形状 {result.shape} 与预期不符"
                )

            return self._sanitize_expression_values(result=result, expression=expression)
        except Exception as e:
            raise RuntimeError(
                f"表达式求值失败: '{expression}'\n"
                f"可用变量: {list(available_names.keys())}\n"
                f"错误: {e}"
            ) from e

    def _sanitize_expression_values(self, result: np.ndarray, expression: str) -> np.ndarray:
        """处理表达式中的少量奇异点，避免 0/0 或 x/0 污染后续统计。"""
        result = np.asarray(result, dtype=float)
        finite_mask = np.isfinite(result)
        finite_count = int(finite_mask.sum())

        if finite_count == len(result):
            return result

        if finite_count < max(3, int(0.8 * len(result))):
            invalid_count = len(result) - finite_count
            raise ValueError(
                f"表达式 `{expression}` 产生过多非有限值，共 {invalid_count}/{len(result)} 个。"
                "这通常说明候选公式含有不稳定除法，请换一个表达式或先避开接近零的变量。"
            )

        sanitized = result.copy()
        invalid_indices = np.flatnonzero(~finite_mask)
        finite_indices = np.flatnonzero(finite_mask)
        sanitized[invalid_indices] = np.interp(
            invalid_indices.astype(float),
            finite_indices.astype(float),
            result[finite_mask],
        )
        return sanitized

    def _summarize_trajectory(self, result: ExperimentResult) -> dict[str, float]:
        q = result.q
        t = result.t
        return {
            "num_points": float(len(t)),
            "q_min": float(np.min(q)),
            "q_max": float(np.max(q)),
            "mean_q": float(np.mean(q)),
        }

    def _resolve_experiment_id(
        self,
        notebook: ScientificNotebook,
        requested_experiment_id: Any,
    ) -> str:
        """允许 LLM 省略 experiment_id 时默认使用最近一次实验。"""
        if requested_experiment_id is not None:
            candidates = self._experiment_id_candidates(requested_experiment_id)
            for experiment_id in candidates:
                if experiment_id in notebook.experiments:
                    return experiment_id

            available = ", ".join(sorted(notebook.experiments.keys())) or "无"
            raise ValueError(
                f"未知实验 ID `{requested_experiment_id}`。可用实验 ID: {available}。"
                "如果想引用第 4 个实验，可使用 `exp_04` 或数字 `4`。"
            )

        latest = notebook.latest_experiment_id()
        if latest is None:
            raise ValueError("当前还没有任何实验，无法解析 experiment_id。")
        return latest

    def _experiment_id_candidates(self, requested_experiment_id: Any) -> list[str]:
        """把 4、'4'、'exp_4'、'exp_04' 都归一化为可匹配的实验 ID。"""
        raw = str(requested_experiment_id).strip()
        if not raw:
            return []

        candidates = [raw]
        match = re.fullmatch(r"(?:exp[_-]?)?0*(\d+)", raw, flags=re.IGNORECASE)
        if match:
            numeric_id = int(match.group(1))
            candidates.extend(
                [
                    f"exp_{numeric_id:02d}",
                    f"exp_{numeric_id}",
                    str(numeric_id),
                ]
            )

        deduplicated: list[str] = []
        for candidate in candidates:
            if candidate not in deduplicated:
                deduplicated.append(candidate)
        return deduplicated

    def _resolve_force_field_type(self, requested_force_field_type: Any) -> ForceFieldType:
        """兼容 LLM 常见的场景命名变体。"""
        normalized = str(requested_force_field_type).strip().lower().replace("-", "_").replace(" ", "_")

        constant_aliases = {
            "constant",
            "constant_force",
            "constant_field",
            "force",
            "forced",
            "恒力",
            "恒定外力",
            "恒定受力",
        }
        free_aliases = {
            "free",
            "none",
            "no_force",
            "zero_force",
            "no_external_force",
            "unforced",
            "无外力",
            "自由",
        }

        if normalized in constant_aliases:
            return ForceFieldType.CONSTANT
        if normalized in free_aliases:
            return ForceFieldType.FREE

        raise ValueError(
            f"未知 force_field_type `{requested_force_field_type}`。"
            "请使用 free/none/no_force 或 constant/constant_force。"
        )

    def _normalize_output_name(self, value: Any, default: str) -> str:
        """把 LLM 给出的序列名转成可在表达式里引用的 Python 标识符。"""
        raw = str(value or default).strip()
        normalized = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_")
        if not normalized:
            normalized = default
        if normalized[0].isdigit():
            normalized = f"{default}_{normalized}"
        return normalized

    def _make_unique_series_name(
        self,
        notebook: ScientificNotebook,
        experiment_id: str,
        desired_name: str,
    ) -> str:
        """生成不覆盖已有序列的名字，避免候选量把 a/v 等观测量冲掉。"""
        normalized = self._normalize_output_name(desired_name, default="series")
        existing = set(notebook.available_series(experiment_id))
        if normalized not in existing:
            return normalized

        index = 2
        while f"{normalized}_{index}" in existing:
            index += 1
        return f"{normalized}_{index}"

    def _coerce_float(self, value: Any, default: float) -> float:
        """把 LLM 生成的参数稳健转换为 float。"""
        if value is None or value == "":
            return default
        return float(value)

    def _coerce_int(self, value: Any, default: int) -> int:
        """把 LLM 生成的参数稳健转换为 int。"""
        if value is None or value == "":
            return default
        return int(value)

    def _coerce_bool(self, value: Any, default: bool) -> bool:
        """把 LLM 生成的参数稳健转换为 bool。"""
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        return bool(value)
