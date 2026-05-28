from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from .code_registry import CodeRegistry
from .code_runner import GeneratedCodeRunner, GeneratedProcessorResult
from .data_brain import DataProcessingBrain
from .hypothesis_registry import HypothesisRegistry
from .reporting import ScientificReporter
from .tool_specs import (
    DEFAULT_TOOL_SPECS,
    TOOL_SPECS_BY_NAME,
    ToolSpec,
    render_tool_descriptions,
    render_tool_name_list,
)
from .universe import ExperimentConfig, ExperimentResult, ForceFieldType, VirtualUniverse


LEDGER_MAINTENANCE_MODES = frozenset(
    {
        "maintain_ledger",
        "observe",
        "observation",
        "phenomenology",
        "phenomenological",
        "derive",
        "define_quantity",
    }
)
HYPOTHESIS_VALIDATION_MODES = frozenset(
    {"validate_hypothesis", "test_hypothesis", "hypothesis_validation", "model_compare"}
)
FAILURE_MARKERS = (
    "数据处理 LLM 路径失败",
    "动作执行失败",
    "自动修复重试仍失败",
)


def _normalize_analysis_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in HYPOTHESIS_VALIDATION_MODES:
        return "validate_hypothesis"
    return "maintain_ledger"


def _is_ledger_maintenance_mode(value: Any) -> bool:
    return _normalize_analysis_mode(value) == "maintain_ledger"


def _is_hypothesis_validation_mode(value: Any) -> bool:
    return _normalize_analysis_mode(value) == "validate_hypothesis"


def _action_failed(observation: str) -> bool:
    return any(marker in observation for marker in FAILURE_MARKERS)


@dataclass(slots=True)
class ExperimentRecord:
    """一次原始实验，宿主只暴露 t,q 和控制参数给 agent。"""

    experiment_id: str
    config: ExperimentConfig
    result: ExperimentResult
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DerivedSeries:
    """数据处理 LLM 追加到实验数据记录表的派生序列。"""

    experiment_id: str
    name: str
    values: np.ndarray
    source_name: str
    provenance: str
    summary_text: str


@dataclass(slots=True)
class LedgerObservation:
    """实验数据记录表中的 OBS 观察条目。"""

    observation_id: str
    step_index: int
    summary: str
    source_data_refs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LedgerValidation:
    """实验数据记录表中的 VAL 假说验证条目。"""

    validation_id: str
    step_index: int
    hypothesis_id: str
    experiment_ids: list[str]
    supports: bool
    metric_name: str
    metric_values: dict[str, float]
    aggregate_score: float | None
    summary: str
    source_data_refs: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ActionDecision:
    """决策 LLM 输出的下一步动作。"""

    thought: str
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionRecord:
    """已经执行的动作和宿主反馈。"""

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
class ScientificNotebook:
    """双账本：实验数据记录表 + 假说表。"""

    experiments: dict[str, ExperimentRecord] = field(default_factory=dict)
    derived_series: dict[str, dict[str, DerivedSeries]] = field(default_factory=dict)
    observations: list[LedgerObservation] = field(default_factory=list)
    validations: list[LedgerValidation] = field(default_factory=list)
    hypothesis_registry: HypothesisRegistry = field(default_factory=HypothesisRegistry)
    action_history: list[ActionRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    _observation_counter: int = 0
    _validation_counter: int = 0

    def register_experiment(self, record: ExperimentRecord) -> None:
        self.experiments[record.experiment_id] = record
        self.notes.append(
            f"{record.experiment_id}: 写入原始 t,q；n={len(record.result.t)}, "
            f"q_range=[{np.min(record.result.q):.6f}, {np.max(record.result.q):.6f}]。"
        )

    def register_series(self, series: DerivedSeries) -> None:
        self.derived_series.setdefault(series.experiment_id, {})[series.name] = series
        self.notes.append(
            f"{series.experiment_id}: 新增派生序列 `{series.name}`，来源 `{series.source_name}`。"
            f"{series.summary_text}"
        )

    def register_observation(
        self,
        *,
        step_index: int,
        summary: str,
        source_data_refs: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
        figures: list[str] | None = None,
    ) -> LedgerObservation:
        self._observation_counter += 1
        observation = LedgerObservation(
            observation_id=f"OBS{self._observation_counter:03d}",
            step_index=step_index,
            summary=summary,
            source_data_refs=list(source_data_refs or []),
            metrics=dict(metrics or {}),
            figures=list(figures or []),
        )
        self.observations.append(observation)
        self.notes.append(f"{observation.observation_id}: {observation.summary}")
        return observation

    def register_validation(
        self,
        *,
        step_index: int,
        hypothesis_id: str,
        experiment_ids: list[str],
        supports: bool,
        metric_name: str,
        metric_values: dict[str, float] | None = None,
        aggregate_score: float | None = None,
        summary: str = "",
        source_data_refs: list[str] | None = None,
        figures: list[str] | None = None,
    ) -> LedgerValidation:
        if not str(hypothesis_id).strip():
            raise ValueError("validation 必须绑定 hypothesis_id。")
        self._validation_counter += 1
        validation = LedgerValidation(
            validation_id=f"VAL{self._validation_counter:03d}",
            step_index=step_index,
            hypothesis_id=str(hypothesis_id).strip(),
            experiment_ids=list(experiment_ids),
            supports=bool(supports),
            metric_name=metric_name,
            metric_values=dict(metric_values or {}),
            aggregate_score=aggregate_score,
            summary=summary,
            source_data_refs=list(source_data_refs or []),
            figures=list(figures or []),
        )
        self.validations.append(validation)
        verdict = "支持" if validation.supports else "反驳"
        self.notes.append(
            f"{validation.validation_id}: {verdict} {validation.hypothesis_id}; "
            f"metric={validation.metric_name}, score={validation.aggregate_score}; {validation.summary}"
        )
        return validation

    def add_action_record(self, action_record: ActionRecord) -> None:
        self.action_history.append(action_record)

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

    def resolve_validation_ids(self, values: Any) -> list[str]:
        if values is None or values == "":
            return []
        if isinstance(values, str):
            requested = [item.strip() for item in re.split(r"[,;\n]", values) if item.strip()]
        else:
            try:
                requested = [str(item).strip() for item in values if str(item).strip()]
            except TypeError:
                requested = [str(values).strip()]
        known = {validation.validation_id for validation in self.validations}
        unknown = [validation_id for validation_id in requested if validation_id not in known]
        if unknown:
            raise ValueError(f"未知 validation_id: {unknown}。")
        return requested

    def latest_validation_for_hypothesis(self, hypothesis_id: str) -> LedgerValidation | None:
        for validation in reversed(self.validations):
            if validation.hypothesis_id == hypothesis_id:
                return validation
        return None


@dataclass(slots=True)
class ScientificCycleResult:
    """一次多轮科学发现循环的输出。"""

    notebook: ScientificNotebook
    final_law: LawHypothesis
    report_markdown: str | None = None
    report_path: str | None = None
    figure_paths: list[str] = field(default_factory=list)


class HypothesisBrain:
    """LLM 决策层：看双账本，选择下一步科研动作。"""

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        timeout_seconds: float = 90.0,
    ) -> None:
        if OpenAI is None:
            raise ImportError("未检测到 openai 官方库。")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=2)
        self.model = model
        self.temperature = temperature

    def summarize_notebook(
        self,
        notebook: ScientificNotebook,
        goal: str,
        max_steps: int,
        *,
        include_full_data: bool = False,
    ) -> str:
        lines = [
            f"研究目标: {goal}",
            f"当前已执行步骤数: {len(notebook.action_history)} / {max_steps}",
            f"实验数量: {len(notebook.experiments)}",
            "世界观提醒: 这是一个人工构造的黑盒虚拟物理世界。"
            "你可以看原始数据和派生数据形成直觉，但不能把未计算的脑内指标当证据。",
            "账本边界: run_experiments 写原始 t,q；analyze_data 维护派生量/OBS/VAL；"
            "manage_hypotheses 只维护 proposed/rejected/accepted 三状态。",
            self._ledger_workflow_status(notebook),
        ]

        if not notebook.experiments:
            lines.append("实验数据记录表为空。")
        else:
            lines.extend(self._format_experiment_coverage(notebook))
            if include_full_data:
                lines.extend(self._format_full_experiment_context(notebook))
            else:
                lines.extend(self._format_compact_experiment_context(notebook))

        if notebook.observations:
            if include_full_data:
                lines.append("FULL_DATA_PROCESSING_RESULTS - 全部 OBS:")
                selected_observations = notebook.observations
            else:
                lines.append("实验数据记录表 - 压缩 OBS:")
                selected_observations = self._selected_observations_for_prompt(notebook)
            for observation in selected_observations:
                payload = {
                    "observation_id": observation.observation_id,
                    "step_index": observation.step_index,
                    "summary": observation.summary
                    if include_full_data
                    else self._truncate_text(observation.summary, 420),
                    "source_data_refs": observation.source_data_refs,
                    "metrics": observation.metrics if include_full_data else self._format_metric_brief(observation.metrics),
                    "figures": observation.figures,
                }
                lines.append(self._json_dumps_safe(payload))
        if notebook.validations:
            if include_full_data:
                lines.append("FULL_DATA_PROCESSING_RESULTS - 全部 VAL:")
                selected_validations = notebook.validations
            else:
                lines.append("实验数据记录表 - VAL 摘要:")
                selected_validations = self._selected_validations_for_prompt(notebook)
            for validation in selected_validations:
                payload = {
                    "validation_id": validation.validation_id,
                    "step_index": validation.step_index,
                    "hypothesis_id": validation.hypothesis_id,
                    "experiment_ids": validation.experiment_ids,
                    "supports": validation.supports,
                    "metric_name": validation.metric_name,
                    "metric_values": validation.metric_values if include_full_data else self._format_metric_brief(validation.metric_values),
                    "aggregate_score": validation.aggregate_score,
                    "summary": validation.summary
                    if include_full_data
                    else self._truncate_text(validation.summary, 320),
                    "source_data_refs": validation.source_data_refs,
                    "figures": validation.figures,
                }
                lines.append(self._json_dumps_safe(payload))
        if notebook.notes and include_full_data:
            lines.append("FULL_DATA_PROCESSING_RESULTS - 全部 notebook notes:")
            for index, note in enumerate(notebook.notes, start=1):
                lines.append(self._json_dumps_safe({"note_index": index, "text": note}))

        lines.append(notebook.hypothesis_registry.summarize_for_prompt())
        reminder = self._pending_hypothesis_evidence_reminder(notebook)
        if reminder:
            lines.append(reminder)

        if notebook.action_history and include_full_data:
            lines.append("FULL_DECISION_AND_TOOL_HISTORY - 全部动作记录:")
            for item in notebook.action_history:
                lines.append(self._json_dumps_safe({
                    "step_index": item.step_index,
                    "thought": item.decision.thought,
                    "action": item.decision.action,
                    "parameters": item.decision.parameters,
                    "observation": item.observation,
                }))
        elif notebook.action_history:
            lines.append("最近动作记录:")
            for item in notebook.action_history[-4:]:
                lines.append(self._json_dumps_safe({
                    "step_index": item.step_index,
                    "thought": self._truncate_text(item.decision.thought, 180),
                    "action": item.decision.action,
                    "observation": self._truncate_text(item.observation, 360),
                }))
        return "\n".join(lines)

    def _format_compact_experiment_context(self, notebook: ScientificNotebook) -> list[str]:
        lines = [
            "实验数据记录表 - 常规规划摘要:",
            "说明: 为避免每轮规划上下文过长，常规规划只显示摘要；当决策准备 propose 时，宿主会重新提供 FULL_RAW_EXPERIMENT_DATA 和全部处理结果进行全量证据审查。",
        ]
        for experiment_id in self._representative_experiment_ids(notebook):
            record = notebook.experiments[experiment_id]
            available_series = notebook.available_series(experiment_id)
            lines.extend(
                [
                    f"- {experiment_id}: force_field_type={record.config.force_field_type.value}, "
                    f"F_ext={self._notebook_force_value(record):.6g}, q0={record.config.initial_q}, "
                    f"v0={record.config.initial_v}",
                    f"  t: {self._format_time_axis_summary(record.result.t)}",
                    f"  q: {self._format_series_sketch(record.result.q)}",
                    f"  available_series_count={len(available_series)}, names={self._format_name_list(available_series)}",
                ]
            )
            for series_name, series in sorted(notebook.derived_series.get(experiment_id, {}).items())[:4]:
                lines.append(f"  derived {series_name}: {series.summary_text}")
        return lines

    def _format_full_experiment_context(self, notebook: ScientificNotebook) -> list[str]:
        lines = [
            "FULL_RAW_EXPERIMENT_DATA:",
            "说明: 以下是决策 agent 可直接观察的完整原始实验数据；raw_series 只包含黑盒观测 t,q，不包含底层 universe.py 方程。",
        ]
        for experiment_id, record in sorted(notebook.experiments.items()):
            derived_payload: dict[str, Any] = {}
            for series_name, series in sorted(notebook.derived_series.get(experiment_id, {}).items()):
                derived_payload[series_name] = {
                    "values": self._array_to_json_list(series.values),
                    "source_name": series.source_name,
                    "provenance": series.provenance,
                    "summary": series.summary_text,
                }
            lines.append(self._json_dumps_safe({
                "experiment_id": experiment_id,
                "config": {
                    "force_field_type": record.config.force_field_type.value,
                    "F_ext": self._notebook_force_value(record),
                    "initial_q": record.config.initial_q,
                    "initial_v": record.config.initial_v,
                    "t_span": list(record.config.t_span),
                    "dt": record.config.dt,
                    "noise_std": record.config.noise_std,
                },
                "raw_series": {
                    "t": self._array_to_json_list(record.result.t),
                    "q": self._array_to_json_list(record.result.q),
                },
                "derived_series": derived_payload,
            }))
        return lines

    def _format_name_list(self, names: list[str], limit: int = 10) -> str:
        shown = names[:limit]
        suffix = f", ...(+{len(names) - limit})" if len(names) > limit else ""
        return "[" + ", ".join(shown) + suffix + "]"

    def _format_experiment_coverage(self, notebook: ScientificNotebook) -> list[str]:
        force_values = sorted({round(self._notebook_force_value(record), 9) for record in notebook.experiments.values()})
        initial_velocities = sorted({round(float(record.config.initial_v), 9) for record in notebook.experiments.values()})
        field_counts: dict[str, int] = {}
        for record in notebook.experiments.values():
            field_counts[record.config.force_field_type.value] = field_counts.get(record.config.force_field_type.value, 0) + 1
        return [
            "实验覆盖摘要:",
            f"- force_field_type_counts={field_counts}",
            f"- F_ext_values={force_values[:12]}{' ...' if len(force_values) > 12 else ''}",
            f"- initial_v_values={initial_velocities[:12]}{' ...' if len(initial_velocities) > 12 else ''}",
        ]

    def _representative_experiment_ids(self, notebook: ScientificNotebook, limit: int = 6) -> list[str]:
        ids = sorted(notebook.experiments.keys())
        if len(ids) <= limit:
            return ids
        selected: list[str] = []
        for experiment_id in ids:
            record = notebook.experiments[experiment_id]
            force_value = self._notebook_force_value(record)
            if force_value == 0 or abs(record.config.initial_v) > 1e-9:
                selected.append(experiment_id)
            if len(selected) >= limit // 2:
                break
        for experiment_id in ids[-limit:]:
            if experiment_id not in selected:
                selected.append(experiment_id)
            if len(selected) >= limit:
                break
        return selected[:limit]

    def _selected_observations_for_prompt(self, notebook: ScientificNotebook, limit: int = 10) -> list[LedgerObservation]:
        selected: list[LedgerObservation] = []
        metric_markers = ("r2", "rmse", "collapse", "diagnostic", "residual", "ratio", "slope")
        for observation in notebook.observations:
            metric_text = " ".join(str(key).lower() for key in observation.metrics.keys())
            summary_text = observation.summary.lower()
            if any(marker in metric_text or marker in summary_text for marker in metric_markers):
                selected.append(observation)
        selected.extend(notebook.observations[-6:])
        deduped: list[LedgerObservation] = []
        seen: set[str] = set()
        for observation in selected:
            if observation.observation_id not in seen:
                seen.add(observation.observation_id)
                deduped.append(observation)
        return deduped[-limit:]

    def _selected_validations_for_prompt(self, notebook: ScientificNotebook, limit: int = 10) -> list[LedgerValidation]:
        if len(notebook.validations) <= limit:
            return list(notebook.validations)
        selected = notebook.validations[-limit:]
        accepted_or_refuting = [
            validation
            for validation in notebook.validations
            if validation.supports or any(marker in validation.metric_name.lower() for marker in ("rmse", "r2", "r²"))
        ][-4:]
        merged = accepted_or_refuting + selected
        deduped: list[LedgerValidation] = []
        seen: set[str] = set()
        for validation in merged:
            if validation.validation_id not in seen:
                seen.add(validation.validation_id)
                deduped.append(validation)
        return deduped[-limit:]

    def _format_metric_brief(self, metrics: dict[str, Any], limit: int = 5) -> str:
        if not metrics:
            return "{}"
        flattened = self._flatten_numeric_metrics_for_prompt(metrics)
        if not flattened:
            return "{}"
        priority = sorted(
            flattened.items(),
            key=lambda item: (
                0 if any(marker in item[0].lower() for marker in ("r2", "r²", "rmse", "mae", "score", "gamma", "alpha")) else 1,
                item[0],
            ),
        )[:limit]
        return "{" + ", ".join(f"{key}={value:.4g}" for key, value in priority) + "}"

    def _flatten_numeric_metrics_for_prompt(self, value: Any, prefix: str = "") -> dict[str, float]:
        flattened: dict[str, float] = {}
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                flattened.update(self._flatten_numeric_metrics_for_prompt(item, child))
            return flattened
        if isinstance(value, bool):
            return {}
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if np.isfinite(numeric):
                flattened[prefix or "value"] = numeric
        return flattened

    def _ledger_workflow_status(self, notebook: ScientificNotebook) -> str:
        latest_run_step = self._latest_action_step(notebook, "run_experiments")
        latest_ledger_step = self._latest_ledger_maintenance_step(notebook)
        if latest_run_step is None:
            return "流程状态: 尚无实验。第一步应先做一组基准实验。"
        if latest_ledger_step < latest_run_step:
            return (
                "流程状态: 最近一批实验尚未由 analyze_data(mode=maintain_ledger) 维护。"
                "下一步应先更新实验数据记录表，再提出或验证假说。"
            )
        return (
            "流程状态: 最近实验已经维护进实验数据记录表。"
            "可以 propose；accept/reject 必须引用 VAL。"
        )

    def _latest_action_step(self, notebook: ScientificNotebook, action: str) -> int | None:
        steps = [item.step_index for item in notebook.action_history if item.decision.action == action]
        return max(steps) if steps else None

    def _latest_ledger_maintenance_step(self, notebook: ScientificNotebook) -> int:
        steps = [
            item.step_index
            for item in notebook.action_history
            if item.decision.action == "analyze_data"
            and _is_ledger_maintenance_mode(item.decision.parameters.get("analysis_mode"))
            and not _action_failed(item.observation)
        ]
        return max(steps) if steps else 0

    def _pending_hypothesis_evidence_reminder(self, notebook: ScientificNotebook) -> str:
        for item in reversed(notebook.action_history):
            if item.decision.action == "manage_hypotheses" and not _action_failed(item.observation):
                operation = str(item.decision.parameters.get("operation", "")).strip().lower()
                if operation in {"accept", "accepted", "reject", "rejected", "confirm", "refute", "failed"}:
                    return ""
            if item.decision.action != "analyze_data":
                continue
            params = item.decision.parameters
            target = str(params.get("hypothesis_id", params.get("candidate_expression", ""))).strip()
            if target and _is_hypothesis_validation_mode(params.get("analysis_mode")):
                return (
                    f"强制证据回写: 上一次 validate_hypothesis 绑定了 `{target}`。"
                    "下一步必须调用 manage_hypotheses accept 或 reject，并引用对应 VAL。"
                )
        return ""

    def plan_next_action(self, notebook_summary: str, max_steps: int) -> ActionDecision:
        tool_name_list = render_tool_name_list(DEFAULT_TOOL_SPECS)
        tool_descriptions = render_tool_descriptions(DEFAULT_TOOL_SPECS)
        system_prompt = (
            "你是一位研究黑盒虚拟物理世界的科学家。你不知道底层方程。"
            "你始终可以观察完整原始 t,q、全部派生序列、全部 OBS/VAL 和工具反馈来形成科学直觉。"
            "数据处理 LLM 只是写代码和计算指标的工具；最终公式必须由你综合全量数据后在 manage_hypotheses propose 中提出。"
            "所有计算型证据必须来自 analyze_data 或已经登记的数据处理结果。"
            "请只输出严格 JSON。"
        )
        user_prompt = f"""
下面是当前双账本摘要：

{notebook_summary}

你可用的 action 只有：
{tool_name_list}

动作说明：
{tool_descriptions}

规划要求：
1. 只有 run_experiments、analyze_data、manage_hypotheses、finish 四个 action。
2. 如果没有实验，先用 run_experiments 批量做基准实验；基准实验必须覆盖 F_ext=0 和至少一个非零 F_ext，最好同时含正/负外力或不同初速。
3. 新实验之后，先用 analyze_data(mode=maintain_ledger) 维护实验数据记录表：派生量、OBS、图像。你必须在 analysis_goal 中说明要维护什么，例如要估计什么变化率、比较哪些实验、输出哪些 OBS 数值事实。
4. 如果你准备 propose，宿主会拦截并重新提供 FULL_RAW_EXPERIMENT_DATA、FULL_DATA_PROCESSING_RESULTS 和 FULL_DECISION_AND_TOOL_HISTORY 进行全量证据审查；提出公式时必须引用 observation_ids 和 source_data_refs，source_data_refs 应尽量同时覆盖原始序列、关键派生序列和支持该公式的数据处理结果。每个公式都要来自全量数据中的具体数值线索。
5. 如果现有 OBS 只说明“拟合好/不好”，但没有揭示跨实验一致性、变量关系、残差结构或可复验的数据事实，请先 analyze_data(mode=maintain_ledger) 做诊断观察，再 propose。
6. 诊断观察可以要求数据处理 LLM 比较响应量、控制量和派生状态量之间的数据关系、残差结构、符号/尺度一致性和跨实验可复验性；这是观察数据结构，不是让它替你宣布定律。不要在 prompt 中指定或暗示某个物理规律形式。
7. 在提出或分析关系前，避免使用未验证的物理类比作为任务目标；先用中性的数据关系描述。
8. accept/reject 必须来自 analyze_data(mode=validate_hypothesis) 生成的 VAL，不能自己声称算过 RMSE/R²/残差。
9. accept 的硬门槛很高：跨实验覆盖、R² 通常必须 >=0.99，RMSE 必须很小；局部高 R²、单实验拟合或只有一类初始条件不能 accept。
10. 验证假说时，你必须在 analysis_goal 中决定验证口径：候选表达式如何生成预测、残差怎么定义、处理哪些实验、报告哪些指标、是否生成残差序列。数据处理 LLM 只负责写代码执行你的验证任务。
11. accepted 的语义是“足以回答当前研究目标的规律”。局部对照事实或只覆盖单一控制子域的事实应写入 OBS 或保持 proposed，不能为了结束而 accept。
12. 只要假说表存在 accepted 假说，就可以调用 finish；manage_hypotheses accept 后宿主也可以结束。
13. 实验噪声固定为 0，不要探索或改变 noise_std。表达幂次用 Python 的 **，不要用 ^。
14. 当 action=manage_hypotheses 且 operation=propose 时，thought 中要简要说明你从完整原始数据、派生序列和 OBS/VAL 中看到了哪些共同结构，再给出单一可证伪公式。
15. 总步数上限为 {max_steps}，优先批量实验和批量分析。

请只返回 JSON：
{{
  "thought": "...",
  "action": "...",
  "parameters": {{...}}
}}
"""
        payload = self._request_json(system_prompt=system_prompt, user_prompt=user_prompt)
        return ActionDecision(
            thought=str(payload.get("thought", "继续收集证据。")),
            action=str(payload.get("action", "run_experiments")),
            parameters=dict(payload.get("parameters", {})),
        )

    def synthesize_final_law(self, notebook_summary: str) -> LawHypothesis:
        system_prompt = (
            "你是一位理论物理学家，正在总结黑盒虚拟物理世界中的动力学规律。"
            "只能依据实验数据记录表、VAL 验证和 accepted 假说总结。请只输出 JSON。"
        )
        user_prompt = f"""
请根据以下双账本总结当前发现：

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
            summary=str(payload.get("summary", "已形成一个 accepted 假说。")),
            proposed_law=str(payload.get("proposed_law", "")),
            evidence=str(payload.get("evidence", "")),
            confidence=str(payload.get("confidence", "medium")),
            next_steps=str(payload.get("next_steps", "")),
            raw_payload=payload,
        )

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
        if not content:
            try:
                content = response.choices[0].message.reasoning_content
            except Exception:
                pass
        if not content:
            try:
                content = response.choices[0].text
            except Exception:
                pass
        if not content:
            raise RuntimeError(f"LLM 返回空响应。当前 response: {response!r}")
        parsed = self._parse_json_object(str(content))
        if parsed is None:
            raise RuntimeError(f"LLM 返回的文本无法解析为 JSON。内容: {content!r}")
        return parsed

    def _parse_json_object(self, text: str) -> dict[str, Any] | None:
        candidates = [text.strip()]
        candidates.extend(match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL))
        object_match = re.search(r"\{.*\}", text, re.DOTALL)
        if object_match:
            candidates.append(object_match.group(0))
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _format_time_axis_summary(self, values: np.ndarray) -> str:
        values_array = np.asarray(values, dtype=float).reshape(-1)
        if values_array.size == 0:
            return "n=0"
        dt_text = ""
        if values_array.size > 1:
            dt_text = f", dt≈{float(np.median(np.diff(values_array))):.6g}"
        return (
            f"n={values_array.size}, range=[{float(np.min(values_array)):.6g}, "
            f"{float(np.max(values_array)):.6g}]{dt_text}, sample={self._format_sampled_values(values_array)}"
        )

    def _format_series_sketch(self, values: np.ndarray) -> str:
        values_array = np.asarray(values, dtype=float).reshape(-1)
        finite_values = values_array[np.isfinite(values_array)]
        if finite_values.size == 0:
            return f"n={values_array.size}, finite=0"
        return (
            f"n={values_array.size}, min={float(np.min(finite_values)):.6g}, "
            f"max={float(np.max(finite_values)):.6g}, mean={float(np.mean(finite_values)):.6g}, "
            f"std={float(np.std(finite_values)):.6g}, sample={self._format_sampled_values(values_array)}"
        )

    def _format_sampled_values(self, values: np.ndarray, edge_count: int = 4) -> str:
        values_array = np.asarray(values, dtype=float).reshape(-1)
        if values_array.size <= edge_count * 2:
            sample = values_array
        else:
            sample = np.concatenate([values_array[:edge_count], values_array[-edge_count:]])
        return json.dumps([float(f"{float(value):.8g}") for value in sample], ensure_ascii=False)

    def _array_to_json_list(self, values: np.ndarray) -> list[float | None]:
        result: list[float | None] = []
        for value in np.asarray(values, dtype=float).reshape(-1):
            numeric = float(value)
            result.append(numeric if np.isfinite(numeric) else None)
        return result

    def _json_dumps_safe(self, value: Any) -> str:
        return json.dumps(
            self._to_jsonable(value),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _to_jsonable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            return numeric if np.isfinite(numeric) else None
        if isinstance(value, np.ndarray):
            return self._array_to_json_list(value)
        if isinstance(value, dict):
            return {str(key): self._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_jsonable(item) for item in value]
        return str(value)

    def _truncate_text(self, text: str, max_chars: int) -> str:
        normalized = re.sub(r"\s+", " ", str(text)).strip()
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."

    def _notebook_force_value(self, record: ExperimentRecord) -> float:
        if record.config.force_field_type is ForceFieldType.CONSTANT:
            return float(record.config.constant_force)
        return 0.0


class ScientistAgent:
    """自主科研循环：四个动作，两个账本。"""

    def __init__(
        self,
        universe: VirtualUniverse,
        brain: HypothesisBrain | None = None,
        data_brain: DataProcessingBrain | None = None,
        reporter: ScientificReporter | None = None,
        generated_code_dir: str | Path | None = None,
        use_generated_processors: bool = True,
        allow_data_processing_fallback: bool = False,
    ) -> None:
        self.universe = universe
        self.brain = brain or HypothesisBrain()
        self.data_brain = data_brain
        self.reporter = reporter or ScientificReporter()
        self.generated_code_dir = Path(generated_code_dir or (Path.cwd() / "generated_processors")).resolve()
        self.generated_code_runner = GeneratedCodeRunner(self.generated_code_dir)
        self.generated_code_registry = CodeRegistry(self.generated_code_dir / "registry.json")
        self.use_generated_processors = use_generated_processors
        self.allow_data_processing_fallback = allow_data_processing_fallback
        self.tool_specs: dict[str, ToolSpec] = dict(TOOL_SPECS_BY_NAME)
        self._experiment_counter = 0

    def run_scientific_cycle(
        self,
        report_dir: str | Path | None = None,
        max_steps: int = 20,
        goal: str | None = None,
        progress_callback: Callable[[ActionRecord], None] | None = None,
    ) -> ScientificCycleResult:
        notebook = ScientificNotebook()
        finished_by_finalize = False
        research_goal = goal or (
            "只能从时间-位置观测和控制参数出发，探索黑盒虚拟宇宙中的运动规律。"
            "可以构造派生物理量，但不能预设底层方程。"
        )

        for step_index in range(1, max_steps + 1):
            notebook_summary = self.brain.summarize_notebook(
                notebook,
                research_goal,
                max_steps,
                include_full_data=False,
            )
            try:
                decision = self.brain.plan_next_action(notebook_summary=notebook_summary, max_steps=max_steps)
                decision = self._enforce_ledger_workflow(notebook=notebook, decision=decision)
                reviewed_decision = self._review_proposal_with_full_context(
                    notebook=notebook,
                    goal=research_goal,
                    max_steps=max_steps,
                    decision=decision,
                )
                if reviewed_decision is not decision:
                    reviewed_decision = self._enforce_ledger_workflow(notebook=notebook, decision=reviewed_decision)
                decision = reviewed_decision
            except Exception as exc:
                decision = ActionDecision(
                    thought="决策 LLM 调用失败，停止本轮科研循环并保留当前账本。",
                    action="planning_error",
                    parameters={},
                )
                observation = f"决策 LLM 调用失败: {exc}"
                notebook.add_action_record(ActionRecord(step_index=step_index, decision=decision, observation=observation))
                if progress_callback is not None:
                    progress_callback(notebook.action_history[-1])
                break
            try:
                observation, should_finish = self._execute_action(
                    notebook=notebook,
                    step_index=step_index,
                    decision=decision,
                )
            except Exception as exc:
                observation = f"动作执行失败: {exc}. 请基于失败反馈重新规划。"
                should_finish = False
            action_record = ActionRecord(step_index=step_index, decision=decision, observation=observation)
            notebook.add_action_record(action_record)
            if progress_callback is not None:
                progress_callback(action_record)
            if should_finish:
                finished_by_finalize = True
                break

        final_summary = self.brain.summarize_notebook(
            notebook,
            research_goal,
            max_steps,
            include_full_data=False,
        )
        if self._final_ready_hypotheses(notebook):
            final_law = self.brain.synthesize_final_law(final_summary)
        else:
            final_law = self._build_inconclusive_law(notebook, finished_by_finalize)
        cycle_result = ScientificCycleResult(notebook=notebook, final_law=final_law)
        cycle_result.report_markdown = self.reporter.generate_markdown(cycle_result)
        if report_dir is not None:
            report_path = self.reporter.save_report(
                markdown_text=cycle_result.report_markdown,
                output_dir=report_dir,
            )
            cycle_result.report_path = str(report_path)
        return cycle_result

    def _enforce_ledger_workflow(self, *, notebook: ScientificNotebook, decision: ActionDecision) -> ActionDecision:
        if not notebook.experiments and decision.action != "run_experiments":
            return ActionDecision(
                thought=f"{decision.thought}\n[workflow_override] 当前没有实验数据，先运行基准实验。",
                action="run_experiments",
                parameters={
                    "experiments": [
                        {"q0": 0.0, "v0": 0.0, "F_ext": 0.0, "t_end": 10.0, "dt": 0.1},
                        {"q0": 0.0, "v0": 5.0, "F_ext": 0.0, "t_end": 10.0, "dt": 0.1},
                        {"q0": 0.0, "v0": 0.0, "F_ext": 2.0, "t_end": 10.0, "dt": 0.1},
                    ],
                },
            )
        if self._needs_ledger_maintenance(notebook):
            return ActionDecision(
                thought=f"{decision.thought}\n[workflow_override] 新实验尚未维护进实验数据记录表。",
                action="analyze_data",
                parameters=self._build_ledger_maintenance_params(notebook, decision),
            )
        if self._needs_control_diversity(notebook) and not self._decision_adds_control_diversity(notebook, decision):
            return ActionDecision(
                thought=f"{decision.thought}\n[workflow_override] 当前实验只覆盖单一控制条件，先补充非零外力实验。",
                action="run_experiments",
                parameters=self._build_control_diversity_params(notebook),
            )
        if self._needs_initial_condition_diversity(notebook) and not self._decision_adds_initial_condition_diversity(decision):
            return ActionDecision(
                thought=f"{decision.thought}\n[workflow_override] 当前实验缺少非零初速度条件，先补充初始速度扰动实验。",
                action="run_experiments",
                parameters=self._build_initial_condition_diversity_params(notebook),
            )
        if self._needs_diagnostic_observation(notebook) and decision.action in {"analyze_data", "manage_hypotheses"}:
            return ActionDecision(
                thought=f"{decision.thought}\n[workflow_override] 提出公式前先做一次诊断观察，确认变量坍缩和残差结构。",
                action="analyze_data",
                parameters=self._build_diagnostic_observation_params(notebook),
            )
        return decision

    def _review_proposal_with_full_context(
        self,
        *,
        notebook: ScientificNotebook,
        goal: str,
        max_steps: int,
        decision: ActionDecision,
    ) -> ActionDecision:
        if not self._is_hypothesis_proposal(decision):
            return decision
        full_summary = self.brain.summarize_notebook(
            notebook,
            goal,
            max_steps,
            include_full_data=True,
        )
        full_summary = (
            full_summary
            + "\n\nFULL_EVIDENCE_PROPOSAL_REVIEW:\n"
            "你上一轮准备提出假说。现在必须基于上面的完整原始 t/q、全部派生序列、全部 OBS/VAL、notebook notes 和动作历史重新判断。"
            "如果证据仍然支持提出单一公式，请继续 manage_hypotheses propose，并在 thought 中说明全量数据里的共同结构。"
            "如果还缺少关键诊断或验证线索，请改为 analyze_data(mode=maintain_ledger) 或 run_experiments。"
        )
        reviewed = self.brain.plan_next_action(notebook_summary=full_summary, max_steps=max_steps)
        if self._is_hypothesis_proposal(reviewed):
            reviewed.thought = (
                f"{reviewed.thought}\n"
                "[full_context_review] 已在提出假说前读取 FULL_RAW_EXPERIMENT_DATA、"
                "FULL_DATA_PROCESSING_RESULTS 和 FULL_DECISION_AND_TOOL_HISTORY。"
            )
        return reviewed

    def _is_hypothesis_proposal(self, decision: ActionDecision) -> bool:
        if decision.action != "manage_hypotheses":
            return False
        params = self._normalize_manage_hypotheses_params(decision.parameters)
        operation = str(params.get("operation", "")).strip().lower()
        if operation in {"propose", "add", "new"}:
            return True
        return bool(params.get("expression")) and operation in {"", "list", "summarize", "summary"}

    def _needs_ledger_maintenance(self, notebook: ScientificNotebook) -> bool:
        return bool(self._unmaintained_experiment_ids(notebook))

    def _needs_control_diversity(self, notebook: ScientificNotebook) -> bool:
        if not notebook.experiments:
            return False
        return len(self._control_values_for_experiments(notebook=notebook, experiment_ids=sorted(notebook.experiments))) < 2

    def _needs_initial_condition_diversity(self, notebook: ScientificNotebook) -> bool:
        if not notebook.experiments:
            return False
        return not any(abs(record.config.initial_v) > 1e-9 for record in notebook.experiments.values())

    def _needs_diagnostic_observation(self, notebook: ScientificNotebook) -> bool:
        if notebook.hypothesis_registry.all_records():
            return False
        if len(notebook.experiments) < 5 or self._unmaintained_experiment_ids(notebook):
            return False
        if len(notebook.observations) < 12:
            return False
        for action_record in notebook.action_history:
            workflow = action_record.decision.parameters.get("workflow", {})
            if isinstance(workflow, dict) and workflow.get("reason") == "diagnostic_observation_pass":
                return False
        return True

    def _build_diagnostic_observation_params(self, notebook: ScientificNotebook) -> dict[str, Any]:
        experiment_ids = sorted(notebook.experiments.keys())
        return {
            "analysis_mode": "maintain_ledger",
            "experiment_ids": experiment_ids,
            "analysis_goal": (
                "做一次观察优先的结构化诊断 pass，目标是帮助决策 LLM 看见数据结构，而不是提出定律。"
                "请只基于已有原始序列、派生序列、控制量和 OBS，系统比较变量之间的数据关系、"
                "跨实验一致性、残差结构、尺度/符号一致性和反例。"
                "可以使用简单数值变换来辅助比较，但不要在 analysis_goal 或 observation 中指定、暗示或命名任何物理规律形式，"
                "也不要使用未验证的物理类比。"
                "不要输出最终公式，不要替决策 LLM propose；只输出可核验的数据事实。"
                "输出 5-10 条 OBS，每条必须包含具体数值、source_data_refs、metrics；"
                "metrics 中包含 diagnostic_pass=true、observation_count，以及最有信息量的 R2/RMSE/残差/一致性指标。"
                "最后列出哪些数据关系被排除，以及哪些仍值得由决策 LLM 进一步提出可证伪关系。"
            ),
            "expected_outputs": [
                "diagnostic_observations_with_numbers",
                "collapse_or_residual_metrics",
                "directions_ruled_out",
                "diagnostic_figures_if_useful",
            ],
            "workflow": {"forced_by_host": True, "reason": "diagnostic_observation_pass"},
        }

    def _decision_adds_control_diversity(self, notebook: ScientificNotebook, decision: ActionDecision) -> bool:
        if decision.action != "run_experiments":
            return False
        existing_values = self._control_values_for_experiments(
            notebook=notebook,
            experiment_ids=sorted(notebook.experiments),
        )
        for raw_experiment in self._raw_experiment_param_list(decision.parameters):
            if self._force_value_from_raw_experiment(raw_experiment) not in existing_values:
                return True
        return False

    def _decision_adds_initial_condition_diversity(self, decision: ActionDecision) -> bool:
        if decision.action != "run_experiments":
            return False
        return any(abs(self._coerce_float(raw_experiment.get("initial_v", raw_experiment.get("v0")), 0.0)) > 1e-9 for raw_experiment in self._raw_experiment_param_list(decision.parameters))

    def _build_control_diversity_params(self, notebook: ScientificNotebook) -> dict[str, Any]:
        dt = 0.05
        t_end = 8.0
        if notebook.experiments:
            first = next(iter(notebook.experiments.values()))
            dt = float(first.config.dt)
            t_end = min(10.0, max(4.0, float(first.config.t_span[1] - first.config.t_span[0])))
        return {
            "experiments": [
                {"q0": 0.0, "v0": 0.0, "F_ext": 1.0, "t_end": t_end, "dt": dt},
                {"q0": 0.0, "v0": 0.0, "F_ext": -1.0, "t_end": t_end, "dt": dt},
                {"q0": 0.0, "v0": 2.0, "F_ext": 1.0, "t_end": t_end, "dt": dt},
            ],
        }

    def _build_initial_condition_diversity_params(self, notebook: ScientificNotebook) -> dict[str, Any]:
        dt = 0.05
        t_end = 8.0
        if notebook.experiments:
            first = next(iter(notebook.experiments.values()))
            dt = float(first.config.dt)
            t_end = min(10.0, max(4.0, float(first.config.t_span[1] - first.config.t_span[0])))
        return {
            "experiments": [
                {"q0": 0.0, "v0": 5.0, "F_ext": 0.0, "t_end": t_end, "dt": dt},
                {"q0": 0.0, "v0": 2.0, "F_ext": 1.0, "t_end": t_end, "dt": dt},
                {"q0": 0.0, "v0": -2.0, "F_ext": 1.0, "t_end": t_end, "dt": dt},
            ],
        }

    def _raw_experiment_param_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        raw_experiments = params.get("experiments", params.get("experiment_configs", params.get("batch")))
        if raw_experiments is None:
            raw_experiments = [params]
        elif isinstance(raw_experiments, dict):
            raw_experiments = [raw_experiments]
        else:
            raw_experiments = list(raw_experiments)
        return [item for item in raw_experiments if isinstance(item, dict)]

    def _force_value_from_raw_experiment(self, params: dict[str, Any]) -> float:
        if any(key in params for key in ("constant_force", "F_ext", "F")):
            return round(self._coerce_float(self._get_first_present_param(params, "constant_force", "F_ext", "F"), 0.0), 9)
        force_type = str(params.get("force_field_type", "")).strip().lower()
        if force_type in {"free", "none", "zero"}:
            return 0.0
        return round(self._coerce_float(params.get("constant_force"), 0.0), 9)

    def _latest_action_step(self, notebook: ScientificNotebook, action: str) -> int | None:
        steps = [item.step_index for item in notebook.action_history if item.decision.action == action]
        return max(steps) if steps else None

    def _latest_ledger_maintenance_step(self, notebook: ScientificNotebook) -> int:
        steps = [
            item.step_index
            for item in notebook.action_history
            if item.decision.action == "analyze_data"
            and _is_ledger_maintenance_mode(item.decision.parameters.get("analysis_mode"))
            and not _action_failed(item.observation)
        ]
        return max(steps) if steps else 0

    def _build_ledger_maintenance_params(
        self,
        notebook: ScientificNotebook,
        decision: ActionDecision | None = None,
    ) -> dict[str, Any]:
        unmaintained_experiment_ids = self._unmaintained_experiment_ids(notebook)
        original_params = decision.parameters if decision and isinstance(decision.parameters, dict) else {}
        original_goal = str(original_params.get("analysis_goal", "")).strip()
        if not original_goal and decision is not None and decision.action == "analyze_data":
            original_goal = decision.thought.strip()
        try:
            requested_experiment_ids = self._resolve_experiment_ids_for_action(
                notebook,
                original_params.get("experiment_ids"),
            )
        except Exception:
            requested_experiment_ids = sorted(notebook.experiments.keys())
        if unmaintained_experiment_ids:
            requested_experiment_ids = self._merge_experiment_id_lists(
                unmaintained_experiment_ids,
                requested_experiment_ids,
            )
        requested_outputs = original_params.get("expected_outputs") or [
            "derived_series_if_useful",
            "observations",
            "diagnostic_figures",
        ]
        if original_goal:
            task_text = f"决策 LLM 指定的数据维护任务是：{original_goal}"
        else:
            task_text = "决策 LLM 未明确指定维护内容；只做最小基础维护：从 t,q 估计必要的一阶/二阶变化率并写入数值观察。"
        return {
            "analysis_mode": "maintain_ledger",
            "experiment_ids": requested_experiment_ids,
            "analysis_goal": (
                "维护实验数据记录表。"
                f"{task_text}"
                f"优先补充尚未维护的实验 {unmaintained_experiment_ids}；"
                "允许重复分析旧实验做比较或复核，但不要重复定义已有派生序列。"
                "如果需要不同定义，请使用新的序列名并说明差异。"
                "除非决策 LLM 明确点名，不要定义动量、能量、阻力、质量、模型参数等新的物理量。"
                "噪声固定为 0；估计变化率时优先使用 np.gradient(values, t, edge_order=2)。"
                "输出 observations，每条包含具体数值和 source_data_refs。"
                "不要宣布最终定律，不要做无来源模型海选。"
            ),
            "expected_outputs": requested_outputs,
            "workflow": {
                "forced_by_host": True,
                "reason": "new_experiments_require_ledger_maintenance",
                "unmaintained_experiment_ids": unmaintained_experiment_ids,
            },
        }

    def _execute_action(
        self,
        *,
        notebook: ScientificNotebook,
        step_index: int,
        decision: ActionDecision,
    ) -> tuple[str, bool]:
        action = decision.action
        params = decision.parameters
        if action not in self.tool_specs:
            return f"未知动作 `{action}`。可用动作只有: {', '.join(self.tool_specs)}。", False
        if action == "run_experiments":
            return self._action_run_experiments(notebook, params), False
        if action == "analyze_data":
            params = self._attach_referenced_hypothesis_to_analysis(
                notebook=notebook,
                params=params,
                thought=decision.thought,
            )
            return self._action_analyze_data(notebook, step_index, params), False
        if action == "manage_hypotheses":
            observation = self._action_manage_hypotheses(notebook, step_index, params)
            return observation, self._should_finish_after_hypothesis_action(notebook, params)
        if action == "finish":
            if self._final_ready_hypotheses(notebook):
                return "已有 accepted 假说，科研循环结束。", True
            return "尚无 accepted 假说，不能结束。", False
        return f"未知动作 `{action}`。", False

    def _action_run_experiments(self, notebook: ScientificNotebook, params: dict[str, Any]) -> str:
        raw_experiments = params.get("experiments", params.get("experiment_configs", params.get("batch")))
        if raw_experiments is None:
            raw_experiments = [params]
        elif isinstance(raw_experiments, dict):
            raw_experiments = [raw_experiments]
        else:
            raw_experiments = list(raw_experiments)
        if not raw_experiments:
            raise ValueError("run_experiments 需要至少一个实验配置。")
        if len(raw_experiments) > 12:
            raise ValueError("一次 run_experiments 最多允许 12 个实验。")
        parts = []
        for index, experiment_params in enumerate(raw_experiments, start=1):
            if not isinstance(experiment_params, dict):
                raise ValueError(f"第 {index} 个实验配置必须是 dict。")
            parts.append(self._action_run_experiment(notebook, experiment_params))
        return self._format_batch_result("批量实验完成", parts)

    def _action_run_experiment(self, notebook: ScientificNotebook, params: dict[str, Any]) -> str:
        self._experiment_counter += 1
        experiment_id = f"exp_{self._experiment_counter:02d}"
        warnings = self._run_experiment_parameter_warnings(params)
        requested_noise = self._coerce_float(params.get("noise_std"), 0.0)
        if abs(requested_noise) > 1e-12:
            warnings.append(f"noise_std 已被全局固定为 0，忽略输入 {requested_noise}")

        force_was_provided = any(key in params for key in ("constant_force", "F_ext", "F"))
        constant_force = self._coerce_float(self._get_first_present_param(params, "constant_force", "F_ext", "F"), 0.0)
        if force_was_provided:
            force_field_type = ForceFieldType.CONSTANT if abs(constant_force) > 1e-12 else ForceFieldType.FREE
        elif "force_field_type" in params:
            force_field_type = self._resolve_force_field_type(params.get("force_field_type"))
        else:
            force_field_type = ForceFieldType.FREE

        config = ExperimentConfig(
            initial_q=self._coerce_float(self._get_first_param(params, "initial_q", "q0"), 0.0),
            initial_v=self._coerce_float(self._get_first_param(params, "initial_v", "v0"), 1.0),
            force_field_type=force_field_type,
            t_span=(0.0, self._coerce_float(params.get("t_end"), 5.0)),
            dt=self._coerce_float(params.get("dt"), 0.05),
            constant_force=constant_force,
            noise_std=0.0,
        )
        result = self.universe.run_experiment(config)
        record = ExperimentRecord(
            experiment_id=experiment_id,
            config=config,
            result=result,
            summary=self._summarize_trajectory(result),
        )
        notebook.register_experiment(record)
        return (
            f"完成实验 {experiment_id}。场景={config.force_field_type.value}，"
            f"F_ext={self._experiment_force_value(record):.6g}，"
            f"q 范围 [{np.min(result.q):.6f}, {np.max(result.q):.6f}]。"
            + (f" 参数提示: {'; '.join(warnings)}" if warnings else "")
        )

    def _action_analyze_data(self, notebook: ScientificNotebook, step_index: int, params: dict[str, Any]) -> str:
        target = self._hypothesis_evidence_target(params)
        if "analysis_mode" not in params and target:
            mode = "validate_hypothesis"
        else:
            mode = _normalize_analysis_mode(params.get("analysis_mode"))
        params = {**params, "analysis_mode": mode}
        if "analysis_goal" not in params:
            params["analysis_goal"] = "根据当前实验数据记录表执行一次明确的数据处理任务。"
        if mode == "validate_hypothesis" and not target:
            raise ValueError("analyze_data mode=validate_hypothesis 需要 hypothesis_id 或 candidate_expression。")
        if mode == "maintain_ledger":
            params["analysis_goal"] += (
                "\n[LEDGER_MAINTENANCE] 返回 derived_series 和/或 observations；"
                "observations 需包含 summary、source_data_refs、metrics。不要宣布最终定律。"
            )
        else:
            params["analysis_goal"] += (
                "\n[VALIDATE_HYPOTHESIS] 返回 validations；每条包含 hypothesis_id、experiment_ids、"
                "supports、metric_name、metric_values、aggregate_score、summary、source_data_refs。"
            )
        observation = self._execute_generated_data_action(
            notebook=notebook,
            step_index=step_index,
            action="analyze_data",
            params=params,
            fallback=lambda: "analyze_data 需要数据处理 LLM 生成代码路径。",
        )
        if _action_failed(observation):
            return observation
        if mode == "validate_hypothesis":
            observation += "\n[HYPOTHESIS_EVIDENCE_REQUIRED] 下一步必须 manage_hypotheses accept/reject，并引用 VAL。"
        else:
            observation += "\n[LEDGER_MAINTENANCE_PASS] 已维护实验数据记录表。"
        return observation

    def _action_manage_hypotheses(self, notebook: ScientificNotebook, step_index: int, params: dict[str, Any]) -> str:
        params = self._normalize_manage_hypotheses_params(params)
        operation = str(params.get("operation", "list")).strip().lower()
        if operation in {"propose", "add", "new"}:
            expression = str(params.get("expression", "")).strip()
            if not expression:
                raise ValueError("manage_hypotheses propose 需要 expression。")
            self._validate_specific_hypothesis_expression(expression)
            observation_ids = self._coerce_string_list(params.get("observation_ids", params.get("source_observation_ids")))
            source_data_refs = self._coerce_string_list(params.get("source_data_refs"))
            if notebook.observations and not observation_ids:
                raise ValueError(
                    "manage_hypotheses propose 必须引用 observation_ids。"
                    "请基于已有 OBS 的具体数值线索提出假说。"
                )
            if notebook.observations and not source_data_refs:
                raise ValueError(
                    "manage_hypotheses propose 必须提供 source_data_refs，说明假说来自哪些原始/派生数据。"
                )
            record, created = notebook.hypothesis_registry.propose(
                expression=self._normalize_expression_syntax(expression),
                step_index=step_index,
                origin_action="manage_hypotheses",
                readable_summary=str(params.get("readable_summary", params.get("summary", ""))),
                variables=self._coerce_string_list(params.get("variables")),
                assumptions=self._coerce_string_list(params.get("assumptions")),
                source_data_refs=source_data_refs,
                observation_ids=observation_ids,
                next_tests=self._coerce_string_list(params.get("next_tests")),
                note=str(params.get("note", params.get("notes", ""))),
            )
            verb = "新增" if created else "更新已有"
            return f"{verb}候选规律 {record.hypothesis_id}: `{record.expression}`，status={record.status}。"
        if operation in {"list", "rank", "summarize", "summary"}:
            return notebook.hypothesis_registry.summarize_for_prompt(limit=10)
        if operation in {"finalize", "finish"}:
            if self._final_ready_hypotheses(notebook):
                return "已有 accepted 假说，可以 finish。"
            return "尚无 accepted 假说，不能结束。"
        if operation in {
            "accept",
            "accepted",
            "approve",
            "approved",
            "reject",
            "rejected",
            "refute",
            "refuted",
            "deny",
            "denied",
            "fail",
            "failed",
            "falsify",
            "falsified",
            "invalid",
            "support",
            "supported",
            "confirm",
            "confirmed",
            "validate",
            "validated",
            "verify",
            "verified",
            "pass",
            "passed",
            "record_evidence",
            "evidence",
            "record",
            "update_status",
            "status",
        }:
            decision = self._manage_hypothesis_decision(operation, params)
            hypothesis_id = str(params.get("hypothesis_id", "")).strip() or None
            expression = str(params.get("expression", "")).strip() or None
            if expression:
                expression = self._normalize_expression_syntax(expression)
            if not hypothesis_id and expression:
                hypothesis_id = notebook.hypothesis_registry.resolve(expression=expression).hypothesis_id
            if not hypothesis_id:
                raise ValueError("manage_hypotheses accept/reject 需要 hypothesis_id 或已登记 expression。")
            validation_ids = self._validation_ids_for_hypothesis_decision(
                notebook=notebook,
                hypothesis_id=hypothesis_id,
                decision=decision,
                params=params,
            )
            if decision == "accepted":
                self._validate_accept_scope(
                    notebook=notebook,
                    hypothesis_id=hypothesis_id,
                    validation_ids=validation_ids,
                    params=params,
                )
            validation_summary = self._summarize_validations_for_decision(notebook, validation_ids)
            record = notebook.hypothesis_registry.decide(
                hypothesis_id=hypothesis_id,
                decision=decision,
                step_index=step_index,
                evidence_type="validation",
                experiment_ids=validation_summary["experiment_ids"],
                metric_name=validation_summary["metric_name"],
                metric_values=validation_summary["metric_values"],
                aggregate_score=validation_summary["aggregate_score"],
                summary=str(params.get("summary", validation_summary["summary"])),
                note=str(params.get("note", params.get("notes", ""))),
                validation_ids=validation_ids,
            )
            verb = "接受" if record.status == "accepted" else "拒绝" if record.status == "rejected" else "保持提出"
            return f"已{verb}候选规律 {record.hypothesis_id}: status={record.status}, validations={validation_ids}。"
        raise ValueError(f"manage_hypotheses 不支持 operation={operation!r}。可用: propose, accept, reject, list。")

    def _manage_hypothesis_decision(self, operation: str, params: dict[str, Any]) -> str:
        if operation in {"record_evidence", "evidence", "record"}:
            return "accepted" if self._coerce_bool(params.get("supports"), True) else "rejected"
        text = str(params.get("status", operation)).strip().lower()
        if text in {
            "accept",
            "accepted",
            "approve",
            "approved",
            "confirm",
            "confirmed",
            "support",
            "supported",
            "validate",
            "validated",
            "verify",
            "verified",
            "final",
            "finalize",
            "finish",
            "pass",
            "passed",
        }:
            return "accepted"
        if text in {
            "reject",
            "rejected",
            "refute",
            "refuted",
            "fail",
            "failed",
            "deny",
            "denied",
            "falsify",
            "falsified",
            "invalid",
        }:
            return "rejected"
        if text in {"propose", "proposed"}:
            return "proposed"
        raise ValueError(f"无法把 {text!r} 解释为 accept 或 reject。")

    def _validation_ids_for_hypothesis_decision(
        self,
        *,
        notebook: ScientificNotebook,
        hypothesis_id: str,
        decision: str,
        params: dict[str, Any],
    ) -> list[str]:
        if decision == "proposed":
            return []
        requested = (
            params.get("validation_ids")
            or params.get("validation_id")
            or params.get("source_validation_ids")
            or params.get("source_validation_id")
        )
        if not requested:
            references = params.get("references", params.get("refs"))
            reference_ids = [
                item
                for item in self._coerce_string_list(references)
                if item.upper().startswith("VAL")
            ]
            requested = reference_ids
        validation_ids = notebook.resolve_validation_ids(requested)
        if not validation_ids:
            latest = notebook.latest_validation_for_hypothesis(hypothesis_id)
            if latest is not None:
                validation_ids = [latest.validation_id]
        if not validation_ids:
            raise ValueError(
                "manage_hypotheses accept/reject 必须引用 VAL。"
                "请先调用 analyze_data(mode=validate_hypothesis)。"
            )
        selected = [validation for validation in notebook.validations if validation.validation_id in set(validation_ids)]
        mismatched = [
            validation.validation_id
            for validation in selected
            if validation.hypothesis_id and validation.hypothesis_id != hypothesis_id
        ]
        if mismatched:
            raise ValueError(f"validation_ids {mismatched} 不属于 hypothesis_id={hypothesis_id}。")
        expected_support = decision == "accepted"
        if not any(validation.supports == expected_support for validation in selected):
            expected_text = "支持" if expected_support else "反驳"
            raise ValueError(f"要执行 {decision}，至少需要一个 {expected_text} 该假说的 VAL。")
        return validation_ids

    def _summarize_validations_for_decision(
        self,
        notebook: ScientificNotebook,
        validation_ids: list[str],
    ) -> dict[str, Any]:
        selected = [validation for validation in notebook.validations if validation.validation_id in set(validation_ids)]
        experiment_ids: list[str] = []
        metric_values: dict[str, float] = {}
        scores: list[float] = []
        summaries: list[str] = []
        metric_name = "validation_metric"
        for validation in selected:
            for experiment_id in validation.experiment_ids:
                if experiment_id not in experiment_ids:
                    experiment_ids.append(experiment_id)
            metric_name = validation.metric_name or metric_name
            for key, value in validation.metric_values.items():
                metric_values[f"{validation.validation_id}.{key}"] = value
            if validation.aggregate_score is not None:
                scores.append(float(validation.aggregate_score))
            summaries.append(f"{validation.validation_id}: {validation.summary}")
        return {
            "experiment_ids": experiment_ids,
            "metric_name": metric_name,
            "metric_values": metric_values,
            "aggregate_score": min(scores) if scores else None,
            "summary": " | ".join(summaries),
        }

    def _execute_generated_data_action(
        self,
        *,
        notebook: ScientificNotebook,
        step_index: int,
        action: str,
        params: dict[str, Any],
        fallback: Callable[[], str],
    ) -> str:
        action_spec = self.tool_specs.get(action)
        if action_spec is not None and not action_spec.generated_code_allowed:
            return fallback() if self.allow_data_processing_fallback else f"动作 `{action}` 不允许生成代码路径。"
        if not self.use_generated_processors or self.data_brain is None:
            if self.allow_data_processing_fallback:
                return fallback()
            return (
                "数据处理 LLM 路径不可用，默认不使用其他数据处理函数。"
                "请启用 USE_GENERATED_PROCESSORS=true 并配置 DataProcessingBrain/API。"
            )

        description = f"step {step_index}: {action} with params={params}"
        notebook_context = self._summarize_notebook_for_data_brain(notebook, params)
        payload = self._build_generated_processor_payload(notebook=notebook, action=action, params=params)
        recent_processors = self._compact_recent_processors(self.generated_code_registry.recent_records(limit=5))
        processor_code: str | None = None
        code_path: Path | None = None
        try:
            processor_code = self.data_brain.write_processor_code(
                action=action,
                parameters=params,
                notebook_context=notebook_context,
                recent_processors=recent_processors,
            )
            code_path, result, series, obs_ids, val_ids = self._run_generated_processor_attempt(
                notebook=notebook,
                step_index=step_index,
                action=action,
                processor_code=processor_code,
                payload=payload,
                repaired=False,
            )
            return self._record_generated_processor_success(
                step_index=step_index,
                action=action,
                code_path=code_path,
                description=description,
                result=result,
                registered_series=series,
                observation_ids=obs_ids,
                validation_ids=val_ids,
            )
        except Exception as first_exc:
            first_error = str(first_exc)
            if code_path is not None:
                self.generated_code_registry.record_failure(
                    step_index=step_index,
                    action=action,
                    code_path=code_path,
                    description=description,
                    error=first_error,
                )
            if processor_code is not None and hasattr(self.data_brain, "repair_processor_code"):
                repaired_path: Path | None = None
                try:
                    repaired_code = self.data_brain.repair_processor_code(
                        action=action,
                        parameters=params,
                        notebook_context=notebook_context,
                        recent_processors=recent_processors,
                        failed_code=processor_code,
                        error=first_error,
                        payload_summary=self._summarize_generated_payload_for_repair(payload),
                    )
                    repaired_path, result, series, obs_ids, val_ids = self._run_generated_processor_attempt(
                        notebook=notebook,
                        step_index=step_index,
                        action=action,
                        processor_code=repaired_code,
                        payload=payload,
                        repaired=True,
                    )
                    observation = self._record_generated_processor_success(
                        step_index=step_index,
                        action=action,
                        code_path=repaired_path,
                        description=description + " (auto-repaired after first failure)",
                        result=result,
                        registered_series=series,
                        observation_ids=obs_ids,
                        validation_ids=val_ids,
                    )
                    return f"数据处理 LLM 首次生成代码失败，已自动修复并重试成功。首次失败原因: {first_error}\n{observation}"
                except Exception as repair_exc:
                    if repaired_path is not None:
                        self.generated_code_registry.record_failure(
                            step_index=step_index,
                            action=action,
                            code_path=repaired_path,
                            description=description + " (auto-repair attempt)",
                            error=str(repair_exc),
                        )
                    first_error = f"{first_error}; 自动修复重试仍失败: {repair_exc}"
            return (
                "数据处理 LLM 路径失败，默认不使用其他数据处理函数回退。"
                f"失败原因: {first_error}\n请基于失败反馈重新规划。"
            )

    def _run_generated_processor_attempt(
        self,
        *,
        notebook: ScientificNotebook,
        step_index: int,
        action: str,
        processor_code: str,
        payload: dict[str, Any],
        repaired: bool,
    ) -> tuple[Path, GeneratedProcessorResult, list[str], list[str], list[str]]:
        code_action = f"{action}_repair" if repaired else action
        code_path = self.generated_code_runner.save_processor(
            code=processor_code,
            step_index=step_index,
            action=code_action,
        )
        result = self.generated_code_runner.run_processor(code_path=code_path, payload=payload)
        series, obs_ids, val_ids = self._register_generated_processor_result(
            notebook=notebook,
            result=result,
            code_path=code_path,
            step_index=step_index,
            params=payload.get("parameters", {}),
        )
        return code_path, result, series, obs_ids, val_ids

    def _record_generated_processor_success(
        self,
        *,
        step_index: int,
        action: str,
        code_path: Path,
        description: str,
        result: GeneratedProcessorResult,
        registered_series: list[str],
        observation_ids: list[str],
        validation_ids: list[str],
    ) -> str:
        self.generated_code_registry.record_success(
            step_index=step_index,
            action=action,
            code_path=code_path,
            description=description,
            observation=result.observation,
            metrics=result.metrics,
            derived_series=registered_series,
            figures=result.figures,
        )
        suffix = [f"数据处理代码={code_path}"]
        if registered_series:
            suffix.append(f"新增序列={registered_series}")
        if observation_ids:
            suffix.append(f"新增OBS={observation_ids}")
        if validation_ids:
            suffix.append(f"新增VAL={validation_ids}")
        if result.figures:
            suffix.append(f"图像={result.figures}")
        return result.observation + "\n" + "；".join(suffix)

    def _build_generated_processor_payload(
        self,
        *,
        notebook: ScientificNotebook,
        action: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        output_dir = self.generated_code_dir / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        experiments: dict[str, Any] = {}
        for experiment_id, record in sorted(notebook.experiments.items()):
            series_payload: dict[str, list[float]] = {
                "t": self._array_to_json_list(record.result.t),
                "q": self._array_to_json_list(record.result.q),
            }
            for series_name, series in sorted(notebook.derived_series.get(experiment_id, {}).items()):
                series_payload[series_name] = self._array_to_json_list(series.values)
            force_value = self._experiment_force_value(record)
            metadata = dict(record.result.metadata)
            metadata["F_ext"] = force_value
            metadata["raw_constant_force"] = metadata.get("constant_force", record.config.constant_force)
            metadata["constant_force"] = force_value
            experiments[experiment_id] = {
                "config": {
                    "initial_q": record.config.initial_q,
                    "initial_v": record.config.initial_v,
                    "force_field_type": record.config.force_field_type.value,
                    "t_span": list(record.config.t_span),
                    "dt": record.config.dt,
                    "noise_std": record.config.noise_std,
                    "constant_force": force_value,
                    "raw_constant_force": record.config.constant_force,
                    "F_ext": force_value,
                },
                "metadata": metadata,
                "available_series": notebook.available_series(experiment_id),
                "series": series_payload,
            }
        return {
            "action": action,
            "parameters": params,
            "experiments": experiments,
            "observations": [
                {
                    "observation_id": obs.observation_id,
                    "step_index": obs.step_index,
                    "summary": obs.summary,
                    "source_data_refs": obs.source_data_refs,
                    "metrics": obs.metrics,
                    "figures": obs.figures,
                }
                for obs in notebook.observations
            ],
            "validations": [
                {
                    "validation_id": val.validation_id,
                    "step_index": val.step_index,
                    "hypothesis_id": val.hypothesis_id,
                    "experiment_ids": val.experiment_ids,
                    "supports": val.supports,
                    "metric_name": val.metric_name,
                    "metric_values": val.metric_values,
                    "aggregate_score": val.aggregate_score,
                    "summary": val.summary,
                    "source_data_refs": val.source_data_refs,
                    "figures": val.figures,
                }
                for val in notebook.validations
            ],
            "hypotheses": [
                {
                    "hypothesis_id": record.hypothesis_id,
                    "expression": record.expression,
                    "status": record.status,
                    "readable_summary": record.readable_summary,
                    "source_data_refs": record.source_data_refs,
                    "observation_ids": record.observation_ids,
                    "validation_ids": record.validation_ids,
                }
                for record in notebook.hypothesis_registry.all_records()
            ],
            "output_dir": str(output_dir),
        }

    def _summarize_generated_payload_for_repair(self, payload: dict[str, Any]) -> dict[str, Any]:
        experiments: dict[str, Any] = {}
        for experiment_id, experiment in sorted(payload.get("experiments", {}).items()):
            series = experiment.get("series", {})
            experiments[experiment_id] = {
                "config": experiment.get("config", {}),
                "available_series": experiment.get("available_series", []),
                "series_lengths": {
                    name: len(values) if hasattr(values, "__len__") else None
                    for name, values in sorted(series.items())
                },
            }
        return {
            "action": payload.get("action"),
            "parameters": payload.get("parameters"),
            "experiment_count": len(experiments),
            "experiments": experiments,
            "observation_count": len(payload.get("observations", [])),
            "validation_count": len(payload.get("validations", [])),
            "hypotheses": payload.get("hypotheses", []),
            "output_dir": payload.get("output_dir"),
        }

    def _compact_recent_processors(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for record in records:
            compact.append(
                {
                    "step_index": record.get("step_index"),
                    "action": record.get("action"),
                    "status": record.get("status"),
                    "description": self._truncate_text(str(record.get("description", "")), 220),
                    "observation": self._truncate_text(str(record.get("observation", record.get("error", ""))), 360),
                    "derived_series": list(record.get("derived_series") or [])[:12],
                    "figure_count": len(record.get("figures") or []),
                    "metric_keys": sorted(list((record.get("metrics") or {}).keys()))[:12]
                    if isinstance(record.get("metrics"), dict)
                    else [],
                }
            )
        return compact

    def _summarize_notebook_for_data_brain(self, notebook: ScientificNotebook, params: dict[str, Any]) -> str:
        if not notebook.experiments:
            return "当前没有实验数据。"
        lines: list[str] = []
        try:
            requested_ids = self._resolve_experiment_ids_for_action(notebook, params.get("experiment_ids"))
        except Exception:
            requested_ids = sorted(notebook.experiments.keys())
        if not requested_ids:
            requested_ids = sorted(notebook.experiments.keys())
        force_values = sorted({round(self._experiment_force_value(record), 9) for record in notebook.experiments.values()})
        initial_velocities = sorted({round(float(record.config.initial_v), 9) for record in notebook.experiments.values()})
        lines.append(
            f"全局覆盖: experiments={len(notebook.experiments)}, "
            f"F_ext_values={force_values[:12]}{' ...' if len(force_values) > 12 else ''}, "
            f"initial_v_values={initial_velocities[:12]}{' ...' if len(initial_velocities) > 12 else ''}"
        )
        lines.append(f"本次重点 experiment_ids={requested_ids}")
        for experiment_id in requested_ids:
            record = notebook.experiments[experiment_id]
            available_series = notebook.available_series(experiment_id)
            lines.append(
                f"{experiment_id}: force_field_type={record.config.force_field_type.value}, "
                f"F_ext={self._experiment_force_value(record)}, q0={record.config.initial_q}, "
                f"v0={record.config.initial_v}, dt={record.config.dt}, points={len(record.result.t)}"
            )
            lines.append(f"  available_series={self._format_name_list(available_series, limit=16)}")
            t = record.result.t
            for series_name in available_series[:8]:
                try:
                    values = notebook.get_series_values(experiment_id, series_name)
                    summary = self._summarize_series_text(t=t, values=values, name=series_name)
                except Exception as exc:
                    summary = f"{series_name}: 无法摘要 ({exc})"
                lines.append(f"  - {summary}")
            if len(available_series) > 8:
                lines.append(f"  - omitted_series_summaries={len(available_series) - 8}")
        if notebook.observations:
            lines.append("OBS:")
            relevant_obs = self._selected_ledger_observations_for_data_context(notebook, requested_ids)
            for observation in relevant_obs:
                lines.append(
                    f"  - {observation.observation_id}: metrics={self._format_metric_brief_for_agent(observation.metrics)}; "
                    f"{self._truncate_text(observation.summary, 360)}"
                )
        if notebook.validations:
            lines.append("VAL:")
            for validation in notebook.validations[-8:]:
                verdict = "supports" if validation.supports else "refutes"
                lines.append(
                    f"  - {validation.validation_id}: {verdict} {validation.hypothesis_id}; "
                    f"n_exp={len(validation.experiment_ids)}; metric={validation.metric_name}; score={validation.aggregate_score}; "
                    f"metrics={self._format_metric_brief_for_agent(validation.metric_values)}"
                )
        records = notebook.hypothesis_registry.all_records()
        if records:
            lines.append("Hypotheses:")
            for record in records:
                lines.append(f"  - {record.hypothesis_id}: status={record.status}; expression={record.expression}")
        return "\n".join(lines)

    def _selected_ledger_observations_for_data_context(
        self,
        notebook: ScientificNotebook,
        experiment_ids: list[str],
        limit: int = 12,
    ) -> list[LedgerObservation]:
        selected: list[LedgerObservation] = []
        experiment_prefixes = tuple(f"{experiment_id}:" for experiment_id in experiment_ids)
        for observation in notebook.observations:
            if any(str(ref).startswith(experiment_prefixes) for ref in observation.source_data_refs):
                selected.append(observation)
        selected.extend(notebook.observations[-6:])
        deduped: list[LedgerObservation] = []
        seen: set[str] = set()
        for observation in selected:
            if observation.observation_id not in seen:
                seen.add(observation.observation_id)
                deduped.append(observation)
        return deduped[-limit:]

    def _format_metric_brief_for_agent(self, metrics: dict[str, Any], limit: int = 6) -> str:
        flattened = self._flatten_numeric_metrics(metrics)
        if not flattened:
            return "{}"
        priority = sorted(
            flattened.items(),
            key=lambda item: (
                0 if any(marker in item[0].lower() for marker in ("r2", "r²", "rmse", "mae", "score", "gamma", "alpha")) else 1,
                item[0],
            ),
        )[:limit]
        return "{" + ", ".join(f"{key}={value:.4g}" for key, value in priority) + "}"

    def _register_generated_processor_result(
        self,
        *,
        notebook: ScientificNotebook,
        result: GeneratedProcessorResult,
        code_path: Path,
        step_index: int,
        params: dict[str, Any],
    ) -> tuple[list[str], list[str], list[str]]:
        registered_series: list[str] = []
        seen_series: set[tuple[str, str]] = set()
        for item in result.derived_series:
            experiment_id = self._resolve_experiment_id(notebook, item.get("experiment_id"))
            raw_name = item.get("name")
            if raw_name is None:
                raise ValueError("generated derived_series 缺少 name。")
            name = self._normalize_output_name(raw_name, default="generated_series")
            if name in {"q", "t"}:
                raise ValueError("generated derived_series 不能覆盖原始序列 q/t。")
            series_key = (experiment_id, name)
            if series_key in seen_series:
                continue
            seen_series.add(series_key)
            values = self._coerce_generated_series_values(
                notebook=notebook,
                experiment_id=experiment_id,
                values=item.get("values"),
            )
            existing = notebook.derived_series.get(experiment_id, {}).get(name)
            if existing is not None:
                if self._same_series_values(existing.values, values):
                    continue
                raise ValueError(
                    f"{experiment_id}:{name} 已存在且数值不同。"
                    "请为新的派生量定义使用不同名称，并在 provenance 中说明差异。"
                )
            t = notebook.get_series_values(experiment_id, "t")
            summary = self._summarize_series_text(t=t, values=values, name=name)
            notebook.register_series(
                DerivedSeries(
                    experiment_id=experiment_id,
                    name=name,
                    values=values,
                    source_name=str(item.get("source_name", f"generated by {code_path.name}")),
                    provenance=str(item.get("provenance", f"generated data processor: {code_path.name}")),
                    summary_text=summary,
                )
            )
            registered_series.append(f"{experiment_id}:{name}")
        obs_ids = self._register_generated_observations(notebook, result, step_index, params)
        val_ids = self._register_generated_validations(notebook, result, step_index, params)
        if result.metrics:
            notebook.notes.append(f"生成代码 `{code_path}` 返回 metrics: {result.metrics}")
        if result.figures:
            notebook.notes.append(f"生成代码 `{code_path}` 返回 figures: {result.figures}")
        return registered_series, obs_ids, val_ids

    def _register_generated_observations(
        self,
        notebook: ScientificNotebook,
        result: GeneratedProcessorResult,
        step_index: int,
        params: dict[str, Any],
    ) -> list[str]:
        raw = result.raw_payload.get("observations", [])
        observations = raw if isinstance(raw, list) else []
        ids: list[str] = []
        for item in observations:
            if not isinstance(item, dict):
                continue
            observation = notebook.register_observation(
                step_index=step_index,
                summary=str(item.get("summary", item.get("observation", ""))).strip() or result.observation,
                source_data_refs=self._coerce_string_list(item.get("source_data_refs")),
                metrics=self._coerce_json_object(item.get("metrics", item.get("numeric_facts", {}))),
                figures=self._coerce_string_list(item.get("figures", result.figures)),
            )
            ids.append(observation.observation_id)
        if not ids and _is_ledger_maintenance_mode(params.get("analysis_mode")):
            observation = notebook.register_observation(
                step_index=step_index,
                summary=result.observation,
                source_data_refs=self._default_source_data_refs(notebook, params),
                metrics=dict(result.metrics),
                figures=list(result.figures),
            )
            ids.append(observation.observation_id)
        return ids

    def _register_generated_validations(
        self,
        notebook: ScientificNotebook,
        result: GeneratedProcessorResult,
        step_index: int,
        params: dict[str, Any],
    ) -> list[str]:
        raw = result.raw_payload.get("validations", [])
        validations = raw if isinstance(raw, list) else []
        ids: list[str] = []
        for item in validations:
            if not isinstance(item, dict):
                continue
            experiment_ids = self._coerce_string_list(item.get("experiment_ids", item.get("experiments")))
            if not experiment_ids:
                experiment_ids = self._validation_experiment_ids(notebook, params)
            else:
                experiment_ids = self._resolve_experiment_ids_for_action(notebook, experiment_ids)
            hypothesis_id = str(item.get("hypothesis_id", params.get("hypothesis_id", ""))).strip()
            if not hypothesis_id and params.get("candidate_expression"):
                record, _ = notebook.hypothesis_registry.propose(
                    expression=self._normalize_expression_syntax(str(params.get("candidate_expression"))),
                    step_index=step_index,
                    origin_action="analyze_data validate_hypothesis",
                )
                hypothesis_id = record.hypothesis_id
            validation = notebook.register_validation(
                step_index=step_index,
                hypothesis_id=hypothesis_id,
                experiment_ids=experiment_ids,
                supports=self._coerce_bool(item.get("supports", item.get("support")), True),
                metric_name=str(item.get("metric_name", item.get("metric", "validation_metric"))),
                metric_values=self._coerce_metric_values(item.get("metric_values", item.get("metrics", {}))),
                aggregate_score=self._coerce_optional_float(item.get("aggregate_score", item.get("score"))),
                summary=str(item.get("summary", result.observation)),
                source_data_refs=self._coerce_string_list(item.get("source_data_refs")),
                figures=self._coerce_string_list(item.get("figures", result.figures)),
            )
            ids.append(validation.validation_id)
        if not ids and _is_hypothesis_validation_mode(params.get("analysis_mode")):
            hypothesis_id = str(params.get("hypothesis_id", "")).strip()
            if not hypothesis_id and params.get("candidate_expression"):
                record, _ = notebook.hypothesis_registry.propose(
                    expression=self._normalize_expression_syntax(str(params.get("candidate_expression"))),
                    step_index=step_index,
                    origin_action="analyze_data validate_hypothesis",
                )
                hypothesis_id = record.hypothesis_id
            if not hypothesis_id:
                raise ValueError("validate_hypothesis 结果缺少 hypothesis_id。")
            validation = notebook.register_validation(
                step_index=step_index,
                hypothesis_id=hypothesis_id,
                experiment_ids=self._validation_experiment_ids(notebook, params),
                supports=self._coerce_bool(result.metrics.get("supports"), True),
                metric_name=str(result.metrics.get("metric_name", "validation_metric")),
                metric_values=self._coerce_metric_values(result.metrics.get("metric_values", result.metrics)),
                aggregate_score=self._coerce_optional_float(result.metrics.get("aggregate_score", result.metrics.get("rmse"))),
                summary=result.observation,
                source_data_refs=self._default_source_data_refs(notebook, params),
                figures=list(result.figures),
            )
            ids.append(validation.validation_id)
        return ids

    def _coerce_generated_series_values(
        self,
        *,
        notebook: ScientificNotebook,
        experiment_id: str,
        values: Any,
    ) -> np.ndarray:
        if values is None:
            raise ValueError("generated derived_series 缺少 values。")
        values_array = np.asarray(values, dtype=float).reshape(-1)
        expected_length = len(notebook.get_series_values(experiment_id, "t"))
        if len(values_array) != expected_length:
            raise ValueError(f"{experiment_id}: generated values 长度 {len(values_array)} 与 t 长度 {expected_length} 不一致。")
        return values_array

    def _build_inconclusive_law(self, notebook: ScientificNotebook, finished_by_finalize: bool) -> LawHypothesis:
        reason = "本轮没有形成 accepted 假说。" if not finished_by_finalize else "结束请求出现，但没有 accepted 假说。"
        evidence = "尚无 accepted 假说。"
        if notebook.validations:
            latest = notebook.validations[-1]
            evidence = f"最近一次验证为 `{latest.validation_id}`，hypothesis={latest.hypothesis_id}, score={latest.aggregate_score}。"
        return LawHypothesis(
            summary=f"当前探索尚未形成可接受的最终定律。{reason}",
            proposed_law="尚未形成 accepted 的最终动力学方程。",
            evidence=evidence,
            confidence="low",
            next_steps="继续维护实验数据记录表，提出可证伪假说，并通过 validate_hypothesis 生成 VAL。",
        )

    def _final_ready_hypotheses(self, notebook: ScientificNotebook) -> list[Any]:
        return [
            record
            for record in notebook.hypothesis_registry.all_records()
            if record.status == "accepted"
        ]

    def _should_finish_after_hypothesis_action(self, notebook: ScientificNotebook, params: dict[str, Any]) -> bool:
        operation = str(params.get("operation", "list")).strip().lower()
        status = str(params.get("status", "")).strip().lower()
        accept_aliases = {
            "accept",
            "accepted",
            "approve",
            "approved",
            "support",
            "supported",
            "confirm",
            "confirmed",
            "validate",
            "validated",
            "verify",
            "verified",
            "finalize",
            "finish",
            "pass",
            "passed",
        }
        record_evidence_accepts = operation in {"record_evidence", "evidence", "record"} and self._coerce_bool(params.get("supports"), True)
        if operation not in accept_aliases and status not in accept_aliases and not record_evidence_accepts:
            return False
        return bool(self._final_ready_hypotheses(notebook))

    def _summarize_trajectory(self, result: ExperimentResult) -> dict[str, float]:
        return {
            "num_points": float(len(result.t)),
            "q_min": float(np.min(result.q)),
            "q_max": float(np.max(result.q)),
            "mean_q": float(np.mean(result.q)),
        }

    def _summarize_series_text(self, *, t: np.ndarray, values: np.ndarray, name: str) -> str:
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return f"{name}: count={arr.size}, no finite values"
        text = (
            f"{name}: count={arr.size}, min={float(np.min(finite)):.6g}, "
            f"max={float(np.max(finite)):.6g}, mean={float(np.mean(finite)):.6g}, "
            f"std={float(np.std(finite)):.6g}, first={float(arr[0]):.6g}, last={float(arr[-1]):.6g}"
        )
        if len(t) == arr.size and arr.size >= 2:
            duration = float(np.asarray(t, dtype=float)[-1] - np.asarray(t, dtype=float)[0])
            if abs(duration) > 1e-12:
                text += f", endpoint_slope={float((arr[-1] - arr[0]) / duration):.6g}"
        return text

    def _truncate_text(self, text: str, max_chars: int) -> str:
        normalized = re.sub(r"\s+", " ", str(text)).strip()
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."

    def _format_name_list(self, names: list[str], limit: int = 10) -> str:
        shown = names[:limit]
        suffix = f", ...(+{len(names) - limit})" if len(names) > limit else ""
        return "[" + ", ".join(shown) + suffix + "]"

    def _same_series_values(self, left: np.ndarray, right: np.ndarray) -> bool:
        left_array = np.asarray(left, dtype=float).reshape(-1)
        right_array = np.asarray(right, dtype=float).reshape(-1)
        if left_array.shape != right_array.shape:
            return False
        return bool(np.allclose(left_array, right_array, rtol=1e-10, atol=1e-12, equal_nan=True))

    def _unmaintained_experiment_ids(self, notebook: ScientificNotebook) -> list[str]:
        ids: list[str] = []
        for experiment_id in sorted(notebook.experiments.keys()):
            has_series = bool(notebook.derived_series.get(experiment_id))
            has_observation = any(
                any(str(ref).startswith(f"{experiment_id}:") for ref in observation.source_data_refs)
                for observation in notebook.observations
            )
            if not has_series and not has_observation:
                ids.append(experiment_id)
        return ids

    def _validate_accept_scope(
        self,
        *,
        notebook: ScientificNotebook,
        hypothesis_id: str,
        validation_ids: list[str],
        params: dict[str, Any],
    ) -> None:
        all_control_values = self._control_values_for_experiments(
            notebook=notebook,
            experiment_ids=sorted(notebook.experiments.keys()),
        )
        validation_experiment_ids = [
            experiment_id
            for validation in notebook.validations
            if validation.validation_id in set(validation_ids)
            for experiment_id in validation.experiment_ids
        ]
        selected_validations = [
            validation
            for validation in notebook.validations
            if validation.validation_id in set(validation_ids)
        ]
        if not selected_validations or any(not validation.supports for validation in selected_validations):
            raise ValueError("accepted 假说只能引用支持该假说的 VAL。")
        unique_validation_experiment_ids = list(dict.fromkeys(validation_experiment_ids))
        if len(notebook.experiments) < 5:
            raise ValueError(
                "accepted 假说需要更充分的实验设计。当前实验数少于 5，"
                "请至少补充不同外力和不同初速度实验后再接受。"
            )
        minimum_validation_count = min(5, len(notebook.experiments))
        if len(unique_validation_experiment_ids) < minimum_validation_count:
            raise ValueError(
                f"accepted 假说至少需要覆盖 {minimum_validation_count} 个实验的 VAL，"
                f"当前只覆盖 {len(unique_validation_experiment_ids)} 个。请做跨实验验证。"
            )
        validation_control_values = self._control_values_for_experiments(
            notebook=notebook,
            experiment_ids=unique_validation_experiment_ids,
        )
        if len(all_control_values) <= 1:
            raise ValueError(
                "accepted 假说必须足以回答当前研究目标。当前实验尚未覆盖至少两个控制条件，"
                "请先补充非零外力或其他控制条件实验。"
            )
        if len(validation_control_values) < 2:
            raise ValueError(
                "accepted 假说必须足以回答当前研究目标。当前已有多个控制条件，"
                "但该 VAL 只覆盖单一控制值；请继续做跨控制条件验证。"
            )
        try:
            record = notebook.hypothesis_registry.resolve(hypothesis_id=hypothesis_id)
        except ValueError:
            return
        self._validate_accept_metrics(selected_validations)
        if self._expression_mentions_velocity(record.expression):
            validation_initial_velocities = self._initial_velocity_values_for_experiments(
                notebook=notebook,
                experiment_ids=unique_validation_experiment_ids,
            )
            if len(validation_initial_velocities) < 2:
                raise ValueError(
                    "该假说包含速度项，accepted 前必须用不同初始速度条件做验证。"
                )
        text = " ".join(
            [
                record.expression,
                record.readable_summary,
                " ".join(record.assumptions),
                str(params.get("summary", "")),
            ]
        ).lower()
        local_scope_markers = (
            "free",
            "无外力",
            "自由场",
            "f_ext=0",
            "f_ext = 0",
            "constant_force=0",
            "force=0",
            "control=0",
        )
        mentions_local_scope = any(marker in text for marker in local_scope_markers)
        mentions_control_variable = any(marker in text for marker in ("f_ext", "force", "control", "外力"))
        if mentions_local_scope or not mentions_control_variable:
            raise ValueError(
                "该假说目前只覆盖单一控制子域，不能作为回答当前研究目标的 accepted 规律。"
                "请把它保留为 proposed/OBS，继续提出并验证能覆盖多个控制条件的更一般假说。"
            )

    def _validate_accept_metrics(self, validations: list[LedgerValidation]) -> None:
        r2_values: list[float] = []
        rmse_values: list[tuple[str, float]] = []
        trajectory_rmse_values: list[float] = []
        metric_name_text = " ".join(validation.metric_name.lower() for validation in validations)
        summary_text = " ".join(validation.summary.lower() for validation in validations)
        for validation in validations:
            flattened = dict(validation.metric_values)
            if validation.aggregate_score is not None and any(marker in validation.metric_name.lower() for marker in ("r2", "r²", "score")):
                flattened[f"{validation.validation_id}.aggregate_score"] = validation.aggregate_score
            for key, value in flattened.items():
                key_text = str(key).lower()
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(numeric):
                    continue
                if "r2" in key_text or "r²" in key_text:
                    r2_values.append(numeric)
                if "rmse" in key_text:
                    rmse_values.append((key_text, abs(numeric)))
                    if any(marker in key_text for marker in ("position", "trajectory", "q_", "q.", "qrmse")):
                        trajectory_rmse_values.append(abs(numeric))
        if r2_values and max(r2_values) < 0.99:
            raise ValueError(
                f"accepted 假说需要高质量跨实验验证；当前最佳 R²={max(r2_values):.6g} < 0.99。"
            )
        if not r2_values and not rmse_values:
            raise ValueError("accepted 假说必须提供 R² 或 RMSE 等可量化 VAL 指标。")
        if rmse_values:
            max_rmse = max(value for _, value in rmse_values)
            if max_rmse > 0.1:
                raise ValueError(
                    f"accepted 假说的 RMSE 过大；当前最大 RMSE={max_rmse:.6g} > 0.1。"
                )
        needs_trajectory_validation = any(marker in metric_name_text or marker in summary_text for marker in ("integr", "积分", "trajectory", "position", "位置", "q预测", "q prediction"))
        if needs_trajectory_validation and trajectory_rmse_values and max(trajectory_rmse_values) > 0.05:
            raise ValueError(
                f"轨迹/位置级验证 RMSE 过大；当前最大位置 RMSE={max(trajectory_rmse_values):.6g} > 0.05。"
            )
        if not trajectory_rmse_values and r2_values:
            # Derivative-level fits are useful evidence, but the final law should eventually survive a
            # trajectory/position check or an extremely tight residual check.
            non_position_rmse = [value for key, value in rmse_values if not any(marker in key for marker in ("position", "trajectory", "q_", "q.", "qrmse"))]
            if non_position_rmse and max(non_position_rmse) > 0.02:
                raise ValueError(
                    "当前只是导数/代数层面的拟合，且残差不够小；"
                    f"最大非位置 RMSE={max(non_position_rmse):.6g} > 0.02。请继续做轨迹预测或更严格残差验证。"
                )

    def _control_values_for_experiments(
        self,
        *,
        notebook: ScientificNotebook,
        experiment_ids: list[str],
    ) -> set[float]:
        values: set[float] = set()
        for experiment_id in experiment_ids:
            record = notebook.experiments.get(experiment_id)
            if record is None:
                continue
            values.add(round(self._experiment_force_value(record), 9))
        return values

    def _initial_velocity_values_for_experiments(
        self,
        *,
        notebook: ScientificNotebook,
        experiment_ids: list[str],
    ) -> set[float]:
        values: set[float] = set()
        for experiment_id in experiment_ids:
            record = notebook.experiments.get(experiment_id)
            if record is None:
                continue
            values.add(round(float(record.config.initial_v), 9))
        return values

    def _expression_mentions_velocity(self, expression: str) -> bool:
        return bool(re.search(r"\bv\b|velocity|速度", str(expression), flags=re.IGNORECASE))

    def _experiment_force_value(self, record: ExperimentRecord) -> float:
        if record.config.force_field_type is ForceFieldType.CONSTANT:
            return float(record.config.constant_force)
        return 0.0

    def _resolve_experiment_id(self, notebook: ScientificNotebook, requested: Any) -> str:
        if requested is not None:
            for candidate in self._experiment_id_candidates(requested):
                if candidate in notebook.experiments:
                    return candidate
            available = ", ".join(sorted(notebook.experiments.keys())) or "无"
            raise ValueError(f"未知实验 ID `{requested}`。可用实验 ID: {available}。")
        latest = notebook.latest_experiment_id()
        if latest is None:
            raise ValueError("当前还没有任何实验。")
        return latest

    def _resolve_experiment_ids_for_action(
        self,
        notebook: ScientificNotebook,
        requested_experiment_ids: Any,
        fallback_experiment_id: Any = None,
    ) -> list[str]:
        if requested_experiment_ids is None:
            if fallback_experiment_id is None:
                return sorted(notebook.experiments.keys())
            return [self._resolve_experiment_id(notebook, fallback_experiment_id)]
        if isinstance(requested_experiment_ids, str):
            text = requested_experiment_ids.strip()
            if text.lower() in {"all", "*", "全部", "所有"}:
                raw_items = sorted(notebook.experiments.keys())
            else:
                raw_items = [item.strip() for item in text.split(",") if item.strip()]
        else:
            try:
                raw_items = list(requested_experiment_ids)
            except TypeError:
                raw_items = [requested_experiment_ids]
        resolved: list[str] = []
        for item in raw_items:
            experiment_id = self._resolve_experiment_id(notebook, item)
            if experiment_id not in resolved:
                resolved.append(experiment_id)
        if not resolved:
            raise ValueError("experiment_ids 为空。")
        return resolved

    def _merge_experiment_id_lists(self, *groups: list[str]) -> list[str]:
        merged: list[str] = []
        for group in groups:
            for experiment_id in group:
                if experiment_id not in merged:
                    merged.append(experiment_id)
        return merged

    def _validation_experiment_ids(self, notebook: ScientificNotebook, params: dict[str, Any]) -> list[str]:
        if params.get("experiment_ids") is not None:
            return self._resolve_experiment_ids_for_action(notebook, params.get("experiment_ids"))
        if params.get("experiment_id") is not None:
            return self._resolve_experiment_ids_for_action(notebook, [params.get("experiment_id")])
        return sorted(notebook.experiments.keys())

    def _format_batch_result(self, title: str, parts: list[str]) -> str:
        if len(parts) == 1:
            return parts[0]
        return f"{title}（{len(parts)} 个实验）:\n" + "\n".join(f"- {part}" for part in parts)

    def _run_experiment_parameter_warnings(self, params: dict[str, Any]) -> list[str]:
        allowed = {
            "initial_q",
            "q0",
            "initial_v",
            "v0",
            "force_field_type",
            "constant_force",
            "F_ext",
            "F",
            "t_end",
            "dt",
            "noise_std",
            "experiment_id",
        }
        warnings: list[str] = []
        unknown = sorted(str(key) for key in params if key not in allowed)
        if unknown:
            warnings.append(f"run_experiments 收到未知参数 {unknown}，已忽略")
        for canonical, alias in [("initial_q", "q0"), ("initial_v", "v0"), ("constant_force", "F_ext"), ("constant_force", "F")]:
            if alias in params:
                warnings.append(f"已将别名 `{alias}` 解析为 `{canonical}`")
        return warnings

    def _get_first_param(self, params: dict[str, Any], primary: str, alias: str) -> Any:
        return params.get(primary) if primary in params else params.get(alias)

    def _get_first_present_param(self, params: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in params:
                return params.get(key)
        return None

    def _experiment_id_candidates(self, requested: Any) -> list[str]:
        raw = str(requested).strip()
        if not raw:
            return []
        candidates = [raw]
        match = re.fullmatch(r"(?:exp[_-]?)?0*(\d+)", raw, flags=re.IGNORECASE)
        if match:
            numeric_id = int(match.group(1))
            candidates.extend([f"exp_{numeric_id:02d}", f"exp_{numeric_id}", str(numeric_id)])
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _resolve_force_field_type(self, value: Any) -> ForceFieldType:
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"constant", "constant_force", "constant_field", "force", "forced", "恒力", "恒定外力"}:
            return ForceFieldType.CONSTANT
        if normalized in {"free", "none", "no_force", "zero_force", "no_external_force", "unforced", "无外力", "自由"}:
            return ForceFieldType.FREE
        raise ValueError(f"未知 force_field_type `{value}`。")

    def _normalize_output_name(self, value: Any, default: str) -> str:
        raw = str(value or default).strip()
        normalized = re.sub(r"[^0-9A-Za-z_]+", "_", raw).strip("_") or default
        if normalized[0].isdigit():
            normalized = f"{default}_{normalized}"
        return normalized

    def _hypothesis_evidence_target(self, params: dict[str, Any]) -> str:
        hypothesis_id = str(params.get("hypothesis_id", "")).strip()
        candidate_expression = str(params.get("candidate_expression", "")).strip()
        hypothesis = params.get("hypothesis")
        if isinstance(hypothesis, dict):
            hypothesis_id = hypothesis_id or str(hypothesis.get("hypothesis_id", hypothesis.get("id", ""))).strip()
            candidate_expression = candidate_expression or str(hypothesis.get("candidate_expression", hypothesis.get("expression", ""))).strip()
        return hypothesis_id or candidate_expression

    def _attach_referenced_hypothesis_to_analysis(
        self,
        *,
        notebook: ScientificNotebook,
        params: dict[str, Any],
        thought: str,
    ) -> dict[str, Any]:
        if self._hypothesis_evidence_target(params):
            return params
        text = f"{thought}\n{params.get('analysis_goal', '')}"
        for match in re.finditer(r"\bH\d{1,4}\b", text, flags=re.IGNORECASE):
            raw_id = match.group(0).upper()
            candidates = [raw_id]
            number_match = re.fullmatch(r"H0*(\d+)", raw_id)
            if number_match:
                candidates.append(f"H{int(number_match.group(1)):03d}")
            for hypothesis_id in candidates:
                try:
                    notebook.hypothesis_registry.resolve(hypothesis_id=hypothesis_id)
                except ValueError:
                    continue
                return {**params, "hypothesis_id": hypothesis_id}
        return params

    def _normalize_manage_hypotheses_params(self, params: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(params)
        hypothesis = normalized.get("hypothesis")
        evidence = normalized.get("evidence")
        metrics = normalized.get("metrics")
        if isinstance(hypothesis, dict):
            self._set_missing(normalized, "hypothesis_id", hypothesis.get("hypothesis_id", hypothesis.get("id")))
            self._set_missing(normalized, "expression", hypothesis.get("expression", hypothesis.get("candidate_expression")))
            self._set_missing(normalized, "readable_summary", hypothesis.get("readable_summary", hypothesis.get("summary")))
            self._set_missing(normalized, "variables", hypothesis.get("variables"))
            self._set_missing(normalized, "assumptions", hypothesis.get("assumptions"))
            self._set_missing(normalized, "source_data_refs", hypothesis.get("source_data_refs"))
            self._set_missing(normalized, "observation_ids", hypothesis.get("observation_ids", hypothesis.get("source_observation_ids")))
            self._set_missing(normalized, "validation_ids", hypothesis.get("validation_ids", hypothesis.get("source_validation_ids")))
            self._set_missing(normalized, "next_tests", hypothesis.get("next_tests"))
            self._set_missing(normalized, "status", hypothesis.get("status"))
            self._set_missing(normalized, "note", hypothesis.get("note", hypothesis.get("notes")))
            nested_evidence = hypothesis.get("evidence")
            if isinstance(nested_evidence, list):
                evidence_refs: list[str] = []
                evidence_obs_ids: list[str] = []
                for item in nested_evidence:
                    if not isinstance(item, dict):
                        continue
                    evidence_refs.extend(self._coerce_string_list(item.get("source_data_refs")))
                    evidence_obs_ids.extend(
                        ref
                        for ref in self._coerce_string_list(item.get("observation_ids", item.get("source_observation_ids")))
                        if ref.upper().startswith("OBS")
                    )
                    evidence_obs_ids.extend(
                        ref
                        for ref in self._coerce_string_list(item.get("source_data_refs"))
                        if ref.upper().startswith("OBS")
                    )
                self._set_missing(normalized, "source_data_refs", evidence_refs)
                self._set_missing(normalized, "observation_ids", evidence_obs_ids)
        if isinstance(evidence, dict):
            self._set_missing(normalized, "validation_ids", evidence.get("validation_ids", evidence.get("validation_id")))
            self._set_missing(normalized, "source_validation_ids", evidence.get("source_validation_ids"))
            self._set_missing(normalized, "supports", evidence.get("supports", evidence.get("support")))
            self._set_missing(normalized, "status", evidence.get("status"))
            self._set_missing(normalized, "summary", evidence.get("summary", evidence.get("note")))
        if isinstance(metrics, dict):
            self._set_missing(normalized, "metric_values", self._flatten_numeric_metrics(metrics))
        self._set_missing(normalized, "expression", normalized.get("candidate_expression"))
        status_text = str(normalized.get("status", "")).strip().lower()
        if "operation" not in normalized:
            if status_text in {"reject", "rejected", "refute", "refuted", "fail", "failed", "deny", "denied", "falsify", "falsified", "invalid"}:
                normalized["operation"] = "reject"
            elif status_text in {"accept", "accepted", "approve", "approved", "support", "supported", "confirm", "confirmed", "validate", "validated", "verify", "verified", "pass", "passed"}:
                normalized["operation"] = "accept"
            elif "expression" in normalized:
                normalized["operation"] = "propose"
        return normalized

    def _set_missing(self, target: dict[str, Any], key: str, value: Any) -> None:
        if key not in target and value not in (None, "", []):
            target[key] = value

    def _flatten_numeric_metrics(self, value: Any, prefix: str = "") -> dict[str, float]:
        flattened: dict[str, float] = {}
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                flattened.update(self._flatten_numeric_metrics(item, child))
            return flattened
        if isinstance(value, bool):
            return {}
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if np.isfinite(numeric):
                flattened[prefix or "value"] = numeric
        return flattened

    def _coerce_json_object(self, value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    def _coerce_float(self, value: Any, default: float) -> float:
        if value is None or value == "":
            return default
        return float(value)

    def _coerce_optional_float(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    def _coerce_bool(self, value: Any, default: bool) -> bool:
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

    def _coerce_string_list(self, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [item.strip() for item in re.split(r"[,;\n]", value) if item.strip()]
        try:
            return [str(item).strip() for item in value if str(item).strip()]
        except TypeError:
            return [str(value).strip()]

    def _coerce_metric_values(self, value: Any) -> dict[str, float]:
        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            return self._flatten_numeric_metrics(value)
        if isinstance(value, (list, tuple)):
            flattened: dict[str, float] = {}
            for index, item in enumerate(value, start=1):
                flattened.update(self._flatten_numeric_metrics(item, f"value_{index}"))
            return flattened
        if isinstance(value, bool):
            return {}
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            return {"value": numeric} if np.isfinite(numeric) else {}
        raise ValueError("metric_values 必须是 dict、list 或数值。")

    def _validation_experiment_ids(self, notebook: ScientificNotebook, params: dict[str, Any]) -> list[str]:
        if params.get("experiment_ids") is not None:
            return self._resolve_experiment_ids_for_action(notebook, params.get("experiment_ids"))
        if params.get("experiment_id") is not None:
            return self._resolve_experiment_ids_for_action(notebook, [params.get("experiment_id")])
        return sorted(notebook.experiments.keys())

    def _default_source_data_refs(self, notebook: ScientificNotebook, params: dict[str, Any]) -> list[str]:
        try:
            experiment_ids = self._validation_experiment_ids(notebook, params)
        except Exception:
            experiment_ids = sorted(notebook.experiments.keys())
        series_names = self._coerce_string_list(params.get("optional_series"))
        refs: list[str] = []
        for experiment_id in experiment_ids:
            if series_names:
                refs.extend(f"{experiment_id}:{series}" for series in series_names)
            else:
                refs.extend(f"{experiment_id}:{series}" for series in notebook.available_series(experiment_id))
        return refs[:24]

    def _array_to_json_list(self, values: np.ndarray) -> list[float]:
        return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]

    def _normalize_expression_syntax(self, expression: str) -> str:
        return str(expression).strip().replace("^", "**")

    def _validate_specific_hypothesis_expression(self, expression: str) -> None:
        normalized = re.sub(r"\s+", " ", expression).strip().lower()
        for marker in [" 或 ", "或者", "待定", "可能", "maybe", "either", " or ", "and/or", "某种", "未知", "?", "？"]:
            if marker in normalized:
                raise ValueError(f"候选规律必须是单一、可证伪公式，不能包含 {marker!r}。")
        if len(normalized) > 180:
            raise ValueError("候选规律 expression 太长，请只放公式。")
