from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field


@dataclass(slots=True)
class HypothesisEvidence:
    """Evidence attached to one accept/reject decision."""

    step_index: int
    decision: str
    evidence_type: str
    experiment_ids: list[str]
    metric_name: str
    metric_values: dict[str, float]
    aggregate_score: float | None
    summary: str


@dataclass(slots=True)
class HypothesisRecord:
    """A candidate law with one of three states: proposed, accepted, rejected."""

    hypothesis_id: str
    expression: str
    normalized_expression: str
    readable_summary: str
    status: str
    origin_step: int
    last_updated_step: int
    variables: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    source_data_refs: list[str] = field(default_factory=list)
    observation_ids: list[str] = field(default_factory=list)
    validation_ids: list[str] = field(default_factory=list)
    evidence: list[HypothesisEvidence] = field(default_factory=list)
    next_tests: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    confidence_score: float = 0.0


class HypothesisRegistry:
    """Small state machine for candidate laws: propose, accept, or reject."""

    _STATUS_ORDER = {
        "accepted": 2,
        "proposed": 1,
        "rejected": 0,
    }
    _STATUS_ALIASES = {
        "propose": "proposed",
        "proposed": "proposed",
        "test": "proposed",
        "testing": "proposed",
        "weaken": "proposed",
        "weakened": "proposed",
        "weak": "proposed",
        "accept": "accepted",
        "accepted": "accepted",
        "approve": "accepted",
        "approved": "accepted",
        "confirm": "accepted",
        "confirmed": "accepted",
        "support": "accepted",
        "supported": "accepted",
        "validate": "accepted",
        "validated": "accepted",
        "verify": "accepted",
        "verified": "accepted",
        "final": "accepted",
        "finalize": "accepted",
        "finish": "accepted",
        "pass": "accepted",
        "passed": "accepted",
        "reject": "rejected",
        "rejected": "rejected",
        "refute": "rejected",
        "refuted": "rejected",
        "fail": "rejected",
        "failed": "rejected",
        "deny": "rejected",
        "denied": "rejected",
        "falsify": "rejected",
        "falsified": "rejected",
        "invalid": "rejected",
    }

    def __init__(self) -> None:
        self._records: dict[str, HypothesisRecord] = {}
        self._counter = 0

    def propose(
        self,
        *,
        expression: str,
        step_index: int,
        origin_action: str = "",
        readable_summary: str = "",
        variables: list[str] | None = None,
        assumptions: list[str] | None = None,
        source_data_refs: list[str] | None = None,
        observation_ids: list[str] | None = None,
        next_tests: list[str] | None = None,
        note: str = "",
    ) -> tuple[HypothesisRecord, bool]:
        normalized = self.normalize_expression(expression)
        existing = self.find_by_normalized(normalized)
        if existing is not None:
            existing.last_updated_step = max(existing.last_updated_step, step_index)
            if readable_summary and not existing.readable_summary:
                existing.readable_summary = readable_summary
            self._extend_unique(existing.variables, variables or [])
            self._extend_unique(existing.assumptions, assumptions or [])
            self._extend_unique(existing.source_data_refs, source_data_refs or [])
            self._extend_unique(existing.observation_ids, observation_ids or [])
            self._extend_unique(existing.next_tests, next_tests or [])
            if note:
                existing.notes.append(note)
            if origin_action:
                existing.notes.append(f"step {step_index} revisited via {origin_action}")
            return existing, False

        self._counter += 1
        hypothesis_id = f"H{self._counter:03d}"
        record = HypothesisRecord(
            hypothesis_id=hypothesis_id,
            expression=expression,
            normalized_expression=normalized,
            readable_summary=readable_summary or expression,
            status="proposed",
            origin_step=step_index,
            last_updated_step=step_index,
            variables=list(variables or []),
            assumptions=list(assumptions or []),
            source_data_refs=list(source_data_refs or []),
            observation_ids=list(observation_ids or []),
            next_tests=list(next_tests or []),
            notes=[note] if note else [],
        )
        if origin_action:
            record.notes.append(f"origin_action={origin_action}")
        self._records[hypothesis_id] = record
        return record, True

    def decide(
        self,
        *,
        decision: str,
        step_index: int,
        hypothesis_id: str | None = None,
        expression: str | None = None,
        evidence_type: str = "analysis",
        experiment_ids: list[str] | None = None,
        metric_name: str = "unspecified",
        metric_values: dict[str, float] | None = None,
        aggregate_score: float | None = None,
        summary: str = "",
        note: str = "",
        validation_ids: list[str] | None = None,
    ) -> HypothesisRecord:
        record = self._resolve_or_propose(
            hypothesis_id=hypothesis_id,
            expression=expression,
            step_index=step_index,
        )
        normalized_decision = self._normalize_status(decision)
        record.status = normalized_decision
        record.last_updated_step = max(record.last_updated_step, step_index)
        if summary or metric_values or aggregate_score is not None or experiment_ids:
            record.evidence.append(
                HypothesisEvidence(
                    step_index=step_index,
                    decision=normalized_decision,
                    evidence_type=evidence_type,
                    experiment_ids=list(experiment_ids or []),
                    metric_name=metric_name,
                    metric_values=dict(metric_values or {}),
                    aggregate_score=aggregate_score,
                    summary=summary,
                )
            )
        if note:
            record.notes.append(note)
        self._extend_unique(record.validation_ids, validation_ids or [])
        record.confidence_score = self._confidence_for_status(record.status)
        return record

    def record_evidence(
        self,
        *,
        expression: str,
        step_index: int,
        evidence_type: str,
        experiment_ids: list[str],
        metric_name: str,
        metric_values: dict[str, float] | None = None,
        aggregate_score: float | None = None,
        summary: str = "",
        supports: bool = True,
    ) -> HypothesisRecord:
        return self.decide(
            expression=expression,
            step_index=step_index,
            decision="accepted" if supports else "rejected",
            evidence_type=evidence_type,
            experiment_ids=experiment_ids,
            metric_name=metric_name,
            metric_values=metric_values,
            aggregate_score=aggregate_score,
            summary=summary,
        )

    def update_status(
        self,
        *,
        hypothesis_id: str | None = None,
        expression: str | None = None,
        status: str,
        step_index: int,
        note: str = "",
    ) -> HypothesisRecord:
        return self.decide(
            hypothesis_id=hypothesis_id,
            expression=expression,
            step_index=step_index,
            decision=status,
            note=note,
        )

    def add_next_tests(
        self,
        *,
        hypothesis_id: str | None = None,
        expression: str | None = None,
        next_tests: list[str],
        step_index: int,
    ) -> HypothesisRecord:
        record = self.resolve(hypothesis_id=hypothesis_id, expression=expression)
        self._extend_unique(record.next_tests, next_tests)
        record.last_updated_step = max(record.last_updated_step, step_index)
        return record

    def resolve(self, *, hypothesis_id: str | None = None, expression: str | None = None) -> HypothesisRecord:
        if hypothesis_id:
            try:
                return self._records[hypothesis_id]
            except KeyError as exc:
                raise ValueError(f"未知 hypothesis_id: {hypothesis_id}") from exc
        if expression:
            record = self.find_by_normalized(self.normalize_expression(expression))
            if record is not None:
                return record
        raise ValueError("需要 hypothesis_id 或已登记过的 expression。")

    def find_by_normalized(self, normalized_expression: str) -> HypothesisRecord | None:
        for record in self._records.values():
            if record.normalized_expression == normalized_expression:
                return record
        return None

    def active(self) -> list[HypothesisRecord]:
        return [record for record in self.ranked() if record.status == "proposed"]

    def ranked(self) -> list[HypothesisRecord]:
        return sorted(
            self._records.values(),
            key=lambda item: (
                -self._STATUS_ORDER.get(item.status, 0),
                item.hypothesis_id,
            ),
        )

    def all_records(self) -> list[HypothesisRecord]:
        return list(self._records.values())

    def summarize_for_prompt(self, limit: int = 6) -> str:
        records = self.ranked()[:limit]
        if not records:
            return "当前没有登记的候选规律。"

        lines = ["当前候选规律账本:"]
        for record in records:
            best_score = self._best_score(record)
            score_text = "N/A" if best_score is None else f"{best_score:.6g}"
            next_tests = "; ".join(record.next_tests[:3]) or "未记录"
            lines.append(
                f"- {record.hypothesis_id}: `{record.expression}` | status={record.status} | "
                f"evidence={len(record.evidence)} | validations={record.validation_ids[-3:]} | "
                f"best_score={score_text} | next_tests={next_tests}"
            )
        lines.append(
            "如果新想法与已有 hypothesis 等价或相近，请引用 hypothesis_id；"
            "只有两个决策层级：先 propose，证据足够后 accept 或 reject。"
        )
        return "\n".join(lines)

    def to_markdown(self) -> list[str]:
        lines = ["## Hypothesis Registry"]
        records = self.ranked()
        if not records:
            lines.extend(["- No hypotheses were registered.", ""])
            return lines

        for record in records:
            lines.extend(
                [
                    f"### {record.hypothesis_id}: `{record.expression}`",
                    f"- Status: `{record.status}`",
                    f"- Summary: {record.readable_summary}",
                    f"- Origin Step: `{record.origin_step}`",
                    f"- Last Updated Step: `{record.last_updated_step}`",
                    f"- Variables: `{record.variables}`",
                    f"- Assumptions: `{record.assumptions}`",
                    f"- Source Data Refs: `{record.source_data_refs}`",
                    f"- Observation IDs: `{record.observation_ids}`",
                    f"- Validation IDs: `{record.validation_ids}`",
                    f"- Next Tests: `{record.next_tests}`",
                ]
            )
            if record.evidence:
                lines.append("- Decision Evidence:")
                for evidence in record.evidence[-5:]:
                    lines.append(
                        f"  - step {evidence.step_index}: decision={evidence.decision}, "
                        f"{evidence.metric_name}, score={evidence.aggregate_score}, "
                        f"experiments={evidence.experiment_ids}; {evidence.summary}"
                    )
            if record.notes:
                lines.append(f"- Notes: {'; '.join(record.notes[-5:])}")
            lines.append("")
        return lines

    def normalize_expression(self, expression: str) -> str:
        text = str(expression).lower().strip()
        text = text.replace("^", "**")
        text = re.sub(r"\bsquare\(([^()]+)\)", r"(\1**2)", text)
        replacements = {
            r"\ba(?:_smooth|_sg|_est|_central_diff|_new)?\b": "a",
            r"\bv(?:_smooth|_sg|_est|_central_diff|_new)?\b": "v",
            r"\bf_ext\b|\bconstant_force\b|\bforce\b": "f",
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        text = re.sub(r"\s+", "", text)
        if text.count("=") == 1 and not any(operator in text for operator in ("==", "<=", ">=", "!=")):
            left, right = text.split("=", 1)
            sides = sorted([self._canonicalize_expression_side(left), self._canonicalize_expression_side(right)])
            return "=".join(sides)
        return self._canonicalize_expression_side(text)

    def _resolve_or_propose(
        self,
        *,
        hypothesis_id: str | None,
        expression: str | None,
        step_index: int,
    ) -> HypothesisRecord:
        if hypothesis_id or expression:
            try:
                return self.resolve(hypothesis_id=hypothesis_id, expression=expression)
            except ValueError:
                if not expression:
                    raise
        if expression:
            record, _ = self.propose(expression=expression, step_index=step_index)
            return record
        raise ValueError("需要 hypothesis_id 或 expression。")

    def _canonicalize_expression_side(self, expression: str) -> str:
        """Normalize harmless syntax differences without trying to prove algebraic identity."""
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return expression
        return ast.dump(tree.body, annotate_fields=False, include_attributes=False)

    def _normalize_status(self, status: str) -> str:
        normalized = str(status).strip().lower()
        if normalized not in self._STATUS_ALIASES:
            raise ValueError(f"不支持的 hypothesis decision: {status}")
        return self._STATUS_ALIASES[normalized]

    def _confidence_for_status(self, status: str) -> float:
        if status == "accepted":
            return 1.0
        if status == "rejected":
            return 0.0
        return 0.0

    def _best_score(self, record: HypothesisRecord) -> float | None:
        scores = [
            evidence.aggregate_score
            for evidence in record.evidence
            if evidence.aggregate_score is not None
        ]
        return min(scores) if scores else None

    def _extend_unique(self, target: list[str], values: list[str]) -> None:
        for value in values:
            text = str(value).strip()
            if text and text not in target:
                target.append(text)
