from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GeneratedProcessorRecord:
    """Metadata for one LLM-generated data processor."""

    step_index: int
    action: str
    code_path: str
    status: str
    created_at: str
    description: str
    observation: str = ""
    error: str = ""
    metrics: dict[str, Any] | None = None
    derived_series: list[str] | None = None
    figures: list[str] | None = None


class CodeRegistry:
    """Small JSON registry for generated data-processing scripts."""

    def __init__(self, registry_path: str | Path) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict[str, Any]]:
        if not self.registry_path.exists():
            return []
        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def recent_records(self, limit: int = 8) -> list[dict[str, Any]]:
        return self.load()[-limit:]

    def add_record(self, record: GeneratedProcessorRecord) -> None:
        records = self.load()
        records.append(asdict(record))
        self.registry_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def record_success(
        self,
        *,
        step_index: int,
        action: str,
        code_path: Path,
        description: str,
        observation: str,
        metrics: dict[str, Any] | None,
        derived_series: list[str],
        figures: list[str],
    ) -> None:
        self.add_record(
            GeneratedProcessorRecord(
                step_index=step_index,
                action=action,
                code_path=str(code_path),
                status="success",
                created_at=datetime.now().isoformat(timespec="seconds"),
                description=description,
                observation=observation[:1200],
                metrics=metrics,
                derived_series=derived_series,
                figures=figures,
            )
        )

    def record_failure(
        self,
        *,
        step_index: int,
        action: str,
        code_path: Path,
        description: str,
        error: str,
    ) -> None:
        self.add_record(
            GeneratedProcessorRecord(
                step_index=step_index,
                action=action,
                code_path=str(code_path),
                status="failure",
                created_at=datetime.now().isoformat(timespec="seconds"),
                description=description,
                error=error[:1200],
            )
        )
