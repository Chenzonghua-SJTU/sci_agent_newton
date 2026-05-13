from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GeneratedProcessorResult:
    """Normalized result returned by an LLM-generated processor."""

    observation: str
    derived_series: list[dict[str, Any]]
    figures: list[str]
    metrics: dict[str, Any]
    raw_payload: dict[str, Any]


class GeneratedCodeRunner:
    """Executes generated data-processing code through a narrow JSON interface."""

    def __init__(
        self,
        generated_root: str | Path,
        timeout_seconds: int = 30,
    ) -> None:
        self.generated_root = Path(generated_root)
        self.generated_root.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

    def save_processor(self, code: str, step_index: int, action: str) -> Path:
        action_slug = self._slugify(action)
        code_path = self.generated_root / f"step_{step_index:03d}_{action_slug}.py"
        code_path.write_text(code.rstrip() + "\n", encoding="utf-8")
        return code_path

    def run_processor(
        self,
        *,
        code_path: str | Path,
        payload: dict[str, Any],
    ) -> GeneratedProcessorResult:
        code_path = Path(code_path)
        if not code_path.exists():
            raise FileNotFoundError(f"生成代码文件不存在: {code_path}")

        with tempfile.TemporaryDirectory(prefix="auto_scientist_data_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.json"
            output_path = tmp_path / "output.json"
            input_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            wrapper = self._build_wrapper_script(
                code_path=code_path,
                input_path=input_path,
                output_path=output_path,
                generated_root=self.generated_root,
            )
            completed = subprocess.run(
                [sys.executable, "-c", wrapper],
                cwd=str(self.generated_root),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                stdout = completed.stdout.strip()
                detail = stderr or stdout or f"returncode={completed.returncode}"
                raise RuntimeError(f"生成的数据处理代码执行失败: {detail}")

            if not output_path.exists():
                raise RuntimeError("生成的数据处理代码没有写出 output.json。")

            raw_payload = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(raw_payload, dict):
                raise RuntimeError("生成的数据处理代码必须返回 JSON object。")

        return self._normalize_result(raw_payload)

    def _build_wrapper_script(
        self,
        *,
        code_path: Path,
        input_path: Path,
        output_path: Path,
        generated_root: Path,
    ) -> str:
        return f"""
import importlib.util
import json
import pathlib
import sys

generated_root = pathlib.Path({str(generated_root)!r})
if str(generated_root) not in sys.path:
    sys.path.insert(0, str(generated_root))

code_path = pathlib.Path({str(code_path)!r})
input_path = pathlib.Path({str(input_path)!r})
output_path = pathlib.Path({str(output_path)!r})

spec = importlib.util.spec_from_file_location("generated_processor", code_path)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)

if not hasattr(module, "process"):
    raise RuntimeError("generated processor must define process(payload)")

payload = json.loads(input_path.read_text(encoding="utf-8"))
result = module.process(payload)
if not isinstance(result, dict):
    raise RuntimeError("process(payload) must return a dict")

output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
"""

    def _normalize_result(self, raw_payload: dict[str, Any]) -> GeneratedProcessorResult:
        observation = str(raw_payload.get("observation", "")).strip()
        if not observation:
            observation = "数据处理 LLM 生成的代码执行完成，但没有提供 observation。"

        raw_series = raw_payload.get("derived_series", [])
        derived_series = raw_series if isinstance(raw_series, list) else []
        derived_series = [item for item in derived_series if isinstance(item, dict)]

        raw_figures = raw_payload.get("figures", [])
        figures = raw_figures if isinstance(raw_figures, list) else []
        figures = [str(item) for item in figures]

        raw_metrics = raw_payload.get("metrics", {})
        metrics = raw_metrics if isinstance(raw_metrics, dict) else {}

        return GeneratedProcessorResult(
            observation=observation,
            derived_series=derived_series,
            figures=figures,
            metrics=metrics,
            raw_payload=raw_payload,
        )

    def _slugify(self, value: str) -> str:
        normalized = "".join(char if char.isalnum() else "_" for char in value.lower())
        normalized = "_".join(part for part in normalized.split("_") if part)
        return normalized or "processor"
