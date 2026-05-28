from __future__ import annotations

import ast
import json
import os
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


@dataclass(frozen=True, slots=True)
class GeneratedCodePolicy:
    """Static and runtime guardrails for LLM-generated processor code."""

    allowed_import_roots: frozenset[str] = frozenset(
        {
            "collections",
            "dataclasses",
            "__future__",
            "functools",
            "itertools",
            "json",
            "math",
            "matplotlib",
            "numpy",
            "pandas",
            "pathlib",
            "re",
            "scipy",
            "sklearn",
            "statistics",
            "typing",
            "warnings",
        }
    )
    blocked_import_roots: frozenset[str] = frozenset(
        {
            "autonomous_scientist",
            "builtins",
            "ctypes",
            "dotenv",
            "ftplib",
            "glob",
            "http",
            "importlib",
            "marshal",
            "os",
            "pickle",
            "requests",
            "runpy",
            "shutil",
            "socket",
            "subprocess",
            "sys",
            "urllib",
        }
    )
    blocked_calls: frozenset[str] = frozenset(
        {
            "__import__",
            "breakpoint",
            "compile",
            "delattr",
            "eval",
            "exec",
            "getattr",
            "globals",
            "input",
            "locals",
            "open",
            "setattr",
            "vars",
        }
    )
    blocked_attribute_calls: frozenset[str] = frozenset(
        {
            "chmod",
            "glob",
            "iterdir",
            "open",
            "read_bytes",
            "read_text",
            "rename",
            "replace",
            "rglob",
            "rmdir",
            "unlink",
            "write_bytes",
            "write_text",
        }
    )
    sensitive_env_markers: frozenset[str] = frozenset(
        {"API_KEY", "AUTH", "CREDENTIAL", "KEY", "PASSWORD", "SECRET", "TOKEN"}
    )


class GeneratedCodePolicyViolation(RuntimeError):
    """Raised when generated code violates the local execution policy."""


class GeneratedCodeRunner:
    """Executes generated data-processing code through a narrow JSON interface."""

    def __init__(
        self,
        generated_root: str | Path,
        timeout_seconds: int = 30,
        policy: GeneratedCodePolicy | None = None,
    ) -> None:
        self.generated_root = Path(generated_root).resolve()
        self.generated_root.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.policy = policy or GeneratedCodePolicy()

    def save_processor(self, code: str, step_index: int, action: str) -> Path:
        self.validate_processor_code(code)
        action_slug = self._slugify(action)
        code_path = self.generated_root / f"step_{step_index:03d}_{action_slug}.py"
        code_path.write_text(code.rstrip() + "\n", encoding="utf-8")
        return code_path

    def validate_processor_code(self, code: str) -> None:
        """Reject generated processors that try to escape the narrow data API."""
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise GeneratedCodePolicyViolation(f"生成代码存在语法错误: {exc}") from exc

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._validate_import_name(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise GeneratedCodePolicyViolation("生成代码禁止使用相对 import。")
                self._validate_import_name(node.module or "")
            elif isinstance(node, ast.Call):
                self._validate_call(node)
            elif isinstance(node, ast.Name) and node.id.startswith("__"):
                raise GeneratedCodePolicyViolation(f"生成代码禁止访问特殊名称 `{node.id}`。")
            elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise GeneratedCodePolicyViolation(f"生成代码禁止访问特殊属性 `.{node.attr}`。")

    def run_processor(
        self,
        *,
        code_path: str | Path,
        payload: dict[str, Any],
    ) -> GeneratedProcessorResult:
        code_path = Path(code_path)
        if not code_path.exists():
            raise FileNotFoundError(f"生成代码文件不存在: {code_path}")
        self.validate_processor_code(code_path.read_text(encoding="utf-8"))

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
                env=self._build_sanitized_env(),
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

        return self._normalize_result(
            raw_payload,
            allowed_output_dir=self.generated_root / "artifacts",
        )

    def _build_wrapper_script(
        self,
        *,
        code_path: Path,
        input_path: Path,
        output_path: Path,
        generated_root: Path,
    ) -> str:
        return f"""
import builtins
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
allowed_output_dir = pathlib.Path(payload.get("output_dir", generated_root / "artifacts")).resolve()
allowed_write_roots = (allowed_output_dir, output_path.parent.resolve())
original_open = builtins.open

def _is_relative_to(path, root):
    try:
        return path == root or path.is_relative_to(root)
    except AttributeError:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return path == root

def _guarded_open(file, mode="r", *args, **kwargs):
    if isinstance(file, (str, bytes, pathlib.Path)) and any(flag in mode for flag in ("w", "a", "x", "+")):
        resolved = pathlib.Path(file).resolve()
        if not any(_is_relative_to(resolved, root) for root in allowed_write_roots):
            raise PermissionError(f"generated processor may only write under {{allowed_output_dir}}")
    return original_open(file, mode, *args, **kwargs)

builtins.open = _guarded_open
result = module.process(payload)
if not isinstance(result, dict):
    raise RuntimeError("process(payload) must return a dict")

def _json_default(obj):
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    raise TypeError(f"Object of type {{obj.__class__.__name__}} is not JSON serializable")

output_path.write_text(json.dumps(result, ensure_ascii=False, default=_json_default), encoding="utf-8")
"""

    def _normalize_result(
        self,
        raw_payload: dict[str, Any],
        *,
        allowed_output_dir: Path,
    ) -> GeneratedProcessorResult:
        observation = str(raw_payload.get("observation", "")).strip()
        if not observation:
            observation = "数据处理 LLM 生成的代码执行完成，但没有提供 observation。"

        raw_series = raw_payload.get("derived_series", [])
        derived_series = raw_series if isinstance(raw_series, list) else []
        derived_series = [item for item in derived_series if isinstance(item, dict)]

        raw_figures = raw_payload.get("figures", [])
        figures = raw_figures if isinstance(raw_figures, list) else []
        figures = [
            str(self._validate_output_path(item, allowed_output_dir=allowed_output_dir))
            for item in figures
        ]

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

    def _validate_import_name(self, module_name: str) -> None:
        root_name = module_name.split(".", 1)[0]
        if not root_name:
            raise GeneratedCodePolicyViolation("生成代码包含空 import。")
        if root_name in self.policy.blocked_import_roots:
            raise GeneratedCodePolicyViolation(f"生成代码禁止 import `{root_name}`。")
        if root_name not in self.policy.allowed_import_roots:
            raise GeneratedCodePolicyViolation(f"生成代码 import `{root_name}` 不在白名单中。")

    def _validate_call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.policy.blocked_calls:
            raise GeneratedCodePolicyViolation(f"生成代码禁止调用 `{node.func.id}`。")
        if isinstance(node.func, ast.Attribute) and node.func.attr in self.policy.blocked_attribute_calls:
            raise GeneratedCodePolicyViolation(f"生成代码禁止调用 `.{node.func.attr}()`。")

    def _validate_output_path(self, value: Any, *, allowed_output_dir: Path) -> Path:
        raw_path = Path(str(value))
        path = raw_path if raw_path.is_absolute() else allowed_output_dir / raw_path
        output_root = allowed_output_dir.resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(output_root):
            raise RuntimeError(f"生成代码返回的图像路径超出允许目录: {value}")
        return resolved

    def _build_sanitized_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for key, value in os.environ.items():
            upper_key = key.upper()
            if any(marker in upper_key for marker in self.policy.sensitive_env_markers):
                continue
            env[key] = value
        env["PYTHONNOUSERSITE"] = "1"
        return env
