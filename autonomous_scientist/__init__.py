"""自主科研智能体的基础包。

当前版本包含：
1. VirtualUniverse: 黑盒虚拟物理宇宙环境。
2. HypothesisBrain: 基于 LLM 的实验规划大脑。
3. DataProcessingBrain: 基于 LLM 的数据处理代码生成器。
4. GeneratedCodeRunner / CodeRegistry: 生成代码执行与登记。
5. ToolSpec / GeneratedCodePolicy: action 元数据与生成代码权限约束。
6. HypothesisRegistry: 假说账本与证据管理。
7. ScientificReporter: 科学报告生成模块。
8. ScientistAgent: 多轮 ReAct 式自主科研 Agent。
"""

from .agent import (
    ActionDecision,
    ActionRecord,
    DerivedSeries,
    ExperimentRecord,
    HypothesisBrain,
    LedgerObservation,
    LedgerValidation,
    LawHypothesis,
    ScientificNotebook,
    ScientificCycleResult,
    ScientistAgent,
)
from .code_registry import CodeRegistry, GeneratedProcessorRecord
from .code_runner import (
    GeneratedCodePolicy,
    GeneratedCodePolicyViolation,
    GeneratedCodeRunner,
    GeneratedProcessorResult,
)
from .data_brain import DataProcessingBrain
from .hypothesis_registry import HypothesisEvidence, HypothesisRecord, HypothesisRegistry
from .reporting import ScientificReporter
from .tool_specs import DEFAULT_TOOL_SPECS, TOOL_SPECS_BY_NAME, ToolSpec
from .universe import (
    ExperimentConfig,
    ExperimentResult,
    ForceFieldType,
    VirtualUniverse,
)

__all__ = [
    "ActionDecision",
    "ActionRecord",
    "CodeRegistry",
    "DataProcessingBrain",
    "DerivedSeries",
    "ExperimentConfig",
    "ExperimentRecord",
    "ExperimentResult",
    "ForceFieldType",
    "GeneratedCodePolicy",
    "GeneratedCodePolicyViolation",
    "GeneratedCodeRunner",
    "GeneratedProcessorRecord",
    "GeneratedProcessorResult",
    "HypothesisBrain",
    "HypothesisEvidence",
    "HypothesisRecord",
    "HypothesisRegistry",
    "LedgerObservation",
    "LedgerValidation",
    "LawHypothesis",
    "ScientificCycleResult",
    "ScientificReporter",
    "ScientificNotebook",
    "ScientistAgent",
    "ToolSpec",
    "DEFAULT_TOOL_SPECS",
    "TOOL_SPECS_BY_NAME",
    "VirtualUniverse",
]
