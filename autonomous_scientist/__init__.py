"""自主科研智能体的基础包。

当前版本包含：
1. VirtualUniverse: 黑盒虚拟物理宇宙环境。
2. DataProcessingTool: 通用轨迹平滑与差分工具。
3. VerificationEngine: 基于 PySR 的符号验证引擎。
4. HypothesisBrain: 基于 LLM 的实验规划大脑。
5. DataProcessingBrain: 基于 LLM 的数据处理代码生成器。
6. GeneratedCodeRunner / CodeRegistry: 生成代码执行与登记。
7. ScientificReporter: 科学报告生成模块。
8. ScientistAgent: 多轮 ReAct 式自主科研 Agent。
"""

from .agent import (
    ActionDecision,
    ActionRecord,
    CandidateLaw,
    DerivedSeries,
    ExperimentRecord,
    GeneralizationCheck,
    HypothesisBrain,
    LawHypothesis,
    ScientificNotebook,
    ScientificCycleResult,
    ScientistAgent,
)
from .code_registry import CodeRegistry, GeneratedProcessorRecord
from .code_runner import GeneratedCodeRunner, GeneratedProcessorResult
from .data_brain import DataProcessingBrain
from .processing import DataProcessingTool, PhaseSpaceData, SeriesSummary
from .reporting import ScientificReporter
from .universe import (
    ExperimentConfig,
    ExperimentResult,
    ForceFieldType,
    VirtualUniverse,
)
from .verification import InvariantSearchResult, VerificationEngine, VerificationResult

__all__ = [
    "ActionDecision",
    "ActionRecord",
    "CandidateLaw",
    "CodeRegistry",
    "DataProcessingBrain",
    "DataProcessingTool",
    "DerivedSeries",
    "ExperimentConfig",
    "ExperimentRecord",
    "ExperimentResult",
    "ForceFieldType",
    "GeneratedCodeRunner",
    "GeneratedProcessorRecord",
    "GeneratedProcessorResult",
    "GeneralizationCheck",
    "HypothesisBrain",
    "InvariantSearchResult",
    "LawHypothesis",
    "PhaseSpaceData",
    "ScientificCycleResult",
    "ScientificReporter",
    "ScientificNotebook",
    "SeriesSummary",
    "ScientistAgent",
    "VirtualUniverse",
    "VerificationEngine",
    "VerificationResult",
]
