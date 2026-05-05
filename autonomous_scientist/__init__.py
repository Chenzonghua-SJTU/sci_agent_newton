"""自主科研智能体的基础包。

当前版本包含：
1. VirtualUniverse: 黑盒虚拟物理宇宙环境。
2. DataProcessingTool: 通用轨迹平滑与差分工具。
3. VerificationEngine: 基于 PySR 的符号验证引擎。
4. HypothesisBrain: 基于 LLM 的实验规划大脑。
5. ScientificReporter: 科学报告生成模块。
6. ScientistAgent: 多轮 ReAct 式自主科研 Agent。
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
    "DataProcessingTool",
    "DerivedSeries",
    "ExperimentConfig",
    "ExperimentRecord",
    "ExperimentResult",
    "ForceFieldType",
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
