# Auto Scientist Sandbox

一个面向 AI4S 的 Neuro-Symbolic Agent 原型工程。它的目标不是去拟合一条你已经知道答案的曲线，而是像一个真正的科研工作者那样，通过与黑盒“虚拟物理宇宙”交互，自主设计实验、观察轨迹、提出动力学假设、验证候选规律，并最终输出结构化科研报告。

为了检验这个 Agent 是否真的具备“发现定律”的能力，项目内置了一个故意反常识的沙盒宇宙：它不保证服从教科书里的经典动力学，而底层方程不会暴露给 Agent。也正因为如此，这个仓库非常适合作为 AI for Science、自动科学发现、神经符号推理和假设验证工作流的实验平台。

## 核心特性

- `VirtualUniverse`：黑盒物理环境，通过 API 返回轨迹数据，而不暴露底层方程。
- `DataProcessingTool`：对 `q(t)` 进行平滑、求导、构造相空间特征 `q/v/a`。
- `VerificationEngine`：保留 PySR / 不变量搜索封装，作为可选验证工具；默认自主研究流程不再依赖它提出公式。
- `HypothesisBrain`：调用兼容 OpenAI SDK 的大模型 API，对轨迹统计摘要进行“物理学家式”判断与反思，并在规划步骤中主动提出可验证的候选公式。
- `DataProcessingBrain`：第二个专门负责数据处理的大模型 API。决策 LLM 选定数据处理动作后，它会生成 Python 代码、执行分析、保存脚本，并把结构化 observation 返回给决策 LLM。
- `ScientificReporter`：将一次科研循环导出为 Markdown 报告。
- `ScientistAgent`：串联实验、分析、回归、反思和报告输出。

## 项目结构

```text
.
├── autonomous_scientist/
│   ├── __init__.py
│   ├── agent.py
│   ├── code_registry.py
│   ├── code_runner.py
│   ├── data_brain.py
│   ├── processing.py
│   ├── reporting.py
│   ├── universe.py
│   └── verification.py
├── run_agent.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 环境安装

建议优先使用 Python `3.11`。不建议直接使用 Python `3.13`，因为部分科学计算依赖和 `PySR` 组合在新版本解释器上可能存在兼容性问题。

### 1. 创建并激活虚拟环境

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. 安装 Python 依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

如果你偏好现代项目安装方式，也可以直接使用：

```bash
pip install -e .
```

如果你已经装过依赖，但运行时遇到类似下面的错误：

```text
'PySRRegressor' object has no attribute '_validate_data'
```

这通常表示 `PySR` 与 `scikit-learn` 版本不兼容。请优先确认：

- 你的 Python 版本建议为 `3.11`
- `scikit-learn` 建议固定在 `<1.7`

例如可执行：

```bash
pip install "scikit-learn<1.7"
```

### 3. 重要提示：PySR 依赖 Julia

这是整个项目最容易卡住的一步，请务必认真看。

`PySR` 的后端依赖 Julia，因此即使 `pip install pysr` 成功，也不代表它已经可以直接运行。第一次使用前，你通常还需要额外执行一次安装/编译初始化。

在 Python 交互环境中运行：

```python
import pysr
pysr.install()
```

说明：

- 这一步会自动拉起 Julia 侧依赖安装，首次执行可能较慢。
- 某些环境下你可能还需要手动安装 Julia，并确保其在系统 `PATH` 中可见。
- 如果 PySR 无法正常启动，优先检查 Julia 版本、网络环境和编译日志。

## 配置与运行

### 1. 配置 API Key

默认推荐使用 `DeepSeek API`，因为本项目已经兼容 OpenAI SDK 调用方式，而且对国内用户更友好。

推荐在项目根目录新建一个 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-v4-flash
```

如果你不想用 `.env`，也可以直接在终端中设置环境变量：

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"
```

如果你之后想切回 OpenAI，也可以改用：

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

### 2. 配置数据处理 LLM（可选）

项目现在支持双 LLM 架构：

- 决策 LLM：由 `DEEPSEEK_MODEL` 或 `OPENAI_MODEL` 指定，负责做实验规划、提出 action 和公式。
- 数据处理 LLM：由 `DATA_PROCESSING_MODEL` 指定，负责为数据处理 action 生成 Python 代码并执行。

如果不额外设置，数据处理 LLM 默认复用决策 LLM 的模型与 base URL。你也可以单独配置：

```env
DATA_PROCESSING_MODEL=deepseek-v4-flash
DATA_PROCESSING_TEMPERATURE=0.0
```

如果需要临时关闭生成代码路径，回到旧版固定函数工具：

```env
USE_GENERATED_PROCESSORS=false
```

生成的代码与注册表会保存到：

```text
generated_processors/
├── step_005_inspect_relationships.py
├── step_006_fit_relationship_model.py
├── artifacts/
└── registry.json
```

### 3. 运行 Agent

```bash
python run_agent.py
```

当前默认任务是探索黑盒运动规律，并尝试仅从轨迹数据中恢复可跨实验复验的动力学结构。

Agent 的公式发现流程现在支持两条路径：

- 默认路径：决策 LLM 优先调用 `custom_data_analysis`，用自然语言说明它希望分析的问题，例如“比较 exp_02/03/04 中加速度与速度平方的关系，并返回必要派生量和残差图”。数据处理 LLM 会自行生成 Python 代码、执行分析、返回 observation、metrics、图像和可复用派生序列。随后决策 LLM 在自己的规划思考中提出候选表达式，用 `test_candidate_expression` 和 `cross_experiment_check` 进行验证，再用 `register_candidate_law` 登记已通过跨实验验证的候选规律。
- 兼容路径：`estimate_kinematics`、`inspect_series`、`inspect_relationships`、`define_derived_quantity`、`fit_relationship_model` 等细粒度数据处理动作仍然保留，也会优先走数据处理 LLM 生成代码；若生成代码失败，会自动回退到旧版固定函数工具。
- 旧路径：`search_invariants` 仍保留在代码中作为手动调试/对照工具，但已经从 Agent 的可选动作列表中移除；自主运行时不会再靠不变量枚举或 PySR 替它提出公式。

候选公式提出后，Agent 都需要继续通过 `cross_experiment_check` 在不同实验条件下复验，并通过 `register_candidate_law` 登记候选规律，才允许进入最终规律总结。

## 预期输出

运行时你会在终端看到类似这样的日志：

```text
========================================================================
🚀 启动自主科研 Agent...
========================================================================
🔑 正在读取环境变量中的 API Key...
🧩 正在初始化系统，当前 LLM 提供商: DeepSeek，模型: deepseek-v4-flash
========================================================================
🧪 实验任务：探索恒定外力场景，设定 F_ext = 10.0
🧠 Agent 正在观察数据并生成统计摘要...
🔎 Agent 正在分析变量关系、提出候选公式并验证...
📘 完成候选公式验证后将自动生成 Markdown 科研报告...
========================================================================
✅ 科研循环执行完成。
========================================================================
【候选规律排序】
combo_expression | score=0.001032 | origin=register_candidate_law | exp=exp_03
========================================================================
📄 报告已保存至: /your/project/path/discovery_report.md
========================================================================
```

运行完成后，项目根目录会生成：

```text
discovery_report.md
```

这份报告通常包含以下部分：

- Experimental Setup
- Data Observations
- Hypothesis Plan
- Symbolic Regression Output
- Scientist Reflection
- Conclusion

## `.env` 支持说明

如果你希望 `run_agent.py` 自动加载 `.env`，可以在文件开头加入下面两行：

```python
from dotenv import load_dotenv
load_dotenv()
```

推荐放置位置：

```python
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
```

这样脚本启动时就会自动读取当前目录下的 `.env` 文件，无需每次手动 `export OPENAI_API_KEY`。

## 未来扩展方向

- 支持更多势场类型，例如位置相关势场、时变外力场和多体耦合系统。
- 将 `HypothesisBrain` 升级为多轮 ReAct 策略，而不是单轮实验。
- 接入更严格的量纲分析与单位系统校验。
- 支持自动比较多个候选公式在跨实验条件下的泛化误差。
- 扩展到 PDE 发现、控制方程发现和实验设计优化。

## License

建议使用 MIT License，以便学术交流和二次开发。
