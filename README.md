# Auto Scientist Sandbox

一个面向 AI4S 的自主科学发现原型。Agent 通过黑盒实验 API 获取轨迹数据，维护实验数据记录表和假说表，并在证据足够时接受一条可复验的规律。底层虚拟世界的实现不暴露给决策 LLM 或数据处理 LLM。

## 核心设计

- `VirtualUniverse`：黑盒实验环境。Agent 只能提交实验参数并获得 `t,q` 轨迹和控制参数摘要。
- `ScientificNotebook`：实验数据记录表，保存每个实验的原始 `t,q`、数据处理 LLM 新定义的派生序列、观察记录 `OBS` 和验证记录 `VAL`。
- `HypothesisRegistry`：假说表。每条假说只有三种状态：`proposed`、`rejected`、`accepted`。
- `HypothesisBrain`：决策 LLM，负责决定下一步做什么实验、让数据处理 LLM 维护哪些数据、如何验证哪条假说，以及何时接受或拒绝假说。
- `DataProcessingBrain`：数据处理 LLM，负责把决策 LLM 指定的数据任务写成 Python 代码并运行。它不决定科学策略，不自行选择要验证的假说，也不替决策 LLM 宣布定律。
- `GeneratedCodeRunner`：执行数据处理 LLM 生成的代码，并限制危险 import、文件访问、网络和环境变量泄露。
- `ScientificReporter`：把实验记录表、假说表和最终结论导出为 Markdown 报告。

## Agent 动作

Agent 的可执行动作只有四个：

- `run_experiments`：单个或批量运行黑盒实验。
- `analyze_data`：交给数据处理 LLM 写代码完成数据任务。
  - `maintain_ledger`：维护实验数据记录表，例如定义派生物理量、计算中间序列、生成观察记录。
  - `validate_hypothesis`：按决策 LLM 给定的验证口径，在一个或多个实验上验证某条假说，生成 `VAL` 记录。
- `manage_hypotheses`：提出假说、拒绝假说、接受假说。接受或拒绝必须引用相应的验证记录。
- `finish`：当至少存在一条 `accepted` 假说后结束科研循环。

## 项目结构

```text
.
├── autonomous_scientist/
│   ├── __init__.py
│   ├── agent.py
│   ├── code_registry.py
│   ├── code_runner.py
│   ├── data_brain.py
│   ├── hypothesis_registry.py
│   ├── reporting.py
│   ├── tool_specs.py
│   └── universe.py
├── run_agent.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 安装

建议使用 Python 3.11。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

也可以用可编辑安装：

```bash
pip install -e .
```

## 配置

推荐在项目根目录创建 `.env`：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-v4-flash
```

也支持 OpenAI 兼容配置：

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

数据处理 LLM 默认复用决策 LLM；也可以单独配置：

```env
DATA_PROCESSING_MODEL=deepseek-v4-flash
DATA_PROCESSING_TEMPERATURE=0.0
```

默认启用数据处理 LLM 生成代码：

```env
USE_GENERATED_PROCESSORS=true
ALLOW_DATA_PROCESSING_FALLBACK=false
```

## 运行

```bash
python run_agent.py
```

运行完成后会生成：

```text
discovery_report.md
generated_processors/
```

`discovery_report.md` 会包含实验数据记录表摘要、观察记录、验证记录、假说表状态和最终 accepted 规律。

## 边界约束

- 决策 LLM 可以看到原始数据摘要和派生数据摘要，但不能读取底层世界源码。
- 数据处理 LLM 只能执行 `analysis_goal` 指定的数据任务。
- 数据处理 LLM 如果认为任务不够明确，应返回结构化观察说明任务欠指定，而不是自行发明科学目标。
- `accepted` 是唯一结束条件；一旦假说被接受，科研循环可以结束，不再强制长步数继续探索。
