from __future__ import annotations

import json
import re
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


class DataProcessingBrain:
    """LLM that writes Python code for data-processing actions."""

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout_seconds: float = 90.0,
    ) -> None:
        if OpenAI is None:
            raise ImportError("未检测到 openai 官方库。")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds, max_retries=2)
        self.model = model
        self.temperature = temperature

    def write_processor_code(
        self,
        *,
        action: str,
        parameters: dict[str, Any],
        notebook_context: str,
        recent_processors: list[dict[str, Any]],
    ) -> str:
        """Ask the data-processing LLM to generate a processor module."""
        system_prompt = (
            "你是一个严谨的数据处理 LLM，只负责为科研 agent 写 Python 数据分析代码。"
            "你不是决策者，不提出新的科研结论或最终物理公式，只实现决策 LLM 已经指定的数据处理 action。"
            "你必须只返回 Python 源码，不要 Markdown，不要解释。"
        )
        user_prompt = f"""
请为下面的数据处理 action 生成一个可执行 Python 文件。

Action:
{action}

Parameters JSON:
{json.dumps(parameters, ensure_ascii=False, indent=2)}

当前公开的数据处理动作只有 analyze_data。

如果 Action 是 analyze_data:
- parameters["analysis_goal"] 是决策 LLM 提出的自然语言任务说明，是本次代码的边界；你只能实现这个任务，不自行扩大科学目标。
- 公式提出权只属于决策 LLM；你只是写代码的工具。即使数据里出现很强的规律，也只能返回可核验数值事实、派生序列、OBS 或 VAL，不能把它宣布为最终定律。
- parameters["analysis_mode"] 只应理解为两种语义模式:
  - "maintain_ledger": 维护实验数据记录表，定义派生量/中间物理量，写入 observations。
  - "validate_hypothesis": 验证某个假说，写入 validations。
- optional_series 是决策 LLM 指定的关注序列；除非 analysis_goal 明确要求补充必要基础序列，否则不要擅自扩大分析范围。
- expected_outputs 是决策 LLM 希望看到的输出类型；如果没有提供，只返回完成 analysis_goal 所必需的 observation、metrics、figures、derived_series、observations 或 validations。
- 你可以选择数值实现细节（例如差分、平滑窗口、最小二乘求解方式），但不能替决策 LLM 决定科学策略、要验证哪个假说、要比较哪些模型族、要定义哪些与任务无关的新物理量。
- 噪声固定为 0；从 t,q 估计导数时优先用 np.gradient(q, t, edge_order=2) 和 np.gradient(v, t, edge_order=2)。如果使用 scipy.signal.savgol_filter 的 deriv 参数，delta 必须是时间步长 dt，不能写成 dt**order 或 dt**2。
- 如果 analysis_goal 对“维护什么表项”或“如何验证假说”描述不足以写代码，请返回清楚的 observation 和 metrics={{"task_under_specified": true}}，不要自行发明一套科研分析。

maintain_ledger 模式的额外要求:
1. 只维护决策 LLM 在 analysis_goal/parameters 中要求维护的实验数据记录表条目。
2. 可以定义派生量/中间量、生成图像或写 observations，但它们必须服务于 analysis_goal。
3. observations 列表中的每条都必须是可核验的数据事实，包含 summary、source_data_refs 和 metrics/numeric_facts。
4. 不要替决策 LLM 提出最终假说；如果 analysis_goal 要求提取可观察线索，可以在 observation 文本中列出数据事实，但不要扩展成模型海选。
5. 除非 analysis_goal 明确点名，不要自行定义动量、能量、阻力、质量、模型参数或其他物理解释性量；默认只补充任务需要的基础数值序列。
6. metrics 中尽量包含 observation_count 以及本任务明确要求的指标。
7. 如果 analysis_goal 要求诊断变量关系，只按决策 LLM 指定的任务报告可核验数值事实；不要自行引入或暗示任何物理规律形式。

validate_hypothesis 模式的额外要求:
1. 必须使用 parameters["hypothesis_id"] 或 parameters["candidate_expression"] 标明要验证的假说。
2. 必须按照决策 LLM 在 analysis_goal 中指定的验证口径写代码，例如残差定义、误差指标、边界剔除策略、单实验/跨实验范围。
3. 必须输出 validations 列表；每条包含 hypothesis_id、experiment_ids、supports、metric_name、metric_values、aggregate_score、summary、source_data_refs。
4. 若输入是多个实验，要按 analysis_goal 报告每个实验的误差/残差，而不是只报告整体平均。
5. 如果需要派生量或残差序列，也可以同时返回 derived_series；这些序列必须是验证任务所需。

当前 notebook 数据上下文摘要:
{notebook_context}

最近已经生成并登记过的数据处理脚本:
{json.dumps(recent_processors, ensure_ascii=False, indent=2)}

运行时会调用你生成文件中的:

def process(payload: dict) -> dict:
    ...

payload 结构:
- payload["action"]: 当前 action 名称。
- payload["parameters"]: 决策 LLM 给出的参数。
- payload["experiments"]: dict，key 是 experiment_id。
- 每个 experiment 包含:
  - "config": 实验控制参数，包括 F_ext, force_field_type, constant_force, raw_constant_force, initial_q, initial_v, t_span, dt, noise_std。noise_std 在当前环境中固定为 0。分析外力时必须使用 F_ext；force_field_type=free 时 F_ext=0，raw_constant_force 仅用于审计旧参数，不可作为实际外力。
  - "series": dict，包含 "t", "q" 以及此前已经登记的派生序列；值都是 list[float]。
  - "available_series": 可用序列名列表。
- payload["observations"]: 已写入实验数据记录表的 OBS 条目。
- payload["validations"]: 已写入实验数据记录表的 VAL 条目。
- payload["hypotheses"]: 当前假说表，包含 hypothesis_id、expression 和 status。
- payload["output_dir"]: 允许写图像等产物的目录。

返回 dict 结构:
{{
  "observation": "给决策 LLM 看的简洁中文观察，包含关键数值指标",
  "derived_series": [
    {{
      "experiment_id": "exp_02",
      "name": "new_series_name",
      "values": [0.1, 0.2],
      "source_name": "简述来源表达式或算法",
      "provenance": "generated data processor: ...",
      "description": "可选说明"
    }}
  ],
  "observations": [
    {{
      "summary": "可核验的数据事实",
      "source_data_refs": ["exp_01:q", "exp_02:derived_name"],
      "metrics": {{"example_metric": 0.0}}
    }}
  ],
  "validations": [
    {{
      "hypothesis_id": "H001",
      "experiment_ids": ["exp_01", "exp_02"],
      "supports": true,
      "metric_name": "residual_rmse",
      "metric_values": {{"exp_01": 0.0, "exp_02": 0.0}},
      "aggregate_score": 0.0,
      "summary": "验证摘要",
      "source_data_refs": ["exp_01:q", "exp_02:q"]
    }}
  ],
  "figures": ["可选图像绝对路径或相对路径"],
  "metrics": {{"rmse": 0.01}}
}}

硬性要求:
1. 只能使用白名单 import：json、math、statistics、itertools、functools、collections、pathlib、typing、numpy、pandas、scipy、sklearn、matplotlib。
2. 不要 import autonomous_scientist、os、sys、subprocess、socket、requests、urllib、dotenv、importlib；不要读取 universe.py，不要读取 .env，不要访问网络。
2a. 判断外力时只能使用 config["F_ext"]；不要用 config["constant_force"] 或 metadata["constant_force"] 推断实际外力。
3. 不要修改 payload 输入数据；新序列必须通过 returned derived_series 返回。
4. 需要图像时，把图像保存到 payload["output_dir"] 下，并把路径写入 figures；output_dir 已经由宿主创建，不需要自行创建目录。
5. derived_series 中 values 的长度必须与对应实验的 t 序列长度一致。
6. 如果参数里包含 experiment_ids，则只处理这些实验；如果只有 experiment_id，则只处理它；如果都没有，则默认处理所有实验。
7. 可以只返回 observation 和 figures；如果构造了后续有复用价值的时间序列，才返回 derived_series。
8. 做拟合时，应在 observation 和 metrics 中报告系数、R2、RMSE、MAE 或更适合当前问题的误差指标。
9. 做运动学估计时，可用 scipy.signal.savgol_filter 从 q(t) 估计 q_smooth/v/a，但要说明窗口、阶数和边界误差风险。
10. 需要计算表达式时，请支持已有序列、F_ext、square/cube/sqrt/log/exp/sin/cos/abs 等常见数学构造。
11. 请优先写一个自包含、可复用的分析脚本；observation 要说明你做了哪些处理、关键数值结果、哪些派生序列或图像已返回，以及这些结果对下一步决策有什么帮助。
12. observation 中的数值结论必须来自当前代码实际计算出的 metrics，不要写死“约等于”“明显降低”“通过验证”等未经计算支持的描述。
13. 不要把局部高 R²、单实验拟合或一组相近参数描述成定律；最多说“该模型在这些实验上拟合较好，需跨实验验证”。
14. validate_hypothesis 中，只有跨实验指标很强时才把 supports 设为 true；一般要求 R2>=0.99 且 RMSE 很小。否则 supports=false，并在 summary 说明反例/残差。
15. 代码要健壮，遇到不合适输入应 raise ValueError，错误信息要清楚。
16. 代码会经过静态权限检查；不要调用 eval/exec/open/__import__/getattr/globals/locals，避免访问双下划线特殊名称（包括 __file__），不要用 pathlib 读取、删除或遍历任意文件。

请只返回完整 Python 源码。"""
        return self._request_code(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            empty_message="数据处理 LLM 返回空代码。",
        )

    def repair_processor_code(
        self,
        *,
        action: str,
        parameters: dict[str, Any],
        notebook_context: str,
        recent_processors: list[dict[str, Any]],
        failed_code: str,
        error: str,
        payload_summary: dict[str, Any],
    ) -> str:
        """Ask the data-processing LLM to repair one failed processor."""
        system_prompt = (
            "你是一个严谨的数据处理 LLM，正在修复一段失败的 Python 数据分析代码。"
            "你必须只返回修复后的完整 Python 源码，不要 Markdown，不要解释。"
        )
        user_prompt = f"""
下面这段数据处理代码在静态检查或运行时失败了。请根据错误信息修复它，并返回完整 Python 源码。

Action:
{action}

Parameters JSON:
{json.dumps(parameters, ensure_ascii=False, indent=2)}

错误信息:
{error}

失败代码:
```python
{failed_code}
```

payload 摘要:
{json.dumps(payload_summary, ensure_ascii=False, indent=2)}

当前 notebook 数据上下文摘要:
{notebook_context}

最近已经生成并登记过的数据处理脚本:
{json.dumps(recent_processors, ensure_ascii=False, indent=2)}

修复要求:
1. 仍然只定义 def process(payload: dict) -> dict，并返回 observation、derived_series、figures、metrics。
2. 不要 import os、sys、subprocess、socket、requests、urllib、dotenv、importlib、autonomous_scientist；不要访问网络或读取外部文件。
3. 不要调用 eval/exec/open/__import__/getattr/globals/locals，不要访问双下划线特殊名称。
4. 如果原错误是禁止 import os，请直接删除 os import；输出路径只使用 payload["output_dir"] 和 pathlib.Path 拼接，不要自行创建目录。
5. 如果原错误是 NameError 或变量未定义，请保证所有变量在使用前赋值，尤其是 observation 字符串里的统计量。
6. 如果原错误是 Singular matrix，请改用 numpy.linalg.lstsq、numpy.linalg.pinv，或给设计矩阵加很小 ridge 正则；不要直接 np.linalg.inv(X.T @ X)。
7. 如果原错误是缺少某个派生序列，请从原始观测序列重新计算，不要假设 notebook 中已有派生序列。
8. 如果需要时间变化率，优先用 np.gradient，并在 observation 中说明边界误差风险。
9. 判断外力只能使用 config["F_ext"]；force_field_type=free 时 F_ext=0。
10. derived_series 的 values 长度必须与对应实验 t 长度一致；如果无法保证，就不要返回该序列。
11. 数值结论必须来自代码实际计算的 metrics，不要写死。
12. 如果 parameters["analysis_mode"]=="maintain_ledger"，修复后仍必须保持记录表维护模式：不要改成模型海选，应返回 observations 或明确说明为什么无法形成观察条目。
13. 如果 parameters["analysis_mode"]=="validate_hypothesis"，修复后仍必须返回 validations，且包含 hypothesis_id、supports 和误差指标。

请只返回修复后的完整 Python 源码。"""
        return self._request_code(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            empty_message="数据处理 LLM 返回空修复代码。",
        )

    def _request_code(self, *, system_prompt: str, user_prompt: str, empty_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
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
                content = response.choices[0].text
            except Exception:
                pass

        if not content:
            raise RuntimeError(empty_message)

        code = self._extract_code(str(content))
        if "def process(" not in code:
            raise RuntimeError("数据处理 LLM 生成的代码没有定义 process(payload)。")
        return code

    def _extract_code(self, content: str) -> str:
        content = content.strip()
        fenced = re.search(r"```(?:python|py)?\s*(.*?)\s*```", content, flags=re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        return content
