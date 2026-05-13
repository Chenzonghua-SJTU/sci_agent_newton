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
    ) -> None:
        if OpenAI is None:
            raise ImportError("未检测到 openai 官方库。")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
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
            "你不是决策者，不提出新的科研结论，只实现决策 LLM 已经指定的数据处理 action。"
            "你必须只返回 Python 源码，不要 Markdown，不要解释。"
        )
        user_prompt = f"""
请为下面的数据处理 action 生成一个可执行 Python 文件。

Action:
{action}

Parameters JSON:
{json.dumps(parameters, ensure_ascii=False, indent=2)}

如果 Action 是 custom_data_analysis:
- parameters["analysis_goal"] 是决策 LLM 提出的自然语言科学问题，是本次代码的主要需求。
- optional_series 可作为优先关注的序列，但你可以根据数据上下文自行选择更多序列。
- expected_outputs 是决策 LLM 希望看到的输出类型；如果没有提供，你也应主动返回最有信息量的 observation、metrics、必要图像和可复用派生序列。
- 你可以自行决定是否平滑、求导、拟合、比较实验、画图、构造残差或派生量，但不要声称发现最终定律；最终判断属于决策 LLM。

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
  - "config": 实验控制参数，包括 force_field_type, constant_force, initial_q, initial_v, t_span, dt, noise_std。
  - "series": dict，包含 "t", "q" 以及此前已经登记的派生序列；值都是 list[float]。
  - "available_series": 可用序列名列表。
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
  "figures": ["可选图像绝对路径或相对路径"],
  "metrics": {{"rmse": 0.01}}
}}

硬性要求:
1. 只能使用 Python 标准库以及 numpy、pandas、scipy、sklearn、matplotlib。
2. 不要 import autonomous_scientist，不要读取 universe.py，不要读取 .env，不要访问网络。
3. 不要修改 payload 输入数据；新序列必须通过 returned derived_series 返回。
4. 需要图像时，把图像保存到 payload["output_dir"] 下，并把路径写入 figures。
5. derived_series 中 values 的长度必须与对应实验的 t 序列长度一致。
6. 如果参数里包含 experiment_ids，则只处理这些实验；如果只有 experiment_id，则只处理它；如果都没有，则默认处理所有实验。
7. 如果 action 是 inspect_series 或 inspect_relationships，可以只返回 observation 和 figures，不必返回 derived_series。
8. 如果 action 是 fit_relationship_model，应返回 prediction/residual 派生序列，并在 observation 中报告系数、R2、RMSE、MAE。
9. 如果 action 是 estimate_kinematics，可用 scipy.signal.savgol_filter 从 q(t) 估计 q_smooth/v/a。
10. 如果 action 是 define_derived_quantity 或 test_candidate_expression，需要支持表达式中的已有序列、F_ext、square/cube/sqrt/log/exp/sin/cos/abs。
11. 如果 action 是 custom_data_analysis，请优先写一个自包含、可复用的分析脚本；observation 要说明你做了哪些处理、关键数值结果、哪些派生序列或图像已返回，以及这些结果对下一步决策有什么帮助。
12. observation 中的数值结论必须来自当前代码实际计算出的 metrics，不要写死“约等于”“明显降低”“通过验证”等未经计算支持的描述。
13. 不要把局部高 R²、单实验拟合或一组相近参数描述成定律；最多说“该模型在这些实验上拟合较好，需跨实验验证”。
14. 代码要健壮，遇到不合适输入应 raise ValueError，错误信息要清楚。

请只返回完整 Python 源码。"""
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
            raise RuntimeError("数据处理 LLM 返回空代码。")

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
