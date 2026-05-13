from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from autonomous_scientist import (
    ActionRecord,
    DataProcessingTool,
    DataProcessingBrain,
    HypothesisBrain,
    ScientistAgent,
    VerificationEngine,
    VirtualUniverse,
)


load_dotenv()


DIVIDER = "=" * 72


def log(message: str) -> None:
    """统一的控制台日志输出。"""
    print(message, flush=True)


def ensure_openai_api_key() -> str:
    """读取并校验可用的 LLM API Key。

    Returns:
        已读取到的 API Key。

    Raises:
        RuntimeError: 如果环境变量未设置。
    """
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        return deepseek_api_key

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return openai_api_key

    raise RuntimeError(
        "未检测到可用的 API Key。\n"
        "你可以优先配置 DeepSeek：\n"
        'export DEEPSEEK_API_KEY="your_deepseek_api_key_here"\n'
        "或配置 OpenAI：\n"
        'export OPENAI_API_KEY="your_openai_api_key_here"'
    )


def resolve_llm_provider() -> tuple[str, str, str | None]:
    """解析当前要使用的 LLM 提供商配置。

    Returns:
        provider_name, model_name, base_url
    """
    if os.getenv("DEEPSEEK_API_KEY"):
        return (
            "DeepSeek",
            os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "https://api.deepseek.com",
        )

    return (
        "OpenAI",
        os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        None,
    )


def build_agent(api_key: str, model: str, base_url: str | None) -> ScientistAgent:
    """初始化自主科研 Agent 的全部核心模块。"""
    universe = VirtualUniverse(
        alpha=1.0,
        base_mass=1.0,
        random_seed=42,
    )
    data_tool = DataProcessingTool(
        window_length= nine_point_window(),
        polyorder=3,
    )
    verification_engine = VerificationEngine(
        niterations=150,
        population_size=50,
        maxsize=40,
    )
    brain = HypothesisBrain(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
    )
    data_brain = DataProcessingBrain(
        model=os.getenv("DATA_PROCESSING_MODEL", model),
        api_key=api_key,
        base_url=os.getenv("DATA_PROCESSING_BASE_URL") or base_url,
        temperature=float(os.getenv("DATA_PROCESSING_TEMPERATURE", "0.0")),
    )
    return ScientistAgent(
        universe=universe,
        data_tool=data_tool,
        verification_engine=verification_engine,
        brain=brain,
        data_brain=data_brain,
        generated_code_dir=Path.cwd() / "generated_processors",
        use_generated_processors=os.getenv("USE_GENERATED_PROCESSORS", "true").lower()
        not in {"0", "false", "no", "n"},
    )


def nine_point_window() -> int:
    """单独抽出窗口配置，便于后续脚本层调参。"""
    return 9


def log_agent_step(action_record: ActionRecord) -> None:
    """实时打印 Agent 的每一步，避免长循环看起来像卡住。"""
    log("-" * 72)
    log(f"Step {action_record.step_index}")
    log(f"Thought: {action_record.decision.thought}")
    log(f"Action: {action_record.decision.action}")
    if action_record.decision.parameters:
        log(f"Parameters: {action_record.decision.parameters}")
    log(f"Observation: {action_record.observation}")


def main() -> int:
    """运行一次完整的自主科研循环。"""
    try:
        log(DIVIDER)
        log("🚀 启动自主科研 Agent...")
        log(DIVIDER)

        log("🔑 正在读取环境变量中的 API Key...")
        api_key = ensure_openai_api_key()
        provider_name, model_name, base_url = resolve_llm_provider()

        log(f"🧩 正在初始化系统，当前 LLM 提供商: {provider_name}，模型: {model_name}")
        agent = build_agent(api_key=api_key, model=model_name, base_url=base_url)

        log(DIVIDER)
        log("🧪 实验任务：Agent 只能从时间-位置数据出发，自主决定下一步实验与分析动作")
        log("🧠 LLM 将逐步规划：是否做新实验、是否平滑、是否求导、是否主动提出候选公式")
        log("🔁 当出现候选规律后，Agent 会尝试在不同初速度/不同外力下做跨实验复验")
        log("📘 最终会输出一份包含动作历史、泛化验证和规律总结的 Markdown 科研报告")
        log(DIVIDER)

        max_steps = int(os.getenv("MAX_AGENT_STEPS", "40"))
        cycle_result = agent.run_scientific_cycle(
            max_steps=max_steps,
            progress_callback=log_agent_step,
        )
        figure_dir = Path.cwd() / "report_assets"
        cycle_result.report_markdown = agent.reporter.generate_markdown(
            cycle_result,
            figure_dir=figure_dir,
        )

        report_filename = "discovery_report.md"
        report_path = agent.reporter.save_report(
            markdown_text=cycle_result.report_markdown or "",
            output_dir=Path.cwd(),
            filename=report_filename,
        )

        log("✅ 科研循环执行完成。")
        log(DIVIDER)
        log("【动作历史】")
        for action_record in cycle_result.notebook.action_history:
            log(
                f"Step {action_record.step_index}: "
                f"{action_record.decision.action} | "
                f"thought={action_record.decision.thought}"
            )
            log(f"Observation: {action_record.observation}")
        log(DIVIDER)
        if cycle_result.notebook.invariant_history:
            latest = cycle_result.notebook.invariant_history[-1]
            log("【最新不变量搜索结果】")
            log(f"候选表达式: {latest.equation}")
            log(f"Score: {latest.score:.12f}")
            log(f"Residual Std: {latest.residual_std:.12f}")
            if latest.complexity is not None:
                log(f"Complexity: {latest.complexity:.6f}")
            else:
                log("Complexity: N/A")
            log(DIVIDER)
        if cycle_result.notebook.generalization_checks:
            latest_check = cycle_result.notebook.generalization_checks[-1]
            log("【最新跨实验验证】")
            log(f"Expression: {latest_check.expression}")
            log(f"Experiments: {latest_check.experiment_ids}")
            log(f"Metric: {latest_check.metric_name}")
            log(f"Aggregate Score: {latest_check.aggregate_score:.6f}")
            log(f"Summary: {latest_check.summary_text}")
            log(DIVIDER)
        if cycle_result.notebook.candidate_laws:
            ranked = sorted(cycle_result.notebook.candidate_laws, key=lambda item: item.score)
            log("【候选规律排序】")
            for candidate in ranked[:5]:
                log(
                    f"{candidate.expression} | score={candidate.score:.6f} | "
                    f"origin={candidate.origin} | exp={candidate.source_experiment_id}"
                )
            log(DIVIDER)
        log("【最终规律总结】")
        log(f"Summary: {cycle_result.final_law.summary}")
        log(f"Proposed Law: {cycle_result.final_law.proposed_law}")
        log(f"Evidence: {cycle_result.final_law.evidence}")
        log(f"Confidence: {cycle_result.final_law.confidence}")
        log(f"Next Steps: {cycle_result.final_law.next_steps}")
        log(DIVIDER)
        if cycle_result.figure_paths:
            log("【图像证据】")
            for figure_path in cycle_result.figure_paths:
                log(figure_path)
            log(DIVIDER)
        log(f"📄 报告已保存至: {report_path}")
        log(DIVIDER)
        return 0

    except KeyboardInterrupt:
        log("\n⚠️ 用户中断了本次科研循环。")
        return 130
    except Exception as exc:
        log(DIVIDER)
        log("❌ 自主科研 Agent 运行失败。")
        log(f"错误信息: {exc}")
        log(
            "排查建议：确认已安装 `openai`, `numpy`, `scipy`, `pandas`, `pysr`，"
            "并确保 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY` 已正确设置。"
        )
        log(DIVIDER)
        return 1


if __name__ == "__main__":
    sys.exit(main())
