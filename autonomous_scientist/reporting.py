from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .agent import ScientificCycleResult


class ScientificReporter:
    """生成和保存科研 Markdown 报告的工具类。"""

    def __init__(self, default_filename_prefix: str = "scientific_report") -> None:
        self.default_filename_prefix = default_filename_prefix

    def generate_markdown(
        self,
        cycle_result: ScientificCycleResult,
        figure_dir: str | Path | None = None,
    ) -> str:
        notebook = cycle_result.notebook
        final_law = cycle_result.final_law
        figure_paths = self.generate_figures(cycle_result, figure_dir) if figure_dir is not None else []
        cycle_result.figure_paths = [str(path) for path in figure_paths]

        lines: list[str] = [
            "# Autonomous Scientist Report",
            "",
            "## Research Process",
            f"- Number of experiments: `{len(notebook.experiments)}`",
            f"- Number of actions executed: `{len(notebook.action_history)}`",
            f"- Number of ledger observations: `{len(notebook.observations)}`",
            f"- Number of hypothesis validations: `{len(notebook.validations)}`",
            f"- Number of hypotheses: `{len(notebook.hypothesis_registry.all_records())}`",
            "",
            "## Experimental Setup",
        ]

        if figure_paths:
            lines.extend(["## Visual Evidence", ""])
            for path in figure_paths:
                lines.append(f"![{path.stem}]({path.resolve()})")
                lines.append("")

        for experiment_id, record in sorted(notebook.experiments.items()):
            force_value = (
                float(record.config.constant_force)
                if record.config.force_field_type.value == "constant"
                else 0.0
            )
            lines.extend(
                [
                    f"### {experiment_id}",
                    f"- Force Field Type: `{record.config.force_field_type.value}`",
                    f"- External Force `F_ext`: `{force_value:.6g}`",
                    f"- Initial Position `q0`: `{record.config.initial_q}`",
                    f"- Initial Velocity `v0`: `{record.config.initial_v}`",
                    f"- Time Span: `{record.config.t_span}`",
                    f"- Sampling Interval `dt`: `{record.config.dt}`",
                    f"- Observation Noise Std: `{record.config.noise_std}`",
                    "",
                ]
            )

        lines.extend(
            [
                "## Action History",
            ]
        )
        for action_record in notebook.action_history:
            lines.extend(
                [
                    f"### Step {action_record.step_index}",
                    f"- Thought: {action_record.decision.thought}",
                    f"- Action: `{action_record.decision.action}`",
                    f"- Parameters: `{action_record.decision.parameters}`",
                    f"- Observation: {action_record.observation}",
                    "",
                ]
            )

        lines.extend(["## Notebook Notes"])
        for note in notebook.notes[-12:]:
            lines.append(f"- {note}")
        lines.append("")

        lines.extend(["## Experiment Data Ledger"])
        if notebook.observations:
            lines.append("### Observations")
            for observation in notebook.observations:
                lines.extend(
                    [
                        f"- `{observation.observation_id}` step `{observation.step_index}`",
                        f"  - Summary: {observation.summary}",
                        f"  - Source Data Refs: `{observation.source_data_refs}`",
                        f"  - Metrics: `{observation.metrics}`",
                    ]
                )
        else:
            lines.append("- No ledger observations were recorded.")
        lines.append("")

        if notebook.validations:
            lines.append("### Validations")
            for validation in notebook.validations:
                verdict = "supports" if validation.supports else "refutes"
                lines.extend(
                    [
                        f"- `{validation.validation_id}` {verdict} `{validation.hypothesis_id}`",
                        f"  - Experiments: `{validation.experiment_ids}`",
                        f"  - Metric: `{validation.metric_name}`",
                        f"  - Aggregate Score: `{validation.aggregate_score}`",
                        f"  - Details: `{validation.metric_values}`",
                        f"  - Summary: {validation.summary}",
                    ]
                )
        else:
            lines.append("- No hypothesis validations were recorded.")
        lines.append("")

        lines.extend(notebook.hypothesis_registry.to_markdown())

        lines.extend(
            [
                "## Final Law Hypothesis",
                f"- Summary: {final_law.summary}",
                f"- Proposed Law: `{final_law.proposed_law}`",
                f"- Evidence: {final_law.evidence}",
                f"- Confidence: `{final_law.confidence}`",
                f"- Next Steps: {final_law.next_steps}",
                "",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    def generate_figures(
        self,
        cycle_result: ScientificCycleResult,
        figure_dir: str | Path,
    ) -> list[Path]:
        """为实验记录与泛化检查生成图像证据。"""
        output_dir = Path(figure_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        notebook = cycle_result.notebook

        generated_paths: list[Path] = []

        for experiment_id, record in sorted(notebook.experiments.items()):
            figure_path = output_dir / f"{experiment_id}_trajectory.png"
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

            axes[0].plot(record.result.t, record.result.q, label="q(t)", linewidth=2)
            axes[0].set_ylabel("Position q")
            axes[0].set_title(f"{experiment_id} trajectory")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            derived = notebook.derived_series.get(experiment_id, {})
            plotted_any = False
            for series_name, series in sorted(derived.items()):
                axes[1].plot(record.result.t, series.values, label=series_name, linewidth=1.5)
                plotted_any = True

            if not plotted_any:
                axes[1].plot(record.result.t, record.result.q, label="q(t)", linewidth=1.5)
            axes[1].set_xlabel("Time t")
            axes[1].set_ylabel("Derived / inspected series")
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            fig.tight_layout()
            fig.savefig(figure_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            generated_paths.append(figure_path)

        for idx, validation in enumerate(notebook.validations, start=1):
            if not validation.metric_values:
                continue
            figure_path = output_dir / f"validation_{idx:02d}.png"
            fig, ax = plt.subplots(figsize=(8, 4.5))
            labels = list(validation.metric_values.keys())
            values = [validation.metric_values[label] for label in labels]

            ax.bar(labels, values)
            ax.set_title(f"{validation.validation_id}: {validation.hypothesis_id}")
            ax.set_xlabel("Metric key")
            ax.set_ylabel(validation.metric_name)
            ax.grid(axis="y", alpha=0.3)

            fig.tight_layout()
            fig.savefig(figure_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            generated_paths.append(figure_path)

        return generated_paths

    def save_report(
        self,
        markdown_text: str,
        output_dir: str | Path,
        filename: str | None = None,
    ) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.default_filename_prefix}_{timestamp}.md"

        report_path = output_path / filename
        report_path.write_text(markdown_text, encoding="utf-8")
        return report_path
