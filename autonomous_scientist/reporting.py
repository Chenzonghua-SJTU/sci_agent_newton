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
            f"- Number of invariant searches: `{len(notebook.invariant_history)}`",
            f"- Number of cross-experiment checks: `{len(notebook.generalization_checks)}`",
            f"- Number of candidate laws: `{len(notebook.candidate_laws)}`",
            "",
            "## Experimental Setup",
        ]

        if figure_paths:
            lines.extend(["## Visual Evidence", ""])
            for path in figure_paths:
                lines.append(f"![{path.stem}]({path.resolve()})")
                lines.append("")

        for experiment_id, record in sorted(notebook.experiments.items()):
            force_text = (
                str(record.config.constant_force)
                if record.config.force_field_type.value == "constant"
                else "N/A (free scene; constant_force ignored)"
            )
            lines.extend(
                [
                    f"### {experiment_id}",
                    f"- Force Field Type: `{record.config.force_field_type.value}`",
                    f"- External Force `F_ext`: `{force_text}`",
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

        lines.extend(["## Invariant Search Results"])
        if notebook.invariant_history:
            for idx, invariant in enumerate(notebook.invariant_history, start=1):
                complexity_text = (
                    "N/A" if invariant.complexity is None else f"{invariant.complexity:.6f}"
                )
                lines.extend(
                    [
                        f"### Invariant Search {idx}",
                        f"- Candidate Equation: `{invariant.equation}`",
                        f"- Loss: `{invariant.loss:.12f}`",
                        f"- Complexity: `{complexity_text}`",
                        f"- Residual Std: `{invariant.residual_std:.12f}`",
                        f"- Predicted Mean: `{invariant.predicted_mean:.12f}`",
                        f"- Score: `{invariant.score:.12f}`",
                        "",
                    ]
                )
        else:
            lines.append("- No invariant search was executed.")
            lines.append("")

        lines.extend(["## Cross-Experiment Generalization Checks"])
        if notebook.generalization_checks:
            for idx, check in enumerate(notebook.generalization_checks, start=1):
                lines.extend(
                    [
                        f"### Check {idx}",
                        f"- Expression: `{check.expression}`",
                        f"- Experiments: `{check.experiment_ids}`",
                        f"- Metric: `{check.metric_name}`",
                        f"- Aggregate Score: `{check.aggregate_score:.6f}`",
                        f"- Details: `{check.metric_values}`",
                        f"- Summary: {check.summary_text}",
                        "",
                    ]
                )
        else:
            lines.append("- No cross-experiment validation was executed.")
            lines.append("")

        lines.extend(["## Candidate Law Ranking"])
        if notebook.candidate_laws:
            ranked = sorted(notebook.candidate_laws, key=lambda item: item.score)
            for idx, candidate in enumerate(ranked, start=1):
                lines.extend(
                    [
                        f"### Candidate {idx}",
                        f"- Expression: `{candidate.expression}`",
                        f"- Source Experiment: `{candidate.source_experiment_id}`",
                        f"- Score: `{candidate.score:.12f}`",
                        f"- Origin: `{candidate.origin}`",
                        f"- Notes: {candidate.notes}",
                        "",
                    ]
                )
        else:
            lines.append("- No candidate laws were ranked.")
            lines.append("")

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

        if notebook.generalization_checks:
            for idx, check in enumerate(notebook.generalization_checks, start=1):
                figure_path = output_dir / f"generalization_check_{idx:02d}.png"
                fig, ax = plt.subplots(figsize=(8, 4.5))
                experiment_ids = list(check.metric_values.keys())
                metric_values = [check.metric_values[exp_id] for exp_id in experiment_ids]

                ax.bar(experiment_ids, metric_values)
                ax.set_title(f"Generalization check {idx}: {check.expression}")
                ax.set_xlabel("Experiment ID")
                ax.set_ylabel(check.metric_name)
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
