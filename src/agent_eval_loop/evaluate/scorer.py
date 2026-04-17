"""Scorer: run judges across conversations and aggregate results."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from agent_eval_loop.models import (
    Conversation,
    ConversationEval,
    EvalCategory,
)
from agent_eval_loop.evaluate.judges import Judge

console = Console()


class CategorySummary(BaseModel):
    """Aggregate stats for one eval category."""

    category: EvalCategory
    mean_score: float
    pass_rate: float
    total_evaluated: int


class EvalSummary(BaseModel):
    """Aggregate stats across all categories and conversations."""

    total_conversations: int = 0
    overall_mean_score: float = 0.0
    overall_pass_rate: float = 0.0
    category_summaries: dict[EvalCategory, CategorySummary] = Field(default_factory=dict)


class Scorer:
    """Run evaluation judges across conversations and aggregate results."""

    def __init__(self, judges: list[Judge]):
        self.judges = judges

    def evaluate_batch(
        self,
        conversations: list[Conversation],
    ) -> list[ConversationEval]:
        """Evaluate all conversations with all judges."""
        results = []

        total_evals = len(conversations) * len(self.judges)
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Running {total_evals} evaluations...",
                total=total_evals,
            )

            for conversation in conversations:
                verdicts = []
                for judge in self.judges:
                    verdict = judge.evaluate(conversation)
                    verdicts.append(verdict)
                    progress.advance(task)

                eval_result = ConversationEval(
                    conversation_id=conversation.id,
                    verdicts=verdicts,
                )
                eval_result.compute_aggregate()
                results.append(eval_result)

        return results

    def summarize(self, results: list[ConversationEval]) -> EvalSummary:
        """Compute aggregate statistics from eval results."""
        if not results:
            return EvalSummary()

        category_scores: dict[EvalCategory, list[float]] = defaultdict(list)
        category_passes: dict[EvalCategory, int] = defaultdict(int)
        category_totals: dict[EvalCategory, int] = defaultdict(int)

        for result in results:
            for verdict in result.verdicts:
                category_scores[verdict.category].append(verdict.score)
                category_totals[verdict.category] += 1
                if verdict.passed:
                    category_passes[verdict.category] += 1

        category_summaries = {}
        for cat in category_scores:
            scores = category_scores[cat]
            category_summaries[cat] = CategorySummary(
                category=cat,
                mean_score=sum(scores) / len(scores),
                pass_rate=category_passes[cat] / category_totals[cat],
                total_evaluated=category_totals[cat],
            )

        all_scores = [r.aggregate_score for r in results]
        overall_pass = sum(1 for r in results if r.passed)

        return EvalSummary(
            total_conversations=len(results),
            overall_mean_score=sum(all_scores) / len(all_scores),
            overall_pass_rate=overall_pass / len(results),
            category_summaries=category_summaries,
        )

    def print_summary(self, results: list[ConversationEval]) -> None:
        """Print a formatted summary table to the console."""
        summary = self.summarize(results)

        table = Table(title="Evaluation Summary", show_header=True)
        table.add_column("Category", style="cyan")
        table.add_column("Mean Score", justify="right")
        table.add_column("Pass Rate", justify="right")
        table.add_column("Evaluated", justify="right")

        for cat, cat_summary in summary.category_summaries.items():
            color = "green" if cat_summary.mean_score >= 0.7 else "yellow" if cat_summary.mean_score >= 0.5 else "red"
            table.add_row(
                cat.value,
                f"[{color}]{cat_summary.mean_score:.2f}[/{color}]",
                f"{cat_summary.pass_rate:.0%}",
                str(cat_summary.total_evaluated),
            )

        table.add_section()
        color = "green" if summary.overall_mean_score >= 0.7 else "yellow" if summary.overall_mean_score >= 0.5 else "red"
        table.add_row(
            "[bold]Overall[/bold]",
            f"[bold {color}]{summary.overall_mean_score:.2f}[/bold {color}]",
            f"[bold]{summary.overall_pass_rate:.0%}[/bold]",
            str(summary.total_conversations),
        )

        console.print(table)
