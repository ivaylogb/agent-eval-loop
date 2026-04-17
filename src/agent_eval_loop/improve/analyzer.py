"""Failure analyzer: identify recurring patterns from eval results.

After evaluation, we have per-conversation verdicts. The analyzer
groups failures by category and component, identifies recurring
patterns, and produces actionable failure reports that feed the
improvement stage.
"""

from __future__ import annotations

from collections import defaultdict

from rich.console import Console
from rich.table import Table

from agent_eval_loop.models import (
    ComponentType,
    ConversationEval,
    EvalCategory,
    FailurePattern,
    JudgeVerdict,
)

console = Console()


class FailureAnalyzer:
    """Analyze eval results to identify recurring failure patterns."""

    def __init__(self, min_frequency: int = 2):
        self.min_frequency = min_frequency

    def analyze(self, results: list[ConversationEval]) -> list[FailurePattern]:
        """Identify failure patterns from evaluation results.

        Groups failures by (category, component) and looks for
        recurring themes in the reasoning.
        """
        # Collect all failures
        failures_by_key: dict[
            tuple[EvalCategory, ComponentType | None],
            list[tuple[str, JudgeVerdict]],
        ] = defaultdict(list)

        for result in results:
            for verdict in result.verdicts:
                if not verdict.passed:
                    key = (verdict.category, verdict.failure_component)
                    failures_by_key[key].append(
                        (result.conversation_id, verdict)
                    )

        # Build failure patterns
        patterns = []
        for (category, component), failures in failures_by_key.items():
            if len(failures) < self.min_frequency:
                continue

            # Aggregate reasoning across failures
            reasonings = [v.reasoning for _, v in failures]
            conv_ids = [cid for cid, _ in failures]

            pattern = FailurePattern(
                category=category,
                component=component or ComponentType.INSTRUCTIONS,  # default
                description=self._summarize_failures(reasonings),
                frequency=len(failures),
                example_conversation_ids=conv_ids[:5],
                suggested_fix=self._suggest_fix(category, component, reasonings),
            )
            patterns.append(pattern)

        # Sort by frequency (most common first)
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns

    def _summarize_failures(self, reasonings: list[str]) -> str:
        """Summarize multiple failure reasonings into a pattern description.

        For now, takes the first reasoning as representative.
        In production, use an LLM to synthesize across all reasonings.
        """
        if not reasonings:
            return "Unknown failure pattern"
        # Simple: use the most common reasoning
        # Advanced: cluster reasonings and summarize each cluster
        return reasonings[0]

    def _suggest_fix(
        self,
        category: EvalCategory,
        component: ComponentType | None,
        reasonings: list[str],
    ) -> str:
        """Generate a suggested fix based on the failure pattern.

        This is a heuristic starting point. The optimizer module
        can propose more sophisticated fixes.
        """
        suggestions = {
            (EvalCategory.TOOL_SELECTION, ComponentType.TOOLS):
                "Review tool descriptions for clarity. Ensure each tool's "
                "description includes when NOT to use it. Add example invocations.",
            (EvalCategory.RESPONSE_ACCURACY, ComponentType.ROUTINES):
                "Check routine for missing steps or ambiguous instructions. "
                "Ensure the routine specifies what information to verify before responding.",
            (EvalCategory.TONE_APPROPRIATENESS, ComponentType.INSTRUCTIONS):
                "Add explicit tone guidance to instructions. Include examples of "
                "appropriate responses for frustrated vs. neutral customers.",
            (EvalCategory.ROUTINE_ADHERENCE, ComponentType.ROUTINES):
                "Simplify the routine or break it into smaller, clearer steps. "
                "Add explicit transition conditions between steps.",
            (EvalCategory.ESCALATION_DECISION, ComponentType.INSTRUCTIONS):
                "Define clear escalation criteria in instructions. List specific "
                "scenarios that require vs. don't require human handoff.",
        }

        key = (category, component)
        comp_label = component.value if component else "agent"
        default = f"Review the {comp_label} component for issues related to {category.value}."
        return suggestions.get(key, default)

    def print_report(self, patterns: list[FailurePattern]) -> None:
        """Print a formatted failure analysis report."""
        if not patterns:
            console.print("[green]No recurring failure patterns found.[/green]")
            return

        table = Table(title="Failure Patterns (sorted by frequency)", show_header=True)
        table.add_column("Category", style="cyan")
        table.add_column("Component", style="yellow")
        table.add_column("Frequency", justify="right", style="red")
        table.add_column("Description", max_width=50)
        table.add_column("Suggested Fix", max_width=40, style="green")

        for pattern in patterns:
            table.add_row(
                pattern.category.value,
                pattern.component.value,
                str(pattern.frequency),
                pattern.description[:50] + ("..." if len(pattern.description) > 50 else ""),
                pattern.suggested_fix[:40] + ("..." if len(pattern.suggested_fix) > 40 else ""),
            )

        console.print(table)
