"""Loop orchestrator: run the full simulate → evaluate → improve cycle.

This is the main entry point. The loop runs entirely offline — only
after convergence does the candidate proceed to human review.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anthropic
import jsonlines
from rich.console import Console
from rich.panel import Panel

from agent_eval_loop.agent.config import load_config
from agent_eval_loop.evaluate.judges import get_standard_judges
from agent_eval_loop.evaluate.scorer import Scorer
from agent_eval_loop.improve.analyzer import FailureAnalyzer
from agent_eval_loop.improve.optimizer import Optimizer
from agent_eval_loop.improve.regression import check_regression
from agent_eval_loop.models import (
    AgentConfig,
    ConversationEval,
    EvalCategory,
    LoopIteration,
    LoopState,
)
from agent_eval_loop.simulate.generator import ConversationGenerator
from agent_eval_loop.simulate.personas import get_all_personas
from agent_eval_loop.simulate.scenarios import load_scenarios

console = Console()


class ImprovementLoop:
    """Orchestrate the full simulate → evaluate → improve cycle."""

    def __init__(
        self,
        config_path: str | Path,
        scenarios_path: str | Path,
        output_dir: str | Path = "outputs",
        max_iterations: int = 5,
        convergence_threshold: float = 0.02,
        max_conversations_per_iteration: int = 20,
        eval_categories: list[EvalCategory] | None = None,
        tool_handlers: dict[str, Any] | None = None,
    ):
        self.config_path = Path(config_path)
        self.scenarios_path = Path(scenarios_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.max_conversations = max_conversations_per_iteration
        self.tool_handlers = tool_handlers or {}

        # Shared Anthropic client for all components
        self.client = anthropic.Anthropic()

        # Load initial config and scenarios
        self.current_config = load_config(config_path)
        self.scenarios = load_scenarios(str(scenarios_path))
        self.personas = get_all_personas()

        # Set up evaluation
        self.judges = get_standard_judges(categories=eval_categories, client=self.client)
        self.scorer = Scorer(self.judges)
        self.analyzer = FailureAnalyzer()

        # Loop state
        self.state = LoopState(
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        )


    def run(self) -> AgentConfig:
        """Run the full improvement loop. Returns the best agent config."""
        console.print(Panel(
            f"[bold]Starting improvement loop[/bold]\n"
            f"Agent: {self.current_config.name}\n"
            f"Max iterations: {self.max_iterations}\n"
            f"Convergence threshold: {self.convergence_threshold}\n"
            f"Scenarios: {len(self.scenarios.scenarios)}\n"
            f"Personas: {len(self.personas)}",
            title="agent-eval-loop",
            border_style="blue",
        ))

        for i in range(self.max_iterations):
            console.print(f"\n[bold cyan]═══ Iteration {i + 1}/{self.max_iterations} ═══[/bold cyan]")

            iteration = self._run_iteration(i + 1)
            self.state.iterations.append(iteration)

            if iteration.converged:
                console.print(f"\n[bold green]✓ Converged after {i + 1} iterations[/bold green]")
                break

            if i < self.max_iterations - 1:
                console.print(f"[yellow]Score: {iteration.aggregate_score:.3f} — continuing...[/yellow]")

        self._save_state()
        self.state.current_best_config = self.current_config.name

        console.print(Panel(
            f"[bold green]Loop complete[/bold green]\n"
            f"Best config: {self.current_config.name}\n"
            f"Final score: {self.state.iterations[-1].aggregate_score:.3f}\n"
            f"Iterations run: {len(self.state.iterations)}",
            title="Results",
            border_style="green",
        ))

        return self.current_config

    def _run_iteration(self, iteration_num: int) -> LoopIteration:
        """Run a single simulate → evaluate → improve iteration."""
        iteration = LoopIteration(
            iteration_number=iteration_num,
            agent_config_name=self.current_config.name,
        )

        # --- SIMULATE ---
        console.print("\n[bold blue]1. SIMULATE[/bold blue] — Generating conversations...")
        generator = ConversationGenerator(
            agent_config=self.current_config,
            tool_handlers=self.tool_handlers,
        )
        conversations = generator.generate_batch(
            scenarios=self.scenarios,
            personas=self.personas,
            max_conversations=self.max_conversations,
        )
        iteration.conversations_generated = len(conversations)
        self._save_conversations(conversations, iteration_num)
        console.print(f"  Generated {len(conversations)} conversations")

        # --- EVALUATE ---
        console.print("\n[bold green]2. EVALUATE[/bold green] — Running judges...")
        eval_results = self.scorer.evaluate_batch(conversations)
        iteration.eval_results = eval_results
        self.scorer.print_summary(eval_results)

        if eval_results:
            iteration.aggregate_score = sum(
                r.aggregate_score for r in eval_results
            ) / len(eval_results)

        self._save_evals(eval_results, iteration_num)

        # --- CHECK CONVERGENCE ---
        if self._check_convergence(iteration):
            iteration.converged = True
            return iteration

        # --- IMPROVE ---
        console.print("\n[bold yellow]3. IMPROVE[/bold yellow] — Analyzing failures and proposing fixes...")
        failure_patterns = self.analyzer.analyze(eval_results)
        iteration.failure_patterns = failure_patterns
        self.analyzer.print_report(failure_patterns)

        if failure_patterns:
            optimizer = Optimizer(
                self.current_config,
                client=self.client,
            )
            candidates = optimizer.propose_improvements(failure_patterns)
            iteration.candidates = candidates

            if candidates:
                best_candidate = candidates[0]
                console.print(f"  Proposed: {best_candidate.change_description}")

                candidate_config = optimizer.apply_candidate(
                    best_candidate,
                    output_dir=self.output_dir / f"iter{iteration_num}",
                )

                # Regression test: re-run the candidate on the SAME
                # (persona, scenario) pairs that just produced this
                # iteration's eval_results. That makes the comparison
                # apples-to-apples — current config vs candidate config
                # on one fixed sample.
                console.print("  Running regression tests...")
                pairs_by_id = {
                    (p.id, s.id): (p, s)
                    for p in self.personas
                    for s in self.scenarios.scenarios
                }
                baseline_pairs = [
                    pairs_by_id[(c.persona_id, c.scenario_id)]
                    for c in conversations
                    if (c.persona_id, c.scenario_id) in pairs_by_id
                ]

                candidate_generator = ConversationGenerator(
                    agent_config=candidate_config,
                    tool_handlers=self.tool_handlers,
                )
                candidate_conversations = candidate_generator.generate_batch(
                    pairs=baseline_pairs,
                )
                candidate_evals = self.scorer.evaluate_batch(candidate_conversations)

                regression = check_regression(
                    conversations,
                    eval_results,
                    candidate_conversations,
                    candidate_evals,
                )
                console.print(f"  Regression: {regression.summary}")

                if regression.passed:
                    console.print("  [green]✓ No regressions — applying improvement[/green]")
                    self.current_config = candidate_config
                    best_candidate.regression_passed = True
                else:
                    console.print("  [red]✗ Regressions detected — keeping current config[/red]")

        return iteration

    def _check_convergence(self, current: LoopIteration) -> bool:
        """Check if the loop has converged.

        Converged means: we've run at least 2 iterations and the score
        improvement is below the threshold.
        """
        if len(self.state.iterations) < 1:
            # First iteration — can't check convergence yet
            return False

        previous_score = self.state.iterations[-1].aggregate_score
        improvement = current.aggregate_score - previous_score

        # Converged if improvement is below threshold (including negative)
        if abs(improvement) < self.convergence_threshold:
            console.print(
                f"  [dim]Score delta ({improvement:+.4f}) within "
                f"threshold (±{self.convergence_threshold})[/dim]"
            )
            return True

        return False

    def _save_conversations(self, conversations, iteration_num: int) -> None:
        path = self.output_dir / f"conversations_iter{iteration_num}.jsonl"
        with jsonlines.open(path, mode="w") as writer:
            for conv in conversations:
                writer.write(conv.model_dump(mode="json"))

    def _save_evals(self, results: list[ConversationEval], iteration_num: int) -> None:
        path = self.output_dir / f"evals_iter{iteration_num}.jsonl"
        with jsonlines.open(path, mode="w") as writer:
            for result in results:
                writer.write(result.model_dump(mode="json"))

    def _save_state(self) -> None:
        path = self.output_dir / "loop_state.json"
        with open(path, "w") as f:
            json.dump(self.state.model_dump(mode="json"), f, indent=2, default=str)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the agent improvement loop")
    parser.add_argument("--config", required=True, help="Path to agent config YAML")
    parser.add_argument("--scenarios", required=True, help="Path to scenarios YAML")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations")
    parser.add_argument("--conversations", type=int, default=20, help="Max conversations per iteration")
    parser.add_argument("--threshold", type=float, default=0.02, help="Convergence threshold")

    args = parser.parse_args()

    loop = ImprovementLoop(
        config_path=args.config,
        scenarios_path=args.scenarios,
        output_dir=args.output,
        max_iterations=args.iterations,
        max_conversations_per_iteration=args.conversations,
        convergence_threshold=args.threshold,
    )

    loop.run()


if __name__ == "__main__":
    main()
