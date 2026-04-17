"""Run the improvement loop on the customer support example."""

from pathlib import Path

from agent_eval_loop.loop import ImprovementLoop


def main():
    example_dir = Path(__file__).parent

    loop = ImprovementLoop(
        config_path=example_dir / "config.yaml",
        scenarios_path=example_dir / "scenarios.yaml",
        output_dir=example_dir / "outputs",
        max_iterations=3,
        max_conversations_per_iteration=10,  # small for demo; use 50+ in practice
        convergence_threshold=0.02,
    )

    best_config = loop.run()
    print(f"\nBest config: {best_config.name}")


if __name__ == "__main__":
    main()
