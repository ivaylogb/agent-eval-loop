"""Run the improvement loop on the customer support example."""

import sys
from pathlib import Path

from agent_eval_loop.loop import ImprovementLoop

# Local import: the mock handlers live alongside this script, not on sys.path.
sys.path.insert(0, str(Path(__file__).parent))
from mocks import get_handlers  # noqa: E402


def main():
    example_dir = Path(__file__).parent

    loop = ImprovementLoop(
        config_path=example_dir / "config.yaml",
        scenarios_path=example_dir / "scenarios.yaml",
        output_dir=example_dir / "outputs",
        max_iterations=3,
        max_conversations_per_iteration=10,  # small for demo; use 50+ in practice
        convergence_threshold=0.02,
        tool_handlers=get_handlers(),
    )

    best_config = loop.run()
    print(f"\nBest config: {best_config.name}")


if __name__ == "__main__":
    main()
