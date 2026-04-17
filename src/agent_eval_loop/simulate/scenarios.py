"""Scenarios: seed conversation starters and known failure cases.

Scenarios define the *situation* — what happened, what the customer
wants. Combined with personas (who the customer *is*), they generate
diverse, targeted test conversations.

Sources for scenarios:
1. Known failure cases from production (anonymized)
2. Edge cases identified during development
3. Happy paths that should always work
4. Adversarial cases designed to break the agent
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ScenarioDifficulty(str, Enum):
    EASY = "easy"          # Happy path, straightforward
    MEDIUM = "medium"      # Some complexity, maybe multi-step
    HARD = "hard"          # Edge case, ambiguous, or adversarial
    ADVERSARIAL = "adversarial"  # Designed to break the agent


class ScenarioCategory(str, Enum):
    HAPPY_PATH = "happy_path"
    ERROR_HANDLING = "error_handling"
    EDGE_CASE = "edge_case"
    MULTI_STEP = "multi_step"
    ESCALATION = "escalation"
    ADVERSARIAL = "adversarial"
    REGRESSION = "regression"  # Previously broken, now should work


class Scenario(BaseModel):
    """A test scenario that seeds a simulated conversation."""
    id: str
    name: str
    description: str
    category: ScenarioCategory
    difficulty: ScenarioDifficulty
    opening_message: str  # The customer's first message
    expected_tools: list[str] = Field(default_factory=list)  # Tools the agent should use
    expected_outcome: str = ""  # What a good resolution looks like
    failure_modes: list[str] = Field(default_factory=list)  # Known ways this can go wrong
    context: dict[str, Any] = Field(default_factory=dict)  # Mock data for tools
    max_turns: int = 10
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScenarioSuite(BaseModel):
    """A collection of scenarios for a domain."""
    name: str
    description: str
    scenarios: list[Scenario]

    def by_category(self, category: ScenarioCategory) -> list[Scenario]:
        return [s for s in self.scenarios if s.category == category]

    def by_difficulty(self, difficulty: ScenarioDifficulty) -> list[Scenario]:
        return [s for s in self.scenarios if s.difficulty == difficulty]


def load_scenarios(path: str) -> ScenarioSuite:
    """Load scenarios from a YAML file."""
    from pathlib import Path
    import yaml

    with open(Path(path)) as f:
        raw = yaml.safe_load(f)

    scenarios = [Scenario(**s) for s in raw.get("scenarios", [])]
    return ScenarioSuite(
        name=raw.get("name", "unnamed"),
        description=raw.get("description", ""),
        scenarios=scenarios,
    )
