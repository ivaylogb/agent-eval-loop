"""Core data models shared across the framework.

This module defines the shared vocabulary for the entire system. Every other
module imports from here rather than defining its own types.

Hierarchy:
    AgentConfig (manifest) → ComponentVersion (pointer to a file + version)
    Conversation (multi-turn exchange) → Message + ToolCall
    ConversationEval (scores for one conversation) → JudgeVerdict (one category)
    FailurePattern (recurring issue) → ImprovementCandidate (proposed fix)
    LoopState → LoopIteration (one cycle of simulate → evaluate → improve)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Agent components
# ---------------------------------------------------------------------------


class ComponentType(str, Enum):
    """The types of independently versioned components that make up an agent.

    Production agents are assembled from these components at runtime.
    Each component is versioned separately, so a change to one routine
    doesn't require re-testing the entire prompt — just the changed piece.

    Best practice: treat every component as a prompt artifact. Tool
    descriptions, macros, and orchestration rules deserve the same
    rigor as the main instructions.
    """

    INSTRUCTIONS = "instructions"      # Role definition, constraints, guardrails
    ROUTINES = "routines"              # Step-by-step SOPs as agent-executable flows
    TOOLS = "tools"                    # Tool schemas, descriptions, usage guidance
    MACROS = "macros"                  # Pre-written templates for compliance-sensitive scenarios
    TOOLS_USAGE = "tools_usage"        # Orchestration rules for tool calling (e.g., "call these 3 tools in parallel on turn 1")


class ComponentVersion(BaseModel):
    """A pointer to a specific version of an agent component."""

    component_type: ComponentType
    path: str
    version: str
    content: str | None = None


class AgentConfig(BaseModel):
    """A manifest that defines an agent by pointing to specific component versions."""

    name: str
    description: str = ""
    components: dict[ComponentType, ComponentVersion]
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.0
    # Tool schemas parsed from the `tools` component YAML into Anthropic API
    # format. The tools component also stays in the system prompt as prose
    # (keeps "when NOT to use" guidance); this field is what goes on the wire.
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class Message(BaseModel):
    """A single message in a conversation."""

    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolCall(BaseModel):
    """A tool invocation by the agent."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any | None = None
    error: str | None = None
    latency_ms: float | None = None


class Conversation(BaseModel):
    """A complete multi-turn conversation between a persona and the agent."""

    id: str
    persona_id: str
    scenario_id: str
    agent_config: str
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvalCategory(str, Enum):
    """Standard eval categories. Extend for domain-specific needs."""

    TOOL_SELECTION = "tool_selection"
    RESPONSE_ACCURACY = "response_accuracy"
    TONE_APPROPRIATENESS = "tone_appropriateness"
    ROUTINE_ADHERENCE = "routine_adherence"
    ESCALATION_DECISION = "escalation_decision"
    COMPLETENESS = "completeness"


class JudgeVerdict(BaseModel):
    """A single judge's assessment of one conversation on one eval category."""

    category: EvalCategory
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence: list[str] = Field(default_factory=list)
    failure_component: ComponentType | None = None


class ConversationEval(BaseModel):
    """All eval results for a single conversation."""

    conversation_id: str
    verdicts: list[JudgeVerdict]
    aggregate_score: float = 0.0
    passed: bool = False

    def compute_aggregate(self) -> None:
        if self.verdicts:
            self.aggregate_score = sum(v.score for v in self.verdicts) / len(self.verdicts)
            self.passed = all(v.passed for v in self.verdicts)


# ---------------------------------------------------------------------------
# Improvement
# ---------------------------------------------------------------------------


class FailurePattern(BaseModel):
    """A recurring failure pattern identified across multiple conversations."""

    category: EvalCategory
    component: ComponentType
    description: str
    frequency: int
    example_conversation_ids: list[str]
    suggested_fix: str = ""


class ImprovementCandidate(BaseModel):
    """A proposed change to an agent component, including the revised content."""

    component: ComponentType
    original_version: str
    proposed_version: str
    proposed_content: str  # the actual revised component text — this is the change
    change_description: str
    target_failure_pattern: str
    eval_scores_before: dict[str, float] = Field(default_factory=dict)
    eval_scores_after: dict[str, float] = Field(default_factory=dict)
    regression_passed: bool = False


# ---------------------------------------------------------------------------
# Loop state
# ---------------------------------------------------------------------------


class LoopIteration(BaseModel):
    """State for a single iteration of the improvement loop."""

    iteration_number: int
    agent_config_name: str
    conversations_generated: int = 0
    eval_results: list[ConversationEval] = Field(default_factory=list)
    failure_patterns: list[FailurePattern] = Field(default_factory=list)
    candidates: list[ImprovementCandidate] = Field(default_factory=list)
    aggregate_score: float = 0.0
    converged: bool = False


class LoopState(BaseModel):
    """Full state of the improvement loop across iterations."""

    iterations: list[LoopIteration] = Field(default_factory=list)
    convergence_threshold: float = 0.02
    max_iterations: int = 5
    current_best_config: str = ""
