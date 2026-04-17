"""Synthetic personas for conversation simulation.

Personas define who the simulated user is — their expertise level,
communication style, emotional state, and goal. The simulator uses
personas to generate realistic, diverse conversations that stress-test
the agent across the full spectrum of real user behavior.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExpertiseLevel(str, Enum):
    NOVICE = "novice"          # First-time user, unfamiliar with product
    INTERMEDIATE = "intermediate"  # Regular user, knows basics
    EXPERT = "expert"          # Power user, knows edge cases


class CommunicationStyle(str, Enum):
    CONCISE = "concise"        # Short messages, minimal detail
    VERBOSE = "verbose"        # Long messages, lots of context
    CONFUSED = "confused"      # Unclear, contradictory, needs guidance
    AGGRESSIVE = "aggressive"  # Frustrated, demanding, impatient
    POLITE = "polite"          # Courteous, patient


class EmotionalState(str, Enum):
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    CONFUSED = "confused"


class Persona(BaseModel):
    """A synthetic user persona for conversation simulation."""
    id: str
    name: str
    expertise: ExpertiseLevel
    style: CommunicationStyle
    emotion: EmotionalState
    goal: str  # What this persona is trying to accomplish
    background: str = ""  # Additional context for the persona
    constraints: list[str] = Field(default_factory=list)  # Things this persona won't do
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_system_prompt(self) -> str:
        """Generate a system prompt for an LLM playing this persona."""
        parts = [
            f"You are simulating a customer named {self.name}.",
            f"Expertise level: {self.expertise.value} — {self._expertise_description()}",
            f"Communication style: {self.style.value} — {self._style_description()}",
            f"Emotional state: {self.emotion.value}",
            f"Your goal: {self.goal}",
        ]

        if self.background:
            parts.append(f"Background: {self.background}")

        if self.constraints:
            constraints_str = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"Constraints (things you will NOT do):\n{constraints_str}")

        parts.extend([
            "",
            "Stay in character throughout the conversation.",
            "React naturally to the agent's responses — if they're helpful, acknowledge it.",
            "If they're unhelpful or wrong, push back as your persona would.",
            "Do NOT break character or mention that you're a simulation.",
        ])

        return "\n".join(parts)

    def _expertise_description(self) -> str:
        return {
            ExpertiseLevel.NOVICE: (
                "You are new to this product. You don't know the terminology"
                " and may describe things incorrectly."
            ),
            ExpertiseLevel.INTERMEDIATE: (
                "You use this product regularly and know the basics, but may"
                " not know advanced features."
            ),
            ExpertiseLevel.EXPERT: (
                "You are a power user. You know the product deeply and may"
                " reference specific features by name."
            ),
        }[self.expertise]

    def _style_description(self) -> str:
        return {
            CommunicationStyle.CONCISE: "Keep messages short. One or two sentences max.",
            CommunicationStyle.VERBOSE: "Provide lots of detail and context in your messages.",
            CommunicationStyle.CONFUSED: (
                "You're not sure what you need. Your messages may be unclear"
                " or contradictory."
            ),
            CommunicationStyle.AGGRESSIVE: (
                "You're frustrated and want this resolved immediately. You may"
                " be curt or demanding."
            ),
            CommunicationStyle.POLITE: (
                "You're patient and courteous, even if things aren't going well."
            ),
        }[self.style]


# ---------------------------------------------------------------------------
# Persona library — common persona archetypes
# ---------------------------------------------------------------------------

PERSONA_ARCHETYPES: list[Persona] = [
    Persona(
        id="happy_path_novice",
        name="Alex",
        expertise=ExpertiseLevel.NOVICE,
        style=CommunicationStyle.POLITE,
        emotion=EmotionalState.NEUTRAL,
        goal="Get basic help with a straightforward request",
        background="First-time customer, just signed up last week.",
    ),
    Persona(
        id="frustrated_intermediate",
        name="Jordan",
        expertise=ExpertiseLevel.INTERMEDIATE,
        style=CommunicationStyle.AGGRESSIVE,
        emotion=EmotionalState.FRUSTRATED,
        goal="Resolve a problem that has persisted across multiple interactions",
        background="Has contacted support twice before about this same issue.",
        constraints=["Will not repeat information already provided"],
    ),
    Persona(
        id="confused_novice",
        name="Sam",
        expertise=ExpertiseLevel.NOVICE,
        style=CommunicationStyle.CONFUSED,
        emotion=EmotionalState.CONFUSED,
        goal="Understand something about the product but can't articulate what",
        background="Not tech-savvy. May mix up product names or features.",
    ),
    Persona(
        id="expert_edge_case",
        name="Morgan",
        expertise=ExpertiseLevel.EXPERT,
        style=CommunicationStyle.CONCISE,
        emotion=EmotionalState.NEUTRAL,
        goal="Get help with an unusual or edge-case scenario",
        background="Long-time power user who knows the product better than most support agents.",
    ),
    Persona(
        id="anxious_high_stakes",
        name="Casey",
        expertise=ExpertiseLevel.INTERMEDIATE,
        style=CommunicationStyle.VERBOSE,
        emotion=EmotionalState.ANXIOUS,
        goal="Resolve an urgent, high-stakes issue",
        background="Something went wrong that has real consequences (financial, time-sensitive).",
    ),
]


def get_persona(persona_id: str) -> Persona:
    """Look up a persona by ID."""
    for p in PERSONA_ARCHETYPES:
        if p.id == persona_id:
            return p
    raise ValueError(f"Unknown persona: {persona_id}")


def get_all_personas() -> list[Persona]:
    """Return all available persona archetypes."""
    return PERSONA_ARCHETYPES.copy()
