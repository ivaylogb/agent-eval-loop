"""LLM-as-judge evaluators with structured rubrics.

Each judge evaluates one dimension of conversation quality. Judges return
structured verdicts with score, reasoning, evidence, and failure component
attribution.

Best practices for judge design:
- Ask for reasoning BEFORE the score. When the model explains its thinking
  first, the subsequent score is more calibrated. If you ask for the score
  first, the reasoning becomes post-hoc justification.
- Use binary pass/fail labels for clear signal. The 0-1 score provides
  granularity, but the pass_threshold makes the final call consistently.
- Calibrate judges against human annotations. A judge is only trustworthy
  if its verdicts correlate with human judgment (Cohen's kappa > 0.6).
- Include few-shot examples in rubrics for difficult categories. Examples
  anchor the judge's calibration far more than additional criteria text.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic
from pydantic import BaseModel

from agent_eval_loop.models import (
    ComponentType,
    Conversation,
    EvalCategory,
    JudgeVerdict,
)


# Define JudgeExample BEFORE JudgeRubric to avoid forward reference issues
class JudgeExample(BaseModel):
    """An example verdict for few-shot calibration."""

    conversation_snippet: str
    expected_score: float
    expected_reasoning: str


class JudgeRubric(BaseModel):
    """A structured rubric that guides an LLM judge's evaluation."""

    category: EvalCategory
    description: str
    criteria: list[str]
    pass_threshold: float = 0.7  # score >= this → pass
    examples: list[JudgeExample] | None = None


class Judge:
    """An LLM-as-judge evaluator for a specific eval category.

    The judge calls an LLM with a structured rubric, then applies
    the rubric's pass_threshold to the returned score to determine
    the boolean pass/fail verdict — so the threshold is enforced
    consistently rather than left to the LLM's interpretation.
    """

    def __init__(
        self,
        rubric: JudgeRubric,
        model: str = "claude-sonnet-4-20250514",
        client: anthropic.Anthropic | None = None,
    ):
        self.rubric = rubric
        self.model = model
        self.client = client or anthropic.Anthropic()

    def evaluate(self, conversation: Conversation) -> JudgeVerdict:
        """Evaluate a single conversation against this judge's rubric."""
        prompt = self._build_eval_prompt(conversation)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.0,
            system=self._system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )

        return self._parse_verdict(response)

    def _system_prompt(self) -> str:
        criteria_list = "\n".join(f"- {c}" for c in self.rubric.criteria)

        examples_section = ""
        if self.rubric.examples:
            example_parts = []
            for ex in self.rubric.examples:
                example_parts.append(
                    f"Example:\n"
                    f"  Conversation: {ex.conversation_snippet}\n"
                    f"  Expected score: {ex.expected_score}\n"
                    f"  Expected reasoning: {ex.expected_reasoning}"
                )
            examples_section = "\n\nCalibration examples:\n" + "\n\n".join(example_parts)

        return f"""You are an expert evaluator assessing AI agent conversation quality.

You are evaluating: {self.rubric.category.value}
Description: {self.rubric.description}

Evaluation criteria:
{criteria_list}
{examples_section}

Think step-by-step through each criterion, then produce your assessment.

Respond with a JSON object containing these fields IN THIS ORDER:
- "reasoning": string (2-3 sentences explaining your assessment — write this FIRST, before deciding the score)
- "evidence": list of strings (specific quotes from the conversation that support your reasoning)
- "failure_component": string or null (one of "instructions", "routines", "tools", "macros", "tools_usage", or null if no issue identified)
- "score": float between 0.0 and 1.0 (granular quality score — decide this LAST, after reasoning)

Important: formulate your reasoning and evidence before committing to a score.
Be strict but fair. Respond ONLY with the JSON object."""

    def _build_eval_prompt(self, conversation: Conversation) -> str:
        """Format the conversation for evaluation."""
        lines = [
            f"Conversation ID: {conversation.id}",
            f"Scenario: {conversation.metadata.get('scenario_name', 'unknown')}",
            f"Persona: {conversation.metadata.get('persona_name', 'unknown')}",
            "",
            "--- Conversation ---",
        ]

        for msg in conversation.messages:
            role_label = "Customer" if msg.role.value == "user" else "Agent"
            lines.append(f"{role_label}: {msg.content}")

        lines.append("--- End of Conversation ---")
        lines.append("")
        lines.append(f"Evaluate for: {self.rubric.category.value}")

        return "\n".join(lines)

    def _parse_verdict(self, response: Any) -> JudgeVerdict:
        """Parse the judge LLM response and apply the pass_threshold."""
        text = ""
        for block in response.content:
            if block.type == "text":
                text += block.text

        # Strip markdown code fences
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return JudgeVerdict(
                category=self.rubric.category,
                passed=False,
                score=0.0,
                reasoning=f"Judge response could not be parsed: {text[:200]}",
                failure_component=None,
            )

        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))  # clamp

        # Apply threshold — the rubric determines pass/fail, not the LLM
        passed = score >= self.rubric.pass_threshold

        fc = data.get("failure_component")
        failure_component = None
        if fc and fc in [ct.value for ct in ComponentType]:
            failure_component = ComponentType(fc)

        return JudgeVerdict(
            category=self.rubric.category,
            passed=passed,
            score=score,
            reasoning=data.get("reasoning", ""),
            evidence=data.get("evidence", []),
            failure_component=failure_component,
        )


# ---------------------------------------------------------------------------
# Standard judge rubrics — one per eval category
# ---------------------------------------------------------------------------

STANDARD_RUBRICS: dict[EvalCategory, JudgeRubric] = {
    EvalCategory.TOOL_SELECTION: JudgeRubric(
        category=EvalCategory.TOOL_SELECTION,
        description="Did the agent use the right tools at the right time?",
        criteria=[
            "Agent called the appropriate tool for the customer's request",
            "Agent did not call unnecessary tools",
            "Tool parameters were correct based on the conversation context",
            "Agent handled tool errors appropriately",
        ],
    ),
    EvalCategory.RESPONSE_ACCURACY: JudgeRubric(
        category=EvalCategory.RESPONSE_ACCURACY,
        description="Was the agent's response factually correct and complete?",
        criteria=[
            "Information provided to the customer was accurate",
            "Agent did not hallucinate or fabricate information",
            "Response addressed the customer's actual question",
            "No contradictions within the agent's responses",
        ],
    ),
    EvalCategory.TONE_APPROPRIATENESS: JudgeRubric(
        category=EvalCategory.TONE_APPROPRIATENESS,
        description="Was the agent's tone appropriate for the situation?",
        criteria=[
            "Tone matched the emotional context of the conversation",
            "Agent showed appropriate empathy when customer was frustrated",
            "Agent was professional without being robotic",
            "Agent did not escalate tension or be dismissive",
        ],
    ),
    EvalCategory.ROUTINE_ADHERENCE: JudgeRubric(
        category=EvalCategory.ROUTINE_ADHERENCE,
        description="Did the agent follow the prescribed routine/procedure?",
        criteria=[
            "Agent followed the steps in the correct order",
            "Agent did not skip required steps",
            "Agent gathered all required information before acting",
            "Agent provided required disclosures or confirmations",
        ],
    ),
    EvalCategory.ESCALATION_DECISION: JudgeRubric(
        category=EvalCategory.ESCALATION_DECISION,
        description="Did the agent correctly decide when to escalate to a human?",
        criteria=[
            "Agent escalated when the situation required human judgment",
            "Agent did not unnecessarily escalate simple requests",
            "Agent communicated the escalation clearly to the customer",
            "Agent provided context to the human agent during escalation",
        ],
    ),
    EvalCategory.COMPLETENESS: JudgeRubric(
        category=EvalCategory.COMPLETENESS,
        description="Did the agent fully resolve the customer's request?",
        criteria=[
            "The customer's core issue was addressed",
            "Agent provided all necessary information without the customer having to ask again",
            "Agent confirmed resolution before closing",
            "No loose ends or unanswered questions remained",
        ],
    ),
}


def get_standard_judges(
    categories: list[EvalCategory] | None = None,
    model: str = "claude-sonnet-4-20250514",
    client: anthropic.Anthropic | None = None,
) -> list[Judge]:
    """Get standard judges for the specified categories (or all).

    Pass a shared client to avoid creating N separate HTTP connections.
    """
    shared_client = client or anthropic.Anthropic()
    cats = categories or list(STANDARD_RUBRICS.keys())
    return [
        Judge(rubric=STANDARD_RUBRICS[cat], model=model, client=shared_client)
        for cat in cats
        if cat in STANDARD_RUBRICS
    ]
