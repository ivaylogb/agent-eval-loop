"""Conversation generator: simulate multi-turn conversations at scale.

The generator pairs personas with scenarios and runs them against
the agent under test. It uses the AgentRunner for agent turns (so tools
actually execute) and a separate LLM call for persona turns.

Architecture:
- The AgentRunner handles agent responses including tool calls
- A separate LLM instance plays the persona (the simulated customer)
- The generator orchestrates the turn-by-turn exchange
- Each conversation is recorded with full metadata for evaluation
"""

from __future__ import annotations

import uuid
from typing import Any

import anthropic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from agent_eval_loop.agent.runner import AgentRunner
from agent_eval_loop.models import (
    AgentConfig,
    Conversation,
    Message,
    MessageRole,
)
from agent_eval_loop.simulate.personas import Persona, get_all_personas
from agent_eval_loop.simulate.scenarios import Scenario, ScenarioSuite

console = Console()


class ConversationGenerator:
    """Generate simulated multi-turn conversations.

    For each (persona, scenario) pair, the generator:
    1. Creates an AgentRunner with the configured components and tools
    2. Sends the scenario's opening message through the runner
    3. Uses a separate LLM to generate persona responses
    4. Alternates turns until completion
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        tool_handlers: dict[str, Any] | None = None,
        simulator_model: str = "claude-sonnet-4-20250514",
    ):
        self.agent_config = agent_config
        self.tool_handlers = tool_handlers or {}
        self.simulator_model = simulator_model
        # Shared client for persona LLM calls (agent has its own via runner)
        self.client = anthropic.Anthropic()

    def generate_batch(
        self,
        scenarios: ScenarioSuite,
        personas: list[Persona] | None = None,
        max_conversations: int | None = None,
    ) -> list[Conversation]:
        """Generate conversations for all persona × scenario combinations."""
        if personas is None:
            personas = get_all_personas()

        pairs = [
            (persona, scenario)
            for persona in personas
            for scenario in scenarios.scenarios
        ]

        if max_conversations and len(pairs) > max_conversations:
            import random

            pairs = random.sample(pairs, max_conversations)

        conversations = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Generating {len(pairs)} conversations...",
                total=len(pairs),
            )
            for persona, scenario in pairs:
                try:
                    conv = self.generate_one(persona, scenario)
                    conversations.append(conv)
                except Exception as e:
                    console.print(
                        f"  [red]Failed: {persona.id} × {scenario.id}: {e}[/red]"
                    )
                progress.advance(task)

        return conversations

    def generate_one(self, persona: Persona, scenario: Scenario) -> Conversation:
        """Generate a single multi-turn conversation.

        Uses the AgentRunner for agent turns (tools execute for real)
        and a separate LLM for persona turns.
        """
        # Fresh runner per conversation — clean state
        runner = AgentRunner(
            config=self.agent_config,
            tool_handlers=self.tool_handlers,
            client=self.client,  # share the HTTP client
        )

        conversation = Conversation(
            id=str(uuid.uuid4()),
            persona_id=persona.id,
            scenario_id=scenario.id,
            agent_config=self.agent_config.name,
            metadata={
                "persona_name": persona.name,
                "scenario_name": scenario.name,
                "scenario_category": scenario.category.value,
                "scenario_difficulty": scenario.difficulty.value,
            },
        )

        # Persona message history (for the persona LLM)
        persona_messages: list[dict] = [
            {
                "role": "user",
                "content": self._build_persona_context(persona, scenario),
            },
            {"role": "assistant", "content": scenario.opening_message},
        ]

        # First user message comes from the scenario
        current_user_message = scenario.opening_message
        conversation.messages.append(
            Message(role=MessageRole.USER, content=current_user_message)
        )

        for turn in range(scenario.max_turns):
            # --- Agent turn (via runner — tools execute here) ---
            agent_msg = runner.send_message(current_user_message)
            agent_text = agent_msg.content
            conversation.messages.append(
                Message(role=MessageRole.ASSISTANT, content=agent_text)
            )

            # Capture any tool calls that happened
            conversation.tool_calls.extend(runner.tool_calls[len(conversation.tool_calls):])

            # --- Check if conversation should end ---
            if self._should_end_conversation(agent_text, turn, scenario.max_turns):
                break

            # --- Persona turn (separate LLM) ---
            persona_messages.append({
                "role": "user",
                "content": f"The support agent said:\n\n{agent_text}\n\nRespond in character.",
            })

            persona_response = self.client.messages.create(
                model=self.simulator_model,
                max_tokens=512,
                temperature=0.7,
                system=persona.to_system_prompt(),
                messages=persona_messages,
            )

            persona_text = self._extract_text(persona_response)
            persona_messages.append({"role": "assistant", "content": persona_text})

            if self._persona_is_done(persona_text):
                conversation.messages.append(
                    Message(role=MessageRole.USER, content=persona_text)
                )
                break

            current_user_message = persona_text
            conversation.messages.append(
                Message(role=MessageRole.USER, content=current_user_message)
            )

        return conversation

    def _build_persona_context(self, persona: Persona, scenario: Scenario) -> str:
        """Build initial context for the persona LLM."""
        return (
            f"You are about to contact customer support.\n"
            f"Situation: {scenario.description}\n"
            f"Your goal: {persona.goal}\n"
            f"Start the conversation with this message: {scenario.opening_message}\n"
            f"Then continue naturally based on the agent's responses."
        )

    def _should_end_conversation(
        self, agent_text: str, turn: int, max_turns: int
    ) -> bool:
        """Heuristic: should the conversation end after this agent turn?"""
        if turn >= max_turns - 1:
            return True

        end_signals = [
            "is there anything else i can help",
            "glad i could help",
            "have a great day",
            "anything else i can assist",
        ]
        lower = agent_text.lower()
        return any(signal in lower for signal in end_signals)

    def _persona_is_done(self, persona_text: str) -> bool:
        """Check if the persona signals the conversation is over."""
        done_signals = [
            "thank you, that's all",
            "that resolves my issue",
            "that's everything i needed",
            "no, that's all",
            "thanks, that's it",
        ]
        lower = persona_text.lower()
        return any(signal in lower for signal in done_signals)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text content from an API response."""
        text_blocks = [b for b in response.content if b.type == "text"]
        return " ".join(b.text for b in text_blocks)
