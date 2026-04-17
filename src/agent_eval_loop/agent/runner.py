"""Agent runner: execute a configured agent against messages via the Anthropic API."""

from __future__ import annotations

import json
import time
from typing import Any

import anthropic

from agent_eval_loop.agent.config import build_system_prompt
from agent_eval_loop.agent.scratchpad import Scratchpad
from agent_eval_loop.models import (
    AgentConfig,
    Message,
    MessageRole,
    ToolCall,
)


class AgentRunner:
    """Runs a configured agent in a multi-turn conversation.

    Fully synchronous — designed for batch simulation, not real-time serving.
    The runner manages system prompt assembly, conversation history,
    tool call execution, and scratchpad updates.
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_handlers: dict[str, Any] | None = None,
        client: anthropic.Anthropic | None = None,
    ):
        self.config = config
        self.client = client or anthropic.Anthropic()
        self.system_prompt = build_system_prompt(config)
        self.tool_handlers = tool_handlers or {}
        self.scratchpad = Scratchpad()
        self.conversation_history: list[Message] = []
        self.tool_calls: list[ToolCall] = []

    def register_tool(self, name: str, handler: Any) -> None:
        """Register a tool handler function."""
        self.tool_handlers[name] = handler

    def send_message(self, user_message: str) -> Message:
        """Send a user message and get the agent's response.

        Handles the full cycle: format history → call API → execute
        tool calls in a loop → return final text response.
        """
        self.conversation_history.append(
            Message(role=MessageRole.USER, content=user_message)
        )

        api_messages = self._format_messages()

        # Inject scratchpad into system prompt if populated
        system = self.system_prompt
        scratchpad_content = self.scratchpad.render()
        if scratchpad_content:
            system += f"\n\n<scratchpad>\n{scratchpad_content}\n</scratchpad>"

        # Build tool definitions for the API
        tools = self._build_tool_definitions() if self.tool_handlers else None

        # Build API kwargs (omit tools if None to avoid API errors)
        api_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": system,
            "messages": api_messages,
        }
        if tools:
            api_kwargs["tools"] = tools

        response = self.client.messages.create(**api_kwargs)

        # Process response — handle tool use in an agentic loop
        final_text = self._process_response(response, api_messages, system, tools)

        assistant_msg = Message(role=MessageRole.ASSISTANT, content=final_text)
        self.conversation_history.append(assistant_msg)
        return assistant_msg

    def _process_response(
        self,
        response: Any,
        messages: list[dict],
        system: str,
        tools: list[dict] | None,
    ) -> str:
        """Process API response, executing tool calls in a loop until text-only."""
        max_tool_rounds = 5

        for _ in range(max_tool_rounds):
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if not tool_use_blocks:
                text_blocks = [b for b in response.content if b.type == "text"]
                return " ".join(b.text for b in text_blocks)

            # Execute each tool call
            tool_results = []
            for tool_block in tool_use_blocks:
                result = self._execute_tool(tool_block.name, tool_block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps(result) if not isinstance(result, str) else result,
                })

            # Serialize assistant content blocks properly for re-submission
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_results})

            # Call API again with tool results
            api_kwargs: dict[str, Any] = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "system": system,
                "messages": messages,
            }
            if tools:
                api_kwargs["tools"] = tools

            response = self.client.messages.create(**api_kwargs)

        # Fallback after max rounds
        text_blocks = [b for b in response.content if b.type == "text"]
        return " ".join(b.text for b in text_blocks)

    def _execute_tool(self, tool_name: str, arguments: dict) -> Any:
        """Execute a tool and record the call."""
        start = time.time()
        error = None
        result = None

        try:
            handler = self.tool_handlers.get(tool_name)
            if handler is None:
                error = f"Unknown tool: {tool_name}"
                result = {"error": error}
            else:
                result = handler(**arguments)
        except Exception as e:
            error = str(e)
            result = {"error": error, "error_type": type(e).__name__}

        latency = (time.time() - start) * 1000

        self.tool_calls.append(ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            error=error,
            latency_ms=latency,
        ))

        return result

    def _format_messages(self) -> list[dict]:
        """Format conversation history for the Anthropic API."""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in self.conversation_history
            if msg.role in (MessageRole.USER, MessageRole.ASSISTANT)
        ]

    def _build_tool_definitions(self) -> list[dict]:
        """Build tool definitions from registered handlers."""
        tools = []
        for name, handler in self.tool_handlers.items():
            schema = getattr(handler, "tool_schema", None)
            if schema:
                tools.append(schema)
            else:
                tools.append({
                    "name": name,
                    "description": handler.__doc__ or f"Tool: {name}",
                    "input_schema": getattr(handler, "input_schema", {
                        "type": "object",
                        "properties": {},
                    }),
                })
        return tools

    def get_text_response(self) -> str:
        """Get the last assistant message text, or empty string."""
        for msg in reversed(self.conversation_history):
            if msg.role == MessageRole.ASSISTANT:
                return msg.content
        return ""

    def reset(self) -> None:
        """Reset conversation state for a new conversation."""
        self.conversation_history.clear()
        self.tool_calls.clear()
        self.scratchpad.clear()
