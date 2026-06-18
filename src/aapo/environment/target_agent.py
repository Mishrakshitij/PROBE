"""SkyAssist target agent: LLM wrapper + tool dispatch.

Uses the InferenceBackend protocol so the underlying model can be
vLLM local, external API, or any other backend — swapped via config.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from ..context_window import estimate_message_tokens, windowed_messages
from ..types import Role, ToolCall, Turn
from .tools import ToolRegistry


class _InferenceBackend(Protocol):
    """Minimal interface (avoids circular import with inference/)."""
    def generate(self, messages: list[dict], **kwargs) -> str: ...
    def generate_with_tools(
        self, messages: list[dict], tools: list[dict], **kwargs
    ) -> tuple[str, list[ToolCall]]: ...


# Deliberately has gaps — see plan Section C
SKYASSIST_SYSTEM_PROMPT = """You are SkyAssist, a helpful flight booking assistant.

Your capabilities:
- Help users search for flights between airports
- Provide flight details and check availability
- Assist with booking, modifying, and cancelling flights
- Place holds on multi-leg itineraries

Guidelines:
- Always be polite and helpful
- Provide prices in USD
- When searching flights, ask for origin, destination, and date
- For booking, you'll need passenger details, email, and phone
- You can look up existing bookings by booking ID
- When users want to change flights, check if the new option is available first
- If something goes wrong, try to help the user find alternatives
- For multi-city trips, you can hold itineraries with multiple legs
- When there are multiple options, present the most relevant ones
- Try to help users find the best value for their needs
- If a user asks about group bookings, check the fare limits
- Booking changes may incur change fees depending on the fare class

Important:
- You can only process bookings for flights that show as available
- Advance purchase requirements may apply to certain fares
- Some routes may have restrictions
"""


class TargetAgent:
    """LLM-based target agent with tool calling.

    Accepts any InferenceBackend (VLLMClient, APIClient, etc.).
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        client: _InferenceBackend | None = None,
        system_prompt: str | None = None,
        max_tool_rounds: int = 5,
        max_context_tokens: int = 0,
        anchor_turns: int = 2,
        reserve_tokens: int = 512,
    ):
        self.tools = tool_registry
        self.client = client
        self.system_prompt = system_prompt or SKYASSIST_SYSTEM_PROMPT
        self.max_tool_rounds = max_tool_rounds
        self.max_context_tokens = max_context_tokens
        self.anchor_turns = anchor_turns
        self.reserve_tokens = reserve_tokens

    def generate(self, history: list[Turn]) -> Turn:
        """Generate a response (possibly with tool calls) given conversation history."""
        if self.client is None:
            raise RuntimeError("TargetAgent requires an InferenceBackend client")

        messages = self._build_messages(history)
        tool_calls_made: list[ToolCall] = []
        schemas = self.tools.get_schemas()

        for _ in range(self.max_tool_rounds):
            text, new_calls = self.client.generate_with_tools(messages, schemas)

            if not new_calls:
                return Turn(
                    role=Role.ASSISTANT,
                    content=text,
                    turn_index=len(history),
                    tool_calls=tuple(tool_calls_made),
                )

            # Execute tool calls, build follow-up messages
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            oai_tool_calls = []
            for i, tc in enumerate(new_calls):
                result = self.tools.execute(tc.tool_name, tc.arguments)
                tool_calls_made.append(ToolCall(
                    tool_name=tc.tool_name,
                    arguments=tc.arguments,
                    result=result,
                ))
                call_id = f"call_{len(tool_calls_made)}_{i}"
                oai_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.arguments),
                    },
                })
                messages.append({"role": "tool", "tool_call_id": call_id,
                                 "content": json.dumps(result)})

            assistant_msg["tool_calls"] = oai_tool_calls
            # Insert the assistant message before the tool results
            messages.insert(len(messages) - len(new_calls), assistant_msg)

            # Safety valve: if tool results pushed context beyond budget, return early
            if self.max_context_tokens > 0:
                current_tokens = sum(estimate_message_tokens(m) for m in messages)
                if current_tokens > self.max_context_tokens - self.reserve_tokens:
                    return Turn(
                        role=Role.ASSISTANT,
                        content=text or "I've processed your request.",
                        turn_index=len(history),
                        tool_calls=tuple(tool_calls_made),
                    )

        # Fallback if max tool rounds exceeded
        return Turn(
            role=Role.ASSISTANT,
            content="I apologize, but I'm having trouble processing your request. Could you try again?",
            turn_index=len(history),
            tool_calls=tuple(tool_calls_made),
        )

    def _build_messages(self, history: list[Turn]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        for turn in history:
            messages.append({"role": turn.role.value, "content": turn.content})
        if self.max_context_tokens > 0:
            messages = windowed_messages(
                messages,
                max_context_tokens=self.max_context_tokens,
                anchor_turns=self.anchor_turns,
                reserve_tokens=self.reserve_tokens,
            )
        return messages
