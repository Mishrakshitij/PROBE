"""SkyAssist target agent: LLM wrapper + tool dispatch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import litellm

from ..types import Role, ToolCall, Turn
from .tools import ToolRegistry

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
    """LLM-based target agent with tool calling."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ):
        self.tools = tool_registry
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or SKYASSIST_SYSTEM_PROMPT

    def generate(self, history: list[Turn]) -> Turn:
        """Generate a response (possibly with tool calls) given conversation history."""
        messages = self._build_messages(history)
        tool_calls_made: list[ToolCall] = []

        # Loop to handle multi-step tool use
        for _ in range(5):  # max 5 tool-call rounds per turn
            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=self.tools.get_schemas(),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            choice = response.choices[0]
            msg = choice.message

            # If no tool calls, return the text response
            if not msg.tool_calls:
                content = msg.content or ""
                turn_idx = len(history)
                return Turn(
                    role=Role.ASSISTANT,
                    content=content,
                    turn_index=turn_idx,
                    tool_calls=tuple(tool_calls_made),
                )

            # Execute each tool call
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                result = self.tools.execute(fn_name, fn_args)
                tool_calls_made.append(ToolCall(
                    tool_name=fn_name,
                    arguments=fn_args,
                    result=result,
                ))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

        # Fallback if max tool rounds exceeded
        return Turn(
            role=Role.ASSISTANT,
            content="I apologize, but I'm having trouble processing your request. Could you try again?",
            turn_index=len(history),
            tool_calls=tuple(tool_calls_made),
        )

    def _build_messages(self, history: list[Turn]) -> list[dict[str, Any]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in history:
            messages.append({
                "role": turn.role.value,
                "content": turn.content,
            })
        return messages
