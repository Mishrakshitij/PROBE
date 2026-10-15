"""OpenAI API client — mirrors VLLMClient interface for external model targets.

Used for Task 2 (transfer experiments): run the trained PROBE auditor
against GPT-4o-mini as the target instead of the local Qwen3-30B.

Set OPENAI_API_KEY env var before using, or pass api_key directly.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from openai import OpenAI

from ..types import ToolCall


class OpenAIClient:
    """Calls the OpenAI API using the same interface as VLLMClient.

    Supports function calling so it can be used as a drop-in target agent.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        timeout: float = 60.0,
        retries: int = 3,
    ):
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )
        self._client = OpenAI(api_key=key, timeout=timeout)

    def generate(self, messages: list[dict], **kwargs) -> str:
        temp = kwargs.pop("temperature", self.temperature)
        max_tok = kwargs.pop("max_tokens", self.max_tokens)

        for attempt in range(self.retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    **kwargs,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def generate_with_tools(
        self, messages: list[dict], tools: list[dict], **kwargs
    ) -> tuple[str, list[ToolCall]]:
        temp = kwargs.pop("temperature", self.temperature)
        max_tok = kwargs.pop("max_tokens", self.max_tokens)

        for attempt in range(self.retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    temperature=temp,
                    max_tokens=max_tok,
                    **kwargs,
                )
                msg = resp.choices[0].message
                text = msg.content or ""
                parsed: list[ToolCall] = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        args = json.loads(tc.function.arguments)
                        parsed.append(ToolCall(tool_name=tc.function.name, arguments=args))
                return text, parsed
            except Exception as e:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return "", []

    def health_check(self) -> bool:
        try:
            self._client.models.list()
            return True
        except Exception:
            return False
