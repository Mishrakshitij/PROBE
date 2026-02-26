"""OpenAI-compatible client wrapping a local vLLM server."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from openai import OpenAI

from ..types import ToolCall


class InferenceBackend(Protocol):
    """Generic interface for LLM inference — all consumers code to this."""

    def generate(self, messages: list[dict], **kwargs) -> str: ...

    def generate_with_tools(
        self, messages: list[dict], tools: list[dict], **kwargs
    ) -> tuple[str, list[ToolCall]]: ...


class VLLMClient:
    """Calls a local vLLM server via its OpenAI-compatible API.

    Works with any vLLM server exposing /v1/chat/completions.
    Supports function calling for tool-augmented agents.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model_name: str = "default",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        retries: int = 3,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self._client = OpenAI(base_url=base_url, api_key="not-needed", timeout=timeout)

    def generate(self, messages: list[dict], **kwargs) -> str:
        """Generate text completion from messages."""
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
        """Generate with function calling. Returns (text, tool_calls)."""
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
                parsed_calls = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        args = json.loads(tc.function.arguments)
                        parsed_calls.append(ToolCall(
                            tool_name=tc.function.name,
                            arguments=args,
                        ))
                return text, parsed_calls
            except Exception as e:
                if "max_tokens" in str(e) and max_tok > 128:
                    max_tok = max_tok // 2  # halve and retry
                    continue
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return "", []

    def health_check(self) -> bool:
        """Check if the server is responding."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False


class APIClient:
    """Fallback: calls external API via litellm (for backwards compat)."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3, max_tokens: int = 1024):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict], **kwargs) -> str:
        import litellm
        temp = kwargs.pop("temperature", self.temperature)
        max_tok = kwargs.pop("max_tokens", self.max_tokens)
        resp = litellm.completion(model=self.model, messages=messages,
                                  temperature=temp, max_tokens=max_tok, **kwargs)
        return resp.choices[0].message.content or ""

    def generate_with_tools(
        self, messages: list[dict], tools: list[dict], **kwargs
    ) -> tuple[str, list[ToolCall]]:
        import litellm
        temp = kwargs.pop("temperature", self.temperature)
        max_tok = kwargs.pop("max_tokens", self.max_tokens)
        resp = litellm.completion(model=self.model, messages=messages, tools=tools,
                                  temperature=temp, max_tokens=max_tok, **kwargs)
        msg = resp.choices[0].message
        text = msg.content or ""
        parsed = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                parsed.append(ToolCall(tool_name=tc.function.name, arguments=args))
        return text, parsed
