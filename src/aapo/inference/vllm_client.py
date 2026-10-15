"""OpenAI-compatible client wrapping a local vLLM server."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol

from openai import OpenAI

from ..types import ToolCall


def _parse_xml_tool_calls(text: str) -> list[ToolCall]:
    """Parse Qwen3 XML-style tool calls from content when SGLang doesn't structure them.

    Handles both formats:
      <tool_call>\n{"name": "fn", "arguments": {...}}\n</tool_call>
      <tool_call>\n<function=fn>\n<parameter=k>v</parameter>\n</function>\n</tool_call>
    """
    calls = []
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        block = block.strip()
        # JSON format: {"name": "fn", "arguments": {...}}
        try:
            obj = json.loads(block)
            if "name" in obj:
                calls.append(ToolCall(
                    tool_name=obj["name"],
                    arguments=obj.get("arguments") or obj.get("parameters") or {},
                ))
                continue
        except (json.JSONDecodeError, ValueError):
            pass
        # XML parameter format: <function=fn><parameter=k>v</parameter>...</function>
        fn_match = re.search(r"<function=([^>]+)>", block)
        if fn_match:
            fn_name = fn_match.group(1).strip()
            args = {}
            for pm in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", block, re.DOTALL):
                args[pm.group(1).strip()] = pm.group(2).strip()
            calls.append(ToolCall(tool_name=fn_name, arguments=args))
    return calls


def _strip_xml_tool_calls(text: str) -> str:
    """Remove <tool_call>...</tool_call> blocks from content."""
    return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()


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
        disable_thinking: bool = True,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self._client = OpenAI(base_url=base_url, api_key="not-needed", timeout=timeout)
        # Qwen3 models default to thinking mode; disable it for agent use cases.
        # SGLang respects chat_template_kwargs; standard vllm servers ignore it safely.
        self._extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else {}

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
                    extra_body=self._extra_body,
                    **kwargs,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                err_msg = str(e).lower()
                if ("context length" in err_msg or "input_tokens" in err_msg
                        or "input length" in err_msg or "maximum allowed length" in err_msg):
                    return ""  # context overflow — signal caller to stop
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
                    extra_body=self._extra_body,
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
                elif text:
                    # SGLang may return Qwen3 XML tool calls in content instead of
                    # structured tool_calls — parse them as a fallback.
                    parsed_calls = _parse_xml_tool_calls(text)
                    if parsed_calls:
                        text = _strip_xml_tool_calls(text)
                return text, parsed_calls
            except Exception as e:
                err_msg = str(e).lower()
                if ("context length" in err_msg or "input_tokens" in err_msg
                        or "input length" in err_msg or "maximum allowed length" in err_msg):
                    return "", []  # context overflow — signal caller to stop
                # Some targets (e.g. Llama via the llama3_json parser) only allow
                # one tool call per turn and reject the OpenAI default of parallel
                # tool calls. Retry with parallel calls disabled. No effect on
                # models that accept parallel calls (they never raise this).
                if ("single tool" in err_msg or "parallel" in err_msg) \
                        and kwargs.get("parallel_tool_calls") is not False:
                    kwargs["parallel_tool_calls"] = False
                    continue
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
