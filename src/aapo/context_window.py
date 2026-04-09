"""Token-budget sliding window for multi-turn conversations.

Strategy: "pinned head + sliding tail"
  - Always keep: system prompt + first `anchor_turns` turn pairs
  - Fill remaining token budget with most recent turns (backward fill)
  - Insert [CONTEXT_TRUNCATED] marker when turns are omitted

The auditor (Qwen3-4B, 32K context) sees full history while the target
(vLLM, 12K context) sees a windowed view. This asymmetry is deliberate —
models real-world scenarios where an attacker reviews full conversation history.
"""

from __future__ import annotations

import json


def estimate_tokens(text: str) -> int:
    """Approximate token count: len(text) / 4, minimum 1."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(msg: dict) -> int:
    """Estimate token count for a single OpenAI-format message."""
    tokens = 4  # role + structural overhead

    content = msg.get("content")
    if content:
        tokens += estimate_tokens(content)

    # Tool call messages carry function name + arguments
    if "tool_calls" in msg:
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            tokens += estimate_tokens(fn.get("name", ""))
            args = fn.get("arguments", "")
            if isinstance(args, dict):
                args = json.dumps(args)
            tokens += estimate_tokens(args)

    return tokens


_TRUNCATION_MARKER = {
    "role": "system",
    "content": "[CONTEXT_TRUNCATED: earlier conversation turns omitted to fit context window]",
}


def windowed_messages(
    messages: list[dict],
    max_context_tokens: int,
    anchor_turns: int = 2,
    reserve_tokens: int = 512,
) -> list[dict]:
    """Apply sliding window to keep messages within token budget.

    Args:
        messages: full OpenAI-format message list (system + turns)
        max_context_tokens: total context budget (e.g. max_model_len)
        anchor_turns: number of turn *pairs* to always keep after system prompt
        reserve_tokens: tokens reserved for generation output

    Returns:
        Windowed message list. Returns input unchanged if everything fits.
    """
    if max_context_tokens <= 0:
        return messages

    budget = max_context_tokens - reserve_tokens

    # Check if everything fits
    total = sum(estimate_message_tokens(m) for m in messages)
    if total <= budget:
        return messages

    # Split: system prompt(s) | anchor messages | middle | tail
    # Find where system messages end
    sys_end = 0
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            sys_end = i + 1
        else:
            break

    sys_messages = messages[:sys_end]
    turn_messages = messages[sys_end:]

    # Anchor: first anchor_turns * 2 messages (user+assistant pairs)
    anchor_count = min(anchor_turns * 2, len(turn_messages))
    anchor_msgs = turn_messages[:anchor_count]
    tail_pool = turn_messages[anchor_count:]

    # Budget used by system + anchor + truncation marker
    sys_tokens = sum(estimate_message_tokens(m) for m in sys_messages)
    anchor_tokens = sum(estimate_message_tokens(m) for m in anchor_msgs)
    marker_tokens = estimate_message_tokens(_TRUNCATION_MARKER)
    used = sys_tokens + anchor_tokens + marker_tokens
    remaining = budget - used

    if remaining <= 0:
        # Even anchor doesn't fit — just return system + anchor (best effort)
        return sys_messages + anchor_msgs

    # Backward fill: take as many recent messages as fit
    tail_selected: list[dict] = []
    for msg in reversed(tail_pool):
        msg_tokens = estimate_message_tokens(msg)
        if msg_tokens > remaining:
            break
        tail_selected.append(msg)
        remaining -= msg_tokens

    tail_selected.reverse()

    # If we kept everything from tail_pool, no truncation happened
    if len(tail_selected) == len(tail_pool):
        return messages  # no-op

    # Build windowed output
    result = sys_messages + anchor_msgs
    if tail_pool and len(tail_selected) < len(tail_pool):
        result.append(_TRUNCATION_MARKER)
    result.extend(tail_selected)

    return result
