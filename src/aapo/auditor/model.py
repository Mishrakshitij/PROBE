"""Auditor policy: LLM wrapper for generating user messages and computing log-probs."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..types import Turn

AUDITOR_SYSTEM = """You are a realistic airline customer interacting with a flight booking assistant.
You have a specific travel need and are having a natural conversation.
Be specific about your requirements. Ask follow-up questions naturally.
You may be confused, change your mind, or ask about edge cases — just like a real customer."""


class AuditorPolicy:
    """Wraps a causal LM as the auditor (user-role) policy.

    Supports:
      - generate(): single message
      - generate_candidates(): G messages for shared-prefix sampling
      - log_prob(): log-probability of a text given history
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        system_prompt: str = AUDITOR_SYSTEM,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, history: list[Turn]) -> str:
        """Generate a single user message."""
        candidates = self.generate_candidates(history, n=1)
        return candidates[0] if candidates else ""

    def generate_candidates(self, history: list[Turn], n: int) -> list[str]:
        """Generate n candidate user messages from the same prefix."""
        messages = self._build_messages(history)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                num_return_sequences=n,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        candidates = []
        prompt_len = inputs["input_ids"].shape[1]
        for seq in outputs:
            text = self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
            candidates.append(text)

        return candidates

    def log_prob(self, history: list[Turn], text: str) -> float:
        """Compute log-probability of text given history (for GRPO/DPO)."""
        messages = self._build_messages(history)
        messages.append({"role": "user", "content": text})
        full = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Tokenize full and prefix separately
        prefix_messages = messages[:-1]
        prefix_str = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True)

        full_ids = self.tokenizer(full, return_tensors="pt")["input_ids"].to(self.device)
        prefix_ids = self.tokenizer(prefix_str, return_tensors="pt")["input_ids"].to(self.device)
        prefix_len = prefix_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits[0, prefix_len - 1:-1]  # shifted by 1
            target_ids = full_ids[0, prefix_len:]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_lps = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

        # Length-normalized log-prob
        total = token_lps.sum().item()
        n_tokens = max(token_lps.shape[0], 1)
        return total / n_tokens

    def _build_messages(self, history: list[Turn]) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for t in history:
            role = "user" if t.role.value == "user" else "assistant"
            messages.append({"role": role, "content": t.content})
        return messages

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
