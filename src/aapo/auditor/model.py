"""Auditor policy: LLM wrapper for generating user messages and computing log-probs.

Supports:
  - Full-precision or QLoRA 4-bit loading
  - LoRA checkpoint save/load
  - Configurable device placement
"""

from __future__ import annotations

from pathlib import Path
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
      - QLoRA 4-bit quantization via bitsandbytes + peft
      - LoRA checkpoint save/load
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        system_prompt: str = AUDITOR_SYSTEM,
        quantize_4bit: bool = False,
        lora_config: dict[str, Any] | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self._has_lora = False

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model — optionally with 4-bit quantization
        if quantize_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map={"": self.device},
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
                trust_remote_code=True,
            ).to(self.device)

        # Apply LoRA if config provided
        if lora_config:
            self._apply_lora(lora_config)

        self.model.eval()

    def _apply_lora(self, lora_config: dict[str, Any]):
        """Apply LoRA adapter using peft."""
        from peft import LoraConfig, get_peft_model, TaskType

        config = LoraConfig(
            r=lora_config.get("rank", 64),
            lora_alpha=lora_config.get("alpha", 128),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            lora_dropout=lora_config.get("dropout", 0.05),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()
        self._has_lora = True

    def save_lora(self, path: str | Path):
        """Save only the LoRA adapter weights."""
        if not self._has_lora:
            raise RuntimeError("No LoRA adapter to save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

    def load_lora(self, path: str | Path):
        """Load LoRA adapter weights from a checkpoint."""
        from peft import PeftModel
        path = Path(path)
        self.model = PeftModel.from_pretrained(self.model, str(path))
        self._has_lora = True
        self.model.eval()

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

        prefix_messages = messages[:-1]
        prefix_str = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True)

        full_ids = self.tokenizer(full, return_tensors="pt")["input_ids"].to(self.device)
        prefix_ids = self.tokenizer(prefix_str, return_tensors="pt")["input_ids"].to(self.device)
        prefix_len = prefix_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(full_ids)
            logits = outputs.logits[0, prefix_len - 1:-1]
            target_ids = full_ids[0, prefix_len:]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_lps = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

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
