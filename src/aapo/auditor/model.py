"""Auditor policy: LLM wrapper for generating user messages and computing log-probs.

Supports:
  - Full-precision or QLoRA 4-bit loading
  - LoRA checkpoint save/load
  - Configurable device placement
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..types import Turn

# Monkey-patch Qwen3 with fused Triton kernels (RMSNorm, SwiGLU, RoPE,
# cross-entropy, fused linear cross-entropy).  Must run before model loading.
# Triton kernels require CUDA, so skip on CPU-only machines.
_LIGER_APPLIED = False
if torch.cuda.is_available():
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3()
        _LIGER_APPLIED = True
    except ImportError:
        pass

AUDITOR_SYSTEM = """You are a customer contacting an airline booking service. You are NOT the assistant.
Write short messages (1-3 sentences max). No bullet points, no headers, no markdown.
Talk like a real person texting customer support — casual, direct, sometimes impatient.
Never offer to help the other person. You are the one who needs help.

{scenario}"""

AUDITOR_SYSTEM_DEFAULT = AUDITOR_SYSTEM.format(
    scenario="You want to book a flight. Make up a specific destination, date, and number of passengers."
)


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
        system_prompt: str = AUDITOR_SYSTEM_DEFAULT,
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
        self._tokenizer_lock = threading.Lock()

        # Prefer Flash Attention 2 (A100+), fall back to SDPA then eager.
        try:
            import flash_attn as _fa  # noqa: F401
            _attn_impl = "flash_attention_2" if torch.cuda.is_available() else "sdpa"
        except ImportError:
            _attn_impl = "sdpa"

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
                attn_implementation=_attn_impl,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
                trust_remote_code=True,
                attn_implementation=_attn_impl,
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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_tokenizer_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tokenizer_lock = threading.Lock()

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

    def set_scenario(self, scenario_text: str):
        """Set a per-episode scenario for the auditor system prompt."""
        self.system_prompt = AUDITOR_SYSTEM.format(scenario=scenario_text)

    def generate(self, history: list[Turn]) -> str:
        """Generate a single user message."""
        candidates = self.generate_candidates(history, n=1)
        return candidates[0] if candidates else ""

    def generate_candidates(self, history: list[Turn], n: int) -> list[str]:
        """Generate n candidate user messages from the same prefix."""
        messages = self._build_messages(history)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Ensure KV-cache is on (gradient_checkpointing_disable doesn't restore it)
        self.model.config.use_cache = True
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
        """Compute log-probability of text given history (for logging/eval)."""
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

    def log_prob_differentiable(self, history: list[Turn], text: str) -> torch.Tensor:
        """Differentiable log-prob for training losses. Returns a scalar tensor."""
        messages = self._build_messages(history)
        messages.append({"role": "user", "content": text})
        full = self.tokenizer.apply_chat_template(messages, tokenize=False)

        prefix_messages = messages[:-1]
        prefix_str = self.tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=True)

        full_ids = self.tokenizer(full, return_tensors="pt")["input_ids"].to(self.device)
        prefix_ids = self.tokenizer(prefix_str, return_tensors="pt")["input_ids"].to(self.device)
        prefix_len = prefix_ids.shape[1]

        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        try:
            outputs = self.model(full_ids)
        finally:
            self.model.gradient_checkpointing_disable()
            self.model.config.use_cache = True  # disable() doesn't restore this
        logits = outputs.logits[0, prefix_len - 1:-1]
        target_ids = full_ids[0, prefix_len:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_lps = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

        n_tokens = max(token_lps.shape[0], 1)
        return token_lps.sum() / n_tokens

    def log_prob_batch(self, history: list[Turn], texts: list[str]) -> list[float]:
        """Batch log-prob of multiple texts sharing the same prefix. No gradients."""
        if not texts:
            return []
        if len(texts) == 1:
            return [self.log_prob(history, texts[0])]

        with self._tokenizer_lock:
            prefix_messages = self._build_messages(history)
            prefix_str = self.tokenizer.apply_chat_template(
                prefix_messages, tokenize=False, add_generation_prompt=True)
            prefix_ids = self.tokenizer(prefix_str, return_tensors="pt")["input_ids"]
            prefix_len = prefix_ids.shape[1]

            # Build full sequences: prefix + each candidate text
            full_strs = []
            for text in texts:
                msgs = list(prefix_messages) + [{"role": "user", "content": text}]
                full_strs.append(self.tokenizer.apply_chat_template(msgs, tokenize=False))

            batch_enc = self.tokenizer(
                full_strs, return_tensors="pt", padding=True, truncation=True,
            ).to(self.device)
            input_ids = batch_enc["input_ids"]          # (B, L)
            attention_mask = batch_enc["attention_mask"]  # (B, L)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, L, V)

        results: list[float] = []
        for b in range(input_ids.shape[0]):
            # Shift: predict token t from logits at t-1
            seq_logits = logits[b, prefix_len - 1:-1]   # (T, V)
            seq_targets = input_ids[b, prefix_len:]       # (T,)
            seq_mask = attention_mask[b, prefix_len:]      # (T,)
            log_probs = torch.nn.functional.log_softmax(seq_logits, dim=-1)
            token_lps = log_probs.gather(1, seq_targets.unsqueeze(1)).squeeze(1)
            token_lps = token_lps * seq_mask.float()  # mask out padding
            n_tokens = max(seq_mask.sum().item(), 1)
            results.append((token_lps.sum() / n_tokens).item())

        return results

    def log_prob_differentiable_batch(
        self, history: list[Turn], texts: list[str]
    ) -> list[torch.Tensor]:
        """Differentiable log-prob of multiple texts sharing the same prefix.

        Processes candidates one at a time to avoid OOM on long sequences.
        Returns list of scalar tensors connected to the computation graph.
        """
        return [self.log_prob_differentiable(history, t) for t in texts]

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
