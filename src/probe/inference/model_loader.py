"""Generic model factory: reads models.yaml, dispatches to the right backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .vllm_client import VLLMClient, APIClient, InferenceBackend
from .openai_client import OpenAIClient


def load_models_config(path: str | Path = "configs/models.yaml") -> dict[str, Any]:
    """Load the models.yaml config."""
    p = Path(path)
    if not p.exists():
        return {}
    cfg = OmegaConf.load(p)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore


def load_inference_backend(
    role: str,
    config_path: str | Path = "configs/models.yaml",
    override_base_url: str | None = None,
) -> InferenceBackend:
    """Create an InferenceBackend for the given role (target, judge, judge_eval).

    Reads models.yaml and dispatches:
      - backend=vllm  → VLLMClient pointing at the configured port
      - backend=api   → APIClient (litellm fallback)

    Args:
        role: key in models.yaml (target, judge, judge_eval)
        config_path: path to models.yaml
        override_base_url: override the constructed base_url
    """
    cfg = load_models_config(config_path)
    models = cfg.get("models", {})
    model_cfg = models.get(role)
    if model_cfg is None:
        raise ValueError(f"No model config found for role '{role}' in {config_path}")

    backend = model_cfg.get("backend", "vllm")

    if backend == "vllm":
        import os
        port = int(os.environ.get("PROBE_SERVER_PORT", model_cfg.get("port", 8001)))
        base_url = override_base_url or f"http://localhost:{port}/v1"
        return VLLMClient(
            base_url=base_url,
            model_name=model_cfg["name"],
            temperature=model_cfg.get("temperature", 0.3),
            max_tokens=model_cfg.get("max_tokens", 1024),
        )
    elif backend == "api":
        return APIClient(
            model=model_cfg["name"],
            temperature=model_cfg.get("temperature", 0.3),
            max_tokens=model_cfg.get("max_tokens", 1024),
        )
    elif backend == "openai":
        import os
        api_key_env = model_cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)
        return OpenAIClient(
            model=model_cfg["name"],
            api_key=api_key,
            temperature=model_cfg.get("temperature", 0.3),
            max_tokens=model_cfg.get("max_tokens", 512),
        )
    else:
        raise ValueError(f"Unknown backend '{backend}' for role '{role}'")


def load_auditor_config(config_path: str | Path = "configs/models.yaml") -> dict[str, Any]:
    """Load auditor-specific config (model name, GPU, LoRA params).

    Returns a dict consumable by AuditorPolicy constructor.
    """
    cfg = load_models_config(config_path)
    return cfg.get("models", {}).get("auditor", {})
