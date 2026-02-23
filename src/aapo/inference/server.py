"""vLLM server lifecycle management: launch, wait, shutdown."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any


def launch_vllm(
    model: str,
    gpu: int | list[int],
    port: int = 8001,
    quantization: str | None = None,
    gpu_memory_utilization: float = 0.45,
    max_model_len: int = 4096,
    enable_prefix_caching: bool = True,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Launch a vLLM OpenAI-compatible server as a subprocess.

    Args:
        model: HuggingFace model ID
        gpu: GPU index or list for tensor parallel
        port: server port
        quantization: awq, gptq, bitsandbytes, or None
        gpu_memory_utilization: fraction of GPU memory to use
        max_model_len: max sequence length
        enable_prefix_caching: turn on vLLM prefix caching
        extra_args: additional CLI flags

    Returns:
        Popen process handle
    """
    if isinstance(gpu, list):
        cuda_visible = ",".join(str(g) for g in gpu)
        tp_size = len(gpu)
    else:
        cuda_visible = str(gpu)
        tp_size = 1

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--tensor-parallel-size", str(tp_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
    ]

    if quantization:
        cmd.extend(["--quantization", quantization])

    if enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return proc


def wait_for_ready(port: int, timeout: int = 300, interval: float = 5.0) -> bool:
    """Poll until the vLLM server responds on the given port.

    Returns True if ready, False if timeout.
    """
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(interval)

    return False


def shutdown_vllm(proc: subprocess.Popen, timeout: int = 30):
    """Gracefully stop a vLLM server process."""
    if proc.poll() is not None:
        return  # already exited

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
