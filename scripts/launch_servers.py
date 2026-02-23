#!/usr/bin/env python3
"""Launch/stop vLLM inference servers based on models.yaml.

Profiles:
  training — target+judge on GPU 7 (shared server, port 8001)
  eval     — target on GPU 7 (port 8001), stronger judge on GPU 3+4 (port 8002)

Usage:
    python scripts/launch_servers.py --profile training
    python scripts/launch_servers.py --profile eval
    python scripts/launch_servers.py --stop
"""

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aapo.inference.model_loader import load_models_config
from aapo.inference.server import launch_vllm, wait_for_ready, shutdown_vllm


def main():
    parser = argparse.ArgumentParser(description="Manage vLLM servers")
    parser.add_argument("--profile", choices=["training", "eval"], default="training")
    parser.add_argument("--config", type=str, default="configs/models.yaml")
    parser.add_argument("--stop", action="store_true", help="Stop all managed servers")
    parser.add_argument("--wait-timeout", type=int, default=300)
    args = parser.parse_args()

    cfg = load_models_config(args.config)
    models = cfg.get("models", {})

    processes = []

    if args.profile == "training":
        # Single server for target + judge (same model, same port)
        target_cfg = models.get("target", {})
        if target_cfg.get("backend") == "vllm":
            print(f"Launching target/judge server: {target_cfg['name']} on GPU {target_cfg['gpu']}")
            proc = launch_vllm(
                model=target_cfg["name"],
                gpu=target_cfg["gpu"],
                port=target_cfg.get("port", 8001),
                quantization=target_cfg.get("quantization"),
                gpu_memory_utilization=target_cfg.get("gpu_memory_utilization", 0.45),
                max_model_len=target_cfg.get("max_model_len", 4096),
            )
            processes.append(("target/judge", proc, target_cfg.get("port", 8001)))

    elif args.profile == "eval":
        # Target server
        target_cfg = models.get("target", {})
        if target_cfg.get("backend") == "vllm":
            print(f"Launching target server: {target_cfg['name']} on GPU {target_cfg['gpu']}")
            proc = launch_vllm(
                model=target_cfg["name"],
                gpu=target_cfg["gpu"],
                port=target_cfg.get("port", 8001),
                quantization=target_cfg.get("quantization"),
                gpu_memory_utilization=target_cfg.get("gpu_memory_utilization", 0.45),
                max_model_len=target_cfg.get("max_model_len", 4096),
            )
            processes.append(("target", proc, target_cfg.get("port", 8001)))

        # Stronger judge server
        judge_cfg = models.get("judge_eval", {})
        if judge_cfg.get("backend") == "vllm":
            print(f"Launching eval judge server: {judge_cfg['name']} on GPU {judge_cfg['gpu']}")
            proc = launch_vllm(
                model=judge_cfg["name"],
                gpu=judge_cfg["gpu"],
                port=judge_cfg.get("port", 8002),
                quantization=judge_cfg.get("quantization"),
                gpu_memory_utilization=judge_cfg.get("gpu_memory_utilization", 0.90),
                max_model_len=judge_cfg.get("max_model_len", 2048),
            )
            processes.append(("judge_eval", proc, judge_cfg.get("port", 8002)))

    if not processes:
        print("No vLLM servers to launch for this profile.")
        return

    # Wait for servers to be ready
    for name, proc, port in processes:
        print(f"Waiting for {name} on port {port}...", end=" ", flush=True)
        if wait_for_ready(port, timeout=args.wait_timeout):
            print("ready.")
        else:
            print("TIMEOUT — check logs.")
            for _, p, _ in processes:
                shutdown_vllm(p)
            sys.exit(1)

    print(f"\nAll servers ready. Profile: {args.profile}")
    print("Press Ctrl+C to stop all servers.\n")

    # Keep running, shutdown on SIGINT/SIGTERM
    def _shutdown(sig, frame):
        print("\nShutting down servers...")
        for name, proc, _ in processes:
            print(f"  Stopping {name}...")
            shutdown_vllm(proc)
        print("Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Block until killed
    while True:
        for name, proc, _ in processes:
            if proc.poll() is not None:
                print(f"WARNING: {name} server exited unexpectedly (code {proc.returncode})")
        time.sleep(10)


if __name__ == "__main__":
    main()
