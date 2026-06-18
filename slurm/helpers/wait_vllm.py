#!/usr/bin/env python3
"""Wait for vLLM server to become ready. Exit 0 on success, 1 on timeout."""
import sys
import time
import urllib.request
import urllib.error

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 600

url = f"http://localhost:{port}/v1/models"
start = time.time()
deadline = start + timeout

while time.time() < deadline:
    try:
        r = urllib.request.urlopen(url, timeout=5)
        if r.status == 200:
            elapsed = int(time.time() - start)
            print(f"[vLLM] Server ready on port {port} after {elapsed}s", flush=True)
            sys.exit(0)
    except Exception:
        pass
    time.sleep(5)

print(f"[vLLM] ERROR: Server not ready after {timeout}s", flush=True)
sys.exit(1)
