#!/usr/bin/env python3
"""Pull FastH3 weights onto the network volume (CPU / tiny GPU — never B200)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_ID = os.environ.get(
    "FASTH3_MODEL_ID",
    "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
)
HF_HOME = Path(os.environ.get("HF_HOME", "/runpod-volume/hf"))
READY = HF_HOME / ".fasth3_ready.json"


def _pip_install() -> None:
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--break-system-packages",
        "--no-cache-dir",
        "huggingface_hub",
        "hf_transfer",
    ]
    print("$", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _du(path: Path) -> str:
    proc = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True)
    return (proc.stdout or proc.stderr or "").strip()


def main() -> None:
    t0 = time.time()
    HF_HOME.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    print(f"HF_HOME={HF_HOME} model={MODEL_ID}", flush=True)
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        _pip_install()
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID)
    info = {
        "ok": True,
        "model": MODEL_ID,
        "path": path,
        "hf_home": str(HF_HOME),
        "du": _du(HF_HOME),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    READY.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)
    print("hydrate complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        fail = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "hf_home": str(HF_HOME),
        }
        try:
            HF_HOME.mkdir(parents=True, exist_ok=True)
            READY.write_text(json.dumps(fail, indent=2) + "\n")
        except Exception:
            pass
        print(json.dumps(fail), flush=True)
        raise
