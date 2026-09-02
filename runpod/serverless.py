#!/usr/bin/env python3
"""RunPod serverless handler for FastH3 T2VA.

Keep one warm replica with workers.min=1 (or flashboot + a long idleTimeout).
Scale to zero with workers.min=0 — next clip redoes load + compile, not the
148GB pull or the in-tree sm100a kernel build (those live on volume + image).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/h3")

from fasth3_lib import generate_fasth3_clip, load_fasth3_generator  # noqa: E402
from worker import DEFAULT_PROMPT, _gpu_name  # noqa: E402

OUTPUTS = Path(os.environ.get("H3_OUTPUTS", "/runpod-volume/outputs"))
os.environ.setdefault("HF_HOME", "/runpod-volume/hf")

_generator = None


def _gen():
    global _generator
    if _generator is None:
        print(f"h3: serverless load gpu={_gpu_name()}", flush=True)
        t0 = time.time()
        _generator = load_fasth3_generator()
        print(f"h3: serverless load {round(time.time() - t0, 2)}s", flush=True)
    return _generator


def handler(job: dict) -> dict:
    inp = job.get("input") or {}
    prompt = inp.get("prompt") or DEFAULT_PROMPT
    seed = int(inp.get("seed", 12345))
    num_frames = int(inp.get("num_frames", 124))
    height = int(inp.get("height", 768))
    width = int(inp.get("width", 1344))
    tag = str(inp.get("tag") or job.get("id") or "clip")
    dest = OUTPUTS / tag / "fasth3.mp4"
    t0 = time.time()
    info = generate_fasth3_clip(
        _gen(),
        prompt=prompt,
        seed=seed,
        num_frames=num_frames,
        height=height,
        width=width,
        dest_mp4=str(dest),
    )
    return {
        "ok": True,
        "mp4": str(dest),
        "wall_seconds": round(time.time() - t0, 3),
        "gpu": _gpu_name(),
        **info,
    }


if __name__ == "__main__":
    import runpod

    _gen()
    runpod.serverless.start({"handler": handler})
