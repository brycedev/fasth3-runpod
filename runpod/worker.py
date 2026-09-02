#!/usr/bin/env python3
"""Keep-warm FastH3 HTTP worker (RunPod pod analogue of Modal FastH3Worker).

Boot loads the pipeline (and optionally compiles via a warmup clip). Later
POSTs /generate skip the 148GB pull and the in-tree kernel compile.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, "/opt/h3")

from fasth3_lib import (  # noqa: E402
    FASTH3_MODEL_ID,
    apply_fasth3_env,
    generate_fasth3_clip,
    load_fasth3_generator,
)

HOST = os.environ.get("H3_WORKER_HOST", "0.0.0.0")
PORT = int(os.environ.get("H3_WORKER_PORT", "8000"))
OUTPUTS = Path(os.environ.get("H3_OUTPUTS", "/runpod-volume/outputs"))
WARMUP = os.environ.get("H3_WARMUP", "1") != "0"
WARMUP_HEIGHT = int(os.environ.get("H3_WARMUP_HEIGHT", "768"))
WARMUP_WIDTH = int(os.environ.get("H3_WARMUP_WIDTH", "1344"))
WARMUP_FRAMES = int(os.environ.get("H3_WARMUP_FRAMES", "124"))

_lock = threading.Lock()
_state: dict = {
    "ready": False,
    "warming": True,
    "error": None,
    "boot_seconds": None,
    "warmup_seconds": None,
    "gpu": None,
    "clips": 0,
}
_generator = None

DEFAULT_PROMPT = (
    "CAMERA / LOOK: Handheld mini DV camcorder footage filmed by the subject herself. "
    "Slight hand shake, occasional focus hunting, imperfect framing, natural zoom "
    "adjustments, soft tape-like image quality, subtle grain, realistic auto-exposure "
    "shifts from moving daylight through a bus window. Natural skin tones, mild "
    "motion blur, authentic consumer camcorder aesthetic rather than polished cinematic "
    "footage. STYLE: Cozy commute vlog with gentle ASMR elements. Relaxed pacing, "
    "minimal dialogue, candid moments. Focus on satisfying everyday sounds: pencil "
    "scratching, small notebook pages flipping, seat fabric shifting, bus hum in "
    "the background. SUBJECT: Young woman in her early 20s, plain jacket, hair loose, "
    "minimal jewelry, no visible logos or branded items. Calm, relaxed energy during a "
    "short commute. SETTING: Small bus seat by the window on a bright afternoon. "
    "Natural daylight, blurred scenery passing outside, no visible route numbers, "
    "signage, or brand names in frame. STORYBOARD: Places camera on the seat beside "
    "her, opens a small notebook, sketches the blurred view outside the window. "
    "AUDIO NOTES: Natural ambience — pencil scratching, page flipping, soft bus hum. "
    "No speech or narration. overall_soundscape: bus interior, pencil on paper, "
    "muffled road noise. non_diegetic_music: N/A."
)


def _gpu_name() -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def boot() -> None:
    global _generator
    apply_fasth3_env()
    os.environ.setdefault("HF_HOME", "/runpod-volume/hf")
    print(
        f"h3: fasth3 worker boot model={FASTH3_MODEL_ID} "
        f"HF_HOME={os.environ.get('HF_HOME')} gpu={_gpu_name()}",
        flush=True,
    )
    t0 = time.time()
    _generator = load_fasth3_generator()
    boot_s = round(time.time() - t0, 2)
    warmup_s = None
    if WARMUP:
        dest = OUTPUTS / "_warmup" / "fasth3.mp4"
        print(
            f"h3: warmup {WARMUP_WIDTH}x{WARMUP_HEIGHT} {WARMUP_FRAMES}f "
            "(shape-specific torch compile; same as Modal first replica request)",
            flush=True,
        )
        tw = time.time()
        generate_fasth3_clip(
            _generator,
            prompt=DEFAULT_PROMPT,
            seed=1,
            num_frames=WARMUP_FRAMES,
            height=WARMUP_HEIGHT,
            width=WARMUP_WIDTH,
            dest_mp4=str(dest),
        )
        warmup_s = round(time.time() - tw, 2)
        print(f"h3: warmup finished in {warmup_s}s", flush=True)
    with _lock:
        _state["ready"] = True
        _state["warming"] = False
        _state["boot_seconds"] = boot_s
        _state["warmup_seconds"] = warmup_s
        _state["gpu"] = _gpu_name()
    print(f"h3: fasth3 boot finished in {boot_s}s ready=1", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] in ("/health", "/ready", "/"):
            with _lock:
                body = dict(_state)
            self._send(200 if body["ready"] else 503, body)
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/generate":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or "0")
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid json"})
            return
        with _lock:
            if not _state["ready"]:
                self._send(503, {"error": "warming", **dict(_state)})
                return
            gen = _generator
        prompt = req.get("prompt") or DEFAULT_PROMPT
        seed = int(req.get("seed", 12345))
        num_frames = int(req.get("num_frames", 124))
        height = int(req.get("height", 768))
        width = int(req.get("width", 1344))
        tag = str(req.get("tag") or "clip")
        dest = OUTPUTS / tag / "fasth3.mp4"
        t0 = time.time()
        try:
            info = generate_fasth3_clip(
                gen,
                prompt=prompt,
                seed=seed,
                num_frames=num_frames,
                height=height,
                width=width,
                dest_mp4=str(dest),
            )
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        wall = round(time.time() - t0, 3)
        with _lock:
            _state["clips"] = int(_state["clips"]) + 1
            clips = _state["clips"]
        self._send(
            200,
            {
                "ok": True,
                "mp4": str(dest),
                "clips": clips,
                "wall_seconds": wall,
                "gpu": _gpu_name(),
                **info,
            },
        )


def main() -> None:
    threading.Thread(target=boot, name="fasth3-boot", daemon=True).start()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"h3: listening on {HOST}:{PORT}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
