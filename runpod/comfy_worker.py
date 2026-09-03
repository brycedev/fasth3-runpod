#!/usr/bin/env python3
"""Keep-warm Comfy larryvrh turbo HTTP worker (RunPod analogue of ComfyWorker).

Assumes ComfyUI is already listening on 8188. Weights are on the volume.
Graph matches app.py ComfyWorker._build_graph for T2VA: v4 @ 8 + Sage + SLA.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = os.environ.get("H3_WORKER_HOST", "0.0.0.0")
PORT = int(os.environ.get("H3_WORKER_PORT", "8000"))
COMFY = os.environ.get("H3_COMFY", "http://127.0.0.1:8188")
OUTPUTS = Path(os.environ.get("H3_OUTPUTS", "/runpod-volume/outputs"))
WARMUP = os.environ.get("H3_WARMUP", "1") not in ("0", "false", "False")

DIT = os.environ.get("H3_COMFY_DIT", "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
CLIP = os.environ.get("H3_COMFY_CLIP", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA = "minimax_h3_turbo_v4_step600_ema.safetensors"

# Same prompt as out/ab_lxsla_pro6000_768p_5s / ab_sla.py.
PROMPT = (
    "CAMERA / LOOK: Handheld mini DV camcorder footage filmed by the subject herself. "
    "Slight hand shake, occasional focus hunting, imperfect framing, natural zoom adjustments, "
    "soft tape-like image quality, subtle grain, realistic auto-exposure shifts from moving "
    "daylight through a bus window. Natural skin tones, mild motion blur, authentic consumer "
    "camcorder aesthetic rather than polished cinematic footage.\n"
    "STYLE: Cozy commute vlog with gentle ASMR elements. Relaxed pacing, minimal dialogue, "
    "candid moments. Focus on satisfying everyday sounds: pencil scratching, small notebook "
    "pages flipping, seat fabric shifting, bus hum in the background.\n"
    "SUBJECT: Young woman in her early 20s, plain jacket, hair loose, minimal jewelry, no "
    "visible logos or branded items. Calm, relaxed energy during a short commute.\n"
    "SETTING: Small bus seat by the window on a bright afternoon. Natural daylight, blurred "
    "scenery passing outside, no visible route numbers, signage, or brand names in frame.\n"
    "STORYBOARD:\n"
    '→ (3s, propped medium shot) Places camera on the seat beside her, opens a small '
    'notebook. "Sketching a bit while I\'ve got time."\n'
    "→ (3s, close-up) Sketches the blurred view outside the window.\n"
    '→ (3s, handheld shot) Glances up at the passing scenery, adds a few more lines. '
    '"Trying to catch it before it\'s gone."\n'
    "→ (3s, detail shot) Shades a small section of the sketch. No dialogue.\n"
    '→ (3s, warm ending shot) Holds up the sketch, smiles at the camera. "See you at the '
    'next stop." Hand covers lens as recording ends.\n'
    "AUDIO NOTES: Natural ambience — pencil scratching, page flipping, soft bus hum should be "
    "clearly audible but subtle. Dialogue quiet and casual.\n"
    "REALISM NOTES: Authentic body language, natural blinking, genuine relaxed smile, "
    "imperfect framing, focus breathing, shifting window light. No copyrighted characters, "
    "logos, brand names, route signage, or recognizable public figures anywhere in frame. "
    "Fully original personal vlog content, not AI-generated or commercial in style."
)

_lock = threading.Lock()
_state: dict = {
    "ready": False,
    "warming": True,
    "error": None,
    "boot_seconds": None,
    "warmup_seconds": None,
    "gpu": None,
    "clips": 0,
    "last_mp4": None,
}
_last_mp4: Path | None = None


def _gpu_name() -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def _http_json(method: str, url: str, data: dict | None = None, timeout: float = 120):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return resp.status, json.loads(raw.decode()) if raw else {}


def _wait_comfy(timeout: float = 300) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            st, _ = _http_json("GET", f"{COMFY}/system_stats", timeout=5)
            if st == 200:
                return
        except Exception as exc:
            last = exc
        time.sleep(2)
    raise RuntimeError(f"ComfyUI not healthy: {last}")


def build_graph(*, prompt: str, seed: int, width: int, height: int, length: int, steps: int) -> dict:
    """T2VA larryvrh v4 @ 8 + Sage + SLA 0.90. Preview off (no TAE required)."""
    g = {
        "unet": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": DIT, "weight_dtype": "default"},
        },
        "clip": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"},
        },
        "vvae": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "cond": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["clip", 0],
                "vae": ["vvae", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
            },
        },
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "lora": {
            "class_type": "MiniMaxH3TurboLoRA",
            "inputs": {
                "model": ["unet", 0],
                "lora_name": TURBO_LORA,
                "strength": 1.0,
                "low_vram": False,
            },
        },
        "sage": {
            "class_type": "PathchSageAttentionKJ",
            "inputs": {
                "model": ["lora", 0],
                "sage_attention": "auto",
                "allow_compile": False,
            },
        },
        "sla": {
            "class_type": "H3SLAAttention",
            "inputs": {
                "model": ["sage", 0],
                "sparsity_ratio": 0.90,
                "block_size": "64",
                "min_seq_len": 8192,
                "dense_last_steps": 0,
                "protect_audio": True,
                "enabled": True,
            },
        },
        "guider": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["sla", 0], "conditioning": ["cond", 0]},
        },
        "sched": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["sla", 0],
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        },
        "sampler": {"class_type": "MiniMaxH3TurboSampler", "inputs": {}},
        "samp": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["noise", 0],
                "guider": ["guider", 0],
                "sampler": ["sampler", 0],
                "sigmas": ["sched", 0],
                "latent_image": ["cond", 1],
            },
        },
        "vdec": {"class_type": "VAEDecode", "inputs": {"samples": ["samp", 0], "vae": ["vvae", 0]}},
        "adec": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["samp", 0], "vae": ["avae", 0]},
        },
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["vdec", 0], "fps": 24.0}},
        "save": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["video", 0],
                "filename_prefix": "h3/out",
                "format": "auto",
                "codec": "auto",
            },
        },
        "asave": {
            "class_type": "SaveAudio",
            "inputs": {"audio": ["adec", 0], "filename_prefix": "h3/aud"},
        },
    }
    return g


def _view(ref: dict, dest: Path) -> None:
    q = urllib.parse.urlencode(
        {
            "filename": ref["filename"],
            "subfolder": ref.get("subfolder", ""),
            "type": ref.get("type", "output"),
        }
    )
    with urllib.request.urlopen(f"{COMFY}/view?{q}", timeout=120) as resp:
        dest.write_bytes(resp.read())


def run_clip(
    *,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    length: int,
    steps: int,
    dest: Path,
) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    graph = build_graph(
        prompt=prompt, seed=seed, width=width, height=height, length=length, steps=steps
    )
    t0 = time.time()
    st, queued = _http_json("POST", f"{COMFY}/prompt", {"prompt": graph, "client_id": dest.stem})
    if st >= 400 or "prompt_id" not in queued:
        raise RuntimeError(f"comfy rejected graph: {st} {queued}")
    pid = queued["prompt_id"]
    deadline = time.time() + 50 * 60
    entry = None
    while time.time() < deadline:
        try:
            _, hist = _http_json("GET", f"{COMFY}/history/{pid}", timeout=30)
        except Exception:
            time.sleep(2)
            continue
        entry = (hist or {}).get(pid)
        if entry and entry.get("status", {}).get("completed"):
            break
        if entry and entry.get("status", {}).get("status_str") == "error":
            raise RuntimeError(f"comfy error: {json.dumps(entry.get('status'))[:800]}")
        time.sleep(2)
    else:
        raise RuntimeError("comfy timed out")
    video_ref = audio_ref = None
    for out in (entry.get("outputs") or {}).values():
        for items in out.values() if isinstance(out, dict) else []:
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                name = str(it.get("filename", ""))
                if name.endswith(".mp4"):
                    video_ref = it
                elif name.endswith((".flac", ".wav", ".mp3", ".opus")):
                    audio_ref = it
    if video_ref is None:
        raise RuntimeError(f"no mp4 in comfy outputs: {str(entry.get('outputs'))[:500]}")
    tmp_v = Path(f"/tmp/{dest.stem}_v.mp4")
    _view(video_ref, tmp_v)
    if audio_ref:
        ext = os.path.splitext(audio_ref["filename"])[1]
        tmp_a = Path(f"/tmp/{dest.stem}_a{ext}")
        _view(audio_ref, tmp_a)
        mux = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(tmp_v), "-i", str(tmp_a),
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(dest),
            ],
            capture_output=True,
            text=True,
        )
        if mux.returncode != 0:
            raise RuntimeError(f"ffmpeg mux failed: {mux.stderr[-500:]}")
    else:
        shutil.copy2(tmp_v, dest)
    gen_s = round(time.time() - t0, 2)
    return {
        "generation_seconds": gen_s,
        "file_size": dest.stat().st_size,
        "mp4": str(dest),
        "prompt_id": pid,
        "seed": seed,
        "dit": DIT,
        "clip": CLIP,
        "lora": TURBO_LORA,
        "steps": steps,
        "sage": True,
        "sla": True,
        "sla_sparsity": 0.90,
        "width": width,
        "height": height,
        "length": length,
    }


def boot() -> None:
    t0 = time.time()
    try:
        _wait_comfy()
    except Exception as exc:
        with _lock:
            _state["warming"] = False
            _state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"h3: comfy wait failed: {exc}", flush=True)
        return
    boot_s = round(time.time() - t0, 2)
    warmup_s = None
    if WARMUP:
        dest = OUTPUTS / "_warmup" / "larry.mp4"
        print("h3: warmup 1344x768 124f discard (cold load)", flush=True)
        tw = time.time()
        try:
            run_clip(
                prompt=PROMPT,
                seed=111,
                width=1344,
                height=768,
                length=124,
                steps=8,
                dest=dest,
            )
            warmup_s = round(time.time() - tw, 2)
            print(f"h3: warmup finished in {warmup_s}s", flush=True)
        except Exception as exc:
            traceback.print_exc()
            with _lock:
                _state["warming"] = False
                _state["error"] = f"warmup: {type(exc).__name__}: {exc}"
                _state["boot_seconds"] = boot_s
                _state["gpu"] = _gpu_name()
            return
    with _lock:
        _state["ready"] = True
        _state["warming"] = False
        _state["boot_seconds"] = boot_s
        _state["warmup_seconds"] = warmup_s
        _state["gpu"] = _gpu_name()
    print(f"h3: larry worker ready boot={boot_s}s warmup={warmup_s}", flush=True)


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
        path = self.path.split("?", 1)[0]
        if path in ("/health", "/ready", "/"):
            with _lock:
                body = dict(_state)
            self._send(200 if body["ready"] else 503, body)
            return
        if path == "/clip":
            with _lock:
                mp4 = _state.get("last_mp4")
            if not mp4 or not Path(mp4).exists():
                self._send(404, {"error": "no clip"})
                return
            data = Path(mp4).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
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
        prompt = req.get("prompt") or PROMPT
        seed = int(req.get("seed", 424242))
        num_frames = int(req.get("num_frames", 124))
        height = int(req.get("height", 768))
        width = int(req.get("width", 1344))
        steps = int(req.get("steps", 8))
        tag = str(req.get("tag") or "clip")
        dest = OUTPUTS / tag / "larry.mp4"
        t0 = time.time()
        try:
            info = run_clip(
                prompt=prompt,
                seed=seed,
                width=width,
                height=height,
                length=num_frames,
                steps=steps,
                dest=dest,
            )
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        wall = round(time.time() - t0, 3)
        with _lock:
            _state["clips"] = int(_state["clips"]) + 1
            _state["last_mp4"] = str(dest)
            clips = _state["clips"]
        self._send(
            200,
            {
                "ok": True,
                "clips": clips,
                "wall_seconds": wall,
                "gpu": _gpu_name(),
                **info,
            },
        )


def main() -> None:
    threading.Thread(target=boot, name="larry-boot", daemon=True).start()
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"h3: larry worker listening on {HOST}:{PORT}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
