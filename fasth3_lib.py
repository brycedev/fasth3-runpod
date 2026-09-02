"""Shared FastH3 image + in-process T2VA generate (sm100a + compiled VAE).

The PyPI cu130 wheel is a 90a+100a+120a fatbin. CMake's 100a shorthand
emits a plain sm_100 image; B200 then launches that empty stub with
``invalid argument``. This image builds the in-tree kernel with
``TORCH_CUDA_ARCH_LIST=10.0a`` only.

Used by the production ``FastH3Worker`` in ``app.py`` and the bench sidecar
``fasth3.py``. Do not install ``fastvideo-kernel`` from PyPI on B200.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

FASTH3_MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
FASTVIDEO_ROOT = "/opt/fastvideo"
# Cache-bust: Blackwell-only in-tree sm100a cubin, not the PyPI fatbin.
FASTH3_IMAGE_REV = "fasth3-sm100a-native-100a-2026-09-02e"

# RunPod / local workers import this module without Modal. The image object is
# only built when Modal is installed (app.py, fasth3.py).
fasth3_image: Any = None


def build_fasth3_image():
    """Modal image: CUDA 13 + in-tree ``TORCH_CUDA_ARCH_LIST=10.0a`` kernel."""
    import modal

    return (
        modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
        .apt_install("git", "ffmpeg", "curl", "ninja-build", "g++", "cmake")
        .pip_install("uv")
        .run_commands(
        f"echo {FASTH3_IMAGE_REV}",
        "git clone --depth 1 --recurse-submodules --shallow-submodules "
        "https://github.com/hao-ai-lab/FastVideo.git "
        f"{FASTVIDEO_ROOT}",
        f"git -C {FASTVIDEO_ROOT} log -1 --oneline",
        f"test -f {FASTVIDEO_ROOT}/fastvideo-kernel/include/cutlass/include/cutlass/cutlass.h",
        f"cd {FASTVIDEO_ROOT} && uv pip install --system --prerelease=allow "
        "-e '.[fasth3]' hf_transfer",
        # Fail the image if the cubin still carries an empty sm_100 stub
        # (B200 reports 10.0 and will prefer that over sm_100a).
        r"""python3 - <<'PY'
import pathlib, re, subprocess, sys
sos = sorted(pathlib.Path("/usr/local/lib/python3.12/site-packages").glob("**/fastvideo_kernel*.so"))
print("kernel_sos", [str(p) for p in sos], flush=True)
if not sos:
    sys.exit("no fastvideo_kernel .so found")
dump = "/usr/local/cuda/bin/cuobjdump"
bad = False
saw_ops = False
for so in sos:
    out = subprocess.check_output([dump, "--list-elf", str(so)], text=True)
    print(f"=== cuobjdump {so.name} ===\n{out}", flush=True)
    arches = set(re.findall(r"sm_\d+[af]?", out))
    print(f"  arches={sorted(arches)}", flush=True)
    if "sm_100" in arches:
        print("FATAL: cubin contains sm_100; B200 will launch the empty stub", flush=True)
        bad = True
    if "fastvideo_kernel_ops" in so.name:
        saw_ops = True
        if "sm_100a" not in arches:
            print("FATAL: ops cubin missing sm_100a", flush=True)
            bad = True
if not saw_ops:
    sys.exit("fastvideo_kernel_ops .so not found")
if bad:
    sys.exit(2)
print("cubin ok: sm_100a only", flush=True)
PY""",
        env={
            "TORCH_CUDA_ARCH_LIST": "10.0a",
            "CUDA_HOME": "/usr/local/cuda",
            "CUDACXX": "/usr/local/cuda/bin/nvcc",
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/g++",
            "CUDAHOSTCXX": "/usr/bin/g++",
            "CMAKE_ARGS": (
                "-DFASTVIDEO_KERNEL_BUILD_TK=OFF "
                "-DFASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER=OFF"
            ),
            "UV_TORCH_BACKEND": "cu130",
        },
    )
    .env(
        {
            "HF_HOME": "/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "CUDA_HOME": "/usr/local/cuda",
            "PYTHONUNBUFFERED": "1",
            "TORCH_CUDA_ARCH_LIST": "10.0a",
            "FASTVIDEO_VSA_SM100A": "1",
        }
    )
    .add_local_python_source("fasth3_lib", "h3_route")
    )


try:
    fasth3_image = build_fasth3_image()
except ImportError:
    pass


def apply_fasth3_env() -> dict[str, str]:
    """Measured 1× B200 ``all`` profile (sm100a + regional compile + compiled VAE)."""
    env = {
        "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3",
        "FASTVIDEO_VSA_SM100A": "1",
        "FASTVIDEO_VSA_CUTEDSL": "0",
        "FASTVIDEO_DISABLE_ATTENTION_COMPILE": "0",
        "FASTVIDEO_FA4": "1",
        "FASTVIDEO_NVFP4_FA4": "0",
        "FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN": "0",
        "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",
        "FASTVIDEO_INFERENCE_TORCH_COMPILE": "1",
        "FASTVIDEO_VAE_PARALLEL_DECODE": "1",
        "FASTVIDEO_VAE_PARALLEL_ENCODE": "0",
        "FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY": "gather",
        "FASTVIDEO_ULYSSES_A2A": "off",
        "FASTVIDEO_STAGE_LOGGING": "1",
    }
    os.environ.pop("FASTVIDEO_H3_VSA_PROBE", None)
    os.environ.update(env)
    return env


def load_fasth3_generator(*, num_gpus: int = 1):
    """Load the distilled T2VA pipeline once per replica (keeps compile warm)."""
    from fastvideo import VideoGenerator
    from fastvideo.api import (
        CompileConfig,
        ComponentConfig,
        EngineConfig,
        GeneratorConfig,
        OffloadConfig,
        ParallelismConfig,
        PipelineSelection,
    )

    apply_fasth3_env()
    return VideoGenerator.from_config(
        GeneratorConfig(
            model_path=FASTH3_MODEL_ID,
            pipeline=PipelineSelection(
                components=ComponentConfig(),
                experimental={
                    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
                    "inference_torch_compile": True,
                    "vae_parallel_decode": True,
                    "vae_parallel_decode_strategy": "gather",
                    "video_decode_backend": "h3-vae",
                    "VSA_sparsity": 0.9,
                    "VSA_tile_size": 64,
                },
            ),
            engine=EngineConfig(
                num_gpus=num_gpus,
                execution_backend="mp",
                use_fsdp_inference=False,
                parallelism=ParallelismConfig(tp_size=1, sp_size=num_gpus),
                offload=OffloadConfig(
                    dit=False,
                    dit_layerwise=False,
                    text_encoder=True,
                    vae=True,
                    pin_cpu_memory=True,
                    lazy_module_load=False,
                ),
                compile=CompileConfig(
                    enabled=False,
                    vae_enabled=True,
                ),
            ),
        )
    )


def generate_fasth3_clip(
    generator,
    *,
    prompt: str,
    seed: int,
    num_frames: int,
    height: int,
    width: int,
    dest_mp4: str,
) -> dict:
    """One T2VA clip. Always generates audio (no mute / audio_copy)."""
    from fastvideo.api import GenerationRequest, OutputConfig, SamplingConfig

    dest = Path(dest_mp4)
    dest.parent.mkdir(parents=True, exist_ok=True)
    requested = dest.parent / f"_{dest.stem}_raw.mp4"
    t0 = time.perf_counter()
    result = generator.generate(
        GenerationRequest(
            prompt=prompt,
            negative_prompt="",
            sampling=SamplingConfig(
                height=height,
                width=width,
                num_frames=num_frames,
                fps=24,
                num_inference_steps=5,
                guidance_scale=1.0,
                batch_cfg=False,
                seed=int(seed),
            ),
            output=OutputConfig(
                output_path=str(requested),
                save_video=True,
                return_frames=False,
            ),
        )
    )
    wall = time.perf_counter() - t0
    src = Path(getattr(result, "video_path", None) or requested)
    if not src.exists():
        raise FileNotFoundError(f"FastH3 wrote no mp4 (looked for {src})")
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return {
        "wall_seconds": round(wall, 3),
        "generation_time": getattr(result, "generation_time", None),
        "bytes": dest.stat().st_size,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "seed": int(seed),
    }
