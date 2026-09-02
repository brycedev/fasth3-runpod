# fasth3-runpod

Public build of the FastH3 RunPod worker image (`runpod/Dockerfile.fasth3` from the h3 tree).

- Image: `ghcr.io/brycedev/fasth3-runpod:sm100a`
- Kernel: in-tree `TORCH_CUDA_ARCH_LIST=10.0a` only (no PyPI cu130 wheel)
- Weights are **not** in this image; attach RunPod network volume `fasth3-hf-cache`
