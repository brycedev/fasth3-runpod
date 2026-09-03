#!/usr/bin/env bash
set -euo pipefail

mkdir -p /run/sshd
if [[ -n "${PUBLIC_KEY:-}" && "${PUBLIC_KEY}" != "null" ]]; then
  mkdir -p /root/.ssh
  chmod 700 /root/.ssh
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd || true

MODELS="${H3_MODELS:-/runpod-volume/models}"
OUTPUTS="${H3_OUTPUTS:-/runpod-volume/outputs}"
mkdir -p "$MODELS" "$OUTPUTS" /comfyui/input /comfyui/output

cat > /comfyui/extra_model_paths.yaml <<YAML
h3:
  base_path: ${MODELS}
  diffusion_models: diffusion_models
  text_encoders: text_encoders
  vae: vae
  loras: loras
  checkpoints: checkpoints
  vae_approx: vae_approx
YAML

# Do not install Comfy or pull weights here. Image = nodes; volume = weights.
echo "h3: starting ComfyUI on 0.0.0.0:8188 models=${MODELS}" >&2
python3 /comfyui/main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch --disable-pinned-memory &
COMFY_PID=$!

deadline=$((SECONDS + 300))
until python3 - <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=3)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
do
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then
    echo "h3: ComfyUI exited before listen" >&2
    wait "$COMFY_PID" || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "h3: ComfyUI did not become healthy in 300s" >&2
    exit 1
  fi
  sleep 2
done
echo "h3: ComfyUI healthy on 8188" >&2

exec python3 /opt/h3/comfy_worker.py
