#!/usr/bin/env bash
# Daemonless linux/amd64 image build. Runs on a RunPod CPU pod (not B200).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
DEST="${DEST:-ghcr.io/brycedev/fasth3-runpod:sm100a}"
CONTEXT_DIR="${CONTEXT_DIR:-/opt/src}"

apt-get update -qq
apt-get install -y -qq git ca-certificates curl python3 xz-utils >/tmp/apt.log

if [[ ! -d "${CONTEXT_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/brycedev/fasth3-runpod.git "${CONTEXT_DIR}"
fi

mkdir -p /kaniko/.docker /tmp/kaniko
curl -fsSL -o /tmp/kaniko/executor \
  https://github.com/GoogleContainerTools/kaniko/releases/download/v1.23.2/executor-linux-amd64
chmod +x /tmp/kaniko/executor

if [[ -z "${GHCR_TOKEN:-}" ]]; then
  echo "GHCR_TOKEN missing" >&2
  exit 2
fi
python3 - <<'PY'
import base64, json, os
from pathlib import Path
user = os.environ.get("GHCR_USER", "brycedev")
token = os.environ["GHCR_TOKEN"]
auth = base64.b64encode(f"{user}:{token}".encode()).decode()
Path("/kaniko/.docker/config.json").write_text(json.dumps({"auths": {"ghcr.io": {"auth": auth}}}))
print("wrote docker config for ghcr.io", flush=True)
PY

echo "starting kaniko -> ${DEST}" >&2
# compressed-caching off: less RAM. Snapshot the clone dir only.
exec /tmp/kaniko/executor \
  --dockerfile=runpod/Dockerfile.fasth3 \
  --context="dir://${CONTEXT_DIR}" \
  --destination="${DEST}" \
  --destination=ghcr.io/brycedev/fasth3-runpod:latest \
  --custom-platform=linux/amd64 \
  --compressed-caching=false \
  --snapshot-mode=redo \
  --use-new-run \
  --verbosity=info
