#!/usr/bin/env bash
# Daemonless linux/amd64 build with buildah (vfs + chroot). CPU pod, never B200.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
DEST="${DEST:-ghcr.io/brycedev/fasth3-runpod:sm100a}"
CONTEXT_DIR="${CONTEXT_DIR:-/opt/src}"

if [[ ! -d "${CONTEXT_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/brycedev/fasth3-runpod.git "${CONTEXT_DIR}"
fi
cd "${CONTEXT_DIR}"

if [[ -z "${GHCR_TOKEN:-}" ]]; then
  echo "GHCR_TOKEN missing" >&2
  exit 2
fi

echo "buildah $(buildah --version || true)" >&2
buildah login --username "${GHCR_USER:-brycedev}" --password-stdin ghcr.io <<EOF
${GHCR_TOKEN}
EOF

# vfs works without privileged overlay; chroot isolation avoids needing runc/kvm.
export STORAGE_DRIVER=vfs
buildah bud \
  --isolation chroot \
  --storage-driver vfs \
  --jobs "$(nproc)" \
  -f runpod/Dockerfile.fasth3 \
  -t "${DEST}" \
  -t ghcr.io/brycedev/fasth3-runpod:latest \
  .

buildah push --storage-driver vfs "${DEST}"
buildah push --storage-driver vfs ghcr.io/brycedev/fasth3-runpod:latest
echo "Pushed ${DEST}" >&2
