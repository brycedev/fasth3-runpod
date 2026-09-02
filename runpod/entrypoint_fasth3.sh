#!/usr/bin/env bash
set -euo pipefail

mkdir -p /run/sshd
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  mkdir -p /root/.ssh
  chmod 700 /root/.ssh
  printf '%s\n' "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd || true

# Pods and serverless both honor a custom mount path; default matches serverless.
export HF_HOME="${HF_HOME:-/runpod-volume/hf}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
mkdir -p "${HF_HOME}" /runpod-volume/outputs /outputs

MODE="${H3_MODE:-auto}"
if [[ "${MODE}" == "auto" ]]; then
  if [[ -n "${RUNPOD_ENDPOINT_ID:-}" ]]; then
    MODE=serverless
  elif [[ "${H3_HYDRATE:-0}" == "1" ]]; then
    MODE=hydrate
  else
    MODE=worker
  fi
fi

case "${MODE}" in
  hydrate)
    exec python3 -u /opt/h3/hydrate.py
    ;;
  serverless)
    exec python3 -u /opt/h3/serverless.py
    ;;
  worker)
    exec python3 -u /opt/h3/worker.py
    ;;
  *)
    echo "unknown H3_MODE=${MODE}" >&2
    exit 2
    ;;
esac
