#!/usr/bin/env python3
"""Start a RunPod CPU kaniko builder and wait until it pushes ghcr.io/brycedev/fasth3-runpod.

Uses REST v1 dockerEntrypoint so we can wrap gcr.io/kaniko-project/executor:debug
(kaniko has no standalone binary; buildah needs user namespaces RunPod denies).
Job-scoped GHCR_TOKEN (GHA GITHUB_TOKEN). Never prints it. Not a B200.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

API_V2 = "https://api.runpod.io/v2"
REST = "https://rest.runpod.io/v1"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36"
)
DEST = os.environ.get("DEST", "ghcr.io/brycedev/fasth3-runpod:sm100a")
KANIKO_IMAGE = "gcr.io/kaniko-project/executor:v1.24.0-debug"


def _key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY missing")
    return key


def req(method: str, url: str, data: dict | None = None):
    body = None if data is None else json.dumps(data).encode()
    r = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {_key()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": UA,
        },
    )
    try:
        with urllib.request.urlopen(r, timeout=90) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw.decode()) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"error": raw[:1500]}
        return exc.code, parsed


def logs(pod_id: str, seconds: float = 8.0) -> str:
    url = f"{API_V2}/pods/{pod_id}/logs?source=container&tail=250"
    r = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {_key()}",
            "Accept": "text/event-stream",
            "User-Agent": UA,
        },
    )
    lines: list[str] = []
    try:
        with urllib.request.urlopen(r, timeout=seconds) as resp:
            try:
                resp.fp.raw._sock.settimeout(seconds)
            except Exception:
                pass
            deadline = time.time() + seconds
            buf = b""
            while time.time() < deadline:
                try:
                    chunk = resp.read(4096)
                except (TimeoutError, socket.timeout):
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    text = raw.decode("utf-8", "replace").strip()
                    if text.startswith("data:"):
                        payload = text[5:].strip()
                        try:
                            evt = json.loads(payload)
                            line = evt.get("line")
                            if line:
                                lines.append(line)
                        except json.JSONDecodeError:
                            lines.append(payload)
    except (TimeoutError, socket.timeout, urllib.error.URLError):
        pass
    return "\n".join(lines)


def _redact(text: str) -> str:
    out = []
    for line in text.splitlines():
        if any(s in line for s in ("ghs_", "gho_", "ghu_", "GHCR_TOKEN", "rpa_")):
            continue
        out.append(line)
    return "\n".join(out)


def main() -> int:
    token = os.environ.get("GHCR_TOKEN")
    if not token:
        sys.exit("GHCR_TOKEN missing")
    # Busybox sh in the kaniko debug image. Write GHCR auth then exec the executor.
    # User namespaces are not required (unlike buildah on ubuntu).
    start = (
        "set -eu; "
        "mkdir -p /tmp /kaniko/.docker; "
        "rm -rf /kaniko/buildcontext /tmp/fasth3-runpod-main /tmp/src.tgz; "
        "AUTH=$(printf '%s' \"${GHCR_USER}:${GHCR_TOKEN}\" | /busybox/base64 | tr -d '\\n'); "
        "printf '{\"auths\":{\"ghcr.io\":{\"auth\":\"%s\"}}}\\n' \"$AUTH\" > /kaniko/.docker/config.json; "
        "unset AUTH; "
        "wget -O /tmp/src.tgz https://codeload.github.com/brycedev/fasth3-runpod/tar.gz/refs/heads/main; "
        "tar -xzf /tmp/src.tgz -C /tmp; "
        "exec /kaniko/executor "
        "--context=dir:///tmp/fasth3-runpod-main "
        "--dockerfile=runpod/Dockerfile.fasth3 "
        f"--destination={DEST} "
        "--destination=ghcr.io/brycedev/fasth3-runpod:latest "
        "--custom-platform=linux/amd64 "
        "--compressed-caching=false "
        "--snapshot-mode=redo "
        "--use-new-run "
        "--build-arg GITHUB_TOKEN=${GHCR_TOKEN} "
        "--verbosity=info"
    )
    payload = {
        "name": "fasth3-kaniko",
        "imageName": KANIKO_IMAGE,
        "cloudType": "SECURE",
        "computeType": "CPU",
        "cpuFlavorIds": ["cpu3m"],
        "vcpuCount": 8,
        "dataCenterIds": ["EU-RO-1"],
        "containerDiskInGb": 80,
        "env": {
            "GHCR_TOKEN": token,
            "GHCR_USER": os.environ.get("GHCR_USER", "brycedev"),
            "DEST": DEST,
        },
        "dockerEntrypoint": ["/busybox/sh", "-c"],
        "dockerStartCmd": [start],
    }
    code, body = req("POST", f"{REST}/pods", payload)
    pod = body if isinstance(body, dict) and body.get("id") else (
        body.get("pod") if isinstance(body, dict) else None
    )
    if code not in (200, 201) or not isinstance(pod, dict) or not pod.get("id"):
        print(json.dumps({"ok": False, "http": code, "body": body}, indent=2)[:4000])
        return 1
    pod_id = pod["id"]
    print(
        json.dumps(
            {
                "ok": True,
                "pod_id": pod_id,
                "cost": pod.get("costPerHr") or pod.get("cost"),
                "dc": (pod.get("machine") or {}).get("dataCenterId") or pod.get("dataCenterId"),
            }
        ),
        flush=True,
    )
    # GHA job timeout-minutes is 240; leave ~10 min for terminate + logs.
    deadline = time.time() + 230 * 60
    last = ""
    try:
        while time.time() < deadline:
            st, pbody = req("GET", f"{REST}/pods/{pod_id}")
            if st in (404, 410):
                print(json.dumps({"ok": False, "status": "GONE", "http": st}), flush=True)
                return 1
            p = pbody if isinstance(pbody, dict) else {}
            status = p.get("desiredStatus") or p.get("lastStatusChange") or p.get("status")
            chunk = _redact(logs(pod_id))
            if chunk and chunk != last:
                delta = chunk[len(last) :] if chunk.startswith(last) else chunk
                sys.stdout.write(delta + "\n")
                sys.stdout.flush()
                last = chunk
            low = chunk.lower()
            if "pushed" in low and "ghcr.io/brycedev/fasth3-runpod" in low:
                print("BUILD_OK", flush=True)
                return 0
            desired = str(p.get("desiredStatus") or "")
            if desired in ("EXITED", "TERMINATED") or str(status) in ("EXITED", "TERMINATED", "FAILED"):
                if "pushed" in last.lower():
                    print("BUILD_OK", flush=True)
                    return 0
                print(json.dumps({"ok": False, "status": desired or status}), flush=True)
                return 1
            time.sleep(20)
        print(json.dumps({"ok": False, "timeout": True}), flush=True)
        return 1
    finally:
        req("DELETE", f"{REST}/pods/{pod_id}")
        print("terminated", pod_id, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
