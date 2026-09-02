#!/usr/bin/env python3
"""Start a RunPod CPU kaniko builder and wait until it pushes ghcr.io/brycedev/fasth3-runpod.

Uses job-scoped GHCR_TOKEN (GHA GITHUB_TOKEN). Never prints it. Not a B200.
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import sys
import time
import urllib.error
import urllib.request

API = "https://api.runpod.io/v2"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/128.0.0.0 Safari/537.36"
)
DEST = os.environ.get("DEST", "ghcr.io/brycedev/fasth3-runpod:sm100a")


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
    url = f"{API}/pods/{pod_id}/logs?source=container&tail=250"
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


def main() -> int:
    token = os.environ.get("GHCR_TOKEN")
    if not token:
        sys.exit("GHCR_TOKEN missing")
    start = (
        "set -euo pipefail; export DEBIAN_FRONTEND=noninteractive; "
        "apt-get update -qq; apt-get install -y -qq git ca-certificates curl buildah uidmap; "
        "if [ ! -d /opt/src/.git ]; then git clone --depth 1 https://github.com/brycedev/fasth3-runpod.git /opt/src; fi; "
        "bash /opt/src/runpod/buildah-build.sh"
    )
    payload = {
        "name": "fasth3-kaniko",
        "image": "ubuntu:24.04",
        "cloud": "SECURE",
        "cpu": {"id": "cpu3m", "vcpuCount": 8},
        "dataCenterIds": ["EU-RO-1"],
        "disk": 80,
        "env": {
            "GHCR_TOKEN": token,
            "GHCR_USER": os.environ.get("GHCR_USER", "brycedev"),
            "DEST": DEST,
            "CONTEXT_DIR": "/opt/src",
            "PYTHONUNBUFFERED": "1",
        },
        "args": "bash -lc " + shlex.quote(start),
    }
    code, body = req("POST", f"{API}/pods", payload)
    pod = body.get("pod") if isinstance(body, dict) and "pod" in body else body
    if code not in (200, 201) or not isinstance(pod, dict):
        print(json.dumps({"ok": False, "http": code, "body": body}, indent=2)[:4000])
        return 1
    pod_id = pod["id"]
    print(json.dumps({"ok": True, "pod_id": pod_id, "cost": pod.get("cost"), "dc": pod.get("dataCenterId")}), flush=True)
    deadline = time.time() + 3 * 3600
    last = ""
    try:
        while time.time() < deadline:
            st, pbody = req("GET", f"{API}/pods/{pod_id}")
            p = pbody.get("pod") if isinstance(pbody, dict) and "pod" in pbody else pbody
            status = (p or {}).get("status") if isinstance(p, dict) else None
            chunk = logs(pod_id)
            if chunk and chunk != last:
                delta = chunk[len(last) :] if chunk.startswith(last) else chunk
                sys.stdout.write(delta + "\n")
                sys.stdout.flush()
                last = chunk
            low = chunk.lower()
            if "pushed" in low and "ghcr.io/brycedev/fasth3-runpod" in low:
                print("BUILD_OK", flush=True)
                return 0
            if status in ("EXITED", "TERMINATED", "FAILED"):
                if "pushed" in last.lower():
                    print("BUILD_OK", flush=True)
                    return 0
                print(json.dumps({"ok": False, "status": status}), flush=True)
                return 1
            time.sleep(20)
        print(json.dumps({"ok": False, "timeout": True}), flush=True)
        return 1
    finally:
        req("DELETE", f"{API}/pods/{pod_id}")
        print("terminated", pod_id, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
