#!/usr/bin/env python3
"""Fail the FastH3 image if the in-tree kernel still has an empty sm_100 stub."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys


def main() -> None:
    roots = (
        pathlib.Path("/usr/local/lib"),
        pathlib.Path("/usr/lib"),
    )
    sos: list[pathlib.Path] = []
    for root in roots:
        sos.extend(root.glob("**/fastvideo_kernel*.so"))
    sos = sorted(set(sos))
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


if __name__ == "__main__":
    main()
