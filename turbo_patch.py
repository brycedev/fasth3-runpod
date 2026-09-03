"""Build-time fix for ComfyUI-MiniMax-H3-Turbo's AdaLN time-row prediction.

The turbo LoRA's pruned-base wrapper predicts unique-timestep rows from the
payload. Larryvrh ≥4274783 (PR #10/#16) mirrors image/audio refs via
``payload["refs"]`` / ``keyframes``, and silently skips the AdaLN delta when
row counts disagree — so MotCtx-pinned audio that lands as a ``layout.segments``
``ref_audio`` row no longer crashes, but also no longer gets the LoRA delta.

Fix: when ``payload["layout"].segments`` is present (Motion Context / similar),
mirror comfy/ldm/minimax/model.py's unique_t from those kinds; otherwise keep
upstream ``_unique_t(payload)``. Applied at image build; asserts if the anchor
drifts.
"""

path = "/comfyui/custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py"
src = open(path).read()

OLD = """        payload = kwargs.get("minimax_payload") or {}
        us = _unique_t(ts, shift_v, shift_a, payload)
        shared["silu_temb"] = _interp_egrid(us, E, ctx.device, ctx.dtype)
"""
NEW = """        payload = kwargs.get("minimax_payload") or {}
        payload = getattr(payload, "cond", payload) or {}
        layout = payload.get("layout") if isinstance(payload, dict) else None
        if getattr(layout, "segments", None) is not None:
            _sv = float((ts.flatten()[0] / 1000.0).clamp(min=1e-6))
            _tv = float(1.0 - _sv)
            _ta = float(1.0 - _time_shift_sigma(_sv, shift_v, shift_a))
            _s = {_tv, _ta}
            _kinds = {k for _, _, k in layout.segments}
            if _kinds & {"cond", "ref_img"}:
                _s.add(max(_tv, float(payload.get("visual_cond_noise_aug", 0.999))))
            if "ref_audio" in _kinds:
                _s.add(max(_ta, float(payload.get("audio_cond_noise_aug", 1.0))))
            us = sorted(_s)
        else:
            us = _unique_t(ts, shift_v, shift_a, payload)
        shared["silu_temb"] = _interp_egrid(us, E, ctx.device, ctx.dtype)
"""

assert OLD in src, "turbo adaln patch anchor not found — upstream changed, refusing to build"
open(path, "w").write(src.replace(OLD, NEW))
print("turbo adaln unique-t patch applied")
