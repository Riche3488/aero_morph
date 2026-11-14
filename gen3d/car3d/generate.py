"""TRELLIS-image-large (MIT) sampling. Handles the RTX 50xx (sm_120) workarounds:
  * spconv native algo (no JIT autotune)
  * xformers' auto dispatcher picks the Hopper-only FA3 kernel and crashes -> force FA2 / Cutlass
  * kaolin has no wheel for torch 2.7 -> ../shims/kaolin provides check_tensor (only FlexiCubes use)
  * no nvdiffrast (no nvcc) -> vertex colours from the decoded 3D Gaussians instead of a texture bake
"""
import os
import sys
import time

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PIPE = None


def _setup():
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    for p in (os.path.join(ROOT, "TRELLIS"), os.path.join(ROOT, "shims")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import xformers.ops as xops
    if getattr(xops.memory_efficient_attention, "_car3d_patched", False):
        return
    orig = xops.memory_efficient_attention

    def safe(q, k, v, attn_bias=None, *a, op=None, **kw):
        if op is None:
            try:
                return orig(q, k, v, attn_bias, *a, op=xops.MemoryEfficientAttentionFlashAttentionOp, **kw)
            except Exception:
                op = xops.MemoryEfficientAttentionCutlassOp
        return orig(q, k, v, attn_bias, *a, op=op, **kw)

    safe._car3d_patched = True
    xops.memory_efficient_attention = safe


def load_pipeline():
    global _PIPE
    if _PIPE is None:
        _setup()
        from trellis.pipelines import TrellisImageTo3DPipeline
        t0 = time.time()
        _PIPE = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        _PIPE.cuda()
        print(f"[generate] TRELLIS loaded in {time.time() - t0:.0f}s", flush=True)
    return _PIPE


def gaussian_vertex_colors(V, gs, k=8):
    """SH DC term -> RGB of the k nearest opaque Gaussians, opacity/distance weighted."""
    from scipy.spatial import cKDTree
    xyz = gs.get_xyz.detach().cpu().numpy()
    rgb = np.clip(0.28209479177387814 * gs._features_dc[:, 0].detach().cpu().numpy() + 0.5, 0, 1)
    op = gs.get_opacity.detach().cpu().numpy()[:, 0]
    sel = op > 0.1
    xyz, rgb, op = xyz[sel], rgb[sel], op[sel]
    d, idx = cKDTree(xyz).query(V, k=k)
    w = op[idx] / (d + 1e-4)
    col = (rgb[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
    return (col * 255).astype(np.uint8)


def generate(rgba_images, seed, gen):
    """rgba_images: list of PIL RGBA (background removed). Returns (raw trimesh, [cond images])."""
    import torch
    pipe = load_pipeline()
    conds = [pipe.preprocess_image(im) for im in rgba_images]
    kw = dict(seed=int(seed), formats=["mesh", "gaussian"],
              sparse_structure_sampler_params={"steps": gen["ss_steps"], "cfg_strength": gen["ss_cfg"]},
              slat_sampler_params={"steps": gen["slat_steps"], "cfg_strength": gen["slat_cfg"]})
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    if len(rgba_images) == 1:
        out = pipe.run(rgba_images[0], **kw)
    else:
        out = pipe.run_multi_image(rgba_images, mode=gen["mode"], **kw)
    mr = out["mesh"][0]
    V = mr.vertices.detach().cpu().numpy().astype(np.float64)
    F = mr.faces.detach().cpu().numpy()
    C = gaussian_vertex_colors(V, out["gaussian"][0])
    raw = trimesh.Trimesh(V, F, vertex_colors=C, process=True)
    print(f"[generate] seed {seed}: {len(raw.faces)} faces in {time.time() - t0:.1f}s, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    del out
    torch.cuda.empty_cache()
    return raw, conds
