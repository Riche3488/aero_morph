#!/usr/bin/env python3
"""
IONIQ 5 image -> 3D mesh with Microsoft TRELLIS (MIT licence).

(Hunyuan3D 2.0 was the original request, but its licence excludes South Korea /
EU / UK, so TRELLIS-image-large is used as a drop-in image-to-3D replacement.)

Steps
  1. background removal (rembg / BiRefNet)            -> inputs/<name>_rgba.png
  2. TRELLIS sparse-structure + SLAT flow sampling     (seeded)
  3. FlexiCubes mesh decode (vertex colours)           -> raw mesh
  4. cleanup: largest component, decimation, metric scaling / alignment to the
     aero_project frame (x rearward from front bumper, y left, z up, ground z=0)
     -> output/<tag>.glb / .obj / .stl, plus a raw .ply

Usage
  .venv/bin/python run_trellis.py inputs/candidates/Hyundai_Ioniq_5_1X7A0847.jpg [--seed 1] [--tag ioniq5_trellis]
  multiple images -> TRELLIS multi-image conditioning
"""
import argparse
import os
import sys
import time

os.environ.setdefault("SPCONV_ALGO", "native")      # no JIT autotune (sm_120 safe)
os.environ.setdefault("ATTN_BACKEND", "xformers")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "TRELLIS"))
sys.path.insert(0, os.path.join(HERE, "shims"))      # kaolin.utils.testing stub

import numpy as np
import torch
import trimesh
from PIL import Image

# --- xformers on Blackwell (sm_120): the automatic dispatcher picks the Hopper-only FA3
# kernel and crashes ("flash_fwd_launch_template.h: invalid argument").  Force FA2 / Cutlass.
import xformers.ops as xops
_mea = xops.memory_efficient_attention


def _mea_safe(q, k, v, attn_bias=None, *a, op=None, **kw):
    if op is None:
        try:
            return _mea(q, k, v, attn_bias, *a, op=xops.MemoryEfficientAttentionFlashAttentionOp, **kw)
        except Exception:
            op = xops.MemoryEfficientAttentionCutlassOp
    return _mea(q, k, v, attn_bias, *a, op=op, **kw)


xops.memory_efficient_attention = _mea_safe

# IONIQ 5 public dimensions (mm) – same basis as ../ioniq5_geometry.py
L_MM, W_MM, H_MM = 4635.0, 1890.0, 1605.0


def gaussian_vertex_colors(V, gs, k=8):
    """Vertex colours from the decoded 3D Gaussians (SH DC term -> RGB), opacity- and
    distance-weighted over the k nearest Gaussians. Replaces TRELLIS' nvdiffrast texture bake."""
    from scipy.spatial import cKDTree
    xyz = gs.get_xyz.detach().cpu().numpy()
    rgb = np.clip(0.28209479177387814 * gs._features_dc[:, 0].detach().cpu().numpy() + 0.5, 0, 1)
    op = gs.get_opacity.detach().cpu().numpy()[:, 0]
    sel = op > 0.1
    xyz, rgb, op = xyz[sel], rgb[sel], op[sel]
    d, idx = cKDTree(xyz).query(V, k=k)
    w = op[idx] / (d + 1e-4)
    col = (rgb[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
    print(f"[colour] {sel.sum()} gaussians, mean vertex rgb {col.mean(0).round(3)}")
    return (col * 255).astype(np.uint8)


def remove_bg(path, out_dir):
    import rembg
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(out_dir, f"{name}_rgba.png")
    if os.path.exists(out):
        return Image.open(out)
    img = Image.open(path).convert("RGB")
    s = min(1.0, 2048 / max(img.size))
    if s < 1:
        img = img.resize((int(img.width * s), int(img.height * s)), Image.Resampling.LANCZOS)
    sess = rembg.new_session("birefnet-general")
    rgba = rembg.remove(img, session=sess)
    # keep only the largest alpha blob (drops neighbouring cars / people)
    import cv2
    a = np.array(rgba)[:, :, 3]
    n, lab, stats, _ = cv2.connectedComponentsWithStats((a > 128).astype(np.uint8), 8)
    if n > 2:
        keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        arr = np.array(rgba)
        arr[:, :, 3] = np.where(lab == keep, a, 0)
        rgba = Image.fromarray(arr)
    rgba.save(out)
    return rgba


def align_to_car_frame(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """TRELLIS meshes are z-up with arbitrary heading: PCA in the ground plane gives the
    length axis; the front is the end with the lower top (hood vs. tall tailgate)."""
    m = m.copy()
    xy = m.vertices[:, :2] - m.vertices[:, :2].mean(0)
    w, vecs = np.linalg.eigh(np.cov(xy.T))
    ax = vecs[:, np.argmax(w)]
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.arctan2(ax[1], ax[0]), [0, 0, 1]))
    V = m.vertices
    x0, x1 = V[:, 0].min(), V[:, 0].max(); Lx = x1 - x0
    top_lo = V[V[:, 0] < x0 + 0.15 * Lx, 2].max()
    top_hi = V[V[:, 0] > x1 - 0.15 * Lx, 2].max()
    if top_lo > top_hi:   # tall end at -x -> rotate so the front bumper sits at x min
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))
    lo, hi = m.bounds
    m.apply_translation([-lo[0], -(lo[1] + hi[1]) / 2, -lo[2]])
    return m


def fit_to_spec(m: trimesh.Trimesh, anisotropic: bool) -> trimesh.Trimesh:
    """Scale to metres. Uniform: match length. Anisotropic: also match height and the
    body width (measured over the rear half, i.e. without mirrors) to the public spec,
    which corrects the depth ambiguity of single-/few-view reconstruction."""
    m = m.copy()
    sx = L_MM / 1000.0 / m.extents[0]
    if not anisotropic:
        m.apply_scale(sx); return m
    V = m.vertices
    x0, Lx = V[:, 0].min(), np.ptp(V[:, 0])
    rear = (V[:, 0] > x0 + 0.55 * Lx) & (V[:, 0] < x0 + 0.85 * Lx)
    w_body = np.ptp(V[rear, 1])
    sy = W_MM / 1000.0 / w_body
    sz = H_MM / 1000.0 / np.percentile(V[:, 2], 99.95)      # ignore antenna tip
    m.apply_transform(np.diag([sx, sy, sz, 1.0]))
    return m


def remove_underbody_slivers(m: trimesh.Trimesh, clearance=0.13, wheel_dx=0.45) -> trimesh.Trimesh:
    """Mesh in metres, x from front bumper. The ground is the lowest point of the tyres; any
    geometry outside the wheel zones lower than `clearance` above it is an artefact
    (ground/shadow slivers).  Re-grounds the mesh on the tyres."""
    V, F = m.vertices, m.faces
    axles = np.array([845.0, 3845.0]) / 1000.0
    vw = np.min(np.abs(V[:, :1] - axles[None]), axis=1) < 0.30
    z_ground = np.percentile(V[vw, 2], 0.2)
    c = V[F].mean(1)
    near_wheel = np.min(np.abs(c[:, :1] - axles[None]), axis=1) < wheel_dx
    keep = ~(((c[:, 2] - z_ground) < clearance) & ~near_wheel) & (c[:, 2] > z_ground - 0.01)
    m2 = m.submesh([np.where(keep)[0]], append=True) if not keep.all() else m.copy()
    parts = m2.split(only_watertight=False)
    m2 = max(parts, key=lambda p: len(p.faces))
    m2.apply_translation([0, 0, -z_ground])
    print(f"[clean] ground at tyre bottom (shift {z_ground * 1000:.0f} mm), removed {np.sum(~keep)} sliver faces")
    return m2


def postprocess(raw: trimesh.Trimesh, faces: int, anisotropic: bool) -> trimesh.Trimesh:
    from scipy.spatial import cKDTree
    parts = raw.split(only_watertight=False)
    m = max(parts, key=lambda p: len(p.faces))
    print(f"[clean] kept largest of {len(parts)} components: {len(m.faces)} f")
    m = align_to_car_frame(m)
    m = fit_to_spec(m, anisotropic=False)
    m = remove_underbody_slivers(m)
    src_v = m.vertices.copy(); src_c = m.visual.vertex_colors.copy()
    if len(m.faces) > faces:
        m = m.simplify_quadric_decimation(face_count=faces)
        # decimation drops vertex colours -> transfer from nearest original vertex
        _, idx = cKDTree(src_v).query(m.vertices)
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=src_c[idx])
    m = fit_to_spec(m, anisotropic=anisotropic)
    trimesh.repair.fix_normals(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="ioniq5_trellis")
    ap.add_argument("--ss-steps", type=int, default=25)
    ap.add_argument("--slat-steps", type=int, default=25)
    ap.add_argument("--ss-cfg", type=float, default=7.5)
    ap.add_argument("--slat-cfg", type=float, default=3.0)
    ap.add_argument("--faces", type=int, default=300_000, help="target faces after decimation")
    ap.add_argument("--mode", default="stochastic", choices=["stochastic", "multidiffusion"])
    ap.add_argument("--uniform", action="store_true", help="scale by length only (keep generated proportions)")
    ap.add_argument("--from-raw", help="skip generation, post-process an existing *_raw.ply")
    args = ap.parse_args()

    out_dir = os.path.join(HERE, "output"); os.makedirs(out_dir, exist_ok=True)
    in_dir = os.path.join(HERE, "inputs"); os.makedirs(in_dir, exist_ok=True)

    if args.from_raw:
        return save(postprocess(trimesh.load(args.from_raw, process=False), args.faces, not args.uniform),
                    out_dir, args.tag)
    imgs = [remove_bg(p, in_dir) for p in args.images]

    from trellis.pipelines import TrellisImageTo3DPipeline
    t0 = time.time()
    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.cuda()
    print(f"[load] {time.time() - t0:.1f}s")
    for i, im in enumerate(imgs):
        pipe.preprocess_image(im).save(os.path.join(out_dir, f"{args.tag}_cond{i}.png"))

    kw = dict(seed=args.seed, formats=["mesh", "gaussian"],
              sparse_structure_sampler_params={"steps": args.ss_steps, "cfg_strength": args.ss_cfg},
              slat_sampler_params={"steps": args.slat_steps, "cfg_strength": args.slat_cfg})
    t0 = time.time()
    if len(imgs) == 1:
        out = pipe.run(imgs[0], **kw)
    else:
        out = pipe.run_multi_image(imgs, mode=args.mode, **kw)
    print(f"[sample] {time.time() - t0:.1f}s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    mr = out["mesh"][0]
    V = mr.vertices.detach().cpu().numpy().astype(np.float64)
    F = mr.faces.detach().cpu().numpy()
    C = gaussian_vertex_colors(V, out["gaussian"][0])
    raw = trimesh.Trimesh(V, F, vertex_colors=C, process=True)
    raw.export(os.path.join(out_dir, f"{args.tag}_raw.ply"))
    print(f"[raw] {len(raw.vertices)} v / {len(raw.faces)} f, extents {raw.extents}")

    save(postprocess(raw, args.faces, not args.uniform), out_dir, args.tag)


def save(m, out_dir, tag):
    ext = m.extents * 1000
    print(f"[metric] L/W/H = {ext[0]:.0f} / {ext[1]:.0f} / {ext[2]:.0f} mm "
          f"(spec {L_MM:.0f} / {W_MM:.0f} w/o mirrors / {H_MM:.0f})")
    print(f"[metric] watertight={m.is_watertight}, faces={len(m.faces)}")
    for e in ("glb", "obj", "stl"):
        m.export(os.path.join(out_dir, f"{tag}.{e}"))
    print("[done]", os.path.join(out_dir, tag + ".*"))


if __name__ == "__main__":
    main()
