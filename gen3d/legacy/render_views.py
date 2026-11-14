#!/usr/bin/env python3
"""Orthographic + iso preview of a generated mesh (vertex colours if present).
Usage: .venv/bin/python render_views.py output/ioniq5_trellis.glb [out.png] [--compare ../output/ioniq5_body.stl]"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection

LIGHT = np.array([-0.4, 0.5, 0.75]); LIGHT /= np.linalg.norm(LIGHT)


def load(path, max_faces=120_000):
    m = trimesh.load(path, force="mesh")
    if len(m.faces) > max_faces:
        from scipy.spatial import cKDTree
        sv, sc = m.vertices.copy(), m.visual.vertex_colors.copy()
        m = m.simplify_quadric_decimation(face_count=max_faces)
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=sc[cKDTree(sv).query(m.vertices)[1]])
    V, F = m.vertices, m.faces
    try:
        vc = m.visual.vertex_colors[:, :3] / 255.0
        base = vc[F].mean(1)
    except Exception:
        base = np.tile([0.62, 0.66, 0.72], (len(F), 1))
    return V, F, base


def draw(ax, V, F, base, axes_idx, view_dir, title, outline=None):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    shade = 0.35 + 0.65 * np.abs(n @ LIGHT)
    col = np.clip(base * shade[:, None], 0, 1)
    depth = V[F].mean(1) @ view_dir
    order = np.argsort(depth)              # far -> near (view_dir points toward viewer)
    P = V[F][order][:, :, list(axes_idx)]
    ax.add_collection(PolyCollection(P, facecolors=col[order], edgecolors="none"))
    if outline is not None:
        ax.add_collection(PolyCollection(outline, facecolors="none", edgecolors="r", linewidths=0.2, alpha=0.05))
    ax.autoscale(); ax.set_aspect("equal"); ax.set_title(title); ax.grid(alpha=0.3)


def iso(ax, V, F, base, az=-50, el=20):
    azr, elr = np.radians(az), np.radians(el)
    d = np.array([np.cos(elr) * np.cos(azr), np.cos(elr) * np.sin(azr), np.sin(elr)])
    up = np.array([0, 0, 1.0]); r = np.cross(up, d); r /= np.linalg.norm(r); u = np.cross(d, r)
    V2 = np.c_[V @ r, V @ u, V @ d]
    draw(ax, V2, F, base, (0, 1), np.array([0, 0, 1.0]), f"iso az={az} el={el}")
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("out", nargs="?", default=None)
    args = ap.parse_args()
    V, F, base = load(args.mesh)
    fig, axs = plt.subplots(2, 3, figsize=(21, 10))
    draw(axs[0, 0], V, F, base, (0, 2), np.array([0, 1, 0]), "Side (left, +y)  x[m] / z[m]")
    draw(axs[0, 1], V, F, base, (1, 2), np.array([-1, 0, 0]), "Front (-x)  y / z")
    axs[0, 1].invert_xaxis()
    draw(axs[0, 2], V, F, base, (0, 1), np.array([0, 0, 1]), "Top (+z)  x / y")
    iso(axs[1, 0], V, F, base, az=-140, el=18)
    iso(axs[1, 1], V, F, base, az=140, el=18)
    iso(axs[1, 2], V, F, base, az=40, el=25)
    ext = (V.max(0) - V.min(0)) * 1000
    fig.suptitle(f"{args.mesh}   L/W/H = {ext[0]:.0f}/{ext[1]:.0f}/{ext[2]:.0f} mm, {len(F)} faces (preview)")
    fig.tight_layout()
    out = args.out or args.mesh.rsplit(".", 1)[0] + "_views.png"
    fig.savefig(out, dpi=80); print(out)


if __name__ == "__main__":
    main()
