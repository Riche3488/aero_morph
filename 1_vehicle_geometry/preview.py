#!/usr/bin/env python3
"""Render orthographic views + iso view and compute frontal area."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
sys.path.insert(0, HERE)
import ioniq5_geometry as g

parts = g.build_all(res=1.0)
coarse = g.build_all(res=0.45)
light = np.array([-0.4, 0.5, 0.75]); light /= np.linalg.norm(light)


def shaded(V, F, base):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    s = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
    return np.clip(np.outer(s, base), 0, 1), n


def ortho(ax, axes_idx, view_dir, title):
    i, j = axes_idx
    for name, V, F in parts:
        base = np.array([0.55, 0.62, 0.70]) if name == "body" else np.array([0.2, 0.2, 0.2])
        col, n = shaded(V, F, base)
        vis = n @ view_dir > 0
        order = np.argsort(-(V[F[vis]].mean(1) @ view_dir))[::-1]
        polys = V[F[vis]][order][:, :, [i, j]]
        ax.add_collection(PolyCollection(polys, facecolors=col[vis][order], edgecolors="none"))
    ax.autoscale(); ax.set_aspect("equal"); ax.set_title(title); ax.grid(alpha=0.3)


fig, axs = plt.subplots(2, 2, figsize=(16, 10))
ortho(axs[0, 0], (0, 2), np.array([0, 1, 0]), "Side (left, +y)")
axs[0, 0].axhline(0, color="k", lw=0.8)
ortho(axs[0, 1], (1, 2), np.array([-1, 0, 0]), "Front (-x view)")
axs[0, 1].invert_xaxis()
ortho(axs[1, 0], (0, 1), np.array([0, 0, 1]), "Top")
ax3 = fig.add_subplot(2, 2, 4, projection="3d"); axs[1, 1].remove()
for name, V, F in coarse:
    base = np.array([0.55, 0.62, 0.70]) if name == "body" else np.array([0.2, 0.2, 0.2])
    col, _ = shaded(V, F, base)
    ax3.add_collection3d(Poly3DCollection(V[F], facecolors=col, edgecolors="none"))
ax3.set_xlim(0, 4635); ax3.set_ylim(-2300, 2300); ax3.set_zlim(-1500, 3100)
ax3.view_init(elev=18, azim=-130); ax3.set_box_aspect((1, 1, 1)); ax3.set_title("Iso (front-left)")
plt.tight_layout(); plt.savefig(f"{HERE}/output/ioniq5_views.png", dpi=110)

# underbody / wheel well close-up
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
ortho(ax[0], (0, 1), np.array([0, 0, -1]), "Underbody (bottom view)")
ortho(ax[1], (0, 2), np.array([0, 1, 0]), "Front wheel arch close-up"); ax[1].set_xlim(300, 1400); ax[1].set_ylim(-50, 1100)
plt.tight_layout(); plt.savefig(f"{HERE}/output/ioniq5_underbody.png", dpi=110)

# frontal area by rasterising y-z projection (2 mm grid)
h = 2.0
ys = np.arange(-1100, 1100, h) + h / 2; zs = np.arange(-10, 1700, h) + h / 2
Y, Z = np.meshgrid(ys, zs)
mask = np.zeros_like(Y, bool)
from matplotlib.path import Path
for name, V, F in coarse:
    _, nrm = shaded(V, F, np.ones(3))
    P = V[F[nrm[:, 0] < 0]][:, :, 1:]
    for k in range(0, len(P)):
        tri = P[k]
        lo, hi = tri.min(0), tri.max(0)
        if (hi - lo).min() < 1e-6:
            continue
        i0, i1 = np.searchsorted(ys, [lo[0], hi[0]]); j0, j1 = np.searchsorted(zs, [lo[1], hi[1]])
        if i1 <= i0 or j1 <= j0:
            continue
        sub = np.c_[Y[j0:j1, i0:i1].ravel(), Z[j0:j1, i0:i1].ravel()]
        mask[j0:j1, i0:i1] |= Path(tri).contains_points(sub).reshape(j1 - j0, i1 - i0)
mask &= (Z >= 0)
print(f"Frontal area (z>=0, incl. mirrors+wheels): {mask.sum()*h*h*1e-6:.3f} m^2")
