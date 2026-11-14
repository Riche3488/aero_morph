"""Matplotlib previews (no GPU / OpenGL needed): orthographic + iso views, inspection sheet."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree

LIGHT = np.array([-0.4, 0.5, 0.75]); LIGHT /= np.linalg.norm(LIGHT)


def load(path_or_mesh, max_faces=120_000):
    m = trimesh.load(path_or_mesh, force="mesh") if isinstance(path_or_mesh, str) else path_or_mesh.copy()
    has_col = m.visual.kind == "vertex"
    if len(m.faces) > max_faces:
        sv, sc = m.vertices.copy(), (m.visual.vertex_colors.copy() if has_col else None)
        m = m.simplify_quadric_decimation(face_count=max_faces)
        if has_col:
            m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=sc[cKDTree(sv).query(m.vertices)[1]])
    V, F = m.vertices, m.faces
    base = (m.visual.vertex_colors[:, :3] / 255.0)[F].mean(1) if has_col else np.tile([0.62, 0.66, 0.72], (len(F), 1))
    return V, F, base


def draw(ax, V, F, base, axes_idx, view_dir, title):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    col = np.clip(base * (0.35 + 0.65 * np.abs(n @ LIGHT))[:, None], 0, 1)
    order = np.argsort(V[F].mean(1) @ view_dir)       # far -> near
    ax.add_collection(PolyCollection(V[F][order][:, :, list(axes_idx)], facecolors=col[order], edgecolors="none"))
    ax.autoscale(); ax.set_aspect("equal"); ax.set_title(title); ax.grid(alpha=0.3)


def iso(ax, V, F, base, az=-50, el=20, title=None):
    azr, elr = np.radians(az), np.radians(el)
    d = np.array([np.cos(elr) * np.cos(azr), np.cos(elr) * np.sin(azr), np.sin(elr)])
    r = np.cross([0, 0, 1.0], d); r /= np.linalg.norm(r); u = np.cross(d, r)
    draw(ax, np.c_[V @ r, V @ u, V @ d], F, base, (0, 1), np.array([0, 0, 1.0]), title or f"iso az={az} el={el}")
    ax.set_axis_off()


def render_views(mesh, out, title=""):
    V, F, base = load(mesh)
    fig, axs = plt.subplots(2, 3, figsize=(21, 10))
    draw(axs[0, 0], V, F, base, (0, 2), np.array([0, 1, 0]), "Side (left, +y)  x / z [m]")
    draw(axs[0, 1], V, F, base, (1, 2), np.array([-1, 0, 0]), "Front (-x)  y / z"); axs[0, 1].invert_xaxis()
    draw(axs[0, 2], V, F, base, (0, 1), np.array([0, 0, 1]), "Top (+z)  x / y")
    iso(axs[1, 0], V, F, base, -140, 18); iso(axs[1, 1], V, F, base, 140, 18); iso(axs[1, 2], V, F, base, 40, 25)
    ext = (V.max(0) - V.min(0)) * 1000
    fig.suptitle(f"{title}   L/W/H = {ext[0]:.0f}/{ext[1]:.0f}/{ext[2]:.0f} mm")
    fig.tight_layout(); fig.savefig(out, dpi=80); plt.close(fig)
    return out


def render_inspect(mesh, out, title=""):
    """Left vs mirrored right side, bottom, rear, low iso views – for spotting defects."""
    V, F, base = load(mesh, 200_000)
    fig, axs = plt.subplots(3, 2, figsize=(20, 15))
    draw(axs[0, 0], V, F, base, (0, 2), np.array([0, 1, 0]), "LEFT side (+y)")
    draw(axs[0, 1], V * [1, -1, 1], F[:, ::-1], base, (0, 2), np.array([0, 1, 0]), "RIGHT side (-y), mirrored")
    draw(axs[1, 0], V, F, base, (0, 1), np.array([0, 0, -1]), "BOTTOM (-z)")
    draw(axs[1, 1], V, F, base, (1, 2), np.array([1, 0, 0]), "REAR (+x)")
    iso(axs[2, 0], V, F, base, -120, -15, "low iso front-right"); iso(axs[2, 1], V, F, base, 60, -15, "low iso rear-left")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=75); plt.close(fig)
    return out


def compare_sheet(pngs, labels, out, cols=2, scale=0.55):
    from PIL import Image, ImageDraw
    ims = [Image.open(p).convert("RGB") for p in pngs]
    ims = [im.resize((int(im.width * scale), int(im.height * scale))) for im in ims]
    W, H = max(i.width for i in ims), max(i.height for i in ims)
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (W * cols, H * rows), "white")
    dr = ImageDraw.Draw(sheet)
    for k, (im, lab) in enumerate(zip(ims, labels)):
        x, y = (k % cols) * W, (k // cols) * H
        sheet.paste(im, (x, y)); dr.text((x + 8, y + 4), lab, fill="red")
    sheet.save(out)
    return out
