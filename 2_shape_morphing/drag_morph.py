#!/usr/bin/env python3
"""
Apply drag-style RBF morph(s) to arbitrary STL geometry.

  python3 2_shape_morphing/drag_morph.py --in car.stl [more.stl ...] \
      --mode 2_shape_morphing/modes/rear_roof_drop.json=1.0 [--mode other.json=0.5 ...] \
      [--freeze wheel*.stl] --out out_dir [--render]

  --in      surfaces to morph (any closed or open triangle mesh, ASCII or binary STL)
  --mode    handle-set JSON (from drag_editor.html or hand-written) with an amplitude
            (linear scale of all its displacements; modes superpose -> DOE variables)
  --freeze  parts that must NOT move (e.g. wheels); clearance to them is checked
  --out     writes <name>_morphed.stl per input, qa.json, and (with --render) morph.png
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # shared `common`
import rbf  # noqa: E402
from common.mesh_io import read_stl  # noqa: E402,F401  (re-exported: drag_morph.read_stl)
from common.mesh_io import write_stl_binary as write_stl  # noqa: E402,F401


def edge_check(F):
    e = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return {"open_or_nonmanifold_edges": int((cnt != 2).sum())}


# ----------------------------------------------------------------------------- render
def render(path, parts0, parts1, frozen, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    off = np.cumsum([0] + [len(p[1]) for p in parts0])[:-1]
    V0 = np.vstack([p[1] for p in parts0]); V1 = np.vstack([p[1] for p in parts1])
    F = np.vstack([p[2] + o for p, o in zip(parts0, off)])
    un = np.einsum("ij,ij->i", V1 - V0, rbf.vertex_normals(V0, F))      # signed normal displacement
    lim = max(np.abs(un).max(), 1e-12)
    norm = matplotlib.colors.TwoSlopeNorm(0, -lim, lim)
    cmap = plt.get_cmap("RdBu_r")
    light = np.array([-0.3, 0.5, 0.8]) / np.linalg.norm([-0.3, 0.5, 0.8])

    # one depth-sorted scene: morphed parts coloured by displacement + frozen parts in grey
    tris, cols = [V1[F]], [cmap(norm(un[F].mean(1)))]
    for Vf, Ff in frozen:
        tris.append(Vf[Ff]); cols.append(np.tile([0.45, 0.45, 0.45, 1.0], (len(Ff), 1)))
    T, C = np.vstack(tris), np.vstack(cols)
    fn = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
    C[:, :3] *= (0.55 + 0.45 * np.clip(fn @ light, 0, 1))[:, None]

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9])
    views = [("side (+y)", (0, 2), np.array([0, 1, 0]), gs[0, 0:2]), ("top", (0, 1), np.array([0, 0, 1]), gs[1, 0:2]),
             ("front (-x)", (1, 2), np.array([-1, 0, 0]), gs[0, 2]), ("rear (+x)", (1, 2), np.array([1, 0, 0]), gs[1, 2])]
    axes = []
    for name, (i, j), d, g in views:
        ax = fig.add_subplot(g); axes.append(ax)
        vis = fn @ d > 0
        o = np.argsort(T[vis].mean(1) @ d)
        ax.add_collection(PolyCollection(T[vis][o][:, :, [i, j]], facecolors=C[vis][o], edgecolors="none"))
        ax.autoscale(); ax.set_aspect("equal"); ax.set_title(name); ax.grid(alpha=.25)
        if name.startswith("front"):
            ax.invert_xaxis()

    # section cuts through the point of largest displacement: original (grey) vs morphed (red)
    k = int(np.argmax(np.abs(un))); p = V0[k]
    labels = "xyz"
    for col, (axis, (i, j)) in enumerate([(1, (0, 2)), (0, (1, 2)), (2, (0, 1))]):
        ax = fig.add_subplot(gs[2, col])
        for V, c, lw in ((V0, "0.55", 2.2), (V1, "tab:red", 1.0)):
            sg = rbf.plane_cut(V, F, axis, p[axis])
            ax.add_collection(LineCollection(sg[:, :, [i, j]], colors=c, linewidths=lw))
        r = max(4 * lim, 0.06 * np.ptp(V0[:, i]), 1e-9)
        ax.set_xlim(p[i] - 6 * r, p[i] + 6 * r); ax.set_ylim(p[j] - 3 * r, p[j] + 3 * r)
        ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title(f"cut {labels[axis]} = {p[axis]:.4g}   (grey original, red morphed)", fontsize=9)
        ax.set_xlabel(labels[i]); ax.set_ylabel(labels[j])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.7,
                 label="normal displacement (+ outward)  [model units]")
    fig.suptitle(title)
    plt.savefig(path, dpi=90); plt.close()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", nargs="+", required=True)
    ap.add_argument("--mode", nargs="+", required=True, metavar="JSON[=AMPLITUDE]")
    ap.add_argument("--freeze", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    specs = []
    for m in a.mode:
        path, amp = (m.rsplit("=", 1) + ["1"])[:2] if "=" in m else (m, "1")
        specs.append((json.load(open(path)), float(amp)))
    spec = rbf.combine_specs(specs)

    parts0 = [(os.path.splitext(os.path.basename(p))[0], *read_stl(p)) for p in a.inp]
    frozen = [read_stl(p) for p in a.freeze]
    Vall = np.vstack([p[1] for p in parts0])
    Fall = np.vstack([p[2] + o for p, o in zip(parts0, np.cumsum([0] + [len(p[1]) for p in parts0])[:-1])])
    morph = rbf.RBFMorph(spec, Vall, rbf.vertex_normals(Vall, Fall))

    parts1, qa = [], {"modes": [{"file": m} for m in a.mode], "parts": {}}
    for name, V, F in parts0:
        V1 = V + morph.displacement(V)
        parts1.append((name, V1, F))
        rep = rbf.quality_report(morph, V, V1, F, frozen=[f[0] for f in frozen])
        rep.update(edge_check(F))
        qa["parts"][name] = rep
        write_stl(os.path.join(a.out, f"{name}_morphed.stl"), V1, F, name)
        print(f"{name:14s} {rep['status']:10s} max|u|={rep['max_displacement']:.4g}  min detJ={rep['min_detJ']:.3f}  "
              f"flips={rep['flipped_triangles']}  moved={rep['moved_vertices_pct']:.1f}%"
              + (f"  clearance {rep['clearance_to_frozen_before']:.4g}->{rep['clearance_to_frozen_after']:.4g}"
                 if frozen else ""))
    json.dump(qa, open(os.path.join(a.out, "qa.json"), "w"), indent=2)
    if a.render:
        render(os.path.join(a.out, "morph.png"), parts0, parts1, frozen, " + ".join(a.mode))
    print("->", a.out)


if __name__ == "__main__":
    main()
