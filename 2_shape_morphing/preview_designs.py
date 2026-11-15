#!/usr/bin/env python3
"""
Geometric preview of every design in a study (no CFD). Works for parametric AND
drag-mode designs because it cuts the exported STL geometry (exports missing ones).

  python3 2_shape_morphing/preview_designs.py <study> [--cases ...]
  -> studies/<study>/previews/<case>.png  and  _all_side.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

import design_table as dt
from common.mesh_io import plane_cut, read_stl
from export_designs import export_case, write_index

SECTIONS_X = (1.5, 3.0, 4.3)     # m
PLAN_Z = 0.55                    # m (plan-view cut height, bumper level)


def load_body(study, case, params):
    export_case(study, case, params, quiet=True)
    d = dt.geometry_dir(study, case)
    parts = {f[:-4]: read_stl(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".stl")}
    return parts


def cut(parts, axis, value, names=("body", "mirror_L", "mirror_R")):
    segs = [plane_cut(V, F, axis, value) for n, (V, F) in parts.items() if n in names]
    return np.vstack([s for s in segs if len(s)]) if any(len(s) for s in segs) else np.zeros((0, 2, 3))


def figure(base, des, title, path, wheels=None):
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3)
    a_side, a_top = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[1, :2])
    a_s1, a_s2 = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    for parts, col, lw in ((base, "0.6", 2.0), (des, "tab:red", 1.0)):
        a_side.add_collection(LineCollection(cut(parts, 1, 0.0)[:, :, [0, 2]], colors=col, linewidths=lw))
        a_top.add_collection(LineCollection(cut(parts, 2, PLAN_Z)[:, :, [0, 1]], colors=col, linewidths=lw))
        for xs, ax in zip(SECTIONS_X, (a_s1, a_s1, a_s2)):
            ax.add_collection(LineCollection(cut(parts, 0, xs)[:, :, [1, 2]], colors=col, linewidths=lw))
    for w in (wheels or {}).values():
        a_side.add_patch(plt.Circle((w["x"], w["z"]), w["r"], fc="0.88", ec="0.6", zorder=0))
    a_side.set_title("centre-line cut y = 0"); a_top.set_title(f"plan cut z = {PLAN_Z} m")
    a_s1.set_title(f"sections x = {SECTIONS_X[0]} & {SECTIONS_X[1]} m"); a_s2.set_title(f"section x = {SECTIONS_X[2]} m")
    for ax in (a_side, a_top, a_s1, a_s2):
        ax.autoscale(); ax.set_aspect("equal"); ax.grid(alpha=.3)
    a_side.plot([], [], color="0.6", lw=2, label="baseline"); a_side.plot([], [], color="tab:red", label="design")
    a_side.legend(loc="lower right")
    fig.suptitle(title)
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study")
    ap.add_argument("--cases", nargs="+")
    a = ap.parse_args()
    rows = [r for r in dt.read_designs(a.study) if not a.cases or r["case"] in a.cases]
    out = os.path.join(dt.study_dir(a.study), "previews")
    os.makedirs(out, exist_ok=True)
    base_row = next((r for r in dt.read_designs(a.study) if dt.label(r["params"]) == "baseline"), None)
    base = load_body(a.study, base_row["case"], base_row["params"]) if base_row else load_body(a.study, "_baseline", {})
    fig, ax = plt.subplots(figsize=(15, 4.5))
    for i, r in enumerate(rows):
        des = load_body(a.study, r["case"], r["params"])
        meta = __import__("json").load(open(os.path.join(dt.geometry_dir(a.study, r["case"]), "meta.json")))
        figure(base, des, f"{r['case']}: {dt.label(r['params'])}   (A = {meta['frontal_area']:.3f} m²)",
               os.path.join(out, f"{r['case']}.png"), meta.get("wheels"))
        ax.add_collection(LineCollection(cut(des, 1, 0.0)[:, :, [0, 2]], colors=f"C{i % 10}", linewidths=0.9, label=r["case"]))
    ax.autoscale(); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(ncol=10, fontsize=7)
    ax.set_title(f"{a.study}: centre-line profiles of all designs")
    plt.tight_layout(); plt.savefig(os.path.join(out, "_all_side.png"), dpi=90); plt.close()
    write_index(a.study)
    print(f"{len(rows)} previews -> {out}")


if __name__ == "__main__":
    main()
