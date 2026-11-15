#!/usr/bin/env python3
"""
Detailed flow visualisation of one finished CFD case.

  python3 3_cfd/visualize_case.py <study> <case> [--no-sample] [--np 8]

1. sample (unless --no-sample, skipped if already done): runs foamPostProcess on the latest
   decomposed solution for extra wake planes (x = 4.7/5.0/5.5/6.5 m, shifted with the car length),
   a wheel-centre plane, an underbody plane and three streamline rakes (system/vizFunctions).
2. plot into cases/<case>/results/viz/:
     01_cp_views.png          surface Cp, side / top / bottom / front / rear (full car, mirrored)
     02_cp_centreline.png     Cp along the upper and lower centre line + profile
     03_drag_buildup.png      cumulative Cd(x), pressure vs friction, body vs wheels
     04_skin_friction.png     Cf, near-wall flow direction and reverse-flow (separation) regions
     05_centre_planes.png     y = 0 / wheel plane: Cp and |U| with in-plane streamlines
     06_horizontal_planes.png z = 0.5 m and underbody plane |U| with streamlines
     07_wake_planes.png       total-pressure loss and cross-flow vectors at the wake stations
     08_streamlines.png       3-D streamline rakes, side and top view
     viewer3d.html            interactive 3-D viewer (Cp / Cf surface + streamlines), opens offline
                              except for the three.js CDN
     forces_breakdown.json    pressure / friction / per-part Cd from the surface integration
Half model: the solution lives in y >= 0 and is mirrored for display. The surface integration uses
the sampled point data at the final iteration, so it matches the last-iteration force, not the
averaged Cd in result.json (the difference is the iteration noise).
"""
import argparse
import base64
import glob
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline_config as cfg  # noqa: E402

U_INF, RHO = 27.7778, 1.204
Q_KIN = 0.5 * U_INF ** 2


# ------------------------------------------------------------------ I/O
def read_vtk(path):
    """Legacy ASCII VTK polydata -> (points, polygons (list of int arrays) or lines, {field: array})."""
    with open(path) as f:
        lines = f.read().split("\n")
    out = {"points": None, "polys": None, "lines": None, "data": {}}

    def take(i, count):
        vals, j = [], i
        while len(vals) < count:
            vals.extend(lines[j].split()); j += 1
        return vals, j

    i = 0
    while i < len(lines):
        t = lines[i].split()
        if not t:
            i += 1; continue
        if t[0] == "POINTS":
            v, i = take(i + 1, 3 * int(t[1]))
            out["points"] = np.array(v, float).reshape(-1, 3); continue
        if t[0] in ("POLYGONS", "LINES"):
            v, i = take(i + 1, int(t[2]))
            v = np.array(v, int); cells, k = [], 0
            while k < len(v):
                m = v[k]; cells.append(v[k + 1:k + 1 + m]); k += m + 1
            out["polys" if t[0] == "POLYGONS" else "lines"] = cells; continue
        if len(t) == 4 and t[3] == "float":
            nc, n = int(t[1]), int(t[2])
            v, i = take(i + 1, nc * n)
            out["data"][t[0]] = np.array(v, float).reshape(n, nc).squeeze(); continue
        i += 1
    return out


def latest_vtk(cdir, name, fname):
    c = [p for p in glob.glob(os.path.join(cdir, "postProcessing", name, "*", fname))
         if os.path.basename(os.path.dirname(p)).replace(".", "").isdigit()]
    return max(c, key=lambda p: float(os.path.basename(os.path.dirname(p)))) if c else None


def fan(polys):
    """Polygons -> triangle index array (fan triangulation, vectorised per polygon size)."""
    tris = []
    sizes = np.array([len(p) for p in polys])
    for m in np.unique(sizes):
        P = np.array([p for p, s in zip(polys, sizes) if s == m])
        for k in range(1, m - 1):
            tris.append(P[:, [0, k, k + 1]])
    return np.vstack(tris)


# ------------------------------------------------------------------ sampling
def sample(cdir, case_json, nproc):
    x_rear = case_json.get("bbox_x_max") or 4.635
    dx = x_rear - 4.635                               # stations follow the tail of a morphed car
    wy = np.mean([w["y"] for w in case_json["wheels"].values()])
    st = [4.7, 5.0, 5.5, 6.5]
    lines = ["FoamFile { format ascii; class dictionary; object vizFunctions; }"]
    for x in st:
        lines.append(f"#includeFunc cutPlaneSurface(name=vizX{x:.1f}".replace(".", "p") +
                     f", point=({x + dx:.4f} 0 0), normal=(1 0 0), fields=(p U))")
    lines += [f"#includeFunc cutPlaneSurface(name=vizYwheel, point=(0 {wy:.4f} 0), normal=(0 1 0), fields=(p U))",
              "#includeFunc cutPlaneSurface(name=vizZunder, point=(0 0 0.10), normal=(0 0 1), fields=(p U))",
              "#includeFunc streamlinesLine(name=vizStreamCentre, start=(-0.6 0.02 0.08), end=(-0.6 0.02 1.9), "
              "nPoints=36, U, p, direction=forward)",
              "#includeFunc streamlinesLine(name=vizStreamSide, start=(-0.6 0.55 0.08), end=(-0.6 1.15 1.4), "
              "nPoints=36, U, p, direction=forward)",
              f"#includeFunc streamlinesLine(name=vizStreamWake, start=({x_rear + 0.115:.4f} 0.02 0.12), "
              f"end=({x_rear + 0.115:.4f} 0.85 1.45), nPoints=40, U, p, direction=both)"]
    open(os.path.join(cdir, "system", "vizFunctions"), "w").write("\n".join(lines) + "\n")
    cmd = (f"source /opt/openfoam12/etc/bashrc && mpirun -np {nproc} foamPostProcess -parallel "
           f"-latestTime -dict system/vizFunctions > log.vizSample 2>&1")
    subprocess.run(["bash", "-c", cmd], cwd=cdir, check=True)


# ------------------------------------------------------------------ surface + forces
class Surface:
    def __init__(self, vtk, wheels):
        self.P = vtk["points"]; self.T = fan(vtk["polys"])
        self.p = vtk["data"]["p"]; self.tau = vtk["data"]["wallShearStress"]
        a, b, c = (self.P[self.T[:, k]] for k in range(3))
        cr = np.cross(b - a, c - a)
        self.area = 0.5 * np.linalg.norm(cr, axis=1)
        self.n = cr / np.maximum(2 * self.area[:, None], 1e-30)       # vtk winding normal
        self.cen = (a + b + c) / 3
        self.cp_f = self.p[self.T].mean(1) / Q_KIN
        self.tau_f = self.tau[self.T].mean(1)
        self.part = np.full(len(self.T), "body", dtype=object)
        for name, w in wheels.items():
            r = np.hypot(self.cen[:, 0] - w["x"], self.cen[:, 2] - w["z"])
            sel = (r < w["r"] + 0.015) & (self.cen[:, 1] > w["y"] - w["width"] / 2 - 0.04)
            self.part[sel] = "wheelFront" if name == "front" else "wheelRear"

    def forces(self, a_ref):
        """Per-face drag/lift coefficient contributions of the FULL car (half model x 2).
        Signs are fixed physically: pressure acts against the outward body normal, and the
        friction drag of the whole car must be positive."""
        fp = -(self.p[self.T].mean(1) * self.area)[:, None] * self.n      # kinematic, for vtk normal
        fv = self.tau_f * self.area[:, None]
        # orientation of the vtk normals: on the nose (stagnation, high p) the outward normal points -x
        nose = self.cen[:, 0] < np.percentile(self.cen[:, 0], 1)
        if np.sum(self.n[nose, 0] * self.area[nose]) > 0:                  # normals point into the body
            fp = -fp
        if fv[:, 0].sum() < 0:
            fv = -fv
        self.tau_sign = 1.0 if np.sum(self.tau_f[:, 0]) * np.sign(fv[:, 0].sum()) >= 0 else -1.0
        k = 2 * RHO / (0.5 * RHO * U_INF ** 2 * a_ref)                      # -> coefficient, full car
        self.dcd_p, self.dcd_v = k * fp[:, 0], k * fv[:, 0]
        self.dcl = k * (fp[:, 2] + fv[:, 2])
        return self.dcd_p, self.dcd_v


def mirror_tris(P, T, field):
    Pm = P * np.array([1, -1, 1])
    return np.vstack([P, Pm]), np.vstack([T, T[:, ::-1] + len(P)]), np.concatenate([field, field])


# ------------------------------------------------------------------ plotting helpers
VIEWS = {   # name: (horizontal axis, vertical axis, depth axis, depth sign towards the viewer, flip horizontal)
    "side": (0, 2, 1, -1, False),
    "top": (0, 1, 2, 1, False),
    "bottom": (0, 1, 2, -1, False),
    "front": (1, 2, 0, -1, True),
    "rear": (1, 2, 0, 1, False),
}


def surface_view(ax, P, T, face_val, view, cmap, norm, normals):
    h, v, d, sd, flip = VIEWS[view]
    X = P[:, h] * (-1 if flip else 1); Y = P[:, v]; D = sd * P[:, d]
    tri_xy = np.stack([X[T], Y[T]], -1)
    facing = sd * normals[:, d] > -0.05                          # back-face cull (keeps edge-on faces)
    order = np.argsort(D[T].mean(1)[facing])                     # painter: far faces first
    idx = np.where(facing)[0][order]
    pc = PolyCollection(tri_xy[idx], array=face_val[idx], cmap=cmap, norm=norm, edgecolors="face", linewidths=0.05)
    ax.add_collection(pc)
    ax.set_xlim(X.min() - 0.05, X.max() + 0.05); ax.set_ylim(Y.min() - 0.05, Y.max() + 0.05)
    ax.set_aspect("equal"); ax.set_title(view, fontsize=10); ax.tick_params(labelsize=7)
    return pc


def coverage(polys2d, gx, gy):
    """Boolean grid mask of the area covered by the sampled cut-plane polygons (the car section and
    anything outside the domain stay False). Rasterised with Agg, one pixel per grid point."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    res = gx[1] - gx[0]
    nx, ny = len(gx), len(gy)                                  # one spare pixel: Agg rounds the size down
    fig = Figure(figsize=((nx + 1) / 100, (ny + 1) / 100), dpi=100)
    ax = fig.add_axes([0, 0, nx / (nx + 1), ny / (ny + 1)]); ax.axis("off")
    ax.set_xlim(gx[0] - res / 2, gx[-1] + res / 2); ax.set_ylim(gy[0] - res / 2, gy[-1] + res / 2)
    ax.add_collection(PolyCollection(polys2d, facecolors="k", edgecolors="k", linewidths=0.3, antialiaseds=False))
    cv = FigureCanvasAgg(fig); cv.draw()
    img = np.asarray(cv.buffer_rgba())[::-1, :, 0]
    return img[:ny, :nx] < 128


def plane_grid(vtk, ia, ib, extent, res=0.02, mirror=False):
    """Interpolate p, U of a cut plane onto a regular grid; returns masked arrays."""
    from scipy.interpolate import LinearNDInterpolator
    P = vtk["points"]
    a, b = P[:, ia], P[:, ib]
    p, U = vtk["data"]["p"], vtk["data"]["U"]
    polys = [np.c_[a[q], b[q]] for q in vtk["polys"]]
    if mirror:                                                 # mirror across y = 0 (y is axis ia or ib)
        s = np.array([-1, 1]) if ia == 1 else np.array([1, -1])
        polys += [q * s for q in polys]
        a, b = (np.concatenate([a, -a]), np.concatenate([b, b])) if ia == 1 else \
               (np.concatenate([a, a]), np.concatenate([b, -b]))
        U2 = U.copy(); U2[:, 1] *= -1
        p, U = np.concatenate([p, p]), np.vstack([U, U2])
    gx = np.arange(extent[0], extent[1] + 1e-9, res); gy = np.arange(extent[2], extent[3] + 1e-9, res)
    GX, GY = np.meshgrid(gx, gy)
    mask = ~coverage(polys, gx, gy)
    vals = LinearNDInterpolator(np.c_[a, b], np.c_[p, U])(GX, GY)
    out = {k: np.ma.masked_array(vals[..., i], mask | np.isnan(vals[..., i]))
           for i, k in enumerate(["p", "Ux", "Uy", "Uz"])}
    return GX, GY, out


def outline(ax, surf, axes=(0, 2), mirror=False, color="k"):
    P = surf.P
    a, b = P[:, axes[0]], P[:, axes[1]]
    if mirror:
        if axes[0] == 1: a = np.concatenate([a, -a]); b = np.concatenate([b, b])
        else: a = np.concatenate([a, a]); b = np.concatenate([b, -b])
    ax.scatter(a, b, s=0.02, c=color, alpha=0.15, linewidths=0, rasterized=True)


# ------------------------------------------------------------------ figures
def fig_cp_views(S, out, title):
    P, T, cpf = mirror_tris(S.P, S.T, S.cp_f)
    nrm = np.vstack([S.n, S.n * np.array([1, -1, 1])]) * S.orient
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05])
    norm = Normalize(-1.2, 1.0)
    axes = [fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[2, :2]),
            fig.add_subplot(gs[0:1, 2]), fig.add_subplot(gs[1:2, 2])]
    for ax, v in zip(axes, ["side", "top", "bottom", "front", "rear"]):
        pc = surface_view(ax, P, T, cpf, v, "jet", norm, nrm)
    cax = fig.add_subplot(gs[2, 2]); cax.axis("off")
    fig.colorbar(pc, ax=cax, orientation="horizontal", fraction=0.25, label="Cp = (p - p∞) / q∞")
    cax.text(0.5, 0.9, "stagnation  Cp ≈ 1  (red)\nsuction  Cp < 0  (blue)", ha="center", va="top",
             transform=cax.transAxes, fontsize=10)
    fig.suptitle(f"{title} — surface pressure coefficient", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_centreline(S, out, title):
    P, cp = S.P, S.p / Q_KIN
    sel = P[:, 1] < 0.004
    x, z, c = P[sel, 0], P[sel, 2], cp[sel]
    bins = np.linspace(x.min(), x.max(), 120)
    ib = np.clip(np.digitize(x, bins) - 1, 0, len(bins) - 2)
    zmid = np.array([(z[ib == k].max() + z[ib == k].min()) / 2 if np.any(ib == k) else np.nan
                     for k in range(len(bins) - 1)])
    up = z > zmid[ib]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for m, lab, col in [(up, "upper surface (hood / windscreen / roof / tailgate)", "C3"),
                        (~up, "underbody", "C0")]:
        o = np.argsort(x[m])
        a1.plot(x[m][o], c[m][o], ".", ms=2, color=col, alpha=0.5)
        xs = np.convolve(x[m][o], np.ones(9) / 9, "valid"); cs = np.convolve(c[m][o], np.ones(9) / 9, "valid")
        a1.plot(xs, cs, "-", color=col, lw=1.6, label=lab)
    a1.invert_yaxis(); a1.axhline(0, color="k", lw=0.6); a1.set_ylabel("Cp  (inverted)")
    a1.grid(alpha=0.3); a1.legend(loc="lower right"); a1.set_title(f"{title} — centre-line Cp (y = 0)")
    a2.scatter(x, z, c=c, cmap="jet", vmin=-1.2, vmax=1.0, s=3)
    a2.set_aspect("equal"); a2.set_xlabel("x [m]"); a2.set_ylabel("z [m]"); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_drag_buildup(S, out, title, cd_ref, cd_last=None):
    o = np.argsort(S.cen[:, 0]); x = S.cen[o, 0]
    cp_, cv_ = np.cumsum(S.dcd_p[o]), np.cumsum(S.dcd_v[o])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2.2, 1]})
    a1.plot(x, cp_ + cv_, "k", lw=2, label=f"total  Cd = {cp_[-1] + cv_[-1]:.3f}")
    a1.plot(x, cp_, "C3", lw=1.4, label=f"pressure  {cp_[-1]:.3f}")
    a1.plot(x, cv_, "C0", lw=1.4, label=f"friction  {cv_[-1]:.3f}")
    for name, w in S.wheels.items():
        a1.axvspan(w["x"] - w["r"], w["x"] + w["r"], color="0.85", zorder=0)
        a1.text(w["x"], 0, f"{name} wheel", ha="center", va="bottom", fontsize=8)
    a1.axhline(cd_ref, color="C2", ls="--", lw=1, label=f"result.json mean Cd = {cd_ref:.3f}")
    a1.set_xlabel("x [m]  (nose → tail)"); a1.set_ylabel("cumulative Cd"); a1.grid(alpha=0.3); a1.legend()
    a1.set_title("drag build-up along the car (negative slope = thrust, e.g. the front-face suction peaks)")
    parts = ["body", "wheelFront", "wheelRear"]
    vp = [S.dcd_p[S.part == p].sum() for p in parts]; vv = [S.dcd_v[S.part == p].sum() for p in parts]
    a2.bar(parts, vp, color="C3", label="pressure"); a2.bar(parts, vv, bottom=vp, color="C0", label="friction")
    for i, (a, b) in enumerate(zip(vp, vv)):
        a2.text(i, a + b, f"{a + b:.3f}\n({(a + b) / (sum(vp) + sum(vv)) * 100:.0f}%)", ha="center", va="bottom", fontsize=9)
    a2.set_ylabel("Cd contribution"); a2.legend(); a2.grid(alpha=0.3, axis="y"); a2.set_title("drag by part")
    note = "" if cd_last is None else (f"\nsurface integral of interpolated point data: {cp_[-1] + cv_[-1]:.3f} vs solver force "
                                       f"{cd_last:.3f} at the same iteration ({(cp_[-1] + cv_[-1]) / cd_last - 1:+.1%}); "
                                       "use it for shares and x-distribution, the solver Cd for the value")
    fig.suptitle(f"{title} — drag decomposition (surface integral, final iteration){note}", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return dict(zip(parts, [a + b for a, b in zip(vp, vv)]))


def fig_skin_friction(S, out, title):
    cf = np.linalg.norm(S.tau_f, axis=1) / Q_KIN
    t = S.tau_f * S.tau_sign                                     # oriented with the near-wall flow
    P, T, cff = mirror_tris(S.P, S.T, cf)
    nrm = np.vstack([S.n, S.n * np.array([1, -1, 1])]) * S.orient
    rev = (t[:, 0] < 0).astype(float)
    _, _, revm = mirror_tris(S.P, S.T, rev)
    fig, axs = plt.subplots(2, 2, figsize=(17, 9))
    norm = Normalize(0, 0.006)
    for ax, v in [(axs[0, 0], "side"), (axs[0, 1], "top")]:
        pc = surface_view(ax, P, T, cff, v, "viridis", norm, nrm)
    fig.colorbar(pc, ax=axs[0, :], shrink=0.8, label="Cf = |τw| / q∞")
    for ax, v in [(axs[1, 0], "side"), (axs[1, 1], "top")]:
        surface_view(ax, P, T, revm, v, matplotlib.colors.ListedColormap(["0.82", "crimson"]), Normalize(0, 1), nrm)
        h, vv_, d = VIEWS[v][:3]
        face = S.orient * S.n[:, d] > 0.3                         # faces of the simulated half seen in this view
        idx = np.where(face)[0]
        idx = idx[np.random.default_rng(0).choice(len(idx), min(2500, len(idx)), replace=False)]
        dirs = t[idx][:, [h, vv_]]; dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
        ax.quiver(S.cen[idx, h], S.cen[idx, vv_], dirs[:, 0], dirs[:, 1], color="k", scale=90, width=0.0012,
                  headwidth=3, alpha=0.7)
        ax.set_title(f"{v}: near-wall flow direction, red = reverse flow (τx < 0)\n"
                     "(on the rotating wheels τ is relative to the moving wall, so red there is not separation)",
                     fontsize=9)
    fig.suptitle(f"{title} — skin friction and separation", fontsize=13)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def stream_panel(ax, vtk, ia, ib, ua, ub, ext, surf=None, axes=None, mirror=False, field="U", res=0.02, title=""):
    GX, GY, F = plane_grid(vtk, ia, ib, ext, res=res, mirror=mirror)
    comp = {0: "Ux", 1: "Uy", 2: "Uz"}
    Ua, Ub = F[comp[ua]], F[comp[ub]]
    if field == "U":
        val = np.ma.sqrt(F["Ux"] ** 2 + F["Uy"] ** 2 + F["Uz"] ** 2) / U_INF
        im = ax.pcolormesh(GX, GY, val, cmap="jet", vmin=0, vmax=1.3, shading="auto", rasterized=True)
        lab = "|U| / U∞"
    else:
        val = F["p"] / Q_KIN
        im = ax.pcolormesh(GX, GY, val, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto", rasterized=True)
        lab = "Cp"
    ax.streamplot(GX[0], GY[:, 0], Ua.filled(np.nan), Ub.filled(np.nan), color="k", linewidth=0.5,
                  density=2.2, arrowsize=0.6)
    ax.set_aspect("equal"); ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:]); ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, shrink=0.75, label=lab, pad=0.01)


def fig_centre_planes(cdir, out, title, wheel_y):
    yn = read_vtk(latest_vtk(cdir, "yNormal", "cutPlane.vtk"))
    yw = read_vtk(latest_vtk(cdir, "vizYwheel", "cutPlane.vtk"))
    ext = (-1.2, 8.5, 0.0, 2.6)
    fig, axs = plt.subplots(3, 1, figsize=(15, 13))
    stream_panel(axs[0], yn, 0, 2, 0, 2, ext, field="p", title="centre plane y = 0 — Cp with in-plane streamlines")
    stream_panel(axs[1], yn, 0, 2, 0, 2, ext, title="centre plane y = 0 — |U|/U∞ (wake recirculation behind the tailgate)")
    stream_panel(axs[2], yw, 0, 2, 0, 2, ext, title=f"wheel plane y = {wheel_y:.2f} m — |U|/U∞ (wheel wakes, arch flow)")
    fig.suptitle(f"{title} — vertical planes", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_horizontal_planes(cdir, out, title):
    zn = read_vtk(latest_vtk(cdir, "zNormal", "cutPlane.vtk"))
    zu = read_vtk(latest_vtk(cdir, "vizZunder", "cutPlane.vtk"))
    ext = (-1.2, 8.5, -2.0, 2.0)
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    stream_panel(axs[0], zn, 0, 1, 0, 1, ext, mirror=True, title="z = 0.5 m — |U|/U∞ (mirrored)")
    stream_panel(axs[1], zu, 0, 1, 0, 1, ext, mirror=True, title="z = 0.10 m (underbody) — |U|/U∞ (mirrored)")
    fig.suptitle(f"{title} — horizontal planes", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_wake(cdir, out, title):
    names = sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))
                    for p in glob.glob(os.path.join(cdir, "postProcessing", "viz*X*", "*", "cutPlane.vtk"))} |
                   {"xWake"}, key=lambda n: float(read_vtk(latest_vtk(cdir, n, "cutPlane.vtk"))["points"][0, 0]))
    fig, axs = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.6))
    ext = (-1.6, 1.6, 0.0, 2.1)
    for ax, n in zip(np.atleast_1d(axs), names):
        v = read_vtk(latest_vtk(cdir, n, "cutPlane.vtk"))
        xpos = v["points"][0, 0]
        GX, GY, F = plane_grid(v, 1, 2, ext, res=0.02, mirror=True)
        cpt = F["p"] / Q_KIN + (F["Ux"] ** 2 + F["Uy"] ** 2 + F["Uz"] ** 2) / U_INF ** 2
        im = ax.pcolormesh(GX, GY, cpt, cmap="inferno", vmin=0, vmax=1, shading="auto", rasterized=True)
        s = 5
        ax.quiver(GX[::s, ::s], GY[::s, ::s], F["Uy"][::s, ::s], F["Uz"][::s, ::s], color="w", scale=60,
                  width=0.004, alpha=0.8)
        ax.set_aspect("equal"); ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
        ax.set_title(f"x = {xpos:.2f} m", fontsize=10); ax.tick_params(labelsize=7); ax.set_xlabel("y [m]")
    np.atleast_1d(axs)[0].set_ylabel("z [m]")
    fig.colorbar(im, ax=axs, shrink=0.85, label="Cp_total  (1 = undisturbed, <1 = wake loss)")
    fig.suptitle(f"{title} — wake: total-pressure loss and cross-flow vectors (looking upstream from behind)",
                 fontsize=12)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def load_tracks(cdir):
    tr = []
    for n in ["vizStreamCentre", "vizStreamSide", "vizStreamWake"]:
        p = latest_vtk(cdir, n, "tracks.vtk")
        if p:
            v = read_vtk(p)
            sp = np.linalg.norm(v["data"]["U"], axis=1) / U_INF
            tr += [(n, v["points"][l], sp[l]) for l in v["lines"] if len(l) > 2]
    return tr


def fig_streamlines(S, tracks, out, title):
    fig, axs = plt.subplots(2, 1, figsize=(15, 9))
    norm = Normalize(0, 1.3)
    for ax, (h, v), mir in [(axs[0], (0, 2), False), (axs[1], (0, 1), True)]:
        P, T, cpf = mirror_tris(S.P, S.T, S.cp_f)
        X, Y = P[:, h], P[:, v]
        ax.tripcolor(X, Y, T, facecolors=np.full(len(T), 0.6), cmap="Greys", vmin=0, vmax=1, alpha=0.5,
                     rasterized=True)
        for _, pts, sp in tracks:
            for sgn in ([1, -1] if mir else [1]):
                q = pts.copy(); q[:, 1] *= sgn
                seg = np.stack([q[:-1][:, [h, v]], q[1:][:, [h, v]]], 1)
                lc = LineCollection(seg, array=sp[:-1], cmap="jet", norm=norm, linewidths=0.6)
                ax.add_collection(lc)
        ax.set_xlim(-0.8, 9); ax.set_ylim((-0.05, 2.4) if not mir else (-2, 2)); ax.set_aspect("equal")
        ax.set_title("side view" if not mir else "top view (mirrored)", fontsize=10)
    fig.colorbar(lc, ax=axs, shrink=0.8, label="|U| / U∞")
    fig.suptitle(f"{title} — 3-D streamlines (centre, side and wake rakes)", fontsize=13)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ 3-D viewer
VIEWER = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>@TITLE@</title>
<style>
:root{--bg:#f4f5f7;--fg:#1d2330;--panel:#ffffffe6;--mute:#5b6475}
@media (prefers-color-scheme:dark){:root{--bg:#14171d;--fg:#e6e9ef;--panel:#1d222be6;--mute:#9aa3b2}}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;overflow:hidden}
#ui{position:absolute;top:12px;left:12px;background:var(--panel);padding:12px 14px;border-radius:10px;max-width:300px;
box-shadow:0 2px 12px #0003}
#ui h1{font-size:15px;margin:0 0 6px}#ui p{margin:4px 0;color:var(--mute);font-size:12px}
#ui label{display:block;margin:5px 0;cursor:pointer}
#bar{height:10px;border-radius:4px;margin:6px 0 2px}#ticks{display:flex;justify-content:space-between;font-size:11px;color:var(--mute)}
canvas{display:block}
</style></head><body><div id="ui"><h1>@TITLE@</h1><p>@SUB@</p>
<label><input type="radio" name="f" value="cp" checked> surface Cp</label>
<label><input type="radio" name="f" value="cf"> skin friction Cf</label>
<label><input type="checkbox" id="st" checked> streamlines (|U|/U∞)</label>
<div id="bar"></div><div id="ticks"><span id="t0"></span><span id="t1"></span></div>
<p>drag: rotate · right-drag: pan · wheel: zoom</p></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const D=@DATA@;
function dec(s,T){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new T(u.buffer)}
const pos=dec(D.pos,Float32Array),idx=dec(D.idx,Uint32Array),cp=dec(D.cp,Float32Array),cf=dec(D.cf,Float32Array);
const jet=t=>{t=Math.min(1,Math.max(0,t));return[Math.min(1,Math.max(0,1.5-Math.abs(4*t-3))),Math.min(1,Math.max(0,1.5-Math.abs(4*t-2))),Math.min(1,Math.max(0,1.5-Math.abs(4*t-1)))]};
const vir=t=>{t=Math.min(1,Math.max(0,t));return[0.267+t*(0.993-0.267)*t,0.005+t*0.9,0.329+0.3*Math.sin(Math.PI*t)-0.2*t]};
const R=new THREE.WebGLRenderer({antialias:true});R.setPixelRatio(devicePixelRatio);document.body.appendChild(R.domElement);
const S=new THREE.Scene();const C=new THREE.PerspectiveCamera(35,1,0.05,200);C.up.set(0,0,1);C.position.set(-4.5,-6.5,3.2);
const O=new THREE.OrbitControls(C,R.domElement);O.target.set(2.3,0,0.7);O.update();
S.add(new THREE.HemisphereLight(0xffffff,0x444444,0.9));const dl=new THREE.DirectionalLight(0xffffff,0.5);dl.position.set(-3,-5,8);S.add(dl);
function mk(sign){const g=new THREE.BufferGeometry();const p=pos.slice();if(sign<0)for(let i=1;i<p.length;i+=3)p[i]=-p[i];
g.setAttribute('position',new THREE.BufferAttribute(p,3));g.setIndex(new THREE.BufferAttribute(idx,1));g.setAttribute('color',new THREE.BufferAttribute(new Float32Array(p.length),3));g.computeVertexNormals();
const m=new THREE.Mesh(g,new THREE.MeshLambertMaterial({vertexColors:true,side:THREE.DoubleSide}));S.add(m);return m}
const M=[mk(1),mk(-1)];
function paint(f){const v=f=='cp'?cp:cf,lo=f=='cp'?-1.2:0,hi=f=='cp'?1.0:0.006,map=f=='cp'?jet:vir;
for(const m of M){const c=m.geometry.attributes.color.array;for(let i=0;i<v.length;i++){const k=map((v[i]-lo)/(hi-lo));c[3*i]=k[0];c[3*i+1]=k[1];c[3*i+2]=k[2]}m.geometry.attributes.color.needsUpdate=true}
const st=[];for(let i=0;i<=10;i++){const k=map(i/10);st.push(`rgb(${k.map(x=>Math.round(255*x))}) ${i*10}%`)}
document.getElementById('bar').style.background=`linear-gradient(90deg,${st})`;
document.getElementById('t0').textContent=(f=='cp'?'Cp ':'Cf ')+lo;document.getElementById('t1').textContent=hi}
paint('cp');document.querySelectorAll('input[name=f]').forEach(e=>e.onchange=()=>paint(e.value));
const SL=new THREE.Group();S.add(SL);
for(const tr of D.tracks){const pts=dec(tr.p,Float32Array),sp=dec(tr.s,Float32Array);for(const sg of tr.m?[1,-1]:[1]){const g=new THREE.BufferGeometry();const p=pts.slice();if(sg<0)for(let i=1;i<p.length;i+=3)p[i]=-p[i];
const c=new Float32Array(p.length);for(let i=0;i<sp.length;i++){const k=jet(sp[i]/1.3);c[3*i]=k[0];c[3*i+1]=k[1];c[3*i+2]=k[2]}
g.setAttribute('position',new THREE.BufferAttribute(p,3));g.setAttribute('color',new THREE.BufferAttribute(c,3));SL.add(new THREE.Line(g,new THREE.LineBasicMaterial({vertexColors:true})))}}
document.getElementById('st').onchange=e=>SL.visible=e.target.checked;
const gr=new THREE.Mesh(new THREE.PlaneGeometry(16,8),new THREE.MeshLambertMaterial({color:0x888888,transparent:true,opacity:0.25}));gr.position.set(3,0,-0.006);S.add(gr);
function rs(){R.setSize(innerWidth,innerHeight);C.aspect=innerWidth/innerHeight;C.updateProjectionMatrix()}addEventListener('resize',rs);rs();
(function loop(){requestAnimationFrame(loop);R.render(S,C)})();
</script></body></html>"""


def b64(a, dt):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dt).tobytes()).decode()


def write_viewer(S, tracks, out, title, sub):
    tr = []
    for name, pts, sp in tracks:
        step = max(1, len(pts) // 400)
        tr.append({"p": b64(pts[::step], np.float32), "s": b64(sp[::step], np.float32),
                   "m": name != "vizStreamCentre"})
    cf_pt = np.linalg.norm(S.tau, axis=1) / Q_KIN
    data = {"pos": b64(S.P, np.float32), "idx": b64(S.T, np.uint32), "cp": b64(S.p / Q_KIN, np.float32),
            "cf": b64(cf_pt, np.float32), "tracks": tr}
    html = VIEWER.replace("@TITLE@", title).replace("@SUB@", sub).replace("@DATA@", json.dumps(data))
    open(out, "w").write(html)


def last_force_cd(cdir):
    """Solver Cd at the last written iteration (forceCoeffs.dat, newest restart folder)."""
    fs = glob.glob(os.path.join(cdir, "postProcessing", "forceCoeffs*", "*", "forceCoeffs.dat"))
    if not fs:
        return None
    f = max(fs, key=lambda p: float(os.path.basename(os.path.dirname(p))))
    hdr = [l for l in open(f) if l.startswith("#")][-1].lstrip("#").split()
    d = np.loadtxt(f, comments="#", ndmin=2)
    return float(d[-1, hdr.index("Cd")])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study"); ap.add_argument("case")
    ap.add_argument("--no-sample", action="store_true"); ap.add_argument("--np", type=int, default=8)
    a = ap.parse_args()
    cdir = os.path.join(cfg.CFD_RUNS, a.study, "cases", a.case)
    cj = json.load(open(os.path.join(cdir, "case.json")))
    res = json.load(open(os.path.join(cdir, "result.json")))
    surf_vtk = latest_vtk(cdir, "car", "patch.vtk")
    if not surf_vtk:
        raise SystemExit(f"{cdir}: no surface sample (postProcessing/car) — has the case been solved?")
    car = read_vtk(surf_vtk)
    cj["bbox_x_max"] = float(car["points"][:, 0].max())
    if not a.no_sample and not latest_vtk(cdir, "vizStreamWake", "tracks.vtk"):
        print("sampling extra planes / streamlines ..."); sample(cdir, cj, a.np)
    out = os.path.join(cdir, "results", "viz"); os.makedirs(out, exist_ok=True)
    title = f"{cj.get('vehicle', a.study)} [{a.case}] {cj.get('label', '')}"
    S = Surface(car, cj["wheels"]); S.wheels = cj["wheels"]
    S.forces(res["frontal_area"])
    nose = S.cen[:, 0] < np.percentile(S.cen[:, 0], 1)
    S.orient = -1.0 if np.sum(S.n[nose, 0] * S.area[nose]) > 0 else 1.0       # -> outward normals
    print("figures ...")
    fig_cp_views(S, os.path.join(out, "01_cp_views.png"), title)
    fig_centreline(S, os.path.join(out, "02_cp_centreline.png"), title)
    cd_last = last_force_cd(cdir)
    parts = fig_drag_buildup(S, os.path.join(out, "03_drag_buildup.png"), title, res["Cd"], cd_last)
    fig_skin_friction(S, os.path.join(out, "04_skin_friction.png"), title)
    wy = float(np.mean([w["y"] for w in cj["wheels"].values()]))
    fig_centre_planes(cdir, os.path.join(out, "05_centre_planes.png"), title, wy)
    fig_horizontal_planes(cdir, os.path.join(out, "06_horizontal_planes.png"), title)
    fig_wake(cdir, os.path.join(out, "07_wake_planes.png"), title)
    tracks = load_tracks(cdir)
    fig_streamlines(S, tracks, os.path.join(out, "08_streamlines.png"), title)
    write_viewer(S, tracks, os.path.join(out, "viewer3d.html"), title,
                 f"Cd {res['Cd']:.3f} · CdA {res['CdA']:.3f} m² · {res['cells'] / 1e6:.2f} M cells · 100 km/h")
    fb = {"Cd_pressure": float(S.dcd_p.sum()), "Cd_friction": float(S.dcd_v.sum()),
          "Cd_surface_total": float(S.dcd_p.sum() + S.dcd_v.sum()), "Cl_surface": float(S.dcl.sum()),
          "Cd_by_part": {k: float(v) for k, v in parts.items()}, "Cd_result_mean": res["Cd"],
          "Cd_solver_same_iteration": cd_last,
          "note": "surface integral of the final-iteration sample; result.json Cd is the window mean"}
    json.dump(fb, open(os.path.join(out, "forces_breakdown.json"), "w"), indent=2)
    print(json.dumps(fb, indent=2)); print(f"-> {out}")


if __name__ == "__main__":
    main()
