#!/usr/bin/env python3
"""Post-process IONIQ 5 half-model RANS: force coefficients, residuals, Cp and flow-field plots."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.tri import Triangulation

U_INF, RHO = 27.7778, 1.204           # 100 km/h, air @ 20 C
A_FULL = 2.650                        # full-car frontal area [m^2]
N_AVG = 500                           # average over last N iterations
PP = "postProcessing"
OUT = "results"
os.makedirs(OUT, exist_ok=True)


def load_dat(path):
    rows = []
    for line in open(path):
        if line.startswith("#"):
            continue
        parts = line.split()
        try:
            rows.append([float(v) for v in parts])
        except ValueError:
            continue
    return np.array(rows)


def latest(pattern):
    c = glob.glob(pattern)
    return max(c, key=lambda p: float(p.split(os.sep)[-2])) if c else None


def read_vtk(path):
    with open(path) as f:
        lines = f.read().split("\n")
    i = 0
    pts, polys, data = None, None, {}
    while i < len(lines):
        ln = lines[i].split()
        if not ln:
            i += 1; continue
        if ln[0] == "POINTS":
            n = int(ln[1]); vals = []
            i += 1
            while len(vals) < 3 * n:
                vals += lines[i].split(); i += 1
            pts = np.array(vals, float).reshape(n, 3); continue
        if ln[0] == "POLYGONS":
            n, size = int(ln[1]), int(ln[2]); vals = []
            i += 1
            while len(vals) < size:
                vals += lines[i].split(); i += 1
            vals = np.array(vals, int); polys = []; k = 0
            for _ in range(n):
                m = vals[k]; polys.append(vals[k + 1:k + 1 + m]); k += m + 1
            continue
        if len(ln) == 4 and ln[3] == "float" and ln[0] not in ("POINTS",):
            name, nc, n = ln[0], int(ln[1]), int(ln[2]); vals = []
            i += 1
            while len(vals) < nc * n:
                vals += lines[i].split(); i += 1
            data[name] = np.array(vals, float).reshape(n, nc).squeeze(); continue
        i += 1
    return pts, polys, data


def fan_tris(polys):
    t = []
    for p in polys:
        for k in range(1, len(p) - 1):
            t.append((p[0], p[k], p[k + 1]))
    return np.array(t)


# ---------------------------------------------------------------- forces
fc = load_dat(f"{PP}/forceCoeffsIncompressible/0/forceCoeffs.dat")
it, Cm, Cd, Cl, Clf, Clr = fc.T[:6]
sel = it > it.max() - N_AVG
q = 0.5 * RHO * U_INF ** 2
cd, cl, clf, clr = Cd[sel].mean(), Cl[sel].mean(), Clf[sel].mean(), Clr[sel].mean()
summary = {
    "iterations": int(it.max()),
    "Cd (mean last %d)" % N_AVG: cd,
    "Cd std": Cd[sel].std(),
    "Cl": cl, "Cl front": clf, "Cl rear": clr,
    "CdA [m2]": cd * A_FULL,
    "Drag force full car [N]": cd * q * A_FULL,
    "Lift force full car [N]": cl * q * A_FULL,
    "Aero drag power @100km/h [kW]": cd * q * A_FULL * U_INF / 1e3,
}
yp = f"{PP}/yPlus/0/yPlus.dat"
if os.path.exists(yp):
    for line in open(yp):
        if line.startswith("#"):
            continue
        s = line.split()
        if len(s) >= 5 and s[1] in ("body", "wheelFront", "wheelRear", "mirror"):
            summary[f"y+ {s[1]} (min/max/avg)"] = f"{float(s[2]):.1f} / {float(s[3]):.1f} / {float(s[4]):.1f}"
with open(f"{OUT}/summary.txt", "w") as f:
    for k, v in summary.items():
        line = f"{k:40s} {v:.4f}" if isinstance(v, float) else f"{k:40s} {v}"
        print(line); f.write(line + "\n")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].plot(it, Cd, label="Cd"); ax[0].plot(it, Cl, label="Cl")
ax[0].plot(it, Clf, lw=0.8, label="Cl front"); ax[0].plot(it, Clr, lw=0.8, label="Cl rear")
ax[0].axvspan(it.max() - N_AVG, it.max(), color="0.9")
ax[0].set_ylim(-0.6, 1.0); ax[0].set_xlabel("iteration"); ax[0].grid(alpha=.3); ax[0].legend()
ax[0].set_title(f"Force coefficients (avg: Cd={cd:.3f}, Cl={cl:.3f})")
rf = glob.glob(f"{PP}/residuals*/0/residuals.dat")[0]
names = [l for l in open(rf) if l.startswith("# Time")][0].split()[2:]
rows = [l.split() for l in open(rf) if not l.startswith("#")]
rows = [r for r in rows if "N/A" not in r]
R = np.array(rows, float)
for j, n in enumerate(names):
    ax[1].semilogy(R[:, 0], R[:, j + 1], label=n)
ax[1].set_title("Initial residuals"); ax[1].set_xlabel("iteration"); ax[1].grid(alpha=.3); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/convergence.png", dpi=110); plt.close()

# ---------------------------------------------------------------- Cp on car
car = latest(f"{PP}/car/*/patch.vtk")
P, polys, D = read_vtk(car)
T = fan_tris(polys)
cp = D["p"] / (0.5 * U_INF ** 2)
fig, axs = plt.subplots(3, 1, figsize=(13, 13))
views = [("Side (+y)", (0, 2), np.array([0, 1, 0])), ("Top", (0, 1), np.array([0, 0, 1])),
         ("Bottom", (0, 1), np.array([0, 0, -1]))]
cmap = plt.get_cmap("jet")
norm = matplotlib.colors.Normalize(-1.2, 1.0)
for ax, (title, (i, j), d) in zip(axs, views):
    tri = P[T]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vis = n @ d < 0          # OpenFOAM wall normals point out of the fluid (into the car)
    depth = tri[vis].mean(1) @ d
    o = np.argsort(depth)
    col = cmap(norm(cp[T[vis]].mean(1)))[o]
    ax.add_collection(PolyCollection(tri[vis][o][:, :, [i, j]], facecolors=col, edgecolors="none"))
    ax.autoscale(); ax.set_aspect("equal"); ax.set_title(f"Cp – {title}")
fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, label="Cp", shrink=0.6)
plt.savefig(f"{OUT}/cp_surface.png", dpi=110); plt.close()

# ---------------------------------------------------------------- planes
def plane_plot(name, ij, xlim, ylim, title, fname, field="Umag"):
    f = latest(f"{PP}/{name}/*/cutPlane.vtk")
    P, polys, D = read_vtk(f)
    T = fan_tris(polys)
    i, j = ij
    tr = Triangulation(P[:, i], P[:, j], T)
    fig, ax = plt.subplots(figsize=(14, 5) if name != "xWake" else (8, 6))
    if field == "Umag":
        v = np.linalg.norm(D["U"], axis=1) / U_INF
        cs = ax.tripcolor(tr, v, cmap="jet", vmin=0, vmax=1.3, shading="gouraud")
        lab = "|U| / U∞"
    else:
        v = D["p"] / (0.5 * U_INF ** 2) + (np.linalg.norm(D["U"], axis=1) / U_INF) ** 2
        cs = ax.tripcolor(tr, v, cmap="viridis", vmin=0, vmax=1, shading="gouraud")
        lab = "Cp,total"
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.set_title(title)
    fig.colorbar(cs, ax=ax, label=lab, shrink=0.8)
    plt.tight_layout(); plt.savefig(f"{OUT}/{fname}", dpi=110); plt.close()


plane_plot("yNormal", (0, 2), (-1.5, 12), (0, 3), "Velocity – symmetry plane (y=0)", "U_symmetry_plane.png")
plane_plot("zNormal", (0, 1), (-1.5, 12), (0, 3), "Velocity – horizontal plane z=0.5 m", "U_z0.5.png")
plane_plot("xWake", (1, 2), (0, 1.8), (0, 2.0), "Total pressure – wake plane x=5.2 m (0.57 m behind car)",
           "Cptot_wake.png", field="Cpt")

# ---------------------------------------------------------------- one-page summary
def draw_cp_side(ax):
    d = np.array([0, 1, 0]); tri = P[T]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); vis = n @ d < 0
    o = np.argsort(tri[vis].mean(1) @ d)
    ax.add_collection(PolyCollection(tri[vis][o][:, :, [0, 2]], facecolors=cmap(norm(cp[T[vis]].mean(1)))[o],
                                     edgecolors="none"))
    ax.set_xlim(-0.2, 4.85); ax.set_ylim(-0.05, 1.75); ax.set_aspect("equal")


def draw_plane(ax, name, ij, xlim, ylim):
    Pp, pp, Dp = read_vtk(latest(f"{PP}/{name}/*/cutPlane.vtk"))
    tr = Triangulation(Pp[:, ij[0]], Pp[:, ij[1]], fan_tris(pp))
    cs = ax.tripcolor(tr, np.linalg.norm(Dp["U"], axis=1) / U_INF, cmap="jet", vmin=0, vmax=1.3, shading="gouraud")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    return cs


fig, axs = plt.subplots(2, 2, figsize=(17, 9.5))
a = axs[0, 0]
a.plot(it, Cd, label="Cd"); a.plot(it, Cl, label="Cl")
a.axvspan(it.max() - N_AVG, it.max(), color="0.9", label="averaging window")
a.axhspan(0.2, 0.5, color="tab:green", alpha=0.08, label="plausible Cd range")
a.axhline(0.288, color="k", ls="--", lw=0.8, label="IONIQ 5 published Cd 0.288")
a.set_ylim(-0.5, 0.8); a.set_xlabel("iteration"); a.grid(alpha=.3); a.legend(fontsize=8, loc="lower right")
a.set_title(f"Convergence  ->  Cd = {cd:.3f} (+/-{Cd[sel].std():.3f}),  Cl = {cl:.3f}")
a = axs[0, 1]; draw_cp_side(a); a.set_title("Surface pressure coefficient Cp (side view)")
fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=a, label="Cp", shrink=0.8)
a = axs[1, 0]; cs = draw_plane(a, "yNormal", (0, 2), (-1.5, 10), (0, 3)); a.set_title("|U|/U∞  - centre plane y = 0")
a = axs[1, 1]; draw_plane(a, "zNormal", (0, 1), (-1.5, 10), (0, 3)); a.set_title("|U|/U∞  - horizontal plane z = 0.5 m (half)")
fig.colorbar(cs, ax=axs[1, :], label="|U| / U∞", shrink=0.8)
fig.suptitle(f"IONIQ 5 virtual wind tunnel - OpenFOAM 12, steady RANS k-omega SST, 100 km/h, moving belt + rotating wheels\n"
             f"Cd = {cd:.3f}   CdA = {cd*A_FULL:.3f} m²   Drag = {cd*q*A_FULL:.0f} N   Cl = {cl:.3f} "
             f"(F {clf:.3f} / R {clr:.3f})   A_ref = {A_FULL} m²", fontsize=13)
plt.savefig(f"{OUT}/ioniq5_wind_tunnel_summary.png", dpi=110, bbox_inches="tight"); plt.close()
print("figures written to", OUT)
