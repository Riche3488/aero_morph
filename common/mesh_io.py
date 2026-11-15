"""
Triangle-mesh utilities shared by all stages: STL io, watertight checks, volume,
frontal area and plane cuts. Pure numpy, unit-agnostic (results are in the units of
the input coordinates), and independent of every stage folder.

A mesh is (V, F): V (n, 3) float vertices, F (m, 3) int triangle indices.
"""
import struct

import numpy as np

_STL_REC = [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]


# ----------------------------------------------------------------------------- STL io
def read_stl(path):
    """ASCII (also multi-solid) or binary STL -> welded indexed mesh (V, F)."""
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<I", data[80:84])[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50 * n and n > 0:
        tri = np.frombuffer(data, dtype=_STL_REC, count=n, offset=84)["v"].astype(float)
    else:
        txt = data.decode("ascii", "ignore").split()
        pts = [(float(txt[i + 1]), float(txt[i + 2]), float(txt[i + 3]))
               for i in range(len(txt) - 3) if txt[i] == "vertex"]
        tri = np.asarray(pts, float).reshape(-1, 3, 3)
    # weld duplicate corners -> indexed mesh (keeps topology for watertight checks)
    P = tri.reshape(-1, 3)
    scale = max(np.abs(P).max(), 1e-12)
    key = np.round(P / scale * 1e9).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return P[idx], inv.reshape(-1, 3)


def _face_normals(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    return a, b, c, n


def write_stl_binary(path, V, F, name="solid"):
    a, b, c, n = _face_normals(V, F)
    data = np.zeros(len(F), dtype=_STL_REC)
    data["n"] = n
    data["v"] = np.stack([a, b, c], 1)
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(F)))
        f.write(data.tobytes())


def write_stl_ascii_multi(path, parts):
    """parts: [(solid_name, V, F), ...] -> one ASCII STL with named solids (OpenFOAM regions)."""
    with open(path, "w") as f:
        for name, V, F in parts:
            a, b, c, n = _face_normals(V, F)
            f.write(f"solid {name}\n")
            for k in range(len(F)):
                f.write(f" facet normal {n[k,0]:.6e} {n[k,1]:.6e} {n[k,2]:.6e}\n  outer loop\n")
                for p in (a[k], b[k], c[k]):
                    f.write(f"   vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")
                f.write("  endloop\n endfacet\n")
            f.write(f"endsolid {name}\n")


# ----------------------------------------------------------------------------- checks
def signed_volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0


def orient_outward(V, F):
    return F if signed_volume(V, F) > 0 else F[:, ::-1]


def check_watertight(V, F, min_area=1e-6):
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    _, cnt = np.unique(np.sort(e, axis=1), axis=0, return_counts=True)
    _, dcnt = np.unique(e, axis=0, return_counts=True)   # directed: each must be unique
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return dict(edges_not_2=int((cnt != 2).sum()), dup_directed=int((dcnt > 1).sum()),
                degenerate=int((area < min_area).sum()), min_area=float(area.min()))


# ----------------------------------------------------------------------------- frontal area
def frontal_area(parts, h, z_min=0.0):
    """Projected frontal area (y-z plane, z >= z_min) by scan-line rasterisation on an h grid.

    parts: [(name, V, F), ...]. The grid is sized from the parts' bounding box (aligned to
    multiples of h), so nothing is clipped whatever the vehicle size. Result in units^2.
    """
    allV = np.vstack([V for _, V, _ in parts])
    (ylo, zlo), (yhi, zhi) = allV[:, 1:].min(0), allV[:, 1:].max(0)
    zlo = max(zlo, z_min)
    ys = np.arange(np.floor(ylo / h) * h, yhi + h, h) + h / 2
    zs = np.arange(np.floor(zlo / h) * h, zhi + h, h) + h / 2
    zs = zs[zs >= z_min]
    mask = np.zeros((len(zs), len(ys)), bool)
    for _, V, F in parts:
        T = V[F][:, :, 1:]                                   # (n,3,2) y,z
        a = 0.5 * ((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1])
                   - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1]))
        T = T[np.abs(a) > 6.25e-11 * h * h]          # (= 1e-9 mm^2 at h = 4 mm)
        for tri in T:
            j0, j1 = np.searchsorted(zs, [tri[:, 1].min(), tri[:, 1].max()])
            if j1 <= j0:
                continue
            zz = zs[j0:j1, None]
            # intersect scan lines with the 3 edges -> y-interval per line
            p, q = tri, np.roll(tri, -1, axis=0)
            dzv = q[:, 1] - p[:, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (zz - p[:, 1]) / dzv
                yv = p[:, 0] + t * (q[:, 0] - p[:, 0])
            ok = (t >= 0) & (t <= 1) & (dzv != 0)
            ylo_ = np.where(ok, yv, np.inf).min(1)
            yhi_ = np.where(ok, yv, -np.inf).max(1)
            i0 = np.searchsorted(ys, ylo_)
            i1 = np.searchsorted(ys, yhi_)
            for r, (u, v) in enumerate(zip(i0, i1)):
                if v > u:
                    mask[j0 + r, u:v] = True
    return float(mask.sum() * h * h)


# ----------------------------------------------------------------------------- cuts
def plane_cut(V, F, axis, value):
    """Intersect a triangle mesh with the plane x[axis]=value -> array of segments (n,2,3)."""
    value = value + 1e-7 * max(np.ptp(V[:, axis]), 1e-12)   # avoid planes through vertex rows
    d = V[:, axis] - value
    dt = d[F]
    s = np.sign(dt)
    cross = (s.min(1) < 0) & (s.max(1) > 0)
    segs = []
    for tri, dd in zip(V[F[cross]], dt[cross]):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (dd[a] < 0) != (dd[b] < 0):
                t = dd[a] / (dd[a] - dd[b])
                pts.append(tri[a] + t * (tri[b] - tri[a]))
        if len(pts) == 2:
            segs.append(pts)
    return np.asarray(segs).reshape(-1, 2, 3)
