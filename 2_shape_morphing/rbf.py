"""
RBF handle morphing for arbitrary triangle meshes ("drag" style shape editing).

A morph is a smooth displacement field u(x) defined in space:

    u(x) = sum_j  w_j * phi( |S_j^-1 (x - c_j)| )

  c_j  : centres = drag handles + anchors (pinned points)
  S_j  : support radius per centre (scalar or anisotropic [rx, ry, rz])
  phi  : Wendland C2 kernel (1-t)^4 (4t+1), t<1, else 0: compact support, C2 smooth,
         positive definite. Outside every support the geometry is untouched.
  w_j  : solved so that u(handle_i) = d_i and u(anchor_k) = 0 exactly.

Applying u to the mesh vertices only moves points: connectivity, and hence
watertightness, is preserved. Because u is a field in space, the same morph can
also be applied to a CFD volume mesh.

Handle-set JSON (units = units of the STL):
{
  "symmetry": "y",                      # optional: mirror handles/anchors about y=0
  "planes": [                           # optional: sliding constraints, e.g. ground
    {"point": [0,0,0], "normal": [0,0,1], "blend": 0.05}   # normal motion -> 0 at the plane
  ],
  "handles": [
    {"name": "roof", "center": [3.9, 0.0, 1.6], "disp": [0, 0, -0.04], "radius": 0.8},
    {"center": [4.6, 0.5, 0.6], "normal_disp": -0.03, "radius": [0.4, 0.6, 0.5]}
  ],
  "anchors": [ {"center": [3.85, 0.82, 0.37], "radius": 0.5} ]
}
"""
import json

import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------------- kernel
def wendland(t):
    t = np.minimum(t, 1.0)
    return (1 - t) ** 4 * (4 * t + 1)


def wendland_d(t):
    """d phi / d t."""
    t = np.minimum(t, 1.0)
    return -20.0 * t * (1 - t) ** 3


def _radii(r):
    r = np.atleast_1d(np.asarray(r, float))
    return np.repeat(r, 3) if r.size == 1 else r


# ----------------------------------------------------------------------------- morph
class RBFMorph:
    def __init__(self, spec, vertices=None, normals=None):
        """spec: dict (handle-set JSON). vertices/normals: used to resolve 'normal_disp'."""
        self.spec = spec
        cen, disp, rad, names = [], [], [], []
        for h in spec.get("handles", []):
            c = np.asarray(h["center"], float)
            if "disp" in h:
                d = np.asarray(h["disp"], float)
            elif "normal_disp" in h:
                if vertices is None:
                    raise ValueError("normal_disp needs the mesh to estimate the surface normal")
                n = surface_normal_at(c, vertices, normals)
                d = n * float(h["normal_disp"])
            else:
                raise ValueError(f"handle needs 'disp' or 'normal_disp': {h}")
            cen.append(c); disp.append(d); rad.append(_radii(h["radius"])); names.append(h.get("name", ""))
        for a in spec.get("anchors", []):
            cen.append(np.asarray(a["center"], float)); disp.append(np.zeros(3))
            rad.append(_radii(a.get("radius", 0.3))); names.append("anchor")
        if spec.get("symmetry") == "y":
            for i in range(len(cen)):
                c, d = cen[i], disp[i]
                if abs(c[1]) > 1e-9:
                    cen.append(c * [1, -1, 1]); disp.append(d * [1, -1, 1]); rad.append(rad[i]); names.append(names[i] + "_mirror")
                else:
                    disp[i] = d * [1, 0, 1]             # on the plane: no lateral motion
        self.c = np.asarray(cen).reshape(-1, 3)
        self.d = np.asarray(disp).reshape(-1, 3)
        self.r = np.asarray(rad).reshape(-1, 3)
        self.names = names
        self.planes = [(np.asarray(p["point"], float), _unit(p["normal"]), float(p.get("blend", 0.05)))
                       for p in spec.get("planes", [])]
        self.w = self._solve()

    # weights: A w = d, A_ij = phi(|S_j^-1 (c_i - c_j)|)
    def _solve(self):
        if len(self.c) == 0:
            return np.zeros((0, 3))
        t = np.linalg.norm((self.c[:, None, :] - self.c[None, :, :]) / self.r[None, :, :], axis=2)
        A = wendland(t)
        return np.linalg.lstsq(A, self.d, rcond=None)[0]

    def _blocks(self, X, chunk=200000):
        for s in range(0, len(X), chunk):
            yield s, X[s:s + chunk]

    def displacement(self, X):
        X = np.asarray(X, float)
        U = np.zeros_like(X)
        for s, x in self._blocks(X):
            for c, r, w in zip(self.c, self.r, self.w):
                q = (x - c) / r
                inside = np.abs(q).max(1) < 1.0
                if not inside.any():
                    continue
                t = np.linalg.norm(q[inside], axis=1)
                U[s:s + len(x)][inside] += wendland(t)[:, None] * w
        return self._apply_planes(X, U)

    def jacobian(self, X):
        """du/dx (n,3,3) of the unconstrained field (planes ignored, which is conservative)."""
        X = np.asarray(X, float)
        J = np.zeros((len(X), 3, 3))
        for c, r, w in zip(self.c, self.r, self.w):
            q = (X - c) / r
            inside = np.abs(q).max(1) < 1.0
            if not inside.any():
                continue
            qi = q[inside]
            t = np.maximum(np.linalg.norm(qi, axis=1), EPS)
            dt_dx = qi / (r * t[:, None])                    # d t / d x_k
            g = wendland_d(t)[:, None] * dt_dx               # grad phi
            J[inside] += w[None, :, None] * g[:, None, :]
        return J

    def _apply_planes(self, X, U):
        for p, n, blend in self.planes:
            dist = np.abs((X - p) @ n)
            f = _smoothstep(dist / blend)                    # 0 on the plane -> 1 away from it
            un = U @ n
            U = U - ((1 - f) * un)[:, None] * n[None, :]
        return U

    def max_handle_ratio(self):
        """|d| / min(radius) per handle: a quick fold indicator (keep below ~0.5)."""
        m = np.linalg.norm(self.d, axis=1) / self.r.min(1)
        return float(m.max()) if len(m) else 0.0


# ----------------------------------------------------------------------------- helpers
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def vertex_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], n)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), EPS)


def surface_normal_at(c, V, N, k=12):
    d = np.linalg.norm(V - c, axis=1)
    idx = np.argsort(d)[:k]
    n = N[idx].mean(0)
    return n / np.linalg.norm(n)


def load_spec(path, scale=1.0):
    spec = json.load(open(path))
    if scale != 1.0:                       # amplitude: linear in all handle displacements
        for h in spec.get("handles", []):
            if "disp" in h:
                h["disp"] = [scale * v for v in h["disp"]]
            if "normal_disp" in h:
                h["normal_disp"] *= scale
    return spec


def combine_specs(specs_and_scales):
    """Superpose several morph modes (each with an amplitude) into one handle set."""
    out = {"handles": [], "anchors": [], "planes": []}
    for spec, a in specs_and_scales:
        for h in spec.get("handles", []):
            h = dict(h)
            if "disp" in h:
                h["disp"] = [a * v for v in h["disp"]]
            if "normal_disp" in h:
                h["normal_disp"] *= a
            out["handles"].append(h)
        out["anchors"] += spec.get("anchors", [])
        out["planes"] += spec.get("planes", [])
        if spec.get("symmetry"):
            out["symmetry"] = spec["symmetry"]
    return out


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


# ----------------------------------------------------------------------------- QA
def quality_report(morph, V0, V1, F, frozen=()):
    """Fold check (det J), triangle flips/stretch vs. original, clearance to frozen parts."""
    J = morph.jacobian(V0)
    detJ = np.linalg.det(np.eye(3)[None] + J)
    n0 = np.cross(V0[F[:, 1]] - V0[F[:, 0]], V0[F[:, 2]] - V0[F[:, 0]])
    n1 = np.cross(V1[F[:, 1]] - V1[F[:, 0]], V1[F[:, 2]] - V1[F[:, 0]])
    a0, a1 = np.linalg.norm(n0, axis=1), np.linalg.norm(n1, axis=1)
    cosang = np.einsum("ij,ij->i", n0, n1) / np.maximum(a0 * a1, EPS)
    moved = np.linalg.norm(V1 - V0, axis=1)
    rep = {
        "max_displacement": float(moved.max()),
        "moved_vertices_pct": float((moved > 1e-9 * max(1.0, np.abs(V0).max())).mean() * 100),
        "min_detJ": float(detJ.min()),
        "flipped_triangles": int((cosang < 0).sum()),
        "max_normal_rotation_deg": float(np.degrees(np.arccos(np.clip(cosang.min(), -1, 1)))),
        "area_ratio_min": float((a1 / np.maximum(a0, EPS)).min()),
        "area_ratio_max": float((a1 / np.maximum(a0, EPS)).max()),
        "handle_ratio_max": morph.max_handle_ratio(),
    }
    if frozen:
        from scipy.spatial import cKDTree
        P = np.vstack([v for v in frozen])
        tree = cKDTree(P)
        d0 = tree.query(V0[moved > 0])[0] if (moved > 0).any() else np.array([np.inf])
        d1 = tree.query(V1[moved > 0])[0] if (moved > 0).any() else np.array([np.inf])
        rep["clearance_to_frozen_before"] = float(d0.min())
        rep["clearance_to_frozen_after"] = float(d1.min())
    ok = rep["min_detJ"] > 0.2 and rep["flipped_triangles"] == 0
    rep["status"] = "OK" if ok else ("FOLD RISK" if rep["min_detJ"] > 0 else "FOLDED - reduce disp or enlarge radius")
    return rep
