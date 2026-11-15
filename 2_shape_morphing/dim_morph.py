"""
Dimensional morphing of a whole-car STL (fused wheels): front / rear overhang, length,
width and roof-spoiler height as design variables  dim:<name>  (values in mm, 0 = baseline).

| variable       | meaning (mm)                                                         | bounds     |
|----------------|----------------------------------------------------------------------|------------|
| dim:foh        | front overhang change; the nose moves forward (+) / back (-)         | -100 .. 100 |
| dim:roh        | rear overhang change; the tail moves rearward (+) / forward (-)      | -100 .. 100 |
| dim:length     | length change at fixed overhangs = wheelbase change (rear axle moves) | -150 .. 150 |
| dim:width      | body width change (track follows the body)                           |  -60 ..  60 |
| dim:spoiler    | roof-spoiler trailing-edge height change                             |  -30 ..  50 |

Total length change = foh + length + roh. The front axle keeps its coordinates (the nose
moves to x < 0 for foh > 0), so the CFD domain does not have to move.

The morph is a smooth displacement field in space, built from 1-D maps:

    u_x = D(x)                       D = -foh (1 - G_f(x)) + length G_l(x) + roh G_r(x)
    u_y = y * width / W              W = body width of the base
    u_z = spoiler * w_x(x) * w_z(z)  roof kick-up behind the spoiler, glass below stays

G_* are monotone C1 transitions (0 -> 1) whose slope is a plateau with smoothstep ends, so
each stretch zone is stretched almost uniformly, and the nose / tail ends, the wheel arches and
the cabin outside the zone move rigidly (shape kept). Zones (base coordinates):
  front overhang : nose + 0.12 m  ..  front tyre - 0.06 m
  wheelbase      : front tyre + 0.06 m  ..  rear tyre - 0.06 m
  rear overhang  : rear tyre + 0.06 m  ..  tail - 0.12 m
J = du/dx is lower triangular, so det(I + J) = (1 + D') (1 + width/W) (1 + d u_z/dz) exactly.
The wheels (tyre zones, smooth falloff) move rigidly with the displacement at their centre, so
tyres keep their shape and the tyre-to-arch gap is kept.

Landmarks (wheel zones, body width, spoiler trailing edge) are detected on the base mesh;
study.json  base.dim_landmarks  overrides any of them (keys as in detect_landmarks()).
"""
import numpy as np

from common.mesh_io import plane_cut

PREFIX = "dim:"
#        name: (baseline, lower, upper, unit, description)
PARAMS = {
    "foh": (0.0, -100.0, 100.0, "mm", "front overhang change (nose forward +)"),
    "roh": (0.0, -100.0, 100.0, "mm", "rear overhang change (tail rearward +)"),
    "length": (0.0, -150.0, 150.0, "mm", "length change at fixed overhangs = wheelbase change"),
    "width": (0.0, -60.0, 60.0, "mm", "body width change (track follows)"),
    "spoiler": (0.0, -30.0, 50.0, "mm", "roof-spoiler trailing-edge height change"),
}

ARCH_GAP = 0.06        # stretch zones stay this far from the tyres [m]
RIGID_END = 0.12       # nose / tail length that moves rigidly [m]
BLEND = 0.08           # length of the smooth ends of a stretch zone [m]
SPOILER_RAMP = 0.45    # roof length over which the spoiler kick-up builds up [m]
SPOILER_DZ = (0.28, 0.10)   # w_z: 0 below z_te - 0.28, 1 above z_te - 0.10 [m]


def _smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _smooth_d(t):
    return np.where((t > 0) & (t < 1), 6 * t * (1 - t), 0.0)


class Transition:
    """G(x): 0 for x <= a, 1 for x >= b, monotone; G' = plateau with smoothstep ends."""

    def __init__(self, a, b, blend=BLEND, n=4001):
        if b - a < 2.5 * blend:
            raise ValueError(f"stretch zone [{a:.3f}, {b:.3f}] m too short")
        self.a, self.b = a, b
        self.xs = np.linspace(a, b, n)
        p = _smooth((self.xs - a) / blend) * _smooth((b - self.xs) / blend)
        G = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(self.xs))])
        self.p, self.G = p / G[-1], G / G[-1]

    def __call__(self, x):
        return np.interp(x, self.xs, self.G, left=0.0, right=1.0)

    def d(self, x):
        return np.interp(x, self.xs, self.p, left=0.0, right=0.0)

    def max_slope(self):
        return float(self.p.max())


# ----------------------------------------------------------------------------- landmarks
SPOILER_BIN = 0.005         # centre-line profile bins [m]
SPOILER_STEP = 4            # drop measured over 4 bins = 2 cm
SPOILER_MIN_DROP = 0.02     # a smaller drop is no spoiler edge -> warning in the landmarks


def body_width(V, wheels):
    """Body width between the arches, below the mirrors [m]."""
    wf, wr = wheels["front"], wheels["rear"]
    m = (V[:, 0] > wf["x"] + wf["r"] + 0.1) & (V[:, 0] < wr["x"] - wr["r"] - 0.1) & (V[:, 2] < 0.9)
    return 2 * float(np.abs(V[m, 1]).max())


def centreline_top(V, F, x_from, step=SPOILER_BIN):
    """Top of the centre-line section (mesh cut with the plane y = 0) behind x_from, binned in x.
    The cut is exact, so it does not depend on how densely vertices sample the centre line (a
    vertex strip |y| < 3 cm misses the roof on coarse meshes and then reports the floor).
    -> (bin start x, top z per bin; NaN where the section has no point)."""
    seg = plane_cut(V, F, 1, 0.0)
    seg = seg[seg[:, :, 0].max(1) >= x_from]
    if not len(seg):
        raise ValueError("no centre-line section behind x = %.3f m" % x_from)
    t = np.linspace(0, 1, 9)[None, :, None]               # densify segments (<= ~3 mm apart)
    P = (seg[:, :1] + t * (seg[:, 1:] - seg[:, :1])).reshape(-1, 3)
    P = P[P[:, 0] >= x_from]
    n = int(np.ceil((P[:, 0].max() - x_from) / step)) + 1
    idx = np.minimum(((P[:, 0] - x_from) / step).astype(int), n - 1)
    ztop = np.full(n, -np.inf)
    np.maximum.at(ztop, idx, P[:, 2])
    ztop[~np.isfinite(ztop)] = np.nan
    return x_from + step * np.arange(n), ztop


def spoiler_edge(V, F, x_from, x_to):
    """Largest drop of the centre-line top profile over 2 cm, for edges in [x_from, x_to].
    -> (x of the edge, z at the edge, drop). Empty bins stay NaN, so the step is always 2 cm.
    x_to keeps the search off the tail end, where the section top curls down into the rear face."""
    xs, ztop = centreline_top(V, F, x_from)
    drop = ztop[:-SPOILER_STEP] - ztop[SPOILER_STEP:]
    drop[xs[:len(drop)] > x_to] = np.nan
    if not np.isfinite(drop).any():
        raise ValueError("no centre-line profile behind x = %.3f m" % x_from)
    k = int(np.nanargmax(drop))
    return float(xs[k]), float(ztop[k]), float(drop[k])


def detect_landmarks(V, F, wheels, override=None):
    """Zones and reference sizes from the base mesh + wheel data (m). `override` wins."""
    wf, wr = wheels["front"], wheels["rear"]
    x0, x1 = float(V[:, 0].min()), float(V[:, 0].max())
    # spoiler trailing edge: in the rear 30 %, ahead of the rigid tail end
    xte, zte, drop = spoiler_edge(V, F, x0 + 0.7 * (x1 - x0), x1 - RIGID_END)
    lm = {"x_nose": x0, "x_tail": x1, "body_width": body_width(V, wheels),
          "foh_zone": [x0 + RIGID_END, wf["x"] - wf["r"] - ARCH_GAP],
          "wheelbase_zone": [wf["x"] + wf["r"] + ARCH_GAP, wr["x"] - wr["r"] - ARCH_GAP],
          "roh_zone": [wr["x"] + wr["r"] + ARCH_GAP, x1 - RIGID_END],
          "spoiler_te": [xte, zte], "spoiler_drop": drop}
    if drop < SPOILER_MIN_DROP:
        lm["warning"] = (f"no distinct spoiler edge (largest 2 cm drop {drop * 1000:.0f} mm at x={xte:.3f}); "
                         f"set study.json base.dim_landmarks.spoiler_te")
    lm.update(override or {})
    return lm


def measure(V, wheels):
    """Overall dimensions [mm] of a (morphed) car with its wheels."""
    wf, wr = wheels["front"], wheels["rear"]
    return {"FOH": (wf["x"] - V[:, 0].min()) * 1e3, "WB": (wr["x"] - wf["x"]) * 1e3,
            "ROH": (V[:, 0].max() - wr["x"]) * 1e3, "L": (V[:, 0].max() - V[:, 0].min()) * 1e3,
            "W": body_width(V, wheels) * 1e3, "trackF": 2 * wf["y"] * 1e3, "trackR": 2 * wr["y"] * 1e3}


def quality(mo, V0, V1, F):
    """QA of a dimensional morph (used by export and the GUI check):
    det J of the field; triangle flips and normal rotation judged on regular triangles
    (>= 1 % of the median area). Near-collinear slivers of the base can flip under a stretch
    although the field does not fold; they are only counted."""
    detJ = np.linalg.det(np.eye(3)[None] + mo.jacobian(V0))
    n0 = np.cross(V0[F[:, 1]] - V0[F[:, 0]], V0[F[:, 2]] - V0[F[:, 0]])
    n1 = np.cross(V1[F[:, 1]] - V1[F[:, 0]], V1[F[:, 2]] - V1[F[:, 0]])
    a0, a1 = np.linalg.norm(n0, axis=1), np.linalg.norm(n1, axis=1)
    reg = a0 >= 0.01 * np.median(a0)
    cos = np.einsum("ij,ij->i", n0, n1) / np.maximum(a0 * a1, 1e-30)
    moved = np.linalg.norm(V1 - V0, axis=1)
    rep = {"max_displacement": float(moved.max()),
           "moved_vertices_pct": float((moved > 1e-9).mean() * 100),
           "min_detJ": float(detJ.min()), "min_detJ_bound": mo.min_detJ_bound(),
           "flipped_triangles": int((cos[reg] < 0).sum()), "flipped_slivers": int((cos[~reg] < 0).sum()),
           "max_normal_rotation_deg_regular": float(np.degrees(np.arccos(np.clip(cos[reg].min(), -1, 1)))),
           "area_ratio_min": float((a1[reg] / a0[reg]).min()), "area_ratio_max": float((a1[reg] / a0[reg]).max())}
    ok = rep["min_detJ"] > 0.2 and rep["flipped_triangles"] == 0
    rep["status"] = "OK" if ok else ("FOLD RISK" if rep["min_detJ"] > 0 else "FOLDED")
    return rep


# ----------------------------------------------------------------------------- morph
class DimMorph:
    """Displacement field of a set of dim:* values (mm) on a base with landmarks `lm` (m).
    Same interface as rbf.RBFMorph for rbf.quality_report (displacement, jacobian)."""

    def __init__(self, values, lm, wheels=None):
        unknown = set(values) - set(PARAMS)
        if unknown:
            raise KeyError(f"unknown dim variables {sorted(unknown)} (known: {sorted(PARAMS)})")
        self.v = {k: float(values.get(k, 0.0)) / 1000 for k in PARAMS}     # -> m
        self.lm, self.wheels = lm, wheels
        self.Gf = Transition(*lm["foh_zone"])
        self.Gl = Transition(*lm["wheelbase_zone"])
        self.Gr = Transition(*lm["roh_zone"])
        self.s = self.v["width"] / lm["body_width"]
        xte, zte = lm["spoiler_te"]
        self.sx = (xte - SPOILER_RAMP, xte - 0.10)
        self.sz = (zte - SPOILER_DZ[0], zte - SPOILER_DZ[1])

    # 1-D pieces
    def D(self, x):
        v = self.v
        return -v["foh"] * (1 - self.Gf(x)) + v["length"] * self.Gl(x) + v["roh"] * self.Gr(x)

    def Dd(self, x):
        v = self.v
        return v["foh"] * self.Gf.d(x) + v["length"] * self.Gl.d(x) + v["roh"] * self.Gr.d(x)

    def _wx(self, x):
        a, b = self.sx
        return _smooth((x - a) / (b - a)), _smooth_d((x - a) / (b - a)) / (b - a)

    def _wz(self, z):
        a, b = self.sz
        return _smooth((z - a) / (b - a)), _smooth_d((z - a) / (b - a)) / (b - a)

    def field(self, X):
        X = np.asarray(X, float)
        wx, _ = self._wx(X[:, 0]); wz, _ = self._wz(X[:, 2])
        return np.c_[self.D(X[:, 0]), X[:, 1] * self.s, self.v["spoiler"] * wx * wz]

    def wheel_disp(self, key, side):
        w = self.wheels[key]
        return np.array([float(self.D(np.array([w["x"]]))[0]), side * w["y"] * self.s, 0.0])

    def displacement(self, X, freeze=None):
        """Field displacement; with `freeze` (1 away from the tyres, 0 on them) the tyre zones
        move rigidly with their wheel centre."""
        U = self.field(X)
        if freeze is not None and self.wheels is not None:
            wf, wr = self.wheels["front"], self.wheels["rear"]
            front = X[:, 0] < 0.5 * (wf["x"] + wr["x"])
            side = np.where(X[:, 1] >= 0, 1.0, -1.0)
            Uw = np.c_[np.where(front, self.wheel_disp("front", 1)[0], self.wheel_disp("rear", 1)[0]),
                       side * np.where(front, wf["y"], wr["y"]) * self.s, np.zeros(len(X))]
            U = freeze[:, None] * U + (1 - freeze[:, None]) * Uw
        return U

    def jacobian(self, X):
        X = np.asarray(X, float)
        wx, dwx = self._wx(X[:, 0]); wz, dwz = self._wz(X[:, 2])
        J = np.zeros((len(X), 3, 3))
        J[:, 0, 0] = self.Dd(X[:, 0])
        J[:, 1, 1] = self.s
        J[:, 2, 0] = self.v["spoiler"] * dwx * wz
        J[:, 2, 2] = self.v["spoiler"] * wx * dwz
        return J

    def max_handle_ratio(self):
        return 0.0

    def min_detJ_bound(self):
        """Analytic lower bound of det(I + J) over all space (zones are disjoint, G' >= 0,
        w_x in [0, 1], w_z' >= 0: only shrinking values lower it)."""
        v = self.v
        dx = 1 + min(0.0, v["foh"] * self.Gf.max_slope(), v["length"] * self.Gl.max_slope(),
                     v["roh"] * self.Gr.max_slope())
        dz = 1 + min(0.0, v["spoiler"]) * 1.5 / (self.sz[1] - self.sz[0])
        return dx * (1 + self.s) * dz

    def transform_wheels(self, wheels):
        out = {}
        for key, w in wheels.items():
            d = self.wheel_disp(key, 1)
            out[key] = {**w, "x": w["x"] + d[0], "y": w["y"] + d[1]}
        return out


def split(params):
    """{name: value} of the non-zero dim:* entries of a design."""
    return {k[len(PREFIX):]: v for k, v in params.items() if k.startswith(PREFIX) and abs(v) > 1e-12}
