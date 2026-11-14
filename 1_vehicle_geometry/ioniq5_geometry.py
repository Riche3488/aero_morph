#!/usr/bin/env python3
"""
Hyundai IONIQ 5 - simplified exterior geometry for external aerodynamics (CFD).

Parametric loft of cross-sections built from public dimensions:
  Length 4635, Width 1890 (w/o mirrors), Height 1605, Wheelbase 3000,
  Front/Rear overhang 845/790, Track F/R 1638/1647, Tire 235/55R19 (D=741),
  Ground clearance ~160 mm.

Coordinate system (mm, SI-ready via --scale):
  x : rearward, x=0 at front bumper tip
  y : left(+)/right(-), y=0 on symmetry plane
  z : up, z=0 at ground plane

Outputs (closed, watertight, outward normals):
  body (with open wheel wells), 4 tires (solid discs), 2 mirror housings,
  plus a combined multi-solid ASCII STL (named solids, OpenFOAM friendly).
"""
import argparse
import os
import struct

import numpy as np
from scipy.interpolate import PchipInterpolator

# ----------------------------------------------------------------------------
# Main dimensions
# ----------------------------------------------------------------------------
L = 4635.0
W_HALF = 945.0
WB = 3000.0
OH_F = 845.0
X_FA, X_RA = OH_F, OH_F + WB
TRACK_F, TRACK_R = 1638.0, 1647.0
TIRE_W = 235.0
TIRE_R = 0.5 * (19 * 25.4 + 2 * 0.55 * 235.0)      # 370.6 mm
TIRE_SINK = 5.0                                   # contact-patch penetration into ground
Z_WHEEL = TIRE_R - TIRE_SINK
ARCH_R = 415.0                                    # wheel-arch opening radius
ARCH_RAMP = 60.0                                  # lower-arch transition length
WELL_GAP = 25.0                                   # tire inner face -> well wall

WHEELS = [  # name, x, y_center
    ("wheel_FL", X_FA, +TRACK_F / 2), ("wheel_FR", X_FA, -TRACK_F / 2),
    ("wheel_RL", X_RA, +TRACK_R / 2), ("wheel_RR", X_RA, -TRACK_R / 2),
]


def pchip(pts):
    a = np.asarray(pts, float)
    return PchipInterpolator(a[:, 0], a[:, 1], extrapolate=True)


# Side-view upper profile (roof / hood / glass)
Z_TOP0 = pchip([
    (0, 560), (8, 640), (25, 705), (60, 760), (120, 800), (250, 840), (600, 890),
    (1000, 950), (1250, 1000), (1320, 1040), (1500, 1190), (1700, 1360),
    (1900, 1505), (2050, 1575), (2200, 1600), (3000, 1605), (3800, 1595),
    (4200, 1580), (4380, 1565), (4450, 1450), (4520, 1260), (4580, 1085),
    (4620, 1015), (4635, 985),
])
# Side-view lower profile (underbody)
Z_BOT0 = pchip([
    (0, 440), (8, 370), (25, 320), (60, 280), (150, 250), (400, 220),
    (845, 195), (1300, 165), (2000, 160), (3300, 160), (3700, 170),
    (4100, 200), (4400, 260), (4560, 330), (4620, 380), (4635, 420),
])
# Plan-view half width
HALF_W0 = pchip([
    (0, 520), (8, 640), (25, 730), (60, 810), (120, 870), (250, 915),
    (500, 938), (900, 945), (3800, 945), (4250, 940), (4450, 925),
    (4570, 895), (4620, 860), (4635, 830),
])
# Beltline (shoulder) height
Z_BELT0 = pchip([(0, 700), (1200, 960), (2000, 990), (4000, 1020), (4635, 1040)])
ROOF_HALF_W = 690.0


# ----------------------------------------------------------------------------
# Shape morphing: design variables -> smooth deformation of the profiles.
# Every variable is 0 (or 1 for scales) at baseline. Units: mm / deg / -.
# ----------------------------------------------------------------------------
MORPH_PARAMS = {
    # name:            (baseline, lower, upper, unit, description)
    "ride_height":      (0.0, -25.0, 30.0, "mm",  "body vertical offset vs. wheels (wheels stay on ground)"),
    "hood_height":      (0.0, -40.0, 40.0, "mm",  "hood / front-end height, fades out at windshield base"),
    "front_corner":     (0.0, -20.0, 80.0, "mm",  "extra plan-view rounding of the front corners"),
    "windshield_rake":  (0.0, -100.0, 100.0, "mm", "+ moves windshield base forward (more raked)"),
    "roof_rear_drop":   (0.0, -20.0, 60.0, "mm",  "+ lowers the rear roof / spoiler lip (fastback-ish)"),
    "spoiler_length":   (0.0, -60.0, 60.0, "mm",  "+ extends the roof spoiler rearward"),
    "tail_taper":       (0.0, 0.0, 80.0, "mm",    "boat-tail: half-width reduction at the tail"),
    "diffuser_angle":   (0.0, -2.0, 6.0, "deg",   "extra rear underbody upsweep from x=4.0 m"),
    "mirror_scale":     (1.0, 0.0, 1.2, "-",      "mirror housing scale (0 = removed)"),
}
DESIGN = {k: v[0] for k, v in MORPH_PARAMS.items()}
RIDE_DZ = 0.0


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _xwarp(f, amp, w):
    """f(x - amp*w(x)): shifts features of f by amp where w=1 (monotonic for small amp*w')."""
    if amp == 0:
        return f
    return lambda x: f(np.asarray(x, float) - amp * w(np.asarray(x, float)))


def set_design(params=None, **kw):
    """Activate a design. Unknown keys raise; missing keys take baseline values."""
    global DESIGN, Z_TOP, Z_BOT, HALF_W, Z_BELT, ROOF_HW, RIDE_DZ
    d = {k: v[0] for k, v in MORPH_PARAMS.items()}
    for k, v in {**(params or {}), **kw}.items():
        if k not in MORPH_PARAMS:
            raise KeyError(f"unknown morph parameter '{k}' (valid: {list(MORPH_PARAMS)})")
        lo, hi = MORPH_PARAMS[k][1:3]
        if not lo <= float(v) <= hi:
            raise ValueError(f"{k}={v} outside validated range [{lo}, {hi}]")
        d[k] = float(v)
    DESIGN, RIDE_DZ = d, d["ride_height"]
    dz = d["ride_height"]

    # --- upper profile: windshield rake + spoiler (x-warps), hood, rear-roof drop (offsets)
    zt = _xwarp(Z_TOP0, -d["windshield_rake"],
                lambda x: smoothstep((x - 900) / 350) * (1 - smoothstep((x - 1500) / 550)))
    zt = _xwarp(zt, d["spoiler_length"],
                lambda x: smoothstep((x - 3900) / 400) * (1 - smoothstep((x - 4450) / 185)))
    hood, drop = d["hood_height"], d["roof_rear_drop"]
    Z_TOP = lambda x, f=zt: (f(x) + dz + hood * (1 - smoothstep((np.asarray(x) - 800) / 500))
                             - drop * smoothstep((np.asarray(x) - 2800) / 1580)
                             * (1 - smoothstep((np.asarray(x) - 4450) / 185)))
    # --- lower profile: ride height + diffuser upsweep
    ta = np.tan(np.radians(d["diffuser_angle"]))
    Z_BOT = lambda x: (Z_BOT0(x) + dz + ta * np.maximum(np.asarray(x) - 4000, 0)
                       * smoothstep((np.asarray(x) - 4000) / 200))
    Z_BELT = lambda x: Z_BELT0(x) + dz
    # --- plan view: front-corner rounding + boat-tail
    fc, tt = d["front_corner"], d["tail_taper"]
    w_tail = lambda x: smoothstep((np.asarray(x) - 3600) / 1035)
    HALF_W = lambda x: HALF_W0(x) - fc * (1 - smoothstep(np.asarray(x) / 450)) - tt * w_tail(x)
    ROOF_HW = lambda x: ROOF_HALF_W - 0.7 * tt * w_tail(x)
    return d


def validate_design(n=600):
    """Cheap geometric sanity checks on the active design; returns list of problems."""
    x = np.linspace(0, L, n)
    zt, zb, hw = Z_TOP(x), Z_BOT(x), HALF_W(x)
    errs = []
    if np.min(zt - zb) < 80:
        errs.append(f"section height < 80 mm at x={x[np.argmin(zt - zb)]:.0f}")
    if np.min(hw) < 300:
        errs.append("half width < 300 mm")
    arch_top_clear = (Z_WHEEL + RIDE_DZ + ARCH_R) - (Z_WHEEL + TIRE_R)
    if arch_top_clear < 10:
        errs.append("wheel arch would touch the tire")
    if Z_BOT(L / 2) < 100:
        errs.append("ground clearance < 100 mm")
    return errs


set_design()


# ----------------------------------------------------------------------------
# 2D helpers
# ----------------------------------------------------------------------------
def fillet_polyline(pts, radii, n_arc=12):
    """Round interior corners of an open polyline with tangent arcs."""
    pts = np.asarray(pts, float)
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        A, B, C = pts[i - 1], pts[i], pts[i + 1]
        r = radii[i]
        u, v = A - B, C - B
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        if r <= 0 or lu < 1e-9 or lv < 1e-9:
            out.append(B)
            continue
        u, v = u / lu, v / lv
        cosang = np.clip(np.dot(u, v), -1, 1)
        theta = np.arccos(cosang)
        if theta > np.pi - 1e-3 or theta < 1e-3:
            out.append(B)
            continue
        d = r / np.tan(theta / 2)
        dmax = 0.45 * min(lu, lv)
        if d > dmax:
            d = dmax
            r = d * np.tan(theta / 2)
        T1, T2 = B + u * d, B + v * d
        bis = (u + v) / np.linalg.norm(u + v)
        Cc = B + bis * (r / np.sin(theta / 2))
        a1 = np.arctan2(*(T1 - Cc)[::-1])
        a2 = np.arctan2(*(T2 - Cc)[::-1])
        da = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        ang = a1 + da * np.linspace(0, 1, n_arc)
        out.extend(np.c_[Cc[0] + r * np.cos(ang), Cc[1] + r * np.sin(ang)])
    out.append(pts[-1])
    return np.asarray(out)


def resample(poly, n):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    t = np.linspace(0, s[-1], n)
    return np.c_[np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])]


def arch_height(x, y_side):
    """Upper z of wheel-well opening at station x for the given side (sign of y)."""
    zmax, yin = -1e9, None
    for _, xw, yw in WHEELS:
        if np.sign(yw) != y_side:
            continue
        dx = abs(x - xw)
        if dx <= ARCH_R:
            z = Z_WHEEL + RIDE_DZ + np.sqrt(ARCH_R ** 2 - dx ** 2)
        elif dx <= ARCH_R + ARCH_RAMP:
            f = (dx - ARCH_R) / ARCH_RAMP
            z = (Z_WHEEL + RIDE_DZ) + (Z_BOT(x) - 2.0 - Z_WHEEL - RIDE_DZ) * f
        else:
            continue
        if z > zmax:
            zmax = z
            yin = abs(yw) - TIRE_W / 2 - WELL_GAP
    return zmax, yin


def half_outline_dense(x):
    zb, zt, hw = float(Z_BOT(x)), float(Z_TOP(x)), float(HALF_W(x))
    zs = min(zt - 110.0, float(Z_BELT(x)))
    zs = max(zs, zb + 0.3 * (zt - zb))
    g = np.clip((zt - zs - 110.0) / 450.0, 0, 1)        # 0 hood/bumper -> 1 greenhouse
    hr = (hw - 110.0) * (1 - g) + float(ROOF_HW(x)) * g
    crown = 25.0 * g
    zm = zb + 0.4 * (zs - zb)
    pts = [(0, zb), (hw - 45, zb), (hw, zm), (hw, zs), (hr, zt - crown), (0, zt)]
    radii = [0, 60, 200, 120 + 60 * (1 - g), 110 + 60 * (1 - g), 0]
    return resample(fillet_polyline(pts, radii, n_arc=40), 4000), zb


def sample_polyline_exact(verts, n):
    """n points on a polyline, keeping every corner vertex exactly."""
    verts = np.asarray(verts, float)
    seg = np.linalg.norm(np.diff(verts, axis=0), axis=1)
    keep = seg > 1e-6
    if not keep.any():
        return np.repeat(verts[:1], n, axis=0)
    starts, ends, seg = verts[:-1][keep], verts[1:][keep], seg[keep]
    k = len(seg)
    ivals = n - 1
    raw = ivals * seg / seg.sum()
    cnt = np.maximum(np.floor(raw).astype(int), 1 if ivals >= k else 0)
    while cnt.sum() > ivals:
        cnt[np.argmax(cnt)] -= 1
    rem = raw - cnt
    while cnt.sum() < ivals:
        i = np.argmax(rem); cnt[i] += 1; rem[i] = -1e9
    out = []
    for a, b, c in zip(starts, ends, cnt):
        t = np.arange(c) / max(c, 1)
        out.append(a + np.outer(t, b - a))
    out.append(ends[-1:])
    return np.vstack(out)


def half_section(x, side, n_low, n_up):
    """Half outline (y>=0 coords) from bottom-centre to roof-centre with the
    wheel-well opening cut in. Point count is fixed (n_low + n_up - 1) and the
    arch crossing is always vertex n_low-1, so the arch edge is crisp."""
    dense, zb = half_outline_dense(x)
    za, yin = arch_height(x, side)
    zsplit = zb if yin is None else max(zb, za)
    z = dense[:, 1]
    i = int(np.argmax(z > zsplit + 1e-6))
    t = (zsplit - z[i - 1]) / max(z[i] - z[i - 1], 1e-12)
    C = dense[i - 1] + np.clip(t, 0, 1) * (dense[i] - dense[i - 1])
    if yin is not None and zsplit > zb + 1e-6 and yin < C[0]:
        low = [(0, zb), (yin, zb), (yin, zsplit), C]
    else:
        low = [(0, zb), C]
    lower = sample_polyline_exact(low, n_low)
    upper = resample(np.vstack([C, dense[i:]]), n_up)
    return np.vstack([lower, upper[1:]])


# ----------------------------------------------------------------------------
# Mesh builders
# ----------------------------------------------------------------------------
def build_body(n_half=170, base_st=320, arch_step=5.0):
    xs = 0.5 * L * (1 - np.cos(np.linspace(0, np.pi, base_st)))
    extra = []
    for xw in (X_FA, X_RA):
        extra.append(np.arange(xw - ARCH_R - ARCH_RAMP - 30, xw + ARCH_R + ARCH_RAMP + 30, arch_step))
    xs = np.unique(np.round(np.r_[xs, np.concatenate(extra)], 3))
    xs = xs[(xs >= 0) & (xs <= L)]

    rings = []
    for x in xs:
        n_low = max(int(0.42 * n_half), 20)
        hl = half_section(x, +1, n_low, n_half - n_low + 1)
        hr_ = half_section(x, -1, n_low, n_half - n_low + 1)
        # closed loop: bottom-centre -> left(+y) -> top-centre -> right(-y) -> back
        left = hl
        right = np.c_[-hr_[::-1, 0], hr_[::-1, 1]][1:-1]
        loop = np.vstack([left, right])
        rings.append(np.c_[np.full(len(loop), x), loop[:, 0], loop[:, 1]])
    rings = np.asarray(rings)                  # (nst, m, 3)
    nst, m, _ = rings.shape
    V = rings.reshape(-1, 3)
    F = []
    idx = np.arange(nst * m).reshape(nst, m)
    a = idx[:-1, :]
    b = idx[1:, :]
    a1 = np.roll(a, -1, axis=1)
    b1 = np.roll(b, -1, axis=1)
    F.append(np.stack([a, b, b1], -1).reshape(-1, 3))
    F.append(np.stack([a, b1, a1], -1).reshape(-1, 3))
    # end caps (fans to centroid)
    for k, flip in ((0, False), (nst - 1, True)):
        c = len(V)
        V = np.vstack([V, rings[k].mean(axis=0)])
        r = idx[k]
        tri = np.c_[np.full(m, c), r, np.roll(r, -1)]
        F.append(tri[:, ::-1] if flip else tri)
    return V, np.vstack(F)


def revolve_y(profile, n_seg, center):
    """Revolve (r, y) profile around the y-axis. Endpoints with r=0 become poles."""
    th = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
    inner = profile[1:-1]
    V = []
    for r, yy in inner:
        V.append(np.c_[r * np.cos(th), np.full(n_seg, yy), r * np.sin(th)])
    V = np.vstack(V)
    p0 = len(V)
    V = np.vstack([V, [0, profile[0, 1], 0], [0, profile[-1, 1], 0]])
    nr = len(inner)
    F = []
    for i in range(nr - 1):
        a = i * n_seg + np.arange(n_seg)
        b = (i + 1) * n_seg + np.arange(n_seg)
        a1, b1 = np.roll(a, -1), np.roll(b, -1)
        F += [np.c_[a, a1, b1], np.c_[a, b1, b]]
    a = np.arange(n_seg)
    F.append(np.c_[np.full(n_seg, p0), np.roll(a, -1), a])
    a = (nr - 1) * n_seg + np.arange(n_seg)
    F.append(np.c_[np.full(n_seg, p0 + 1), a, np.roll(a, -1)])
    return V + np.asarray(center), np.vstack(F)


def build_tire(xc, yc, zc, n_seg=128):
    w2, R, rs = TIRE_W / 2, TIRE_R, 45.0
    rim_r, dish = 0.5 * 19 * 25.4, 30.0      # shallow wheel-face recess
    prof = np.array([(0, -w2 + dish), (rim_r - 20, -w2 + dish), (rim_r, -w2),
                     (R, -w2), (R, w2), (rim_r, w2), (rim_r - 20, w2 - dish), (0, w2 - dish)])
    prof = fillet_polyline(prof, [0, 15, 10, rs, rs, 10, 15, 0], n_arc=10)
    return revolve_y(prof, n_seg, (xc, yc, zc))


def build_ellipsoid(c, ax, n_u=48, n_v=24):
    v = np.linspace(-np.pi / 2, np.pi / 2, n_v + 1)
    prof = np.c_[np.cos(v), np.sin(v)]          # (r, y) on unit sphere, poles at y=±1
    prof[0, 0] = prof[-1, 0] = 0
    V, F = revolve_y(prof, n_u, (0, 0, 0))
    return V * np.asarray(ax) + np.asarray(c), F


def build_mirror(side, scale=1.0):
    # door-mounted housing; inner edge stays embedded in the body (y = 840 mm)
    a = np.array([95, 105, 62]) * scale
    Vh, Fh = build_ellipsoid((1430, side * (840 + a[1]), 1065 + RIDE_DZ), a)
    return Vh, Fh


# ----------------------------------------------------------------------------
# Checks & IO
# ----------------------------------------------------------------------------
def signed_volume(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0


def orient_outward(V, F):
    return F if signed_volume(V, F) > 0 else F[:, ::-1]


def check_watertight(V, F):
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.sort(e, axis=1)
    _, cnt = np.unique(key, axis=0, return_counts=True)
    _, dcnt = np.unique(e, axis=0, return_counts=True)   # directed: each must be unique
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return dict(edges_not_2=int((cnt != 2).sum()), dup_directed=int((dcnt > 1).sum()),
                degenerate=int((area < 1e-6).sum()), min_area=float(area.min()))


def write_stl_binary(path, V, F, name="solid"):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    data = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    data["n"] = n
    data["v"] = np.stack([a, b, c], 1)
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b" "))
        f.write(struct.pack("<I", len(F)))
        f.write(data.tobytes())


def write_stl_ascii_multi(path, parts):
    with open(path, "w") as f:
        for name, V, F in parts:
            a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            n = np.cross(b - a, c - a)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
            f.write(f"solid {name}\n")
            for k in range(len(F)):
                f.write(f" facet normal {n[k,0]:.6e} {n[k,1]:.6e} {n[k,2]:.6e}\n  outer loop\n")
                for p in (a[k], b[k], c[k]):
                    f.write(f"   vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n")
                f.write("  endloop\n endfacet\n")
            f.write(f"endsolid {name}\n")


def build_all(res=1.0, design=None):
    if design is not None:
        set_design(design)
    parts = []
    V, F = build_body(n_half=int(170 * res), base_st=int(320 * res), arch_step=5.0 / res)
    parts.append(("body", V, orient_outward(V, F)))
    for name, xw, yw in WHEELS:
        V, F = build_tire(xw, yw, Z_WHEEL, n_seg=int(128 * res))
        parts.append((name, V, orient_outward(V, F)))
    if DESIGN["mirror_scale"] > 0.05:
        for name, s in (("mirror_L", +1), ("mirror_R", -1)):
            V, F = build_mirror(s, DESIGN["mirror_scale"])
            parts.append((name, V, orient_outward(V, F)))
    return parts


def frontal_area(parts, h=4.0):
    """Projected frontal area [m^2] (y-z plane, z>=0) by scan-line rasterisation (h mm grid)."""
    ys = np.arange(-1200, 1200, h) + h / 2
    zs = np.arange(0, 1800, h) + h / 2
    mask = np.zeros((len(zs), len(ys)), bool)
    for _, V, F in parts:
        T = V[F][:, :, 1:]                                   # (n,3,2) y,z
        a = 0.5 * ((T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1])
                   - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1]))
        T = T[np.abs(a) > 1e-9]
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
            ylo = np.where(ok, yv, np.inf).min(1)
            yhi = np.where(ok, yv, -np.inf).max(1)
            i0 = np.searchsorted(ys, ylo)
            i1 = np.searchsorted(ys, yhi)
            for r, (u, v) in enumerate(zip(i0, i1)):
                if v > u:
                    mask[j0 + r, u:v] = True
    return mask.sum() * h * h * 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    ap.add_argument("--scale", type=float, default=0.001, help="mm -> output unit (default m)")
    ap.add_argument("--res", type=float, default=1.0, help="mesh resolution multiplier")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="morph parameters, e.g. --set roof_rear_drop=30 tail_taper=40")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    set_design({k: float(v) for k, v in (a.split("=") for a in args.set)})
    errs = validate_design()
    if errs:
        raise SystemExit("invalid design: " + "; ".join(errs))
    parts = build_all(args.res)
    print(f"{'part':10s} {'tris':>8s} {'volume[m3]':>11s}  watertight-check")
    for name, V, F in parts:
        chk = check_watertight(V, F)
        print(f"{name:10s} {len(F):8d} {signed_volume(V, F)*1e-9:11.4f}  {chk}")

    scaled = [(n, V * args.scale, F) for n, V, F in parts]
    for n, V, F in scaled:
        write_stl_binary(os.path.join(args.out, f"ioniq5_{n}.stl"), V, F, n)
    write_stl_ascii_multi(os.path.join(args.out, "ioniq5_full_multisolid.stl"), scaled)

    allV = np.vstack([V for _, V, _ in parts])
    lo, hi = allV.min(0), allV.max(0)
    print("bbox [mm] min", lo.round(1), "max", hi.round(1), "size", (hi - lo).round(1))


if __name__ == "__main__":
    main()
