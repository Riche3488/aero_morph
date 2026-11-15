"""High-resolution smooth refinement of an (already roughly symmetric) car mesh.

    .venv/bin/python smooth_refine.py output/ioniq5_sym.stl vehicles/ioniq5.yaml --tag ioniq5_smooth

Unlike car3d/repair.py (binary voxel solid -> EDT -> terraced, coarse), this builds an *exact*
signed distance field from the input surface:
  sign      = generalised winding number (robust to holes / non-manifold patches)
  magnitude = exact point-to-triangle distance in a narrow band (EDT further out)
and then fixes known defect classes directly on the SDF:
  1. tyres      : material outside the tyre circle below the axle is cut (contact-patch "feet"),
                  then a small flat contact patch is made (CFD friendly, no zero-angle wedge)
  2. upper body : dents / pits (e.g. windshield) and narrow grooves filled by an SDF closing
  3. sills      : vertical ribs between the wheel arches replaced by an x-extrusion (median)
  4. underbody  : lumps / steps smoothed via a smoothed bottom height map (>= floor height)
  5. plates     : licence-plate holes (front / rear; winding-number membrane -> proud blob with
                  side notches) re-created from the surrounding bumper by tension-spline inpainting
  6. windshield : ~0.4 m trough between two transverse ridges on the glass replaced by a fitted
                  low-order (mirror symmetric) surface; pits at the A-pillar feet filled
  7. everywhere : stair-step ripples removed by SDF Gaussian + Taubin smoothing
The result is exactly mirror symmetric (half-space SDF mirrored, half mesh decimated + welded),
watertight, and fitted to the spec L/W/H and axle positions.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from car3d.repair import decimate_half, detect_axles, find_symmetry, fit_spec, warp_axles  # noqa: E402
from car3d.spec import load_config  # noqa: E402

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------- input
def load_input(path, spec):
    m = trimesh.load(path, force="mesh", process=True)
    parts = m.split(only_watertight=False)
    big = max(len(p.faces) for p in parts)
    m = trimesh.util.concatenate([p for p in parts if len(p.faces) > 0.01 * big])
    log(f"input {len(m.faces)} faces ({len(parts)} components -> kept {sum(len(p.faces) > 0.01 * big for p in parts)})")
    M, sym = find_symmetry(m)
    m.apply_transform(M)
    m.apply_translation([-m.bounds[0][0], 0, -m.bounds[0][2]])
    axles = detect_axles(m, spec)
    m, warp = warp_axles(m, axles, spec)
    m, fit = fit_spec(m, spec)
    log(f"fitted extents {np.round(m.extents * 1000)} mm")
    return m, dict(symmetry=sym, axle_warp=warp, pre_fit=fit, tyre_contact_x_m=[round(a, 4) for a in axles])


# ----------------------------------------------------------------------------- exact SDF
def grid(m, h, pad=12):
    lo, hi = m.bounds
    ny = int(np.ceil(max(abs(lo[1]), abs(hi[1])) / h)) + pad
    xs = np.arange(lo[0] - pad * h, hi[0] + pad * h, h)
    ys = (np.arange(-ny, ny) + 0.5) * h          # y centres at +-(k+.5)h -> exact mirror pairs
    zs = np.arange(-pad * h, hi[2] + pad * h, h)
    return xs, ys, zs


def exact_sdf(m, xs, ys, zs, h, band_vox=6):
    """Half grid (y>0) -> winding-number sign + exact distance in a narrow band -> mirrored."""
    import igl
    import open3d as o3d
    yh = ys[ys > 0]
    shape = (len(xs), len(yh), len(zs))
    V = np.ascontiguousarray(m.vertices, dtype=np.float64)
    F = np.ascontiguousarray(m.faces, dtype=np.int64)
    inside = np.zeros(shape, bool)
    for i0 in range(0, len(xs), 64):
        X, Y, Z = np.meshgrid(xs[i0:i0 + 64], yh, zs, indexing="ij")
        Q = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
        inside[i0:i0 + 64] = (igl.fast_winding_number(V, F, Q) > 0.5).reshape(X.shape)
    log(f"winding number on {np.prod(shape) / 1e6:.0f} M half-grid nodes, inside volume x2 = {2 * inside.sum() * h ** 3:.2f} m^3")
    # keep the main body (+ big attached parts); drop floating crumbs
    lab, n = ndimage.label(inside)
    if n > 1:
        sz = ndimage.sum(np.ones_like(lab, np.int8), lab, range(1, n + 1))
        inside = np.isin(lab, np.where(sz > 0.002 * sz.max())[0] + 1)
        log(f"inside components {n} -> {int((sz > 0.002 * sz.max()).sum())}")
    del lab
    full_in = np.concatenate([inside[:, ::-1], inside], axis=1)
    d_out = ndimage.distance_transform_edt(~full_in).astype(np.float32)
    d_in = ndimage.distance_transform_edt(full_in).astype(np.float32)
    sdf = (d_out - d_in) * h                       # coarse, ~voxel accurate, used beyond the band
    del d_out, d_in, full_in
    half = sdf[:, len(yh):]                        # view on y>0
    band = np.abs(half) < band_vox * h
    idx = np.argwhere(band)
    P = np.c_[xs[idx[:, 0]], yh[idx[:, 1]], zs[idx[:, 2]]].astype(np.float32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    d = scene.compute_distance(o3d.core.Tensor(P)).numpy()
    s = np.sign(half[band]); s[s == 0] = 1
    # exact unsigned distance, but never let it exceed the EDT estimate's consistency range
    half[band] = s * np.minimum(d, band_vox * h)
    sdf[:, :len(yh)] = half[:, ::-1]
    log(f"exact distance on {len(P) / 1e6:.1f} M band nodes")
    return sdf


# ----------------------------------------------------------------------------- helpers
def edt_sdf(solid, h):
    return (ndimage.distance_transform_edt(~solid).astype(np.float32)
            - ndimage.distance_transform_edt(solid).astype(np.float32)) * h


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def tyre_y(spec, ax):
    return (spec.track_front if ax < 0.5 * spec.length else spec.track_rear) / 2


def fit_tyre(sdf, xs, ys, zs, h, ax, spec):
    """Tyre circle (cx, cz, R) from the side silhouette of the tyre band (holes between spokes
    filled): outer edge along rays from the centre, bottom (contact blobs) and top (arch) ignored,
    robust least squares. Falls back to the spec tyre if the fit is implausible."""
    yt = tyre_y(spec, ax)
    jy = np.where(np.abs(np.abs(ys) - yt) < spec.tire_width / 2 - 0.02)[0]
    jy = jy[ys[jy] > 0]
    sil = ndimage.binary_fill_holes((sdf[:, jy, :] < 0).any(axis=1))
    sl = ndimage.gaussian_filter(np.where(sil, -1.0, 1.0).astype(np.float32), 0.7)
    cx, cz, R = ax, spec.tire_radius, spec.tire_radius
    ths = np.radians(np.r_[np.arange(-160, -114, 3), np.arange(-66, 21, 3), np.arange(159, 201, 3)])
    for it in range(6):
        P = []
        for th in ths:
            r = np.arange(0.6 * R, 1.3 * R, h / 4)
            px, pz = cx + r * np.cos(th), cz + r * np.sin(th)
            v = ndimage.map_coordinates(sl, [(px - xs[0]) / h, (pz - zs[0]) / h], order=1, mode="nearest")
            k = np.argmax(v > 0)
            if k > 0 and v[0] < 0:
                rr = r[k - 1] + (r[k] - r[k - 1]) * (-v[k - 1]) / (v[k] - v[k - 1])
                P.append((cx + rr * np.cos(th), cz + rr * np.sin(th)))
        P = np.array(P)
        keep = np.ones(len(P), bool)
        for _ in range(4):
            A = np.c_[2 * P[keep], np.ones(keep.sum())]
            sol = np.linalg.lstsq(A, (P[keep] ** 2).sum(1), rcond=None)[0]
            c = sol[:2]; Rf = np.sqrt(sol[2] + c @ c)
            res = np.abs(np.hypot(*(P - c).T) - Rf)
            keep = res < max(0.006, 2.5 * np.median(res))
        cx, cz, R = float(c[0]), float(c[1]), float(Rf)
    p90 = float(np.percentile(res[keep], 90))
    if abs(R - spec.tire_radius) > 0.12 * spec.tire_radius or abs(cx - ax) > 0.1:
        log(f"WARNING: tyre fit @x={ax:.3f} implausible (R={R:.3f}, cx={cx:.3f}); using the spec tyre")
        cx, cz, R = ax, spec.tire_radius, spec.tire_radius
    return cx, cz, R, p90


# ----------------------------------------------------------------------------- defect fixes
def fill_voids(sdf):
    solid = sdf < 0
    filled = ndimage.binary_fill_holes(solid)
    n = int((filled & ~solid).sum())
    sdf[filled & ~solid] = -np.abs(sdf[filled & ~solid]) - 1e-4
    log(f"internal voids filled: {n} voxels")
    return n


def fix_tyres(sdf, xs, ys, zs, h, spec, patch=0.008):
    """Cut everything outside the fitted tyre circle in the bottom sector (contact-patch blobs),
    then flatten the bottom `patch` metres -> flat contact patch at the new ground."""
    X = xs[:, None, None]; Y = np.abs(ys)[None, :, None]; Z = zs[None, None, :]
    out, zg = [], []
    for ax in (spec.axle_f, spec.axle_r):
        cx, cz, R, res = fit_tyre(sdf, xs, ys, zs, h, ax, spec)
        yt = tyre_y(spec, ax)
        ix = np.where(np.abs(xs - cx) < R)[0]; iz = np.where(zs < cz - 0.6 * R)[0]
        iy = np.where(np.abs(np.abs(ys) - yt) < spec.tire_width / 2 + 0.05)[0]
        sub = np.ix_(ix, iy, iz)
        r = np.hypot(X[ix] - cx, Z[:, :, iz] - cz)            # (nx,1,nz)
        before = (sdf[sub] < 0).sum()
        sdf[sub] = np.maximum(sdf[sub], np.broadcast_to(r - R, sdf[sub].shape))
        removed = int(before - (sdf[sub] < 0).sum())
        out.append(dict(axle_x=round(cx, 4), centre_z=round(cz, 4), radius=round(R, 4),
                        fit_residual_p90_mm=round(res * 1000, 2), blob_voxels_removed=removed))
        zg.append(cz - R)
        log(f"tyre @x={cx:.3f}: centre z {cz:.3f}, R {R * 1000:.0f} mm (fit p90 {res * 1000:.1f} mm), "
            f"blob voxels removed {removed}")
    zc = max(zg) + patch                                        # flat contact patch for both axles
    k = zs < zc
    sdf[:, :, k] = np.maximum(sdf[:, :, k], (zc - zs[k])[None, None, :])
    w = 2 * np.sqrt(2 * np.mean([o["radius"] for o in out]) * patch)
    log(f"contact patch: ground moved to z={zc:.3f} (patch length ~{w * 1000:.0f} mm)")
    return dict(tyres=out, ground_z=round(float(zc), 4), contact_patch_length_mm=round(w * 1000))


def keepout(xs, ys, zs, spec, tyres, mirror_box=(1.35, 1.95, 0.78, 1.35), spoiler=(4.2, 1.3)):
    """Weight 1 where gap-filling operations are allowed: not in the wheel / arch zones (the tyre-
    arch gap must stay open), not around the mirrors (stalk gap), not under the roof-spoiler lip."""
    X = xs[:, None, None]; Y = np.abs(ys)[None, :, None]; Z = zs[None, None, :]
    w = np.ones(np.broadcast_shapes(X.shape, Y.shape, Z.shape), np.float32)
    for t in tyres:
        r = np.hypot(X - t["axle_x"], Z - t["centre_z"])
        w *= 1 - smoothstep((t["radius"] + 0.22 - r) / 0.06) * smoothstep((Y - 0.40) / 0.05)
    mx0, mx1, my, mz = mirror_box
    w *= 1 - smoothstep((X - mx0) / 0.08) * smoothstep((mx1 - X) / 0.08) * smoothstep((Y - my) / 0.04) \
        * smoothstep((mz - Z) / 0.05)
    w *= 1 - smoothstep((X - spoiler[0]) / 0.08) * smoothstep((Z - spoiler[1]) / 0.05)
    w *= smoothstep((Z - 0.22) / 0.05)          # underbody is handled separately
    return w


def close_body(sdf, xs, ys, zs, h, spec, tyres, r=0.04):
    """SDF closing (dilate r, erode r): fills pits and grooves narrower than 2r (bumper holes,
    door-line grooves, cowl notch) everywhere outside the keep-out zones."""
    dil = sdf < r
    closed = r - ndimage.distance_transform_edt(dil).astype(np.float32) * h
    closed = np.where(dil, closed, sdf)
    del dil
    closed = ndimage.gaussian_filter(closed, 1.0)
    w = keepout(xs, ys, zs, spec, tyres)
    delta = w * np.maximum(sdf - closed, 0)
    del closed, w
    filled = (sdf >= 0) & (sdf - delta < 0)
    depth = float(delta[filled].max()) if filled.any() else 0.0
    sdf -= delta
    log(f"body closing r={r * 1000:.0f} mm: {filled.sum()} voxels filled ({filled.sum() * h ** 3 * 1000:.1f} L, "
        f"max depth {depth * 1000:.1f} mm)")
    return dict(radius_mm=r * 1000, filled_litres=round(float(filled.sum() * h ** 3 * 1000), 2), max_depth_mm=round(depth * 1000, 1))


def fill_ledges(sdf, xs, ys, zs, h, axis, outward, r, region2d, max_tilt=60, trend_sigma=0.12, depth=0.06):
    """Outer height map seen along `axis` -> grey-scale closing of its residual about a smooth
    trend (flat-SE closing alone leaves any monotone slope, and so a ledge on an inclined glass,
    unchanged) -> fill only. Removes steps / pits / pockets narrower than 2r, keeps convex shapes."""
    coords = (xs, ys, zs)[axis]
    H = surface_height(sdf, coords, axis, outward)
    has = np.isfinite(H)
    Ho = np.where(has, H * outward, -1e3)                         # "height" increasing outward
    hf = has.astype(np.float32)
    sg = trend_sigma / h
    trend = ndimage.gaussian_filter(np.where(has, Ho, 0), sg) / np.maximum(ndimage.gaussian_filter(hf, sg), 1e-6)
    res = np.where(has, Ho - trend, -1.0)
    k = int(round(r / h)); yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    lift = ndimage.grey_closing(res, footprint=xx ** 2 + yy ** 2 <= k * k) - res
    g = np.gradient(ndimage.gaussian_filter(np.where(has, Ho, 0), 2), h)
    w2 = ndimage.gaussian_filter((has & (np.degrees(np.arctan(np.hypot(*g))) < max_tilt)).astype(np.float32), 1.5)
    w2 *= region2d.astype(np.float32)
    lift = np.where(has, np.maximum(lift, 0) * w2, 0)
    lift = np.where(lift > 0.5 * h, lift, 0)
    shape2 = [1, 1, 1]; shape2[axis] = -1
    C = coords.reshape(shape2) * outward                          # outward coordinate
    Ho3, Hn3 = np.expand_dims(Ho, axis), np.expand_dims(Ho + lift, axis)
    slab = np.maximum(C - Hn3, (Ho3 - depth) - C)                 # >0 outside the fill slab
    m3 = np.expand_dims(lift > 0, axis) & (slab < np.abs(sdf) + 0.02)
    np.minimum(sdf, np.where(m3, slab, np.inf), out=sdf)
    m2 = lift > 0
    log(f"ledge fill axis={'xyz'[axis]}{'+' if outward > 0 else '-'} r={r * 1000:.0f} mm: {m2.sum() * h * h:.3f} m^2 lifted, "
        f"max {lift.max() * 1000:.1f} mm, mean {lift[m2].mean() * 1000 if m2.any() else 0:.1f} mm")
    return dict(area_m2=round(float(m2.sum() * h * h), 3), max_lift_mm=round(float(lift.max()) * 1000, 1))


def hole_region(sdf, xs, ys, zs, h, axis, outward, mesh, gap=0.006, grow=0.06):
    """2-D mask (in the height-map plane) of surface patches that do not come from the input
    surface: the winding-number field spans open holes there (bumper grille / plate openings)."""
    import open3d as o3d
    coords = (xs, ys, zs)[axis]
    H = surface_height(sdf, coords, axis, outward)
    ok = np.isfinite(H)
    other = [c for i, c in enumerate((xs, ys, zs)) if i != axis]
    U, V = np.meshgrid(*other, indexing="ij")
    P = np.zeros((ok.sum(), 3), np.float32)
    oi = [i for i in range(3) if i != axis]
    P[:, oi[0]], P[:, oi[1]], P[:, axis] = U[ok], V[ok], H[ok]
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(mesh.vertices.astype(np.float32)), o3d.core.Tensor(mesh.faces.astype(np.uint32)))
    d = np.zeros(H.shape, np.float32); d[ok] = sc.compute_distance(o3d.core.Tensor(P)).numpy()
    hole = ok & (d > gap)
    hole = ndimage.binary_opening(hole, iterations=2)
    reg = ndimage.binary_dilation(hole, iterations=int(grow / h))
    return np.clip(ndimage.gaussian_filter(reg.astype(np.float32), 0.02 / h) * 1.6, 0, 1), float(hole.sum() * h * h)


def fill_all_ledges(sdf, xs, ys, zs, h, spec, tyres, mesh, r_top=0.09, r_end=0.12, exclude=None):
    out = {}
    # top: glass / hood / roof field between the shoulders (the shoulder-to-window junction,
    # mirror bases and the roof-spoiler joint are design concavities, not defects)
    Y = np.abs(ys)
    reg = smoothstep((0.68 - Y)[None, :] / 0.05) * smoothstep((4.1 - xs)[:, None] / 0.08)
    out["top"] = fill_ledges(sdf, xs, ys, zs, h, 2, +1, r_top, reg)
    # front / rear fascia: only around patches that span holes of the input (grille / plate
    # openings) -> the pockets they leave are filled, the real fascia shape is untouched
    for nm, outward in (("front", -1), ("rear", +1)):
        reg, area = hole_region(sdf, xs, ys, zs, h, 0, outward, mesh)
        if exclude is not None and nm in exclude:                  # re-created by smooth_plates
            reg = reg * (1 - exclude[nm])
        log(f"{nm}: input-hole patches {area:.3f} m^2")
        out[nm] = fill_ledges(sdf, xs, ys, zs, h, 0, outward, r_end, reg)
        out[nm]["hole_patch_m2"] = round(area, 3)
    return out


def displace_along(sdf, coords, h, axis, outward, H, d, band):
    """Move the outer surface (height map H seen along `axis`) outward by d (may be negative):
    sdf -= d * dsdf/daxis in a band around the surface. Valid for |d| << band."""
    grad = np.gradient(sdf, h, axis=axis)
    shape2 = [1, 1, 1]; shape2[axis] = -1
    C = coords.reshape(shape2)
    Hx = np.expand_dims(np.where(np.isfinite(H), H, np.inf), axis)
    fade = smoothstep((band - np.abs(C - Hx)) / (0.5 * band))
    sdf -= outward * np.expand_dims(d.astype(np.float32), axis) * grad * fade


def detect_windshield(H, xs, ys, h, spec):
    """Windshield x-range on the centreline of the top height map: hood -> glass where the slope
    first exceeds 0.3, glass -> roof where it falls back below 0.15."""
    j0 = np.argsort(np.abs(ys))[:2]
    zc = ndimage.gaussian_filter1d(np.nan_to_num(H[:, j0].mean(axis=1)), 4)
    sl = np.gradient(zc, h)
    i = np.where((xs > 0.2 * spec.length) & (sl > 0.3))[0][0]
    i2 = i + np.where(sl[i:] < 0.15)[0][0]
    return float(xs[i]), float(xs[i2])


def fair_windshield(sdf, xs, ys, zs, h, spec, ym=0.6, deg_x=6, band=0.03):
    """Windshield + front roof: the generated glass carries a ~0.4 m wavelength trough (dish
    between two transverse ridges) that the ripple / closing filters keep. Fit a low-order surface
    z = sum c_ij u^i (y/ym)^2j (mirror symmetric) to the top height map over glass + front roof and
    move the glass onto it (fill and cut), blended out towards cowl, roof and A-pillars."""
    H = surface_height(sdf, zs, 2, +1)
    xa, xb = detect_windshield(H, xs, ys, h, spec)
    x0, x1 = xa - 0.05, xb + 0.7
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    reg = (X > x0) & (X < x1) & (np.abs(Y) < ym) & np.isfinite(H)
    u = (X - x0) / (x1 - x0)
    A = lambda u_, y_: np.stack([u_ ** i * (y_ / ym) ** (2 * j) for i in range(deg_x + 1) for j in range(3)], -1)
    c = np.linalg.lstsq(A(u[reg], Y[reg]), H[reg], rcond=None)[0]
    T = A(u, Y) @ c
    w = smoothstep((X - (xa + 0.08)) / 0.12) * smoothstep((xb + 0.4 - X) / 0.15) * smoothstep((ym - 0.04 - np.abs(Y)) / 0.08)
    d = np.where(reg, w * (T - H), 0)
    res_before = float(np.std((H - T)[reg & (w > 0.9)]))
    displace_along(sdf, zs, h, 2, +1, H, d, band)
    log(f"windshield x {xa:.3f}..{xb:.3f}: refit to smooth surface, residual std {res_before * 1000:.1f} mm, "
        f"shift {d.min() * 1000:.1f}..{d.max() * 1000:.1f} mm")
    return dict(x_range=[round(xa, 3), round(xb, 3)], residual_std_mm=round(res_before * 1000, 2),
                shift_mm=[round(float(d.min()) * 1000, 1), round(float(d.max()) * 1000, 1)]), (xa, xb)


def fill_pillar_pits(sdf, xs, ys, zs, h, xa, r=0.12):
    """Pits at the A-pillar foot (glass corner / cowl end): ledge filling with a wide disk, only
    in the two corner zones."""
    Y = np.abs(ys)[None, :]; X = xs[:, None]
    reg = smoothstep((X - (xa + 0.05)) / 0.08) * smoothstep((xa + 0.85 - X) / 0.08) \
        * smoothstep((Y - 0.42) / 0.05) * smoothstep((0.74 - Y) / 0.03)
    return fill_ledges(sdf, xs, ys, zs, h, 2, +1, r, reg, max_tilt=70, trend_sigma=0.15)


def plate_holes(sdf, xs, ys, zs, h, mesh, max_area=0.15):
    """Central holes of the input surface in the front / rear fascia (licence-plate recesses),
    taken on the *base* SDF (later fills move the surface off the input and would inflate them)."""
    out = {}
    for nm, outward in (("front", -1), ("rear", +1)):
        soft, _ = hole_region(sdf, xs, ys, zs, h, 0, outward, mesh, grow=h)
        lab, n = ndimage.label(soft > 0.5)
        Yc = np.abs(ys)[:, None] * np.ones(lab.shape)
        keep = [k for k in range(1, n + 1)
                if Yc[lab == k].min() < 0.05 and (lab == k).sum() * h * h < max_area]
        out[nm] = np.isin(lab, keep)
        log(f"plate hole {nm}: {out[nm].sum() * h * h:.3f} m^2")
    return out


def inpaint_smooth2d(img, mask, order=1, tension=0.0, valid=None):
    """Fill img[mask] from the known pixels: order 1 = harmonic (Lap u = 0; bounded by the
    boundary values, never bulges), 2 = biharmonic (C1, but extrapolates boundary slopes);
    order 2 with tension t solves (Lap^2 - t Lap) u = 0: boundary slopes decay over ~1/sqrt(t) px.
    5-point Laplacian; window edges and pixels outside `valid` act as Neumann (no-flux) boundaries,
    so only valid known pixels set the fill. Sparse direct solve."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    ny, nz = img.shape
    idx = np.arange(ny * nz).reshape(ny, nz)
    vf = np.ones(ny * nz, bool) if valid is None else (valid | mask).ravel()
    rows, cols, vals = [], [], []
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        a = idx[max(di, 0):ny + min(di, 0), max(dj, 0):nz + min(dj, 0)].ravel()
        b = idx[max(-di, 0):ny + min(-di, 0), max(-dj, 0):nz + min(-dj, 0)].ravel()
        e = vf[a] & vf[b]; a, b = a[e], b[e]
        rows += [a, a]; cols += [b, a]; vals += [np.ones(len(a)), -np.ones(len(a))]
    L = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ny * nz, ny * nz))
    A = (L @ L - tension * L).tocsr() if order == 2 else L
    u = mask.ravel(); k = ~u
    x = img.ravel().astype(np.float64).copy()
    x[u] = spsolve(A[u][:, u].tocsc(), -A[u][:, k] @ np.nan_to_num(x[k]))
    return x.reshape(ny, nz)


def smooth_plates(sdf, xs, ys, zs, h, holes, grow=(0.14, 0.03), fade=0.04, depth=0.10, max_dev=0.06, tension=0.05):
    """Licence-plate recesses (front / rear): TRELLIS leaves an open hole there, the winding number
    spans it with a flat membrane, and the later filters turn that into a blob with notches at its
    sides. The input has no real surface there, so the fascia height map is re-created inside the
    hole (+grow) by biharmonic inpainting from the surrounding bumper, and the SDF is rebuilt
    from it (fill and cut)."""
    out, excl = {}, {}
    for nm, outward in (("front", -1), ("rear", +1)):
        hole = holes[nm]
        if not hole.any():
            out[nm] = dict(area_m2=0.0); continue
        H = surface_height(sdf, xs, 0, outward)                      # (ny, nz)
        # grow mostly sideways: the membrane's side notches reach ~10 cm beyond the hole in y,
        # while above / below the bumper curls away within a few cm
        gy, gz = int(grow[0] / h), int(grow[1] / h)
        se = ((np.arange(-gy, gy + 1)[:, None] / max(gy, 1)) ** 2 + (np.arange(-gz, gz + 1)[None, :] / max(gz, 1)) ** 2) <= 1
        grown = ndimage.binary_dilation(hole, structure=se)
        ring = ndimage.binary_dilation(grown, iterations=int(0.03 / h)) & ~grown
        ref = np.nanmedian((H * outward)[ring])                      # surrounding bumper level
        near = np.abs(H * outward - ref) < max_dev
        unk = grown & near
        lab, _ = ndimage.label(unk | hole)                           # the part connected to the hole
        unk = np.isin(lab, np.unique(lab[hole])) & (unk | hole)       # the membrane itself always
        # inpaint over unk + a fade band, then blend: features ending at the hole (bumper grooves)
        # fade out over `fade` instead of stopping abruptly (which would leave a notch)
        dist = ndimage.distance_transform_edt(~unk) * h
        big = ((dist < fade) & near) | unk
        w = np.where(unk, 1.0, np.where(big, smoothstep(1 - dist / fade) * smoothstep((max_dev - np.abs(H * outward - ref)) / 0.02), 0)).astype(np.float32)
        I, J = np.where(ndimage.binary_dilation(big, iterations=int(0.12 / h)))
        sl = (slice(I.min(), I.max() + 1), slice(J.min(), J.max() + 1))
        Ho = H * outward
        win = Ho[sl]
        if not np.isfinite(win[~big[sl]]).all():
            win = np.where(np.isfinite(win), win, np.nanmedian(win))
        new = inpaint_smooth2d(win, big[sl], order=2, tension=tension, valid=(near | big)[sl])
        Hn = Ho.copy(); Hn[sl] = np.where(big[sl], new, win)
        Hn = np.where(np.isfinite(Ho), Ho + w * (Hn - Ho), Ho)
        ww = np.where(np.isfinite(Ho), w, 0)
        act = ww > 0
        lo_, hi_ = np.nanmin(Hn[act]) - depth, np.nanmax(np.maximum(Hn, Ho)[act]) + 0.03
        ix = np.where((xs * outward > lo_) & (xs * outward < hi_))[0]
        C = xs[ix][:, None, None] * outward                          # outward coordinate
        Hn3 = np.where(act, Hn, 0)[None]; ww3 = ww[None].astype(np.float32)
        sub = sdf[ix]
        m3 = act[None] & (C > Hn3 - depth)
        sub[m3] = (ww3 * (C - Hn3) + (1 - ww3) * sub)[m3]
        sdf[ix] = sub
        ch = np.abs(Hn - Ho)[unk]
        excl[nm] = np.clip(ndimage.gaussian_filter(ndimage.binary_dilation(ww > 0, iterations=int(0.04 / h)).astype(np.float32), 0.01 / h) * 2, 0, 1)
        out[nm] = dict(area_m2=round(float(unk.sum() * h * h), 3), mean_change_mm=round(float(ch.mean()) * 1000, 1),
                       max_change_mm=round(float(ch.max()) * 1000, 1))
        log(f"plate {nm}: hole {hole.sum() * h * h:.3f} m^2, inpainted {unk.sum() * h * h:.3f} m^2, "
            f"|change| mean {ch.mean() * 1000:.1f} mm, max {ch.max() * 1000:.1f} mm")
    return out, excl


def surface_height(sdf, coords, axis, outward):
    """Sub-voxel height of the outermost surface seen from `outward` (+1/-1) along `axis`."""
    a = np.moveaxis(sdf, axis, -1)
    if outward > 0:
        a = a[..., ::-1]
    c = coords if outward < 0 else coords[::-1]
    ins = a < 0
    has = ins.any(axis=-1)
    k = np.argmax(ins, axis=-1)
    k1 = np.clip(k - 1, 0, a.shape[-1] - 1)
    s0 = np.take_along_axis(a, k1[..., None], -1)[..., 0]
    s1 = np.take_along_axis(a, k[..., None], -1)[..., 0]
    t = np.clip(s0 / np.maximum(s0 - s1, 1e-9), 0, 1)
    H = c[k1] + t * (c[k] - c[k1])
    return np.where(has & (k > 0), H, np.nan)


def panel_smooth(sdf, xs, ys, zs, h, axis, outward, sigma=0.06, keep_dev=0.004, max_tilt=50, band=0.03, region=None):
    """Remove low-amplitude terraces on a family of panels: outer height map along `axis` ->
    masked Gaussian (sigma); points deviating < keep_dev from it are ripples and are moved onto the
    smooth surface, larger deviations (edges, fin, creases, lamps) are design features and kept.
    The SDF is displaced locally:  sdf -= d * dsdf/daxis  in a band around the surface."""
    coords = (xs, ys, zs)[axis]
    H = surface_height(sdf, coords, axis, outward)
    valid = np.isfinite(H)
    Hf = np.where(valid, H, 0)
    g = np.gradient(ndimage.gaussian_filter(Hf, 2), h)
    tilt = np.degrees(np.arctan(np.hypot(*g)))
    ok = valid & (tilt < max_tilt)
    ok = ndimage.binary_erosion(ok, iterations=3)
    if region is not None:
        ok &= region
    sg = sigma / h
    T = ndimage.gaussian_filter(np.where(ok, Hf, 0), sg) / np.maximum(ndimage.gaussian_filter(ok.astype(np.float32), sg), 1e-6)
    dev = np.where(ok, T - H, 0)
    w = smoothstep((keep_dev - np.abs(dev)) / (0.5 * keep_dev)) * ndimage.gaussian_filter(ok.astype(np.float32), 1.0)
    w = ndimage.minimum_filter(w, size=3)            # features: also leave their immediate rim alone
    d = (np.where(ok, dev, 0) * w).astype(np.float32)
    grad = np.gradient(sdf, h, axis=axis)
    shape2 = [1, 1, 1]; shape2[axis] = -1
    C = coords.reshape(shape2)
    Hx = np.expand_dims(np.where(ok, H, np.inf), axis)
    fade = smoothstep((band - np.abs(C - Hx)) / (0.5 * band))
    dd = np.expand_dims(d, axis)
    sdf -= outward * dd * grad * fade
    m = ok & (w > 0.5)
    log(f"panel smoothing axis={'xyz'[axis]}{'+' if outward > 0 else '-'}: {m.sum() * h * h:.2f} m^2 smoothed, "
        f"|shift| mean {np.abs(d[m]).mean() * 1000 if m.any() else 0:.2f} mm, max {np.abs(d).max() * 1000:.1f} mm")
    return dict(area_m2=round(float(m.sum() * h * h), 2), mean_shift_mm=round(float(np.abs(d[m]).mean()) * 1000 if m.any() else 0, 2))


def adaptive_smooth(sdf, h, s_light=1.2, s_heavy=3.0, k0=6.0, k1=25.0):
    """Curvature-adaptive SDF smoothing: heavy Gaussian on flat-ish panels (kills stair-step
    ripples), light Gaussian at creases / small features (|mean curvature| from the heavy field)."""
    heavy = ndimage.gaussian_filter(sdf, s_heavy)
    kappa = np.abs(ndimage.laplace(heavy)) / (h * h)            # ~ |k1 + k2| [1/m]
    kappa = ndimage.maximum_filter(kappa, size=3)
    w = smoothstep((kappa - k0) / (k1 - k0)).astype(np.float32)  # 1 -> light
    del kappa
    light = ndimage.gaussian_filter(sdf, s_light)
    near = np.abs(heavy) < 0.02
    log(f"adaptive smoothing: light share near surface {w[near].mean():.2f}")
    out = heavy + w * (light - heavy)
    return out


def extrude_sills(sdf, xs, ys, zs, h, spec, tyres, z1=0.55, ymin=0.55):
    """Between the wheel arches, the sill (z<z1, |y|>ymin) is replaced by its x-median profile."""
    xa = tyres[0]["axle_x"] + tyres[0]["radius"] + 0.12
    xb = tyres[1]["axle_x"] - tyres[1]["radius"] - 0.12
    ix = np.where((xs > xa) & (xs < xb))[0]
    core = (xs[ix] > xa + 0.1) & (xs[ix] < xb - 0.1)
    iy = np.where(np.abs(ys) > ymin)[0]; iz = np.where(zs < z1 + 0.1)[0]
    sub = sdf[np.ix_(ix, iy, iz)]
    med = np.median(sub[core], axis=0)                          # (ny, nz)
    X = xs[ix][:, None, None]; Y = np.abs(ys[iy])[None, :, None]; Z = zs[iz][None, None, :]
    w = smoothstep((X - xa) / 0.1) * smoothstep((xb - X) / 0.1) * smoothstep((z1 + 0.08 - Z) / 0.08) \
        * smoothstep((Y - ymin) / 0.05)
    new = (1 - w) * sub + w * med[None]
    dev = np.abs(new - sub)[np.abs(sub) < 0.01]
    sdf[np.ix_(ix, iy, iz)] = new
    log(f"sill extrusion x {xa:.2f}..{xb:.2f}: mean |change| near surface {dev.mean() * 1000:.1f} mm, max {dev.max() * 1000:.1f} mm")
    return dict(x_range=[round(xa, 3), round(xb, 3)], mean_change_mm=round(float(dev.mean()) * 1000, 2))


def smooth_underbody(sdf, xs, ys, zs, h, spec, tyres, sigma=0.09, thick=0.25):
    """Bottom height map of the body footprint -> masked Gaussian (>= floor) -> rebuilt underside.
    Removes lumps, and turns the flat-floor end steps into smooth ramps."""
    solid = sdf < 0
    kz = np.where(zs < 0.8)[0]
    s = solid[:, :, kz]
    has = s.any(axis=2)
    zb = np.where(has, zs[kz][np.argmax(s, axis=2)], np.nan)  # lowest solid z per column
    # "thick" columns: solid continuously from zb to zb+thick (body, not a lip / arch / tyre)
    nth = int(thick / h)
    first = np.argmax(s, axis=2)
    idx = np.clip(first[..., None] + np.arange(nth)[None, None, :], 0, s.shape[2] - 1)
    run = np.take_along_axis(s, idx, axis=2).all(axis=2)
    del idx
    X = xs[:, None]; Y = np.abs(ys)[None, :]
    wheel = np.zeros(has.shape, bool)
    for t in tyres:
        yt = tyre_y(spec, t["axle_x"])
        wheel |= (np.abs(X - t["axle_x"]) < t["radius"] + 0.15) & (Y > yt - spec.tire_width / 2 - 0.12)
    foot = ndimage.binary_fill_holes(ndimage.binary_closing(has & run, iterations=3)) & ~wheel
    foot = ndimage.binary_erosion(foot, iterations=int(0.03 / h))
    floor_z = float(np.percentile(zb[foot], 1))
    sg = sigma / h
    wsum = ndimage.gaussian_filter(foot.astype(np.float32), sg)
    zt = ndimage.gaussian_filter(np.where(foot, zb, 0).astype(np.float32), sg) / np.maximum(wsum, 1e-6)
    zt = np.maximum(zt, floor_z)
    wcol = np.clip(ndimage.gaussian_filter(foot.astype(np.float32), 0.02 / h) * 2 - 1, 0, 1)  # inside -> 1
    wcol = smoothstep(wcol)
    Z = zs[kz][None, None, :]
    plane = zt[..., None] - Z                                    # >0 below the target underside
    wz = smoothstep((zt[..., None] + 0.16 - Z) / 0.06)           # 1 near the underside, 0 above +16 cm
    w = (wcol[..., None] * wz).astype(np.float32)
    sub = sdf[:, :, kz]
    new = (1 - w) * sub + w * plane
    ch = np.abs(np.where(foot, zt - zb, 0))
    sdf[:, :, kz] = new
    log(f"underbody: footprint {foot.sum() * h * h:.2f} m^2, |height change| mean {ch[foot].mean() * 1000:.1f} mm, "
        f"max {ch[foot].max() * 1000:.0f} mm")
    return dict(floor_z=round(floor_z, 4), footprint_m2=round(float(foot.sum() * h * h), 2), mean_change_mm=round(float(ch[foot].mean()) * 1000, 1),
                max_change_mm=round(float(ch[foot].max()) * 1000))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("config")
    ap.add_argument("--tag", default="ioniq5_smooth")
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--cache", default=None, help="npz cache of the base SDF (speeds up re-runs)")
    ap.add_argument("--stage", default="all", help="'sdf' = stop after the base SDF")
    ap.add_argument("--faces", type=int, default=1_000_000, help="final face count")
    ap.add_argument("--sigma", type=float, default=1.2, help="SDF Gaussian sigma at creases [voxels]")
    ap.add_argument("--sigma-flat", type=float, default=3.0, help="SDF Gaussian sigma on flat panels [voxels]")
    ap.add_argument("--panel-sigma", type=float, default=0.06, help="panel smoothing scale [m]")
    ap.add_argument("--panel-dev", type=float, default=0.004, help="deviations below this are ripples [m]")
    ap.add_argument("--ledge-r", type=float, default=0.09, help="top-surface ledge filling disk radius [m]")
    ap.add_argument("--taubin", type=int, default=20)
    ap.add_argument("--close-r", type=float, default=0.04, help="upper-body closing radius [m]")
    ap.add_argument("--patch", type=float, default=0.008, help="tyre flattening depth [m]")
    ap.add_argument("--dump", default=None, help="save the final SDF (debug)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    spec = cfg["spec"]
    h = args.voxel

    if args.cache and os.path.exists(args.cache):
        c = np.load(args.cache, allow_pickle=False)
        m = trimesh.Trimesh(c["V"], c["F"], process=False)
        sdf, xs, ys, zs = c["sdf"], c["xs"], c["ys"], c["zs"]
        info = json.loads(str(c["info"]))
        log(f"base SDF from cache {args.cache}")
    else:
        m, info = load_input(args.mesh, spec)
        xs, ys, zs = grid(m, h)
        sdf = exact_sdf(m, xs, ys, zs, h)
        if args.cache:
            np.savez(args.cache, sdf=sdf, xs=xs, ys=ys, zs=zs, V=m.vertices, F=m.faces, info=json.dumps(info))
    if args.stage == "sdf":
        return
    holes = plate_holes(sdf, xs, ys, zs, h, m)
    ap_ = args
    fix = {}
    fix["voids_filled_voxels"] = fill_voids(sdf)
    fix["tyres"] = fix_tyres(sdf, xs, ys, zs, h, spec, patch=ap_.patch)
    tyres = fix["tyres"]["tyres"]
    fix["sills"] = extrude_sills(sdf, xs, ys, zs, h, spec, tyres)
    fix["underbody"] = smooth_underbody(sdf, xs, ys, zs, h, spec, tyres)
    fix["plates"], plate_zone = smooth_plates(sdf, xs, ys, zs, h, holes)
    fix["body_closing"] = close_body(sdf, xs, ys, zs, h, spec, tyres, r=ap_.close_r)
    fix["ledges"] = fill_all_ledges(sdf, xs, ys, zs, h, spec, tyres, m, r_top=ap_.ledge_r, exclude=plate_zone)
    fix["windshield"], (xa, _) = fair_windshield(sdf, xs, ys, zs, h, spec)
    fix["pillar_pits"] = fill_pillar_pits(sdf, xs, ys, zs, h, xa)
    fix["panels"] = {}
    for nm, axis, outward in (("top", 2, +1), ("front", 0, -1), ("rear", 0, +1), ("bottom", 2, -1)):
        fix["panels"][nm] = panel_smooth(sdf, xs, ys, zs, h, axis, outward, sigma=ap_.panel_sigma, keep_dev=ap_.panel_dev)
    # sides: smooth the +y side, then mirror it onto -y
    ny2 = len(ys) // 2
    half = np.ascontiguousarray(sdf[:, ny2:])
    fix["panels"]["side"] = panel_smooth(half, xs, ys[ny2:], zs, h, 1, +1, sigma=ap_.panel_sigma, keep_dev=ap_.panel_dev)
    sdf[:, ny2:] = half; sdf[:, :ny2] = half[:, ::-1]
    del half
    fill_voids(sdf)
    info["fixes"] = fix

    sdf = 0.5 * (sdf + sdf[:, ::-1, :])
    sdf = adaptive_smooth(sdf, h, s_light=ap_.sigma, s_heavy=ap_.sigma_flat)
    if ap_.dump:
        np.savez(ap_.dump, sdf=sdf, xs=xs, ys=ys, zs=zs)
    verts, fcs, _, _ = measure.marching_cubes(sdf, 1e-3 * h, spacing=(h, h, h))
    del sdf
    mc = trimesh.Trimesh(verts + [xs[0], ys[0], zs[0]], fcs[:, ::-1], process=True)
    parts = mc.split(only_watertight=False)
    if len(parts) > 1:
        mc = max(parts, key=lambda p: len(p.faces))
    if mc.volume < 0:
        mc.invert()
    log(f"marching cubes: {len(mc.faces)} faces, {len(parts)} components, watertight={mc.is_watertight}")
    trimesh.smoothing.filter_taubin(mc, lamb=0.5, nu=-0.53, iterations=ap_.taubin)

    half0 = trimesh.intersections.slice_mesh_plane(mc, plane_normal=[0, 1, 0], plane_origin=[0, 0, 0], cap=False)
    for attempt in range(10):
        half = decimate_half(half0, int(ap_.faces // 2 * (1 + 0.011 * attempt)))
        V, F = half.vertices, half.faces
        full = trimesh.Trimesh(np.vstack([V, V * [1, -1, 1]]), np.vstack([F, F[:, ::-1] + len(V)]), process=False)
        full.merge_vertices(digits_vertex=7)
        full.remove_unreferenced_vertices()
        if full.is_watertight and full.is_winding_consistent:
            break
        log(f"decimation attempt {attempt + 1}: not watertight, retrying")
    info["decimation_attempts"] = attempt + 1
    m, info["final_fit"] = fit_spec(full, spec)
    sx = spec.length / m.extents[0]                     # hole / pocket fills can nudge the ends
    m.apply_transform(np.diag([sx, 1.0, 1.0, 1.0]))
    info["final_fit"]["length_scale"] = float(sx)
    trimesh.repair.fix_normals(m)
    d, _ = cKDTree(m.vertices).query(m.vertices * [1, -1, 1])
    lo, hi = m.bounds
    rear = (m.vertices[:, 0] > 0.55 * spec.length) & (m.vertices[:, 0] < 0.85 * spec.length)
    info.update(extents_mm=[round(float(e) * 1000) for e in m.extents],
                body_width_mm=round(float(2 * np.abs(m.vertices[rear, 1]).max()) * 1000),
                faces=len(m.faces), watertight=bool(m.is_watertight),
                winding_consistent=bool(m.is_winding_consistent), euler=int(m.euler_number),
                volume_m3=round(float(m.volume), 3), mirror_deviation_max_mm=float(d.max() * 1000),
                edge_mm_mean=round(float(m.edges_unique_length.mean()) * 1000, 2), seconds=round(time.time() - T0, 1))
    log("result " + json.dumps({k: info[k] for k in ("extents_mm", "body_width_mm", "faces", "watertight", "volume_m3", "mirror_deviation_max_mm")}))
    os.makedirs(ap_.out, exist_ok=True)
    base = os.path.join(ap_.out, ap_.tag)
    for e in ("stl", "glb"):
        m.export(f"{base}.{e}")
    try:
        sys.path.insert(0, HERE)
        from car3d.render import render_views
        render_views(m, f"{base}_views.png", title=f"{ap_.tag}  {info['extents_mm']} mm, {len(m.faces)} faces")
    except Exception as e:                               # previews are optional
        log(f"preview skipped: {e}")
    with open(f"{base}_report.json", "w") as f:
        json.dump(info, f, indent=2)
    log(f"saved {base}.stl/.glb + _report.json")


if __name__ == "__main__":
    main()
