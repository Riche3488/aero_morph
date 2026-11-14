"""Aligned candidate -> watertight, exactly mirror-symmetric solid fitted to the spec.

  1. symmetry plane: optimise yaw / roll / lateral offset so the mesh best matches its mirror
     image (trimmed chamfer), move it to y = 0
  2. voxel solid: TRELLIS meshes are thick closed shells (outer + inner skin) with small holes
     through both layers -> Euclidean closing with a radius grown until the cavity is sealed,
     outside flood fill
  3. clean: slivers below the ground clearance (outside the wheels) removed, optional flat
     underbody between the axles at the ground clearance, morphological opening (spikes)
  4. symmetric SDF  (sdf + mirror_y(sdf)) / 2, Gaussian smoothing, marching cubes, pymeshfix,
     Taubin smoothing
  5. cut at y = 0, decimate the left half with a locked seam, colour it, mirror + weld
     -> watertight and exactly symmetric
  6. smooth x-warp: tyre contacts -> spec axle positions (wheel zones rigid), then L / W / H
"""
import time

import numpy as np
import trimesh
from scipy import ndimage, optimize
from scipy.spatial import cKDTree
from skimage import measure


def log(*a):
    print("[repair]", *a, flush=True)


# ----------------------------------------------------------------------------- symmetry plane
def _plane_transform(params):
    yaw, roll, dy = params
    T = trimesh.transformations.translation_matrix([0, -dy, 0])
    Rz = trimesh.transformations.rotation_matrix(-yaw, [0, 0, 1])
    Rx = trimesh.transformations.rotation_matrix(-roll, [1, 0, 0])
    return Rx @ Rz @ T


def _mirror_cost(params, P, trim=0.9):
    M = _plane_transform(params)
    Q = P @ M[:3, :3].T + M[:3, 3]
    d, _ = cKDTree(Q).query(Q * [1, -1, 1], k=1)
    return np.sort(d)[: int(trim * len(d))].mean()


def find_symmetry(mesh, n=150_000):
    P, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    c = P.mean(0); P = P - [0, c[1], 0]
    best = None
    for yaw0 in np.radians([-1.5, 0, 1.5]):
        r = optimize.minimize(_mirror_cost, [yaw0, 0.0, 0.0], args=(P,), method="Nelder-Mead",
                              options=dict(xatol=1e-5, fatol=1e-7, maxiter=300))
        if best is None or r.fun < best.fun:
            best = r
    before = _mirror_cost([0, 0, 0], P)
    yaw, roll, dy = best.x
    info = dict(yaw_deg=float(np.degrees(yaw)), roll_deg=float(np.degrees(roll)), dy_mm=float((dy + c[1]) * 1000),
                mirror_error_before_mm=float(before * 1000), mirror_error_after_mm=float(best.fun * 1000))
    log(f"symmetry plane yaw={info['yaw_deg']:+.3f} deg roll={info['roll_deg']:+.3f} deg "
        f"dy={info['dy_mm']:+.1f} mm | mirror error {info['mirror_error_before_mm']:.1f} -> {info['mirror_error_after_mm']:.1f} mm")
    return _plane_transform(best.x) @ trimesh.transformations.translation_matrix([0, -c[1], 0]), info


# ----------------------------------------------------------------------------- voxel solid
def make_grid(mesh, h, pad=6):
    lo, hi = mesh.bounds
    nyh = int(np.ceil(max(abs(lo[1]), abs(hi[1])) / h)) + pad   # y centres at +-(k+0.5)h -> exact mirror
    xs = np.arange(lo[0] - pad * h, hi[0] + pad * h, h)
    ys = (np.arange(-nyh, nyh) + 0.5) * h
    zs = np.arange(-pad * h, hi[2] + pad * h, h)
    return xs, ys, zs


def voxelise_surface(mesh, xs, ys, zs, h):
    n = int(min(25e6, max(4e6, 6 * mesh.area / (h * h))))
    P, _ = trimesh.sample.sample_surface(mesh, n, seed=1)
    P = np.vstack([P, mesh.vertices])
    idx = np.floor((P - [xs[0] - h / 2, ys[0] - h / 2, zs[0] - h / 2]) / h).astype(np.int64)
    shape = (len(xs), len(ys), len(zs))
    ok = np.all((idx >= 0) & (idx < shape), axis=1)
    occ = np.zeros(shape, bool)
    occ[tuple(idx[ok].T)] = True
    log(f"grid {shape} ({np.prod(shape) / 1e6:.0f} M voxels, h={h * 1000:.0f} mm), {n / 1e6:.1f} M samples")
    return occ


def solidify(surf, h, radii_mm=(12, 24, 36, 48, 64, 80, 100), min_fill=0.45):
    st = ndimage.generate_binary_structure(3, 1)
    d_out = ndimage.distance_transform_edt(~surf).astype(np.float32) * h
    for R in radii_mm:
        lab, _ = ndimage.label(~(d_out <= R / 1000.0), structure=st)
        border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
                                           lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
        solid = ~np.isin(lab, border[border > 0])
        del lab
        solid = ndimage.distance_transform_edt(solid).astype(np.float32) * h > R / 1000.0 - 0.5 * h
        idx = np.argwhere(solid)
        fill = solid.sum() / np.prod(idx.max(0) - idx.min(0) + 1)
        log(f"closing R={R} mm: volume {solid.sum() * h ** 3:.2f} m^3, bbox fill {fill:.2f}")
        if fill >= min_fill:
            return solid, R
    log("WARNING: cavity not sealed even at the largest closing radius")
    return solid, R


def detect_axles(mesh, spec):
    """Tyre contact x positions (two clusters of ground-touching vertices)."""
    V = mesh.vertices
    xs = np.sort(V[V[:, 2] < 0.03, 0])
    ok = len(xs) > 20
    if ok:
        gap = np.argmax(np.diff(xs))
        fa, ra = float(np.median(xs[:gap + 1])), float(np.median(xs[gap + 1:]))
        ok = abs((ra - fa) - spec.wheelbase) < 0.25 * spec.wheelbase
    if not ok:
        log("WARNING: tyre contacts not found, using spec axle positions")
        fa, ra = spec.axle_f, spec.axle_r
    log(f"tyre contact x: front {fa:.3f} m, rear {ra:.3f} m (wheelbase {(ra - fa) * 1000:.0f} mm, "
        f"spec {spec.wheelbase * 1000:.0f})")
    return fa, ra


def clean_solid(solid, xs, ys, zs, axles, spec, flat_floor, open_r):
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij", sparse=True)
    fa, ra = axles
    zone = spec.tire_radius + 0.05
    clr = spec.ground_clearance
    wheel = ((np.abs(X - fa) < zone) | (np.abs(X - ra) < zone)) & \
            (np.abs(Y) > spec.track_min / 2 - spec.tire_width - 0.08)
    below = Z < clr
    n0 = solid.sum()
    solid[np.broadcast_to(below & ~wheel, solid.shape)] = False
    if flat_floor:          # between the wheels: fill from the clearance up to the body
        region = (X > fa + zone) & (X < ra - zone)
        band = (Z >= clr) & (Z <= clr + 0.44)
        s = solid & np.broadcast_to(band, solid.shape)
        above = np.flip(np.logical_or.accumulate(np.flip(s, axis=2), axis=2), axis=2)
        foot = np.broadcast_to(region, solid.shape) & np.broadcast_to(band, solid.shape) & above
        body_col = (solid & np.broadcast_to((Z > clr + 0.09) & (Z < clr + 0.39), solid.shape)).any(axis=2, keepdims=True)
        solid |= foot & body_col
    if open_r > 0:
        solid = ndimage.binary_opening(solid, ndimage.generate_binary_structure(3, 1), iterations=open_r)
    lab, n = ndimage.label(solid)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        solid = np.isin(lab, np.where(sizes > 0.002 * sizes.max())[0] + 1)
    log(f"clean: solid voxels {n0 / 1e6:.2f} M -> {solid.sum() / 1e6:.2f} M")
    return solid


def sdf_from_solid(solid, h):
    return (ndimage.distance_transform_edt(~solid).astype(np.float32)
            - ndimage.distance_transform_edt(solid).astype(np.float32)) * h


# ----------------------------------------------------------------------------- colours
def transfer_colours(new_v, src_mesh, k=16):
    """Robust colour from the original surface and its mirror: drop the darkest third of the k
    nearest samples (speckles / interior-skin colours), then per-channel median."""
    V = src_mesh.vertices
    C = src_mesh.visual.vertex_colors[:, :3].astype(np.float32)
    P, Cc = np.vstack([V, V * [1, -1, 1]]), np.vstack([C, C])
    _, idx = cKDTree(P).query(new_v, k=k)
    nb = Cc[idx]
    order = np.argsort(nb.mean(2), axis=1)[:, k // 3:]
    col = np.median(np.take_along_axis(nb, order[..., None], axis=1), axis=1)
    return np.c_[col, np.full(len(col), 255)].astype(np.uint8)


# ----------------------------------------------------------------------------- mesh ops
def decimate_half(half, target):
    import open3d as o3d
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(half.vertices), o3d.utility.Vector3iVector(half.faces))
    om = om.simplify_quadric_decimation(target_number_of_triangles=int(target), boundary_weight=1e4)
    om.remove_degenerate_triangles(); om.remove_duplicated_triangles(); om.remove_unreferenced_vertices()
    out = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), process=True)
    uniq, cnt = np.unique(out.edges_sorted, axis=0, return_counts=True)
    bverts = np.unique(uniq[cnt == 1])
    # the half's only boundary is the seam (the full mesh is watertight); decimation may let
    # seam vertices drift slightly off the plane -> snap every boundary vertex near it
    seam = bverts[np.abs(out.vertices[bverts, 1]) < 0.05]
    v = out.vertices.copy(); v[seam, 1] = 0.0
    out.vertices = v
    inplane = np.all(np.abs(v[out.faces, 1]) < 1e-9, axis=1)   # would be duplicated by the mirror
    if inplane.any():
        out.update_faces(~inplane); out.remove_unreferenced_vertices()
    return out


def mirror_weld(half, fix=True):
    V, F, C = half.vertices, half.faces, half.visual.vertex_colors
    full = trimesh.Trimesh(np.vstack([V, V * [1, -1, 1]]), np.vstack([F, F[:, ::-1] + len(V)]),
                           vertex_colors=np.vstack([C, C]), process=False)
    full.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=7)
    full.remove_unreferenced_vertices()
    if not fix:
        return full
    if not full.is_watertight:          # safety net 1: tiny seam holes
        trimesh.repair.fill_holes(full)
        log(f"seam holes filled -> watertight={full.is_watertight}")
    if not full.is_watertight:          # safety net 2: local non-manifold folds (e.g. at the
        import pymeshfix                # flat-floor ends) -> pymeshfix; breaks exact symmetry
        sv, sc = full.vertices.copy(), full.visual.vertex_colors.copy()     # only locally
        mf = pymeshfix.MeshFix(full.vertices, full.faces)
        mf.repair(joincomp=True, remove_smallest_components=True)
        full = trimesh.Trimesh(mf.points, mf.faces, process=True)
        full.visual = trimesh.visual.ColorVisuals(full, vertex_colors=sc[cKDTree(sv).query(full.vertices)[1]])
        log(f"pymeshfix -> watertight={full.is_watertight}")
    return full


def warp_axles(m, axles, spec):
    """Smooth monotone x-warp: tyre contacts -> spec axles; wheel zones keep slope ~1."""
    from scipy.interpolate import PchipInterpolator
    fa, ra = axles
    hz = spec.tire_radius + 0.03
    x0, x1 = m.bounds[0][0], m.bounds[1][0]
    s = spec.length / (x1 - x0)
    FA, RA, L = spec.axle_f, spec.axle_r, spec.length
    src = np.array([x0, fa - hz / s, fa, fa + hz / s, ra - hz / s, ra, ra + hz / s, x1])
    dst = np.array([0, FA - hz, FA, FA + hz, RA - hz, RA, RA + hz, L])
    if np.any(np.diff(src) <= 0) or np.any(np.diff(dst) <= 0):
        log("WARNING: axle warp skipped (inconsistent knots)")
        return m, {}
    f = PchipInterpolator(src, dst)
    v = m.vertices.copy(); v[:, 0] = f(v[:, 0]); m.vertices = v
    info = dict(front_overhang_scale=float((dst[1] - dst[0]) / (src[1] - src[0])),
                wheelbase_scale=float((dst[4] - dst[3]) / (src[4] - src[3])),
                rear_overhang_scale=float((dst[7] - dst[6]) / (src[7] - src[6])))
    log("axle warp " + ", ".join(f"{k} {v:.3f}" for k, v in info.items()))
    return m, info


def fit_spec(m, spec):
    V = m.vertices
    m.apply_translation([-V[:, 0].min(), 0, -V[:, 2].min()])
    V = m.vertices
    rear = (V[:, 0] > 0.55 * spec.length) & (V[:, 0] < 0.85 * spec.length)
    sy = spec.width / (2 * np.abs(V[rear, 1]).max())
    sz = spec.height / np.percentile(V[:, 2], 99.95)
    m.apply_transform(np.diag([1.0, sy, sz, 1.0]))
    return m, dict(width_scale=float(sy), height_scale=float(sz))


# ----------------------------------------------------------------------------- main entry
def repair(src, spec, voxel=0.012, smooth=1.5, open=2, taubin=30, faces=60_000, flat_floor=True):
    t0 = time.time()
    h = voxel
    info = {}
    src = trimesh.Trimesh(src.vertices, src.faces, vertex_colors=src.visual.vertex_colors, process=True)

    M, info["symmetry"] = find_symmetry(src)
    src.apply_transform(M)
    src.apply_translation([-src.bounds[0][0], 0, -np.percentile(src.vertices[:, 2], 0.05)])
    axles = detect_axles(src, spec)
    info["tyre_contact_x_m"] = [round(a, 4) for a in axles]

    xs, ys, zs = make_grid(src, h)
    solid, info["closing_radius_mm"] = solidify(voxelise_surface(src, xs, ys, zs, h), h)
    solid = clean_solid(solid, xs, ys, zs, axles, spec, flat_floor, open)

    sdf = sdf_from_solid(solid, h)
    del solid
    info["sdf_asymmetry_mm"] = float(np.mean(np.abs(sdf - sdf[:, ::-1, :])[np.abs(sdf) < 3 * h]) * 1000)
    sdf = 0.5 * (sdf + sdf[:, ::-1, :])
    if smooth > 0:
        sdf = ndimage.gaussian_filter(sdf, smooth)

    # iso-level slightly off 0: the mirror average (+h + -h)/2 gives exact zeros on grid nodes
    # (flat floor), which makes marching cubes emit non-manifold edges
    verts, fcs, _, _ = measure.marching_cubes(sdf, 1e-3 * h, spacing=(h, h, h))
    del sdf
    m = trimesh.Trimesh(verts + [xs[0], ys[0], zs[0]], fcs[:, ::-1], process=True)
    info["mc_pymeshfix"] = not m.is_watertight
    if not m.is_watertight:
        import pymeshfix
        mf = pymeshfix.MeshFix(m.vertices, m.faces)
        mf.repair(joincomp=True, remove_smallest_components=True)
        m = trimesh.Trimesh(mf.points, mf.faces, process=True)
    if m.volume < 0:
        m.invert()
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=taubin)
    log(f"marching cubes: {len(m.faces)} faces, watertight={m.is_watertight} (pymeshfix used: {info['mc_pymeshfix']})")

    half0 = trimesh.intersections.slice_mesh_plane(m, plane_normal=[0, 1, 0], plane_origin=[0, 0, 0], cap=False)
    # QEM decimation occasionally leaves a non-manifold fold (e.g. at the flat-floor ends); retry
    # with slightly different targets so the result stays exactly symmetric AND watertight
    for attempt in range(8):
        half = decimate_half(half0, int(faces // 2 * (1 + 0.013 * attempt)))
        half.visual = trimesh.visual.ColorVisuals(half, vertex_colors=transfer_colours(half.vertices, src))
        full = mirror_weld(half, fix=False)
        if full.is_watertight:
            break
        log(f"decimation attempt {attempt + 1}: not watertight, retrying")
    else:
        full = mirror_weld(half, fix=True)
    info["decimation_attempts"] = attempt + 1
    m = full

    m, info["axle_warp"] = warp_axles(m, axles, spec)
    m, info["final_fit"] = fit_spec(m, spec)
    trimesh.repair.fix_normals(m)

    d, _ = cKDTree(m.vertices).query(m.vertices * [1, -1, 1])
    info.update(extents_mm=[round(float(e) * 1000) for e in m.extents], faces=len(m.faces),
                watertight=bool(m.is_watertight), winding_consistent=bool(m.is_winding_consistent),
                volume_m3=round(float(m.volume), 3), mirror_deviation_max_mm=float(d.max() * 1000),
                mirror_deviation_p99_mm=float(np.percentile(d, 99) * 1000),
                seconds=round(time.time() - t0, 1))
    log(f"result L/W/H {info['extents_mm']} mm, {info['faces']} faces, watertight={info['watertight']}, "
        f"volume {info['volume_m3']} m^3, mirror dev max {info['mirror_deviation_max_mm']:.3f} / p99 {info['mirror_deviation_p99_mm']:.3f} mm ({info['seconds']} s)")
    return m, info
