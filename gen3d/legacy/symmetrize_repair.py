#!/usr/bin/env python3
"""
Symmetrise and repair the TRELLIS IONIQ 5 mesh -> watertight, exactly mirror-symmetric solid.

  1. symmetry plane: optimise yaw / roll / lateral offset so the mesh best matches its mirror
     image (trimmed chamfer distance), then move that plane to y = 0
  2. voxelise the surface (dense point sampling), morphological closing to seal cracks/holes,
     flood-fill the outside -> solid occupancy (removes double/interior surfaces)
  3. clean the solid:
       - clip slivers below the ground clearance (outside the wheels)
       - optional flat underbody between the axles at the ground clearance (DrivAer-style,
         matches ../ioniq5_geometry.py)
       - morphological opening to remove thin spikes
  4. signed distance field, symmetrised: sdf_sym = (sdf + mirror_y(sdf)) / 2, light smoothing
  5. marching cubes -> watertight mesh, decimation, vertex colours transferred from the
     original surface (+ its mirror) by a robust k-NN median (kills dark speckles)

Usage
  .venv/bin/python symmetrize_repair.py output/ioniq5_final.glb --tag ioniq5_sym [--voxel 0.006]
"""
import argparse
import os
import time

import numpy as np
import trimesh
from scipy import ndimage, optimize
from scipy.spatial import cKDTree
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
L_SPEC, W_SPEC, H_SPEC = 4.635, 1.890, 1.605
CLEARANCE = 0.160


# ----------------------------------------------------------------------------- symmetry plane
def plane_transform(params):
    """params = (yaw, roll, dy): returns 4x4 that maps the candidate symmetry plane to y=0."""
    yaw, roll, dy = params
    T = trimesh.transformations.translation_matrix([0, -dy, 0])
    Rz = trimesh.transformations.rotation_matrix(-yaw, [0, 0, 1])
    Rx = trimesh.transformations.rotation_matrix(-roll, [1, 0, 0])
    return Rx @ Rz @ T


def mirror_cost(params, P, trim=0.9):
    M = plane_transform(params)
    Q = P @ M[:3, :3].T + M[:3, 3]
    Qm = Q * [1, -1, 1]
    d, _ = cKDTree(Q).query(Qm, k=1)
    d = np.sort(d)[: int(trim * len(d))]
    return d.mean()


def find_symmetry(mesh, n=150_000):
    P, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    c = P.mean(0); P = P - [0, c[1], 0]
    best = None
    for yaw0 in np.radians([-1.5, 0, 1.5]):          # coarse start, then Nelder-Mead
        r = optimize.minimize(mirror_cost, [yaw0, 0.0, 0.0], args=(P,), method="Nelder-Mead",
                              options=dict(xatol=1e-5, fatol=1e-7, maxiter=300))
        if best is None or r.fun < best.fun:
            best = r
    before = mirror_cost([0, 0, 0], P)
    yaw, roll, dy = best.x
    print(f"[sym] plane yaw={np.degrees(yaw):+.3f} deg, roll={np.degrees(roll):+.3f} deg, "
          f"dy={(dy + c[1]) * 1000:+.1f} mm | mirror error {before * 1000:.1f} -> {best.fun * 1000:.1f} mm")
    return plane_transform(best.x) @ trimesh.transformations.translation_matrix([0, -c[1], 0])


# ----------------------------------------------------------------------------- voxel solid
def make_grid(mesh, h, pad=6):
    lo, hi = mesh.bounds
    ymax = max(abs(lo[1]), abs(hi[1]))
    nyh = int(np.ceil(ymax / h)) + pad                # y grid symmetric: centres at +-(k+0.5)h
    xs = np.arange(lo[0] - pad * h, hi[0] + pad * h, h)
    ys = (np.arange(-nyh, nyh) + 0.5) * h
    zs = np.arange(-pad * h, hi[2] + pad * h, h)      # ground z=0 included
    return xs, ys, zs


def voxelise_surface(mesh, xs, ys, zs, h):
    area = mesh.area
    n = int(min(25e6, max(4e6, 6 * area / (h * h))))
    P, _ = trimesh.sample.sample_surface(mesh, n, seed=1)
    P = np.vstack([P, mesh.vertices])
    idx = np.floor((P - [xs[0] - h / 2, ys[0] - h / 2, zs[0] - h / 2]) / h).astype(np.int64)
    shape = (len(xs), len(ys), len(zs))
    ok = np.all((idx >= 0) & (idx < shape), axis=1)
    occ = np.zeros(shape, bool)
    occ[tuple(idx[ok].T)] = True
    print(f"[vox] grid {shape} ({np.prod(shape) / 1e6:.0f} M voxels), {n / 1e6:.1f} M samples")
    return occ


def solidify(surf, h, radii_mm=(12, 24, 36, 48, 64, 80), min_fill=0.45):
    """Outside = flood fill from the grid border through the shell thickened by R (Euclidean),
    solid = complement, then eroded back by R (= morphological closing with a ball).
    TRELLIS meshes are thick closed shells with an inner cavity; tiny holes through both layers
    let the flood leak into the cavity, so R is increased until the cavity is sealed
    (solid fills >= min_fill of the bounding box)."""
    st = ndimage.generate_binary_structure(3, 1)
    d_out = ndimage.distance_transform_edt(~surf).astype(np.float32) * h
    for R in radii_mm:
        shell = d_out <= R / 1000.0
        lab, _ = ndimage.label(~shell, structure=st)
        border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(),
                                           lab[:, -1].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
        outside = np.isin(lab, border[border > 0])
        del lab
        solid = ~outside
        solid = ndimage.distance_transform_edt(solid).astype(np.float32) * h > R / 1000.0 - 0.5 * h
        idx = np.argwhere(solid)
        bb = np.prod(idx.max(0) - idx.min(0) + 1)
        fill = solid.sum() / bb
        print(f"[solid] closing R={R} mm: volume {solid.sum() * h ** 3:.2f} m^3, bbox fill {fill:.2f}")
        if fill >= min_fill:
            return solid
    print("[solid] WARNING: cavity not sealed even at the largest radius")
    return solid


def detect_axles(mesh):
    V = mesh.vertices
    low = V[V[:, 2] < 0.03]
    xs = np.sort(low[:, 0])
    gap = np.argmax(np.diff(xs))
    fa, ra = np.median(xs[:gap + 1]), np.median(xs[gap + 1:])
    print(f"[geo] tyre contact x: front {fa:.3f} m, rear {ra:.3f} m (wheelbase {(ra - fa) * 1000:.0f} mm, spec 3000)")
    return fa, ra


def clean_solid(solid, xs, ys, zs, axles, flat_floor, open_r, h):
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij", sparse=True)
    fa, ra = axles
    wheel = ((np.abs(X - fa) < 0.42) | (np.abs(X - ra) < 0.42)) & (np.abs(Y) > 0.50)
    below = Z < CLEARANCE
    n0 = solid.sum()
    solid[np.broadcast_to(below & ~wheel, solid.shape)] = False
    if flat_floor:
        # between the wheels: fill from the clearance up to the body (flat floor)
        region = (X > fa + 0.42) & (X < ra - 0.42)
        band = (Z >= CLEARANCE) & (Z <= 0.60)
        s = solid & np.broadcast_to(band, solid.shape)
        above = np.flip(np.logical_or.accumulate(np.flip(s, axis=2), axis=2), axis=2)  # solid at/above (in band)
        foot = np.broadcast_to(region, solid.shape) & np.broadcast_to(band, solid.shape) & above
        # only where the column really is body (solid in 0.25..0.55 somewhere)
        body_col = (solid & np.broadcast_to((Z > 0.25) & (Z < 0.55), solid.shape)).any(axis=2, keepdims=True)
        solid |= foot & body_col
    if open_r > 0:
        st = ndimage.generate_binary_structure(3, 1)
        solid = ndimage.binary_opening(solid, st, iterations=open_r)
    # keep the main body + wheels (largest components that touch the body)
    lab, n = ndimage.label(solid)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = np.where(sizes > 0.002 * sizes.max())[0] + 1
        solid = np.isin(lab, keep)
    print(f"[clean] solid voxels {n0 / 1e6:.2f} M -> {solid.sum() / 1e6:.2f} M, components kept: "
          f"{len(np.unique(ndimage.label(solid)[0])) - 1}")
    return solid


def sdf_from_solid(solid, h):
    out = ndimage.distance_transform_edt(~solid).astype(np.float32)
    inn = ndimage.distance_transform_edt(solid).astype(np.float32)
    return (out - inn + 0.0) * h          # >0 outside, <0 inside (voxel-centre approx)


# ----------------------------------------------------------------------------- colours
def transfer_colours(new_v, src_mesh, k=16):
    V = src_mesh.vertices
    C = src_mesh.visual.vertex_colors[:, :3].astype(np.float32)
    P = np.vstack([V, V * [1, -1, 1]])
    Cc = np.vstack([C, C])
    d, idx = cKDTree(P).query(new_v, k=k)
    nb = Cc[idx]                                    # (n,k,3)
    lum = nb.mean(2)
    # drop the darkest third of neighbours (speckle / interior-surface colours) then median
    order = np.argsort(lum, axis=1)[:, k // 3:]
    nb = np.take_along_axis(nb, order[..., None], axis=1)
    col = np.median(nb, axis=1)
    return np.c_[col, np.full(len(col), 255)].astype(np.uint8)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--tag", default="ioniq5_sym")
    ap.add_argument("--voxel", type=float, default=0.006, help="voxel size [m]")
    ap.add_argument("--open", type=int, default=1, help="opening radius [voxels] to remove spikes")
    ap.add_argument("--smooth", type=float, default=1.0, help="Gaussian sigma on the SDF [voxels]")
    ap.add_argument("--no-flat-floor", action="store_true")
    ap.add_argument("--faces", type=int, default=400_000)
    ap.add_argument("--taubin", type=int, default=15, help="Taubin smoothing iterations")
    args = ap.parse_args()
    t0 = time.time()
    h = args.voxel
    out_dir = os.path.join(HERE, "output")

    src = trimesh.load(args.mesh, force="mesh", process=False)
    src = trimesh.Trimesh(src.vertices, src.faces, vertex_colors=src.visual.vertex_colors, process=True)

    # 1. symmetry plane -> y = 0, then re-ground on the tyres
    M = find_symmetry(src)
    src.apply_transform(M)
    src.apply_translation([-src.bounds[0][0], 0, -np.percentile(src.vertices[:, 2], 0.05)])
    axles = detect_axles(src)

    # 2. voxel solid
    xs, ys, zs = make_grid(src, h)
    surf = voxelise_surface(src, xs, ys, zs, h)
    solid = solidify(surf, h)
    del surf
    print(f"[solid] done ({time.time() - t0:.0f}s)")

    # 3. clean
    solid = clean_solid(solid, xs, ys, zs, axles, not args.no_flat_floor, args.open, h)

    # 4. symmetric SDF
    sdf = sdf_from_solid(solid, h)
    del solid
    asym = np.mean(np.abs(sdf - sdf[:, ::-1, :])[np.abs(sdf) < 3 * h])
    sdf = 0.5 * (sdf + sdf[:, ::-1, :])
    if args.smooth > 0:
        sdf = ndimage.gaussian_filter(sdf, args.smooth)
    print(f"[sdf] mean |sdf - mirror| near surface before symmetrisation: {asym * 1000:.1f} mm")

    # 5. surface: marching cubes -> repair -> Taubin smoothing
    verts, faces, _, _ = measure.marching_cubes(sdf, 0.0, spacing=(h, h, h))
    del sdf
    verts += [xs[0], ys[0], zs[0]]
    m = trimesh.Trimesh(verts, faces[:, ::-1], process=True)   # skimage winding -> outward normals
    print(f"[mc] {len(m.faces)} faces, watertight={m.is_watertight}")
    if not m.is_watertight:
        import pymeshfix
        mf = pymeshfix.MeshFix(m.vertices, m.faces)
        mf.repair(joincomp=True, remove_smallest_components=True)
        m = trimesh.Trimesh(mf.points, mf.faces, process=True)
        print(f"[mc] pymeshfix -> {len(m.faces)} faces, watertight={m.is_watertight}")
    if m.volume < 0:
        m.invert()
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=args.taubin)

    # 6. exact symmetry + watertight decimation: cut at y=0, decimate the left half with a
    #    locked boundary, colour it, mirror it and weld the seam
    half = trimesh.intersections.slice_mesh_plane(m, plane_normal=[0, 1, 0], plane_origin=[0, 0, 0], cap=False)
    half = decimate_half(half, args.faces // 2)
    half.visual = trimesh.visual.ColorVisuals(half, vertex_colors=transfer_colours(half.vertices, src))
    m = mirror_weld(half)

    # 7. proportions: axles to the spec wheelbase (wheel zones rigid), then L / W / H to spec
    m = warp_axles(m, axles)
    m = fit_spec(m)
    trimesh.repair.fix_normals(m)

    d, _ = cKDTree(m.vertices).query(m.vertices * [1, -1, 1])
    ext = m.extents * 1000
    print(f"[result] L/W/H = {ext[0]:.0f}/{ext[1]:.0f}/{ext[2]:.0f} mm, faces {len(m.faces)}, "
          f"watertight={m.is_watertight}, winding_consistent={m.is_winding_consistent}, "
          f"volume {m.volume:.2f} m^3, mirror deviation max {d.max() * 1000:.4f} mm")
    for e in ("glb", "obj", "stl"):
        m.export(os.path.join(out_dir, f"{args.tag}.{e}"))
    print(f"[done] {os.path.join(out_dir, args.tag)}.*  ({time.time() - t0:.0f}s)")


def decimate_half(half, target):
    import open3d as o3d
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(half.vertices),
                                   o3d.utility.Vector3iVector(half.faces))
    om = om.simplify_quadric_decimation(target_number_of_triangles=int(target), boundary_weight=1e4)
    om.remove_degenerate_triangles(); om.remove_duplicated_triangles(); om.remove_unreferenced_vertices()
    out = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles), process=True)
    # snap the seam exactly onto the symmetry plane
    edges = out.edges_sorted
    uniq, cnt = np.unique(edges, axis=0, return_counts=True)
    bverts = np.unique(uniq[cnt == 1])
    seam = bverts[np.abs(out.vertices[bverts, 1]) < 0.01]
    v = out.vertices.copy(); v[seam, 1] = 0.0
    v[:, 1] = np.maximum(v[:, 1], 0.0)
    out.vertices = v
    # faces lying in the symmetry plane would be duplicated by the mirror -> drop them
    inplane = np.all(np.abs(v[out.faces, 1]) < 1e-9, axis=1)
    if inplane.any():
        out.update_faces(~inplane); out.remove_unreferenced_vertices()
        print(f"[half] dropped {inplane.sum()} faces lying in the symmetry plane")
    print(f"[half] {len(out.faces)} faces, seam vertices {len(seam)} (open boundary verts {len(bverts)})")
    return out


def mirror_weld(half):
    V, F = half.vertices, half.faces
    C = half.visual.vertex_colors
    Vm = V * [1, -1, 1]
    full = trimesh.Trimesh(np.vstack([V, Vm]), np.vstack([F, F[:, ::-1] + len(V)]),
                           vertex_colors=np.vstack([C, C]), process=False)
    full.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=7)
    full.remove_unreferenced_vertices()
    print(f"[mirror] {len(full.faces)} faces, watertight={full.is_watertight}")
    return full


def warp_axles(m, axles, half_zone=0.40):
    """Smooth monotone x-warp: tyre contacts -> spec axle positions (845 / 3845 mm from the
    front bumper), wheel zones keep slope ~1 (rigid), overhangs / wheelbase stretch."""
    from scipy.interpolate import PchipInterpolator
    fa, ra = axles
    x0, x1 = m.bounds[0][0], m.bounds[1][0]
    s = L_SPEC / (x1 - x0)                       # overall length scale first
    FA, RA = 0.845, 3.845
    src = np.array([x0, fa - half_zone / s, fa, fa + half_zone / s, ra - half_zone / s, ra, ra + half_zone / s, x1])
    dst = np.array([0, FA - half_zone, FA, FA + half_zone, RA - half_zone, RA, RA + half_zone, L_SPEC])
    f = PchipInterpolator(src, dst)
    v = m.vertices.copy(); v[:, 0] = f(v[:, 0]); m.vertices = v
    print(f"[warp] axles {fa:.3f}/{ra:.3f} -> {FA}/{RA} m, segment scales "
          f"F-overhang {np.diff(dst[:2])[0] / np.diff(src[:2])[0]:.3f}, "
          f"wheelbase {(dst[4] - dst[3]) / (src[4] - src[3]):.3f}, R-overhang {(dst[7] - dst[6]) / (src[7] - src[6]):.3f}")
    return m


def fit_spec(m):
    """width over the rear half (no mirrors) -> 1890 mm, roof (99.95 pct) -> 1605 mm; ground z=0."""
    V = m.vertices
    m.apply_translation([-V[:, 0].min(), 0, -V[:, 2].min()])
    V = m.vertices
    rear = (V[:, 0] > 0.55 * L_SPEC) & (V[:, 0] < 0.85 * L_SPEC)
    sy = W_SPEC / (2 * np.abs(V[rear, 1]).max())
    sz = H_SPEC / np.percentile(V[:, 2], 99.95)
    m.apply_transform(np.diag([1.0, sy, sz, 1.0]))
    print(f"[fit] width scale {sy:.3f}, height scale {sz:.3f}")
    return m


if __name__ == "__main__":
    main()
