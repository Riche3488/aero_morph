"""Raw TRELLIS mesh -> car frame (x rearward from front bumper, y left, z up, ground on tyres),
sliver cleanup, decimation, anisotropic fit to the published L / W / H."""
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def align_to_car_frame(m, orientation="auto"):
    """TRELLIS meshes are z-up with arbitrary heading: PCA in the ground plane gives the length
    axis; the front is the end with the lower top (hood vs. roof/tailgate/trunk).
    orientation='flip' reverses the automatic decision."""
    m = m.copy()
    xy = m.vertices[:, :2] - m.vertices[:, :2].mean(0)
    w, vecs = np.linalg.eigh(np.cov(xy.T))
    ax = vecs[:, np.argmax(w)]
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.arctan2(ax[1], ax[0]), [0, 0, 1]))
    V = m.vertices
    x0, x1 = V[:, 0].min(), V[:, 0].max(); Lx = x1 - x0
    top_lo = V[V[:, 0] < x0 + 0.15 * Lx, 2].max()
    top_hi = V[V[:, 0] > x1 - 0.15 * Lx, 2].max()
    flip = top_lo > top_hi
    if orientation == "flip":
        flip = not flip
    if flip:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]))
    lo, hi = m.bounds
    m.apply_translation([-lo[0], -(lo[1] + hi[1]) / 2, -lo[2]])
    return m


def fit_length(m, spec):
    m = m.copy()
    m.apply_scale(spec.length / m.extents[0])
    return m


def fit_lwh(m, spec):
    """Length -> L, body width over the rear half (no mirrors) -> W, roof (99.95 pct, ignores an
    antenna tip) -> H. Corrects the depth ambiguity of single / few-view reconstruction."""
    m = m.copy()
    V = m.vertices
    x0, Lx = V[:, 0].min(), np.ptp(V[:, 0])
    rear = (V[:, 0] > x0 + 0.55 * Lx) & (V[:, 0] < x0 + 0.85 * Lx)
    sx = spec.length / Lx
    sy = spec.width / np.ptp(V[rear, 1])
    sz = spec.height / np.percentile(V[:, 2], 99.95)
    m.apply_transform(np.diag([sx, sy, sz, 1.0]))
    return m


def remove_underbody_slivers(m, spec):
    """Mesh in metres, x from the front bumper (length already fitted). The ground is the lowest
    tyre point; geometry outside the wheel zones lower than 0.8 x ground clearance is an
    artefact (ground / shadow slivers). Re-grounds the mesh on the tyres."""
    V, F = m.vertices, m.faces
    axles = np.array([spec.axle_f, spec.axle_r])
    vw = np.min(np.abs(V[:, :1] - axles[None]), axis=1) < 0.8 * spec.tire_radius
    z_ground = np.percentile(V[vw, 2], 0.2)
    c = V[F].mean(1)
    near_wheel = np.min(np.abs(c[:, :1] - axles[None]), axis=1) < spec.tire_radius + 0.08
    keep = ~(((c[:, 2] - z_ground) < 0.8 * spec.ground_clearance) & ~near_wheel) & (c[:, 2] > z_ground - 0.01)
    m2 = m.submesh([np.where(keep)[0]], append=True) if not keep.all() else m.copy()
    m2 = max(m2.split(only_watertight=False), key=lambda p: len(p.faces))
    m2.apply_translation([0, 0, -z_ground])
    return m2, int((~keep).sum())


def align_candidate(raw, spec, faces=300_000, orientation="auto"):
    parts = raw.split(only_watertight=False)
    m = max(parts, key=lambda p: len(p.faces))
    m = align_to_car_frame(m, orientation)
    m = fit_length(m, spec)
    m, n_sliver = remove_underbody_slivers(m, spec)
    src_v, src_c = m.vertices.copy(), m.visual.vertex_colors.copy()
    if len(m.faces) > faces:
        m = m.simplify_quadric_decimation(face_count=faces)
        _, idx = cKDTree(src_v).query(m.vertices)          # decimation drops vertex colours
        m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=src_c[idx])
    m = fit_lwh(m, spec)
    info = dict(components=len(parts), sliver_faces_removed=n_sliver, faces=len(m.faces),
                extents_mm=[round(float(e) * 1000) for e in m.extents])
    return m, info
