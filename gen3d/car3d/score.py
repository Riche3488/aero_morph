"""Automatic candidate scoring: are all four wheels there, and do left and right agree?"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def wheel_counts(m, spec):
    V = m.vertices
    r = spec.tire_radius
    out = {}
    for name, x, track, side in [("FL", spec.axle_f, spec.track_front, 1), ("FR", spec.axle_f, spec.track_front, -1),
                                 ("RL", spec.axle_r, spec.track_rear, 1), ("RR", spec.axle_r, spec.track_rear, -1)]:
        yc = track / 2
        sel = (np.abs(V[:, 0] - x) < 0.9 * r) & (V[:, 2] < 0.8 * r) & \
              (V[:, 1] * side > yc - spec.tire_width) & (V[:, 1] * side < yc + spec.tire_width)
        out[name] = int(sel.sum())
    return out


def mirror_error(m, n=60_000, trim=0.9):
    P, _ = trimesh.sample.sample_surface(m, n, seed=0)
    d, _ = cKDTree(P).query(P * [1, -1, 1])
    return float(np.sort(d)[: int(trim * n)].mean())


def score_candidate(m, spec):
    wc = wheel_counts(m, spec)
    vals = np.array(list(wc.values()), float)
    wheel_ratio = float(vals.min() / max(vals.max(), 1))      # 1 = all four wheels equally present
    merr = mirror_error(m)
    score = wheel_ratio - merr / 0.05                           # 10 mm mirror error costs 0.2
    return dict(wheels=wc, wheel_ratio=round(wheel_ratio, 3), mirror_error_mm=round(merr * 1000, 2),
                score=round(score, 3))
