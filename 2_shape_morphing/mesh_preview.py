"""Light display meshes for the browser GUIs (numpy only)."""
import numpy as np


def cluster_decimate(V, F, target_faces, iters=6):
    """Vertex-clustering decimation (numpy only) for the browser preview: vertices in one grid
    cell are merged into their mean, collapsed and duplicate triangles are dropped. Not
    watertight-safe and not for export, only for display."""
    if target_faces <= 0 or len(F) <= target_faces:
        return V, F
    e = np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1).mean()
    h = e * np.sqrt(len(F) / target_faces)
    for _ in range(iters):
        key = np.floor(V / h).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        n = inv.max() + 1
        cnt = np.bincount(inv, minlength=n)
        Vc = np.stack([np.bincount(inv, V[:, k], minlength=n) for k in range(3)], 1) / cnt[:, None]
        Fc = inv[F]
        Fc = Fc[(Fc[:, 0] != Fc[:, 1]) & (Fc[:, 1] != Fc[:, 2]) & (Fc[:, 0] != Fc[:, 2])]
        _, first = np.unique(np.sort(Fc, axis=1), axis=0, return_index=True)
        Fc = Fc[np.sort(first)]
        if 0.85 * target_faces <= len(Fc) <= 1.15 * target_faces:
            break
        h *= np.sqrt(len(Fc) / target_faces)
    # clustering folds some triangles over (they show as dark / bright specks): drop those whose
    # normal opposes the summed original normals of their cells
    n_orig = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    cell_n = np.zeros((len(Vc), 3))
    for k in range(3):
        np.add.at(cell_n, inv[F[:, k]], n_orig)
    n_new = np.cross(Vc[Fc[:, 1]] - Vc[Fc[:, 0]], Vc[Fc[:, 2]] - Vc[Fc[:, 0]])
    ref = cell_n[Fc[:, 0]] + cell_n[Fc[:, 1]] + cell_n[Fc[:, 2]]
    Fc = Fc[np.einsum("ij,ij->i", n_new, ref) > 0]
    used = np.unique(Fc)
    remap = np.full(len(Vc), -1, np.int64); remap[used] = np.arange(len(used))
    return Vc[used], remap[Fc]
