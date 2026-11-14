import sys; sys.path.insert(0, "/home/riche3488/aero_project/gen3d")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import render_views as rv
path, out = sys.argv[1], sys.argv[2]
V, F, base = rv.load(path, 200_000)
fig, axs = plt.subplots(3, 2, figsize=(20, 15))
rv.draw(axs[0,0], V, F, base, (0,2), np.array([0,1,0]), "LEFT side (+y)")
rv.draw(axs[0,1], V*[1,-1,1], F[:, ::-1], base, (0,2), np.array([0,1,0]), "RIGHT side (-y), mirrored to match left")
rv.draw(axs[1,0], V, F, base, (0,1), np.array([0,0,-1]), "BOTTOM (-z)")
rv.draw(axs[1,1], V, F, base, (1,2), np.array([1,0,0]), "REAR (+x)")
rv.iso(axs[2,0], V, F, base, az=-120, el=-15); axs[2,0].set_title("low iso front-right")
rv.iso(axs[2,1], V, F, base, az=60, el=-15); axs[2,1].set_title("low iso rear-left")
fig.tight_layout(); fig.savefig(out, dpi=75); print(out)
