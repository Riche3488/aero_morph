#!/usr/bin/env python3
"""Export simplified CFD surfaces (half model: +y side only) in metres."""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", "1_vehicle_geometry"))
import ioniq5_geometry as g

RES = 0.5                 # coarser surface tessellation (snappy cells >= 15 mm)
KEEP = {"body": "body", "wheel_FL": "wheelFront", "wheel_RL": "wheelRear", "mirror_L": "mirror"}
out = os.path.join(here, "constant", "geometry")
os.makedirs(out, exist_ok=True)
for name, V, F in g.build_all(res=RES):
    if name in KEEP:
        g.write_stl_binary(os.path.join(out, KEEP[name] + ".stl"), V * 1e-3, F, KEEP[name])
        print(f"{KEEP[name]:12s} {len(F):7d} tris  {g.check_watertight(V, F)}")
