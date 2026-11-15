#!/usr/bin/env python3
"""Copy the CFD surfaces of the IONIQ 5 baseline (half model: +y side only, metres) into constant/geometry.

Source = a stage-2 export of the baseline design (default 2_shape_morphing/studies/demo/geometry/c000,
IONIQ 5 generator at baseline, res 0.5), so stage 3 reads hand-off files only and imports no stage-1/2 code.
The triangles are byte-identical to what the generator writes directly (the 80-byte binary header
names the stage-2 part; stage 3 ignores it). The export must be at res 0.5 (checked).

  python3 make_geometry.py [EXPORT_DIR]
"""
import json
import os
import shutil
import sys

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(here))
src = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(
    root, "2_shape_morphing", "studies", "demo", "geometry", "c000")
KEEP = {"body": "body", "wheel_FL": "wheelFront", "wheel_RL": "wheelRear", "mirror_L": "mirror"}
RES = 0.5                     # surface resolution of the baseline (CFD default of export_designs.py)

meta_path = os.path.join(src, "meta.json")
if not os.path.exists(meta_path):
    sys.exit(f"no stage-2 export in {src}; run: python3 2_shape_morphing/export_designs.py demo --cases c000")
meta = json.load(open(meta_path))
if meta.get("base") != "ioniq5_parametric" or meta.get("label") != "baseline":
    sys.exit(f"{src} is not the IONIQ 5 baseline design (base={meta.get('base')}, label={meta.get('label')})")
res = meta.get("signature", {}).get("res")
if meta.get("status") != "ok" or res != RES:
    sys.exit(f"{src} was exported at res={res} (status {meta.get('status')}), the baseline needs res={RES}: "
             f"python3 2_shape_morphing/export_designs.py demo --cases c000 --res {RES} --force")
out = os.path.join(here, "constant", "geometry")
os.makedirs(out, exist_ok=True)
for part, patch in KEEP.items():
    shutil.copy(os.path.join(src, part + ".stl"), os.path.join(out, patch + ".stl"))
    print(f"{patch:12s} {meta['parts'][part]['triangles']:7d} tris  <- {os.path.relpath(src, root)}/{part}.stl")
