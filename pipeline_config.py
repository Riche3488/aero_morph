"""
Shared paths for the 5-stage pipeline. Every stage script starts with

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import pipeline_config as cfg
    cfg.add_stage_paths()

after which modules from all stage folders can be imported directly
(e.g. `import ioniq5_geometry`, `import design_table`, `import rbf`).
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

GEOMETRY_DIR = os.path.join(ROOT, "1_vehicle_geometry")      # stage 1: base vehicle geometry
MORPH_DIR = os.path.join(ROOT, "2_shape_morphing")           # stage 2: designs, morph modes, exported geometry
CFD_DIR = os.path.join(ROOT, "3_cfd")                        # stage 3: OpenFOAM cases and results
DL_DIR = os.path.join(ROOT, "4_dl_training")                 # stage 4: surrogate model training
INFER_DIR = os.path.join(ROOT, "5_inference")                # stage 5: model-based prediction

MORPH_STUDIES = os.path.join(MORPH_DIR, "studies")           # <study>/designs.csv, geometry/<case>/, previews/
MORPH_MODES = os.path.join(MORPH_DIR, "modes")               # drag (RBF) morph modes *.json
CFD_RUNS = os.path.join(CFD_DIR, "runs")                     # <study>/cases/<case>/, results.csv, report/
CFD_TEMPLATE = os.path.join(CFD_DIR, "template")
CFD_BASELINE = os.path.join(CFD_DIR, "baseline_case")        # converged baseline (mapped initial field)

FOAM_BASHRC = "/opt/openfoam12/etc/bashrc"


def add_stage_paths():
    for d in (GEOMETRY_DIR, MORPH_DIR, CFD_DIR, DL_DIR, INFER_DIR):
        if d not in sys.path:
            sys.path.insert(0, d)
