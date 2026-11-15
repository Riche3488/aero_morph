"""
Shared paths and hand-off locations for the 5-stage pipeline.

Every stage script starts with

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import pipeline_config as cfg

which makes `pipeline_config` and the shared `common` package importable. Stage folders are
NOT put on sys.path: a script imports only modules from its own folder (Python adds the
script's folder automatically) plus `common`. Stages exchange data through files only
(see README "Data hand-offs"). The single code-level dependency, stage 2 using the stage-1
IONIQ 5 generator for parametric studies, goes through `load_module()` and is loaded lazily.
"""
import importlib.util
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

IONIQ5_PARAMETRIC = "ioniq5_parametric"                      # base id of the stage-1 IONIQ 5 generator


# ----------------------------------------------------------------------------- hand-off 2 -> 3
def geometry_root(study):
    """Stage-2 output = stage-3 input: <study>/geometry/{index.json, <case>/meta.json, <case>/*.stl}."""
    return os.path.join(MORPH_STUDIES, study, "geometry")


def geometry_dir(study, case):
    return os.path.join(geometry_root(study), case)


# ----------------------------------------------------------------------------- helpers
def rel(path):
    """Path relative to the project root if it lies inside it, else absolute."""
    path = os.path.abspath(path)
    return os.path.relpath(path, ROOT) if path.startswith(ROOT + os.sep) else path


def load_module(stage_dir, name):
    """Import <stage_dir>/<name>.py without putting stage_dir on sys.path (no name clashes)."""
    key = f"_stage_{os.path.basename(stage_dir)}_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, os.path.join(stage_dir, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
    return sys.modules[key]
