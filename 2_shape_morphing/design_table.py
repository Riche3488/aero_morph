"""
Design table and study settings.

  2_shape_morphing/studies/<study>/designs.csv   one row per design
  2_shape_morphing/studies/<study>/study.json    base geometry of the study (optional)

Base geometry
  * "ioniq5_parametric" (default): the IONIQ 5 generator of stage 1. Design variables =
    parametric morph parameters (MORPH_PARAMS) + drag-mode amplitudes.
  * "stl": any watertight STL (e.g. gen3d/runs/ioniq6/ioniq6_rough.stl). Design variables =
    drag-mode amplitudes only. Wheels that are fused into the mesh are described in study.json
    so that stage 3 can split them into rotating-wheel patches.

A drag mode is 2_shape_morphing/modes/<name>.json; its column is  mode:<name>  (0 = off).
Modes record the geometry they were drawn on ("source_geometry"); only matching modes are
allowed in a study.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline_config as cfg  # noqa: E402

cfg.add_stage_paths()
import ioniq5_geometry as geo  # noqa: E402

PARAMS = list(geo.MORPH_PARAMS)
BASE = {k: v[0] for k, v in geo.MORPH_PARAMS.items()}
MODE_PREFIX = "mode:"
MODE_RANGE = (-2.0, 2.0)          # default validated amplitude range for drag modes
IONIQ5_ID = "ioniq5_parametric"
# drag modes drawn on these files share the coordinates (metres) of the IONIQ 5 generator
IONIQ5_ALIASES = ("3_cfd/baseline_case/constant/geometry/body.stl",)

# IONIQ 5 generator wheels (m) - written to meta.json for the CFD boundary conditions
IONIQ5_WHEELS = {
    "front": {"x": geo.X_FA / 1000, "y": geo.TRACK_F / 2000, "z": geo.Z_WHEEL / 1000,
              "r": geo.Z_WHEEL / 1000, "width": geo.TIRE_W / 1000},
    "rear": {"x": geo.X_RA / 1000, "y": geo.TRACK_R / 2000, "z": geo.Z_WHEEL / 1000,
             "r": geo.Z_WHEEL / 1000, "width": geo.TIRE_W / 1000},
}


# ----------------------------------------------------------------------------- study settings
def study_dir(study):
    return os.path.join(cfg.MORPH_STUDIES, study)


def geometry_dir(study, case):
    return os.path.join(study_dir(study), "geometry", case)


def study_config(study):
    p = os.path.join(study_dir(study), "study.json")
    c = json.load(open(p)) if os.path.exists(p) else {}
    c.setdefault("base", {"type": IONIQ5_ID})
    return c


def write_study_config(study, conf):
    os.makedirs(study_dir(study), exist_ok=True)
    json.dump(conf, open(os.path.join(study_dir(study), "study.json"), "w"), indent=2)


def base_id(study):
    b = study_config(study)["base"]
    return IONIQ5_ID if b["type"] == IONIQ5_ID else b["path"]


def is_parametric(study):
    return study_config(study)["base"]["type"] == IONIQ5_ID


def rel(path):
    path = os.path.abspath(path)
    return os.path.relpath(path, cfg.ROOT) if path.startswith(cfg.ROOT + os.sep) else path


def stl_base_from_gen3d(stl_path, z_offset=-0.005):
    """Base config for a gen3d model: wheels from the report.json next to the STL."""
    rep = os.path.join(os.path.dirname(os.path.abspath(stl_path)), "report.json")
    if not os.path.exists(rep):
        raise FileNotFoundError(f"no report.json next to {stl_path}; give the wheels explicitly (--wheels)")
    s = json.load(open(rep))["spec_mm"]
    return stl_base(stl_path, s["front_overhang"] / 1000, (s["front_overhang"] + s["wheelbase"]) / 1000,
                    s["track_front"] / 1000, s["track_rear"] / 1000, s["tire_diameter"] / 1000,
                    s["tire_width"] / 1000, z_offset, name=json.load(open(rep)).get("name"))


def stl_base(stl_path, x_front, x_rear, track_f, track_r, tire_d, tire_w, z_offset=-0.005, name=None):
    r = tire_d / 2
    return {"type": "stl", "path": rel(stl_path), "name": name or os.path.splitext(os.path.basename(stl_path))[0],
            "units": "m", "z_offset": z_offset,
            # wheel centres in the ORIGINAL STL coordinates (tyre bottom at z = 0); z_offset is applied on export
            "wheels": {"front": {"x": x_front, "y": track_f / 2, "z": r, "r": r, "width": tire_w},
                       "rear": {"x": x_rear, "y": track_r / 2, "z": r, "r": r, "width": tire_w}},
            "split_wheels": True}


# ----------------------------------------------------------------------------- modes
def mode_path(name):
    return os.path.join(cfg.MORPH_MODES, name + ".json")


def mode_spec(name):
    return json.load(open(mode_path(name)))


def mode_range(name):
    return tuple(mode_spec(name).get("amplitude_range", MODE_RANGE))


def mode_compatible(name, study):
    src = mode_spec(name).get("source_geometry")
    b = base_id(study)
    if src is None:
        return b == IONIQ5_ID                   # legacy modes were drawn on the IONIQ 5 CFD body
    if b == IONIQ5_ID:
        return src == IONIQ5_ID or src in IONIQ5_ALIASES
    return os.path.normpath(src) == os.path.normpath(b)


def available_modes(study=None):
    if not os.path.isdir(cfg.MORPH_MODES):
        return []
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(cfg.MORPH_MODES) if f.endswith(".json"))
    return [n for n in names if study is None or mode_compatible(n, study)]


def split(params):
    """-> (parametric dict, {mode_name: amplitude}) with zero-amplitude modes dropped."""
    geo_p = {k: v for k, v in params.items() if not k.startswith(MODE_PREFIX)}
    modes = {k[len(MODE_PREFIX):]: v for k, v in params.items() if k.startswith(MODE_PREFIX) and abs(v) > 1e-12}
    return geo_p, modes


def validate(params, study):
    """Raise on unknown names / out-of-range values; return list of geometric problems."""
    geo_p, modes = split(params)
    errs = []
    if is_parametric(study):
        geo.set_design(geo_p)
        errs = geo.validate_design()
        geo.set_design()
    else:
        changed = [k for k, v in geo_p.items() if k in BASE and abs(v - BASE[k]) > 1e-9]
        if changed:
            raise ValueError(f"parametric variables {changed} only apply to the IONIQ 5 generator; "
                             f"study base is {base_id(study)} (use drag modes)")
    for m, a in modes.items():
        if not os.path.exists(mode_path(m)):
            raise KeyError(f"unknown drag mode '{m}' (available: {available_modes(study)})")
        if not mode_compatible(m, study):
            raise ValueError(f"mode '{m}' was drawn on {mode_spec(m).get('source_geometry', 'the IONIQ 5 body')}, "
                             f"not on this study's base {base_id(study)}")
        lo, hi = mode_range(m)
        if not lo <= a <= hi:
            raise ValueError(f"mode:{m}={a} outside [{lo}, {hi}]")
    return errs


def full(params):
    return {**BASE, **params}


# ----------------------------------------------------------------------------- table io
def columns(rows, study=None):
    modes = sorted({k for r in rows for k in r["params"] if k.startswith(MODE_PREFIX)})
    return (PARAMS if study is None or is_parametric(study) else []) + modes


def read_designs(study):
    path = os.path.join(study_dir(study), "designs.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            p = {k: float(v) for k, v in r.items() if k != "case" and v not in ("", None)}
            rows.append({"case": r["case"], "params": full(p)})
    return rows


def write_designs(study, rows):
    os.makedirs(study_dir(study), exist_ok=True)
    cols = columns(rows, study)
    with open(os.path.join(study_dir(study), "designs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case"] + cols)
        for r in rows:
            p = full(r["params"])
            w.writerow([r["case"]] + [f"{p.get(k, 0.0):g}" for k in cols])


def label(params):
    """Short human label: only what differs from the baseline."""
    out = []
    for k, v in params.items():
        if k.startswith(MODE_PREFIX):
            if abs(v) > 1e-12:
                out.append(f"{k}×{v:g}")
        elif k in BASE and abs(v - BASE[k]) > 1e-9:
            u = geo.MORPH_PARAMS[k][3]
            out.append(f"{k}={v:g}{u if u != '-' else ''}")
    return ", ".join(out) if out else "baseline"
