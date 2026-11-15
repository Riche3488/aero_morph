"""
Design table and study settings.

  2_shape_morphing/studies/<study>/designs.csv   one row per design
  2_shape_morphing/studies/<study>/study.json    base geometry of the study (optional)

Base geometry
  * "ioniq5_parametric" (default): the IONIQ 5 generator of stage 1. Design variables =
    parametric morph parameters (MORPH_PARAMS) + drag-mode amplitudes.
  * "stl": any watertight STL (e.g. gen3d/runs/ioniq6/ioniq6_rough.stl). Design variables =
    drag-mode amplitudes + dimensional variables dim:foh / dim:roh / dim:length / dim:width /
    dim:spoiler (mm, see dim_morph.py). Wheels that are fused into the mesh are described in study.json
    so that stage 3 can split them into rotating-wheel patches.

The stage-1 generator is loaded lazily (generator()), only for parametric studies, so STL-base
studies work without stage 1.

A drag mode is 2_shape_morphing/modes/<name>.json; its column is  mode:<name>  (0 = off).
A dimensional variable has the column  dim:<name>  (mm, 0 = baseline).
Modes record the geometry they were drawn on ("source_geometry"); only matching modes are
allowed in a study.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline_config as cfg  # noqa: E402
import dim_morph  # noqa: E402

MODE_PREFIX = "mode:"
DIM_PREFIX = dim_morph.PREFIX
OFFSET_PREFIXES = (MODE_PREFIX, DIM_PREFIX)    # 0 = baseline, zero entries are dropped
MODE_RANGE = (-2.0, 2.0)          # default validated amplitude range for drag modes
IONIQ5_ID = cfg.IONIQ5_PARAMETRIC
IONIQ5_REFERENCE_CD = 0.288       # published Cd of the IONIQ 5
# drag modes drawn on these files share the coordinates (metres) of the IONIQ 5 generator
IONIQ5_ALIASES = ("3_cfd/baseline_case/constant/geometry/body.stl",
                  "1_vehicle_geometry/output/ioniq5_body.stl")


# ----------------------------------------------------------------------------- stage-1 generator (lazy)
def generator():
    """The stage-1 IONIQ 5 generator module (1_vehicle_geometry/ioniq5_geometry.py)."""
    return cfg.load_module(cfg.GEOMETRY_DIR, "ioniq5_geometry")


def generator_file():
    return os.path.join(cfg.GEOMETRY_DIR, "ioniq5_geometry.py")


def morph_params():
    """{name: (baseline, lower, upper, unit, description)} of the parametric generator."""
    return generator().MORPH_PARAMS


def base_values():
    return {k: v[0] for k, v in morph_params().items()}


def ioniq5_wheels():
    """Generator wheels (m) - written to meta.json for the CFD boundary conditions."""
    g = generator()
    return {"front": {"x": g.X_FA / 1000, "y": g.TRACK_F / 2000, "z": g.Z_WHEEL / 1000,
                      "r": g.Z_WHEEL / 1000, "width": g.TIRE_W / 1000},
            "rear": {"x": g.X_RA / 1000, "y": g.TRACK_R / 2000, "z": g.Z_WHEEL / 1000,
                     "r": g.Z_WHEEL / 1000, "width": g.TIRE_W / 1000}}


# ----------------------------------------------------------------------------- study settings
def study_dir(study):
    return os.path.join(cfg.MORPH_STUDIES, study)


def geometry_dir(study, case):
    return cfg.geometry_dir(study, case)


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


def reference_cd(study):
    b = study_config(study)["base"]
    return b.get("reference_cd", IONIQ5_REFERENCE_CD if b["type"] == IONIQ5_ID else None)


rel = cfg.rel


def stl_base_from_gen3d(stl_path, z_offset=-0.005):
    """Base config for a gen3d model: wheels from the report.json next to the STL."""
    rep = os.path.join(os.path.dirname(os.path.abspath(stl_path)), "report.json")
    if not os.path.exists(rep):
        raise FileNotFoundError(f"no report.json next to {stl_path}; give the wheels explicitly (--wheels)")
    r = json.load(open(rep))
    s = r["spec_mm"]
    return stl_base(stl_path, s["front_overhang"] / 1000, (s["front_overhang"] + s["wheelbase"]) / 1000,
                    s["track_front"] / 1000, s["track_rear"] / 1000, s["tire_diameter"] / 1000,
                    s["tire_width"] / 1000, z_offset, name=r.get("name"))


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
        return src == IONIQ5_ID or os.path.normpath(src) in {os.path.normpath(a) for a in IONIQ5_ALIASES}
    return os.path.normpath(src) == os.path.normpath(b)


def available_modes(study=None):
    if not os.path.isdir(cfg.MORPH_MODES):
        return []
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(cfg.MORPH_MODES) if f.endswith(".json"))
    return [n for n in names if study is None or mode_compatible(n, study)]


# ----------------------------------------------------------------------------- design params
def split(params):
    """-> (parametric dict, {mode_name: amplitude}) with zero-amplitude modes dropped.
    dim:* variables are in neither (see split_dims)."""
    geo_p = {k: v for k, v in params.items() if not k.startswith(OFFSET_PREFIXES)}
    modes = {k[len(MODE_PREFIX):]: v for k, v in params.items() if k.startswith(MODE_PREFIX) and abs(v) > 1e-12}
    return geo_p, modes


def split_dims(params):
    """{dim_name: value_mm} of the non-zero dim:* variables."""
    return dim_morph.split(params)


def dim_range(name):
    return dim_morph.PARAMS[name][1:3]


def available_dims(study=None):
    """dim:* variables apply to STL-base studies."""
    return [] if study is None or is_parametric(study) else list(dim_morph.PARAMS)


def full(params, study):
    """Complete design: parametric studies get every generator variable (baseline for missing ones);
    STL studies keep only what was given (drag modes)."""
    return {**base_values(), **params} if is_parametric(study) else dict(params)


def design_key(params, study):
    """Hashable identity of a design (zero-amplitude modes ignored)."""
    p = full(params, study)
    return tuple(sorted((k, round(v, 6)) for k, v in p.items() if not k.startswith(OFFSET_PREFIXES) or abs(v) > 1e-12))


def validate(params, study):
    """Raise on unknown names / out-of-range values; return list of geometric problems."""
    geo_p, modes = split(params)
    errs = []
    if is_parametric(study):
        g = generator()
        g.set_design(geo_p)
        errs = g.validate_design()
        g.set_design()
    elif geo_p:
        raise ValueError(f"parametric variables {sorted(geo_p)} only apply to the IONIQ 5 generator; "
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
    dims = split_dims(params)
    if dims and is_parametric(study):
        raise ValueError(f"dimensional variables {sorted(dims)} apply to STL-base studies; the IONIQ 5 "
                         f"generator has its own parametric variables")
    for d, v in dims.items():
        if d not in dim_morph.PARAMS:
            raise KeyError(f"unknown variable dim:{d} (available: {['dim:' + k for k in dim_morph.PARAMS]})")
        lo, hi = dim_range(d)
        if not lo <= v <= hi:
            raise ValueError(f"dim:{d}={v} outside [{lo:g}, {hi:g}] mm")
    return errs


def variables(params, study):
    """Description of the design variables in params (for meta.json -> stage 3/4):
    {name: {"baseline", "lower", "upper", "unit"}}."""
    out = {}
    mp = morph_params() if is_parametric(study) else {}
    for k in params:
        if k.startswith(MODE_PREFIX):
            name = k[len(MODE_PREFIX):]
            lo, hi = mode_range(name) if os.path.exists(mode_path(name)) else MODE_RANGE
            out[k] = {"baseline": 0.0, "lower": lo, "upper": hi, "unit": "amp"}
        elif k.startswith(DIM_PREFIX) and k[len(DIM_PREFIX):] in dim_morph.PARAMS:
            b, lo, hi, u, _ = dim_morph.PARAMS[k[len(DIM_PREFIX):]]
            out[k] = {"baseline": b, "lower": lo, "upper": hi, "unit": u}
        elif k in mp:
            b, lo, hi, u, _ = mp[k]
            out[k] = {"baseline": b, "lower": lo, "upper": hi, "unit": u}
    return out


def label(params, study=None):
    """Short human label: only what differs from the baseline. With an STL-base study the
    generator is not loaded (parametric keys there are invalid and shown as given)."""
    out = []
    geo_keys = [k for k in params if not k.startswith(OFFSET_PREFIXES)]
    if geo_keys and study is not None and not is_parametric(study):
        return ", ".join([f"{k}={params[k]:g}" for k in geo_keys]
                         + [f"{k}×{v:g}" for k, v in params.items() if k.startswith(MODE_PREFIX) and abs(v) > 1e-12]
                         + [f"{k}={v:+g}mm" for k, v in params.items() if k.startswith(DIM_PREFIX) and abs(v) > 1e-12])
    mp = morph_params() if geo_keys else {}
    for k, v in params.items():
        if k.startswith(MODE_PREFIX):
            if abs(v) > 1e-12:
                out.append(f"{k}×{v:g}")
        elif k.startswith(DIM_PREFIX):
            if abs(v) > 1e-12:
                out.append(f"{k}={v:+g}mm")
        elif k in mp and abs(v - mp[k][0]) > 1e-9:
            u = mp[k][3]
            out.append(f"{k}={v:g}{u if u != '-' else ''}")
    return ", ".join(out) if out else "baseline"


# ----------------------------------------------------------------------------- table io
def columns(rows, study):
    modes = sorted({k for r in rows for k in r["params"] if k.startswith(MODE_PREFIX)})
    dims = [DIM_PREFIX + d for d in dim_morph.PARAMS if any(DIM_PREFIX + d in r["params"] for r in rows)]
    return (list(morph_params()) if is_parametric(study) else []) + dims + modes


def read_designs(study):
    path = os.path.join(study_dir(study), "designs.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            p = {k: float(v) for k, v in r.items() if k != "case" and v not in ("", None)}
            rows.append({"case": r["case"], "params": full(p, study)})
    return rows


def write_designs(study, rows):
    os.makedirs(study_dir(study), exist_ok=True)
    cols = columns(rows, study)
    with open(os.path.join(study_dir(study), "designs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case"] + cols)
        for r in rows:
            p = full(r["params"], study)
            w.writerow([r["case"]] + [f"{p.get(k, 0.0):g}" for k in cols])
