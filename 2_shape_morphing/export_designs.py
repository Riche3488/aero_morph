#!/usr/bin/env python3
"""
Export the morphed geometry of every design in a study (the hand-off to stage 3).

  python3 2_shape_morphing/export_designs.py <study> [--cases c001 c002] [--res 0.5] [--force]

Base = IONIQ 5 generator (default) or any STL (study.json, see doe.py --base-stl).

For each design (IONIQ 5 base):
  1. parametric morph   (MORPH_PARAMS -> 1_vehicle_geometry/ioniq5_geometry.py)
  2. drag (RBF) modes   (mode:<name> amplitudes -> 2_shape_morphing/modes/<name>.json), applied
                        to body + mirrors; wheels are frozen and the clearance is checked
  3. write studies/<study>/geometry/<case>/
        body.stl wheel_FL.stl wheel_FR.stl wheel_RL.stl wheel_RR.stl [mirror_L.stl mirror_R.stl]
        meta.json   (params, variables, label, frontal area, wheels, QA, units = m)
  4. write studies/<study>/geometry/index.json: the designs of designs.csv whose export is
     up to date. This folder is the complete hand-off to stage 3 (CFD reads nothing else).
For an STL base: drag modes are applied in the STL's own coordinates (as drawn in the GUI) with
the wheel zones frozen (smooth falloff), then the dimensional variables dim:* (dim_morph.py:
overhangs, length, width, spoiler height; tyres move rigidly with their wheel centre), then
the z offset (tyre sink) is applied and the fused
wheels are split into regions -> body.stl = multi-solid ASCII STL with solids
body / wheelFront / wheelRear (one closed surface; stage 3 turns the regions into patches).

Up-to-date designs are skipped (same params, same mode files, same base / resolution, and for
the IONIQ 5 base the same stage-1 generator source).
"""
import argparse
import hashlib
import inspect
import json
import os

import numpy as np

import design_table as dt
import dim_morph
import rbf
from common import mesh_io as mio
from design_table import cfg

FROZEN = ("wheel_FL", "wheel_FR", "wheel_RL", "wheel_RR")
_BASE_CACHE = {}


def _smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def wheel_freeze_mask(V, wheels, gap=0.01, blend=0.06):
    """1 away from the wheels, 0 on the tyres (smooth in radius and lateral position)."""
    f = np.ones(len(V))
    for w in wheels.values():
        rho = np.hypot(V[:, 0] - w["x"], V[:, 2] - w["z"])
        fr = _smooth((rho - (w["r"] + gap)) / blend)
        fy = _smooth((np.abs(np.abs(V[:, 1]) - w["y"]) - (w["width"] / 2 + 0.03)) / blend)
        f *= 1 - (1 - fr) * (1 - fy)
    return f


def split_wheels(V, F, wheels, tol_r=0.004, tol_y=0.025):
    """Triangle -> region index (0 body, 1 wheelFront, 2 wheelRear) by tyre cylinders."""
    C = V[F].mean(1)
    reg = np.zeros(len(F), int)
    for k, key in ((1, "front"), (2, "rear")):
        w = wheels[key]
        rho = np.hypot(C[:, 0] - w["x"], C[:, 2] - w["z"])
        inside = (rho < w["r"] + tol_r) & (np.abs(np.abs(C[:, 1]) - w["y"]) <= w["width"] / 2 + tol_y)
        reg[inside] = k
    return reg


def _load_base(path):
    key = (path, os.path.getmtime(path))
    if key not in _BASE_CACHE:
        _BASE_CACHE[key] = mio.read_stl(path)
    return _BASE_CACHE[key]


def export_stl_base(study, case, params, conf, out, quiet):
    base = conf["base"]
    path = _base_path(base)
    _, modes = dt.split(params)
    V0, F = _load_base(path)
    chk = mio.check_watertight(V0, F)
    if chk["edges_not_2"]:
        raise RuntimeError(f"base STL {base['path']} is not watertight ({chk['edges_not_2']} open edges)")
    wheels0 = base["wheels"]
    dims = dt.split_dims(params)
    V, qa = V0.copy(), {}
    if modes:
        spec = rbf.combine_specs([(dt.mode_spec(m), a) for m, a in modes.items()])
        morph = rbf.RBFMorph(spec, V0, rbf.vertex_normals(V0, F))
        U = morph.displacement(V0) * wheel_freeze_mask(V0, wheels0)[:, None]
        V = V0 + U
        rep = rbf.quality_report(morph, V0, V, F)
        reg0 = split_wheels(V0, F, wheels0)
        wv = np.unique(F[reg0 > 0])
        rep["max_wheel_displacement"] = float(np.linalg.norm(U[wv], axis=1).max()) if len(wv) else 0.0
        qa["body"] = rep
        if rep["status"] != "OK":
            raise RuntimeError(f"{case}: drag morph quality failed: detJ={rep['min_detJ']:.3f} "
                               f"flips={rep['flipped_triangles']}")
    if dims:
        lm = dim_morph.detect_landmarks(V0, F, wheels0, base.get("dim_landmarks"))
        dm = dim_morph.DimMorph(dims, lm, wheels0)
        Vb = V
        V = Vb + dm.displacement(Vb, freeze=wheel_freeze_mask(Vb, wheels0))
        rep = dim_morph.quality(dm, Vb, V, F)
        rep["landmarks"] = {k: (np.round(v, 4).tolist() if isinstance(v, (list, tuple)) else
                                v if isinstance(v, str) else round(float(v), 4)) for k, v in lm.items()}
        qa["dims"] = rep
        if rep["status"] != "OK":
            raise RuntimeError(f"{case}: dimensional morph quality failed: detJ={rep['min_detJ']:.3f} "
                               f"flips={rep['flipped_triangles']}")
        wheels0 = dm.transform_wheels(wheels0)
    parts, files, wheels = _write_fused(out, V, F, wheels0, base)
    return parts, files, wheels, qa


def _write_fused(out, V, F, wheels0, base):
    """Tyre-sink offset, wheels split into regions -> out/body.stl (multi-solid ASCII)."""
    dz = base.get("z_offset", 0.0)
    V = V + np.array([0, 0, dz])
    wheels = {k: {**w, "z": w["z"] + dz} for k, w in wheels0.items()}
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith(".stl"):
            os.remove(os.path.join(out, f))
    reg = split_wheels(V, F, wheels) if base.get("split_wheels", True) else np.zeros(len(F), int)
    names = ["body", "wheelFront", "wheelRear"]
    solids = [(n, V, F[reg == k]) for k, n in enumerate(names) if (reg == k).any()]
    mio.write_stl_ascii_multi(os.path.join(out, "body.stl"), solids)
    files = {"body": {"triangles": int(len(F)), "volume_m3": round(float(mio.signed_volume(V, F)), 6),
                      "regions": {n: int(len(f)) for n, _, f in solids}}}
    return [("body", V, F)], files, wheels


def _import_path(imp):
    return imp["file"] if os.path.isabs(imp["file"]) else os.path.join(cfg.ROOT, imp["file"])


def export_imported(study, case, params, conf, out):
    """A case whose geometry is a given STL of a shape set (study.json "imports", see
    shape_set.py): aligned to the base frame, wheels at the morpher's positions for the case's
    dimensional values, then written like a morphed case."""
    import shape_set
    imp = conf["imports"][case]
    path = _import_path(imp)
    V, F, scale = shape_set.read_given(path)
    chk = mio.check_watertight(V, F)
    if chk["edges_not_2"]:
        raise RuntimeError(f"{case}: {imp['file']} is not watertight ({chk['edges_not_2']} open edges); "
                           f"stage 3 needs a closed surface")
    base = conf["base"]
    V0, F0 = _load_base(_base_path(base))
    dims = dt.split_dims(params)
    lm = dim_morph.detect_landmarks(V0, F0, base["wheels"], base.get("dim_landmarks"))
    wheels0 = dim_morph.DimMorph(dims, lm, base["wheels"]).transform_wheels(base["wheels"])
    t = np.asarray(imp["translation_mm"], float) / 1e3 if imp.get("translation_mm") is not None else None
    if t is None:                                   # not compared at import time: align now
        b = shape_set.study_base(study)
        t = shape_set.Prediction(b, dims).align(V[:: max(1, len(V) // 200_000)])
    parts, files, wheels = _write_fused(out, V + t, F, wheels0, base)
    qa = {"import": {"source": imp["file"], "set": imp.get("set"), "units_scale": scale,
                     "translation_mm": np.round(t * 1e3, 2).tolist(),
                     "deviation_vs_morpher": imp.get("deviation_vs_morpher", {})}}
    return parts, files, wheels, qa


def _base_path(base):
    return base["path"] if os.path.isabs(base["path"]) else os.path.join(cfg.ROOT, base["path"])


def _mode_hash(modes):
    h = hashlib.sha1()
    for m in sorted(modes):
        h.update(m.encode()); h.update(open(dt.mode_path(m), "rb").read())
    return h.hexdigest()[:12]


def file_hash(path):
    """Short sha1 of a file (read in chunks: fused case STLs are hundreds of MB)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


_CODE_HASH = {}


def code_hash(kind):
    """Hash of the source that shapes the exported geometry of a kind of case, so that code
    edits invalidate the cache: the export / region-split / writer functions of this file and
    common/mesh_io.py always; plus rbf.py (drag modes), dim_morph.py (dim:*), the shape-set
    reader (imported STLs). Only geometry code, so meta / report edits keep the cache."""
    if kind not in _CODE_HASH:
        import shape_set
        from common import mesh_io
        objs = [wheel_freeze_mask, split_wheels, export_stl_base, _write_fused, export_case, mesh_io]
        objs += {"modes": [rbf], "dims": [dim_morph], "import": [shape_set.read_given, export_imported]}.get(kind, [])
        h = hashlib.sha1()
        for o in objs:
            h.update(inspect.getsource(o).encode())
        _CODE_HASH[kind] = h.hexdigest()[:12]
    return _CODE_HASH[kind]


def signature(study, params, res, conf=None, case=None):
    """Everything the exported geometry depends on (cache key)."""
    conf = conf or dt.study_config(study)
    _, modes = dt.split(params)
    sig = {"params": params, "res": res, "modes_hash": _mode_hash(modes), "base": conf["base"],
           "code_hash": code_hash("base")}
    if modes:
        sig["rbf_hash"] = code_hash("modes")
    imp = conf.get("imports", {}).get(case) if case else None
    if imp:                                          # given STL: the file is the geometry
        p = _import_path(imp)
        sig["import"] = {**imp, "mtime": os.path.getmtime(p), "size": os.path.getsize(p)}
        sig["import_code_hash"] = code_hash("import")
        return sig
    if conf["base"]["type"] == "stl":
        sig["base_mtime"] = os.path.getmtime(_base_path(conf["base"]))
        if dt.split_dims(params):
            sig["dim_morph_hash"] = code_hash("dims")                 # morph code edits invalidate the cache
    else:
        sig["generator_hash"] = file_hash(dt.generator_file())      # stage-1 code edits invalidate the cache
    return sig


def export_case(study, case, params, res=0.5, force=False, quiet=False):
    params = dt.full(params, study)
    geo_p, modes = dt.split(params)
    conf = dt.study_config(study)
    out = dt.geometry_dir(study, case)
    meta_path = os.path.join(out, "meta.json")
    sig = signature(study, params, res, conf, case)
    if not force and os.path.exists(meta_path):
        old = json.load(open(meta_path))
        if old.get("signature") == sig and old.get("status") == "ok":
            return old

    if case in conf.get("imports", {}):            # geometry = a given STL (values may lie outside the morpher range)
        unknown = [k for k in dt.split_dims(params) if k not in dim_morph.PARAMS]
        if unknown:
            raise RuntimeError(f"{case}: unknown dimensional variables {unknown}")
        if modes or geo_p:                         # the given geometry is used as is: nothing else is applied
            raise RuntimeError(f"{case}: imported geometry takes only dim:* values (they label the given STL); "
                               f"{sorted([dt.MODE_PREFIX + m for m in modes] + list(geo_p))} would not be applied "
                               f"and would mislabel the case")
        parts, files, wheels, qa = export_imported(study, case, params, conf, out)
        return _write_meta(study, case, params, conf, out, parts, files, wheels, qa, modes, sig, quiet,
                           layout="fused_regions", vehicle=conf["base"].get("name", "stl"))
    errs = dt.validate(params, study)
    if errs:
        raise RuntimeError(f"{case}: invalid design: {errs}")
    if conf["base"]["type"] == "stl":
        parts, files, wheels, qa = export_stl_base(study, case, params, conf, out, quiet)
        return _write_meta(study, case, params, conf, out, parts, files, wheels, qa, modes, sig, quiet,
                           layout="fused_regions", vehicle=conf["base"].get("name", "stl"))
    geo = dt.generator()
    geo.set_design(geo_p)
    parts = [(n, V * 1e-3, F) for n, V, F in geo.build_all(res=res)]      # -> metres
    geo.set_design()

    qa = {}
    if modes:
        spec = rbf.combine_specs([(json.load(open(dt.mode_path(m))), a) for m, a in modes.items()])
        movable = [p for p in parts if p[0] not in FROZEN]
        Vm = np.vstack([p[1] for p in movable])
        Fm = np.vstack([p[2] + o for p, o in zip(movable, np.cumsum([0] + [len(p[1]) for p in movable])[:-1])])
        morph = rbf.RBFMorph(spec, Vm, rbf.vertex_normals(Vm, Fm))
        frozen_pts = [p[1] for p in parts if p[0] in FROZEN]
        new = []
        for n, V, F in parts:
            if n in FROZEN:
                new.append((n, V, F)); continue
            V1 = V + morph.displacement(V)
            qa[n] = rbf.quality_report(morph, V, V1, F, frozen=frozen_pts if n == "body" else ())
            new.append((n, V1, F))
        parts = new
        bad = [n for n, r in qa.items() if r["status"] != "OK"]
        if bad:
            raise RuntimeError(f"{case}: drag morph quality failed for {bad}: "
                               + "; ".join(f"{n}: detJ={qa[n]['min_detJ']:.3f} flips={qa[n]['flipped_triangles']}" for n in bad))
        if "clearance_to_frozen_after" in qa.get("body", {}) and qa["body"]["clearance_to_frozen_after"] < 0.01:
            raise RuntimeError(f"{case}: body comes closer than 10 mm to a wheel")

    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith(".stl"):
            os.remove(os.path.join(out, f))
    files = {}
    for n, V, F in parts:
        chk = mio.check_watertight(V, F)
        if chk["edges_not_2"] or chk["dup_directed"]:
            raise RuntimeError(f"{case}: {n} not watertight {chk}")
        mio.write_stl_binary(os.path.join(out, n + ".stl"), V, F, n)
        files[n] = {"triangles": int(len(F)), "volume_m3": round(float(mio.signed_volume(V, F)), 6)}
    return _write_meta(study, case, params, conf, out, parts, files, dt.ioniq5_wheels(), qa, modes, sig, quiet,
                       layout="separate_parts", vehicle="IONIQ 5")


def _write_meta(study, case, params, conf, out, parts, files, wheels, qa, modes, sig, quiet, layout, vehicle):
    area = mio.frontal_area([(n, V * 1e3, F) for n, V, F in parts], h=4.0) * 1e-6     # 4 mm grid -> m^2
    Vb = parts[0][1]
    meta = {"case": case, "study": study, "label": dt.label(params), "params": params,
            "variables": dt.variables(params, study), "vehicle": vehicle, "base": dt.base_id(study),
            "reference_cd": dt.reference_cd(study), "units": "m", "frontal_area": area, "parts": files,
            "cfd_layout": layout,
            "wheels": wheels, "bbox": [Vb.min(0).round(4).tolist(), Vb.max(0).round(4).tolist()],
            "drag_modes": modes, "dims_mm": dt.split_dims(params), "qa": qa, "signature": sig, "status": "ok"}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    if not quiet:
        print(f"  {case}: A={area:.4f} m²  {dt.label(params)}")
    return meta


def write_index(study, res=0.5):
    """geometry/index.json = designs of designs.csv whose exported geometry is up to date.
    Stage 3 runs exactly these cases; stale or failed exports are left out (and listed)."""
    cases, stale = [], []
    conf = dt.study_config(study)
    for r in dt.read_designs(study):
        mp = os.path.join(dt.geometry_dir(study, r["case"]), "meta.json")
        m = json.load(open(mp)) if os.path.exists(mp) else {}
        if m.get("status") == "ok" and m.get("signature") == signature(study, r["params"], m.get("signature", {}).get("res", res), conf, r["case"]):
            cases.append({"case": r["case"], "label": m["label"]})
        else:
            stale.append(r["case"])
    os.makedirs(cfg.geometry_root(study), exist_ok=True)
    json.dump({"study": study, "base": dt.base_id(study), "cases": cases, "not_exported": stale},
              open(os.path.join(cfg.geometry_root(study), "index.json"), "w"), indent=2)
    return cases, stale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study")
    ap.add_argument("--cases", nargs="+")
    ap.add_argument("--res", type=float, default=0.5, help="surface resolution (0.5 = CFD default)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    rows = [r for r in dt.read_designs(a.study) if not a.cases or r["case"] in a.cases]
    if not rows:
        raise SystemExit(f"no designs (create them with doe.py)")
    print(f"exporting {len(rows)} designs of '{a.study}' -> {os.path.join(dt.study_dir(a.study), 'geometry')}")
    failed = []
    for r in rows:
        try:
            export_case(a.study, r["case"], r["params"], a.res, a.force)
        except Exception as e:  # noqa: BLE001
            print(f"  {r['case']}: FAILED {e}")
            failed.append(r["case"])
    write_index(a.study, a.res)
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
