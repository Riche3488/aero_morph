#!/usr/bin/env python3
"""
Run one design of a study through the full CFD chain (resumable):

  prepare -> mesh -> init -> solve -> post

  python3 3_cfd/run_case.py <study> <case> [--np 8] [--iters N] [--init map|potential]
                            [--map-from CASE_DIR] [--stop-after STAGE] [--clean]

Input = the stage-2 hand-off folder only: 2_shape_morphing/studies/<study>/geometry/
  index.json          cases to run (written by export_designs.py / preview_designs.py)
  <case>/meta.json    design variables, label, frontal area, wheels, CFD layout
  <case>/*.stl        surfaces (m)
Stage 3 imports no stage-1/2 code and never regenerates geometry: export it first with
  python3 2_shape_morphing/export_designs.py <study>
The STLs are copied into the case (constant/geometry) and their hash is stored in case.json;
if the hand-off geometry changes later, the case is reported as changed (--rerun-changed).

Each finished stage leaves a marker file (.done_<stage>) in the case directory,
so a crashed/interrupted run continues where it stopped. Use --redo STAGE to
force a stage (and everything after it) to run again.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline_config as cfg  # noqa: E402

FOAM_BASHRC, TEMPLATE, BASELINE_CASE = cfg.FOAM_BASHRC, cfg.CFD_TEMPLATE, cfg.CFD_BASELINE
STAGES = ["prepare", "mesh", "init", "solve", "post"]
# stage-2 part name -> OpenFOAM patch / file name (half model: +y side only)
STL_PARTS = {"body": "body", "wheel_FL": "wheelFront", "wheel_RL": "wheelRear", "mirror_L": "mirror"}


def case_dir(study, case):
    return os.path.join(cfg.CFD_RUNS, study, "cases", case)


# ----------------------------------------------------------------------------- stage-2 hand-off
def _export_hint(study):
    return f"export it first: python3 2_shape_morphing/export_designs.py {study}"


def study_cases(study):
    """Cases of the hand-off folder: index.json (written by stage 2) or, for older exports
    without an index, every <case>/meta.json with status ok. -> [{"case", "label"}]"""
    root = cfg.geometry_root(study)
    idx = os.path.join(root, "index.json")
    if os.path.exists(idx):
        return json.load(open(idx))["cases"]
    out = []
    for c in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        mp = os.path.join(root, c, "meta.json")
        if not c.startswith("_") and os.path.exists(mp):
            m = json.load(open(mp))
            if m.get("status") == "ok":
                out.append({"case": c, "label": m.get("label", c)})
    return out


def load_meta(study, case):
    mp = os.path.join(cfg.geometry_dir(study, case), "meta.json")
    if not os.path.exists(mp):
        raise SystemExit(f"no exported geometry for {study}/{case} ({mp}); {_export_hint(study)}")
    meta = json.load(open(mp))
    if meta.get("status") != "ok":
        raise SystemExit(f"geometry export of {study}/{case} is not ok; {_export_hint(study)}")
    return meta


def geometry_files(study, case, meta):
    """[(file name in constant/geometry, source STL)] of the hand-off geometry, in patch order."""
    src = cfg.geometry_dir(study, case)
    if meta.get("cfd_layout", "separate_parts") == "fused_regions":
        files = [("body.stl", os.path.join(src, "body.stl"))]
    else:
        files = [(patch + ".stl", os.path.join(src, part + ".stl")) for part, patch in STL_PARTS.items()]
    return [(n, f) for n, f in files if os.path.exists(f)]


def _update_stl(h, path, chunk=1 << 22):
    """Feed the STL bytes that define the surface to h, streamed (fused ASCII STLs are hundreds
    of MB): a binary STL's 80-byte header (solid name only) is skipped, ASCII STLs are taken whole
    (their solid names are the patch regions)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(84)
        n = int.from_bytes(head[80:84], "little") if len(head) >= 84 else -1
        h.update(head[80:] if size == 84 + 50 * n else head)
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)


_HASH_MEMO = {}
HASH_CACHE = os.path.join(cfg.CFD_RUNS, ".geometry_hash_cache.json")


def _hash(named_paths):
    """Geometry hash of [(name, path)] (same value as always: sha1 over name + surface bytes).
    Memoised per process and cached on disk by (path, size, mtime): a study run checks every
    case's geometry several times (run_study, run_case, compare), and re-reading hundreds of MB
    per check would dominate."""
    key = json.dumps([(n, os.path.abspath(p), os.path.getsize(p), os.stat(p).st_mtime_ns) for n, p in named_paths])
    if key in _HASH_MEMO:
        return _HASH_MEMO[key]
    try:
        disk = json.load(open(HASH_CACHE))
    except (OSError, ValueError):
        disk = {}
    if key not in disk:
        h = hashlib.sha1()
        for name, path in named_paths:
            h.update(name.encode()); _update_stl(h, path)
        disk = {k: v for k, v in disk.items()                           # drop entries of rewritten / deleted files
                if all(os.path.exists(p) and os.path.getsize(p) == sz and os.stat(p).st_mtime_ns == mt
                       for _, p, sz, mt in json.loads(k))}
        disk[key] = h.hexdigest()[:16]
        try:                                                            # atomic: parallel runs may race
            os.makedirs(os.path.dirname(HASH_CACHE), exist_ok=True)
            tmp = f"{HASH_CACHE}.{os.getpid()}"
            json.dump(disk, open(tmp, "w"))
            os.replace(tmp, HASH_CACHE)
        except OSError:
            pass
    _HASH_MEMO[key] = disk[key]
    return disk[key]


def geometry_changed(study, case):
    """True if the hand-off geometry differs from the geometry the CFD case was built from,
    None if that cannot be told (case not prepared, or its geometry copy is gone)."""
    cdir = case_dir(study, case)
    cj = os.path.join(cdir, "case.json")
    mp = os.path.join(cfg.geometry_dir(study, case), "meta.json")
    if not (os.path.exists(cj) and os.path.exists(mp)):
        return None
    files = geometry_files(study, case, json.load(open(mp)))
    if not files:
        return None
    used = json.load(open(cj)).get("geometry_hash")
    if used is None:                                   # older cases: compare with their own copy
        copies = [(n, os.path.join(cdir, "constant", "geometry", n)) for n, _ in files]
        if not all(os.path.exists(p) for _, p in copies):
            return None
        used = _hash(copies)
    return used != _hash(files)


# ----------------------------------------------------------------------------- helpers
def sh(cmd, cwd, log):
    """Run a shell command with OpenFOAM sourced; output -> log file. Raises on failure."""
    full = f"source {FOAM_BASHRC} >/dev/null 2>&1; {cmd}"
    with open(os.path.join(cwd, log), "w") as f:
        r = subprocess.run(["bash", "-c", full], cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    text = open(os.path.join(cwd, log)).read()
    if r.returncode != 0 or "FOAM FATAL" in text:
        raise RuntimeError(f"'{cmd}' failed (see {os.path.join(cwd, log)})")
    return text


def foam_set(cwd, entry, value, file):
    sh(f"foamDictionary -entry '{entry}' -set '{value}' {file}", cwd, "log.foamDictionary")


def edit(path, fn):
    """Read-modify-write a text file (read fully before truncating)."""
    with open(path) as f:
        text = f.read()
    with open(path, "w") as f:
        f.write(fn(text))


def done(cdir, stage):
    return os.path.exists(os.path.join(cdir, f".done_{stage}"))


def mark(cdir, stage, info=None):
    with open(os.path.join(cdir, f".done_{stage}"), "w") as f:
        json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})}, f)


def set_status(cdir, **kw):
    p = os.path.join(cdir, "status.json")
    st = json.load(open(p)) if os.path.exists(p) else {}
    st.update(kw)
    json.dump(st, open(p, "w"), indent=2)


# ----------------------------------------------------------------------------- stages
def stage_prepare(cdir, case, params, opt):
    if os.path.isdir(cdir):
        keep = {"status.json"}
        for e in os.listdir(cdir):
            if e not in keep:
                p = os.path.join(cdir, e)
                shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    shutil.copytree(TEMPLATE, cdir, dirs_exist_ok=True)

    meta = load_meta(opt.study, case)                            # stage-2 hand-off (read only)
    src = cfg.geometry_dir(opt.study, case)
    area = meta["frontal_area"]
    gdir = os.path.join(cdir, "constant", "geometry")
    os.makedirs(gdir, exist_ok=True)
    files = geometry_files(opt.study, case, meta)
    for n, f in files:                                           # the case keeps its own copy
        shutil.copy(f, os.path.join(gdir, n))
    names = [n[:-4] for n, _ in files]
    layout = meta.get("cfd_layout", "separate_parts")
    wall = "patchInfo { type wall; }"
    if layout == "fused_regions":
        # one closed STL whose solids body / wheelFront / wheelRear become separate patches
        regs = [r for r in ("wheelFront", "wheelRear") if meta["parts"]["body"].get("regions", {}).get(r)]
        g = ('    body { type triSurfaceMesh; file "body.stl";\n           regions { body { name body; } '
             + " ".join(f"{r} {{ name {r}; }}" for r in regs) + " } }")
        surfs = (f"        body {{ level (5 5); {wall}\n               regions {{ "
                 + " ".join(f"{r} {{ level (4 5); {wall} }}" for r in regs) + " } }")
        patches = ["body"] + regs
    else:
        g = "\n".join(f'    {n:10s} {{ type triSurfaceMesh; file "{n}.stl"; }}' for n in names)
        lv = {"body": "(5 5)", "mirror": "(5 6)"}
        surfs = "\n".join(f"        {n:10s} {{ level {lv.get(n, '(4 5)')}; {wall} }}" for n in names)
        patches = names
    if "body" not in names:
        raise RuntimeError(f"no body.stl in {src}")

    feats = "\n".join(f'        {{ file "{n}.eMesh"; level 5; }}' for n in names)
    edit(os.path.join(cdir, "system", "snappyHexMeshDict"),
         lambda t: t.replace("@GEOMETRY@", g).replace("@FEATURES@", feats).replace("@SURFACES@", surfs))
    w = meta["wheels"]
    fill = {"@XFRONT@": w["front"]["x"], "@YFRONT@": w["front"]["y"], "@XREAR@": w["rear"]["x"],
            "@YREAR@": w["rear"]["y"], "@WHEEL_R@": w["front"]["z"],
            "@LREF@": w["rear"]["x"] - w["front"]["x"], "@XMID@": 0.5 * (w["rear"]["x"] + w["front"]["x"])}
    for f in ("0/U", "system/forceCoeffsIncompressible"):
        edit(os.path.join(cdir, f), lambda t: _fill(t, fill))
    edit(os.path.join(cdir, "system", "surfaceFeaturesDict"),
         lambda t: t.replace("@SURFACES@", " ".join(f'"{n}.stl"' for n in names)))
    edit(os.path.join(cdir, "system", "forceCoeffsIncompressible"), lambda t: t.replace("@AREF@", f"{area / 2:.5f}"))
    edit(os.path.join(cdir, "system", "decomposeParDict"),
         lambda t: re.sub(r"numberOfSubdomains\s+\d+;", f"numberOfSubdomains  {opt.np};", t))

    ref_cd = meta.get("reference_cd", 0.288 if meta.get("base") == cfg.IONIQ5_PARAMETRIC else None)
    json.dump({"case": case, "study": opt.study, "label": meta["label"], "params": params,
               "variables": meta.get("variables", {}), "frontal_area": area,
               "geometry_source": cfg.rel(src), "geometry_hash": _hash(files), "base": meta.get("base"),
               "vehicle": meta.get("vehicle", "IONIQ 5"), "cfd_layout": layout,
               "patches": patches, "wheels": w, "reference_cd": ref_cd,
               "n_iter": opt.iters, "n_avg": opt.n_avg, "init": opt.init},
              open(os.path.join(cdir, "case.json"), "w"), indent=2)
    return {"frontal_area": area, "patches": patches}


def _fill(text, fill):
    for k, v in fill.items():
        text = text.replace(k, f"{v:.6g}")
    return text


def stage_mesh(cdir, opt):
    mpi = f"mpirun -np {opt.np}"
    sh("blockMesh", cdir, "log.blockMesh")
    sh("surfaceFeatures", cdir, "log.surfaceFeatures")
    sh("decomposePar -force -copyZero", cdir, "log.decomposePar")
    t = sh(f"{mpi} snappyHexMesh -parallel -overwrite", cdir, "log.snappyHexMesh")
    if "Finished meshing without any errors" not in t:
        raise RuntimeError("snappyHexMesh did not finish cleanly")
    c = sh(f"{mpi} checkMesh -parallel", cdir, "log.checkMesh")
    cells = int(re.search(r"cells:\s+(\d+)", c).group(1))
    failed = re.search(r"Failed (\d+) mesh checks", c)
    layers = re.findall(r"^(body|mirror)\s+\d+\s+([\d.]+)\s+[\d.e-]+\s+([\d.]+)", t, re.M)
    return {"cells": cells, "failed_checks": int(failed.group(1)) if failed else 0,
            "layer_coverage_pct": {n: float(p) for n, _, p in layers}}


def stage_init(cdir, opt):
    mpi = f"mpirun -np {opt.np}"
    src = opt.map_from
    if opt.init == "map":
        latest = [d for d in glob.glob(os.path.join(src, "processor0", "*")) if os.path.basename(d).replace(".", "").isdigit()
                  and float(os.path.basename(d)) > 0]
        if not latest:
            print(f"    no decomposed solution in {src}; falling back to potentialFoam")
            opt.init = "potential"
        else:
            try:
                sh(f"{mpi} mapFieldsPar {src} -sourceTime latestTime -consistent -parallel", cdir, "log.mapFieldsPar")
                return {"init": "map", "source": src}
            except RuntimeError as e:
                print(f"    mapFieldsPar failed ({e}); falling back to potentialFoam")
                opt.init = "potential"
    sh(f"{mpi} potentialFoam -parallel", cdir, "log.potentialFoam")
    # potential start needs the longer schedule
    opt.iters, opt.n_avg = max(opt.iters, 1500), max(opt.n_avg, 500)
    cj = os.path.join(cdir, "case.json")
    c = json.load(open(cj)); c.update(init="potential", n_iter=opt.iters, n_avg=opt.n_avg)
    json.dump(c, open(cj, "w"), indent=2)
    return {"init": "potential"}


def stage_solve(cdir, opt):
    mpi = f"mpirun -np {opt.np}"
    cd, fs, fv = "system/controlDict", "system/fvSchemes", "system/fvSolution"
    # stage A: robust start
    if opt.init == "map":
        n_a, scheme_a, turb_a = 50, "bounded Gauss linearUpwind limited", "bounded Gauss limitedLinear 1"
    else:
        n_a, scheme_a, turb_a = 300, "bounded Gauss upwind", "bounded Gauss upwind"
    foam_set(cdir, "divSchemes/div(phi,U)", scheme_a, fs)
    foam_set(cdir, "divSchemes/turbulence", turb_a, fs)
    foam_set(cdir, "relaxationFactors/fields/p", 0.2, fv)
    foam_set(cdir, "relaxationFactors/equations/U", 0.5, fv)
    foam_set(cdir, "relaxationFactors/equations/\"(k|omega|epsilon)\"", 0.5, fv)
    n_a = max(1, min(n_a, opt.iters // 2))
    foam_set(cdir, "endTime", n_a, cd)
    foam_set(cdir, "writeInterval", n_a, cd)        # stage A must be written: stage B restarts from it
    foam_set(cdir, "purgeWrite", 1, cd)
    _clear_fo_output(cdir, _latest_time(cdir))
    sh(f"{mpi} foamRun -parallel", cdir, "log.foamRun.stageA")
    if not glob.glob(os.path.join(cdir, "processor0", str(n_a))):
        raise RuntimeError(f"stage A wrote no time {n_a}; stage B would restart from 0")
    # stage B: second order, standard relaxation
    foam_set(cdir, "divSchemes/div(phi,U)", "bounded Gauss linearUpwind limited", fs)
    foam_set(cdir, "divSchemes/turbulence", "bounded Gauss limitedLinear 1", fs)
    foam_set(cdir, "relaxationFactors/fields/p", 0.3, fv)
    foam_set(cdir, "relaxationFactors/equations/U", 0.7, fv)
    foam_set(cdir, "relaxationFactors/equations/\"(k|omega|epsilon)\"", 0.7, fv)
    foam_set(cdir, "writeInterval", opt.iters, cd)
    foam_set(cdir, "endTime", opt.iters, cd)
    _clear_fo_output(cdir, n_a)
    t = sh(f"{mpi} foamRun -parallel", cdir, "log.foamRun.stageB")   # startFrom latestTime = n_a
    if re.search(r"\bnan\b", t, re.I):
        raise RuntimeError("solution diverged (NaN)")
    for d in glob.glob(os.path.join(cdir, "processor*", str(n_a))):   # drop stage-A snapshot
        if n_a != opt.iters:
            shutil.rmtree(d, ignore_errors=True)
    return {}


def _latest_time(cdir):
    """Latest time directory of the decomposed case (where foamRun -parallel starts)."""
    t = [float(os.path.basename(d)) for d in glob.glob(os.path.join(cdir, "processor0", "*"))
         if re.fullmatch(r"\d+(\.\d+)?", os.path.basename(d))]
    return max(t) if t else 0


def _clear_fo_output(cdir, start):
    """Remove function-object output that a previous (killed / retried) run left in
    postProcessing/<fo>/<start>/. Otherwise OpenFOAM writes the new run next to it as
    <file>_<start>.dat and the old, partial file keeps the plain name."""
    name = f"{start:g}"
    for d in glob.glob(os.path.join(cdir, "postProcessing", "*", name)):
        shutil.rmtree(d, ignore_errors=True)


def stage_post(cdir, opt):
    # post.py has no case-specific edits (it reads case.json): always run the current template,
    # so fixes to it also reach cases that were prepared earlier
    shutil.copy(os.path.join(cfg.CFD_TEMPLATE, "post.py"), os.path.join(cdir, "post.py"))
    sh("python3 post.py", cdir, "log.post")
    res = json.load(open(os.path.join(cdir, "result.json")))
    if opt.clean and not getattr(opt, "is_reference", False):
        for d in glob.glob(os.path.join(cdir, "processor*")):
            shutil.rmtree(d, ignore_errors=True)
    return {"Cd": res["Cd"], "CdA": res["CdA"]}


# ----------------------------------------------------------------------------- driver
def run(study, case, opt):
    cases = {c["case"]: c for c in study_cases(study)}
    if case not in cases:
        raise SystemExit(f"case {case} is not in the exported geometry of '{study}' "
                         f"({cfg.geometry_root(study)}/index.json); {_export_hint(study)}")
    meta = load_meta(study, case)
    params = meta["params"]
    opt.study = study
    cdir = case_dir(study, case)
    os.makedirs(cdir, exist_ok=True)
    resolve_init(study, case, cases, meta, opt)
    if done(cdir, "prepare") and opt.redo != "prepare" and geometry_changed(study, case):
        if not getattr(opt, "rerun_changed", False):
            print(f"[{study}/{case}] the exported geometry changed since this case was prepared; "
                  f"re-run with --rerun-changed (or --redo prepare)")
            set_status(cdir, state="geometry changed")
            return False
        opt.redo = "prepare"
    if opt.redo:
        for s in STAGES[STAGES.index(opt.redo):]:
            p = os.path.join(cdir, f".done_{s}")
            os.path.exists(p) and os.remove(p)
    if os.path.exists(os.path.join(cdir, "case.json")) and done(cdir, "init"):  # resume: keep what init used
        c = json.load(open(os.path.join(cdir, "case.json")))
        opt.init, opt.iters, opt.n_avg = c["init"], c["n_iter"], c["n_avg"]

    print(f"[{study}/{case}] {meta['label']}")
    set_status(cdir, state="running", error=None)
    t_case = time.time()
    for st in STAGES:
        if done(cdir, st):
            print(f"  {st:8s} (done)")
            continue
        t0 = time.time()
        print(f"  {st:8s} ...", end="", flush=True)
        try:
            fn = {"prepare": lambda: stage_prepare(cdir, case, params, opt), "mesh": lambda: stage_mesh(cdir, opt),
                  "init": lambda: stage_init(cdir, opt), "solve": lambda: stage_solve(cdir, opt),
                  "post": lambda: stage_post(cdir, opt)}[st]
            info = fn() or {}
        except Exception as e:  # noqa: BLE001
            print(f" FAILED: {e}")
            set_status(cdir, state="failed", stage=st, error=str(e))
            return False
        info["seconds"] = round(time.time() - t0, 1)
        mark(cdir, st, info)
        set_status(cdir, **{st: info})
        print(f" {info['seconds']:.0f}s  " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                      for k, v in info.items() if k != "seconds"))
        if opt.stop_after == st:
            set_status(cdir, state=f"stopped after {st}")
            return True
    set_status(cdir, state="done", wall_seconds=round(time.time() - t_case, 1))
    return True


def add_args(ap):
    ap.add_argument("--np", type=int, default=8, help="MPI processes (default 8 = physical cores)")
    ap.add_argument("--init", choices=["map", "potential"], default="map",
                    help="map = start from converged baseline (fast), potential = potentialFoam (robust)")
    ap.add_argument("--iters", type=int, default=None, help="total iterations (default: map 800, potential 1500)")
    ap.add_argument("--n-avg", type=int, default=None, help="averaging window (default: map 300, potential 500)")
    ap.add_argument("--map-from", default=None,
                    help="decomposed converged case for the initial field (default: study baseline c000 if solved, "
                         "else 3_cfd/baseline_case for IONIQ 5 studies, else potentialFoam)")
    ap.add_argument("--stop-after", choices=STAGES, help="stop after this stage (e.g. prepare or mesh)")
    ap.add_argument("--redo", choices=STAGES, help="re-run from this stage")
    ap.add_argument("--clean", action="store_true", help="delete processor* dirs after post (saves ~600 MB/case)")
    ap.add_argument("--rerun-changed", action="store_true",
                    help="re-run (from prepare) cases whose exported geometry changed since they were run")


def has_solution(case_path):
    return any(os.path.basename(d).replace(".", "").isdigit() and float(os.path.basename(d)) > 0
               for d in glob.glob(os.path.join(case_path, "processor0", "*")))


def resolve_init(study, case, cases, meta, opt):
    """Pick the initial field: explicit --map-from > study's own converged baseline (c000) >
    global IONIQ 5 baseline (IONIQ 5 studies only) > potentialFoam."""
    ref = next((c for c, r in cases.items() if r["label"] == "baseline"), None)
    opt.is_reference = (case == ref)
    if opt.init == "potential":
        src = None
    elif opt.map_from:
        src = opt.map_from
    elif ref and case != ref and has_solution(case_dir(study, ref)):
        src = case_dir(study, ref)
    elif meta.get("base") == cfg.IONIQ5_PARAMETRIC:
        src = BASELINE_CASE
    else:
        src = None                      # new vehicle without a converged reference yet
    opt.map_from = os.path.abspath(src) if src else None
    opt.init = "map" if src else "potential"
    if opt.iters_arg is None:
        opt.iters = 800 if opt.init == "map" else 1500
    if opt.n_avg_arg is None:
        opt.n_avg = 300 if opt.init == "map" else 500


def finalize_opts(opt):
    opt.iters_arg, opt.n_avg_arg = opt.iters, opt.n_avg
    return opt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study")
    ap.add_argument("case")
    add_args(ap)
    o = finalize_opts(ap.parse_args())
    raise SystemExit(0 if run(o.study, o.case, o) else 1)
