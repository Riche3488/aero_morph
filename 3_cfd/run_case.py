#!/usr/bin/env python3
"""
Run one design of a study through the full CFD chain (resumable):

  prepare -> mesh -> init -> solve -> post

  python3 3_cfd/run_case.py <study> <case> [--np 8] [--iters N] [--init map|potential]
                            [--map-from CASE_DIR] [--stop-after STAGE] [--clean]

Geometry comes from stage 2 (2_shape_morphing/studies/<study>/geometry/<case>/,
exported on demand), so parametric and drag-morph designs are handled the same way.

Each finished stage leaves a marker file (.done_<stage>) in the case directory,
so a crashed/interrupted run continues where it stopped. Use --redo STAGE to
force a stage (and everything after it) to run again.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import time

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline_config as cfg  # noqa: E402

cfg.add_stage_paths()
from design_table import geometry_dir, is_parametric, label, read_designs, study_config  # noqa: E402
from export_designs import export_case  # noqa: E402

FOAM_BASHRC, TEMPLATE, BASELINE_CASE = cfg.FOAM_BASHRC, cfg.CFD_TEMPLATE, cfg.CFD_BASELINE
STAGES = ["prepare", "mesh", "init", "solve", "post"]
# stage-2 part name -> OpenFOAM patch / file name (half model: +y side only)
STL_PARTS = {"body": "body", "wheel_FL": "wheelFront", "wheel_RL": "wheelRear", "mirror_L": "mirror"}


def case_dir(study, case):
    return os.path.join(cfg.CFD_RUNS, study, "cases", case)


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

    meta = export_case(opt.study, case, params, quiet=True)      # stage-2 geometry (cached)
    src = geometry_dir(opt.study, case)
    area = meta["frontal_area"]
    gdir = os.path.join(cdir, "constant", "geometry")
    os.makedirs(gdir, exist_ok=True)
    names = []
    layout = meta.get("cfd_layout", "separate_parts")
    wall = "patchInfo { type wall; }"
    if layout == "fused_regions":
        # one closed STL whose solids body / wheelFront / wheelRear become separate patches
        shutil.copy(os.path.join(src, "body.stl"), os.path.join(gdir, "body.stl"))
        names = ["body"]
        regs = [r for r in ("wheelFront", "wheelRear") if meta["parts"]["body"].get("regions", {}).get(r)]
        g = ('    body { type triSurfaceMesh; file "body.stl";\n           regions { body { name body; } '
             + " ".join(f"{r} {{ name {r}; }}" for r in regs) + " } }")
        surfs = (f"        body {{ level (5 5); {wall}\n               regions {{ "
                 + " ".join(f"{r} {{ level (4 5); {wall} }}" for r in regs) + " } }")
        patches = ["body"] + regs
    else:
        for part, patch in STL_PARTS.items():
            f = os.path.join(src, part + ".stl")
            if os.path.exists(f):
                shutil.copy(f, os.path.join(gdir, patch + ".stl"))
                names.append(patch)
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

    base = study_config(opt.study)["base"]
    json.dump({"case": case, "study": opt.study, "label": label(params), "params": params, "frontal_area": area,
               "geometry_source": src, "vehicle": meta.get("vehicle", "IONIQ 5"), "cfd_layout": layout,
               "patches": patches, "wheels": w,
               "reference_cd": base.get("reference_cd", 0.288 if base["type"] == "ioniq5_parametric" else None),
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
    foam_set(cdir, "endTime", n_a, cd)
    foam_set(cdir, "writeInterval", opt.iters, cd)
    foam_set(cdir, "purgeWrite", 1, cd)
    sh(f"{mpi} foamRun -parallel", cdir, "log.foamRun.stageA")
    # stage B: second order, standard relaxation
    foam_set(cdir, "divSchemes/div(phi,U)", "bounded Gauss linearUpwind limited", fs)
    foam_set(cdir, "divSchemes/turbulence", "bounded Gauss limitedLinear 1", fs)
    foam_set(cdir, "relaxationFactors/fields/p", 0.3, fv)
    foam_set(cdir, "relaxationFactors/equations/U", 0.7, fv)
    foam_set(cdir, "relaxationFactors/equations/\"(k|omega|epsilon)\"", 0.7, fv)
    foam_set(cdir, "endTime", opt.iters, cd)
    t = sh(f"{mpi} foamRun -parallel", cdir, "log.foamRun.stageB")
    if re.search(r"\bnan\b", t, re.I):
        raise RuntimeError("solution diverged (NaN)")
    for d in glob.glob(os.path.join(cdir, "processor*", str(n_a))):   # drop stage-A snapshot
        if n_a != opt.iters:
            shutil.rmtree(d, ignore_errors=True)
    return {}


def stage_post(cdir, opt):
    sh("python3 post.py", cdir, "log.post")
    res = json.load(open(os.path.join(cdir, "result.json")))
    if opt.clean and not getattr(opt, "is_reference", False):
        for d in glob.glob(os.path.join(cdir, "processor*")):
            shutil.rmtree(d, ignore_errors=True)
    return {"Cd": res["Cd"], "CdA": res["CdA"]}


# ----------------------------------------------------------------------------- driver
def run(study, case, opt):
    rows = {r["case"]: r for r in read_designs(study)}
    if case not in rows:
        raise SystemExit(f"case {case} not in 2_shape_morphing/studies/{study}/designs.csv")
    params = rows[case]["params"]
    opt.study = study
    cdir = case_dir(study, case)
    os.makedirs(cdir, exist_ok=True)
    resolve_init(study, case, rows, opt)
    if opt.redo:
        for s in STAGES[STAGES.index(opt.redo):]:
            p = os.path.join(cdir, f".done_{s}")
            os.path.exists(p) and os.remove(p)
    if os.path.exists(os.path.join(cdir, "case.json")) and done(cdir, "init"):  # resume: keep what init used
        c = json.load(open(os.path.join(cdir, "case.json")))
        opt.init, opt.iters, opt.n_avg = c["init"], c["n_iter"], c["n_avg"]

    print(f"[{study}/{case}] {label(params)}")
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


def has_solution(case_path):
    return any(os.path.basename(d).replace(".", "").isdigit() and float(os.path.basename(d)) > 0
               for d in glob.glob(os.path.join(case_path, "processor0", "*")))


def resolve_init(study, case, rows, opt):
    """Pick the initial field: explicit --map-from > study's own converged baseline (c000) >
    global IONIQ 5 baseline (IONIQ 5 studies only) > potentialFoam."""
    ref = next((c for c, r in rows.items() if label(r["params"]) == "baseline"), None)
    opt.is_reference = (case == ref)
    if opt.init == "potential":
        src = None
    elif opt.map_from:
        src = opt.map_from
    elif ref and case != ref and has_solution(case_dir(study, ref)):
        src = case_dir(study, ref)
    elif is_parametric(study):
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
