#!/usr/bin/env python3
"""
Gather every result actually produced for the IONIQ 5 (gen3d) vehicle into test_ioniq5/.

  python3 test_ioniq5/collect_results.py

Copies (never moves) the lightweight artefacts of each pipeline stage, so the pipeline folders stay
untouched and this script can be re-run after more designs have been simulated:

  01_gen3d_reconstruction/   photos -> TRELLIS candidates -> rough / final mesh (gen3d/runs/ioniq5)
  02_base_geometry/          repaired base surface ioniq5_fixed.stl + repair report (gen3d/output)
  03_shape_morphing/         study definition, DOE table, per-design meta.json (2_shape_morphing/studies/ioniq5)
  04_cfd/                    OpenFOAM cases that ran: setup dicts, logs, force histories, plots, results

Not copied (regenerable and large): the exported morph STLs (~236 MB each, export_designs.py ioniq5),
the OpenFOAM mesh / processor* fields, and the TRELLIS candidate meshes.
"""
import glob
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEN3D_RUN = os.path.join(ROOT, "gen3d", "runs", "ioniq5")
GEN3D_OUT = os.path.join(ROOT, "gen3d", "output")
STUDY = os.path.join(ROOT, "2_shape_morphing", "studies", "ioniq5")
CFD = os.path.join(ROOT, "3_cfd", "runs", "ioniq5")


def copy(src, dst_dir, name=None):
    if not os.path.exists(src):
        return None
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, name or os.path.basename(src))
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)
              and os.path.getmtime(dst) >= os.path.getmtime(src)):
        shutil.copy2(src, dst)
    return dst


def gen3d():
    d = os.path.join(HERE, "01_gen3d_reconstruction")
    for f in ["ATTRIBUTION.md", "report.json", "inputs_sheet.png", "candidates_sheet.png",
              "ioniq5_rough_views.png", "ioniq5_rough_inspect.png", "ioniq5_final_views.png",
              "ioniq5_final_inspect.png", "ioniq5_rough.stl", "ioniq5_final.glb"]:
        copy(os.path.join(GEN3D_RUN, f), d)
    for f in glob.glob(os.path.join(GEN3D_RUN, "inputs", "*")):
        copy(f, os.path.join(d, "inputs"))
    for c in sorted(glob.glob(os.path.join(GEN3D_RUN, "candidates", "*"))):
        for f in ["info.json", "views.png"]:
            copy(os.path.join(c, f), os.path.join(d, "candidates", os.path.basename(c)))


def base_geometry():
    d = os.path.join(HERE, "02_base_geometry")
    for f in ["ioniq5_fixed.stl", "ioniq5_fixed_report.json", "ioniq5_fixed_views.png"]:
        copy(os.path.join(GEN3D_OUT, f), d)


def morphing():
    d = os.path.join(HERE, "03_shape_morphing")
    for f in ["study.json", "designs.csv"]:
        copy(os.path.join(STUDY, f), d)
    copy(os.path.join(STUDY, "geometry", "index.json"), d, "export_index.json")
    for f in glob.glob(os.path.join(STUDY, "previews", "*")):
        copy(f, os.path.join(d, "previews"))
    for m in sorted(glob.glob(os.path.join(STUDY, "geometry", "c*", "meta.json"))):
        copy(m, os.path.join(d, "designs", os.path.basename(os.path.dirname(m))))


def cfd():
    d = os.path.join(HERE, "04_cfd")
    for f in ["results.csv", "report"] + [os.path.basename(p) for p in glob.glob(os.path.join(CFD, "*.log"))]:
        copy(os.path.join(CFD, f), d)
    done = []
    for cdir in sorted(glob.glob(os.path.join(CFD, "cases", "c*"))):
        case = os.path.basename(cdir)
        out = os.path.join(d, "cases", case)
        for f in ["result.json", "status.json", "case.json", "post.py"]:
            copy(os.path.join(cdir, f), out)
        for f in glob.glob(os.path.join(cdir, "log.*")):
            copy(f, os.path.join(out, "logs"))
        copy(os.path.join(cdir, "system"), os.path.join(out, "setup"))
        for f in glob.glob(os.path.join(cdir, "constant", "*")):
            if os.path.isfile(f):
                copy(f, os.path.join(out, "setup", "constant"))
        copy(os.path.join(cdir, "0"), os.path.join(out, "setup"))
        for f in glob.glob(os.path.join(cdir, "postProcessing", "force*")):
            copy(f, os.path.join(out, "postProcessing"))
        copy(os.path.join(cdir, "results"), out, "plots")
        if os.path.exists(os.path.join(cdir, "result.json")):
            done.append(case)
    write_status(d)
    return done


def write_status(d):
    """04_cfd/STATUS.md: one row per CFD case (state, mesh, Cd) straight from status.json / result.json."""
    idx = json.load(open(os.path.join(STUDY, "geometry", "index.json")))
    rows = ["| case | design | state | cells | layer cover | Cd | CdA [m²] | ΔCdA vs c000 |",
            "|---|---|---|---|---|---|---|---|"]
    base = None
    for c in idx["cases"]:
        cdir = os.path.join(CFD, "cases", c["case"])
        st = json.load(open(os.path.join(cdir, "status.json"))) if os.path.exists(os.path.join(cdir, "status.json")) else {}
        rs = json.load(open(os.path.join(cdir, "result.json"))) if os.path.exists(os.path.join(cdir, "result.json")) else {}
        if c["case"] == "c000" and rs:
            base = rs["CdA"]
        mesh = st.get("mesh", {})
        cov = mesh.get("layer_coverage_pct", {}).get("body")
        rows.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            c["case"], c["label"], st.get("state", "not started"),
            f"{mesh['cells'] / 1e6:.2f} M" if mesh.get("cells") else "",
            f"{cov:.1f} %" if cov is not None else "",
            f"{rs['Cd']:.4f}" if rs else "", f"{rs['CdA']:.4f}" if rs else "",
            f"{(rs['CdA'] / base - 1) * 100:+.2f} %" if rs and base else ""))
    open(os.path.join(d, "STATUS.md"), "w").write(
        "# CFD case status (generated by collect_results.py)\n\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    gen3d(); base_geometry(); morphing()
    done = cfd()
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(HERE) for f in fs)
    print(f"collected into {HERE} ({total / 1e6:.0f} MB); finished CFD cases: {', '.join(done) or 'none'}")
