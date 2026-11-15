#!/usr/bin/env python3
"""
Create / extend a design table: 2_shape_morphing/studies/<study>/designs.csv

Design variables are the parametric morph parameters (IONIQ 5 base), drag-mode amplitudes
(`mode:<name>` for every 2_shape_morphing/modes/<name>.json) and, for an STL base, the
dimensional variables `dim:foh dim:roh dim:length dim:width dim:spoiler` (mm, dim_morph.py).

Examples
  python3 2_shape_morphing/doe.py --list
  python3 2_shape_morphing/doe.py rear --oat tail_taper roof_rear_drop diffuser_angle
  python3 2_shape_morphing/doe.py rear --sweep mode:rear_roof_drop=0.5,1,1.5
  python3 2_shape_morphing/doe.py lhs --lhs 16 --params tail_taper diffuser_angle mode:front_bumper_push --seed 1
  python3 2_shape_morphing/doe.py rear --add tail_taper=40 mode:rear_roof_drop=1

  # any STL as the base (drag modes only); gen3d models read their wheels from report.json
  python3 2_shape_morphing/doe.py ioniq6 --base-stl gen3d/runs/ioniq6/ioniq6_rough.stl
  python3 2_shape_morphing/doe.py ioniq6 --sweep mode:ioniq6_trunk=0.5,1
  python3 2_shape_morphing/doe.py ioniq6 --oat dim:foh dim:roh dim:length dim:width dim:spoiler
  # other STL: --wheels x_front x_rear track_front track_rear tire_diameter tire_width  (m)
"""
import argparse
import sys

import numpy as np

import design_table as dt


def bounds(name, study):
    if name.startswith(dt.MODE_PREFIX):
        lo, hi = dt.mode_range(name[len(dt.MODE_PREFIX):])
        return 0.0, lo, hi
    if name.startswith(dt.DIM_PREFIX):
        d = name[len(dt.DIM_PREFIX):]
        if d not in dt.dim_morph.PARAMS:
            sys.exit(f"unknown variable '{name}' (see --list)")
        if dt.is_parametric(study):
            sys.exit(f"'{name}': dimensional variables apply to STL-base studies (doe.py <study> --base-stl ...)")
        return (0.0, *dt.dim_range(d))
    if not dt.is_parametric(study):
        sys.exit(f"'{name}': study '{study}' has an STL base, only drag modes (mode:<name>) and "
                 f"dimensional variables (dim:<name>) are variables")
    if name not in dt.morph_params():
        sys.exit(f"unknown variable '{name}' (see --list)")
    return dt.morph_params()[name][:3]


def oat(names, levels, study):
    out = []
    for n in names:
        b, lo, hi = bounds(n, study)
        vals = np.linspace(lo, hi, levels) if levels > 2 else [lo, hi]
        out += [{n: float(v)} for v in vals if abs(v - b) > 1e-9]
    return out


def lhs(names, n, seed, study):
    rng = np.random.default_rng(seed)
    cols = []
    for name in names:
        _, lo, hi = bounds(name, study)
        u = (rng.permutation(n) + rng.random(n)) / n
        cols.append(lo + u * (hi - lo))
    return [{name: round(float(c[i]), 3) for name, c in zip(names, cols)} for i in range(n)]


def kv(s):
    k, v = s.split("=", 1)
    return k, v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", nargs="?")
    ap.add_argument("--list", action="store_true", help="list design variables")
    ap.add_argument("--oat", nargs="+", metavar="VAR", help="one-at-a-time at the bounds")
    ap.add_argument("--levels", type=int, default=2, help="levels per variable for --oat (default 2 = bounds)")
    ap.add_argument("--sweep", nargs="+", metavar="VAR=v1,v2,...")
    ap.add_argument("--lhs", type=int, metavar="N")
    ap.add_argument("--params", nargs="+", help="variables for --lhs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--add", nargs="+", metavar="VAR=VALUE", help="add one combined design")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--base-stl", metavar="STL", help="use this STL as the study base (new study only)")
    ap.add_argument("--wheels", nargs=6, type=float, metavar=("XF", "XR", "TF", "TR", "D", "W"),
                    help="wheel data for --base-stl in m (default: gen3d report.json next to the STL)")
    ap.add_argument("--z-offset", type=float, default=-0.005, help="vertical shift of the base STL (tyre sink)")
    a = ap.parse_args()

    if a.base_stl:
        if not a.study:
            sys.exit("--base-stl needs a study name")
        if dt.read_designs(a.study) and dt.base_id(a.study) != dt.rel(a.base_stl):
            sys.exit(f"study '{a.study}' already has designs on base {dt.base_id(a.study)}; use a new study name")
        base = (dt.stl_base(a.base_stl, *a.wheels, z_offset=a.z_offset) if a.wheels
                else dt.stl_base_from_gen3d(a.base_stl, z_offset=a.z_offset))
        conf = dt.study_config(a.study); conf["base"] = base
        dt.write_study_config(a.study, conf)
        print(f"study '{a.study}': base = {base['path']}  (wheels: front x={base['wheels']['front']['x']:.3f}, "
              f"rear x={base['wheels']['rear']['x']:.3f}, r={base['wheels']['front']['r']:.4f} m)")

    if a.list or not a.study or not (a.oat or a.sweep or a.lhs or a.add or a.base_stl):
        study = a.study
        print(f"base: {dt.base_id(study) if study else dt.IONIQ5_ID + ' (default)'}")
        print(f"{'variable':26s} {'base':>6s} {'lower':>7s} {'upper':>7s} unit  description")
        if not study or dt.is_parametric(study):
            for k, (b, lo, hi, u, d) in dt.morph_params().items():
                print(f"{k:26s} {b:6g} {lo:7g} {hi:7g} {u:4s}  {d}")
        for d in dt.available_dims(study):
            b, lo, hi, u, desc = dt.dim_morph.PARAMS[d]
            print(f"{'dim:' + d:26s} {b:6g} {lo:7g} {hi:7g} {u:4s}  {desc}")
        for m in dt.available_modes(study):
            spec = dt.mode_spec(m)
            lo, hi = dt.mode_range(m)
            src = spec.get("source_geometry", dt.IONIQ5_ID)
            print(f"{'mode:' + m:26s} {0:6g} {lo:7g} {hi:7g} amp   {spec.get('description', 'drag mode')}  [{src}]")
        if study and not dt.available_modes(study):
            print("  (no drag modes for this base yet: create one with "
                  f"python3 2_shape_morphing/morph_gui.py {dt.base_id(study)})")
        if not (a.oat or a.sweep or a.lhs or a.add or a.base_stl):
            return

    new = []
    if a.oat:
        new += oat(a.oat, a.levels, a.study)
    if a.sweep:
        for s in a.sweep:
            k, vals = kv(s)
            bounds(k, a.study)
            new += [{k: float(v)} for v in vals.split(",")]
    if a.lhs:
        if not a.params:
            sys.exit("--lhs needs --params")
        new += lhs(a.params, a.lhs, a.seed, a.study)
    if a.add:
        new.append({k: float(v) for k, v in map(kv, a.add)})

    rows = dt.read_designs(a.study)
    if not rows and not a.no_baseline:
        rows.append({"case": "c000", "params": dt.full({}, a.study)})
    keyf = lambda p: dt.design_key(p, a.study)
    seen = {keyf(r["params"]) for r in rows}
    nxt = 1 + max([int(r["case"][1:]) for r in rows if r["case"][1:].isdigit()] or [0])
    added = 0
    for p in new:
        p = dt.full(p, a.study)
        try:
            errs = dt.validate(p, a.study)
        except (KeyError, ValueError) as e:
            print(f"  rejected {dt.label(p, a.study)}: {e}")
            continue
        if errs:
            print(f"  skip {dt.label(p, a.study)}: {errs}")
            continue
        if keyf(p) in seen:
            continue
        seen.add(keyf(p))
        rows.append({"case": f"c{nxt:03d}", "params": p})
        nxt += 1
        added += 1
    dt.write_designs(a.study, rows)
    print(f"study '{a.study}': {len(rows)} designs ({added} new)")
    for r in rows:
        print(f"  {r['case']}  {dt.label(r['params'])}")


if __name__ == "__main__":
    main()
