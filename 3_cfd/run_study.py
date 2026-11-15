#!/usr/bin/env python3
"""
Run every exported design of a study (sequentially, each case on --np cores), then
build the comparison report. Finished cases are skipped, failed ones are reported and
the batch continues. The case list is the stage-2 hand-off
2_shape_morphing/studies/<study>/geometry/index.json (no stage-1/2 code is imported).

  python3 3_cfd/run_study.py <study> [--cases c001 c003] [run_case options...]

Typical workflow
  python3 2_shape_morphing/doe.py             rear --oat roof_rear_drop tail_taper diffuser_angle
  python3 2_shape_morphing/preview_designs.py rear   # export + check the shapes (seconds)
                                                     # (or export_designs.py rear: export only)
  python3 3_cfd/run_study.py                  rear --clean   # ~20 min per design
  python3 3_cfd/compare.py                    rear   # (also run automatically at the end)
"""
import argparse
import json
import os
import time

import compare
import run_case
from run_case import case_dir, geometry_changed, study_cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study")
    ap.add_argument("--cases", nargs="+", help="subset of case ids")
    ap.add_argument("--retry-failed", action="store_true", help="re-run cases whose last attempt failed")
    run_case.add_args(ap)
    a = ap.parse_args()

    rows = study_cases(a.study)
    if not rows:
        raise SystemExit(f"no exported designs for '{a.study}': create them with 2_shape_morphing/doe.py, "
                         f"then python3 2_shape_morphing/export_designs.py {a.study}")
    todo = [r["case"] for r in rows if not a.cases or r["case"] in a.cases]
    t0 = time.time()
    summary = []
    for i, case in enumerate(todo, 1):
        cdir = case_dir(a.study, case)
        st_file = os.path.join(cdir, "status.json")
        st = json.load(open(st_file)) if os.path.exists(st_file) else {}
        changed = geometry_changed(a.study, case)
        if os.path.exists(os.path.join(cdir, "result.json")) and st.get("state") == "done":
            if not changed and not a.redo:              # --redo STAGE re-runs done cases from STAGE
                summary.append((case, "done (cached)"))
                continue
            if changed and not (a.rerun_changed or a.redo == "prepare"):
                summary.append((case, "done, but the exported geometry changed since "
                                      "(use --rerun-changed or --redo prepare)"))
                continue
        if st.get("state") == "failed" and not a.retry_failed:
            summary.append((case, f"skipped (failed before in {st.get('stage')}; use --retry-failed)"))
            continue
        print(f"\n=== [{i}/{len(todo)}]  elapsed {(time.time() - t0) / 60:.1f} min")
        opt = run_case.finalize_opts(argparse.Namespace(**vars(a)))
        ok = run_case.run(a.study, case, opt)
        summary.append((case, "ok" if ok else "FAILED"))

    print("\n=== batch summary")
    for c, s in summary:
        print(f"  {c}: {s}")
    print(f"total {(time.time() - t0) / 60:.1f} min")
    compare.main(a.study)


if __name__ == "__main__":
    main()
