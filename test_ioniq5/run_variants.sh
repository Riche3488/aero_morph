#!/usr/bin/env bash
# Solve the 10 morphed IONIQ 5 variants (c001-c010), visualise each one and refresh test_ioniq5/.
# The meshes are already built (run_study ... --stop-after mesh); run_study resumes at init, which maps
# the converged c000 solution (800 iterations per case, about 20 min each on 8 cores).
#   bash test_ioniq5/run_variants.sh            # all 10
#   bash test_ioniq5/run_variants.sh c003 c004  # a subset
set -euo pipefail
cd "$(dirname "$0")/.."
set +eu; source /opt/openfoam12/etc/bashrc; set -eu   # the OpenFOAM env script is not -e/-u clean
CASES=("${@:-c001 c002 c003 c004 c005 c006 c007 c008 c009 c010}")
python3 3_cfd/run_study.py ioniq5 --cases ${CASES[@]} 2>&1 | tee -a 3_cfd/runs/ioniq5/run_study_solve.log
for c in ${CASES[@]}; do
  [ -f "3_cfd/runs/ioniq5/cases/$c/result.json" ] && python3 3_cfd/visualize_case.py ioniq5 "$c" || true
done
python3 test_ioniq5/collect_results.py
