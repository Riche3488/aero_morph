#!/bin/bash
# Two-stage steady RANS: stage 1 first-order upwind (robust start), stage 2 second order.
cd "${0%/*}" || exit 1
source /opt/openfoam12/etc/bashrc
NP=8
set_kw() { foamDictionary -entry "$1" -set "$2" "$3" > /dev/null; }

# ---- stage 1: 0 -> 300, upwind, low relaxation
set_kw endTime 300 system/controlDict
mpirun -np $NP foamRun -parallel > log.foamRun.stage1 2>&1 || exit 1

# ---- stage 2: 300 -> 1500, linearUpwind, standard relaxation
set_kw "divSchemes/div(phi,U)" "bounded Gauss linearUpwind limited" system/fvSchemes
set_kw "divSchemes/turbulence" "bounded Gauss limitedLinear 1" system/fvSchemes
set_kw "relaxationFactors/fields/p" 0.3 system/fvSolution
set_kw "relaxationFactors/equations/U" 0.7 system/fvSolution
set_kw endTime 1500 system/controlDict
mpirun -np $NP foamRun -parallel > log.foamRun.stage2 2>&1 || exit 1
echo DONE
