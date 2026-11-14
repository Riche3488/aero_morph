# 3 · CFD (OpenFOAM 12 virtual wind tunnel)

Setup: steady RANS k-ω SST at 100 km/h, moving belt, rotating wheels, half model with a symmetry plane,
about 1.6 M cells (snappyHexMesh, 2 layers, wall functions), 8 MPI processes.

## Run a study
```bash
python3 3_cfd/run_study.py rear --clean          # all designs of 2_shape_morphing/studies/rear
python3 3_cfd/run_case.py  rear c003 [--stop-after mesh] [--redo solve]
python3 3_cfd/compare.py   rear                  # also run automatically at the end of run_study
```
Works for both kinds of study base. IONIQ 5 studies get separate wheel and mirror STLs. STL-based studies such as
IONIQ 6 get one multi-region STL whose wheel regions become the `wheelFront`/`wheelRear` patches. Wheel centres,
rotation speed, wheelbase (lRef), moment centre and Aref are filled per case from the stage-2 `meta.json`.

Initial field: an explicit `--map-from` wins. Otherwise the study's own converged baseline (c000) is used. If there is
none, IONIQ 5 studies use `baseline_case/`, and a new vehicle is cold-started with potentialFoam (1500 it).
A new vehicle's c000 therefore runs first (cold start) and the other designs then start from it. The reference case
keeps its `processor*` folders even with `--clean`.

Per design: **prepare** (copies the stage-2 STLs, sets Aref = frontal area / 2) → **mesh** → **init**
(`mapFieldsPar` from the source chosen above, falls back to potentialFoam) → **solve** (50 robust + 750 second-order
iterations) → **post**. Each stage resumes where it stopped (`.done_*` markers). A failed design is
logged in `status.json` and the batch continues.

| option | default | meaning |
|---|---|---|
| `--init map\|potential` | map | map = use a converged source if one exists (800 it); potential = always cold start (1500 it) |
| `--iters`, `--n-avg` | 800 / 300 | iterations / averaging window for Cd |
| `--map-from DIR` | auto | any decomposed converged case (auto: study c000 → baseline_case → potential) |
| `--clean` | off | delete `processor*` after post (~600 MB per case) |
| `--cases`, `--retry-failed` | | run_study subset / re-run failed cases |

## Layout
| path | content |
|---|---|
| `baseline_case/` | original full baseline run (1500 it, Cd 0.315) and the source for mapped init |
| `template/` | case template (dicts with `@GEOMETRY@`/`@AREF@` placeholders, `post.py`) |
| `runs/<study>/cases/<case>/` | OpenFOAM case, `result.json`, `results/*.png` |
| `runs/<study>/results.csv` | **training table for stage 4** (design variables + Cd, CdA, Cl, …) |
| `runs/<study>/report/` | ranking, sensitivity, best-profile plots |

Accuracy: iteration noise is about ±6 Cd counts. The mapped-init baseline reproduces the full run
(0.3169 vs 0.3154). Compare **CdA** when the frontal area changes.
