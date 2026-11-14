# Vehicle external-aero pipeline (IONIQ 5)

```
1_vehicle_geometry ──► 2_shape_morphing ──► 3_cfd ──► 4_dl_training ──► 5_inference
 base shape (STL)       designs + morphed      OpenFOAM     surrogate model    fast prediction /
                        geometry (STL+meta)    Cd, CdA, Cl  (planned)          optimisation (planned)
```

| stage | folder | status | entry points |
|---|---|---|---|
| 1 Vehicle geometry | `1_vehicle_geometry/` | ✅ | `ioniq5_geometry.py` (parametric IONIQ 5), `gen3d_image_to_3d/` → `../gen3d` (image-to-3D) |
| 2 Shape morphing | `2_shape_morphing/` | ✅ | `doe.py` (base: IONIQ 5 generator or any STL), `export_designs.py`, `preview_designs.py`, **`morph_gui.py`** / **`run-local.bat`** (drag morphing GUI) |
| 3 CFD | `3_cfd/` | ✅ | `run_study.py`, `run_case.py`, `compare.py`, `baseline_case/` |
| 4 DL training | `4_dl_training/` | 📝 plan | README (data contract) |
| 5 Inference | `5_inference/` | 📝 plan | README |

## Quick start
```bash
# 2 · define designs (parametric variables and/or drag modes), export and check geometry
python3 2_shape_morphing/doe.py rear --oat tail_taper diffuser_angle
python3 2_shape_morphing/preview_designs.py rear

# 2 · GUI: drag-morph any STL in the browser, save the result as a reusable mode
#     (Windows: double-click run-local.bat; it shows a model menu, or drop an STL on it)
python3 2_shape_morphing/morph_gui.py gen3d/runs/ioniq6/ioniq6_rough.stl

# 2 · a study on another vehicle (any watertight STL as the base, drag modes as variables)
python3 2_shape_morphing/doe.py ioniq6 --base-stl gen3d/runs/ioniq6/ioniq6_rough.stl
python3 2_shape_morphing/doe.py ioniq6 --oat mode:ioniq6_trunk_drop mode:ioniq6_rear_taper

# 3 · CFD for all designs, then the report (~20 min per design on 8 cores)
python3 3_cfd/run_study.py rear --clean
```

## Data hand-offs
| from → to | location | content |
|---|---|---|
| 1 → 2 | `ioniq5_geometry.MORPH_PARAMS` / any STL | base shape and its parametric variables |
| 2 → 3 | `2_shape_morphing/studies/<study>/geometry/<case>/` | `*.stl` (m, watertight) + `meta.json` (variables, frontal area, QA) |
| 3 → 4 | `3_cfd/runs/<study>/results.csv`, `cases/<case>/postProcessing/` | design variables → Cd, CdA, Cl; surface/plane fields |
| 4 → 5 | `4_dl_training/checkpoints/<name>/` | trained model (planned) |

Shared paths: `pipeline_config.py`. Coordinates everywhere: x rearward from the front bumper, y left,
z up from the ground. CFD and morph files use metres.

## Notes
- `gen3d/` (image-to-3D, TRELLIS, 11 GB with its own `.venv`) is kept at the top level. Its virtual
  environment has absolute paths, so moving it would break it. Stage 1 links to it.
- Studies: `demo` (IONIQ 5, parametric + drag modes) and `ioniq6` (IONIQ 6 rough STL base, drag modes).
  ioniq6/c000 has been meshed (1.54 M cells, wheel patches OK) but not solved yet.
- The CFD runs of the `demo` study were paused at the user's request (c000 and c001 are done). Restart with
  `python3 3_cfd/run_study.py demo --clean`; finished stages are skipped.
