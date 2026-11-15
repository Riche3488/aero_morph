# 4 · Deep-learning training on CFD results (planned, not implemented yet)

## Inputs (already produced by stages 2–3)
| source | content | use |
|---|---|---|
| `3_cfd/runs/<study>/results.csv` | design variables as run (`tail_taper`, …, `mode:<name>`; a missing mode column = 0) → Cd, CdA, Cl, Cl_front/rear | tabular surrogate |
| `3_cfd/runs/<study>/cases/<case>/case.json` + `constant/geometry/*.stl` | variables with baseline/bounds/unit; the surface that was simulated | geometry-based models |
| `3_cfd/runs/<study>/cases/<case>/postProcessing/car/<t>/patch.vtk` | surface p (and wall shear stress) | field prediction |
| `…/postProcessing/{yNormal,zNormal,xWake}/<t>/cutPlane.vtk` | p, U on cut planes | wake / flow-field models |

## Planned model types
1. **Design variables → Cd/Cl**: Gaussian process or MLP. Works with tens of samples and gives
   uncertainty for active learning, which tells stage 2 which designs to run next.
2. **Geometry → Cd**: point cloud / mesh networks (PointNet, GNN) on `body.stl`. Independent of the
   parameterisation, so it also covers drag-mode and arbitrary shapes.
3. **Geometry → surface Cp field**: GNN or neural operator on the surface mesh, using the `patch.vtk` labels.

## Suggested layout
```
4_dl_training/
  datasets/   build_dataset.py -> npz/parquet from all runs/*/results.csv + geometry
  models/     model definitions
  train.py    -> checkpoints/<name>/ (weights, normalisation, metrics.json)
```
