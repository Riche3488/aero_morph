# 5 · Inference with the trained model (planned, not implemented yet)

Goal: predict aerodynamics in milliseconds instead of running a ~20-minute CFD case.

## Planned uses
- `predict.py --set tail_taper=40 mode:rear_roof_drop=1` returns Cd, CdA and Cl with uncertainty
  (uses the same design variables as stage 2).
- `predict.py --stl new_shape.stl` runs a geometry-based model on any shape (output of stage 1, or a
  drag-morphed STL from stage 2).
- **Optimisation loop**: search the design space on the surrogate (e.g. minimise CdA subject to
  constraints), then send the top designs back to `2_shape_morphing/doe.py --add …`. Stage 3 verifies
  them with CFD, and the new results go back into stage 4 training (active learning).

## Expected inputs
`4_dl_training/checkpoints/<name>/` (weights, normalisation, list of design variables / model type).
