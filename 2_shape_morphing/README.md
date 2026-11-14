# 2 · Shape morphing

Two ways to change the shape. Both can be combined in one design.

| | Parametric morph | Drag (RBF) morph |
|---|---|---|
| Works on | the IONIQ 5 generator (stage 1) | **any** triangle mesh / STL |
| Defined by | 9 named variables (`MORPH_PARAMS`) | a *mode* file `modes/<name>.json` (handles + anchors) |
| Created with | `doe.py` values | `drag_editor.html` (mouse drag) or JSON by hand |
| In a design table | column `tail_taper`, ... | column `mode:<name>` = amplitude (0 = off) |

## Base geometry of a study
| base | design variables | set up with |
|---|---|---|
| IONIQ 5 generator (default) | parametric variables + drag modes drawn on the IONIQ 5 | nothing (default) |
| **any STL**, e.g. IONIQ 6 | drag modes drawn on that STL | `doe.py <study> --base-stl <file.stl>` |

```bash
python3 2_shape_morphing/doe.py ioniq6 --base-stl gen3d/runs/ioniq6/ioniq6_rough.stl   # study.json + c000
python3 2_shape_morphing/morph_gui.py gen3d/runs/ioniq6/ioniq6_rough.stl              # draw modes, Save to modes/
python3 2_shape_morphing/doe.py ioniq6 --list                                          # only modes drawn on this STL
python3 2_shape_morphing/doe.py ioniq6 --oat mode:ioniq6_trunk_drop mode:ioniq6_rear_taper
```
- **Wheels:** for a gen3d model, the wheels come from `report.json` next to the STL. For other files give them
  explicitly: `--wheels x_front x_rear track_front track_rear tire_diameter tire_width` (m).
- **Tyre sink:** `--z-offset` (default −5 mm) lowers the car so the tyres sink into the ground slightly, as in the IONIQ 5 model.
- **Export:** drag modes are applied in the STL's own coordinates, the same ones the GUI shows. The wheel zones are
  frozen with a smooth falloff. The fused wheels are then split into the regions `wheelFront` / `wheelRear`
  (`body.stl` becomes a multi-solid STL, still one closed surface), so stage 3 can give them rotating-wheel conditions.
- **Mode matching:** every mode records `source_geometry`. A mode can only be used in a study with the same base.
  Mixing them is rejected, as are IONIQ 5 parametric variables in an STL-based study.

## Workflow
```bash
python3 2_shape_morphing/doe.py --list                                  # all design variables
python3 2_shape_morphing/doe.py rear --oat tail_taper diffuser_angle    # designs.csv (c000 = baseline)
python3 2_shape_morphing/doe.py rear --sweep mode:rear_roof_drop=0.5,1,1.5
python3 2_shape_morphing/doe.py rear --add tail_taper=40 mode:front_bumper_push=1
python3 2_shape_morphing/export_designs.py rear     # -> studies/rear/geometry/<case>/*.stl + meta.json
python3 2_shape_morphing/preview_designs.py rear    # -> studies/rear/previews/*.png (exports if needed)
```
DOE modes: `--oat` (each variable at its bounds, `--levels N`), `--sweep VAR=v1,v2`, `--lhs N --params ...`
(Latin hypercube), `--add` (one combined design). Each call appends to the table and skips duplicates.

**Output for stage 3**: `studies/<study>/geometry/<case>/`
- `body.stl`, `wheel_{FL,FR,RL,RR}.stl`, `mirror_{L,R}.stl` in metres, all watertight
- `meta.json` with the design variables, frontal area, bbox and the drag-morph check results
  (det J, flipped triangles, clearance to the wheels)

## Drag morphing (RBF) for any geometry
The displacement field is `u(x) = Σ w_j φ(|S_j⁻¹(x − c_j)|)` with the Wendland C2 kernel
`φ(t) = (1−t)⁴(4t+1)`:
- **Compact support.** Outside the radius nothing moves.
- **Smooth.** The field is C2 continuous.
- **Exact at the control points.** Handles reach exactly the dragged displacement and anchors stay exactly fixed.
- **Topology unchanged.** Only vertex positions move, so a watertight mesh stays watertight.
- **Defined in space.** The same field can later deform a CFD volume mesh.

1. **GUI (recommended)**: on Windows, double-click **`run-local.bat`** in the project root. It shows a model menu,
   accepts an STL dragged onto it, or takes a path argument. From a shell:
   `python3 2_shape_morphing/morph_gui.py <any.stl>`, for example
   `gen3d/runs/ioniq6/ioniq6_rough.stl`. This opens `http://localhost:8765` with the model loaded (in the
   Windows browser when run from WSL). The project panel adds:
   - **Save to modes/**: the mode becomes the design variable `mode:<name>`. The JSON records `source_geometry`.
   - **Load**: continue editing a saved mode.
   - **Apply in Python**: full-resolution morph with `rbf.py`, plus checks and a render, written to
     `gui_output/<stem>/<stem>__<name>.stl|png|_qa.json`.

   `drag_editor.html` can also be opened directly as a file (load the STL by hand, download JSON/STL).
   In the editor:
   - press on the surface and drag to create and move a handle. Shift drags along the surface normal; there are also axis locks.
   - *Anchor* pins points that must not move.
   - sliders set the radius and stretch the falloff per axis (x, y, z); there are y-symmetry and ground-lock options.
   - the panel shows a live colour map and the min det J fold check.
   - **Save mode JSON** into `modes/`. The mode can then be used as a design variable.
2. **Apply to any STL (CLI)**:
   `python3 2_shape_morphing/drag_morph.py --in car.stl --mode modes/x.json=1.0 [--freeze wheels.stl] --out out --render`
   This writes the morphed STL, `qa.json`, and `morph.png` (colour map plus section cuts).
3. **Mode JSON format** (units = STL units; the IONIQ pipeline uses metres):
   ```json
   {"symmetry": "y", "planes": [{"point": [0,0,0], "normal": [0,0,1], "blend": 0.05}],
    "handles": [{"name": "lip", "center": [4.38,0,1.565], "disp": [0,0,-0.05], "radius": [0.9,1.4,0.6]}],
    "anchors": [{"center": [2.9,0,1.605], "radius": [0.9,1.4,0.6]}],
    "amplitude_range": [-2, 2]}
   ```
   `normal_disp` can replace `disp` to push or pull along the local surface normal.
   Amplitudes scale every displacement linearly, and several modes add together.

**Quality rule of thumb:** keep |d| / radius below about 0.5. `min det J > 0.2` is enforced on export;
below 0 the space folds and the design is rejected. The JS editor and `rbf.py` give the same
result (checked to 1e-12).

## Files
| file | role |
|---|---|
| `design_table.py` | read/write `designs.csv`, labels, validation (parametric + modes) |
| `doe.py` | design generation |
| `export_designs.py` | morph → STL + meta.json (cached by signature) |
| `preview_designs.py` | centre-line, plan and section cuts vs baseline |
| `rbf.py` | RBF morph engine, plane cuts, quality checks |
| `drag_morph.py` | CLI: apply modes to arbitrary STL(s) |
| `drag_editor.html` | interactive drag editor (three.js) |
| `morph_gui.py` | local server: opens the editor on any STL, saves modes, applies the morph at full resolution |
| `modes/` | saved drag modes (`rear_roof_drop`, `front_bumper_push` examples) |
