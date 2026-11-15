# 2 · Shape morphing

Three ways to change the shape. They can be combined in one design.

| | Parametric morph | Dimensional morph | Drag (RBF) morph |
|---|---|---|---|
| Works on | the IONIQ 5 generator (stage 1) | **any** whole-car STL (fused wheels) | **any** triangle mesh / STL |
| Defined by | 9 named variables (`MORPH_PARAMS`) | 5 fixed variables (`dim_morph.py`) | a *mode* file `modes/<name>.json` (handles + anchors) |
| Created with | `doe.py` values | `doe.py` values | `drag_editor.html` (mouse drag) or JSON by hand |
| In a design table | column `tail_taper`, ... | column `dim:foh`, ... = change in mm (0 = baseline) | column `mode:<name>` = amplitude (0 = off) |

## Base geometry of a study
| base | design variables | set up with |
|---|---|---|
| IONIQ 5 generator (default) | parametric variables + drag modes drawn on the IONIQ 5 | nothing (default) |
| **any STL**, e.g. IONIQ 6 | dimensional variables + drag modes drawn on that STL | `doe.py <study> --base-stl <file.stl>` |

## Dimensional morph (STL base): overhangs, length, width, spoiler
| variable | change (mm) | bounds |
|---|---|---|
| `dim:foh` | front overhang; the nose moves forward (+) | −100 … 100 |
| `dim:roh` | rear overhang; the tail moves rearward (+) | −100 … 100 |
| `dim:length` | length at fixed overhangs = wheelbase (the rear axle moves) | −150 … 150 |
| `dim:width` | body width (the track follows the body) | −60 … 60 |
| `dim:spoiler` | height of the roof-spoiler trailing edge | −30 … 50 |

Total length change = `foh + length + roh`. The front axle keeps its coordinates, so for `foh > 0` the nose moves to x < 0.
- **Field:** `u = (D(x), y·Δw/W, Δs·w_x(x)·w_z(z))`. `D` stretches three zones almost uniformly: nose+0.12 m … front tyre−0.06 m,
  between the arches, and rear tyre+0.06 m … tail−0.12 m. The nose and tail ends, the arches and everything outside a zone move rigidly.
  The spoiler kick-up builds up over the last 0.45 m of the roof. The rear glass more than 0.28 m below the trailing edge stays put.
  The field is C1, and `det(I+J) = (1+D')(1+Δw/W)(1+∂u_z/∂z)` exactly (lower-triangular J), so it cannot fold within the bounds.
- **Wheels:** the tyre zones move rigidly with their wheel centre (smooth falloff), so tyres keep their shape and the arch gap is kept.
  `meta.json` gets the moved wheels (x for `length`, y for `width`).
- **Landmarks:** wheel zones, body width and spoiler trailing edge are detected on the base mesh. The trailing edge is
  the largest 2 cm drop of the exact centre-line section (mesh cut at y = 0) in the rear 30 %, excluding the rigid tail
  end. A drop below 20 mm adds a warning. They are written to `meta.json → qa.dims.landmarks`. Override them with
  `study.json → base.dim_landmarks`.
- **Order:** drag modes (base coordinates), then dimensional morph, then the tyre-sink z offset.

**GUI (drag the car):** `python3 2_shape_morphing/dim_gui.py ioniq5` opens `http://localhost:8766`.
- **Handles.** There is one coloured grab handle per variable: at the nose (FOH), tail (ROH), rear wheel (length), side
  panel (width) and spoiler edge (spoiler). Each shows a double arrow along its direction of motion and a label with the
  current value. Handles keep a constant size on screen and have an enlarged grab zone. The width and length handles move
  to the side that faces the camera. Drag a handle along its arrow and it follows the mouse. "show handles" toggles them.
- **Drag the surface.** Press on the car and drag. The picked point follows the mouse through the same field: the editor
  computes the sensitivity ∂u/∂v_k of that point for every variable, and the drag direction on screen picks one.
  Typical results: nose → `foh`, tail/hatch → `roh`, roof/doors/rear wheel along x → `length`, side panel sideways →
  `width`, roof end up/down → `spoiler`. When several variables move the point the same way, the most local one wins.
  <kbd>Shift</kbd> gives 10 mm steps and <kbd>Ctrl+Z</kbd> undoes. Sliders and number fields give the same values.
- **Preview only.** The browser gets a light preview mesh (~200k triangles, vertex clustering, `--preview-faces`)
  for display. The field is the one in `dim_morph.py`, evaluated in JS on the preview vertices, with the landmarks,
  zone tables and wheel mask sent by the server; it matches the Python field to float precision.
- **Full resolution on request:** *Check in Python* morphs the full base on the server and returns measured dimensions
  plus det J / flips, with no files. *Add to designs* appends the values to `designs.csv`. *Export case* runs
  `export_designs` for that case.

**Shape sets (given STLs with known values):** a folder `1.stl … n.stl` plus a manifest (`manifest.txt|csv|json`, mm):
```text
1.stl : -40mm FOH
2.stl : +50mm ROH, +20 mm spoiler        # also: 2.stl roh=50 spoiler=20 · CSV: file,foh,roh,length,width,spoiler
```
```bash
python3 2_shape_morphing/dim_gui.py ioniq5 --set path/to/set            # browse in the GUI ("Shape set" panel)
python3 2_shape_morphing/shape_set.py path/to/set --study ioniq5 --compare             # deviation report -> compare.json
python3 2_shape_morphing/shape_set.py path/to/set --study ioniq5 --to-designs          # values -> designs (morpher geometry)
python3 2_shape_morphing/shape_set.py path/to/set --study ioniq5 --import [STUDY]      # given STLs -> new study
```
- **Compare.** The study base is morphed with the listed values. The given STL is aligned by a trimmed point-to-plane
  translation, which absorbs e.g. the −5 mm tyre sink of exported cases; STLs in mm are scaled to m. Then its signed
  deviation is measured: `match` if p99 ≤ 2 mm, `close` if ≤ 10 mm, otherwise `differs`. A wrong manifest entry shows up
  as `differs`.
- **GUI.** Picking an STL sets the sliders and handles to its values. Views: *Morpher*, *Given*, *Overlay* (translucent
  given STL over the morpher's shape) or *Deviation* (signed colour map). ← → step through the set. Editing from there
  works as usual.
- **Import.** The given STLs become the case geometry of a new study. `study.json → imports` maps case → file plus the
  alignment. `export_designs.py` writes them aligned, with the tyre-sink offset and the wheels split into regions; the
  wheels are placed where the morpher puts them for the case's values. `meta.json` carries the design values and the
  deviation, so stage 3 runs them and stage 4 learns values → geometry. Values outside the morpher range are allowed
  here, and duplicate values or `differs` entries are reported as warnings.

```bash
python3 2_shape_morphing/doe.py ioniq5 --base-stl gen3d/output/ioniq5_fixed.stl --wheels 0.8287 3.8663 1.607 1.587 0.730 0.235
python3 2_shape_morphing/doe.py ioniq5 --oat dim:foh dim:roh dim:length dim:width dim:spoiler
python3 2_shape_morphing/doe.py ioniq5 --lhs 20 --params dim:foh dim:roh dim:length dim:width dim:spoiler --seed 1
python3 2_shape_morphing/export_designs.py ioniq5 && python3 2_shape_morphing/preview_designs.py ioniq5
```

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
python3 2_shape_morphing/export_designs.py rear     # -> studies/rear/geometry/index.json, <case>/*.stl + meta.json
python3 2_shape_morphing/preview_designs.py rear    # -> studies/rear/previews/*.png (exports if needed)
```
DOE modes: `--oat` (each variable at its bounds, `--levels N`), `--sweep VAR=v1,v2`, `--lhs N --params ...`
(Latin hypercube), `--add` (one combined design). Each call appends to the table and skips duplicates.

**Output for stage 3** (the complete hand-off; stage 3 imports no stage-2 code): `studies/<study>/geometry/`
- `index.json`: the designs of `designs.csv` whose export is up to date. Stage 3 runs exactly these.
- `<case>/body.stl`, `wheel_{FL,FR,RL,RR}.stl`, `mirror_{L,R}.stl` in metres, all watertight
  (STL base: one multi-region `body.stl`)
- `<case>/meta.json` with the design variables (values + baseline/bounds/unit), label, frontal area, wheels,
  bbox, reference Cd and the drag-morph check results (det J, flipped triangles, clearance to the wheels)

The export cache key covers the design, the mode files, the base and, for the IONIQ 5 base, the stage-1
generator source, so editing `ioniq5_geometry.py` re-exports. The generator is only loaded for parametric
studies; STL-base studies do not need stage 1.

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
| `rbf.py` | RBF morph engine, quality checks |
| `dim_morph.py` | dimensional morph (overhangs, length, width, spoiler height) for STL bases |
| `dim_gui.py`, `dim_editor.html` | GUI for the dimensional morph: drag the car surface, live preview, add / export designs |
| `shape_set.py` | shape sets: manifest parsing, comparison with the morpher, designs / import hand-off (CLI) |
| `mesh_preview.py` | light display meshes for the GUIs (vertex clustering) |
| `drag_morph.py` | CLI: apply modes to arbitrary STL(s) |
| `drag_editor.html` | interactive drag editor (three.js) |
| `morph_gui.py` | local server: opens the editor on any STL, saves modes, applies the morph at full resolution |
| `modes/` | saved drag modes (`rear_roof_drop`, `front_bumper_push` examples) |
