# car3d: photos to a symmetric, watertight 3D car model

This pipeline turns photos of a car into a rough 3D model that is watertight and exactly mirror-symmetric.
It is built on **TRELLIS** image-to-3D (MIT licence).
The same steps that produced `output/ioniq5_rough` now run for any vehicle: you write one YAML file per car.

> **Why not Hunyuan3D 2.0?**
> The original request was Hunyuan3D 2.0 (`Comfy-Org/Hunyuan-3D-2.0`).
> Its licence does **not apply in South Korea, the EU or the UK**, so this pipeline uses Microsoft TRELLIS-image-large instead.

## Quick start
```bash
cd gen3d
# re-run IONIQ 5 (reproduces output/ioniq5_rough), or the sedan test case
.venv/bin/python pipeline.py run vehicles/ioniq5.yaml
.venv/bin/python pipeline.py run vehicles/ioniq6.yaml
```

## New vehicle in 4 steps
1. **Find photos** on Wikimedia Commons. The search lists each photo's licence and saves a contact sheet:
   ```bash
   .venv/bin/python pipeline.py search "Kia EV6" --exclude "interior,concept,gt"
   # -> runs/_search/kia_ev6/sheet.png + candidates.json
   ```
   Pick 1–4 photos of the **same car**: same colour, trim and wheels.
   A front-left 3/4 view plus a rear 3/4 view works well.
   Photos of different cars mixed together make TRELLIS produce broken wheels.
2. **Write `vehicles/<id>.yaml`**, starting from a copy of `vehicles/TEMPLATE.yaml`.
   Fill in the published dimensions in mm (L, W without mirrors, H, wheelbase, front overhang, tracks, tyre, ground clearance) and the photo titles.
   You can also give your own photos with `path:`.
3. **Run it:**
   ```bash
   .venv/bin/python pipeline.py run vehicles/<id>.yaml
   ```
4. **Review the outputs:**
   - `runs/<id>/candidates_sheet.png` shows the candidate meshes.
   - `runs/<id>/<id>_rough_views.png` and `_inspect.png` show the final model.
   - If the automatic pick is not the most faithful candidate, set `generation.select: <tag>` and re-run only the later stages:
     ```bash
     .venv/bin/python pipeline.py run vehicles/<id>.yaml --stages select,repair,render
     ```

## Stages (`--stages`, default: all)
| Stage | What it does | Output in `runs/<id>/` |
|---|---|---|
| `fetch` | Downloads the photos with their licence data, then removes the background with BiRefNet on the GPU, keeping only the largest object. Runs in a separate process. | `inputs/`, `inputs_sheet.png`, `ATTRIBUTION.md` |
| `generate` | Runs TRELLIS once for every image set × seed. Each raw mesh is aligned to the car frame, cleaned of slivers, fitted to L/W/H and scored. Candidates that already exist are skipped unless you pass `--force`. | `candidates/<tag>/{raw.ply, aligned.glb, views.png, info.json}` |
| `select` | Picks the best candidate automatically, or the one named in `generation.select`. | `<id>_final.glb`, `candidates_sheet.png` |
| `repair` | Finds the symmetry plane, builds a voxel solid, adds a flat floor, averages the SDF with its mirror, extracts the surface, decimates one half and mirrors it, moves the axles to their spec positions and fits L/W/H. | `<id>_<preset>.{stl,glb,obj}` |
| `render` | Draws preview and inspection images with Matplotlib (no OpenGL needed). | `*_views.png`, `*_inspect.png` |

All the numbers are collected in `runs/<id>/report.json`: candidate scores, symmetry, closing radius, axle warp, final L/W/H, watertightness, volume and per-stage timing.

**Car frame (metres):** x points rearward from the front bumper, y points left, z points up, and z = 0 is the bottom of the tyres.

### Repair presets (`repair.presets`)
| Preset | Voxel size | SDF σ | Opening | Taubin | Faces |
|---|---|---|---|---|---|
| `rough` (default) | 12 mm | 1.5 vox | 2 vox | 30 | ~60k |
| `fine` | 6 mm | 1.0 vox | 1 vox | 15 | ~400k |

You can override these per vehicle, for example `repair: {presets: [rough], rough: {voxel: 0.015, faces: 40000}}`, or from the CLI with `--preset rough,fine`.

## Things to know
- **Automatic selection only checks the geometry.** It checks that all four wheels are present and that left and right agree; it does not check how faithful the model is to the photos.
  For IONIQ 6 the top score went to a single-photo candidate, while the two-photo candidate captures the duck-tail spoiler. `vehicles/ioniq6.yaml` therefore sets `select:` by hand.
  Always look at `candidates_sheet.png`.
- **TRELLIS is not bit-reproducible.** The sparse GPU kernels are non-deterministic, so the same seed gives a different mesh on each run.
  Generate several seeds and let the selection choose.
  To keep a specific earlier mesh, list it under `generation.import: {tag: path/raw.ply}`; `vehicles/ioniq5.yaml` does this with the original IONIQ 5 mesh.
- **Front/rear detection.** By default (`orientation: auto`) the lower end of the car is taken as the front, which suits hatchbacks, SUVs and sedans.
  If the preview shows the car backwards, set `orientation: flip`.
- **Axle warp.** The tyre contact points are moved smoothly to the spec axle positions, keeping the wheel zones rigid, so the overhangs and wheelbase get stretched or compressed.
  `report.json → repair.<preset>.axle_warp` records how much; for IONIQ 6 the front overhang was compressed to ×0.82.
  Large values mean the generated proportions were off.
- **Flat floor.** Between the wheels the underbody is replaced by a flat floor at the ground-clearance height. This is DrivAer-style and matches `../ioniq5_geometry.py`.
  Pass `flat_floor: false` in a preset override to keep the generated underbody.
- **Ready for CFD?** The output is watertight with consistent outward normals, so it can be fed to snappyHexMesh, but it is approximate.
  It is AI-generated from photos, the glass, wheels and underbody are simplified, and the tyres touch the ground plane.
  Use it for trend and comparative studies, not for absolute Cd.

## Validated runs (RTX 5070 Ti 16 GB, WSL2)
| Vehicle | Candidates | Selected | Rough result L/W/H (mm) | Watertight | Mirror deviation |
|---|---|---|---|---|---|
| IONIQ 5 (`vehicles/ioniq5.yaml`) | 3 new seeds + 1 imported | `orig_fl+rr_s1` (auto) | 4635 / 1911\* / 1605, 8.73 m³ | yes | 0.000 mm |
| IONIQ 6 (`vehicles/ioniq6.yaml`) | 2 image sets × 3 seeds | `fl+rl_s2` (manual) | 4855 / 1880 / 1495, 8.24 m³ | yes | 0.000 mm |

\* The overall width includes the mirrors; the body width is fitted to the spec value.

Each TRELLIS run takes about 8 s and about 10.4 GiB of VRAM; loading the model takes about 25 s.
The `rough` repair takes about 60 s and under 4 GB of RAM.

## Setup (fresh clone)
```bash
cd gen3d
python3 -m venv .venv            # or: python3 -m venv --without-pip .venv + bootstrap pip from its wheel
.venv/bin/pip install torch==2.7.1 torchvision==0.22.1 xformers==0.0.31.post1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
git clone https://github.com/microsoft/TRELLIS.git && git -C TRELLIS checkout 193e0502b5cb3eb5997ec457d5340e95d09b158e
git -C TRELLIS submodule update --init --depth 1 trellis/representations/mesh/flexicubes
```
The model weights download on the first run:
- TRELLIS-image-large from Hugging Face
- DINOv2 through torch.hub
- BiRefNet into `~/.u2net`

Not in git (see `../.gitignore`):
- `.venv/`, `TRELLIS/`, `inputs/`
- `runs/*/candidates/` and `runs/*/inputs/` (they can be regenerated)
- most of `output/`; only `output/ioniq5_final_raw.ply` is kept, because `vehicles/ioniq5.yaml` imports it

## Environment notes (RTX 50xx / sm_120)
- The `.venv` was created with stdlib `venv --without-pip`, and pip was bootstrapped from its wheel.
- Main package versions: torch 2.7.1+cu128, xformers 0.0.31.post1, spconv-cu126, libigl, open3d, scikit-image, pymeshfix.
- **xformers:** on sm_120 the automatic dispatcher picks the Hopper-only FA3 kernel, which crashes. `car3d/generate.py` forces FA2, with Cutlass as the fallback.
- **kaolin:** there is no wheel for torch 2.7. `shims/kaolin` provides `check_tensor`, the only kaolin function FlexiCubes uses.
- **No nvdiffrast** (there is no nvcc). Vertex colours are taken from TRELLIS' decoded 3D Gaussians instead of a texture bake.
- **rembg / BiRefNet:** it is forced onto the GPU with `onnxruntime.preload_dlls()`, which uses torch's CUDA 12 wheels. On the CPU it needs about 12 GB of RAM and was OOM-killed.
- **Marching cubes** runs at iso-level `1e-3·h`, not 0. Averaging the SDF with its mirror gives exact zeros on grid nodes, and with an iso-level of 0 that produced non-manifold edges.

## Layout
```
pipeline.py            CLI: search | run
car3d/spec.py          YAML config -> Spec (mm -> m), presets
car3d/images.py        Wikimedia search/download, attribution, background removal, contact sheets
car3d/generate.py      TRELLIS loading + sampling (sm_120 workarounds, Gaussian vertex colours)
car3d/align.py         car frame, sliver removal, L/W/H fit
car3d/score.py         wheel presence + left/right consistency
car3d/repair.py        symmetric watertight solid, axle warp, spec fit
car3d/render.py        preview / inspection images
vehicles/*.yaml        one config per vehicle (TEMPLATE.yaml to start)
runs/<id>/             pipeline outputs
output/                first IONIQ 5 session (incl. ioniq5_rough.*, ioniq5_final_raw.ply)
legacy/                the original one-off scripts (run_trellis.py, symmetrize_repair.py, ...)
```

## Licences / attribution
- TRELLIS code and weights: MIT (Microsoft). FlexiCubes: Apache-2.0. DINOv2: Apache-2.0. BiRefNet: MIT.
- Photo credits for each run are in `runs/<id>/ATTRIBUTION.md`. Models derived from CC BY or CC BY-SA photos must carry the same attribution.
  - IONIQ 5: "Hyundai Ioniq 5 1X7A0847.jpg" by Alexander-93 (CC BY-SA 4.0); "Hyundai Ioniq 5 rear.jpg" by TTTNIS (CC0).
  - IONIQ 6: "Hyundai Ioniq 6 1X7A7258.jpg" and "1X7A7260.jpg" by Alexander-93 (CC BY-SA 4.0).
- Vehicle designs and names belong to their manufacturers. The models are for research use.
