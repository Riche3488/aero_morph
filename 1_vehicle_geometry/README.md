# 1 · Vehicle geometry: IONIQ 5 simplified exterior for external aero CFD

Regenerate: `python3 1_vehicle_geometry/ioniq5_geometry.py [--scale 0.001] [--res 1.0] [--set tail_taper=40 ...]`,
previews: `python3 1_vehicle_geometry/preview.py`

The parametric morph variables (`MORPH_PARAMS`, `set_design()`) live in this module, because they deform its
key-point profiles. Stage 2 drives them. `gen3d_image_to_3d/` is a link to `../gen3d`, a separate
image-to-3D (TRELLIS) model of the same car. That folder was left in place, see the top-level README.

## Basis (public spec, mm)
| Item | Value |
|---|---|
| Length / Width (no mirrors) / Height | 4635 / 1890 / 1605 |
| Wheelbase, overhang F/R | 3000, 845 / 790 |
| Track F/R | 1638 / 1647 |
| Tire | 235/55R19 (D = 741) |
| Ground clearance | 160 |
| Width incl. mirrors (model) | 2100 |

## Coordinates (output in **metres** by default)
- x: rearward, x=0 at front bumper tip; front axle x=0.845, rear axle x=3.845
- y: +left, symmetric about y=0
- z: up, z=0 = ground. Tires penetrate ground by 5 mm (contact patch) – trim with the ground plane or use as-is for snappyHexMesh.

## Files (`output/`)
| File | Content |
|---|---|
| `ioniq5_body.stl` | Body, closed, with open wheel wells (~490k tris) |
| `ioniq5_wheel_{FL,FR,RL,RR}.stl` | Solid tires w/ wheel face recess (for rotating-wall BC) |
| `ioniq5_mirror_{L,R}.stl` | Mirror housings (overlap body – union in mesher) |
| `ioniq5_full_multisolid.stl` | All parts as named ASCII solids (OpenFOAM) |
| `ioniq5_views.png`, `ioniq5_underbody.png` | Preview renders |

All parts: watertight, 2-manifold, consistent outward normals, no degenerate faces.
Projected frontal area (incl. wheels + mirrors): **≈ 2.65 m²**.

## Simplifications (fidelity level ≈ DrivAer "smooth underbody")
- Flat smooth underbody, no grille openings / cooling flow, no door gaps, wipers, antenna, roof rails.
- Wheel wells are box-shaped (wall at tire inner face + 25 mm), arch radius 415 mm.
- Surfaces are a parametric loft fitted to the published dimensions and the general silhouette,
  **not** OEM CAD/scan data – use for trend/comparative studies, not absolute Cd validation.
  Profiles (`Z_TOP`, `Z_BOT`, `HALF_W`, `Z_BELT`) are editable key-point tables at the top of the script.
