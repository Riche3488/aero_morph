"""car3d – photo(s) -> TRELLIS image-to-3D -> aligned, symmetric, watertight car model.

Stages (see ../pipeline.py):
  fetch     download photos (Wikimedia Commons or local files) + background removal
  generate  TRELLIS sampling for every (image set x seed) candidate, aligned to the car frame
  select    automatic candidate scoring (all four wheels present, left/right consistency)
  repair    symmetry plane, voxel solid, flat floor, symmetric SDF, watertight symmetric mesh,
            axle positions and L/W/H fitted to the published spec
  render    preview images

Car frame (metres): x rearward from the front bumper, y left (+), z up, ground z = 0.
"""
