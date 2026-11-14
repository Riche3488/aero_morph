"""Vehicle config (YAML) loading. Dimensions are written in mm in the YAML, used in metres here."""
import os
from dataclasses import asdict, dataclass

import yaml

GENERATION_DEFAULTS = dict(
    image_sets=None,          # list of lists of image keys; default: [all images]
    seeds=[1],
    mode="stochastic",        # TRELLIS multi-image mode: stochastic | multidiffusion
    ss_steps=25, slat_steps=25, ss_cfg=7.5, slat_cfg=3.0,
    align_faces=300_000,      # faces of the aligned candidate mesh
    orientation="auto",       # auto (lower end = front) | flip (if auto gets it backwards)
    select=None,              # force a candidate tag, e.g. "fl+rr_s1"
    import_=None,             # {tag: raw mesh path} previously generated meshes added as candidates
)

REPAIR_PRESETS = {
    "rough": dict(voxel=0.012, smooth=1.5, open=2, taubin=30, faces=60_000, flat_floor=True),
    "fine": dict(voxel=0.006, smooth=1.0, open=1, taubin=15, faces=400_000, flat_floor=True),
}


@dataclass
class Spec:
    """Published exterior dimensions in metres."""
    length: float
    width: float              # body width without mirrors
    height: float
    wheelbase: float
    front_overhang: float
    track_front: float
    track_rear: float
    tire_diameter: float
    tire_width: float
    ground_clearance: float

    @property
    def axle_f(self):
        return self.front_overhang

    @property
    def axle_r(self):
        return self.front_overhang + self.wheelbase

    @property
    def rear_overhang(self):
        return self.length - self.axle_r

    @property
    def tire_radius(self):
        return 0.5 * self.tire_diameter

    @property
    def track_min(self):
        return min(self.track_front, self.track_rear)

    def check(self):
        assert 0 < self.front_overhang < self.length and self.axle_r < self.length, "overhang/wheelbase inconsistent with length"
        assert self.width < 3 and self.length < 8, "dimensions must be given in mm in the YAML"

    def describe(self):
        mm = {k: round(v * 1000) for k, v in asdict(self).items()}
        mm["rear_overhang"] = round(self.rear_overhang * 1000)
        return mm


def _tire_diameter_mm(t):
    """'235/55R19' -> 740.5 mm"""
    w, rest = t.upper().split("/")
    ar, rim = rest.split("R")
    return float(rim) * 25.4 + 2 * float(w) * float(ar) / 100.0, float(w)


def load_config(path):
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    s = dict(cfg["spec"])
    if "tire" in s:
        d, w = _tire_diameter_mm(str(s.pop("tire")))
        s.setdefault("tire_diameter", d)
        s.setdefault("tire_width", w)
    if "track" in s:
        s.setdefault("track_front", s["track"]); s.setdefault("track_rear", s.pop("track"))
    spec = Spec(**{k: float(v) / 1000.0 for k, v in s.items()})
    spec.check()

    gen = dict(GENERATION_DEFAULTS); gen.update(cfg.get("generation") or {})
    gen["import"] = gen.pop("import_", None) or gen.get("import")
    images = cfg.get("images") or {}
    if not images:
        raise ValueError("config needs at least one image under 'images:'")
    if not gen["image_sets"]:
        gen["image_sets"] = [list(images.keys())]
    for st in gen["image_sets"]:
        for k in st:
            if k not in images:
                raise ValueError(f"image set refers to unknown image key '{k}'")

    rep_cfg = dict(cfg.get("repair") or {})
    presets = rep_cfg.pop("presets", None) or [rep_cfg.pop("preset", "rough")]
    repair = {}
    for p in presets:
        r = dict(REPAIR_PRESETS[p]); r.update(rep_cfg.get(p, {}) if isinstance(rep_cfg.get(p), dict) else {})
        repair[p] = r

    vid = cfg.get("id") or os.path.splitext(os.path.basename(path))[0]
    return dict(id=vid, name=cfg.get("name", vid), spec=spec, images=images, generation=gen,
                repair=repair, config_path=path, config_dir=os.path.dirname(path))
