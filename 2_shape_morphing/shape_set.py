#!/usr/bin/env python3
"""
Shape sets: given STL files (1.stl, 2.stl, ... n.stl) with known changes of the dimensional
design variables (1.stl: -40 mm FOH, ..., n.stl: +50 mm ROH), connected to the morpher, the GUI
and the study pipeline.

  python3 2_shape_morphing/shape_set.py <set dir | manifest> [--study ioniq5] [--compare]
                                         [--to-designs] [--import [STUDY]]
  python3 2_shape_morphing/dim_gui.py ioniq5 --set <set dir | manifest>      # browse it in the GUI

Manifest (in the set folder: manifest.txt / .csv / .json, or params.*; or pass the file itself),
values in mm, variable names foh / roh / length / width / spoiler (also FOH, dim:foh,
front_overhang, rear_overhang, wheelbase, spoiler_height, ...):
  text    1.stl : -40mm FOH
          2.stl : +50mm ROH, +20mm spoiler          (or: 2.stl roh=50 spoiler=20)
  csv     file,foh,roh,length,width,spoiler
          1.stl,-40,0,0,0,0
  json    {"1.stl": {"foh": -40}, "2.stl": {"roh": 50}}

Compare: the base of the study is morphed with dim_morph.py at the given values. The given STL is
aligned to it by a translation (point-to-plane least squares; this absorbs e.g. the -5 mm tyre-sink
offset of exported cases) and its signed deviation from the morphed surface is measured
(+ = given STL outside). A set made with this morpher matches to ~0; a set from another tool
shows where it differs; a wrong manifest entry shows up as a large deviation.
STLs in mm (extent > 100) are scaled to m.

Study hand-off:
  --to-designs    add the values as designs of the study (geometry made by the morpher)
  --import STUDY  make STUDY (default <study>_<set>) whose cases ARE the given STLs:
                  study.json "imports" maps case -> file; export_designs.py writes them
                  (aligned, tyre-sink offset, wheels split into regions) with the design values
                  in meta.json, so stage 3 runs them and stage 4 learns values -> geometry.
"""
import argparse
import csv
import json
import os
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import design_table as dt  # noqa: E402
import dim_morph  # noqa: E402
import export_designs as ex  # noqa: E402
from rbf import vertex_normals  # noqa: E402
from common import mesh_io as mio  # noqa: E402
from design_table import cfg  # noqa: E402
from mesh_preview import cluster_decimate  # noqa: E402

MANIFEST_NAMES = ("manifest.txt", "manifest.csv", "manifest.json", "params.txt", "params.csv", "params.json")
ALIASES = {
    "foh": "foh", "frontoverhang": "foh", "fronthang": "foh",
    "roh": "roh", "rearoverhang": "roh",
    "length": "length", "len": "length", "l": "length", "wheelbase": "length", "wb": "length",
    "width": "width", "w": "width", "bodywidth": "width",
    "spoiler": "spoiler", "spoilerheight": "spoiler", "spoilerz": "spoiler",
}
MATCH_MM = 2.0              # deviation p99 below this: the STL is reproduced by the morpher
CLOSE_MM = 10.0             # (p99, not p95: a wrong overhang value only affects ~5 % of the surface)
MAX_SHIFT = 0.05            # alignment translation beyond this = different coordinate frame


# ----------------------------------------------------------------------------- manifest
def var_name(s):
    k = re.sub(r"[\s_\-]", "", s.lower().replace("dim:", ""))
    if k not in ALIASES:
        raise ValueError(f"unknown design variable '{s}' (use {', '.join(dim_morph.PARAMS)})")
    return ALIASES[k]


NUM = r"[+-]?\d+(?:\.\d+)?"


def parse_text_entry(rest):
    """'-40mm FOH, +20 mm spoiler' | 'FOH -40' | 'foh=-40 roh=20' -> {var: mm}"""
    out = {}
    if "=" in rest:
        for n, v in re.findall(rf"([A-Za-z][\w:]*)\s*=\s*({NUM})", rest):
            out[var_name(n)] = float(v)
        return out
    for part in re.split(r"[,;]", rest):
        part = part.strip()
        if not part:
            continue
        m = (re.fullmatch(rf"({NUM})\s*(?:mm)?\s+([A-Za-z][\w: ]*)", part)
             or re.fullmatch(rf"({NUM})\s*mm\s*([A-Za-z][\w: ]*)", part))
        if m:
            out[var_name(m.group(2).strip())] = float(m.group(1)); continue
        m = re.fullmatch(rf"([A-Za-z][\w: ]*?)\s*[:]?\s*({NUM})\s*(?:mm)?", part)
        if m:
            out[var_name(m.group(1).strip())] = float(m.group(2)); continue
        raise ValueError(f"cannot read '{part}' (write e.g. '-40mm FOH' or 'foh=-40')")
    return out


def find_manifest(path):
    if os.path.isfile(path):
        return os.path.abspath(path)
    for n in MANIFEST_NAMES:
        p = os.path.join(path, n)
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(f"no manifest in {path} (one of {', '.join(MANIFEST_NAMES)}; format in shape_set.py)")


def read_manifest(path):
    """-> list of {"name", "file" (abs), "dims" {var: mm}} in manifest order."""
    mf = find_manifest(path)
    root = os.path.dirname(mf)
    rows = []
    if mf.endswith(".json"):
        data = json.load(open(mf))
        items = data.items() if isinstance(data, dict) else [(d["file"], {k: v for k, v in d.items() if k != "file"}) for d in data]
        rows = [(f, {var_name(k): float(v) for k, v in d.items()}) for f, d in items]
    elif mf.endswith(".csv"):
        with open(mf, newline="") as fh:
            for r in csv.DictReader(fh):
                f = r.pop("file", None) or r.pop("stl", None) or r.pop("name", None)
                rows.append((f, {var_name(k): float(v) for k, v in r.items() if k and v not in ("", None)}))
    else:
        for ln, line in enumerate(open(mf), 1):
            line = line.split("#")[0].strip()
            if not line:
                continue
            m = re.match(r"^\s*(\S+?\.stl)\s*[:,=]?\s*(.*)$", line, re.I)
            if not m:
                raise ValueError(f"{mf}:{ln}: expected '<file>.stl : <change> <variable>, ...'")
            try:
                rows.append((m.group(1), parse_text_entry(m.group(2))))
            except ValueError as e:
                raise ValueError(f"{mf}:{ln}: {e}") from None
    out = []
    for f, dims in rows:
        p = f if os.path.isabs(f) else os.path.join(root, f)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{f} (listed in {mf}) not found")
        out.append({"name": os.path.basename(f), "file": os.path.abspath(p),
                    "dims": {k: v for k, v in dims.items() if abs(v) > 1e-12}})
    if not out:
        raise ValueError(f"{mf}: no entries")
    return mf, out


def dims_label(dims):
    return ", ".join(f"dim:{k}={v:+g}mm" for k, v in dims.items()) or "baseline"


def out_of_range(dims):
    bad = []
    for k, v in dims.items():
        lo, hi = dim_morph.PARAMS[k][1:3]
        if not lo <= v <= hi:
            bad.append(f"dim:{k}={v:+g} outside [{lo:g}, {hi:g}]")
    return bad


# ----------------------------------------------------------------------------- comparison
def read_given(path):
    V, F = mio.read_stl(path)
    scale = 1e-3 if np.ptp(V, axis=0).max() > 100 else 1.0      # mm -> m
    return V * scale, F, scale


class Prediction:
    """The morpher's shape for one set of values on the study base (full resolution)."""

    def __init__(self, base, dims):
        mo = dim_morph.DimMorph(dims, base.lm, base.wheels)
        self.V = base.V + mo.displacement(base.V, freeze=np.asarray(base.freeze, float))
        self.N = vertex_normals(self.V, base.F)
        self.tree = cKDTree(self.V)
        self.wheels = mo.transform_wheels(base.wheels)

    def signed(self, P):
        _, i = self.tree.query(P)
        return np.einsum("ij,ij->i", P - self.V[i], self.N[i]), i

    def align(self, P, iters=8):
        """Translation t minimising sum ((p + t - q) . n)^2 (point-to-plane), trimmed: points
        deviating more than max(2 mm, 3 x median) are left out, so a region that really differs
        (e.g. a wrong overhang value) does not drag the alignment along."""
        t = np.zeros(3)
        for _ in range(iters):
            d, i = self.signed(P + t)
            keep = np.abs(d) <= max(0.002, 3 * np.median(np.abs(d)))
            n, d = self.N[i][keep], d[keep]
            A = n.T @ n + 1e-9 * np.eye(3)
            dt_ = np.linalg.solve(A, -(n * d[:, None]).sum(0))
            t += dt_
            if np.linalg.norm(dt_) < 1e-6:
                break
        return t


def compare(base, item, preview_faces=150_000):
    """Deviation of a given STL from the morpher at the item's values. Returns (stats, preview)
    with preview = (V aligned, F, signed deviation mm per preview vertex)."""
    V, F, scale = read_given(item["file"])
    pred = Prediction(base, item["dims"])
    sub = V[:: max(1, len(V) // 200_000)]                       # alignment on <= 200k points
    t = pred.align(sub)
    d, _ = pred.signed(V + t)
    a = np.abs(d) * 1e3
    p95, p99 = (float(np.percentile(a, q)) for q in (95, 99))
    status = "match" if p99 <= MATCH_MM else "close" if p99 <= CLOSE_MM else "differs"
    shift = float(np.linalg.norm(t))
    chk = mio.check_watertight(V, F)
    stats = {"name": item["name"], "dims": item["dims"], "label": dims_label(item["dims"]),
             "triangles": int(len(F)), "units_scale": scale, "watertight": chk["edges_not_2"] == 0,
             "translation_mm": np.round(t * 1e3, 2).tolist(),
             "dev_mean_mm": round(float(a.mean()), 2), "dev_rms_mm": round(float(np.sqrt((a ** 2).mean())), 2),
             "dev_p95_mm": round(p95, 2), "dev_p99_mm": round(p99, 2), "dev_max_mm": round(float(a.max()), 2),
             "dev_signed_mean_mm": round(float(d.mean() * 1e3), 2),
             "status": status if shift <= MAX_SHIFT else "frame?", "out_of_range": out_of_range(item["dims"])}
    Vp, Fp = cluster_decimate(V, F, preview_faces)
    dp, _ = pred.signed(Vp + t)
    return stats, (Vp + t, Fp, dp * 1e3)


# ----------------------------------------------------------------------------- study hand-off
def study_base(study):
    """V, F, wheels, landmarks, freeze mask of an STL-base study (what Prediction needs)."""
    conf = dt.study_config(study)
    if conf["base"]["type"] != "stl":
        raise SystemExit(f"study '{study}' is not STL-based")

    class B:
        pass
    b = B()
    b.study, b.base = study, conf["base"]
    b.path = ex._base_path(b.base)
    b.V, b.F = ex._load_base(b.path)
    b.wheels = b.base["wheels"]
    b.lm = dim_morph.detect_landmarks(b.V, b.F, b.wheels, b.base.get("dim_landmarks"))
    b.freeze = ex.wheel_freeze_mask(b.V, b.wheels)
    return b


def to_designs(study, items):
    """Add the values as designs (geometry by the morpher). -> [(item name, case, new?)]"""
    rows = dt.read_designs(study)
    if not rows:
        rows.append({"case": "c000", "params": dt.full({}, study)})
    keys = {dt.design_key(r["params"], study): r["case"] for r in rows}
    nxt = 1 + max([int(r["case"][1:]) for r in rows if r["case"][1:].isdigit()] or [0])
    out = []
    for it in items:
        p = dt.full({dt.DIM_PREFIX + k: v for k, v in it["dims"].items()}, study)
        dt.validate(p, study)                                   # raises outside the morpher range
        k = dt.design_key(p, study)
        if k in keys:
            out.append((it["name"], keys[k], False)); continue
        case = f"c{nxt:03d}"; nxt += 1
        rows.append({"case": case, "params": p}); keys[k] = case
        out.append((it["name"], case, True))
    dt.write_designs(study, rows)
    return out


def import_set(src_study, items, new_study, set_name, stats=None):
    """New study whose cases are the given STLs (+ c000 = morpher baseline if the set has none)."""
    if dt.read_designs(new_study):
        raise ValueError(f"study '{new_study}' already exists; choose another name")
    conf = dt.study_config(src_study)
    base = dict(conf["base"])
    imports, rows, seen, warn = {}, [], {}, []
    if not any(not it["dims"] for it in items):
        rows.append({"case": "c000", "params": {}})
    for n, it in enumerate(items, 1):
        case = f"c{n:03d}"
        p = {dt.DIM_PREFIX + k: v for k, v in it["dims"].items()}
        key = tuple(sorted(p.items()))
        if key in seen:                              # allowed (cases are separate geometries), but suspicious
            warn.append(f"{it['name']} has the same values as {seen[key]}")
        seen.setdefault(key, it["name"])
        st = (stats or {}).get(it["name"], {})
        if st.get("status") in ("differs", "frame?"):
            warn.append(f"{it['name']}: {st['status']} from the morpher at its listed values (p99 {st.get('dev_p99_mm')} mm)")
        rows.append({"case": case, "params": p})
        imports[case] = {"file": cfg.rel(it["file"]), "sha1": ex.file_hash(it["file"]), "set": set_name,
                         "translation_mm": st.get("translation_mm"), "units_scale": st.get("units_scale"),
                         "deviation_vs_morpher": {k: st[k] for k in ("dev_mean_mm", "dev_p99_mm", "dev_max_mm", "status") if k in st}}
    dt.write_study_config(new_study, {"base": base, "imports": imports,
                                      "imported_from": {"set": set_name, "study": src_study}})
    dt.write_designs(new_study, rows)
    return rows, imports, warn


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("set", help="set folder (with manifest.*) or the manifest file")
    ap.add_argument("--study", default="ioniq5", help="STL-base study the set belongs to")
    ap.add_argument("--compare", action="store_true", help="measure each STL against the morpher")
    ap.add_argument("--to-designs", action="store_true", help="add the values to the study's designs.csv")
    ap.add_argument("--import", dest="imp", nargs="?", const="", metavar="STUDY",
                    help="new study whose cases are the given STLs (default <study>_<set>)")
    a = ap.parse_args()
    mf, items = read_manifest(a.set)
    set_name = os.path.basename(os.path.dirname(mf)) or "set"
    print(f"shape set '{set_name}' ({cfg.rel(mf)}): {len(items)} STLs")
    for it in items:
        w = out_of_range(it["dims"])
        print(f"  {it['name']:20s} {dims_label(it['dims'])}" + (f"   ! {'; '.join(w)}" if w else ""))
    stats = {}
    if a.compare or a.imp is not None:
        b = study_base(a.study)
        print(f"compare with the morpher on '{a.study}' ({cfg.rel(b.path)}):")
        for it in items:
            s, _ = compare(b, it)
            stats[it["name"]] = s
            print(f"  {it['name']:20s} {s['status']:8s} deviation mean {s['dev_mean_mm']:.2f} / p99 {s['dev_p99_mm']:.2f} / "
                  f"max {s['dev_max_mm']:.2f} mm, shift {s['translation_mm']} mm" + ("" if s["watertight"] else "  (not watertight)"))
        json.dump(stats, open(os.path.join(os.path.dirname(mf), "compare.json"), "w"), indent=2)
    if a.to_designs:
        for name, case, new in to_designs(a.study, items):
            print(f"  {name} -> {a.study}/{case}" + ("" if new else " (already there)"))
    if a.imp is not None:
        new = a.imp or f"{a.study}_{re.sub(r'[^A-Za-z0-9_]', '_', set_name)}"
        rows, imports, warn = import_set(a.study, items, new, set_name, stats)
        for w in warn:
            print(f"  ! {w}")
        print(f"study '{new}': {len(rows)} cases ({len(imports)} imported STLs). Next:\n"
              f"  python3 2_shape_morphing/export_designs.py {new}")


if __name__ == "__main__":
    main()
