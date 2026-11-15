#!/usr/bin/env python3
"""
GUI for the dimensional morph (dim:foh / roh / length / width / spoiler) of an STL-base study:
drag handles on the car with the mouse and watch the shape change live.

  python3 2_shape_morphing/dim_gui.py [study] [--port 8766] [--no-browser]
  e.g. python3 2_shape_morphing/dim_gui.py ioniq5

The browser (dim_editor.html) is for visualisation only. It gets a light preview mesh, the base
decimated by vertex clustering to ~--preview-faces triangles, and evaluates the same field as
dim_morph.py on it. The field is defined in space, so the preview moves exactly like the full
mesh. The server also sends the wheel freeze mask (at the preview vertices), the landmarks and
the stretch-zone transition tables, all computed by the Python code. Full-resolution geometry is
only made on request:
  * Check in Python  -> morph on the server: measured dimensions + QA (det J, flips), no files
  * Add to designs   -> appends the current values to studies/<study>/designs.csv (validated, no duplicates)
  * Export case      -> export_designs.export_case for that case (full STL + meta.json, updates index.json)
With --set <folder | manifest> a shape set (given STLs + their design values, see shape_set.py) is
listed in the editor: picking an STL sets the sliders / handles to its values and shows the given
STL next to the morpher's shape (overlay or deviation map). From the set panel:
  * Compare all            -> deviation of every STL from the morpher (also written to compare.json)
  * Set -> designs         -> the set's values as designs of the study (geometry by the morpher)
  * Import set as study    -> new study whose cases are the given STLs (export_designs writes them)
The server binds to 127.0.0.1 only and refuses cross-site / foreign-host requests (local_server.py). It writes designs.csv / study.json of the study (or of the
new import study), the study geometry/ folders and the set's compare.json.
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import design_table as dt  # noqa: E402
import dim_morph  # noqa: E402
import export_designs as ex  # noqa: E402
from mesh_preview import cluster_decimate  # noqa: E402
import shape_set  # noqa: E402
from local_server import request_problem  # noqa: E402
from design_table import cfg  # noqa: E402

CASE_RE = re.compile(r"^c\d{3,5}$")


class Base:
    """Base mesh of the study + everything the editor needs (computed once)."""

    def __init__(self, study, preview_faces=200_000):
        conf = dt.study_config(study)
        if conf["base"]["type"] != "stl":
            raise SystemExit(f"study '{study}' is not STL-based; dimensional variables need an STL base "
                             f"(python3 2_shape_morphing/doe.py <study> --base-stl <file.stl>)")
        self.study, self.base = study, conf["base"]
        self.path = ex._base_path(self.base)
        print(f"loading {cfg.rel(self.path)} ...", flush=True)
        self.V, self.F = ex._load_base(self.path)
        self.wheels = self.base["wheels"]
        self.lm = dim_morph.detect_landmarks(self.V, self.F, self.wheels, self.base.get("dim_landmarks"))
        self.freeze = ex.wheel_freeze_mask(self.V, self.wheels).astype(np.float32)
        m0 = dim_morph.DimMorph({}, self.lm, self.wheels)
        self.zones = {k: {"a": g.a, "b": g.b, "G": np.round(g.G, 7).tolist(), "p": np.round(g.p, 5).tolist(),
                          "max_slope": g.max_slope()}
                      for k, g in (("foh", m0.Gf), ("length", m0.Gl), ("roh", m0.Gr))}
        self.ref = dim_morph.measure(self.V, self.wheels)
        self.Vp, self.Fp = cluster_decimate(self.V, self.F, preview_faces)
        self.freeze_p = ex.wheel_freeze_mask(self.Vp, self.wheels).astype(np.float32)
        self.blob = self._mesh_blob()
        print(f"preview mesh for the browser: {len(self.Fp):,} triangles (full: {len(self.F):,})", flush=True)

    def _mesh_blob(self):
        V = self.Vp.astype(np.float32); F = self.Fp.astype(np.uint32)
        return struct.pack("<II", len(V), len(F)) + V.tobytes() + F.tobytes() + self.freeze_p.tobytes()

    def check(self, dims):
        """Full-resolution morph on the server: measured dimensions + the export QA (no files)."""
        mo = dim_morph.DimMorph(dims, self.lm, self.wheels)
        V1 = self.V + mo.displacement(self.V, freeze=self.freeze.astype(float))
        meas = dim_morph.measure(V1, mo.transform_wheels(self.wheels))
        zte = self.lm["spoiler_te"][1]
        meas["spoiler_z"] = (zte + mo.v["spoiler"]) * 1e3
        q = dim_morph.quality(mo, self.V, V1, self.F)
        return {"dims": dims, "measured_mm": {k: round(v, 1) for k, v in meas.items()},
                "change_mm": {k: round(meas[k] - self.ref.get(k, zte * 1e3), 1) for k in meas},
                "min_detJ": round(q["min_detJ"], 4), "min_detJ_bound": round(q["min_detJ_bound"], 4),
                "flipped_triangles": q["flipped_triangles"], "flipped_slivers": q["flipped_slivers"],
                "max_displacement_mm": round(q["max_displacement"] * 1e3, 1), "status": q["status"]}


def designs_info(study):
    out = []
    for r in dt.read_designs(study):
        out.append({"case": r["case"], "label": dt.label(r["params"]), "dims": dt.split_dims(r["params"]),
                    "exported": os.path.exists(os.path.join(dt.geometry_dir(study, r["case"]), "meta.json"))})
    return out


def add_design(study, dims):
    dims = {k: round(float(v), 1) for k, v in dims.items() if k in dim_morph.PARAMS and abs(float(v)) > 1e-9}
    params = dt.full({dt.DIM_PREFIX + k: v for k, v in dims.items()}, study)
    errs = dt.validate(params, study)
    if errs:
        raise ValueError(f"invalid design: {errs}")
    rows = dt.read_designs(study)
    if not rows:
        rows.append({"case": "c000", "params": dt.full({}, study)})
    key = dt.design_key(params, study)
    for r in rows:
        if dt.design_key(r["params"], study) == key:
            return {"case": r["case"], "label": dt.label(r["params"]), "new": False}
    nxt = 1 + max([int(r["case"][1:]) for r in rows if r["case"][1:].isdigit()] or [0])
    case = f"c{nxt:03d}"
    rows.append({"case": case, "params": params})
    dt.write_designs(study, rows)
    return {"case": case, "label": dt.label(params), "new": True}


class SetState:
    """A shape set opened in the GUI; comparisons are computed on demand and cached."""

    def __init__(self, path, B):
        self.manifest, self.items = shape_set.read_manifest(path)
        self.name = os.path.basename(os.path.dirname(self.manifest)) or "set"
        self.B, self.stats, self.blobs = B, {}, {}
        self.lock = threading.Lock()

    def info(self):
        return {"name": self.name, "manifest": cfg.rel(self.manifest),
                "items": [{"i": i, "name": it["name"], "file": cfg.rel(it["file"]), "dims": it["dims"],
                           "label": shape_set.dims_label(it["dims"]),
                           "out_of_range": shape_set.out_of_range(it["dims"]), "stats": self.stats.get(it["name"])}
                          for i, it in enumerate(self.items)]}

    def item(self, i):
        if not 0 <= i < len(self.items):
            raise ValueError("no such set item")
        with self.lock:
            it = self.items[i]
            if it["name"] not in self.blobs:
                st, (V, F, dev) = shape_set.compare(self.B, it)
                self.stats[it["name"]] = st
                V = V.astype(np.float32); F = F.astype(np.uint32)
                self.blobs[it["name"]] = struct.pack("<II", len(V), len(F)) + V.tobytes() + F.tobytes() + dev.astype(np.float32).tobytes()
                json.dump(self.stats, open(os.path.join(os.path.dirname(self.manifest), "compare.json"), "w"), indent=2)
            return self.stats[it["name"]], self.blobs[it["name"]]


def make_handler(B, SET=None):
    editor = os.path.join(HERE, "dim_editor.html")
    lock = threading.Lock()                      # designs.csv / export: one at a time

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("  [dim-gui] " + fmt % args + "\n")

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            bad = request_problem(self)
            if bad:
                return self._send(403, {"error": bad})
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, open(editor, "rb").read(), "text/html; charset=utf-8")
                if u.path == "/api/info":
                    return self._send(200, {
                        "study": B.study, "base": cfg.rel(B.path), "name": B.base.get("name", ""),
                        "wheels": B.wheels, "landmarks": B.lm, "zones": B.zones,
                        "spoiler": {"ramp": dim_morph.SPOILER_RAMP, "dz": list(dim_morph.SPOILER_DZ)},
                        "params": {k: {"baseline": v[0], "lower": v[1], "upper": v[2], "unit": v[3], "desc": v[4]}
                                   for k, v in dim_morph.PARAMS.items()},
                        "reference_mm": {k: round(v, 1) for k, v in B.ref.items()},
                        "full_triangles": int(len(B.F)), "preview_triangles": int(len(B.Fp)),
                        "designs": designs_info(B.study)})
                if u.path == "/api/mesh.bin":
                    return self._send(200, B.blob, "application/octet-stream")
                if u.path == "/api/set":
                    return self._send(200, SET.info() if SET else None)
                if u.path == "/api/set/mesh.bin":
                    if not SET:
                        raise ValueError("no shape set (start with --set)")
                    _, blob = SET.item(int((parse_qs(u.query).get("i") or ["-1"])[0]))
                    return self._send(200, blob, "application/octet-stream")
                if u.path == "/api/designs":
                    return self._send(200, designs_info(B.study))
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                return self._send(400, {"error": str(e)})

        def do_POST(self):
            u = urlparse(self.path); q = parse_qs(u.query)
            bad = request_problem(self)
            if bad:
                return self._send(403, {"error": bad})
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 100_000:
                    raise ValueError("request too large")
                body = json.loads(self.rfile.read(n) or b"{}")
                if u.path == "/api/check":
                    dims = {k: float(v) for k, v in body.get("dims", {}).items() if abs(float(v)) > 1e-9}
                    return self._send(200, B.check(dims))
                if u.path == "/api/add_design":
                    with lock:
                        return self._send(200, add_design(B.study, body.get("dims", {})))
                if u.path.startswith("/api/set/"):
                    if not SET:
                        raise ValueError("no shape set (start with --set)")
                    if u.path == "/api/set/compare":
                        idx = body.get("items") or list(range(len(SET.items)))
                        return self._send(200, [SET.item(int(i))[0] for i in idx])
                    if u.path == "/api/set/to_designs":
                        with lock:
                            res = shape_set.to_designs(B.study, SET.items)
                        return self._send(200, [{"name": n, "case": c, "new": new} for n, c, new in res])
                    if u.path == "/api/set/import":
                        new = body.get("study") or f"{B.study}_{re.sub(r'[^A-Za-z0-9_]', '_', SET.name)}"
                        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,64}", new):
                            raise ValueError("study name: letters, digits, _ or -")
                        for i in range(len(SET.items)):
                            SET.item(i)                     # alignment + deviation for every STL first
                        with lock:
                            rows, imports, warn = shape_set.import_set(B.study, SET.items, new, SET.name, SET.stats)
                        return self._send(200, {"study": new, "cases": len(rows), "imported": len(imports), "warnings": warn,
                                                "next": f"python3 2_shape_morphing/export_designs.py {new}"})
                    raise ValueError("unknown set action")
                if u.path == "/api/export":
                    case = (q.get("case") or [""])[0]
                    if not CASE_RE.match(case):
                        raise ValueError("bad case name")
                    row = next((r for r in dt.read_designs(B.study) if r["case"] == case), None)
                    if row is None:
                        raise ValueError(f"{case} is not in designs.csv")
                    with lock:
                        meta = ex.export_case(B.study, case, row["params"], quiet=True)
                        ex.write_index(B.study)
                    return self._send(200, {"case": case, "label": meta["label"], "frontal_area": meta["frontal_area"],
                                            "qa": meta["qa"].get("dims", {}), "wheels": meta["wheels"],
                                            "path": cfg.rel(dt.geometry_dir(B.study, case))})
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                return self._send(400, {"error": str(e)})

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", nargs="?", default="ioniq5")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--set", metavar="DIR|MANIFEST", help="shape set to browse (given STLs + design values, see shape_set.py)")
    ap.add_argument("--preview-faces", type=int, default=200_000,
                    help="triangles of the browser preview mesh (0 = full resolution)")
    a = ap.parse_args()
    B = Base(a.study, a.preview_faces)
    SET = SetState(a.set, B) if a.set else None
    if SET:
        print(f"shape set '{SET.name}': {len(SET.items)} STLs ({cfg.rel(SET.manifest)})", flush=True)
    srv = None
    for port in range(a.port, a.port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(B, SET))
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit(f"no free port in {a.port}..{a.port + 19}")
    url = f"http://localhost:{port}/"
    print(f"dimensional morph GUI for study '{a.study}' ({cfg.rel(B.path)}, {len(B.F):,} triangles, "
          f"preview {len(B.Fp):,})\n"
          f"  open {url}   (Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        try:
            if "microsoft" in open("/proc/version").read().lower():
                subprocess.Popen(["cmd.exe", "/c", "start", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
