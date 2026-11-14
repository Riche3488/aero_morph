#!/usr/bin/env python3
"""
GUI drag-morphing for ANY STL: local web server + drag_editor.html.

  python3 2_shape_morphing/morph_gui.py <geometry.stl> [--port 8765] [--out DIR] [--no-browser]

  e.g. python3 2_shape_morphing/morph_gui.py gen3d/runs/ioniq6/ioniq6_rough.stl

Opens the editor in the browser with the model already loaded. In the editor:
  * drag handles / place anchors (live RBF preview, fold check)
  * "Save to modes/"         -> 2_shape_morphing/modes/<name>.json (usable as design variable mode:<name>)
  * "Apply in Python"        -> full-resolution morph with rbf.py on the server:
                                <out>/<stem>__<name>.stl, <stem>__<name>_qa.json, <stem>__<name>.png
Server binds to 127.0.0.1 only. It serves just the editor, the given STL and its own outputs,
and only writes inside modes/ and the output folder.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import pipeline_config as cfg  # noqa: E402
import rbf  # noqa: E402
from drag_morph import read_stl, render, write_stl  # noqa: E402

NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def make_handler(model_path, out_dir):
    model_name = os.path.basename(model_path)
    stem = os.path.splitext(model_name)[0]
    editor = os.path.join(HERE, "drag_editor.html")
    state = {"mesh": None}
    lock = threading.Lock()

    def mesh():
        with lock:
            if state["mesh"] is None:
                state["mesh"] = read_stl(model_path)
            return state["mesh"]

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stderr.write("  [gui] " + fmt % args + "\n")

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

        def _name(self, q):
            name = (q.get("name") or [""])[0]
            if not NAME_RE.match(name):
                raise ValueError("mode name must be 1-64 chars of letters, digits, _ or -")
            return name

        def do_GET(self):
            u = urlparse(self.path); q = parse_qs(u.query)
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, open(editor, "rb").read(), "text/html; charset=utf-8")
                if u.path == "/api/info":
                    return self._send(200, {"model": os.path.relpath(model_path, cfg.ROOT) if model_path.startswith(cfg.ROOT) else model_path,
                                            "model_name": model_name, "modes_dir": os.path.relpath(cfg.MORPH_MODES, cfg.ROOT),
                                            "out_dir": os.path.relpath(out_dir, cfg.ROOT) if out_dir.startswith(cfg.ROOT) else out_dir,
                                            "default_mode_name": f"{stem}_mode"})
                if u.path == "/api/model.stl":
                    return self._send(200, open(model_path, "rb").read(), "application/octet-stream")
                if u.path == "/api/modes":
                    os.makedirs(cfg.MORPH_MODES, exist_ok=True)
                    return self._send(200, sorted(f[:-5] for f in os.listdir(cfg.MORPH_MODES) if f.endswith(".json")))
                if u.path == "/api/mode":
                    p = os.path.join(cfg.MORPH_MODES, self._name(q) + ".json")
                    return self._send(200, open(p, "rb").read())
                if u.path.startswith("/api/out/"):
                    f = os.path.basename(u.path)                  # no sub-paths
                    p = os.path.join(out_dir, f)
                    if not (NAME_RE.match(os.path.splitext(f)[0].replace("__", "_")) and f.endswith(".png") and os.path.exists(p)):
                        return self._send(404, {"error": "not found"})
                    return self._send(200, open(p, "rb").read(), "image/png")
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                return self._send(400, {"error": str(e)})

        def do_POST(self):
            u = urlparse(self.path); q = parse_qs(u.query)
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 20_000_000:
                    raise ValueError("request too large")
                spec = json.loads(self.rfile.read(n) or b"{}")
                name = self._name(q)
                if u.path == "/api/save_mode":
                    spec.pop("preview_amplitude", None)
                    spec["description"] = spec.get("description") or f"drag mode made on {model_name}"
                    spec["source_geometry"] = os.path.relpath(model_path, cfg.ROOT) if model_path.startswith(cfg.ROOT) else model_path
                    spec.setdefault("amplitude_range", [-2, 2])
                    os.makedirs(cfg.MORPH_MODES, exist_ok=True)
                    p = os.path.join(cfg.MORPH_MODES, name + ".json")
                    json.dump(spec, open(p, "w"), indent=2)
                    return self._send(200, {"name": name, "path": os.path.relpath(p, cfg.ROOT)})
                if u.path == "/api/apply":
                    amp = float((q.get("amp") or ["1"])[0])
                    spec = rbf.combine_specs([(spec, amp)])
                    V, F = mesh()
                    morph = rbf.RBFMorph(spec, V, rbf.vertex_normals(V, F))
                    V1 = V + morph.displacement(V)
                    rep = rbf.quality_report(morph, V, V1, F)
                    os.makedirs(out_dir, exist_ok=True)
                    base = f"{stem}__{name}"
                    stl = os.path.join(out_dir, base + ".stl")
                    write_stl(stl, V1, F, name)
                    json.dump({"source": model_path, "spec": spec, "amplitude": amp, "qa": rep},
                              open(os.path.join(out_dir, base + "_qa.json"), "w"), indent=2)
                    render(os.path.join(out_dir, base + ".png"), [(stem, V, F)], [(stem, V1, F)], [],
                           f"{model_name}: {name} (amplitude {amp:g})")
                    return self._send(200, {"qa": rep, "stl": os.path.relpath(stl, cfg.ROOT) if stl.startswith(cfg.ROOT) else stl,
                                            "png_url": f"api/out/{base}.png"})
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                return self._send(400, {"error": str(e)})

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stl")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", default=None, help="output folder (default: 2_shape_morphing/gui_output/<stem>)")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    model = os.path.abspath(a.stl)
    if not os.path.exists(model):
        raise SystemExit(f"not found: {model}")
    out = os.path.abspath(a.out or os.path.join(HERE, "gui_output", os.path.splitext(os.path.basename(model))[0]))
    srv = None
    for port in range(a.port, a.port + 20):          # next free port if busy (e.g. a second GUI)
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(model, out))
            break
        except OSError:
            continue
    if srv is None:
        raise SystemExit(f"no free port in {a.port}..{a.port + 19}")
    url = f"http://localhost:{port}/"
    print(f"morph GUI for {model}\n  open {url}   (Ctrl+C to stop)")
    if not a.no_browser:
        try:  # WSL: open in the Windows browser; elsewhere use the default browser
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
