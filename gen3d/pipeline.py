#!/usr/bin/env python3
"""
Car photo(s) -> 3D model pipeline (TRELLIS image-to-3D + symmetric watertight repair).

  # 1) find candidate photos for a new vehicle (Wikimedia Commons, licence shown)
  .venv/bin/python pipeline.py search "Kia EV6" --limit 24
  #    -> runs/_search/<query>/sheet.png + candidates.json ; pick titles into a config

  # 2) write vehicles/<id>.yaml (copy vehicles/TEMPLATE.yaml): spec in mm + images + options

  # 3) run everything (or some stages)
  .venv/bin/python pipeline.py run vehicles/ioniq5.yaml
  .venv/bin/python pipeline.py run vehicles/ev6.yaml --stages generate,select --seeds 1,2,3,4
  .venv/bin/python pipeline.py run vehicles/ev6.yaml --stages repair,render --preset fine

Outputs: runs/<id>/  (inputs/, candidates/<tag>/, <id>_final.glb, <id>_<preset>.stl/.glb/.obj,
         *_views.png, report.json, ATTRIBUTION.md)
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import trimesh  # noqa: E402

from car3d import images as im  # noqa: E402
from car3d.spec import load_config  # noqa: E402

STAGES = ["fetch", "generate", "select", "repair", "render"]


class Run:
    def __init__(self, cfg, out_root):
        self.cfg = cfg
        self.dir = os.path.join(out_root, cfg["id"])
        self.inputs = os.path.join(self.dir, "inputs")
        self.cands = os.path.join(self.dir, "candidates")
        for d in (self.dir, self.inputs, self.cands):
            os.makedirs(d, exist_ok=True)
        self.report_path = os.path.join(self.dir, "report.json")
        self.report = json.load(open(self.report_path)) if os.path.exists(self.report_path) else {}
        self.report.update(id=cfg["id"], name=cfg["name"], config=cfg["config_path"], spec_mm=cfg["spec"].describe())

    def p(self, name):
        return os.path.join(self.dir, name)

    def save(self):
        with open(self.report_path, "w") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)


def log(stage, *a):
    print(f"[{stage}]", *a, flush=True)


# ----------------------------------------------------------------------------- stages
def stage_fetch(run, force=False):
    metas, rgbas = {}, []
    for key, entry in run.cfg["images"].items():
        jpg, meta = im.fetch_image(key, entry, run.inputs, run.cfg["config_dir"])
        rgba = os.path.join(run.inputs, f"{key}_rgba.png")
        if force and os.path.exists(rgba):
            os.remove(rgba)
        im.remove_background(jpg, rgba)
        metas[key] = meta; rgbas.append(rgba)
        log("fetch", f"{key}: {meta['title']}  ({meta.get('license')}, {meta.get('artist')})")
    im.write_attribution(metas, run.p("ATTRIBUTION.md"))
    im.contact_sheet([os.path.join(run.inputs, f"{k}.jpg") for k in metas] + rgbas,
                     run.p("inputs_sheet.png"), labels=list(metas) + [f"{k} (matte)" for k in metas],
                     cols=max(2, len(metas)))
    run.report["images"] = metas


def candidate_tags(cfg, seeds=None):
    g = cfg["generation"]
    for st in g["image_sets"]:
        for s in (seeds or g["seeds"]):
            yield "+".join(st) + f"_s{s}", st, s


def stage_generate(run, seeds=None, force=False):
    from PIL import Image

    from car3d.align import align_candidate
    from car3d.generate import generate
    from car3d.render import render_views
    from car3d.score import score_candidate
    cfg, g = run.cfg, run.cfg["generation"]
    for tag, path in (g.get("import") or {}).items():
        d = os.path.join(run.cands, tag); os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, "info.json")) and not force:
            continue
        src = path if os.path.isabs(path) else os.path.join(cfg["config_dir"], path)
        raw = trimesh.load(src, force="mesh", process=False)
        raw.export(os.path.join(d, "raw.ply"))
        m, info = align_candidate(raw, cfg["spec"], g["align_faces"], g["orientation"])
        m.export(os.path.join(d, "aligned.glb"))
        info.update(score_candidate(m, cfg["spec"]), tag=tag, imported_from=os.path.abspath(src))
        render_views(m, os.path.join(d, "views.png"), f"{cfg['id']} {tag} (imported)")
        json.dump(info, open(os.path.join(d, "info.json"), "w"), indent=2)
        log("generate", f"{tag}: imported {src} -> wheel ratio {info['wheel_ratio']}, "
                        f"mirror err {info['mirror_error_mm']} mm -> score {info['score']}")
    for tag, keys, seed in candidate_tags(cfg, seeds):
        d = os.path.join(run.cands, tag); os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, "info.json")) and not force:
            log("generate", f"{tag}: exists, skipped (--force to redo)")
            continue
        rgbas = [Image.open(os.path.join(run.inputs, f"{k}_rgba.png")) for k in keys]
        raw, conds = generate(rgbas, seed, g)
        raw.export(os.path.join(d, "raw.ply"))
        for i, c in enumerate(conds):
            c.save(os.path.join(d, f"cond{i}_{keys[i]}.png"))
        m, info = align_candidate(raw, cfg["spec"], g["align_faces"], g["orientation"])
        m.export(os.path.join(d, "aligned.glb"))
        info.update(score_candidate(m, cfg["spec"]), tag=tag, images=keys, seed=seed, mode=g["mode"])
        render_views(m, os.path.join(d, "views.png"), f"{cfg['id']} {tag}")
        json.dump(info, open(os.path.join(d, "info.json"), "w"), indent=2)
        log("generate", f"{tag}: wheels {info['wheels']} ratio {info['wheel_ratio']}, "
                        f"mirror err {info['mirror_error_mm']} mm -> score {info['score']}")


def stage_select(run):
    from car3d.render import compare_sheet
    infos = []
    for tag in sorted(os.listdir(run.cands)):
        f = os.path.join(run.cands, tag, "info.json")
        if os.path.exists(f):
            infos.append(json.load(open(f)))
    if not infos:
        raise SystemExit("no candidates – run the generate stage first")
    forced = run.cfg["generation"].get("select")
    if forced:
        best = next(i for i in infos if i["tag"] == forced)
    else:
        best = max(infos, key=lambda i: i["score"])
    for i in sorted(infos, key=lambda i: -i["score"]):
        log("select", f"{'*' if i is best else ' '} {i['tag']:24s} score {i['score']:+.3f}  "
                      f"wheel ratio {i['wheel_ratio']:.2f}  mirror {i['mirror_error_mm']:.1f} mm")
    shutil.copy(os.path.join(run.cands, best["tag"], "aligned.glb"), run.p(f"{run.cfg['id']}_final.glb"))
    top = sorted(infos, key=lambda i: -i["score"])[:6]
    compare_sheet([os.path.join(run.cands, i["tag"], "views.png") for i in top],
                  [f"{i['tag']}  score {i['score']:+.2f}" for i in top], run.p("candidates_sheet.png"))
    run.report["candidates"] = infos
    run.report["selected"] = best["tag"]


def stage_repair(run, presets=None):
    from car3d.repair import repair
    src = trimesh.load(run.p(f"{run.cfg['id']}_final.glb"), force="mesh", process=False)
    run.report.setdefault("repair", {})
    for preset, params in run.cfg["repair"].items():
        if presets and preset not in presets:
            continue
        log("repair", f"preset '{preset}': {params}")
        m, info = repair(src, run.cfg["spec"], **params)
        for e in ("stl", "glb", "obj"):
            m.export(run.p(f"{run.cfg['id']}_{preset}.{e}"))
        info["params"] = params
        run.report["repair"][preset] = info


def stage_render(run):
    from car3d.render import render_inspect, render_views
    vid = run.cfg["id"]
    targets = [f"{vid}_final.glb"] + [f"{vid}_{p}.glb" for p in run.cfg["repair"]]
    for t in targets:
        if os.path.exists(run.p(t)):
            base = t[:-4]
            render_views(run.p(t), run.p(f"{base}_views.png"), base)
            render_inspect(run.p(t), run.p(f"{base}_inspect.png"), base)
            log("render", run.p(f"{base}_views.png"))


# ----------------------------------------------------------------------------- CLI
def cmd_run(a):
    cfg = load_config(a.config)
    if a.preset:
        from car3d.spec import REPAIR_PRESETS
        cfg["repair"] = {p: cfg["repair"].get(p, dict(REPAIR_PRESETS[p])) for p in a.preset.split(",")}
    run = Run(cfg, a.out)
    stages = a.stages.split(",") if a.stages else STAGES
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    log("run", f"{cfg['name']} ({cfg['id']}) -> {run.dir}, stages {stages}")
    for st in stages:
        t0 = time.time()
        if st == "fetch" and len(stages) > 1:
            # separate process: the BiRefNet session keeps several GB that onnxruntime does not
            # release, which together with TRELLIS exceeds 16 GB RAM (OOM kill)
            import subprocess
            cmd = [sys.executable, "-u", os.path.abspath(__file__), "--out", a.out, "run", a.config, "--stages", "fetch"]
            subprocess.run(cmd + (["--force"] if a.force else []), check=True)
            run.report = json.load(open(run.report_path))
            continue
        if st == "fetch":
            stage_fetch(run, a.force)
        elif st == "generate":
            stage_generate(run, seeds, a.force)
        elif st == "select":
            stage_select(run)
        elif st == "repair":
            stage_repair(run)
        elif st == "render":
            stage_render(run)
        else:
            raise SystemExit(f"unknown stage {st}; choose from {STAGES}")
        run.report.setdefault("timing_s", {})[st] = round(time.time() - t0, 1)
        run.save()
    log("run", f"done -> {run.dir}")


def cmd_search(a):
    slug = re.sub(r"[^a-z0-9]+", "_", a.query.lower()).strip("_")
    d = os.path.join(a.out, "_search", slug); os.makedirs(d, exist_ok=True)
    res = im.wikimedia_search(a.query, a.limit, width=640)
    if a.exclude:
        bad = [s.strip().lower() for s in a.exclude.split(",") if s.strip()]
        res = [r for r in res if not any(re.search(rf"\b{re.escape(b)}\b", r["title"].lower()) for b in bad)]
    if not res:
        raise SystemExit("no results – try another query / fewer --exclude terms")
    paths = []
    for i, r in enumerate(res):
        p = os.path.join(d, f"{i:02d}.jpg")
        if not os.path.exists(p):
            im.download(r["thumb"], p)
        paths.append(p)
        print(f"{i:2d}  {r['license']:14s} {r['title']}")
    json.dump(res, open(os.path.join(d, "candidates.json"), "w"), indent=2, ensure_ascii=False)
    im.contact_sheet(paths, os.path.join(d, "sheet.png"), labels=[f"{i}: {r['license']}" for i, r in enumerate(res)])
    print(f"\ncontact sheet: {os.path.join(d, 'sheet.png')}")
    print("Pick 1-4 views of the SAME colour/trim (e.g. front-left 3/4 + rear-right 3/4) and add their titles\n"
          "under 'images:' in vehicles/<id>.yaml as  key: {wikimedia: \"File:...\", view: front-left}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "runs"), help="output root (default gen3d/runs)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the pipeline for a vehicle config")
    r.add_argument("config")
    r.add_argument("--stages", help=f"comma list from {STAGES} (default: all)")
    r.add_argument("--seeds", help="override generation seeds, e.g. 1,2,3,4")
    r.add_argument("--preset", help="override repair presets, e.g. rough or rough,fine")
    r.add_argument("--force", action="store_true", help="recompute existing candidates / mattes")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("search", help="search Wikimedia Commons for input photos")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=24)
    s.add_argument("--exclude", help="comma list of title substrings to drop, e.g. 'interior,concept'")
    s.set_defaults(func=cmd_search)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
