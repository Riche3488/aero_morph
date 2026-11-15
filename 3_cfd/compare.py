#!/usr/bin/env python3
"""
Collect CFD results of a study and draw the comparison report.

  python3 3_cfd/compare.py <study>

Outputs (3_cfd/runs/<study>/):
  results.csv                 one row per finished case (design variables + Cd, CdA, Cl, ...)
                              -> this is the training table for 4_dl_training
  report/ranking.png          dCd / dCdA vs baseline per design, best first
  report/sensitivity.png      Cd and CdA vs each varied design variable (+ linear slopes)
  report/profiles_best.png    centre-line / plan cuts of the 3 lowest-CdA designs vs baseline

Reads only stage-3 data: cases/<case>/result.json, case.json (the design variables that were
actually run) and constant/geometry (the case's own copy of the surfaces).
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline_config as cfg  # noqa: E402
from common.mesh_io import plane_cut, read_stl  # noqa: E402
from run_case import geometry_changed  # noqa: E402

MODE_PREFIX = "mode:"

RESULT_KEYS = ["Cd", "Cd_std", "CdA", "Cl", "Cl_front", "Cl_rear", "drag_N", "frontal_area", "cells", "iterations"]


def run_dir(study):
    return os.path.join(cfg.CFD_RUNS, study)


def load_results(study):
    """Finished cases with the design variables they were run with (case.json), not the current
    design table, so results and variables always belong together."""
    out, changed = [], []
    cdir = os.path.join(run_dir(study), "cases")
    for case in sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []:
        f, cj = os.path.join(cdir, case, "result.json"), os.path.join(cdir, case, "case.json")
        if not (os.path.exists(f) and os.path.exists(cj)):
            continue
        res, c = json.load(open(f)), json.load(open(cj))
        variables = c.get("variables")
        if variables is None:                    # cases prepared before case.json carried them: hand-off meta
            mp = os.path.join(cfg.geometry_dir(study, case), "meta.json")
            variables = json.load(open(mp)).get("variables", {}) if os.path.exists(mp) else {}
        res.update(case=case, params=c.get("params", res.get("params", {})), label=c.get("label", case),
                   variables=variables)
        if geometry_changed(study, case):
            changed.append(case)
        out.append(res)
    if changed:
        print(f"note: the exported geometry of {changed} changed after they were run; their rows keep the "
              f"variables they were run with (re-run: run_study.py {study} --rerun-changed)")
    return out


def columns(res):
    """Design variables in order of first appearance (stage-2 order: parametric, then modes)."""
    cols = []
    for r in res:
        cols += [k for k in r["params"] if k not in cols]
    return cols


def write_csv(study, res):
    cols = columns(res)
    path = os.path.join(run_dir(study), "results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study", "case", "label"] + cols + RESULT_KEYS)
        for r in res:
            w.writerow([study, r["case"], r["label"]] + [r["params"].get(k, 0.0) for k in cols]
                       + [r.get(k) for k in RESULT_KEYS])
    return path


def baseline_of(res):
    return next((r for r in res if r["label"] == "baseline"), None)


def _var(k, res):
    return next((r["variables"][k] for r in res if k in r.get("variables", {})), {})


def unit(k, res):
    return _var(k, res).get("unit", "amp" if k.startswith(MODE_PREFIX) else "-")


def base_value(k, res, base):
    v = _var(k, res)
    if "baseline" in v:
        return v["baseline"]
    return 0.0 if k.startswith(MODE_PREFIX) else base["params"].get(k, 0.0)


def ranking(study, res, base, out):
    res = sorted(res, key=lambda r: r["CdA"])
    names = [f"{r['case']}  {r['label']}" for r in res]
    dcd = np.array([r["Cd"] - base["Cd"] for r in res]) * 1000
    dcda = np.array([(r["CdA"] / base["CdA"] - 1) * 100 for r in res])
    err = np.array([r["Cd_std"] for r in res]) * 1000
    fig, ax = plt.subplots(1, 2, figsize=(15, max(3.5, 0.38 * len(res) + 1.5)), sharey=True)
    y = np.arange(len(res))
    ax[0].barh(y, dcd, xerr=err, color=["tab:green" if v < 0 else "tab:red" for v in dcd], alpha=.8, capsize=2)
    ax[0].set_yticks(y); ax[0].set_yticklabels(names, fontsize=8); ax[0].invert_yaxis()
    ax[0].axvline(0, color="k", lw=.8); ax[0].set_xlabel("ΔCd [counts = 0.001]  (error bar: ±1σ iteration noise)")
    ax[0].set_title(f"ΔCd vs baseline (Cd = {base['Cd']:.3f})")
    ax[1].barh(y, dcda, color=["tab:green" if v < 0 else "tab:red" for v in dcda], alpha=.8)
    ax[1].axvline(0, color="k", lw=.8); ax[1].set_xlabel("ΔCdA [%]  (drag force at equal speed)")
    ax[1].set_title(f"ΔCdA vs baseline (CdA = {base['CdA']:.3f} m²)")
    for a in ax:
        a.grid(alpha=.3, axis="x")
    fig.suptitle(f"study '{study}': {len(res)} designs, sorted by CdA")
    plt.tight_layout(); plt.savefig(os.path.join(out, "ranking.png"), dpi=100); plt.close()


def sensitivity(study, res, base, out):
    cols = columns(res)
    varied = [k for k in cols if len({round(r["params"].get(k, 0.0), 6) for r in res}) > 1]
    if not varied:
        return
    n = len(varied); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.8 * nrow), squeeze=False)
    X = np.array([[r["params"].get(k, 0.0) for k in varied] for r in res])
    slopes = None
    if len(res) >= n + 2:
        slopes = np.linalg.lstsq(np.c_[np.ones(len(res)), X], np.array([r["Cd"] for r in res]), rcond=None)[0][1:]
    for ax, (j, k) in zip(axs.flat, enumerate(varied)):
        only = [r for r in res if all(abs(r["params"].get(q, 0.0) - base_value(q, res, base)) < 1e-9
                                      for q in cols if q != k)]
        others = [r for r in res if r not in only]
        if others:
            ax.scatter([r["params"].get(k, 0.0) for r in others], [r["Cd"] for r in others], s=18, c="0.6", label="other designs")
        if only:
            only = sorted(only, key=lambda r: r["params"].get(k, 0.0))
            xs = [r["params"].get(k, 0.0) for r in only]
            ax.errorbar(xs, [r["Cd"] for r in only], yerr=[r["Cd_std"] for r in only], fmt="o-", color="tab:blue",
                        capsize=3, label="Cd (one-at-a-time)")
            ax2 = ax.twinx()
            ax2.plot(xs, [r["CdA"] for r in only], "s--", color="tab:orange", ms=4)
            ax2.set_ylabel("CdA [m²]", color="tab:orange", fontsize=8); ax2.tick_params(labelsize=7, colors="tab:orange")
        ax.axhline(base["Cd"], color="k", lw=.6, ls=":")
        title = k + (f"   fit: {slopes[j] * 1000:+.2f} counts/{unit(k, res)}" if slopes is not None else "")
        ax.set_title(title, fontsize=10); ax.set_xlabel(f"{k} [{unit(k, res)}]"); ax.set_ylabel("Cd"); ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc="best")
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"study '{study}': sensitivity of Cd / CdA to design variables")
    plt.tight_layout(); plt.savefig(os.path.join(out, "sensitivity.png"), dpi=100); plt.close()


def _cut(study, case, axis, value):
    f = os.path.join(run_dir(study), "cases", case, "constant", "geometry", "body.stl")   # the case's own copy
    if not os.path.exists(f):
        return np.zeros((0, 2, 3))
    V, F = read_stl(f)
    return plane_cut(V, F, axis, value)


def profiles_best(study, res, base, out, n=3):
    best = sorted(res, key=lambda r: r["CdA"])[:n]
    fig, ax = plt.subplots(2, 1, figsize=(14, 7.5))
    for r, c, lw in [(base, "0.6", 2.2)] + list(zip(best, ["tab:green", "tab:blue", "tab:purple"], [1.1] * 3)):
        lab = "baseline" if r is base else f"{r['case']}  Cd={r['Cd']:.3f}  CdA={r['CdA']:.3f}  ({r['label']})"
        ax[0].add_collection(LineCollection(_cut(study, r["case"], 1, 0.0)[:, :, [0, 2]], colors=c, linewidths=lw, label=lab))
        ax[1].add_collection(LineCollection(_cut(study, r["case"], 2, 0.55)[:, :, [0, 1]], colors=c, linewidths=lw))
    for a in ax:
        a.autoscale(); a.set_aspect("equal"); a.grid(alpha=.3)
    ax[0].legend(fontsize=8, loc="lower center"); ax[0].set_title("centre-line cut (y = 0): lowest-CdA designs")
    ax[1].set_title("plan cut (z = 0.55 m)")
    plt.tight_layout(); plt.savefig(os.path.join(out, "profiles_best.png"), dpi=100); plt.close()


def main(study):
    res = load_results(study)
    if not res:
        print("no finished cases yet")
        return
    path = write_csv(study, res)
    base = baseline_of(res) or res[0]
    out = os.path.join(run_dir(study), "report")
    os.makedirs(out, exist_ok=True)
    ranking(study, res, base, out)
    sensitivity(study, res, base, out)
    profiles_best(study, res, base, out)
    print(f"\n{'case':6s} {'Cd':>7s} {'ΔCd':>7s} {'CdA':>7s} {'ΔCdA%':>7s} {'Cl':>7s}  design")
    for r in sorted(res, key=lambda r: r["CdA"]):
        print(f"{r['case']:6s} {r['Cd']:7.4f} {(r['Cd'] - base['Cd']) * 1000:+6.1f}c {r['CdA']:7.4f} "
              f"{(r['CdA'] / base['CdA'] - 1) * 100:+6.2f}% {r['Cl']:7.3f}  {r['label']}")
    print(f"\nresults: {path}\nreport : {out}/")


if __name__ == "__main__":
    main(sys.argv[1])
