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

cfg.add_stage_paths()
import design_table as dt  # noqa: E402
import rbf  # noqa: E402
from drag_morph import read_stl  # noqa: E402

RESULT_KEYS = ["Cd", "Cd_std", "CdA", "Cl", "Cl_front", "Cl_rear", "drag_N", "frontal_area", "cells", "iterations"]


def run_dir(study):
    return os.path.join(cfg.CFD_RUNS, study)


def load_results(study):
    out = []
    for r in dt.read_designs(study):
        f = os.path.join(run_dir(study), "cases", r["case"], "result.json")
        if os.path.exists(f):
            res = json.load(open(f))
            res["params"] = r["params"]
            out.append(res)
    return out


def write_csv(study, res):
    cols = dt.columns([{"params": r["params"]} for r in res])
    path = os.path.join(run_dir(study), "results.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study", "case", "label"] + cols + RESULT_KEYS)
        for r in res:
            w.writerow([study, r["case"], dt.label(r["params"])] + [r["params"].get(k, 0.0) for k in cols]
                       + [r.get(k) for k in RESULT_KEYS])
    return path


def baseline_of(res):
    return next((r for r in res if dt.label(r["params"]) == "baseline"), None)


def unit(k):
    return "amp" if k.startswith(dt.MODE_PREFIX) else dt.geo.MORPH_PARAMS[k][3]


def base_value(k):
    return 0.0 if k.startswith(dt.MODE_PREFIX) else dt.BASE[k]


def ranking(study, res, base, out):
    res = sorted(res, key=lambda r: r["CdA"])
    names = [f"{r['case']}  {dt.label(r['params'])}" for r in res]
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
    cols = dt.columns([{"params": r["params"]} for r in res])
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
        only = [r for r in res if all(abs(r["params"].get(q, 0.0) - base_value(q)) < 1e-9 for q in cols if q != k)]
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
        title = k + (f"   fit: {slopes[j] * 1000:+.2f} counts/{unit(k)}" if slopes is not None else "")
        ax.set_title(title, fontsize=10); ax.set_xlabel(f"{k} [{unit(k)}]"); ax.set_ylabel("Cd"); ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc="best")
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"study '{study}': sensitivity of Cd / CdA to design variables")
    plt.tight_layout(); plt.savefig(os.path.join(out, "sensitivity.png"), dpi=100); plt.close()


def _cut(study, case, axis, value):
    f = os.path.join(dt.geometry_dir(study, case), "body.stl")
    if not os.path.exists(f):
        return np.zeros((0, 2, 3))
    V, F = read_stl(f)
    return rbf.plane_cut(V, F, axis, value)


def profiles_best(study, res, base, out, n=3):
    best = sorted(res, key=lambda r: r["CdA"])[:n]
    fig, ax = plt.subplots(2, 1, figsize=(14, 7.5))
    for r, c, lw in [(base, "0.6", 2.2)] + list(zip(best, ["tab:green", "tab:blue", "tab:purple"], [1.1] * 3)):
        lab = "baseline" if r is base else f"{r['case']}  Cd={r['Cd']:.3f}  CdA={r['CdA']:.3f}  ({dt.label(r['params'])})"
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
              f"{(r['CdA'] / base['CdA'] - 1) * 100:+6.2f}% {r['Cl']:7.3f}  {dt.label(r['params'])}")
    print(f"\nresults: {path}\nreport : {out}/")


if __name__ == "__main__":
    main(sys.argv[1])
