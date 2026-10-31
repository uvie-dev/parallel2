#!/usr/bin/env python3
"""plots.py - turns data/results.csv into figures (PNG+SVG+PDF), per-figure source CSVs and a summary table.

Usage:  python3 scripts/plots.py [--csv data/results.csv] [--fixed-grain 64] [--outdir plots]
Requires: pandas, matplotlib, numpy      (pip install pandas matplotlib numpy)
Editable output: every figure is saved as SVG (open in Inkscape/PowerPoint/Illustrator), and the exact
numbers behind each figure are saved in plots/data/<figure>.csv (open in Excel to re-plot).
Edit this file to restyle - all plotting code is here.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/results.csv")
ap.add_argument("--outdir", default="plots")
ap.add_argument("--fixed-grain", type=int, default=64, help="grain used for the 'fixed granularity' plots")
a = ap.parse_args()

os.makedirs(os.path.join(a.outdir, "data"), exist_ok=True)
raw = pd.read_csv(a.csv)
bad = (raw.checksum_ok == 0).sum()
if bad:
    print(f"WARNING: {bad} runs failed the checksum test - results of those runs are NOT trustworthy")

SCHED_STYLE = {"static": ("tab:red", "o"), "static1": ("tab:orange", "s"), "dynamic": ("tab:blue", "^"),
               "guided": ("tab:green", "v"), "task": ("tab:purple", "D")}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 120})

# ---------- aggregate repetitions: median + std over reps ----------
par = raw[raw.sched != "serial"].copy()
keys = ["workload", "size", "order", "cv_target", "cv_measured", "sched", "grain", "threads"]
agg = par.groupby(keys).agg(
    t_par_med=("t_par", "median"), t_par_mean=("t_par", "mean"), t_par_std=("t_par", "std"),
    t_serial=("t_serial_med", "first"), imbalance_pct=("imbalance_pct", "median"),
    idle_pct=("idle_pct", "median"), barrier_wait_pct=("barrier_wait_pct", "median"),
    sched_overhead_pct=("sched_overhead_pct", "median"), reps=("rep", "count")).reset_index()
agg["t_par_cv_pct"] = 100 * agg.t_par_std / agg.t_par_mean
agg["speedup"] = agg.t_serial / agg.t_par_med
agg["efficiency"] = agg.speedup / agg.threads
agg.to_csv(os.path.join(a.outdir, "data", "aggregated_all.csv"), index=False)

def save(fig, name, df=None):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(a.outdir, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    if df is not None:
        df.to_csv(os.path.join(a.outdir, "data", f"{name}.csv"), index=False)
    plt.close(fig)
    print("saved", name)

def focus(workload="synthetic", order="random"):
    d = agg[(agg.workload == workload) & (agg.order == order)]
    if d.empty:
        return d
    return d[d["size"] == d["size"].max()]

syn = focus()
if syn.empty:
    raise SystemExit("No synthetic/random data found - run scripts/run_all.sh first")
Pmax = syn.threads.max()
cvs = sorted(syn.cv_target.unique())

# ---------- Fig 1: workload variance vs execution time ----------
d = syn[syn.threads == Pmax]
best = d.loc[d.groupby(["cv_target", "sched"]).t_par_med.idxmin()]
fixed = d[d.grain == a.fixed_grain]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for s, (c, m) in SCHED_STYLE.items():
    b = best[best.sched == s].sort_values("cv_measured")
    f = fixed[fixed.sched == s].sort_values("cv_measured")
    if not b.empty:
        ax[0].errorbar(b.cv_measured, b.t_par_med, yerr=b.t_par_std, color=c, marker=m, label=s, capsize=2)
    if not f.empty:
        ax[1].errorbar(f.cv_measured, f.t_par_med, yerr=f.t_par_std, color=c, marker=m, label=s, capsize=2)
ideal = d.groupby("cv_measured").t_serial.first() / Pmax
ax[0].plot(ideal.index, ideal.values, "k--", lw=1, label="ideal T1/p")
ax[1].plot(ideal.index, ideal.values, "k--", lw=1, label="ideal T1/p")
ax[0].set_title(f"(a) best granularity per schedule (p={Pmax})"); ax[1].set_title(f"(b) fixed granularity g={a.fixed_grain} (p={Pmax})")
for x in ax: x.set_xlabel("workload variance: CV of task cost (σ/μ)"); x.set_ylabel("execution time (s)")
ax[0].legend(ncol=2, fontsize=7)
save(fig, "fig1_variance_vs_time", pd.concat([best.assign(panel="best_grain"), fixed.assign(panel="fixed_grain")]))

# ---------- Fig 2: load imbalance vs threads ----------
d = syn[(syn.grain == a.fixed_grain)]
fig, axs = plt.subplots(1, len(cvs), figsize=(2.6 * len(cvs), 3.2), sharey=True, squeeze=False)
for ax, cv in zip(axs[0], cvs):
    for s, (c, m) in SCHED_STYLE.items():
        x = d[(d.cv_target == cv) & (d.sched == s)].sort_values("threads")
        if not x.empty: ax.plot(x.threads, x.imbalance_pct, color=c, marker=m, label=s)
    ax.set_title(f"CV={cv:g}"); ax.set_xlabel("threads"); ax.set_xscale("log", base=2)
axs[0][0].set_ylabel(f"load imbalance λ (%) , g={a.fixed_grain}"); axs[0][0].legend(fontsize=7)
save(fig, "fig2_imbalance_vs_threads", d[["cv_target", "sched", "threads", "grain", "imbalance_pct", "idle_pct"]])

# ---------- Fig 3: granularity vs efficiency (main result) ----------
d = syn[syn.threads == Pmax]
fig, axs = plt.subplots(1, len(cvs), figsize=(2.6 * len(cvs), 3.2), sharey=True, squeeze=False)
opt_rows = []
for ax, cv in zip(axs[0], cvs):
    for s, (c, m) in SCHED_STYLE.items():
        x = d[(d.cv_target == cv) & (d.sched == s)].sort_values("grain")
        if x.empty: continue
        ax.plot(x.grain, x.efficiency, color=c, marker=m, label=s, ms=4)
        i = x.efficiency.idxmax(); ax.plot(x.loc[i, "grain"], x.loc[i, "efficiency"], "k*", ms=9, zorder=5)
        opt_rows.append(x.loc[i])
    ax.set_xscale("log", base=2); ax.set_title(f"CV={cv:g}"); ax.set_xlabel("granularity g (tasks/chunk)")
axs[0][0].set_ylabel(f"parallel efficiency (p={Pmax})"); axs[0][0].legend(fontsize=7)
save(fig, "fig3_granularity_vs_efficiency", d[["cv_target", "sched", "grain", "threads", "efficiency", "speedup", "t_par_med"]])
opt = pd.DataFrame(opt_rows)

# ---------- Fig 4: idle-time breakdown (barrier wait vs scheduling overhead) ----------
rows = [(cvs[0], "static"), (cvs[0], "dynamic"), (cvs[-1], "static"), (cvs[-1], "dynamic")]
fig, axs = plt.subplots(2, 2, figsize=(8, 5), sharey=True)
for ax, (cv, s) in zip(axs.flat, rows):
    x = d[(d.cv_target == cv) & (d.sched == s)].sort_values("grain")
    if x.empty: continue
    pos = np.arange(len(x))
    ax.bar(pos, x.barrier_wait_pct, label="barrier wait (imbalance tail)", color="tab:red")
    ax.bar(pos, x.sched_overhead_pct, bottom=x.barrier_wait_pct, label="scheduling overhead", color="tab:gray")
    ax.set_xticks(pos); ax.set_xticklabels(x.grain.astype(int), rotation=90, fontsize=7)
    ax.set_title(f"{s}, CV={cv:g}"); ax.set_xlabel("granularity g"); ax.set_ylabel("% of T_par per thread")
axs[0][0].legend(fontsize=7)
save(fig, "fig4_idle_breakdown", d[["cv_target", "sched", "grain", "barrier_wait_pct", "sched_overhead_pct", "idle_pct"]])

# ---------- Fig 5: scalability (speedup vs threads, best grain) ----------
fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
rows5 = []
for ax, s in zip(axs, ["static", "dynamic"]):
    for cv in cvs:
        x = syn[(syn.cv_target == cv) & (syn.sched == s)]
        if x.empty: continue
        bestg = x[x.threads == Pmax].sort_values("efficiency").iloc[-1].grain
        y = x[x.grain == bestg].sort_values("threads")
        ax.plot(y.threads, y.speedup, marker="o", label=f"CV={cv:g} (g*={int(bestg)})")
        rows5.append(y)
    ax.plot(sorted(syn.threads.unique()), sorted(syn.threads.unique()), "k--", lw=1, label="ideal")
    ax.set_title(f"{s}: speedup at best g"); ax.set_xlabel("threads"); ax.set_ylabel("speedup"); ax.legend(fontsize=7)
save(fig, "fig5_scalability", pd.concat(rows5) if rows5 else None)

# ---------- Fig 6: optimal granularity vs variance ----------
fig, ax = plt.subplots(figsize=(4.5, 3.3))
for s, (c, m) in SCHED_STYLE.items():
    o = opt[opt.sched == s].sort_values("cv_measured")
    if not o.empty: ax.plot(o.cv_measured, o.grain, color=c, marker=m, label=s)
ax.set_yscale("log", base=2); ax.set_xlabel("workload variance (CV)"); ax.set_ylabel("optimal granularity g* (tasks/chunk)")
ax.legend(fontsize=7)
save(fig, "fig6_optimal_grain_vs_variance", opt[["cv_target", "cv_measured", "sched", "grain", "efficiency", "t_par_med"]])

# ---------- Fig 7: real irregular kernels ----------
wl = [w for w in ("prime", "mandel") if not focus(w, "natural").empty]
if wl:
    fig, axs = plt.subplots(1, len(wl), figsize=(4.3 * len(wl), 3.3), sharey=True, squeeze=False)
    rows7 = []
    for ax, w in zip(axs[0], wl):
        x0 = focus(w, "natural"); x0 = x0[x0.threads == x0.threads.max()]
        for s, (c, m) in SCHED_STYLE.items():
            x = x0[x0.sched == s].sort_values("grain")
            if not x.empty: ax.plot(x.grain, x.efficiency, color=c, marker=m, label=s, ms=4); rows7.append(x)
        ax.set_xscale("log", base=2); ax.set_xlabel("granularity g")
        ax.set_title(f"{w} (CV={x0.cv_measured.iloc[0]:.2f}, n={int(x0['size'].iloc[0])})")
    axs[0][0].set_ylabel("efficiency"); axs[0][0].legend(fontsize=7)
    save(fig, "fig7_real_kernels_efficiency", pd.concat(rows7))

# ---------- Summary table: best configuration per workload/variance level ----------
tabs = []
for (w, o, sz, cv), g in agg[agg.threads == Pmax].groupby(["workload", "order", "size", "cv_target"]):
    for s, h in g.groupby("sched"):
        r = h.loc[h.efficiency.idxmax()]
        tabs.append(dict(workload=w, order=o, size=sz, cv_target=cv, cv_measured=round(r.cv_measured, 3), sched=s,
                         best_grain=int(r.grain), t_par_s=round(r.t_par_med, 4), t_par_std_s=round(r.t_par_std, 4),
                         speedup=round(r.speedup, 2), efficiency=round(r.efficiency, 3),
                         imbalance_pct=round(r.imbalance_pct, 1), idle_pct=round(r.idle_pct, 1)))
tab = pd.DataFrame(tabs)
tab.to_csv(os.path.join(a.outdir, "data", "summary_table.csv"), index=False)
with open(os.path.join(a.outdir, "data", "summary_table.md"), "w") as f:
    try:
        f.write(tab.to_markdown(index=False))
    except ImportError:  # 'tabulate' not installed
        f.write(tab.to_string(index=False))
print(f"summary table -> {a.outdir}/data/summary_table.csv  ({len(tab)} rows)")
