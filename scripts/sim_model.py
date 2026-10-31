#!/usr/bin/env python3
"""sim_model.py - discrete-event model of chunked scheduling (NOT measured data).
Predicts how the optimal granularity g* moves with task-cost variance (CV), for static / dynamic / guided.
Model: N tasks with lognormal cost (mean 1.0, given CV); chunks of g consecutive tasks; each chunk dispatch
costs h (scheduling overhead, in units of mean task cost); dynamic = greedy earliest-free-thread assignment.
Usage: python3 scripts/sim_model.py [--p 16] [--n 4000] [--h 0.2] [--order random|sorted] [--outdir plots/model]
Use it to (1) form hypotheses before running real experiments, (2) compare against measured curves."""
import argparse, heapq, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=int, default=16); ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--h", type=float, default=0.2); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--order", default="random", choices=["random","sorted"])
ap.add_argument("--outdir", default="plots/model")
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
rng = np.random.default_rng(a.seed)

def costs(cv):
    if cv == 0: return np.ones(a.n)
    s2 = np.log(1 + cv * cv); x = rng.lognormal(-s2 / 2, np.sqrt(s2), a.n)
    return np.sort(x) if a.order == 'sorted' else x

def makespan(c, g, kind, p, h):
    ch = np.add.reduceat(c, np.arange(0, len(c), g)) + h      # chunk cost incl. dispatch overhead
    if kind == "static":                                      # contiguous blocks of chunks per thread
        return max(blk.sum() for blk in np.array_split(ch, p))
    if kind == "dynamic":
        free = [0.0] * p; heapq.heapify(free)
        for x in ch:
            t = heapq.heappop(free); heapq.heappush(free, t + x)
        return max(free)
    if kind == "guided":                                      # chunk = max(g, remaining/p) tasks
        free = [0.0] * p; heapq.heapify(free); i = 0; n = len(c)
        while i < n:
            size = max(g, (n - i) // p); size = min(size, n - i)
            t = heapq.heappop(free); heapq.heappush(free, t + c[i:i + size].sum() + h); i += size
        return max(free)

rows = []
grains = [2 ** k for k in range(0, 13)]
for cv in [0, 0.25, 1.0, 2.0]:
    c = costs(cv); t1 = c.sum()
    for kind in ["static", "dynamic", "guided"]:
        for g in grains:
            tp = makespan(c, g, kind, a.p, a.h)
            rows.append(dict(cv=cv, sched=kind, grain=g, t_par=tp, speedup=t1 / tp, efficiency=t1 / tp / a.p))
df = pd.DataFrame(rows); df.to_csv(os.path.join(a.outdir, "model_results.csv"), index=False)

fig, axs = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
col = {"static": "tab:red", "dynamic": "tab:blue", "guided": "tab:green"}
for ax, cv in zip(axs, [0, 0.25, 1.0, 2.0]):
    for k in col:
        x = df[(df.cv == cv) & (df.sched == k)]
        ax.plot(x.grain, x.efficiency, marker="o", ms=3, color=col[k], label=k)
        i = x.efficiency.idxmax(); ax.plot(x.loc[i, "grain"], x.loc[i, "efficiency"], "k*", ms=9)
    ax.set_xscale("log", base=2); ax.set_title(f"CV={cv:g}"); ax.set_xlabel("granularity g"); ax.grid(alpha=.3)
axs[0].set_ylabel(f"efficiency (model, p={a.p})"); axs[0].legend(fontsize=7)
fig.savefig(os.path.join(a.outdir, "model_granularity_vs_efficiency.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(a.outdir, "model_granularity_vs_efficiency.svg"), bbox_inches="tight")
best = df.loc[df.groupby(["cv", "sched"]).efficiency.idxmax()][["cv", "sched", "grain", "efficiency"]]
print(best.to_string(index=False))
