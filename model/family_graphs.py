#!/usr/bin/env python3
"""Figure: one representative variable-incidence graph per E23 family.

Picks the smallest completed instance of each family (layout has to converge),
re-downloads it, builds the VIG, colors nodes by Louvain community, and draws
all 14 panels in one figure. The point is to make Q legible: high-Q families
should show visibly separated blobs, low-Q families one tangle. Each panel is
annotated with the instance's Q and its family's median ppd.

Communities are anonymous partitions, not named series: adjacent communities
just need distinct colors, so a small categorical set is reused per panel and
no legend is drawn. Edges are subsampled above EDGE_CAP for drawability.

Output: figures/family_graphs.png
"""
from __future__ import annotations

import csv
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import median

import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "instrument"))
from run_midrange import fetch  # noqa: E402
from modularity_industrial import read_cnf, MAX_CLAUSE  # noqa: E402

ROOT = Path(__file__).resolve().parent
EDGE_CAP = 12_000     # max edges drawn per panel
NODE_CAP = 4_000      # largest component is truncated beyond this
# categorical set from the skill's reference palette (light-surface steps);
# identity is per-panel and anonymous, so reuse across panels is fine
COLORS = ["#3d7fe0", "#e0653d", "#2fa377", "#8a63d2", "#d2a53f",
          "#c25793", "#5aa9c9", "#8c8c46"]


def build_vig(clauses):
    weights = {}
    for cl in clauses:
        vs = sorted({abs(l) for l in cl})
        if len(vs) < 2 or len(vs) > MAX_CLAUSE:
            continue
        for e in ((a, b) for i, a in enumerate(vs) for b in vs[i + 1:]):
            weights[e] = weights.get(e, 0.0) + 1.0
    ids = {v: i for i, v in enumerate({v for e in weights for v in e})}
    return ig.Graph(n=len(ids), edges=[(ids[a], ids[b]) for a, b in weights])


def main() -> None:
    rows = list(csv.DictReader(open(ROOT / "modularity_industrial.csv")))
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    reps = {f: min(rs, key=lambda r: int(r["n_clauses"])) for f, rs in by_fam.items()}
    med_ppd = {f: median(float(r["ppd"]) for r in rs) for f, rs in by_fam.items()}

    fams = sorted(reps, key=lambda f: -med_ppd[f])
    fig, axes = plt.subplots(4, 4, figsize=(17, 17), facecolor="white")
    rng = random.Random(0)

    with tempfile.TemporaryDirectory() as td:
        for ax, fam in zip(axes.flat, fams):
            r = reps[fam]
            cnf = fetch(f"https://benchmark-database.de/file/{r['hash']}",
                        Path(td), timeout=180.0, retries=4, backoff=5.0)
            g = build_vig(read_cnf(cnf))
            cnf.unlink()
            g = g.components().giant()
            if g.vcount() > NODE_CAP:
                g = g.subgraph(rng.sample(range(g.vcount()), NODE_CAP))
                g = g.components().giant()
            ig.set_random_number_generator(random.Random(0))
            memb = g.community_multilevel().membership
            xy = g.layout_fruchterman_reingold(niter=300)
            xs = [p[0] for p in xy]
            ys = [p[1] for p in xy]
            edges = g.get_edgelist()
            if len(edges) > EDGE_CAP:
                edges = rng.sample(edges, EDGE_CAP)
            segs = [((xs[a], ys[a]), (xs[b], ys[b])) for a, b in edges]
            ax.add_collection(LineCollection(segs, colors="#b8bcc2",
                                             linewidths=0.25, alpha=0.35, zorder=1))
            ax.scatter(xs, ys, s=3.5,
                       c=[COLORS[m % len(COLORS)] for m in memb],
                       linewidths=0, zorder=2)
            ax.set_title(fam, fontsize=12, color="#1f2328", pad=6)
            ax.text(0.02, 0.02,
                    f"Q={float(r['modularity']):.2f}   median ppd={med_ppd[fam]:.0f}",
                    transform=ax.transAxes, fontsize=9.5, color="#57606a")
            ax.set_aspect("equal")
            ax.axis("off")
            print(f"{fam}: {g.vcount()} nodes drawn", file=sys.stderr, flush=True)
    for ax in axes.flat[len(fams):]:
        ax.axis("off")
    axes.flat[len(fams)].text(
        0.0, 0.85,
        "Variable incidence graphs,\none representative instance per family.\n\n"
        "Node = variable; edge = shared clause;\ncolor = Louvain community.\n\n"
        "Q = this instance's modularity.\nppd = family median (MiniSat, E23).\n\n"
        "Panels sorted by median ppd.",
        fontsize=11, color="#1f2328", va="top")
    fig.tight_layout()
    out = ROOT.parent / "figures" / "family_graphs.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
