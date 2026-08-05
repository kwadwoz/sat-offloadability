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
from collections import Counter, defaultdict
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
TAIL = "#c9ced4"   # communities beyond the 8 largest
# families whose within-family Q->ppd slope survived preregistered
# out-of-sample replication (E25/E26), with the replicated rho
VALIDATED = {"bitvector": -0.64, "prime-factoring": +0.74,
             "planning": +0.63, "scheduling": +0.58, "miter": +0.46}
GOOD = "#1a7f4b"   # status-good ink for the validated tag


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

    # only the five replication-validated families; the rest are described
    # in prose in the paper. Single row, no legend panel -- everything else
    # belongs in the caption.
    fams = sorted((f for f in reps if f in VALIDATED), key=lambda f: -med_ppd[f])
    # smaller canvas at the same aspect: marks and text are sized in points,
    # so a narrower figure survives the shrink to \textwidth with the strokes
    # still readable
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.3), facecolor="white")
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
            # colour only the len(COLORS) largest communities and grey the
            # tail: cycling the palette would make two unrelated communities
            # share a colour, which a reader would misread as one cluster
            sizes = Counter(memb)
            top = {c: i for i, (c, _) in enumerate(sizes.most_common(len(COLORS)))}
            node_c = [COLORS[top[m]] if m in top else TAIL for m in memb]
            n_comm = len(sizes)
            xy = g.layout_fruchterman_reingold(niter=1500)
            # scale each layout uniformly into the same unit square: with
            # equal aspect, differing extents would give the panels different
            # box heights and stagger the titles
            xs = [p[0] for p in xy]
            ys = [p[1] for p in xy]
            cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
            half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 or 1.0
            xs = [(x - cx) / half for x in xs]
            ys = [(y - cy) / half for y in ys]
            edges = g.get_edgelist()
            if len(edges) > EDGE_CAP:
                edges = rng.sample(edges, EDGE_CAP)
            segs = [((xs[a], ys[a]), (xs[b], ys[b])) for a, b in edges]
            ax.add_collection(LineCollection(segs, colors="#8b929a",
                                             linewidths=0.6, alpha=0.6, zorder=1))
            ax.scatter(xs, ys, s=14.0, c=node_c, edgecolors="white",
                       linewidths=0.35, zorder=2, rasterized=True)
            ax.set_title(fam, fontsize=14, color="#1f2328", pad=7)
            ax.text(0.5, -0.045,
                    f"$Q$={float(r['modularity']):.2f}    "
                    f"median ppd={med_ppd[fam]:.0f}    "
                    f"$\\rho$={VALIDATED[fam]:+.2f}",
                    transform=ax.transAxes, fontsize=10.5, color="#57606a",
                    ha="center", va="top")
            ax.set_xlim(-1.09, 1.09)
            ax.set_ylim(-1.09, 1.09)
            ax.set_aspect("equal")
            ax.axis("off")
            print(f"{fam}: {g.vcount()} nodes drawn", file=sys.stderr, flush=True)
    fig.tight_layout()
    fig.text(0.5, -0.02,
             "Colour marks the 8 largest Louvain communities per instance; "
             "all smaller communities are grey. Communities are numbered "
             "per instance, so colours are not comparable across panels.",
             ha="center", va="top", fontsize=11, color="#57606a")
    out = ROOT.parent / "figures" / "family_graphs.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)", file=sys.stderr)


if __name__ == "__main__":
    main()
