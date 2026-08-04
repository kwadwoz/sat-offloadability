#!/usr/bin/env python3
"""E20: retest G0's UP features against a denoised target.

E14 showed single-run instance PPD is mostly noise (ICC 0.06-0.31), but that
verdict is driven by var_rename and phases -- perturbations that do not occur
at deployment, where a CNF arrives with its variable numbering fixed and the
solver's phase policy is fixed. Under clause_shuffle alone, ICC stays high
(0.68-0.84 for the uf families).

So the prediction target G0 should have been aimed at is the *expected* PPD
over clause-order noise, not one noisy draw. This script

  1. rebuilds the target: per instance, mean PPD over the 12 clause_shuffle
     repeats in ppd_stability.csv (no new solver runs);
  2. states the new ceiling per family: sqrt(ICC_k) for the mean-of-k target
     via Spearman-Brown, vs E14's single-run sqrt(ICC);
  3. re-runs G0's correlation table (bridge_check.csv features up_mean/up_max)
     against old and new targets side by side, within family so log n is
     held constant by construction.

If the feature correlations move materially toward the new ceiling, G0's
negative was an artifact of target noise. If they stay flat, the features are
genuinely uninformative and the family-median story stands.

Outputs: model/target_redefinition.csv, model/target_redefinition_corr.csv
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
K = 12  # clause_shuffle repeats per instance in ppd_stability.csv


def icc1(groups: list[list[float]]) -> float:
    """One-way random-effects ICC(1): between-instance variance share."""
    k = len(groups[0])
    grand = mean(v for g in groups for v in g)
    n = len(groups)
    msb = k * sum((mean(g) - grand) ** 2 for g in groups) / (n - 1)
    msw = sum((v - mean(g)) ** 2 for g in groups for v in g) / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)


def main() -> None:
    # -- 1. rebuild the target from existing E14 repeats ---------------------
    shuffles: dict[str, list[float]] = defaultdict(list)
    fam_of: dict[str, str] = {}
    orig: dict[str, float] = {}
    with open(ROOT / "ppd_stability.csv") as fh:
        for r in csv.DictReader(fh):
            if r["mode"] != "clause_shuffle":
                continue
            shuffles[r["instance"]].append(float(r["ppd"]))
            fam_of[r["instance"]] = r["family"]
            orig[r["instance"]] = float(r["ppd_orig"])
    clean = {i: mean(v) for i, v in shuffles.items()}

    with open(ROOT / "target_redefinition.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "instance", "ppd_orig", "ppd_clean_mean",
                    "n_reps", "shuffle_cv"])
        for i, v in sorted(clean.items(), key=lambda kv: (fam_of[kv[0]], kv[0])):
            vals = shuffles[i]
            cv = (mean((x - v) ** 2 for x in vals) ** 0.5) / v if v else 0.0
            w.writerow([fam_of[i], i, orig[i], f"{v:.4f}", len(vals), f"{cv:.4f}"])

    # -- 2. ceilings: single-run vs mean-of-K target -------------------------
    print("== ceilings under clause_shuffle only (deployment-realistic noise)")
    print(f"{'family':<12} {'ICC1':>6} {'ceil_1run':>9} {'ICC_k':>6} {'ceil_meanK':>10}")
    fam_insts = defaultdict(list)
    for i in clean:
        fam_insts[fam_of[i]].append(i)
    for fam in sorted(fam_insts):
        groups = [shuffles[i] for i in fam_insts[fam]]
        r1 = icc1(groups)
        # Spearman-Brown: reliability of the mean of K repeats
        rk = K * r1 / (1 + (K - 1) * r1) if r1 > 0 else 0.0
        print(f"{fam:<12} {r1:6.3f} {max(r1,0)**.5:9.3f} {rk:6.3f} {rk**.5:10.3f}")

    # -- 3. G0 features vs old and new target, within family -----------------
    bridge = list(csv.DictReader(open(ROOT / "bridge_check.csv")))
    out = open(ROOT / "target_redefinition_corr.csv", "w", newline="")
    w = csv.writer(out)
    w.writerow(["depth_mode", "family", "n", "feature",
                "rho_vs_orig", "p_vs_orig", "rho_vs_clean", "p_vs_clean"])
    print("\n== G0 UP features: old single-run target vs denoised target")
    print(f"{'depth':<8} {'family':<12} {'n':>3} {'feature':<8} "
          f"{'rho_old':>8} {'rho_new':>8} {'delta':>7}")
    for depth in sorted({r["depth_mode"] for r in bridge}):
        by_fam = defaultdict(list)
        for r in bridge:
            if r["depth_mode"] == depth and r["instance"] in clean:
                by_fam[r["family"]].append(r)
        for fam in sorted(by_fam):
            rows = by_fam[fam]
            if len(rows) < 8:
                continue
            old_t = [orig[r["instance"]] for r in rows]
            new_t = [clean[r["instance"]] for r in rows]
            for feat in ("up_mean", "up_max"):
                x = [float(r[feat]) for r in rows]
                ro, po = spearmanr(x, old_t)
                rn, pn = spearmanr(x, new_t)
                w.writerow([depth, fam, len(rows), feat,
                            f"{ro:.4f}", f"{po:.4f}", f"{rn:.4f}", f"{pn:.4f}"])
                mark = " *" if pn < 0.05 else ""
                print(f"{depth:<8} {fam:<12} {len(rows):>3} {feat:<8} "
                      f"{ro:8.3f} {rn:8.3f} {rn - ro:+7.3f}{mark}")
    out.close()


if __name__ == "__main__":
    main()
