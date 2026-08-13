"""AltSignal - what the decision rule costs. Sweeping gate 2 instead of assuming it.

The full pipeline buys 1.53 vendors at 77.2% precision and 22.7% recall. Precision is fine;
recall is not. Four of five real vendors are missed every time, and on seed 42 one of the
rejected vendors earned +1.31 on sealed data - better than the vendor that was bought.

The cause is not the ranking. On seed 42 the three real vendors held the three lowest p-values
out of thirty. The cause is the CUTOFF: Benjamini-Hochberg is a step-up procedure, so it stops
at the first p-value that fails its threshold and rejects everything behind it. At thirty
candidates the rank-2 threshold is 2/30 x q, and one lucky null landing there blocks every
genuine vendor below it. Raising q does not help, because the null's p-value scales with the
threshold.

So this sweeps the decision rule and reports the whole precision/recall curve, rather than
picking one operating point and hoping. Four families:

  bh        Benjamini-Hochberg at various q          - the current rule
  fixed     a flat p-value threshold                 - no step-up, so no blocking
  topk      buy the k lowest p-values                - fixes the shortlist size
  fixed_be  flat threshold plus a breakeven floor    - does economics bind once more get through

The economics gate is included because the pipeline showed it doing nothing: the count after
gate 3 exactly equalled the count after gate 2 in every seed. Turnover control pushed breakeven
so high that everything surviving significance clears 15bps easily. Whether that stays true
when the significance gate lets three times as many through is the open question.

Run:  python -m src.gate_sweep --seeds 20
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.generate import build
from src.pipeline import run, benjamini_hochberg

RULES = (
    [("bh", q) for q in (0.10, 0.20, 0.40)]
    + [("fixed", p) for p in (0.01, 0.025, 0.05, 0.10)]
    + [("topk", k) for k in (2, 3, 5)]
    + [("fixed_be", p) for p in (0.05, 0.10)]
)
BE_FLOOR = 30.0          # breakeven floor for the fixed_be family, in bps


def decide(df, family, param):
    """Boolean buy vector under one decision rule. Reads no ground truth."""
    p = df["p_value"].values
    if family == "bh":
        keep = benjamini_hochberg(p, param)
    elif family == "fixed":
        keep = p < param
    elif family == "topk":
        keep = np.zeros(len(p), dtype=bool)
        keep[np.argsort(p)[:param]] = True
    elif family == "fixed_be":
        keep = (p < param) & (df["breakeven_bps"].values >= BE_FLOOR)
    else:
        raise ValueError(family)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    # Each universe is scored under every rule, so the comparison is paired and differences
    # are not confounded by which seeds happened to be easy.
    rows, t0 = [], time.time()
    for i, seed in enumerate(range(args.seeds), 1):
        u = build(seed)
        df = run(u["returns"], u["vendors"], u["live_starts"])
        truth = np.array([u["true_ics"][int(v[-2:])] for v in df["vendor"]])
        n_real = int((truth > 0).sum())

        for family, param in RULES:
            keep = decide(df, family, param)
            hold = df["holdout_sharpe"].values
            rows.append({
                "rule": f"{family}_{param}",
                "family": family,
                "seed": seed,
                "n_bought": int(keep.sum()),
                "precision": float((truth[keep] > 0).mean()) if keep.any() else np.nan,
                "recall": float((truth[keep] > 0).sum() / n_real),
                "n_fake_bought": int(((truth == 0) & keep).sum()),
                "holdout_bought": float(hold[keep].mean()) if keep.any() else np.nan,
                "holdout_rejected": float(hold[~keep].mean()) if (~keep).any() else np.nan,
            })
        if i % 5 == 0 or i == args.seeds:
            print(f"  {i}/{args.seeds} seeds ({(time.time() - t0) / i:.1f}s each)", flush=True)

    d = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    d.to_csv("results/gate_sweep.csv", index=False)

    g = (d.groupby(["family", "rule"], sort=False)
           .agg(bought=("n_bought", "mean"), precision=("precision", "mean"),
                recall=("recall", "mean"), fakes=("n_fake_bought", "mean"),
                hold_buy=("holdout_bought", "mean"), hold_rej=("holdout_rejected", "mean"))
           .reset_index())
    # F1 balances the two error directions; a rule that buys nothing scores perfectly on
    # precision alone, which is why precision is never reported on its own here.
    g["f1"] = 2 * g["precision"] * g["recall"] / (g["precision"] + g["recall"])
    g["hold_gap"] = g["hold_buy"] - g["hold_rej"]

    pd.set_option("display.width", 165)
    print(f"\nDECISION RULE SWEEP   ({args.seeds} seeds, 30 vendors, 5 real, paired)")
    print(g.drop(columns="family").to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best = g.loc[g["f1"].idxmax()]
    cur = g[g["rule"] == "bh_0.1"].iloc[0]
    print(f"\n  current rule  bh_0.1     bought {cur['bought']:.2f}"
          f"   precision {cur['precision']:.1%}   recall {cur['recall']:.1%}"
          f"   F1 {cur['f1']:.3f}")
    print(f"  best F1       {best['rule']:<10} bought {best['bought']:.2f}"
          f"   precision {best['precision']:.1%}   recall {best['recall']:.1%}"
          f"   F1 {best['f1']:.3f}")

    print("\n  hold_gap is the only column a real desk could observe without an answer key.")
    print("  If it stays positive as recall rises, the extra vendors are genuinely earning.")
    fb = g[g["family"] == "fixed_be"]
    fx = g[(g["family"] == "fixed") & (g["rule"].isin(["fixed_0.05", "fixed_0.1"]))]
    if len(fb) and len(fx):
        print(f"\n  economics gate: fixed rules buy {fx['bought'].mean():.2f} vendors,"
              f" the same rules plus a {BE_FLOOR:.0f}bps floor buy {fb['bought'].mean():.2f}."
              " If those match, it is still non-binding.")


if __name__ == "__main__":
    main()