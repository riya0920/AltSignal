"""AltSignal - multi-seed robustness harness. Works for any scorer.

A single seed tells you which vendors got lucky in one draw. It is an anecdote, not a result.
"vendor_09 outranked a real vendor" is a fact about seed 42; the finding you can defend is
"a zero-skill vendor outranks a genuine one in X% of worlds."

This rebuilds the entire synthetic universe on each of N seeds, runs a chosen scorer on each,
and reports the distribution. It is parameterised by scorer so every guard added in Steps 3-5
is measured the same way and the rows of the final table are directly comparable. On one seed
you cannot tell a working guard from a lucky one.

Two metrics do the work:

  inversions   - how many (fake, real) pairs the ranking gets backwards. Needs no arbitrary
                 Sharpe cutoff, which is why it replaced "cleared 1.0".
  max_fake     - the highest score reached by a vendor with ZERO true skill. Collected across
                 seeds this measures the noise band directly.

Note the noise band is computed from vendors KNOWN to be fake, so it is an oracle: it is the
target a real detector aims at, not a detector itself. Steps 4-5 must estimate the same band
without the answer key.

Run:  python -m src.robustness --scorer naive --seeds 200
      python -m src.robustness --scorer walkforward --seeds 200
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.generate import build
from src.naive import score_all
from src.walkforward import score_all_wf
from src.ensemble import score_all_ens

SCORERS = {
    "naive": (score_all, "best_sharpe"),
    "walkforward": (score_all_wf, "wf_sharpe"),
    "ensemble": (score_all_ens, "ens_sharpe"),
    "fracpos": (score_all_ens, "frac_pos"),
}


def run_one(seed, scorer, score_col):
    """Build one universe, score it, return (per-vendor rows, per-seed summary)."""
    u = build(seed)
    df = scorer(u["returns"], u["vendors"]).copy()

    truth = dict(zip([f"vendor_{i:02d}" for i in range(len(u["true_ics"]))], u["true_ics"]))
    df["seed"] = seed
    df["true_ic"] = df["vendor"].map(truth)
    df["kind"] = np.where(df["true_ic"] > 0, "REAL", "fake")
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    fakes = df[df["true_ic"] == 0]
    reals = df[df["true_ic"] > 0]
    inversions = int((fakes[score_col].values[:, None] > reals[score_col].values).sum())

    # the cost of having looked more than once, under whichever scorer is running
    if "selection_cost" in df:
        gap = float(df["selection_cost"].mean())
    else:
        gap = float((df["best_sharpe"] - df["single_sharpe"]).mean())

    summary = {
        "seed": seed,
        "inversions": inversions,
        "any_inversion": inversions > 0,
        "max_fake": float(fakes[score_col].max()),
        "min_real": float(reals[score_col].min()),
        "top_is_fake": bool(df.iloc[0]["true_ic"] == 0),
        "best_fake_rank": int(fakes["rank"].min()),
        "weakest_real_rank": int(reals["rank"].max()),
        "mean_search_gap": gap,
        "max_score": float(df[score_col].max()),
    }
    return df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", choices=sorted(SCORERS), default="naive")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--tag", default=None, help="output filename suffix (defaults to scorer)")
    args = ap.parse_args()

    scorer, score_col = SCORERS[args.scorer]
    tag = args.tag or args.scorer

    detail, summaries = [], []
    t0 = time.time()
    for i, seed in enumerate(range(args.start, args.start + args.seeds), 1):
        df, s = run_one(seed, scorer, score_col)
        detail.append(df)
        summaries.append(s)
        if i % 10 == 0 or i == args.seeds:
            rate = (time.time() - t0) / i
            print(f"  {i}/{args.seeds} seeds   ({rate:.1f}s each, "
                  f"~{rate * (args.seeds - i) / 60:.1f} min left)", flush=True)

    detail = pd.concat(detail, ignore_index=True)
    summ = pd.DataFrame(summaries)

    out = Path("results")
    out.mkdir(exist_ok=True)
    detail.to_csv(out / f"robustness_{tag}_detail.csv", index=False)
    summ.to_csv(out / f"robustness_{tag}.csv", index=False)

    print(f"\nROBUSTNESS - scorer={args.scorer}   ({len(summ)} seeds, {time.time() - t0:.0f}s)")
    print("\nHOW OFTEN THE RANKING GETS IT WRONG")
    print(f"  seeds with >=1 fake outranking a real vendor:  {summ['any_inversion'].mean():6.1%}")
    print(f"  seeds where the TOP-RANKED vendor is fake:     {summ['top_is_fake'].mean():6.1%}")
    print(f"  mean inversions per seed:                      {summ['inversions'].mean():6.2f}"
          f"   (out of 35 possible fake-real pairs)")
    print(f"  best fake reached the top 4 in:                {(summ['best_fake_rank'] <= 4).mean():6.1%}"
          f" of seeds")

    print("\nTHE NOISE BAND  (score of a ZERO-skill vendor; oracle, uses the answer key)")
    q = summ["max_fake"].quantile([0.05, 0.5, 0.95])
    print(f"  median:              {q[0.5]:.2f}")
    print(f"  5th-95th percentile: {q[0.05]:.2f} to {q[0.95]:.2f}")
    print(f"  worst case seen:     {summ['max_fake'].max():.2f}")

    print("\nREAL VENDORS: is each one reliably findable?")
    band = summ["max_fake"].quantile(0.95)
    for ic, g in detail[detail["true_ic"] > 0].groupby("true_ic"):
        print(f"  true_ic {ic:5.3f}   median score {g[score_col].median():5.2f}"
              f"   clears the noise band in {(g[score_col] > band).mean():6.1%} of seeds")
    print("  -> BOTH columns matter. A guard that kills inversions by rejecting everything")
    print("     has not helped; watch the weak real vendor (0.010) as the cost side.")

    print("\nCOST OF SELECTION")
    print(f"  mean gap: {summ['mean_search_gap'].mean():.2f}")
    print(f"  max score seen anywhere: {summ['max_score'].max():.2f}"
          f"   (a value > 5 signals a breadth/leakage bug)")


if __name__ == "__main__":
    main()