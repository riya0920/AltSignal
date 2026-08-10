"""AltSignal - Probability of Backtest Overfitting (PBO) via combinatorially symmetric
cross-validation. Bailey, Borwein, Lopez de Prado & Zhu (2014).

Everything measured so far leaned on the answer key. "A fake outranks a real vendor in 43% of
seeds" is only computable because true_ics exists. PBO asks nearly the same question with no
ground truth at all:

    when I pick the best-looking candidate on half the data, does it stay good on the other
    half, or does it fall to the bottom?

If the in-sample winner routinely lands below median out-of-sample, the selection procedure is
finding luck rather than skill - and you can establish that about a real vendor set, on real
data, where nobody knows the truth. That is the piece the rest of this repo cannot do.

METHOD
  1. Split T observations into S=16 contiguous blocks.
  2. Every choice of S/2 blocks is an in-sample set; its complement is out-of-sample.
     C(16,8) = 12870 splits, so the result is a distribution rather than one number.
  3. Per split: rank all N candidates in-sample, take the argmax, then find that same
     candidate's rank out-of-sample.
  4. omega = OOS rank of the IS winner, scaled to (0,1). lambda = logit(omega).
     PBO = fraction of splits where lambda <= 0, i.e. the winner landed below median.

  0%   selection always generalises
  50%  selection is a coin flip - worthless
  80%+ selection is actively anti-predictive

Note this deliberately ignores time order. Walk-forward asks "could I have traded this?";
PBO asks "does the RANKING survive fresh data?" - a narrower question that isolates the
selection step, which is exactly what Step 3 identified as the problem.

WHY TWO CANDIDATE SETS
  naive     108 columns: every vendor x config book. This is what Step 2 chose among.
  ensemble   12 columns: one equal-weight book per vendor. This is what Step 3 chose among.

Step 3 argued that removing config selection was the fix. PBO tests that claim directly, and
without the answer key: if the argument holds, the 108-column PBO should be far worse than
the 12-column PBO.

Run:  python -m src.pbo                 (seed 42, both candidate sets)
      python -m src.pbo --sweep 50      (distribution across seeds)
"""

import argparse
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.generate import build
from src.naive import CONFIGS, smooth, top_k, weights

S_BLOCKS = 16                     # must be even; C(16,8) = 12870 splits
TRADING_DAYS = 252


def candidate_books(returns, vendors, mode="ensemble"):
    """P&L matrix of shape (n_days, n_candidates), plus the candidate names.

    mode='naive'    -> 108 columns, one per vendor x config. What Step 2 selected among.
    mode='ensemble' ->  12 columns, one equal-weight book per vendor. What Step 3 selects
                        among. No config choice is made inside a column.
    """
    fwd = returns[1:]
    cols, names = [], []
    for name, sig in vendors.items():
        books = [(weights(top_k(smooth(sig, s), k)) * fwd).sum(axis=1)
                 for _, s, k in CONFIGS]
        if mode == "naive":
            for (label, _, _), b in zip(CONFIGS, books):
                cols.append(b)
                names.append(f"{name}|{label}")
        else:
            cols.append(np.mean(books, axis=0))
            names.append(name)
    return np.column_stack(cols), names


def _block_moments(M, s_blocks=S_BLOCKS):
    """Per-block count, sum and sum-of-squares for every candidate.

    The whole method rests on this: mean and variance over ANY union of blocks can be
    rebuilt from block-level sums, so 12870 splits cost 12870 cheap additions instead of
    12870 passes over the full P&L matrix. Without it this would be unusable on a laptop.
    """
    t = M.shape[0] // s_blocks * s_blocks         # drop the ragged tail
    parts = np.array_split(M[:t], s_blocks, axis=0)
    counts = np.array([p.shape[0] for p in parts], dtype=float)
    sums = np.array([p.sum(axis=0) for p in parts])
    sumsq = np.array([(p ** 2).sum(axis=0) for p in parts])
    return counts, sums, sumsq


def _sharpe_from_moments(counts, sums, sumsq, mask):
    """Annualised Sharpe of every candidate over the blocks selected by mask."""
    n = counts[mask].sum()
    tot = sums[mask].sum(axis=0)
    tot2 = sumsq[mask].sum(axis=0)
    mean = tot / n
    var = np.maximum(tot2 / n - mean ** 2, 1e-24)
    return mean / np.sqrt(var) * np.sqrt(TRADING_DAYS)


def cscv(M, s_blocks=S_BLOCKS):
    """Run combinatorially symmetric cross-validation. Returns a dict of diagnostics."""
    counts, sums, sumsq = _block_moments(M, s_blocks)
    n_cand = M.shape[1]

    lambdas, is_sr, oos_sr, winners = [], [], [], []
    for combo in combinations(range(s_blocks), s_blocks // 2):
        m_is = np.zeros(s_blocks, dtype=bool)
        m_is[list(combo)] = True

        sr_is = _sharpe_from_moments(counts, sums, sumsq, m_is)
        sr_oos = _sharpe_from_moments(counts, sums, sumsq, ~m_is)

        win = int(np.argmax(sr_is))
        # rank of the winner among OOS Sharpes: 1 = worst, n_cand = best
        rank = int((sr_oos <= sr_oos[win]).sum())
        omega = rank / (n_cand + 1.0)             # scaled into (0,1), never 0 or 1
        lambdas.append(np.log(omega / (1.0 - omega)))
        is_sr.append(sr_is[win])
        oos_sr.append(sr_oos[win])
        winners.append(win)

    lambdas = np.array(lambdas)
    is_sr, oos_sr = np.array(is_sr), np.array(oos_sr)

    # Performance degradation: regress the winner's OOS Sharpe on its IS Sharpe. A positive
    # slope means a better in-sample score really does buy a better out-of-sample one. A flat
    # or negative slope means the in-sample number carries no information about the future.
    slope, intercept = np.polyfit(is_sr, oos_sr, 1)

    return {
        "pbo": float((lambdas <= 0).mean()),
        "n_splits": len(lambdas),
        "n_candidates": n_cand,
        "median_lambda": float(np.median(lambdas)),
        "median_is_sharpe": float(np.median(is_sr)),
        "median_oos_sharpe": float(np.median(oos_sr)),
        "degradation": float(np.median(is_sr) - np.median(oos_sr)),
        "slope": float(slope),
        "intercept": float(intercept),
        "prob_oos_negative": float((oos_sr < 0).mean()),
        "winners": np.array(winners),
        "lambdas": lambdas,
    }


def report(returns, vendors, true_ics=None):
    rows = []
    for mode in ["naive", "ensemble"]:
        M, names = candidate_books(returns, vendors, mode)
        r = cscv(M)
        # answer key used ONLY for this display column, never inside cscv
        if true_ics is not None:
            fake_idx = {i for i, n in enumerate(names)
                        if true_ics[int(n.split("|")[0][-2:])] == 0}
            r["winner_was_fake"] = float(np.isin(r["winners"], list(fake_idx)).mean())
        r["mode"] = mode
        rows.append(r)
    return pd.DataFrame(rows)[[
        "mode", "n_candidates", "n_splits", "pbo", "median_is_sharpe",
        "median_oos_sharpe", "degradation", "slope", "prob_oos_negative"
    ] + (["winner_was_fake"] if true_ics is not None else [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=0)
    args = ap.parse_args()

    if args.sweep:
        rows = []
        t0 = time.time()
        for i, seed in enumerate(range(args.sweep), 1):
            u = build(seed)
            df = report(u["returns"], u["vendors"], u["true_ics"])
            df["seed"] = seed
            rows.append(df)
            if i % 5 == 0 or i == args.sweep:
                rate = (time.time() - t0) / i
                print(f"  {i}/{args.sweep} seeds   ({rate:.1f}s each, "
                      f"~{rate * (args.sweep - i) / 60:.1f} min left)", flush=True)
        allr = pd.concat(rows, ignore_index=True)
        Path("results").mkdir(exist_ok=True)
        allr.to_csv("results/pbo_sweep.csv", index=False)
        print(f"\nPBO ACROSS {args.sweep} SEEDS")
        g = allr.groupby("mode")
        print(g[["pbo", "degradation", "slope", "winner_was_fake"]]
              .agg(["mean", "std"]).to_string(float_format=lambda x: f"{x:.3f}"))
        return

    returns = np.load("data/returns.npy")
    vendors = dict(np.load("data/vendors.npz"))
    key = pd.read_csv("data/answer_key.csv")

    df = report(returns, vendors, key["true_ic"].tolist())
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/pbo.csv", index=False)

    pd.set_option("display.width", 170)
    print(f"PROBABILITY OF BACKTEST OVERFITTING   ({S_BLOCKS} blocks, "
          f"{df['n_splits'].iloc[0]} splits, no answer key inside the method)")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n  pbo               how often the in-sample winner landed BELOW MEDIAN out of"
          " sample. 0.5 = coin flip")
    print("  degradation       median in-sample Sharpe minus median out-of-sample Sharpe")
    print("  slope             OOS Sharpe regressed on IS Sharpe. >0 means a better"
          " in-sample score buys something")
    print("  prob_oos_negative how often the chosen candidate actually LOST money out of"
          " sample")
    print("  winner_was_fake   display only - how often the chosen candidate had zero true"
          " skill")


if __name__ == "__main__":
    main()