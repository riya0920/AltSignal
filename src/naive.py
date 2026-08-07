"""AltSignal Step 2 - the naive backtest. Deliberately wrong, on purpose.

Every protection is omitted here. There is no train/test split, no correction for the fact
that twelve vendors were tested, and no penalty for the nine configurations tried per vendor.
The point is to establish the baseline: how badly does a rushed, honest-looking evaluation
mislead you? Steps 3-5 are measured against this row.

Nothing is fitted. There is no model anywhere in this file - a vendor's numbers are turned
into a long/short book by arithmetic, the book is multiplied by what actually happened, and
the result is scored. The overfitting comes entirely from SELECTION: running that arithmetic
nine ways and keeping the best. Selection alone is enough to turn a money-losing null vendor
into an impressive-looking one.

score_all() is pure - arrays in, dataframe out, no files, no printing - so src/robustness.py
can call it inside a loop over seeds. main() is the thin wrapper that loads data/, writes
results/step2_naive.csv, and prints the scorecard.

Run:  python -m src.naive      (from the repo root, after python -m src.generate)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# --- the configuration grid: 3 smoothing windows x 3 concentration levels = 9 trials ---
# Smoothing exploits factor momentum; concentration exploits conviction. Neither is
# unreasonable on its own, which is exactly why searching over them is so easy to justify
# to yourself and so damaging in practice.
SMOOTH_WINDOWS = (1, 5, 20)
TOP_K = (None, 50, 25)          # None = hold all names; 50 = 50 long + 50 short; etc.

CONFIGS = [
    (f"smooth{s}_" + ("all" if k is None else f"top{k}"), s, k)
    for s in SMOOTH_WINDOWS
    for k in TOP_K
]

# The one configuration a disciplined analyst would have committed to in advance: no
# smoothing, hold everything. Its score is reported as single_sharpe, and the gap between
# it and best_sharpe is the measurable cost of having looked nine times.
DEFAULT_CONFIG = "smooth1_all"


def smooth(sig, n):
    """Trailing n-day mean of the signal. The first n rows are left raw (no history yet)."""
    if n == 1:
        return sig
    c = np.cumsum(sig, axis=0)
    out = np.empty_like(sig)
    out[:n] = sig[:n]
    out[n:] = (c[n:] - c[:-n]) / n
    return out


def top_k(sig, k):
    """Keep only the k most-liked and k most-disliked names, at equal weight.

    Discards the vendor's conviction in the middle of the book. Concentration raises both
    the return and the variance, so whether it helps is a matter of luck on any given draw -
    which is precisely why it makes a useful knob for the search to abuse.
    """
    if k is None:
        return sig
    out = np.zeros_like(sig)
    idx = np.argsort(sig, axis=1)
    rows = np.arange(sig.shape[0])[:, None]
    out[rows, idx[:, -k:]] = 1.0
    out[rows, idx[:, :k]] = -1.0
    return out


def weights(sig):
    """Dollar-neutral long/short book with gross exposure 1.

    Demeaning each day removes the market factor, so the book bets on the CROSS-SECTIONAL
    ranking rather than on the market going up. Note this weights by conviction: being right
    about your large positions matters more than being right often.
    """
    w = sig - sig.mean(axis=1, keepdims=True)
    return w / np.abs(w).sum(axis=1, keepdims=True)


def sharpe(sig, fwd):
    """Annualised Sharpe of the book implied by sig, applied to next-day returns fwd.

    Row t of sig is aligned to fwd[t], which is the return on day t+1. Getting this
    off by one is look-ahead bias and silently inflates everything downstream.
    """
    pnl = (weights(sig) * fwd).sum(axis=1)
    sd = pnl.std()
    if sd == 0:
        return 0.0
    return pnl.mean() / sd * np.sqrt(252)


def score_vendor(sig, fwd):
    """Score one vendor across all nine configs. Returns (single, best, best_label)."""
    scores = {}
    for label, s, k in CONFIGS:
        scores[label] = sharpe(top_k(smooth(sig, s), k), fwd)
    best_label = max(scores, key=scores.get)
    return scores[DEFAULT_CONFIG], scores[best_label], best_label


def score_all(returns, vendors):
    """Run the naive backtest over every vendor. Pure: no files, no printing.

    returns : (n_days, n_assets)
    vendors : dict of name -> (n_days-1, n_assets)

    Returns a dataframe sorted by best_sharpe descending. The answer key is NOT touched
    here; callers join true_ic in afterwards, for display only.
    """
    fwd = returns[1:]
    rows = []
    for name, sig in vendors.items():
        single, best, label = score_vendor(sig, fwd)
        rows.append({
            "vendor": name,
            "single_sharpe": single,
            "best_sharpe": best,
            "best_config": label,
            "n_configs_tried": len(CONFIGS),
        })
    return (pd.DataFrame(rows)
            .sort_values("best_sharpe", ascending=False)
            .reset_index(drop=True))


def main():
    returns = np.load("data/returns.npy")
    vendors = dict(np.load("data/vendors.npz"))
    key = pd.read_csv("data/answer_key.csv")

    df = score_all(returns, vendors)

    # answer key joined for DISPLAY ONLY, after all scoring is complete
    df = df.merge(key, on="vendor", how="left")
    df["kind"] = np.where(df["true_ic"] > 0, "REAL", "fake")

    out = Path("results")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "step2_naive.csv", index=False)

    pd.set_option("display.width", 140)
    print("STEP 2 NAIVE BACKTEST  (sorted by best_sharpe)")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    fakes = df[df["true_ic"] == 0]
    reals = df[df["true_ic"] > 0]
    # An inversion is the finding that matters: a vendor with ZERO true skill scoring above
    # a vendor with real skill. Unlike a Sharpe threshold, this needs no arbitrary cutoff.
    inversions = int((fakes["best_sharpe"].values[:, None] > reals["best_sharpe"].values).sum())

    print("\nSUMMARY")
    print(f"  top-ranked vendor overall:   {df.iloc[0]['vendor']} -> {df.iloc[0]['kind']}"
          f"  (best_sharpe {df.iloc[0]['best_sharpe']:.2f})")
    print(f"  max best_sharpe anywhere:    {df['best_sharpe'].max():.2f}"
          f"   (a value > 5 signals a breadth/leakage bug)")
    print(f"  best fake:                   {fakes.iloc[0]['vendor']}"
          f"  ({fakes.iloc[0]['best_sharpe']:.2f}, single {fakes.iloc[0]['single_sharpe']:.2f})")
    print(f"  fake-over-real inversions:   {inversions}"
          f"   (zero-skill vendors outranking genuine ones)")
    print(f"  mean best-minus-single gap:  {(df['best_sharpe'] - df['single_sharpe']).mean():.2f}"
          f"   (the cost of knob-twisting)")
    print("\n  NOTE: this is one seed. See src/robustness.py for the distribution.")


if __name__ == "__main__":
    main()