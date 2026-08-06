"""AltSignal Step 2 - the deliberately wrong naive backtest.

This is the sloppy baseline. Every protection is omitted on purpose so we can record
how many pure-noise vendors survive. It is SUPPOSED to produce wrong answers. The four
deliberate mistakes, none of which we fix here:

  1. Fit and evaluate on the same full 1500-day sample. No train/test split.
  2. Search a grid of configs per vendor and keep only the best-scoring one.
  3. Report the maximum across twelve vendors with no multiple-testing correction.
  4. Report raw in-sample Sharpe with no deflation.

answer_key.csv is joined in for display only AFTER all scoring is complete, never
used in the decision logic.

Run:  python -m src.naive      (from the repo root, after src.generate)
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

returns = np.load(DATA / "returns.npy")                 # (1500, 400)
vendors_npz = np.load(DATA / "vendors.npz")
vendors = {k: vendors_npz[k] for k in vendors_npz.files}

# config grid: 3 smoothing windows x 3 concentrations = 9 configs per vendor
SMOOTH_WINDOWS = [1, 5, 20]
CONCENTRATIONS = [("all", None), ("top50", 50), ("top25", 25)]


def smooth(sig, window):
    """Trailing mean over `window` days (window=1 is a no-op)."""
    if window == 1:
        return sig
    return pd.DataFrame(sig).rolling(window, min_periods=1).mean().to_numpy()


def concentrate(w, k):
    """Keep only the k largest longs and k largest shorts per day, else keep all."""
    if k is None:
        return w
    out = np.zeros_like(w)
    for t in range(w.shape[0]):
        row = w[t]
        longs = np.argsort(row)[-k:]      # k most positive
        shorts = np.argsort(row)[:k]      # k most negative
        keep = np.concatenate([longs, shorts])
        out[t, keep] = row[keep]
    return out


def sharpe_for_config(sig, window, k):
    """Build a dollar-neutral long/short book from a vendor signal and return its
    annualised in-sample Sharpe. Vendor row t is applied to returns[t+1]."""
    s = smooth(sig, window)

    # 2.1 dollar-neutral weights: demean per day (kills market exposure), gross = 1
    w = s - s.mean(axis=1, keepdims=True)
    w = concentrate(w, k)
    gross = np.abs(w).sum(axis=1, keepdims=True)
    gross[gross == 0] = 1.0
    w = w / gross

    # 2.2 P&L: vendor row t predicts return t+1. Assert alignment - an off-by-one
    # here is look-ahead bias and silently inflates everything.
    fwd = returns[1:]                       # (1499, 400), aligned to vendor rows
    assert w.shape == fwd.shape, f"shape mismatch {w.shape} vs {fwd.shape}"
    pnl = (w * fwd).sum(axis=1)
    sd = pnl.std()
    if sd == 0:
        return 0.0
    return pnl.mean() / sd * np.sqrt(252)


rows = []
for name, sig in vendors.items():
    single = sharpe_for_config(sig, window=1, k=None)   # default: smoothing 1, all names
    best_sharpe, best_config = single, "smooth1_all"
    for window in SMOOTH_WINDOWS:
        for cname, k in CONCENTRATIONS:
            sh = sharpe_for_config(sig, window, k)
            if sh > best_sharpe:
                best_sharpe, best_config = sh, f"smooth{window}_{cname}"
    rows.append(
        {
            "vendor": name,
            "single_sharpe": round(single, 4),
            "best_sharpe": round(best_sharpe, 4),
            "best_config": best_config,
            "n_configs_tried": len(SMOOTH_WINDOWS) * len(CONCENTRATIONS),
        }
    )

table = pd.DataFrame(rows).sort_values("best_sharpe", ascending=False).reset_index(drop=True)
table.to_csv(RESULTS / "step2_naive.csv", index=False)

# --- join the answer key for DISPLAY ONLY, after all scoring is done ---
key = pd.read_csv(DATA / "answer_key.csv")
display = table.merge(key, on="vendor", how="left")
display["kind"] = np.where(display["true_ic"] > 0, "REAL", "fake")

print("STEP 2 NAIVE BACKTEST  (sorted by best_sharpe)")
print(display.to_string(index=False))

fakes = display[display["true_ic"] == 0]
reals = display[display["true_ic"] > 0]
top = display.iloc[0]

print("\nSUMMARY")
print(f"  zero-IC vendors clearing Sharpe 1.0 (best-of-9):  {(fakes['best_sharpe'] > 1.0).sum()} of {len(fakes)}")
print(f"  zero-IC vendors clearing Sharpe 0.5 (best-of-9):  {(fakes['best_sharpe'] > 0.5).sum()} of {len(fakes)}")
print(f"  top-ranked vendor overall:  {top['vendor']}  ->  {top['kind']}  (best_sharpe {top['best_sharpe']:.2f})")
print(f"  max best_sharpe anywhere:   {display['best_sharpe'].max():.2f}   (a value > 5 signals a breadth/leakage bug)")
mean_gap = (display['best_sharpe'] - display['single_sharpe']).mean()
print(f"  mean best-minus-single gap: {mean_gap:.2f}   (the cost of knob-twisting)")
