"""AltSignal - change-point detection. Finding the backfill seam without being told where it is.

src/backfill.py trusts the vendor. It takes the declared live-start date, splits there, and
compares the two halves. That matches stage 3 of how a real quant fund evaluates a dataset -
asking about structural changes, panel changes, and backfills in the history - but it stops
there, and a vendor who shades the date defeats it entirely.

Practitioners do not stop there. At the backtest stage they build a model portfolio from the
dataset and read its track record looking for the point where the nature of the data changed,
describing the problem explicitly as change-point detection, and reporting that quality
deterioration is sometimes locatable to within a month. The declared date is a hypothesis to be
checked, not evidence.

This module does that. It scans the P&L series for the single split that best separates two
regimes, calibrates a threshold from a bootstrap null in which no break exists, and then
compares what it found against what the vendor claimed.

THE STATISTIC
For every candidate split t, the standardised difference in mean daily P&L between the two
segments, pooled variance:

    stat(t) = (mean_before - mean_after) / (pooled_sd * sqrt(1/n1 + 1/n2))

The test statistic is max over t. Taking the maximum is itself a multiple-testing problem - a
series with no break at all will still produce some largest split - which is why the threshold
comes from a bootstrap null rather than a t-table. The stationary bootstrap is used so that
autocorrelation in daily P&L survives resampling; an iid shuffle would make the null too tight
and every vendor would look broken.

Directional on purpose: only a DROP is flagged. Reconstructed history is better than live
history, never worse, so a mid-sample improvement is a different phenomenon and not this one.

WHAT IS SCORED
  detection    how often a backfilled vendor is flagged
  localisation how far the detected break sits from the true one, in days
  agreement    whether the detected break corroborates the declared date
  false alarms how often a clean vendor is flagged

The last one matters most. A detector that fires on everything localises nothing.

Run:  python -m src.changepoint
      python -m src.changepoint --sweep 20
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.generate import build
from src.naive import CONFIGS, smooth, top_k, weights
from src.turnover import apply_rate

MIN_SEG = 250           # days required either side; below this the statistic is unstable
N_BOOT = 300            # bootstrap replications for the null
MEAN_BLOCK = 20         # stationary bootstrap mean block length, in days
ALPHA = 0.05
TRADE_RATE = 0.20
AGREE_DAYS = 60         # detected and declared count as agreeing within this many days
TRADING_DAYS = 252


def book(sig):
    tgt = np.mean([weights(top_k(smooth(sig, s), k)) for _, s, k in CONFIGS], axis=0)
    return np.nan_to_num(apply_rate(tgt, TRADE_RATE), nan=0.0)


def scan(pnl, min_seg=MIN_SEG):
    """Standardised two-sample statistic at every candidate split. Returns (stats, offset).

    Vectorised through cumulative sums: means and variances for every split come from prefix
    sums, so the whole scan is a handful of array operations rather than a Python loop over
    a thousand candidate cut points.
    """
    n = len(pnl)
    if n < 2 * min_seg + 1:
        return np.array([]), min_seg

    c1 = np.cumsum(pnl)
    c2 = np.cumsum(pnl ** 2)
    t = np.arange(min_seg, n - min_seg)

    n1 = t.astype(float)
    n2 = float(n) - n1
    s1, s2 = c1[t - 1], c1[-1] - c1[t - 1]
    q1, q2 = c2[t - 1], c2[-1] - c2[t - 1]

    m1, m2 = s1 / n1, s2 / n2
    v1 = np.maximum(q1 / n1 - m1 ** 2, 1e-24)
    v2 = np.maximum(q2 / n2 - m2 ** 2, 1e-24)
    pooled = np.sqrt((n1 * v1 + n2 * v2) / (n1 + n2))
    return (m1 - m2) / (pooled * np.sqrt(1 / n1 + 1 / n2)), min_seg


def stationary_bootstrap(pnl, n_boot, mean_block, rng):
    """Resampled series preserving autocorrelation, under the null of NO break."""
    n = len(pnl)
    p = 1.0 / mean_block
    out = np.empty((n_boot, n))
    starts = rng.integers(0, n, size=(n_boot, n))
    restart = rng.random((n_boot, n)) < p
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        cur = starts[b, 0]
        for i in range(n):
            if i > 0:
                cur = starts[b, i] if restart[b, i] else (cur + 1) % n
            idx[i] = cur
        out[b] = pnl[idx]
    return out


def detect(pnl, n_boot=N_BOOT, alpha=ALPHA, seed=0):
    """Locate the most likely downward break and test it against a no-break null."""
    stats, off = scan(pnl)
    if len(stats) == 0:
        return {"break_day": -1, "stat": 0.0, "threshold": np.inf, "flagged": False}

    k = int(np.argmax(stats))
    obs = float(stats[k])

    rng = np.random.default_rng(seed)
    boot = stationary_bootstrap(pnl, n_boot, MEAN_BLOCK, rng)
    null_max = np.array([scan(b)[0].max() if len(scan(b)[0]) else 0.0 for b in boot])
    thresh = float(np.quantile(null_max, 1 - alpha))

    return {"break_day": int(k + off), "stat": obs, "threshold": thresh,
            "flagged": bool(obs > thresh)}


def analyse(u, n_boot=N_BOOT):
    fwd = u["returns"][1:]
    rows = []
    for i, (name, sig) in enumerate(u["vendors"].items()):
        pnl = (book(sig) * fwd).sum(axis=1)
        d = detect(pnl, n_boot, seed=i)
        true_ls = int(u["live_starts"][name])
        d.update({
            "vendor": name,
            "true_live_start": true_ls,
            "is_backfilled": true_ls > 0,
            "true_ic": u["true_ics"][i],
            "error_days": abs(d["break_day"] - true_ls) if true_ls > 0 else np.nan,
            "agrees": (abs(d["break_day"] - true_ls) <= AGREE_DAYS) if true_ls > 0 else np.nan,
        })
        rows.append(d)
    return pd.DataFrame(rows)


def summarise(df):
    bf = df[df["is_backfilled"]]
    clean = df[~df["is_backfilled"]]
    hits = bf[bf["flagged"]]
    return {
        "detection": float(bf["flagged"].mean()),
        "false_alarm": float(clean["flagged"].mean()),
        "median_error_days": float(hits["error_days"].median()) if len(hits) else np.nan,
        "agree_rate": float(hits["agrees"].mean()) if len(hits) else np.nan,
        "n_backfilled": len(bf),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=0)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    if args.sweep:
        rows, t0 = [], time.time()
        for i, seed in enumerate(range(args.sweep), 1):
            s = summarise(analyse(build(seed), args.boot))
            s["seed"] = seed
            rows.append(s)
            if i % 5 == 0 or i == args.sweep:
                print(f"  {i}/{args.sweep} seeds ({(time.time()-t0)/i:.1f}s each)", flush=True)
        d = pd.DataFrame(rows)
        Path("results").mkdir(exist_ok=True)
        d.to_csv("results/changepoint_sweep.csv", index=False)
        print(f"\nCHANGE-POINT DETECTION   ({args.sweep} seeds, no declared date used)")
        print(f"  backfilled vendors flagged:     {d['detection'].mean():.1%}")
        print(f"  clean vendors falsely flagged:  {d['false_alarm'].mean():.1%}"
              f"   (nominal {ALPHA:.0%})")
        print(f"  median localisation error:      {d['median_error_days'].mean():.0f} days")
        print(f"  detected break within {AGREE_DAYS}d of the")
        print(f"    declared date:                {d['agree_rate'].mean():.1%}")
        print("\n  Compare with src/backfill.py, which is TOLD the date and catches ~73%.")
        return

    u = build()
    df = analyse(u, args.boot)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/changepoint.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"CHANGE-POINT DETECTION   (seed 42, {args.boot} bootstraps, "
          f"min segment {MIN_SEG}d, no declared date used)")
    show = df[df["is_backfilled"]].sort_values("stat", ascending=False)
    print("\nVENDORS THAT ACTUALLY HAVE RECONSTRUCTED HISTORY")
    print(show[["vendor", "true_ic", "true_live_start", "break_day", "error_days",
                "stat", "threshold", "flagged", "agrees"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    clean = df[~df["is_backfilled"]]
    s = summarise(df)
    print(f"\n  detected {int(show['flagged'].sum())} of {len(show)}"
          f"   median error {s['median_error_days']:.0f} days"
          f"   agree with declared date {s['agree_rate']:.0%}")
    print(f"  false alarms among {len(clean)} clean vendors: "
          f"{int(clean['flagged'].sum())}")
    print("\n  A detected break that does NOT match the declared date is the interesting")
    print("  case: either the vendor understated the reconstruction, or something else")
    print("  changed in the data that nobody mentioned.")


if __name__ == "__main__":
    main()