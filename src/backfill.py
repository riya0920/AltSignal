"""AltSignal - backfill detection. The highest-powered test in the harness uses no statistics.

When a vendor sells "10 years of history", they typically began COLLECTING only a few years
ago. The earlier portion was reconstructed afterwards, by people who already knew how the
period turned out - a memoir, not a diary. It flatters the vendor enormously, and it is the
single most common reason a real alternative-data backtest falls apart in production.

The test is almost embarrassingly simple:

    ask the vendor when they started collecting, split the history there, score both halves.

No bootstrap, no correction, no null distribution over strategies. The live-start date is not
secret - it is on the data sheet, and a vendor who will not disclose it has answered the
question a different way.

WHAT THIS IS AND IS NOT
This does not replace the significance tests. It answers a DIFFERENT question: "was this
history reconstructed?" rather than "is this edge real?" A vendor can have honest history and
no edge, or reconstructed history and a genuine edge - both exist in this synthetic data on
purpose, because one real vendor is given backfill too. Comparing this test's detection rate
against deflated Sharpe's would be comparing a metal detector to a thermometer. Report it as a
separate gate.

CALIBRATION
An honest vendor will not score identically in both halves either; randomness alone produces a
gap. So the threshold is measured rather than assumed: vendors that declared no backfill are
split at the same day and their gaps pooled, giving the distribution of drops that honest data
produces. Anything past the 95th percentile of that is flagged. No answer key is involved.

Run:  python -m src.backfill
      python -m src.backfill --sweep 50
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.generate import build
from src.naive import CONFIGS, smooth, top_k, weights

TRADING_DAYS = 252
MIN_HALF = 200          # days needed either side for a meaningful comparison
FLAG_PCTILE = 95        # honest-gap percentile above which a vendor is flagged


def ensemble_weights(sig):
    """Positions of the equal-weight ensemble book across all nine configs."""
    return np.mean([weights(top_k(smooth(sig, s), k)) for _, s, k in CONFIGS], axis=0)


def _sharpe(pnl):
    sd = pnl.std()
    return 0.0 if sd == 0 else float(pnl.mean() / sd * np.sqrt(TRADING_DAYS))


def split_sharpes(pnl, cut):
    """Sharpe before and after a cut point. Returns (pre, post, gap)."""
    pre, post = _sharpe(pnl[:cut]), _sharpe(pnl[cut:])
    return pre, post, pre - post


def analyse(returns, vendors, live_starts):
    """Score every vendor's pre/post gap and flag the ones that look reconstructed.

    Pure: no files, no printing, and the answer key's true_ic is never read - only live_start,
    which is disclosed information.
    """
    fwd = returns[1:]
    pnls = {n: (ensemble_weights(s) * fwd).sum(axis=1) for n, s in vendors.items()}

    declared = {n: ls for n, ls in live_starts.items() if ls > 0}
    clean = [n for n, ls in live_starts.items() if ls <= 0]

    # Calibration: split the vendors that declared NO backfill at each declared cut point and
    # pool the gaps. That is what an honest vendor's drop looks like at that point in history.
    honest_gaps = []
    for cut in set(declared.values()):
        if cut < MIN_HALF or len(fwd) - cut < MIN_HALF:
            continue
        for n in clean:
            honest_gaps.append(split_sharpes(pnls[n], cut)[2])
    honest_gaps = np.array(honest_gaps)
    threshold = float(np.percentile(honest_gaps, FLAG_PCTILE)) if len(honest_gaps) else np.inf

    rows = []
    for n, ls in live_starts.items():
        if ls <= 0:
            rows.append({"vendor": n, "live_start": ls, "pre": np.nan, "post": np.nan,
                         "gap": np.nan, "flagged": False, "testable": False})
            continue
        pre, post, gap = split_sharpes(pnls[n], ls)
        testable = ls >= MIN_HALF and len(fwd) - ls >= MIN_HALF
        rows.append({"vendor": n, "live_start": ls, "pre": pre, "post": post, "gap": gap,
                     "flagged": bool(testable and gap > threshold), "testable": testable})

    return pd.DataFrame(rows), threshold, honest_gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=0)
    args = ap.parse_args()

    if args.sweep:
        rows, t0 = [], time.time()
        for i, seed in enumerate(range(args.sweep), 1):
            u = build(seed)
            df, thr, _ = analyse(u["returns"], u["vendors"], u["live_starts"])
            df["true_ic"] = [u["true_ics"][int(v[-2:])] for v in df["vendor"]]
            bf = df[df["live_start"] > 0]
            rows.append({
                "seed": seed,
                "n_backfilled": len(bf),
                "caught": int(bf["flagged"].sum()),
                "caught_rate": float(bf["flagged"].mean()),
                "false_flags": int(df[df["live_start"] <= 0]["flagged"].sum()),
                "threshold": thr,
            })
            if i % 10 == 0 or i == args.sweep:
                r = (time.time() - t0) / i
                print(f"  {i}/{args.sweep} seeds ({r:.1f}s each)", flush=True)
        s = pd.DataFrame(rows)
        Path("results").mkdir(exist_ok=True)
        s.to_csv("results/backfill_sweep.csv", index=False)
        print(f"\nBACKFILL DETECTION   ({args.sweep} seeds)")
        print(f"  backfilled vendors caught:  {s['caught_rate'].mean():.1%}")
        print(f"  seeds catching all of them: {(s['caught'] == s['n_backfilled']).mean():.1%}")
        print(f"  false flags per seed:       {s['false_flags'].mean():.2f}"
              f"   (expected ~{(100 - FLAG_PCTILE) / 100:.2f} by construction)")
        return

    returns = np.load("data/returns.npy")
    vendors = dict(np.load("data/vendors.npz"))
    key = pd.read_csv("data/answer_key.csv")
    live_starts = dict(zip(key["vendor"], key["live_start"]))

    df, thr, gaps = analyse(returns, vendors, live_starts)
    df = df.merge(key[["vendor", "true_ic"]], on="vendor", how="left")   # DISPLAY ONLY
    df["kind"] = np.where(df["true_ic"] > 0, "REAL", "fake")

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/backfill.csv", index=False)

    shown = df[df["live_start"] > 0].sort_values("gap", ascending=False)
    pd.set_option("display.width", 150)
    print(f"BACKFILL TEST   (flag threshold {thr:.3f}, "
          f"the {FLAG_PCTILE}th percentile of {len(gaps)} honest gaps)")
    print("\nVENDORS DECLARING A LIVE-START DATE")
    print(shown[["vendor", "kind", "true_ic", "live_start", "pre", "post", "gap", "flagged"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    caught = int(shown["flagged"].sum())
    print(f"\n  caught {caught} of {len(shown)} reconstructed histories")
    print(f"  false flags among vendors declaring no backfill: "
          f"{int(df[df['live_start'] <= 0]['flagged'].sum())}")
    print("\n  NOTE: a flag means the HISTORY was reconstructed, not that the vendor is fake.")
    print("  One genuinely predictive vendor is backfilled here on purpose.")


if __name__ == "__main__":
    main()