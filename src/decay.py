"""AltSignal - signal decay and transaction costs. Why statistically real vendors still fail.

Every IC in this repo so far has meant one thing: today's signal against TOMORROW's return.
One day ahead. But a signal does not switch off at midnight, and how long it persists decides
what it is worth.

    fast decay   the edge is gone in two days, so the book must be rebuilt constantly and
                 every rebuild is paid for in spread, impact and borrow
    slow decay   the edge lasts a month, so positions can be held and the cost of running the
                 strategy collapses

Two vendors with identical 1-day IC can differ by an order of magnitude in value. Nothing
else in this project can see that, because Sharpe on a frictionless book is blind to it.

This module adds the two missing views:

  ic_decay      IC at horizons 1..60 days, and the HALF-LIFE - the number of days for the
                edge to fall to half its day-1 value.
  cost_curve    Sharpe after transaction costs, swept across cost levels. Turnover is
                measured from the actual position changes, so a fast-decaying signal is
                penalised exactly as hard as it deserves.

The breakeven cost is the headline number: the level at which a vendor stops making money.
A vendor whose breakeven sits below realistic trading costs is worthless no matter how
significant its Sharpe was in Step 4. This is the most common real-world reason an
alternative-data vendor gets rejected, and it is not a statistical objection at all.

Run:  python -m src.decay
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.naive import CONFIGS, smooth, top_k, weights

HORIZONS = [1, 2, 3, 5, 10, 20, 40, 60]
COST_BPS = [0, 1, 2, 5, 10, 20, 50]        # per unit of gross turnover, one way
TRADING_DAYS = 252


def _std_rows(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


def ic_decay(sig, returns, horizons=HORIZONS):
    """Cross-sectional IC of the signal against the return h days ahead, for each h.

    Row t of sig is what the vendor said at the end of day t, and z[t+1] is the next day's
    return - so horizon 1 reproduces the IC used everywhere else in the repo. Larger h just
    walks the target further forward, leaving the signal where it is.
    """
    z = _std_rows(returns)
    v = _std_rows(sig)
    n_assets = sig.shape[1]
    out = {}
    for h in horizons:
        a = v[:len(v) - h + 1] if h > 1 else v
        b = z[h:h + len(a)]
        m = min(len(a), len(b))
        out[h] = float((a[:m] * b[:m]).sum(axis=1).mean() / n_assets)
    return out


def half_life(decay, horizons=HORIZONS):
    """Days for the IC to fall to half its day-1 value, by log-linear fit.

    Fitted only on horizons where IC is still positive - once it crosses zero the log is
    undefined and the points are noise anyway. Returns nan when the signal is too weak or
    too noisy to fit, which is itself informative.
    """
    xs = [h for h in horizons if decay[h] > 0]
    if len(xs) < 3 or decay[horizons[0]] <= 0:
        return float("nan")
    ys = np.log([decay[h] for h in xs])
    slope, _ = np.polyfit(xs, ys, 1)
    return float(np.log(0.5) / slope) if slope < 0 else float("inf")


def ensemble_weights(sig):
    """Positions of the Step 3 ensemble book: the nine config books, equally weighted."""
    return np.mean([weights(top_k(smooth(sig, s), k)) for _, s, k in CONFIGS], axis=0)


def cost_curve(sig, fwd, costs=COST_BPS):
    """Turnover, and net Sharpe at each cost level.

    Turnover is the gross change in positions from one day to the next - the amount actually
    traded. A signal that flips its book daily racks this up fast, which is precisely the
    economic penalty that fast decay implies and that a frictionless Sharpe hides.
    """
    w = ensemble_weights(sig)
    gross = np.abs(np.diff(w, axis=0)).sum(axis=1)
    turnover = float(gross.mean())                 # per day, gross exposure is 1

    pnl = (w * fwd).sum(axis=1)
    out = {"turnover": turnover, "ann_turnover": turnover * TRADING_DAYS}
    for c in costs:
        charge = np.concatenate([[0.0], gross]) * (c / 10000.0)
        net = pnl - charge[:len(pnl)]
        sd = net.std()
        out[f"sharpe_{c}bps"] = 0.0 if sd == 0 else float(net.mean() / sd * np.sqrt(TRADING_DAYS))
    return out


def breakeven_bps(sig, fwd, hi=200.0):
    """The cost level at which the vendor stops making money, by bisection.

    This is the number to quote. It converts a statistical result into a procurement
    decision: if realistic all-in trading costs exceed it, the vendor is unbuyable however
    significant its Sharpe was.
    """
    w = ensemble_weights(sig)
    gross = np.concatenate([[0.0], np.abs(np.diff(w, axis=0)).sum(axis=1)])
    pnl = (w * fwd).sum(axis=1)

    def net_mean(c):
        return (pnl - gross[:len(pnl)] * (c / 10000.0)).mean()

    if net_mean(0.0) <= 0:
        return 0.0
    if net_mean(hi) > 0:
        return float(hi)
    lo = 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if net_mean(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def report(returns, vendors):
    fwd = returns[1:]
    rows = []
    for name, sig in vendors.items():
        d = ic_decay(sig, returns)
        r = {"vendor": name,
             "ic_1d": d[1], "ic_5d": d[5], "ic_20d": d[20], "ic_60d": d[60],
             "half_life": half_life(d)}
        r.update(cost_curve(sig, fwd))
        r["breakeven_bps"] = breakeven_bps(sig, fwd)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    returns = np.load("data/returns.npy")
    vendors = dict(np.load("data/vendors.npz"))
    key = pd.read_csv("data/answer_key.csv")

    df = report(returns, vendors)
    df = df.merge(key, on="vendor", how="left")      # DISPLAY ONLY
    df["kind"] = np.where(df["true_ic"] > 0, "REAL", "fake")
    df = df.sort_values("true_ic", ascending=False)

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/decay.csv", index=False)

    pd.set_option("display.width", 190)
    print("SIGNAL DECAY   (IC at horizon h, cross-sectional)")
    print(df[["vendor", "kind", "true_ic", "ic_1d", "ic_5d", "ic_20d", "ic_60d", "half_life"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nTRANSACTION COSTS   (net Sharpe of the ensemble book)")
    cols = ["vendor", "kind", "ann_turnover"] + [f"sharpe_{c}bps" for c in COST_BPS] \
        + ["breakeven_bps"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n  half_life      days for the edge to fall to half its day-1 value")
    print("  ann_turnover   times the book is fully rebuilt per year")
    print("  breakeven_bps  cost level at which the vendor stops making money."
          " Realistic all-in")
    print("                 costs for a liquid US equity book are roughly 5-15 bps one way.")


if __name__ == "__main__":
    main()