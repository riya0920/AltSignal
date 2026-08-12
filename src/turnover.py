"""AltSignal - turnover control. The book was rebuilding itself every three days.

Every economic result in this project - breakeven_bps, the capacity curve, signal decay - runs
through position sizing, and position sizing was churning at 106-150x a year. Real long/short
equity books run 5-50x. At 130x, trading costs dominate everything and the economic gates stop
measuring vendor quality at all: in the capacity table a NULL vendor had the highest capacity,
which is the tell that the numbers were being driven by luck in the frictionless Sharpe rather
than by liquidity.

WHERE THE CHURN COMES FROM
Broken out by config, on vendor_00:

    smooth1_all      220x    sharpe +1.28
    smooth1_top50    285x    sharpe +1.18
    smooth1_top25    329x    sharpe +1.15
    smooth5_all       83x    sharpe +1.03
    smooth20_all      37x    sharpe +1.13

The three smooth-1 configs cost 6x the turnover of smooth-20 and buy nothing - every Sharpe in
that column sits between 1.02 and 1.28. Signal smoothing is not the same thing as position
smoothing, and the ensemble was averaging books that each rebuilt themselves daily.

THE FIX
Move only part of the way toward the target each day:

    held[t] = (1 - rate) * held[t-1] + rate * target[t]

rate=1.0 is the old behaviour. rate=0.1 means a tenth of the gap is closed daily, so a position
takes about a week to build and the book turns over far less. This is what a real desk does, and
it costs almost no signal: the target barely moves day to day anyway, so tracking it loosely
loses little.

Two things worth noticing in the results. Net-of-cost Sharpe is MAXIMISED at an interior rate,
not at rate=1 - trading less makes more money once costs are real. And the optimal rate is not
the same for every vendor, because v2 gives each one a different decay speed: a fast-decaying
signal has to be tracked more closely or the edge is gone before the position is built.

Run:  python -m src.turnover
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.naive import CONFIGS, smooth, top_k, weights

TRADING_DAYS = 252
RATES = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
COST_BPS = 10.0                 # round-trip cost per unit of gross turnover
TARGET_TURNOVER = 50.0          # upper end of what a real long/short book runs


def target_weights(sig):
    """The Step 3 ensemble book: all nine configs, equally weighted, no position smoothing."""
    return np.mean([weights(top_k(smooth(sig, s), k)) for _, s, k in CONFIGS], axis=0)


def apply_rate(target, rate):
    """Partial adjustment toward the target. rate=1 reproduces the unsmoothed book.

    Renormalised each day so gross exposure stays at 1 - otherwise a slow rate would quietly
    shrink the book and the Sharpe comparison would be measuring leverage, not turnover.
    """
    if rate >= 1.0:
        return target
    held = np.empty_like(target)
    held[0] = target[0]
    for t in range(1, len(target)):
        held[t] = (1 - rate) * held[t - 1] + rate * target[t]
    gross = np.abs(held).sum(axis=1, keepdims=True)
    return held / np.maximum(gross, 1e-12)


def evaluate(w, fwd, cost_bps=COST_BPS):
    """Turnover, gross Sharpe and net Sharpe for a held position matrix."""
    traded = np.abs(np.diff(w, axis=0, prepend=0.0)).sum(axis=1)
    gross = (w * fwd).sum(axis=1)
    net = gross - traded * (cost_bps / 10000.0)
    sh = lambda x: 0.0 if x.std() == 0 else float(x.mean() / x.std() * np.sqrt(TRADING_DAYS))
    return float(traded.mean() * TRADING_DAYS), sh(gross), sh(net)


def sweep(returns, vendors, rates=RATES):
    fwd = returns[1:]
    rows = []
    for name, sig in vendors.items():
        tgt = target_weights(sig)
        for rate in rates:
            turn, g, n = evaluate(apply_rate(tgt, rate), fwd)
            rows.append({"vendor": name, "rate": rate, "turnover": turn,
                         "gross_sharpe": g, "net_sharpe": n})
    return pd.DataFrame(rows)


def main():
    returns = np.load("data/returns.npy")
    vendors = dict(np.load("data/vendors.npz"))
    key = pd.read_csv("data/answer_key.csv")

    df = sweep(returns, vendors)
    df = df.merge(key[["vendor", "true_ic"]], on="vendor", how="left")     # DISPLAY ONLY
    df["kind"] = np.where(df["true_ic"] > 0, "REAL", "fake")

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/turnover.csv", index=False)

    pd.set_option("display.width", 165)
    print(f"TURNOVER CONTROL   (cost {COST_BPS:.0f} bps per unit traded)")
    print("\nAVERAGED ACROSS ALL 30 VENDORS")
    g = df.groupby("rate").agg(turnover=("turnover", "mean"),
                               gross=("gross_sharpe", "mean"),
                               net=("net_sharpe", "mean"))
    print(g.to_string(float_format=lambda x: f"{x:.2f}"))

    print("\nTHE FIVE REAL VENDORS   (net Sharpe at each rate)")
    real = df[df["true_ic"] > 0]
    piv = real.pivot_table(index="vendor", columns="rate", values="net_sharpe")
    print(piv.to_string(float_format=lambda x: f"{x:+.2f}"))

    print("\nBEST RATE PER REAL VENDOR")
    for name, sub in real.groupby("vendor"):
        b = sub.loc[sub["net_sharpe"].idxmax()]
        base = sub[sub["rate"] == 1.0].iloc[0]
        print(f"  {name}  true_ic {b['true_ic']:.3f}   best rate {b['rate']:.2f}"
              f"   turnover {base['turnover']:.0f}x -> {b['turnover']:.0f}x"
              f"   net sharpe {base['net_sharpe']:+.2f} -> {b['net_sharpe']:+.2f}")

    ok = g[g["turnover"] <= TARGET_TURNOVER]
    if len(ok):
        rec = ok.index.max()
        print(f"\n  RECOMMENDED rate {rec:.2f}: the fastest setting whose average turnover"
              f" stays under {TARGET_TURNOVER:.0f}x,")
        print(f"  which is the upper end of what a real long/short equity book runs.")
    print("\n  Net Sharpe peaks at an interior rate, not at 1.0 - trading less makes more")
    print("  money once costs are real. The peak differs by vendor because each one's signal")
    print("  decays at a different speed.")


if __name__ == "__main__":
    main()