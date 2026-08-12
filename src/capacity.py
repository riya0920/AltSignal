"""AltSignal - capacity. Every backtest so far assumed infinite liquidity.

Positions in this project have been free to take. Any stock absorbs any size at the closing
price. Real books cannot: buying moves the price against you, and the more you buy relative to
what trades that day, the worse it gets. This is why a signal can be genuinely predictive,
survive every statistical gate, and still be worth nothing to a fund of any size.

The standard model is the square-root law, which holds remarkably well empirically:

    cost (in return terms) = K * daily_vol * sqrt( traded / average daily volume )

Two consequences that matter more than the formula:

  Cost is not linear in size. Doubling the fund does not double the cost per dollar - it
  multiplies it by sqrt(2). So gross P&L scales with AUM while cost scales with AUM^1.5, and
  every strategy has a size at which the second overtakes the first.

  Turnover is punished twice. A fast-decaying signal both trades more often AND pays more per
  trade at a given fund size. The v2 generator gives each vendor its own decay rate, so this
  module finally has something to discriminate on.

CAPACITY is reported as the AUM at which net Sharpe falls to half its frictionless value. That
is a more useful number than the point where it hits zero, because no desk runs a strategy to
the edge of its own viability.

Run:  python -m src.capacity
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.turnover import apply_rate

from src.naive import CONFIGS, smooth, top_k, weights

# Fund sizes to sweep, in millions of dollars.
AUM_GRID = np.array([10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000], dtype=float)

IMPACT_K = 0.5          # square-root law constant; empirical estimates cluster near 0.3-1.0
ADV_MEDIAN = 50.0       # median stock's average daily volume, in $m
ADV_SIGMA = 1.2         # lognormal spread - real volume is enormously dispersed
TRADING_DAYS = 252


def ensemble_weights(sig):
    tgt = np.mean([weights(top_k(smooth(sig, s), k)) for _, s, k in CONFIGS], axis=0)
    return apply_rate(tgt, 0.20)


def _sharpe(pnl):
    sd = pnl.std()
    return 0.0 if sd == 0 else float(pnl.mean() / sd * np.sqrt(TRADING_DAYS))


def adv_profile(n_assets, seed=0):
    """Average daily volume per stock, in $m. Lognormal: a few huge names, a long thin tail."""
    rng = np.random.default_rng(seed)
    return ADV_MEDIAN * np.exp(ADV_SIGMA * rng.standard_normal(n_assets))


def net_pnl(sig, fwd, adv, aum, vol):
    """Daily P&L net of square-root impact, at a given fund size.

    Impact is charged on the CHANGE in position, not the position itself - you pay to trade,
    not to hold. Everything is in return units, so it subtracts directly from gross P&L.
    """
    w = ensemble_weights(sig)
    gross = (w * fwd).sum(axis=1)

    traded = np.abs(np.diff(w, axis=0, prepend=0.0)) * aum          # $m traded per name per day
    participation = traded / adv[None, :]
    cost_ret = IMPACT_K * vol[None, :] * np.sqrt(participation)     # return cost per $ traded
    cost = (cost_ret * np.abs(np.diff(w, axis=0, prepend=0.0))).sum(axis=1)
    return gross - cost


def capacity_curve(sig, fwd, adv, vol):
    frictionless = _sharpe((ensemble_weights(sig) * fwd).sum(axis=1))
    sharpes = np.array([_sharpe(net_pnl(sig, fwd, adv, a, vol)) for a in AUM_GRID])

    # capacity = AUM where net Sharpe first falls below half the frictionless value
    target = frictionless / 2.0
    cap = np.nan
    if frictionless > 0:
        below = np.where(sharpes < target)[0]
        cap = float(AUM_GRID[below[0]]) if len(below) else float(AUM_GRID[-1])
    return frictionless, sharpes, cap


def report(returns, vendors, seed=0):
    fwd = returns[1:]
    adv = adv_profile(returns.shape[1], seed)
    vol = returns.std(axis=0)                       # per-stock daily volatility

    rows = []
    for name, sig in vendors.items():
        fl, sh, cap = capacity_curve(sig, fwd, adv, vol)
        w = ensemble_weights(sig)
        rows.append({
            "vendor": name,
            "ann_turnover": float(np.abs(np.diff(w, axis=0)).sum(axis=1).mean() * TRADING_DAYS),
            "sharpe_0": fl,
            **{f"sharpe_{int(a)}m": s for a, s in zip(AUM_GRID, sh)},
            "capacity_musd": cap,
        })
    return pd.DataFrame(rows)


def main():
    returns = np.load("data/returns.npy")
    vendors = dict(np.load("data/vendors.npz"))
    key = pd.read_csv("data/answer_key.csv")

    df = report(returns, vendors)
    df = df.merge(key[["vendor", "true_ic"]], on="vendor", how="left")   # DISPLAY ONLY
    df["kind"] = np.where(df["true_ic"] > 0, "REAL", "fake")
    df = df.sort_values("true_ic", ascending=False)

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/capacity.csv", index=False)

    pd.set_option("display.width", 175)
    print(f"CAPACITY   (square-root impact, K={IMPACT_K}, median ADV ${ADV_MEDIAN:.0f}m)")
    cols = ["vendor", "kind", "true_ic", "ann_turnover", "sharpe_0",
            "sharpe_100m", "sharpe_500m", "sharpe_2500m", "capacity_musd"]
    print(df[cols].head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n  sharpe_0        frictionless, what every earlier module reported")
    print("  sharpe_Nm       net of impact at a fund of $N million")
    print("  capacity_musd   fund size at which net Sharpe falls to HALF the frictionless value")
    print("\n  A vendor whose capacity sits below your AUM is unbuyable no matter how")
    print("  significant it looked. Note this is a HARDER test than breakeven_bps, which")
    print("  assumes a fixed cost per trade rather than one that grows with your size.")


if __name__ == "__main__":
    main()