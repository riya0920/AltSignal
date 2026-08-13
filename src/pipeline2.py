"""AltSignal - the pipeline with a switchable evidence base. Returns vs fundamentals.

The decision-rule sweep showed twelve different cutoff rules all sitting on ONE
precision/recall curve. Tightening bought precision and lost recall; loosening did the reverse;
nothing beat anything. That is the signature of a gate that has run out of evidence, not one
with a badly chosen threshold - every rule was reading the same return-based p-values, and
those p-values are weak at IC 0.02 by construction.

So this runs the identical pipeline twice on the identical worlds, changing only what gate 2
looks at:

    returns       Newey-West p-value on the ensemble book. 1500 noisy daily observations of a
                  signal buried under market and idiosyncratic noise.
    fundamentals  accuracy against reported revenue, AND incremental value over consensus.
                  9600 clean firm-quarter observations of the thing the vendor claims to
                  measure.

Both gates see only the research window; the holdout is sliced off before either runs, and the
fundamentals gate is restricted to quarters that end before the cut. Gate 1 (provenance) runs
first in both arms and discards reconstructed history, so neither arm gets to use it.

WHY THE FUNDAMENTALS GATE NEEDS TWO CONDITIONS
Accuracy alone is not enough and the world is built to prove it. The mirror vendors track
consensus almost perfectly - they are the most ACCURATE vendors in the shortlist and they are
worth nothing, because everything they know is already in the price. A gate that filters on
accuracy alone buys them. The incremental test - how much the nowcast adds to a regression that
already contains consensus - is what removes them.

Run:  python -m src.pipeline2                (seed 42, both arms)
      python -m src.pipeline2 --sweep 20     (paired comparison across worlds)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.world import build, QUARTER_LEN, N_QUARTERS
from src.naive import CONFIGS, smooth, top_k, weights
from src.turnover import apply_rate

HOLDOUT_DAYS = 250
TRADE_RATE = 0.20
COST_BPS = 10.0
# Lower than the 30bps used on the return-driven world. A quarterly signal holds a view for
# only part of each quarter and sits flat the rest of the time, so its book trades less AND
# earns over fewer days; breakeven is not comparable across the two worlds.
MIN_BREAKEVEN = 5.0
P_THRESHOLD = 0.10          # returns arm, from the gate sweep
MIN_ACCURACY = 0.010        # fundamentals arm: R2 against reported revenue
MIN_INCREMENTAL = 0.002     # fundamentals arm: R2 added over consensus
NW_LAGS = 5
TRADING_DAYS = 252


def book(sig):
    """Turnover-controlled ensemble positions.

    Days after the final announcement carry no view at all - the signal rows are exactly zero,
    and normalising them divides by zero, which propagates NaN through every downstream Sharpe.
    Flat days are a real feature of a quarterly signal, not an error, so they are held flat.
    """
    tgt = np.mean([weights(top_k(smooth(sig, s), k)) for _, s, k in CONFIGS], axis=0)
    return np.nan_to_num(apply_rate(tgt, TRADE_RATE), nan=0.0)


def _sharpe(p):
    return 0.0 if p.std() == 0 else float(p.mean() / p.std() * np.sqrt(TRADING_DAYS))


def _nw_pvalue(pnl, lags=NW_LAGS):
    x = pnl - pnl.mean()
    n = len(x)
    tot = (x @ x) / n
    for L in range(1, lags + 1):
        tot += 2.0 * (1.0 - L / (lags + 1.0)) * (x[L:] @ x[:-L]) / n
    se = np.sqrt(max(tot, 1e-18) / n)
    return float(1.0 - stats.norm.cdf(pnl.mean() / se)) if se > 0 else 1.0


def _fit_r2(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1.0 - (y - X @ b).var() / y.var()


def incremental_r2(g, c, n):
    """R2 the nowcast ADDS to a regression already containing consensus.

    The obvious version - correlating (actual - consensus) with (nowcast - consensus) - is
    wrong and flatters junk, because both differences share the same -consensus term and that
    alone produces a correlation of about 0.07 from pure noise. Nesting the regressions removes
    it.
    """
    g, c, n = g.ravel(), c.ravel(), n.ravel()
    ones = np.ones(len(g))
    return float(max(_fit_r2(g, np.column_stack([ones, c, n]))
                     - _fit_r2(g, np.column_stack([ones, c])), 0.0))


def breakeven_bps(pnl, traded, hi=200.0):
    f = lambda cst: (pnl - traded * (cst / 10000.0)).mean()
    if f(0.0) <= 0:
        return 0.0
    if f(hi) > 0:
        return float(hi)
    lo = 0.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return float((lo + hi) / 2)


def run(u):
    """Both arms, on one world. No ground truth (kind, true_acc) is read here."""
    fwd = u["returns"][1:]
    cut = fwd.shape[0] - HOLDOUT_DAYS
    q_cut = cut // QUARTER_LEN                    # quarters fully inside the research window
    g, c = u["g"], u["c"]

    rows = []
    for name, sig in u["vendors"].items():
        w = book(sig)
        pnl = (w * fwd).sum(axis=1)
        traded = np.abs(np.diff(w, axis=0, prepend=0.0)).sum(axis=1)

        # --- GATE 1: provenance, applied to BOTH arms ---
        ls = int(u["live_starts"][name])
        start = ls if 0 < ls < cut else 0
        q_start = start // QUARTER_LEN
        res_pnl, res_traded = pnl[start:cut], traded[start:cut]

        gq, cq = g[q_start:q_cut], c[q_start:q_cut]
        nq = u["nowcasts"][name][q_start:q_cut]

        rows.append({
            "vendor": name,
            "live_start": ls,
            "quarters_used": len(gq),
            "gross_sharpe": _sharpe(res_pnl),
            "p_value": _nw_pvalue(res_pnl),
            "accuracy_r2": float(np.corrcoef(gq.ravel(), nq.ravel())[0, 1] ** 2),
            "incremental_r2": incremental_r2(gq, cq, nq),
            "breakeven_bps": breakeven_bps(res_pnl, res_traded),
            "holdout_sharpe": _sharpe(pnl[cut:]),
        })

    df = pd.DataFrame(rows)

    # --- GATE 2, two ways ---
    df["sig_returns"] = df["p_value"] < P_THRESHOLD
    df["sig_fund"] = ((df["accuracy_r2"] >= MIN_ACCURACY)
                      & (df["incremental_r2"] >= MIN_INCREMENTAL))

    # --- GATE 3: economics, applied after each ---
    econ = df["breakeven_bps"] >= MIN_BREAKEVEN
    df["buy_returns"] = df["sig_returns"] & econ
    df["buy_fund"] = df["sig_fund"] & econ
    return df.sort_values("accuracy_r2", ascending=False).reset_index(drop=True)


def score(df, u):
    """Ground truth attached AFTER both arms have decided."""
    df = df.copy()
    df["kind"] = [u["kinds"][v] for v in df["vendor"]]
    real = df["kind"] == "genuine"
    n_real = int(real.sum())

    out = {}
    for arm in ["returns", "fund"]:
        for stage, col in [("sig", f"sig_{arm}"), ("buy", f"buy_{arm}")]:
            k = df[col]
            out[f"{arm}_{stage}_n"] = int(k.sum())
            out[f"{arm}_{stage}_prec"] = float(real[k].mean()) if k.any() else np.nan
            out[f"{arm}_{stage}_recall"] = float((real & k).sum() / n_real)
        k = df[f"buy_{arm}"]
        out[f"{arm}_mirrors_bought"] = int(((df["kind"] == "mirror") & k).sum())
        out[f"{arm}_holdout_buy"] = float(df.loc[k, "holdout_sharpe"].mean()) if k.any() else np.nan
        out[f"{arm}_holdout_rej"] = float(df.loc[~k, "holdout_sharpe"].mean())
    return df, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=0)
    args = ap.parse_args()

    if args.sweep:
        rows, t0 = [], time.time()
        for i, seed in enumerate(range(args.sweep), 1):
            u = build(seed)
            _, s = score(run(u), u)
            rows.append(s)
            if i % 5 == 0 or i == args.sweep:
                print(f"  {i}/{args.sweep} seeds ({(time.time()-t0)/i:.1f}s each)", flush=True)
        d = pd.DataFrame(rows)
        Path("results").mkdir(exist_ok=True)
        d.to_csv("results/pipeline2_sweep.csv", index=False)

        print(f"\nGATE 2: RETURNS vs FUNDAMENTALS   ({args.sweep} paired worlds, "
              f"30 vendors, 5 genuine, 2 mirror)")
        hdr = f"  {'':<14}{'bought':>8}{'precision':>11}{'recall':>9}{'mirrors':>9}{'hold gap':>10}"
        print(hdr)
        for arm, label in [("returns", "returns"), ("fund", "fundamentals")]:
            print(f"  {label:<14}{d[f'{arm}_buy_n'].mean():>8.2f}"
                  f"{d[f'{arm}_buy_prec'].mean():>10.1%}"
                  f"{d[f'{arm}_buy_recall'].mean():>9.1%}"
                  f"{d[f'{arm}_mirrors_bought'].mean():>9.2f}"
                  f"{(d[f'{arm}_holdout_buy'] - d[f'{arm}_holdout_rej']).mean():>+10.3f}")
        print("\n  before the economics gate:")
        for arm, label in [("returns", "returns"), ("fund", "fundamentals")]:
            print(f"  {label:<14}{d[f'{arm}_sig_n'].mean():>8.2f}"
                  f"{d[f'{arm}_sig_prec'].mean():>10.1%}"
                  f"{d[f'{arm}_sig_recall'].mean():>9.1%}")
        print("\n  mirrors = accurate vendors that add nothing over consensus. A fundamentals")
        print("  gate filtering on accuracy alone would buy them; the incremental test removes them.")
        return

    u = build()
    df, s = score(run(u), u)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/pipeline2.csv", index=False)

    pd.set_option("display.width", 180)
    print("GATE 2: RETURNS vs FUNDAMENTALS   (seed 42)")
    show = ["vendor", "kind", "gross_sharpe", "p_value", "accuracy_r2", "incremental_r2",
            "breakeven_bps", "buy_returns", "buy_fund", "holdout_sharpe"]
    print(df[show].head(12).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    for arm, label in [("returns", "returns"), ("fund", "fundamentals")]:
        print(f"\n  {label}: bought {s[f'{arm}_buy_n']}"
              f"  precision {s[f'{arm}_buy_prec']:.0%}"
              f"  recall {s[f'{arm}_buy_recall']:.0%}"
              f"  mirrors {s[f'{arm}_mirrors_bought']}"
              f"  holdout {s[f'{arm}_holdout_buy']:+.3f} vs {s[f'{arm}_holdout_rej']:+.3f}")


if __name__ == "__main__":
    main()