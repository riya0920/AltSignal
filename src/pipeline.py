"""AltSignal - the full pipeline with asynchronous latency integration.

Run:  python -m src.pipeline
      python -m src.pipeline --sweep 30
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.generate import build
from src.naive import CONFIGS, smooth, top_k, weights
from src.turnover import apply_rate

HOLDOUT_DAYS = 250
FDR_Q = 0.10
TRADE_RATE = 0.20
COST_BPS = 10.0
MIN_BREAKEVEN = 30.0
NW_LAGS = 5
TRADING_DAYS = 252


def apply_lag(sig: np.ndarray, lag: int = 0) -> np.ndarray:
    """Delays signal availability by `lag` trading days."""
    if lag <= 0:
        return sig
    pad = np.zeros((lag, sig.shape[1]), dtype=sig.dtype)
    return np.vstack([pad, sig[:-lag]])


def book(sig):
    tgt = np.mean([weights(top_k(smooth(sig, s), k)) for _, s, k in CONFIGS], axis=0)
    return apply_rate(tgt, TRADE_RATE)


def _sharpe(pnl):
    sd = pnl.std()
    return 0.0 if sd == 0 else float(pnl.mean() / sd * np.sqrt(TRADING_DAYS))


def _nw_pvalue(pnl, lags=NW_LAGS):
    x = pnl - pnl.mean()
    n = len(x)
    tot = (x @ x) / n
    for L in range(1, lags + 1):
        tot += 2.0 * (1.0 - L / (lags + 1.0)) * (x[L:] @ x[:-L]) / n
    se = np.sqrt(max(tot, 1e-18) / n)
    return float(1.0 - stats.norm.cdf(pnl.mean() / se)) if se > 0 else 1.0


def benjamini_hochberg(pvals, q=FDR_Q):
    p = np.asarray(pvals)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    passed = p[order] <= (np.arange(1, m + 1) / m) * q
    out = np.zeros(m, dtype=bool)
    if passed.any():
        out[order[:np.max(np.where(passed)[0]) + 1]] = True
    return out


def breakeven_bps(pnl, traded, hi=200.0):
    def net_mean(c):
        return (pnl - traded * (c / 10000.0)).mean()
    if net_mean(0.0) <= 0:
        return 0.0
    if net_mean(hi) > 0:
        return float(hi)
    lo = 0.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if net_mean(mid) > 0 else (lo, mid)
    return float((lo + hi) / 2)


def run(returns, vendors, live_starts, vendor_lags=None):
    if vendor_lags is None:
        vendor_lags = {name: 0 for name in vendors}

    fwd = returns[1:]
    cut = fwd.shape[0] - HOLDOUT_DAYS

    rows = []
    for name, sig in vendors.items():
        lag = int(vendor_lags.get(name, 0))
        sig_delayed = apply_lag(sig, lag=lag)

        w = book(sig_delayed)
        pnl = (w * fwd).sum(axis=1)
        traded = np.abs(np.diff(w, axis=0, prepend=0.0)).sum(axis=1)

        # --- GATE 1: provenance ---
        ls = int(live_starts[name])
        start = ls if 0 < ls < cut else 0
        res_pnl, res_traded = pnl[start:cut], traded[start:cut]

        net = res_pnl - res_traded * (COST_BPS / 10000.0)
        rows.append({
            "vendor": name,
            "live_start": ls,
            "lag_days": lag,
            "days_used": len(res_pnl),
            "days_discarded": start,
            "gross_sharpe": _sharpe(res_pnl),
            "net_sharpe": _sharpe(net),
            "p_value": _nw_pvalue(res_pnl),
            "turnover": float(res_traded.mean() * TRADING_DAYS),
            "breakeven_bps": breakeven_bps(res_pnl, res_traded),
            "holdout_sharpe": _sharpe(pnl[cut:]),
        })

    df = pd.DataFrame(rows)

    # --- GATE 2: significance ---
    df["pass_significance"] = df["p_value"] < 0.10

    # --- GATE 3: economics ---
    df["pass_economics"] = df["pass_significance"] & (df["breakeven_bps"] >= MIN_BREAKEVEN)
    df["buy"] = df["pass_economics"]
    return df.sort_values("p_value").reset_index(drop=True)


def score(df, true_ics):
    df = df.copy()
    df["true_ic"] = [true_ics[int(v[-2:])] for v in df["vendor"]]
    df["kind"] = np.where(df["true_ic"] > 0, "REAL", "fake")
    n_real = int((df["true_ic"] > 0).sum())

    bought = df[df["buy"]]
    sig_only = df[df["pass_significance"]]
    return df, {
        "n_bought": len(bought),
        "precision": float((bought["true_ic"] > 0).mean()) if len(bought) else np.nan,
        "recall": float((bought["true_ic"] > 0).sum() / n_real),
        "n_after_significance": len(sig_only),
        "precision_after_significance": float((sig_only["true_ic"] > 0).mean()) if len(sig_only) else np.nan,
        "backfill_days_discarded": int(df["days_discarded"].sum()),
        "holdout_bought": float(bought["holdout_sharpe"].mean()) if len(bought) else np.nan,
        "holdout_rejected": float(df.loc[~df["buy"], "holdout_sharpe"].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=0)
    args = ap.parse_args()

    if args.sweep:
        rows, t0 = [], time.time()
        for i, seed in enumerate(range(args.sweep), 1):
            u = build(seed)
            df = run(u["returns"], u["vendors"], u["live_starts"], vendor_lags=u.get("lags"))
            _, s = score(df, u["true_ics"])
            s["seed"] = seed
            rows.append(s)
            if i % 5 == 0 or i == args.sweep:
                print(f"  {i}/{args.sweep} seeds ({(time.time() - t0) / i:.1f}s each)", flush=True)
        s = pd.DataFrame(rows)
        Path("results").mkdir(exist_ok=True)
        s.to_csv("results/pipeline_sweep.csv", index=False)

        print(f"\nFULL PIPELINE   ({args.sweep} seeds, 30 vendors, 5 real)")
        print(f"  after significance:  {s['n_after_significance'].mean():.2f} vendors,"
              f" {s['precision_after_significance'].mean():.1%} real")
        print(f"  after economics:     {s['n_bought'].mean():.2f} vendors,"
              f" {s['precision'].mean():.1%} real")
        print(f"  recall:              {s['recall'].mean():.1%} of the 5 real vendors")
        print(f"\n  holdout Sharpe, bought:   {s['holdout_bought'].mean():+.3f}")
        print(f"  holdout Sharpe, rejected: {s['holdout_rejected'].mean():+.3f}")
        print("\n  The holdout gap is the number a real desk could actually observe.")
        return

    returns = np.load("data/returns.npy")
    vendors = dict(np.load("data/vendors.npz"))
    key = pd.read_csv("data/answer_key.csv")

    lags = dict(zip(key["vendor"], key["lag_days"])) if "lag_days" in key.columns else None
    live_starts = dict(zip(key["vendor"], key["live_start"]))

    df = run(returns, vendors, live_starts, vendor_lags=lags)
    df, s = score(df, key["true_ic"].tolist())

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/pipeline.csv", index=False)

    pd.set_option("display.width", 175)
    print(f"FULL PIPELINE   (rate {TRADE_RATE}, cost {COST_BPS:.0f}bps, "
          f"BH q={FDR_Q}, min breakeven {MIN_BREAKEVEN:.0f}bps, {HOLDOUT_DAYS}d sealed)")
    show = ["vendor", "kind", "true_ic", "lag_days", "days_discarded", "gross_sharpe", "p_value",
            "pass_significance", "breakeven_bps", "buy", "holdout_sharpe"]
    print(df[show].head(12).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nGATE BY GATE   (30 vendors in, 5 of them real)")
    print(f"  1 provenance   {s['backfill_days_discarded']} vendor-days of reconstructed history discarded")
    print(f"  2 significance {s['n_after_significance']} survive, {s['precision_after_significance']:.0%} real")
    print(f"  3 economics    {s['n_bought']} survive, {s['precision']:.0%} real")
    print(f"  4 holdout      bought {s['holdout_bought']:+.3f}   rejected {s['holdout_rejected']:+.3f}")
    print(f"\n  recall: {s['recall']:.0%} of the real vendors found")


if __name__ == "__main__":
    main()