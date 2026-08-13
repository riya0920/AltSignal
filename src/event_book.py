"""AltSignal - the event book. Trading a quarterly signal the way a real earnings desk does.

The fundamentals gate recovers all five genuine vendors in every world. The economics gate then
throws two of them away, and it is not wrong to do so - it is being handed an unrealistic book.

The problem is how the quarterly view was being traded. The generic ensemble holds a position
every single day of the quarter, but the signal only pays on the ~23 days when a company
actually reports. So the book carries risk for sixty days to collect on one, and pays turnover
the whole time. Breakeven computed on that book understates enormously: on seed 42, vendor_03
posts a breakeven of 0.00 bps and then earns +2.04 on the holdout.

A real earnings desk does not do this. It builds into the print and exits after it:

    HOLD_BEFORE   days ahead of the announcement to start carrying the position
    HOLD_AFTER    days after the print to exit (the drift is not all captured on day one)

Everything outside those windows is flat. Fewer days in the market, the same P&L, far less
turnover - which is exactly the trade the cost gate is supposed to reward.

WHAT TO WATCH
Concentration is not free. A book that is flat 80% of the time has a much smaller denominator,
so Sharpe can move in either direction: gross P&L per day in the market rises, but the variance
of that P&L rises too. The honest test is whether the ECONOMICS gate stops discarding genuine
vendors, not whether the raw Sharpe looks bigger.

Run:  python -m src.event_book
      python -m src.event_book --sweep 14
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.world import build, QUARTER_LEN
from src.turnover import apply_rate
from src.pipeline2 import (book as base_book, breakeven_bps, incremental_r2, _sharpe,
                           HOLDOUT_DAYS, TRADE_RATE, MIN_ACCURACY, MIN_INCREMENTAL)

HOLD_BEFORE = 10        # start building the position this many days before the print
HOLD_AFTER = 3          # exit this many days after it
MIN_BREAKEVEN = 5.0
TRADING_DAYS = 252


def event_windows(announce, n_rows, before=HOLD_BEFORE, after=HOLD_AFTER):
    """Boolean mask of the days on which the book is allowed to carry a position."""
    live = np.zeros(n_rows, dtype=bool)
    for day in announce.values():
        live[max(0, day - before):min(n_rows, day + after + 1)] = True
    return live


def event_book(sig, live):
    """The SAME book as the pipeline, masked flat outside the announcement windows.

    Both arms must share a construction or the comparison measures the wrong thing: an earlier
    version rebuilt the baseline from the raw signal instead of the nine-config ensemble, which
    scored 0.25 against the pipeline's 1.39 and made concentration look catastrophic. The only
    difference between the two arms here is the mask.
    """
    return base_book(sig) * live[:, None]


def analyse(u, before=HOLD_BEFORE, after=HOLD_AFTER):
    fwd = u["returns"][1:]
    n_rows = fwd.shape[0]
    cut = n_rows - HOLDOUT_DAYS
    q_cut = cut // QUARTER_LEN
    live = event_windows(u["announce"], n_rows, before, after)
    g, c = u["g"], u["c"]

    rows = []
    for name, sig in u["vendors"].items():
        for label, w in [("always_on", base_book(sig)),
                         ("event", event_book(sig, live))]:
            pnl = (w * fwd).sum(axis=1)
            traded = np.abs(np.diff(w, axis=0, prepend=0.0)).sum(axis=1)

            ls = int(u["live_starts"][name])                    # GATE 1
            start = ls if 0 < ls < cut else 0
            q_start = start // QUARTER_LEN
            res_pnl, res_traded = pnl[start:cut], traded[start:cut]

            gq, cq = g[q_start:q_cut], c[q_start:q_cut]
            nq = u["nowcasts"][name][q_start:q_cut]

            rows.append({
                "vendor": name, "book": label,
                "days_in_market": float((np.abs(w[start:cut]).sum(1) > 0).mean()),
                "turnover": float(res_traded.mean() * TRADING_DAYS),
                "gross_sharpe": _sharpe(res_pnl),
                "accuracy_r2": float(np.corrcoef(gq.ravel(), nq.ravel())[0, 1] ** 2),
                "incremental_r2": incremental_r2(gq, cq, nq),
                "breakeven_bps": breakeven_bps(res_pnl, res_traded),
                "holdout_sharpe": _sharpe(pnl[cut:]),
            })

    df = pd.DataFrame(rows)
    df["kind"] = [u["kinds"][v] for v in df["vendor"]]
    df["pass_fund"] = ((df["accuracy_r2"] >= MIN_ACCURACY)
                       & (df["incremental_r2"] >= MIN_INCREMENTAL))
    df["buy"] = df["pass_fund"] & (df["breakeven_bps"] >= MIN_BREAKEVEN)
    return df


def summarise(df):
    real = df["kind"] == "genuine"
    out = []
    for label, sub in df.groupby("book", sort=False):
        r = sub["kind"] == "genuine"
        k = sub["buy"]
        out.append({
            "book": label,
            "days_in_market": sub["days_in_market"].mean(),
            "turnover": sub["turnover"].mean(),
            "n_bought": int(k.sum()),
            "precision": float(r[k].mean()) if k.any() else np.nan,
            "recall": float((r & k).sum() / r.sum()),
            "genuine_breakeven": float(sub.loc[r, "breakeven_bps"].mean()),
            "hold_gap": (float(sub.loc[k, "holdout_sharpe"].mean()) if k.any() else np.nan)
                        - float(sub.loc[~k, "holdout_sharpe"].mean()),
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=0)
    args = ap.parse_args()

    if args.sweep:
        rows, t0 = [], time.time()
        for i, seed in enumerate(range(args.sweep), 1):
            s = summarise(analyse(build(seed)))
            s["seed"] = seed
            rows.append(s)
            if i % 5 == 0 or i == args.sweep:
                print(f"  {i}/{args.sweep} seeds ({(time.time()-t0)/i:.1f}s each)", flush=True)
        d = pd.concat(rows, ignore_index=True)
        Path("results").mkdir(exist_ok=True)
        d.to_csv("results/event_book_sweep.csv", index=False)
        g = d.groupby("book", sort=False).mean(numeric_only=True).drop(columns="seed")
        print(f"\nEVENT BOOK vs ALWAYS-ON   ({args.sweep} paired worlds, "
              f"hold {HOLD_BEFORE}d before / {HOLD_AFTER}d after the print)")
        print(g.to_string(float_format=lambda x: f"{x:.3f}"))
        print("\n  recall is what matters: the fundamentals gate finds all five genuine")
        print("  vendors, and the question is how many the economics gate then discards.")
        return

    u = build()
    df = analyse(u)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/event_book.csv", index=False)

    pd.set_option("display.width", 175)
    print(f"EVENT BOOK   (seed 42, hold {HOLD_BEFORE}d before / {HOLD_AFTER}d after)")
    gen = df[df["kind"] == "genuine"].sort_values(["vendor", "book"])
    print(gen[["vendor", "book", "days_in_market", "turnover", "gross_sharpe",
               "breakeven_bps", "buy", "holdout_sharpe"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nSUMMARY")
    print(summarise(df).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()