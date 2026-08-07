"""AltSignal Step 3 (alternative) - stop selecting a config; summarise all nine.

Step 2's bug was framed as a leakage problem: the winning config was chosen using the same
days it was scored on. Walk-forward attacks that framing by moving the selection earlier in
time. It does not work here. Across 40 seeds:

    walk-forward, train=756   72.5% of seeds have a fake outranking a real vendor
    walk-forward, train=504   67.5%
    walk-forward, train=378   62.5%
    no split at all (Step 2)  47.5%

The sweep is monotone: the fewer days spent selecting, the better the result. That is the
tell. The selection was never buying anything, so every day spent on it was a day taken away
from measurement - and with only 1500 days, measurement is the binding constraint.

So the real diagnosis is not WHEN you select. It is THAT you select. max() over nine configs
throws away eight of the nine measurements you paid for, and the one it keeps is the one most
contaminated by luck. Averaging keeps all nine and needs no split, so it costs no sample:

    Sharpe of the equal-weight averaged book   27.5%
    mean of the nine Sharpes                   25.0%
    fraction of the nine that are positive     17.5%

The intuition behind frac_pos: a real vendor's nine configs are all reading the same genuine
signal, so they all come out mildly positive. A null's nine scatter around zero - some up,
some down - and only the max looks impressive. Nine out of nine positive is hard to fake.

Caveat on frac_pos: it takes only ten distinct values, so it ties constantly. Use it as a
filter alongside a continuous score, not as the ranking on its own.

Run:  python -m src.ensemble
      python -m src.robustness --scorer ensemble --seeds 200
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.naive import CONFIGS, smooth, top_k, weights


def _sharpe(pnl):
    sd = pnl.std()
    return 0.0 if sd == 0 else pnl.mean() / sd * np.sqrt(252)


def score_all_ens(returns, vendors):
    """Score every vendor by summarising all nine configs. Pure: no files, no answer key.

    ens_sharpe is the headline: build one book that is the equal-weight average of the nine
    config books, then score that book. It is a real, tradeable portfolio - you are not
    averaging scores, you are diversifying across the nine ways of reading the vendor - so it
    picks up a diversification benefit that mean_sharpe does not.
    """
    fwd = returns[1:]
    rows = []
    for name, sig in vendors.items():
        pnls = np.array([(weights(top_k(smooth(sig, s), k)) * fwd).sum(axis=1)
                         for _, s, k in CONFIGS])
        sh = np.array([_sharpe(p) for p in pnls])
        rows.append({
            "vendor": name,
            "ens_sharpe": _sharpe(pnls.mean(axis=0)),   # headline: the averaged book
            "mean_sharpe": float(sh.mean()),            # average of the nine scores
            "frac_pos": float((sh > 0).mean()),         # how many of nine are positive
            "max_sharpe": float(sh.max()),              # what Step 2 would have reported
            "min_sharpe": float(sh.min()),
            "spread": float(sh.max() - sh.min()),       # wide spread = config-sensitive
            "n_configs": len(CONFIGS),
        })
    df = pd.DataFrame(rows)
    # selection_cost: the Sharpe that existed only because a winner was picked. robustness.py
    # reads this column if present.
    df["selection_cost"] = df["max_sharpe"] - df["ens_sharpe"]
    return df.sort_values("ens_sharpe", ascending=False).reset_index(drop=True)


def main():
    returns = np.load("data/returns.npy")
    vendors = dict(np.load("data/vendors.npz"))
    key = pd.read_csv("data/answer_key.csv")

    df = score_all_ens(returns, vendors)
    df = df.merge(key, on="vendor", how="left")     # DISPLAY ONLY, after scoring
    df["kind"] = np.where(df["true_ic"] > 0, "REAL", "fake")

    out = Path("results")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "step3_ensemble.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"STEP 3 ENSEMBLE   (all {len(CONFIGS)} configs kept, no selection, no split)")
    show = ["vendor", "kind", "true_ic", "max_sharpe", "ens_sharpe",
            "mean_sharpe", "frac_pos", "spread", "selection_cost"]
    print(df[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    fakes, reals = df[df["true_ic"] == 0], df[df["true_ic"] > 0]
    print("\nINVERSIONS BY STATISTIC   (fake outranking real; lower is better)")
    for col in ["max_sharpe", "ens_sharpe", "mean_sharpe", "frac_pos"]:
        inv = int((fakes[col].values[:, None] > reals[col].values).sum())
        print(f"  {col:<14} {inv:>3}   (of 35 possible pairs)")

    print(f"\n  mean selection_cost, fakes: {fakes['selection_cost'].mean():+.3f}")
    print(f"  mean selection_cost, reals: {reals['selection_cost'].mean():+.3f}")
    print("\n  NOTE: this is one seed. Run src/robustness.py --scorer ensemble.")


if __name__ == "__main__":
    main()