"""AltSignal - the true holdout. What the whole pipeline is worth on data it never saw.

Every number in this project so far was computed on data the method had already touched. Even
walk-forward reuses the same 1500 days across folds. That is fine for comparing methods, but
it cannot answer the question a fund actually asks before signing a contract:

    if I run this whole decision process, what fraction of what I buy will still work?

So the last 250 days are sealed. Nothing reads them - not the backfill gate, not the p-values,
not the config grid. The pipeline runs on the first 1249 days only, produces a buy list, and
THEN the holdout is opened once to score that list. This is the synthetic version of a paper
trial: the gate real desks weight most heavily, because it is the only one that cannot be
gamed by anything the researcher did.

THE PIPELINE, IN ORDER
  gate 1  provenance   a vendor declaring a live-start date is scored ONLY on data after it.
                       Reconstructed history is not evidence. This runs first because it
                       changes what the later gates are allowed to look at.
  gate 2  significance Newey-West p-value on the ensemble book, then Benjamini-Hochberg across
                       vendors to control the false discovery rate.
  gate 3  holdout      the sealed 250 days, opened once.

The interesting output is not whether the pipeline is accurate - the answer key tells you that
already. It is whether the HOLDOUT would have told you the same thing without the answer key.
If holdout performance of the buy list tracks its true composition, then a real desk running
this process on real vendors gets an honest read on its own hit rate.

Run:  python -m src.holdout
      python -m src.holdout --sweep 30
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.generate import build
from src.naive import CONFIGS, smooth, top_k, weights

HOLDOUT_DAYS = 250
FDR_Q = 0.10
TRADING_DAYS = 252
NW_LAGS = 5


def ensemble_weights(sig):
    return np.mean([weights(top_k(smooth(sig, s), k)) for _, s, k in CONFIGS], axis=0)


def _sharpe(pnl):
    sd = pnl.std()
    return 0.0 if sd == 0 else float(pnl.mean() / sd * np.sqrt(TRADING_DAYS))


def _nw_pvalue(pnl, lags=NW_LAGS):
    """One-sided p-value, robust to autocorrelation from held positions."""
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
    order = np.argsort(p)
    passed = p[order] <= (np.arange(1, m + 1) / m) * q
    out = np.zeros(m, dtype=bool)
    if passed.any():
        out[order[:np.max(np.where(passed)[0]) + 1]] = True
    return out


def run_pipeline(returns, vendors, live_starts):
    """Run gates 1-2 on the research window, then open the holdout. No answer key anywhere."""
    fwd = returns[1:]
    n = fwd.shape[0]
    cut = n - HOLDOUT_DAYS                      # everything at or past cut is sealed

    rows = []
    for name, sig in vendors.items():
        pnl = (ensemble_weights(sig) * fwd).sum(axis=1)

        # GATE 1: reconstructed history is not evidence. Start the research window at the
        # declared live-start date, not at day zero.
        ls = int(live_starts[name])
        start = ls if 0 < ls < cut else 0
        research = pnl[start:cut]
        holdout = pnl[cut:]

        rows.append({
            "vendor": name,
            "live_start": ls,
            "research_days": len(research),
            "research_sharpe": _sharpe(research),
            "p_value": _nw_pvalue(research),
            "holdout_sharpe": _sharpe(holdout),
            # what the naive reading would have been: full history, backfill included
            "naive_sharpe": _sharpe(pnl[:cut]),
        })

    df = pd.DataFrame(rows)
    df["accept"] = benjamini_hochberg(df["p_value"])           # GATE 2
    df["degradation"] = df["research_sharpe"] - df["holdout_sharpe"]
    return df.sort_values("research_sharpe", ascending=False).reset_index(drop=True)


def summarise(df, true_ics):
    """Score the buy list. The answer key is used HERE only, after the decision is made."""
    df = df.copy()
    df["true_ic"] = [true_ics[int(v[-2:])] for v in df["vendor"]]
    df["kind"] = np.where(df["true_ic"] > 0, "REAL", "fake")
    bought = df[df["accept"]]
    return df, {
        "n_bought": len(bought),
        "precision": float((bought["true_ic"] > 0).mean()) if len(bought) else np.nan,
        "n_real_found": int((bought["true_ic"] > 0).sum()),
        "holdout_sharpe_bought": float(bought["holdout_sharpe"].mean()) if len(bought) else np.nan,
        "holdout_sharpe_bought_real": float(
            bought.loc[bought["true_ic"] > 0, "holdout_sharpe"].mean()) if (bought["true_ic"] > 0).any() else np.nan,
        "holdout_sharpe_bought_fake": float(
            bought.loc[bought["true_ic"] == 0, "holdout_sharpe"].mean()) if (bought["true_ic"] == 0).any() else np.nan,
        "holdout_sharpe_rejected": float(df.loc[~df["accept"], "holdout_sharpe"].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=0)
    args = ap.parse_args()

    if args.sweep:
        rows, t0 = [], time.time()
        for i, seed in enumerate(range(args.sweep), 1):
            u = build(seed)
            df = run_pipeline(u["returns"], u["vendors"], u["live_starts"])
            _, s = summarise(df, u["true_ics"])
            s["seed"] = seed
            rows.append(s)
            if i % 5 == 0 or i == args.sweep:
                print(f"  {i}/{args.sweep} seeds ({(time.time() - t0) / i:.1f}s each)", flush=True)
        s = pd.DataFrame(rows)
        Path("results").mkdir(exist_ok=True)
        s.to_csv("results/holdout_sweep.csv", index=False)
        print(f"\nTRUE HOLDOUT   ({args.sweep} seeds, last {HOLDOUT_DAYS} days sealed)")
        print(f"  vendors bought per seed:       {s['n_bought'].mean():.2f}")
        print(f"  precision (bought that are real): {s['precision'].mean():.1%}")
        print(f"  real vendors found:            {s['n_real_found'].mean():.2f} of 5")
        print(f"\n  holdout Sharpe, bought:        {s['holdout_sharpe_bought'].mean():+.3f}")
        print(f"  holdout Sharpe, rejected:      {s['holdout_sharpe_rejected'].mean():+.3f}")
        print(f"  holdout Sharpe, bought & real: {s['holdout_sharpe_bought_real'].mean():+.3f}")
        print(f"  holdout Sharpe, bought & fake: {s['holdout_sharpe_bought_fake'].mean():+.3f}")
        print("\n  If bought > rejected on holdout, the pipeline's own out-of-sample read")
        print("  would have told a real desk it was working - with no answer key.")
        return

    returns = np.load("data/returns.npy")
    vendors = dict(np.load("data/vendors.npz"))
    key = pd.read_csv("data/answer_key.csv")

    df = run_pipeline(returns, vendors, dict(zip(key["vendor"], key["live_start"])))
    df, s = summarise(df, key["true_ic"].tolist())

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/holdout.csv", index=False)

    pd.set_option("display.width", 165)
    print(f"TRUE HOLDOUT   (last {HOLDOUT_DAYS} days sealed; gate 1 = provenance, "
          f"gate 2 = BH at q={FDR_Q})")
    show = ["vendor", "kind", "true_ic", "live_start", "naive_sharpe", "research_sharpe",
            "p_value", "accept", "holdout_sharpe"]
    print(df[show].head(14).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n  bought {s['n_bought']}   of which real: {s['n_real_found']}"
          f"   precision {s['precision']:.0%}")
    print(f"  holdout Sharpe - bought {s['holdout_sharpe_bought']:+.3f}"
          f"   rejected {s['holdout_sharpe_rejected']:+.3f}")
    print("\n  naive_sharpe includes reconstructed history; research_sharpe starts at the")
    print("  declared live-start date. The gap between them is what gate 1 removed.")


if __name__ == "__main__":
    main()