"""AltSignal Step 4 - pay for your searches.

Steps 2 and 3 produced a RANKING. A ranking cannot tell you where to stop. Even after the
ensemble cut the error rate from 53.0% to 43.0%, there is no defensible line on the page:
"buy the top four" is a number picked by eye, and "buy anything above 0.9" is a threshold
borrowed from an oracle noise band that only exists because the answer key exists.

Step 4 replaces the ranking with a DECISION. Each vendor gets a p-value and a rule, and
"survives" finally means something precise: not rejected at the chosen significance level.

The thing being corrected for is the size of the search. Twelve vendors were each read nine
ways, so 108 books were scored. The best of 108 looks good even when all 108 are worthless -
that is not a flaw in any one test, it is arithmetic. Three corrections, in increasing
sophistication:

  Bonferroni       - demand N times stronger evidence from each test. Controls the chance of
                     even ONE false positive, at the cost of destroying weak real signal.
                     Expect it to reject the 0.010 vendor essentially always.
  Benjamini-Hochberg - control the FALSE DISCOVERY RATE instead: allow a stated fraction of
                     accepted vendors to be junk. Far gentler on genuine weak signal, and the
                     right framing here, because buying one dud out of five picks is a normal
                     business outcome while missing every weak-but-real vendor is not.
  Deflated Sharpe  - Bailey & Lopez de Prado. Instead of adjusting a p-value after the fact,
                     it asks directly: given that I ran N trials whose Sharpes had variance V,
                     what Sharpe would the LUCKIEST worthless strategy have produced? Then it
                     tests the observed Sharpe against that benchmark rather than against zero,
                     and corrects for the non-normality of the P&L while it is at it.

The permutation null in this file is the piece that matters most for the project's premise.
Every noise band quoted so far came from vendors KNOWN to be fake - an oracle, unavailable in
real life. Circularly shifting a vendor's signal in time destroys its alignment with returns
while preserving its own autocorrelation and cross-sectional structure, so the resulting
Sharpes are a null distribution estimated WITHOUT the answer key. That is the first thing in
this repo that could be pointed at a real vendor.

Run:  python -m src.significance                  (one seed, full report + permutation check)
      python -m src.significance --sweep 200      (decision error rates across seeds)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.generate import build
from src.naive import CONFIGS, smooth, top_k, weights

EULER_GAMMA = 0.5772156649015329
TRADING_DAYS = 252

# Total trials in the search: 12 vendors x 9 configs. This is the number the corrections are
# paid against. Using 9 here instead would be quietly self-serving - the search that produced
# the winner was the whole exercise, not one vendor's slice of it.
N_VENDORS = 12
N_TRIALS = N_VENDORS * len(CONFIGS)

ALPHA = 0.05        # significance level for Bonferroni and deflated Sharpe
FDR_Q = 0.10        # tolerated false discovery rate for Benjamini-Hochberg


# ---------------------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------------------

def config_books(sig, fwd):
    """Daily P&L for each of the nine configs. Shape (n_configs, n_days)."""
    return np.array([(weights(top_k(smooth(sig, s), k)) * fwd).sum(axis=1)
                     for _, s, k in CONFIGS])


def ensemble_weights(sig):
    """Position matrix of the Step 3 ensemble book: the nine config books, equally weighted.

    P&L is linear in the weights, so averaging the nine weight matrices and scoring once is
    identical to scoring nine books and averaging - and it leaves a single position matrix
    that can be shifted in time, which the permutation null needs.
    """
    return np.mean([weights(top_k(smooth(sig, s), k)) for _, s, k in CONFIGS], axis=0)


def daily_sharpe(pnl):
    sd = pnl.std(ddof=1)
    return 0.0 if sd == 0 else pnl.mean() / sd


def newey_west_se(pnl, lags=5):
    """Standard error of the mean, robust to autocorrelation in daily P&L.

    A plain t-test assumes each day is independent. Smoothed signals hold positions for days
    at a time, so consecutive P&L is correlated and the naive standard error is too small -
    which makes p-values too optimistic in exactly the direction that flatters the result.
    Newey-West adds the autocovariance terms back in with Bartlett weights.
    """
    x = pnl - pnl.mean()
    n = len(x)
    gamma0 = (x @ x) / n
    total = gamma0
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        total += 2.0 * w * (x[L:] @ x[:-L]) / n
    return np.sqrt(max(total, 1e-18) / n)


def sharpe_pvalue(pnl, lags=5):
    """One-sided p-value for H0: the strategy's true mean return is zero."""
    se = newey_west_se(pnl, lags)
    t = pnl.mean() / se if se > 0 else 0.0
    return float(1.0 - stats.norm.cdf(t)), float(t)


# ---------------------------------------------------------------------------------------
# multiple testing corrections
# ---------------------------------------------------------------------------------------

def bonferroni(pvals, alpha=ALPHA, n_trials=N_TRIALS):
    """Reject only where p < alpha / n_trials. Controls the family-wise error rate.

    Blunt on purpose: it protects against even a single false positive, which is a stricter
    promise than this problem needs.
    """
    return np.asarray(pvals) < alpha / n_trials


def benjamini_hochberg(pvals, q=FDR_Q):
    """Control the expected fraction of ACCEPTED vendors that are junk at q.

    Sort p ascending; find the largest i with p_(i) <= (i/m) * q; reject everything up to it.
    The step-up structure is what makes it gentler than Bonferroni: a vendor with a mediocre
    p-value can still be accepted if enough vendors ahead of it look strong.
    """
    p = np.asarray(pvals)
    m = len(p)
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * q
    passed = p[order] <= thresh
    out = np.zeros(m, dtype=bool)
    if passed.any():
        cut = np.max(np.where(passed)[0])
        out[order[:cut + 1]] = True
    return out


def expected_max_sharpe(trial_sharpes, n_trials=N_TRIALS):
    """The Sharpe the LUCKIEST of n_trials worthless strategies would be expected to reach.

    Bailey & Lopez de Prado's benchmark. It scales with the spread of the trial Sharpes: if
    every configuration produces a similar number, luck has little room to work and the bar
    stays low. If they scatter widely, the maximum is mostly noise and the bar rises.
    """
    v = np.std(trial_sharpes, ddof=1)
    if v == 0 or n_trials < 2:
        return 0.0
    a = stats.norm.ppf(1.0 - 1.0 / n_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(v * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe(pnl, sr0, sr=None):
    """Probability the strategy's true Sharpe exceeds the luck benchmark sr0.

    All quantities are per-observation (daily), never annualised - annualising inside the
    formula is the classic way to get this wrong by a factor of sqrt(252).

    The denominator carries the non-normality correction: negative skew and fat tails both
    make an observed Sharpe less trustworthy, and daily P&L has plenty of both.
    """
    t = len(pnl)
    sr = daily_sharpe(pnl) if sr is None else sr
    sk = float(stats.skew(pnl))
    ku = float(stats.kurtosis(pnl, fisher=False))
    denom = np.sqrt(max(1.0 - sk * sr + ((ku - 1.0) / 4.0) * sr ** 2, 1e-12))
    z = (sr - sr0) * np.sqrt(t - 1) / denom
    return float(stats.norm.cdf(z))


# ---------------------------------------------------------------------------------------
# permutation null - the noise band WITHOUT the answer key
# ---------------------------------------------------------------------------------------

def permutation_null(vendors, fwd, n_perm=200, rng=None, min_shift=63):
    """Null distribution of the ensemble Sharpe, estimated with no ground truth.

    Circularly shifting a vendor's signal in time breaks its alignment with returns but keeps
    its own autocorrelation, its cross-sectional structure, and the market's volatility
    clustering intact. Whatever Sharpe survives that shift is what the vendor's shape alone
    is worth - which is the definition of the noise band.

    Returns (per_vendor_null, max_null): the pooled null Sharpes, and the per-permutation
    maximum across vendors. The second is the family-wise band: the score to beat when you
    are picking the best of twelve, not evaluating one in isolation.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = fwd.shape[0]
    # The shift must be applied to the POSITIONS, before they meet returns. Rolling a
    # finished P&L series is a no-op: it reorders the same daily numbers, so mean and
    # standard deviation are unchanged and the "null" comes back exactly equal to the
    # observed Sharpe. The break has to happen where signal and returns are joined.
    W = {name: ensemble_weights(sig) for name, sig in vendors.items()}

    pooled, maxima = [], []
    for _ in range(n_perm):
        this = []
        for w in W.values():
            k = int(rng.integers(min_shift, n - min_shift))
            pnl = (np.roll(w, k, axis=0) * fwd).sum(axis=1)
            this.append(daily_sharpe(pnl) * np.sqrt(TRADING_DAYS))
        pooled.extend(this)
        maxima.append(max(this))
    return np.array(pooled), np.array(maxima)


# ---------------------------------------------------------------------------------------
# the scorer
# ---------------------------------------------------------------------------------------

def score_all_sig(returns, vendors, alpha=ALPHA, q=FDR_Q):
    """Score every vendor and attach a DECISION under each correction.

    Pure: no files, no printing, and the answer key is never read here.
    """
    fwd = returns[1:]

    books = {name: config_books(sig, fwd) for name, sig in vendors.items()}

    rows = []
    for name, b in books.items():
        trials = np.array([daily_sharpe(p) for p in b])
        # The luck benchmark is built from THIS vendor's own nine trial Sharpes, not from the
        # spread across all twelve vendors. Cross-vendor spread is mostly genuine difference
        # in skill, and feeding it in as if it were luck sets the bar absurdly high - it
        # rejected all twelve on the first run. A vendor's own config spread is the right
        # scale: tight spread means the reading is robust and luck had little room; wide
        # spread means the result depends on which lens you used.
        sr0 = expected_max_sharpe(trials, len(CONFIGS))
        ens = b.mean(axis=0)                       # the Step 3 ensemble book
        p, t = sharpe_pvalue(ens)
        rows.append({
            "vendor": name,
            "ens_sharpe": daily_sharpe(ens) * np.sqrt(TRADING_DAYS),
            "t_stat": t,
            "p_value": p,
            "dsr": deflated_sharpe(ens, sr0),
            "sr0_benchmark": sr0 * np.sqrt(TRADING_DAYS),
            "frac_pos": float((trials > 0).mean()),
        })

    df = pd.DataFrame(rows)
    df["accept_raw"] = df["p_value"] < alpha
    df["accept_bonf"] = bonferroni(df["p_value"], alpha)
    df["accept_bh"] = benjamini_hochberg(df["p_value"], q)
    df["accept_dsr"] = df["dsr"] > (1.0 - alpha)
    return df.sort_values("ens_sharpe", ascending=False).reset_index(drop=True)


METHODS = ["accept_raw", "accept_bonf", "accept_bh", "accept_dsr"]


# ---------------------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------------------

def single_seed_report(n_perm=200):
    returns = np.load("data/returns.npy")
    vendors = dict(np.load("data/vendors.npz"))
    key = pd.read_csv("data/answer_key.csv")

    df = score_all_sig(returns, vendors)
    df = df.merge(key, on="vendor", how="left")        # DISPLAY ONLY, after all scoring
    df["kind"] = np.where(df["true_ic"] > 0, "REAL", "fake")

    out = Path("results")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "step4_significance.csv", index=False)

    pd.set_option("display.width", 170)
    print(f"STEP 4 SIGNIFICANCE   (N_TRIALS = {N_TRIALS}, alpha = {ALPHA}, FDR q = {FDR_Q})")
    print("  sr0 = per-vendor luck benchmark: the Sharpe the luckiest of its own"
          f" {len(CONFIGS)} configs would reach by chance\n")
    show = (["vendor", "kind", "true_ic", "ens_sharpe", "t_stat", "p_value",
             "sr0_benchmark", "dsr"] + METHODS)
    print(df[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nDECISION ERRORS   (this seed)")
    for m in METHODS:
        fp = int(((df["true_ic"] == 0) & df[m]).sum())
        fn = int(((df["true_ic"] > 0) & ~df[m]).sum())
        print(f"  {m:<12} bought {int(df[m].sum()):2d}   false positives {fp}   missed real {fn}")

    fwd = returns[1:]
    print(f"\nPERMUTATION NULL   ({n_perm} circular shifts, NO answer key used)")
    pooled, maxima = permutation_null(vendors, fwd, n_perm)
    print(f"  single-vendor null: median {np.median(pooled):+.3f},"
          f" 95th pct {np.quantile(pooled, 0.95):.3f}")
    print(f"  best-of-12 null:    median {np.median(maxima):+.3f},"
          f" 95th pct {np.quantile(maxima, 0.95):.3f}   <- the band to beat")
    band = np.quantile(maxima, 0.95)
    print(f"  vendors clearing the estimated band: "
          f"{', '.join(df.loc[df.ens_sharpe > band, 'vendor'].tolist()) or 'none'}")
    print("  (compare against the ORACLE band from robustness.py - if these agree, the")
    print("   project has a detector that works without ground truth)")


def sweep(n_seeds, start=0):
    """Decision error rates across seeds. Steps 2-3 measured ranking; this measures choices."""
    rows = []
    t0 = time.time()
    for i, seed in enumerate(range(start, start + n_seeds), 1):
        u = build(seed)
        df = score_all_sig(u["returns"], u["vendors"])
        truth = dict(zip([f"vendor_{j:02d}" for j in range(len(u["true_ics"]))], u["true_ics"]))
        df["true_ic"] = df["vendor"].map(truth)
        r = {"seed": seed}
        for m in METHODS:
            r[f"{m}_fp"] = int(((df["true_ic"] == 0) & df[m]).sum())
            r[f"{m}_fn"] = int(((df["true_ic"] > 0) & ~df[m]).sum())
            r[f"{m}_weak_found"] = bool(df.loc[df["true_ic"] == 0.010, m].any())
            r[f"{m}_n"] = int(df[m].sum())
        rows.append(r)
        if i % 10 == 0 or i == n_seeds:
            rate = (time.time() - t0) / i
            print(f"  {i}/{n_seeds} seeds   ({rate:.1f}s each, "
                  f"~{rate * (n_seeds - i) / 60:.1f} min left)", flush=True)

    s = pd.DataFrame(rows)
    out = Path("results")
    out.mkdir(exist_ok=True)
    s.to_csv(out / "step4_sweep.csv", index=False)

    print(f"\nSTEP 4 DECISION ERRORS   ({n_seeds} seeds, {time.time() - t0:.0f}s)")
    print(f"  {'method':<12}{'bought':>8}{'false pos':>11}{'>=1 FP':>9}"
          f"{'missed real':>13}{'weak found':>12}")
    for m in METHODS:
        print(f"  {m:<12}{s[m + '_n'].mean():>8.2f}{s[m + '_fp'].mean():>11.2f}"
              f"{(s[m + '_fp'] > 0).mean():>9.1%}{s[m + '_fn'].mean():>13.2f}"
              f"{s[m + '_weak_found'].mean():>12.1%}")
    print("\n  false pos   = zero-skill vendors bought (out of 7)")
    print("  missed real = genuine vendors passed over (out of 5)")
    print("  A method that buys nothing scores perfectly on false positives. Read both.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=0, help="run the multi-seed decision sweep")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--perm", type=int, default=200, help="permutation draws in single-seed mode")
    args = ap.parse_args()

    if args.sweep:
        sweep(args.sweep, args.start)
    else:
        single_seed_report(args.perm)


if __name__ == "__main__":
    main()