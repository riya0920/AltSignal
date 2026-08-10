"""AltSignal - White's Reality Check, Hansen's SPA, and Romano-Wolf stepdown.

Bonferroni and Benjamini-Hochberg both assume the tests are independent. Yours are not. All
twelve vendors trade the same 400 names, in the same market, on the same days: when the market
has a bad month most of their books have a bad month together. Correcting as if they were
independent charges for twelve separate lottery tickets when the tickets share most of their
numbers. The result is over-penalisation, which is why Bonferroni kept losing the weak vendor.

Reality Check does not correct a p-value after the fact. It builds the null distribution of
"best of twelve" directly, by resampling all twelve books ON THE SAME resampled days a
thousand times. The cross-strategy correlation survives because the strategies are never
pulled apart.

Three pieces:

  stationary bootstrap   resamples blocks of days with geometric lengths, not single days.
                         Daily P&L is autocorrelated - smoothed signals hold positions for
                         days - and shuffling single days would destroy that and make every
                         result look more certain than it is.
  Hansen's SPA           studentises each strategy and excludes hopeless ones from setting
                         the benchmark. Under plain Reality Check a vendor losing money badly
                         still raises the bar for everyone; SPA recentres so it cannot.
  Romano-Wolf stepdown   plain Reality Check answers only "is the BEST one real?". Stepdown
                         gives a verdict per vendor: test the best, and if it passes remove
                         it and re-run on the rest. Each round faces a slightly easier bar
                         because the accepted ones are no longer competing.

Run:  python -m src.reality_check
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.significance import ensemble_weights, newey_west_se
from src.generate import build

N_BOOT = 1000
MEAN_BLOCK = 20        # average bootstrap block length in days
ALPHA = 0.05
TRADING_DAYS = 252


def stationary_bootstrap_idx(n, n_boot, mean_block, rng):
    """Politis-Romano stationary bootstrap: index sets of geometric-length blocks.

    Each step either continues the current block or starts a new one at a random point, with
    probability 1/mean_block of restarting. Block lengths are therefore geometric with mean
    mean_block, which keeps the resampled series stationary - unlike fixed-length blocks,
    which leave artefacts at the seams.
    """
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    starts = rng.integers(0, n, size=(n_boot, n))
    restart = rng.random((n_boot, n)) < p
    for b in range(n_boot):
        cur = starts[b, 0]
        row = idx[b]
        for t in range(n):
            if t > 0:
                cur = starts[b, t] if restart[b, t] else (cur + 1) % n
            row[t] = cur
    return idx


def _stats(F, lags=5):
    """Observed mean, HAC standard error, and studentised statistic per strategy."""
    means = F.mean(axis=0)
    ses = np.array([newey_west_se(F[:, k], lags) for k in range(F.shape[1])])
    ses = np.maximum(ses, 1e-18)
    return means, ses, means / ses


def bootstrap_matrix(F, n_boot=N_BOOT, mean_block=MEAN_BLOCK, seed=0):
    """Studentised, recentred bootstrap statistics. Shape (n_boot, n_strategies).

    The same resampled day-index is applied to EVERY strategy in a given replication. That is
    the whole point: it preserves the correlation between the books, which is exactly what
    Bonferroni and BH throw away.
    """
    rng = np.random.default_rng(seed)
    n = F.shape[0]
    idx = stationary_bootstrap_idx(n, n_boot, mean_block, rng)
    means, ses, _ = _stats(F)
    boot_means = np.array([F[i].mean(axis=0) for i in idx])
    return (boot_means - means) / ses            # recentred under the null


def reality_check(F, n_boot=N_BOOT, seed=0):
    """White's Reality Check p-value for H0: the best strategy has no genuine edge."""
    _, _, tstats = _stats(F)
    Z = bootstrap_matrix(F, n_boot, seed=seed)
    v_obs = max(0.0, float(tstats.max()))
    v_boot = np.maximum(Z.max(axis=1), 0.0)
    return float((v_boot >= v_obs).mean())


def hansen_spa(F, n_boot=N_BOOT, seed=0):
    """Hansen's SPA p-value. Same idea, but hopeless strategies stop setting the bar.

    A strategy whose mean is worse than -A_k is treated as having zero edge under the null
    rather than its own (very negative) mean. Without this, one badly losing vendor drags the
    benchmark down and makes everything else look better than it is.
    """
    means, ses, tstats = _stats(F)
    n = F.shape[0]
    thresh = 0.25 * ses * n ** (-0.25) * np.sqrt(2.0 * np.log(np.log(n)))
    keep = means >= -thresh                       # strategies allowed to set the benchmark

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_idx(n, n_boot, MEAN_BLOCK, rng)
    boot_means = np.array([F[i].mean(axis=0) for i in idx])
    g = np.where(keep, means, 0.0)
    Z = (boot_means - g) / ses

    v_obs = max(0.0, float(tstats.max()))
    v_boot = np.maximum(Z.max(axis=1), 0.0)
    return float((v_boot >= v_obs).mean())


def romano_wolf(F, alpha=ALPHA, n_boot=N_BOOT, seed=0):
    """Stepdown multiple test. Returns a boolean accept per strategy, controlling FWER.

    Round 1 asks whether the best of all N clears the max-statistic critical value. Every
    strategy that clears it is accepted and REMOVED, and round 2 recomputes the critical
    value over what is left. Removing the winners lowers the bar for the rest, which is why
    stepdown recovers weak-but-real signal that single-step methods lose.
    """
    _, _, tstats = _stats(F)
    Z = bootstrap_matrix(F, n_boot, seed=seed)
    n_str = F.shape[1]

    accept = np.zeros(n_str, dtype=bool)
    active = np.ones(n_str, dtype=bool)
    for _ in range(n_str):
        if not active.any():
            break
        crit = np.quantile(np.maximum(Z[:, active].max(axis=1), 0.0), 1.0 - alpha)
        newly = active & (tstats > crit)
        if not newly.any():
            break
        accept |= newly
        active &= ~newly
    return accept, tstats


def books(returns, vendors):
    """Ensemble P&L matrix, shape (n_days, n_vendors), and the vendor names."""
    fwd = returns[1:]
    names = list(vendors)
    F = np.column_stack([(ensemble_weights(vendors[n]) * fwd).sum(axis=1) for n in names])
    return F, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=0)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    if args.sweep:
        rows = []
        for seed in range(args.sweep):
            u = build(seed)
            F, names = books(u["returns"], u["vendors"])
            acc, _ = romano_wolf(F, n_boot=args.boot, seed=seed)
            ics = np.array(u["true_ics"])
            rows.append({
                "seed": seed,
                "fp": int((acc & (ics == 0)).sum()),
                "fn": int((~acc & (ics > 0)).sum()),
                "n": int(acc.sum()),
                "weak_found": bool(acc[4]),
                "rc_p": reality_check(F, args.boot, seed),
                "spa_p": hansen_spa(F, args.boot, seed),
            })
            print(f"  seed {seed + 1}/{args.sweep}", flush=True)
        s = pd.DataFrame(rows)
        Path("results").mkdir(exist_ok=True)
        s.to_csv("results/reality_check_sweep.csv", index=False)
        print(f"\nROMANO-WOLF STEPDOWN   ({args.sweep} seeds)")
        print(f"  bought                 {s['n'].mean():.2f}")
        print(f"  false positives        {s['fp'].mean():.2f}   (>=1 in {(s['fp'] > 0).mean():.1%} of seeds)")
        print(f"  missed real            {s['fn'].mean():.2f}")
        print(f"  weak vendor found      {s['weak_found'].mean():.1%}")
        return

    returns = np.load("data/returns.npy")
    vendors = dict(np.load("data/vendors.npz"))
    key = pd.read_csv("data/answer_key.csv")

    F, names = books(returns, vendors)
    acc, tstats = romano_wolf(F, n_boot=args.boot)

    df = pd.DataFrame({
        "vendor": names,
        "sharpe": F.mean(axis=0) / F.std(axis=0) * np.sqrt(TRADING_DAYS),
        "t_stat": tstats,
        "accept_rw": acc,
    }).merge(key, on="vendor", how="left")              # DISPLAY ONLY
    df["kind"] = np.where(df["true_ic"] > 0, "REAL", "fake")
    df = df.sort_values("t_stat", ascending=False)

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/reality_check.csv", index=False)

    print(f"REALITY CHECK   ({args.boot} stationary bootstraps, mean block {MEAN_BLOCK} days)")
    print(f"  White's Reality Check p = {reality_check(F, args.boot):.4f}"
          "   (H0: the BEST vendor has no edge)")
    print(f"  Hansen's SPA p          = {hansen_spa(F, args.boot):.4f}"
          "   (same, with hopeless vendors excluded from the benchmark)")
    print("\nROMANO-WOLF STEPDOWN   (per-vendor, FWER controlled at 5%)")
    print(df[["vendor", "kind", "true_ic", "sharpe", "t_stat", "accept_rw"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    fp = int((df["accept_rw"] & (df["true_ic"] == 0)).sum())
    fn = int((~df["accept_rw"] & (df["true_ic"] > 0)).sum())
    print(f"\n  bought {int(df['accept_rw'].sum())}   false positives {fp}   missed real {fn}")


if __name__ == "__main__":
    main()