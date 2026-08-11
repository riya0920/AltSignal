"""AltSignal Step 1 (v2) - synthetic universe with a REALISTIC vendor shortlist.

v1 built an easy problem and the naive backtest mostly solved it: the top four vendors sorted
themselves correctly in almost every seed, and the entire 53% failure rate came from one
vendor (IC 0.010) colliding with the nulls. PBO said the same thing more sharply - 0.000
across all twelve candidates, 0.598 restricted to the seven nulls. The easy cases were easy.

Three changes make the shortlist look like one a fund would actually receive.

  1. BASE RATE. v1 had 5 real out of 12 (42%). Real alternative-data shortlists run closer to
     1 in 6 or worse. This matters more than it sounds: with 25 nulls competing instead of 7,
     far more of them get lucky, so even an accurate test approves mostly junk. There are
     simply more ways to be wrong.

  2. COMPRESSED ICs. v1 spanned 0.010 to 0.050 - a fivefold range, so the strong vendors won
     under any method. v2 spans 0.010 to 0.030, all inside the band where v1's hard case
     lived. No free wins; every vendor is now a hard case.

  3. PER-VENDOR DECAY. v1 gave every factor the same persistence, so every vendor's signal
     decayed identically and decay.py's half-life column was noise. v2 gives each style factor
     its own persistence and points each real vendor at a different primary factor. A vendor
     reading a fast-mean-reverting factor now genuinely decays faster than one reading a slow
     factor - which creates real turnover and cost differences between vendors with IDENTICAL
     IC. That is the most common way real vendors differ in value.

Expect every headline number from v1 to get worse. That is the point: a harness that only
works on easy problems has not been tested.

Run:  python -m src.generate      (from the repo root)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_ASSETS = 400
N_DAYS = 1500

# --- the shortlist ---------------------------------------------------------------------
# Five real vendors, twenty-five nulls: a 1-in-6 base rate. ICs compressed into 0.010-0.030,
# the range where telling real from lucky is genuinely hard. v1's 0.050 and 0.040 are gone.
REAL_ICS = [0.030, 0.025, 0.020, 0.015, 0.010]
N_NULLS = int(os.environ.get("ALTSIGNAL_NNULLS", "25"))
TRUE_ICS = REAL_ICS + [0.0] * N_NULLS
N_VENDORS = len(TRUE_ICS)

# --- generator parameters ---------------------------------------------------------------
N_STYLE = int(os.environ.get("ALTSIGNAL_NSTYLE", "6"))
STYLE_VOL = float(os.environ.get("ALTSIGNAL_STYLEVOL", "0.0055"))
IDIO_VOL = float(os.environ.get("ALTSIGNAL_IDIOVOL", "0.006"))
FAC_PRED = float(os.environ.get("ALTSIGNAL_FACPRED", "0.10"))
NOISE_PHI = float(os.environ.get("ALTSIGNAL_NOISEPHI", "0.6"))
JUNK_FRAC = float(os.environ.get("ALTSIGNAL_JUNKFRAC", "0.15"))

# Per-factor persistence. v1 used one shared value; these span fast mean-reversion to slow
# trend, mirroring real style factors (short-term reversal turns over in days, value in
# months). A vendor's decay rate is inherited from whichever factor it primarily reads.
PHI_LO = float(os.environ.get("ALTSIGNAL_PHILO", "0.55"))
PHI_HI = float(os.environ.get("ALTSIGNAL_PHIHI", "0.97"))

# How concentrated a real vendor is on its primary factor. The remainder spreads over the
# others, which keeps effective breadth above 1 so Sharpe stays in a realistic band while
# still giving each vendor a distinct decay profile.
PRIMARY_W = float(os.environ.get("ALTSIGNAL_PRIMARYW", "0.75"))

# --- backfill (reconstructed history) ---------------------------------------------------
# When a vendor sells "10 years of history", they typically only began COLLECTING a few years
# ago. The earlier portion is reconstructed after the fact, by people who already knew how the
# period turned out. It is a memoir, not a diary, and it flatters the vendor enormously.
#
# Some vendors here are given a hidden live-start day. Before it, their data carries a strong
# injected signal (BACKFILL_IC). After it, they revert to whatever their true skill is. The
# live-start date is DISCLOSED in the answer key, because in reality the vendor tells you - it
# is on the data sheet. The test is then simply to score the two halves separately.
#
# One REAL vendor is given backfill too. Otherwise "has backfill" would be identical to "is
# fake", and the test would look far more powerful than it is. Reconstructed history and
# worthless data are two different problems, and the test only detects the first.
BACKFILL_IC = float(os.environ.get("ALTSIGNAL_BACKFILLIC", "0.05"))
N_BACKFILL_NULLS = int(os.environ.get("ALTSIGNAL_NBACKFILL", "8"))
BACKFILL_REAL_IDX = 2                    # vendor_02, true IC 0.020
LIVE_START_LO, LIVE_START_HI = 0.50, 0.80    # as a fraction of the history


def _std_rows(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


def _mean_ic(a, b):
    """Average daily cross-sectional correlation between two standardised panels."""
    return float((a * b).sum(axis=1).mean() / a.shape[1])


def build(seed=SEED):
    """Generate one synthetic universe and its vendor shortlist. Pure: no files, no printing."""
    rng = np.random.default_rng(seed)

    def fat(size):
        """Student-t(4) rescaled to unit variance. Same middle as a normal, far heavier ends."""
        return rng.standard_t(4, size=size) / np.sqrt(4 / (4 - 2))

    # --- market volatility, sticky (AR(1)) ---
    mkt_vol = np.zeros(N_DAYS)
    mkt_vol[0] = 0.010
    for t in range(1, N_DAYS):
        mkt_vol[t] = 0.95 * mkt_vol[t - 1] + 0.05 * 0.010 + 0.002 * rng.normal()
    mkt_vol = np.abs(mkt_vol)

    market = mkt_vol * fat(N_DAYS)
    beta = rng.normal(1.0, 0.3, N_ASSETS)
    style_load = rng.normal(0.0, 1.0, (N_ASSETS, N_STYLE))

    # --- factor scores, each with its OWN persistence ---
    phis = np.linspace(PHI_HI, PHI_LO, N_STYLE)      # factor 0 slowest, last one fastest
    score = np.empty((N_DAYS, N_STYLE))
    score[0] = rng.standard_normal(N_STYLE)
    for t in range(1, N_DAYS):
        score[t] = phis * score[t - 1] + np.sqrt(1 - phis ** 2) * rng.standard_normal(N_STYLE)
    shock = rng.standard_normal((N_DAYS, N_STYLE))

    fac_ret = np.empty((N_DAYS, N_STYLE))
    fac_ret[0] = shock[0]
    fac_ret[1:] = FAC_PRED * score[:-1] + np.sqrt(1 - FAC_PRED ** 2) * shock[1:]
    style = (STYLE_VOL * fac_ret) @ style_load.T

    idio = IDIO_VOL * fat((N_DAYS, N_ASSETS))
    returns = market[:, None] * beta[None, :] + style + idio
    z = _std_rows(returns)
    z_fwd = z[1:]

    def vendor_alpha(primary):
        """The forecastable direction a vendor reads, concentrated on one style factor.

        Weighting toward a single factor is what gives each vendor its own decay rate: the
        signal inherits that factor's persistence. The remaining weight keeps breadth above
        one so the resulting Sharpe stays realistic.
        """
        w = np.full(N_STYLE, (1.0 - PRIMARY_W) / (N_STYLE - 1))
        w[primary] = PRIMARY_W
        a = (FAC_PRED * score[:-1] * w) @ style_load.T
        return _std_rows(a)

    def ar1_fat(n_rows, n_cols, phi):
        x = np.empty((n_rows, n_cols))
        x[0] = fat(n_cols)
        for t in range(1, n_rows):
            x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * fat(n_cols)
        return x

    def style_noise():
        """Persistent fat-tailed error living in the same style subspace as the signal."""
        in_style = ar1_fat(N_DAYS - 1, N_STYLE, NOISE_PHI) @ style_load.T
        idio_err = fat((N_DAYS - 1, N_ASSETS))
        e = np.sqrt(1 - JUNK_FRAC) * _std_rows(in_style) + np.sqrt(JUNK_FRAC) * idio_err
        return _std_rows(e)

    def make_series(target_ic, primary):
        """One vendor series with a given injected IC. target_ic of 0 gives pure noise."""
        noise = style_noise()
        if target_ic == 0:
            # Nulls keep their raw noise. Orthogonalising them too would force in-sample IC to
            # exactly zero, cleaner than reality - a worthless vendor DOES show a spurious IC
            # by chance, and seeing through that is the whole job.
            return noise
        a = vendor_alpha(primary)
        # Project the noise off alpha before mixing: both live in the same style subspace, so a
        # raw draw is correlated with alpha, and restandardising the mixture then rescales the
        # signal component and throws realised IC off target.
        dot = (noise * a).sum(axis=1, keepdims=True)
        noise = _std_rows(noise - dot / (a * a).sum(axis=1, keepdims=True) * a)
        rho = _mean_ic(a, z_fwd)
        s = min(target_ic / rho, 1.0) if rho > 0 else 0.0
        return s * a + np.sqrt(max(1 - s ** 2, 0.0)) * noise

    # which vendors ship reconstructed history
    null_ids = np.arange(len(REAL_ICS), N_VENDORS)
    backfilled = set(rng.choice(null_ids, N_BACKFILL_NULLS, replace=False).tolist())
    backfilled.add(BACKFILL_REAL_IDX)

    n_rows = N_DAYS - 1
    vendors, primaries, live_starts = {}, {}, {}
    for i, target in enumerate(TRUE_ICS):
        name = f"vendor_{i:02d}"
        primary = i % N_STYLE
        series = make_series(target, primary)          # the vendor's TRUE behaviour

        if i in backfilled:
            ls = int(rng.integers(int(LIVE_START_LO * n_rows), int(LIVE_START_HI * n_rows)))
            # everything before the live start is rebuilt with hindsight, so it looks great
            series = series.copy()
            series[:ls] = make_series(BACKFILL_IC, primary)[:ls]
        else:
            ls = -1

        vendors[name] = series
        primaries[name] = primary if target > 0 else -1
        live_starts[name] = ls

    return {
        "returns": returns,
        "vendors": vendors,
        "true_ics": TRUE_ICS,
        "live_starts": live_starts,
        "z": z,
        "phis": phis,
        "primaries": primaries,
        "seed": seed,
    }


def sanity_block(u):
    returns, vendors, z = u["returns"], u["vendors"], u["z"]
    z_fwd = z[1:]

    daily_vol = returns.std(axis=1)
    roll20 = pd.Series(daily_vol).rolling(20).mean()
    sample_corr = np.corrcoef(returns[:, :50].T)[np.triu_indices(50, 1)].mean()

    print(f"STEP 1 SANITY CHECKS   (seed = {u['seed']}, {N_VENDORS} vendors, "
          f"{len(REAL_ICS)} real = {len(REAL_ICS) / N_VENDORS:.0%} base rate)")
    print(f"  panel-wide daily vol   {returns.std():.4f}   (want ~0.018-0.022)")
    print(f"  annualised vol         {returns.std() * np.sqrt(252):.1%}   (want ~28-35%)")
    print(f"  mean pairwise corr     {sample_corr:.3f}   (want ~0.25-0.35)")
    print(f"  calm/wild spread       {roll20.max() / roll20.min():.2f}x")
    print(f"  factor persistences    {np.round(u['phis'], 2)}   (slow -> fast)")

    print("\nREALISED IC vs TRUTH   (real vendors only; nulls checked in aggregate)")
    null_ics = []
    for (name, v), truth in zip(vendors.items(), u["true_ics"]):
        ic = _mean_ic(_std_rows(v), z_fwd)
        if truth == 0:
            null_ics.append(ic)
        else:
            print(f"  {name}  true {truth:5.3f}   realised {ic:+.4f}"
                  f"   primary factor {u['primaries'][name]}"
                  f" (phi {u['phis'][u['primaries'][name]]:.2f})"
                  f"   live_start {u['live_starts'][name]}")
    null_ics = np.array(null_ics)
    print(f"  {len(null_ics)} nulls: mean {null_ics.mean():+.4f}, "
          f"max |ic| {np.abs(null_ics).max():.4f}   (want near zero)")


def main():
    u = build(seed=SEED)

    out = Path("data")
    out.mkdir(exist_ok=True)
    np.save(out / "returns.npy", u["returns"])
    np.savez_compressed(out / "vendors.npz", **u["vendors"])
    pd.DataFrame({
        "vendor": list(u["vendors"]),
        "true_ic": u["true_ics"],
        # DISCLOSED, not secret: in reality the vendor tells you when they began collecting.
        # -1 means the entire history is genuine.
        "live_start": [u["live_starts"][n] for n in u["vendors"]],
    }).to_csv(out / "answer_key.csv", index=False)

    sanity_block(u)


if __name__ == "__main__":
    main()