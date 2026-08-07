"""AltSignal Step 1 - synthetic universe plus twelve candidate vendor datasets.

Everything here is fake on purpose. We generate a fake equity market first, then build
twelve vendor datasets whose true predictive power we choose ourselves. That chosen list
of ICs is the answer key. The point of synthetic data is that the truth is known, so the
harness can be validated before it is ever pointed at real data.

DEVIATIONS FROM THE LITERAL SPEC (deliberate, documented, see NOTES.md for the full
derivation):

  1. Returns get a few shared style/sector factors in addition to the market factor
     (spec 1.3 had market + idiosyncratic only).
  2. A vendor's real signal lives ONLY in that low-dimensional factor subspace, not in
     every idiosyncratic name (spec 1.5 built the signal from the full standardised
     next-day return).

Why: the spec wants real vendors at Sharpe 0.5-2.5 in Step 2. The fundamental law of
active management is IR = IC * sqrt(breadth). If a vendor predicts all 400 names
independently, breadth is ~400 per day and an IC of only 0.05 forces annual Sharpe
= 0.05 * sqrt(400 * 252) ~ 16. No value of the noise-correlation knob RHO fixes this:
correlated vendor noise only lowers the book's measured variance and nudges Sharpe UP.
The lever that actually matters is BREADTH. Real vendor data (a credit-card panel, a
satellite feed) predicts sector and style moves, not 400 independent idiosyncratic
wiggles, so confining the signal to ~K factor directions drops the effective breadth to
single digits and lands IC 0.05 near Sharpe 2. The realised cross-sectional IC is still
calibrated to match the answer key, so the spec's IC semantics are preserved.

The generator is exposed as build(seed) so that src/robustness.py can re-run the whole
pipeline across many seeds. A single seed tells you which vendors got lucky in one draw;
only the distribution across seeds is a result. main() is the thin wrapper that writes
files and prints the Step 1 sanity block.

Run:  python -m src.generate      (from the repo root)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# --- fixed parameters (see spec section "Parameters") ---
SEED = 42
N_ASSETS = 400
N_DAYS = 1500

# five real vendors, seven pure noise. vendor_04 at 0.01 is the deliberate hard case.
TRUE_ICS = [0.05, 0.04, 0.03, 0.025, 0.01, 0, 0, 0, 0, 0, 0, 0]

# ---------------------------------------------------------------------------------------
# Generator parameters. Every value below was fixed on PRINCIPLED grounds (realistic panel
# volatility, cross sectional correlation, factor breadth and factor momentum) BEFORE the
# Step 2 fake-vendor outcome was looked at, and was not touched afterward. Fitting these to
# force a particular fake-vendor pattern would be fitting the gate to the answer, which the
# spec forbids. Each is overridable by an env var only so the choices can be reproduced and
# swept, never so the run silently changes. See NOTES.md for the full sweep record.
#
# N_STYLE is the number of style/sector factors a real vendor can span. It is the effective
# breadth of the naive book, so it sets the Step 2 Sharpe level: annual Sharpe ~ IC *
# sqrt(N_STYLE * 252). Six factors put the strongest real vendor (IC 0.05) near Sharpe 2.6
# and the 0.01 hard case near 0.8, inside the spec's band, give a mean pairwise correlation
# in the middle of the target range, and keep rho_pz safely above the top IC of 0.05.
N_STYLE = int(os.environ.get("ALTSIGNAL_NSTYLE", "6"))
STYLE_VOL = float(os.environ.get("ALTSIGNAL_STYLEVOL", "0.0055"))  # style factor daily vol
IDIO_VOL = float(os.environ.get("ALTSIGNAL_IDIOVOL", "0.006"))   # pure per-stock daily vol

# How predictable each factor is one day ahead: the realized factor return is only this
# correlated with the score a vendor could have seen the day before. This is what puts
# genuine, un-hedgeable risk INSIDE the low-dimensional subspace, so a book built on a
# vendor's forecast wins only on average rather than every single day. Kept low on purpose:
# a higher value makes the edge too clean and pushes the naive Sharpes back into double digits.
FAC_PRED = float(os.environ.get("ALTSIGNAL_FACPRED", "0.05"))

# Persistence (momentum) of the factor scores. Real style and sector factors trend, and
# that trend is what lets the Step 2 smoothing configs lock onto a lucky exposure and
# manufacture apparent skill from pure noise, which is the whole point of the naive backtest.
PHI = float(os.environ.get("ALTSIGNAL_PHI", "0.92"))

# Persistence of the vendor error itself. Real vendor feeds are somewhat sticky. Kept modest:
# too much stickiness inflates the spurious in-sample IC of the pure-null vendors (see the IC
# calibration note in NOTES.md), and Step 1's trustworthy ground truth takes priority.
NOISE_PHI = float(os.environ.get("ALTSIGNAL_NOISEPHI", "0.6"))

# Fraction of a real vendor's own error that is pure idiosyncratic name-level noise; the
# rest lives in the same style subspace as its signal. A little idiosyncratic error keeps
# the vendor from being perfectly low-rank without changing the book's breadth much.
JUNK_FRAC = float(os.environ.get("ALTSIGNAL_JUNKFRAC", "0.15"))


def _std_rows(x):
    """Standardise each row (each day) to zero mean and unit cross-sectional variance."""
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


def build(seed=SEED):
    """Generate one complete synthetic universe and its twelve vendor datasets.

    Pure: touches no files and prints nothing, so it can be called in a loop over seeds.
    Returns a dict rather than a bare tuple so callers take only what they need and the
    diagnostics (z, rho_pz) travel with the data instead of being recomputed.
    """
    rng = np.random.default_rng(seed)

    def fat(size):
        """Student-t draws with 4 degrees of freedom, rescaled to unit variance.

        Real daily returns have far fatter tails than a normal allows. df=4 gives heavy
        tails; dividing by sqrt(df/(df-2)) = sqrt(2) rescales so a value of 1 still means
        one typical move.
        """
        return rng.standard_t(4, size=size) / np.sqrt(4 / (4 - 2))

    # --- 1.2 market volatility with clustering (AR(1), sticky) ---
    mkt_vol = np.zeros(N_DAYS)
    mkt_vol[0] = 0.010
    for t in range(1, N_DAYS):
        mkt_vol[t] = 0.95 * mkt_vol[t - 1] + 0.05 * 0.010 + 0.002 * rng.normal()
    mkt_vol = np.abs(mkt_vol)

    # --- 1.3 returns: market factor + a few style factors + small idiosyncratic ---
    # The style factors carry the ONLY forecastable structure, and even it is only partly
    # forecastable: the realized factor return on day t+1 is FAC_PRED-correlated with a
    # score that was knowable on day t, plus an independent shock. That shock is the risk a
    # vendor cannot design away, and it keeps the naive Sharpe realistic instead of infinite.
    market = mkt_vol * fat(N_DAYS)                 # one number per day, shared by all names
    beta = rng.normal(1.0, 0.3, N_ASSETS)          # each stock's sensitivity to the market

    style_load = rng.normal(0.0, 1.0, (N_ASSETS, N_STYLE))  # each stock's factor loadings

    # forecastable factor scores, persistent (AR(1)) so the factors trend rather than flicker
    score = np.empty((N_DAYS, N_STYLE))
    score[0] = rng.standard_normal(N_STYLE)
    for t in range(1, N_DAYS):
        score[t] = PHI * score[t - 1] + np.sqrt(1 - PHI ** 2) * rng.standard_normal(N_STYLE)
    shock = rng.standard_normal((N_DAYS, N_STYLE))          # independent day-t+1 shocks

    # realized factor return on day t is FAC_PRED * yesterday's score + an independent shock
    fac_ret = np.empty((N_DAYS, N_STYLE))
    fac_ret[0] = shock[0]
    fac_ret[1:] = FAC_PRED * score[:-1] + np.sqrt(1 - FAC_PRED ** 2) * shock[1:]
    style = (STYLE_VOL * fac_ret) @ style_load.T   # (N_DAYS, N_ASSETS), correlated residual

    idio = IDIO_VOL * fat((N_DAYS, N_ASSETS))      # small pure per-stock news
    returns = market[:, None] * beta[None, :] + style + idio        # (N_DAYS, N_ASSETS)

    # --- 1.4 cross-sectional standardisation (rank the 400 names against each other) ---
    z = _std_rows(returns)

    # The forecastable expected next-day return in name space: what a vendor that reads the
    # factor scores perfectly would report on day t. Rank N_STYLE (low breadth) AND only a
    # partial predictor, because the shock is not in it. This is the single quantity a real
    # vendor is competing to estimate.
    alpha = (FAC_PRED * score[:-1]) @ style_load.T          # (N_DAYS-1, N_ASSETS)
    alpha = _std_rows(alpha)

    # rho_pz: daily cross-sectional correlation between that forecastable direction and the
    # realized next-day return. It is the ceiling on a vendor's realised IC, so we scale to it.
    rho_pz = np.mean([np.corrcoef(alpha[k], z[k + 1])[0, 1] for k in range(N_DAYS - 1)])

    def ar1_fat(n_rows, n_cols, phi):
        """Persistent (AR(1)) fat-tailed factor paths, unit variance, persistence phi."""
        x = np.empty((n_rows, n_cols))
        x[0] = fat(n_cols)
        for t in range(1, n_rows):
            x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * fat(n_cols)
        return x

    def style_noise():
        """Fat-tailed error living in the SAME style subspace as the signal, plus a little
        pure idiosyncratic noise. Standardised per day to unit cross-sectional variance.

        Living in the style subspace is what makes Step 2 honest: every vendor's book bets
        the same handful of style factors (fixed breadth), so a vendor that is mostly error
        is making mostly WRONG style bets and earns a genuinely low Sharpe, while a strong
        vendor's signal dominates and earns a high one. That is what spreads the real
        vendors by IC and lets the weak ones overlap the lucky fakes.

        The style-space part is PERSISTENT (AR(1)), like real vendor data: a vendor that is
        accidentally long a trending factor stays long it, so the Step 2 smoothing configs
        can lock onto that lucky exposure and turn pure noise into an impressive in-sample
        Sharpe.
        """
        in_style = ar1_fat(N_DAYS - 1, N_STYLE, NOISE_PHI) @ style_load.T   # alpha's space
        idio_err = fat((N_DAYS - 1, N_ASSETS))                  # small pure per-stock error
        e = np.sqrt(1 - JUNK_FRAC) * _std_rows(in_style) + np.sqrt(JUNK_FRAC) * idio_err
        return _std_rows(e)

    def make_vendor(target_ic):
        """A vendor's value on day t, built to predict the return on day t+1.

        Signal and error both live in the style subspace (see style_noise), so the naive
        book always bets the same low number of style factors. s is chosen so the realised
        full-sample cross-sectional IC lands on target_ic: since alpha correlates with the
        true return at rho_pz, a signal that is a fraction s of alpha has IC ~ s * rho_pz.
        """
        noise = style_noise()
        if target_ic == 0:
            return noise                                        # pure null vendor
        s = target_ic / rho_pz                                  # fraction of signal
        return s * alpha + np.sqrt(max(1 - s ** 2, 0.0)) * noise

    vendors = {f"vendor_{i:02d}": make_vendor(ic) for i, ic in enumerate(TRUE_ICS)}

    return {
        "returns": returns,
        "vendors": vendors,
        "true_ics": TRUE_ICS,
        "z": z,
        "rho_pz": rho_pz,
        "seed": seed,
    }


def sanity_block(u):
    """Print the Step 1 acceptance criteria for one universe. All must hold."""
    returns, vendors, z = u["returns"], u["vendors"], u["z"]

    daily_vol = returns.std(axis=1)
    roll20 = pd.Series(daily_vol).rolling(20).mean()
    sample_corr = np.corrcoef(returns[:, :50].T)[np.triu_indices(50, 1)].mean()

    print(f"STEP 1 SANITY CHECKS   (seed = {u['seed']})")
    print(f"  panel-wide daily vol   {returns.std():.4f}   (want ~0.018-0.022)")
    print(f"  annualised vol         {returns.std() * np.sqrt(252):.1%}   (want ~28-35%)")
    print(f"  mean pairwise corr     {sample_corr:.3f}   (want ~0.25-0.35)")
    print(f"  calmest 20d window     {roll20.min():.4f}")
    print(f"  wildest 20d window     {roll20.max():.4f}")
    print(f"  calm/wild spread       {roll20.max() / roll20.min():.2f}x   (want > ~1.3x)")
    print(f"  worst market-wide day  {returns.mean(1).min():.2%}")
    print(f"  factor subspace K      {N_STYLE}   (rho_pz = {u['rho_pz']:.3f}, the IC ceiling)")

    print(f"\nREALISED IC vs TRUTH   (JUNK_FRAC = {JUNK_FRAC})")
    for (name, v), truth in zip(vendors.items(), u["true_ics"]):
        ic = np.mean([np.corrcoef(v[k], z[k + 1])[0, 1] for k in range(N_DAYS - 1)])
        flag = "" if truth != 0 else "   (null)"
        print(f"  {name}  true {truth:5.3f}   realised {ic:+.4f}{flag}")


def main():
    """Build the canonical seed-42 universe, write it to data/, print the sanity block."""
    u = build(seed=SEED)

    # --- 1.7 outputs ---
    out = Path("data")
    out.mkdir(exist_ok=True)
    np.save(out / "returns.npy", u["returns"])
    np.savez_compressed(out / "vendors.npz", **u["vendors"])
    pd.DataFrame({"vendor": list(u["vendors"]), "true_ic": u["true_ics"]}).to_csv(
        out / "answer_key.csv", index=False
    )

    sanity_block(u)


if __name__ == "__main__":
    main()