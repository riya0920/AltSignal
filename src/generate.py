"""AltSignal Step 1 (v2) - synthetic universe with a REALISTIC vendor shortlist and delivery lag.

Run:  python -m src.generate
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_ASSETS = 400
N_DAYS = 1500

REAL_ICS = [0.030, 0.025, 0.020, 0.015, 0.010]
N_NULLS = int(os.environ.get("ALTSIGNAL_NNULLS", "25"))
TRUE_ICS = REAL_ICS + [0.0] * N_NULLS
N_VENDORS = len(TRUE_ICS)

N_STYLE = int(os.environ.get("ALTSIGNAL_NSTYLE", "6"))
STYLE_VOL = float(os.environ.get("ALTSIGNAL_STYLEVOL", "0.0055"))
IDIO_VOL = float(os.environ.get("ALTSIGNAL_IDIOVOL", "0.006"))
FAC_PRED = float(os.environ.get("ALTSIGNAL_FACPRED", "0.10"))
NOISE_PHI = float(os.environ.get("ALTSIGNAL_NOISEPHI", "0.6"))
JUNK_FRAC = float(os.environ.get("ALTSIGNAL_JUNKFRAC", "0.15"))

PHI_LO = float(os.environ.get("ALTSIGNAL_PHILO", "0.55"))
PHI_HI = float(os.environ.get("ALTSIGNAL_PHIHI", "0.97"))
PRIMARY_W = float(os.environ.get("ALTSIGNAL_PRIMARYW", "0.75"))

BACKFILL_IC = float(os.environ.get("ALTSIGNAL_BACKFILLIC", "0.05"))
N_BACKFILL_NULLS = int(os.environ.get("ALTSIGNAL_NBACKFILL", "8"))
BACKFILL_REAL_IDX = 2
LIVE_START_LO, LIVE_START_HI = 0.50, 0.80

LAG_MAX = int(os.environ.get("ALTSIGNAL_LAGMAX", "4"))


def _std_rows(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


def _mean_ic(a, b):
    return float((a * b).sum(axis=1).mean() / a.shape[1])


def build(seed=SEED):
    rng = np.random.default_rng(seed)

    def fat(size):
        return rng.standard_t(4, size=size) / np.sqrt(4 / (4 - 2))

    mkt_vol = np.zeros(N_DAYS)
    mkt_vol[0] = 0.010
    for t in range(1, N_DAYS):
        mkt_vol[t] = 0.95 * mkt_vol[t - 1] + 0.05 * 0.010 + 0.002 * rng.normal()
    mkt_vol = np.abs(mkt_vol)

    market = mkt_vol * fat(N_DAYS)
    beta = rng.normal(1.0, 0.3, N_ASSETS)
    style_load = rng.normal(0.0, 1.0, (N_ASSETS, N_STYLE))

    phis = np.linspace(PHI_HI, PHI_LO, N_STYLE)
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
        in_style = ar1_fat(N_DAYS - 1, N_STYLE, NOISE_PHI) @ style_load.T
        idio_err = fat((N_DAYS - 1, N_ASSETS))
        e = np.sqrt(1 - JUNK_FRAC) * _std_rows(in_style) + np.sqrt(JUNK_FRAC) * idio_err
        return _std_rows(e)

    def make_series(target_ic, primary):
        noise = style_noise()
        if target_ic == 0:
            return noise
        a = vendor_alpha(primary)
        dot = (noise * a).sum(axis=1, keepdims=True)
        noise = _std_rows(noise - dot / (a * a).sum(axis=1, keepdims=True) * a)
        rho = _mean_ic(a, z_fwd)
        s = min(target_ic / rho, 1.0) if rho > 0 else 0.0
        return s * a + np.sqrt(max(1 - s ** 2, 0.0)) * noise

    null_ids = np.arange(len(REAL_ICS), N_VENDORS)
    backfilled = set(rng.choice(null_ids, N_BACKFILL_NULLS, replace=False).tolist())
    backfilled.add(BACKFILL_REAL_IDX)

    n_rows = N_DAYS - 1
    vendors, primaries, live_starts, lags = {}, {}, {}, {}
    for i, target in enumerate(TRUE_ICS):
        name = f"vendor_{i:02d}"
        primary = i % N_STYLE
        series = make_series(target, primary)

        if i in backfilled:
            ls = int(rng.integers(int(LIVE_START_LO * n_rows), int(LIVE_START_HI * n_rows)))
            series = series.copy()
            series[:ls] = make_series(BACKFILL_IC, primary)[:ls]
        else:
            ls = -1

        lag_days = int(rng.integers(1, LAG_MAX + 1)) if target > 0 else int(rng.integers(0, LAG_MAX + 1))

        vendors[name] = series
        primaries[name] = primary if target > 0 else -1
        live_starts[name] = ls
        lags[name] = lag_days

    return {
        "returns": returns,
        "vendors": vendors,
        "true_ics": TRUE_ICS,
        "live_starts": live_starts,
        "lags": lags,
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
                  f"   live_start {u['live_starts'][name]}"
                  f"   lag {u['lags'][name]}d")
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
        "live_start": [u["live_starts"][n] for n in u["vendors"]],
        "lag_days": [u["lags"][n] for n in u["vendors"]],
    }).to_csv(out / "answer_key.csv", index=False)

    sanity_block(u)


if __name__ == "__main__":
    main()