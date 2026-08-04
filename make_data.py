"""AltSignal Step 1 — synthetic universe + 12 candidate vendor datasets."""

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n_assets, n_days = 400, 1500


def fat(size):
    """Student-t draws, rescaled to unit variance. Fat tails."""
    return rng.standard_t(4, size=size) / np.sqrt(2)


# --- market volatility: sticky (clustering) ---
mkt_vol = np.zeros(n_days)
mkt_vol[0] = 0.01
for t in range(1, n_days):
    mkt_vol[t] = 0.95 * mkt_vol[t - 1] + 0.05 * 0.01 + 0.002 * rng.normal()
mkt_vol = np.abs(mkt_vol)

# --- returns: shared market factor + stock-specific noise ---
market = mkt_vol * fat(n_days)
beta = rng.normal(1.0, 0.3, n_assets)
idio = 0.015 * fat((n_days, n_assets))
returns = market[:, None] * beta[None, :] + idio

# --- cross-sectional standardisation (rank stocks against each other, per day) ---
z = (returns - returns.mean(1, keepdims=True)) / returns.std(1, keepdims=True)


def make_vendor(ic):
    """Vendor value on day t, built to predict the return on day t+1."""
    noise = rng.normal(0, 1, size=(n_days - 1, n_assets))
    return ic * z[1:] + np.sqrt(1 - ic ** 2) * noise


true_ics = [0.05, 0.04, 0.03, 0.025, 0.01, 0, 0, 0, 0, 0, 0, 0]
vendors = {f"vendor_{i:02d}": make_vendor(ic) for i, ic in enumerate(true_ics)}

# --- save ---
out = Path("data")
out.mkdir(exist_ok=True)
np.save(out / "returns.npy", returns)
np.savez_compressed(out / "vendors.npz", **vendors)
pd.DataFrame({"vendor": list(vendors), "true_ic": true_ics}).to_csv(
    out / "answer_key.csv", index=False
)

# --- sanity checks ---
daily_vol = returns.std(1)
print("SANITY CHECKS")
print(f"  avg daily vol      {returns.std():.4f}   (want ~0.02)")
print(f"  annualised vol     {returns.std()*np.sqrt(252):.1%}")
print(f"  calmest 20d window {pd.Series(daily_vol).rolling(20).mean().min():.4f}")
print(f"  wildest 20d window {pd.Series(daily_vol).rolling(20).mean().max():.4f}")
print(f"  worst single day   {returns.mean(1).min():.2%}  (market-wide)")
print(f"  avg pairwise corr  {np.corrcoef(returns[:, :50].T)[np.triu_indices(50,1)].mean():.3f}")

print("\nREALISED IC vs TRUTH")
rows = []
for (name, v), truth in zip(vendors.items(), true_ics):
    ic = np.mean([np.corrcoef(v[k], z[k + 1])[0, 1] for k in range(n_days - 1)])
    rows.append((name, truth, ic))
    print(f"  {name}  true {truth:5.3f}   realised {ic:+.4f}")

print("\nhighest realised IC among the SEVEN fakes: "
      f"{max(abs(r[2]) for r in rows if r[1] == 0):.4f}")
