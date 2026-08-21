"""AltSignal - signal decay, transaction costs, and latency evaluation.

Run:  python -m src.decay
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.turnover import apply_rate
from src.naive import CONFIGS, smooth, top_k, weights

HORIZONS = [1, 2, 3, 5, 10, 20, 40, 60]
COST_BPS = [0, 1, 2, 5, 10, 20, 50]
TRADING_DAYS = 252


def _std_rows(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


def apply_lag(sig: np.ndarray, lag: int = 0) -> np.ndarray:
    """Delays vendor signal availability by `lag` trading days."""
    if lag <= 0:
        return sig
    pad = np.zeros((lag, sig.shape[1]), dtype=sig.dtype)
    return np.vstack([pad, sig[:-lag]])


def ic_decay(sig, returns, horizons=HORIZONS):
    z = _std_rows(returns)
    v = _std_rows(sig)
    n_assets = sig.shape[1]
    out = {}
    for h in horizons:
        a = v[:len(v) - h + 1] if h > 1 else v
        b = z[h:h + len(a)]
        m = min(len(a), len(b))
        out[h] = float((a[:m] * b[:m]).sum(axis=1).mean() / n_assets)
    return out


def half_life(decay, horizons=HORIZONS):
    xs = [h for h in horizons if decay[h] > 0]
    if len(xs) < 3 or decay[horizons[0]] <= 0:
        return float("nan")
    ys = np.log([decay[h] for h in xs])
    slope, _ = np.polyfit(xs, ys, 1)
    return float(np.log(0.5) / slope) if slope < 0 else float("inf")


def ensemble_weights(sig):
    tgt = np.mean([weights(top_k(smooth(sig, s), k)) for _, s, k in CONFIGS], axis=0)
    return apply_rate(tgt, 0.20)


def cost_curve(sig, fwd, costs=COST_BPS):
    w = ensemble_weights(sig)
    gross = np.abs(np.diff(w, axis=0)).sum(axis=1)
    turnover = float(gross.mean())

    pnl = (w * fwd).sum(axis=1)
    out = {"turnover": turnover, "ann_turnover": turnover * TRADING_DAYS}
    for c in costs:
        charge = np.concatenate([[0.0], gross]) * (c / 10000.0)
        net = pnl - charge[:len(pnl)]
        sd = net.std()
        out[f"sharpe_{c}bps"] = 0.0 if sd == 0 else float(net.mean() / sd * np.sqrt(TRADING_DAYS))
    return out


def breakeven_bps(sig, fwd, hi=200.0):
    w = ensemble_weights(sig)
    gross = np.concatenate([[0.0], np.abs(np.diff(w, axis=0)).sum(axis=1)])
    pnl = (w * fwd).sum(axis=1)

    def net_mean(c):
        return (pnl - gross[:len(pnl)] * (c / 10000.0)).mean()

    if net_mean(0.0) <= 0:
        return 0.0
    if net_mean(hi) > 0:
        return float(hi)
    lo = 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if net_mean(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def report(returns, vendors, lags=None):
    fwd = returns[1:]
    rows = []
    for name, sig in vendors.items():
        lag = lags.get(name, 0) if lags else 0
        sig_delayed = apply_lag(sig, lag=lag)

        d = ic_decay(sig, returns)
        r = {
            "vendor": name,
            "ic_1d": d[1],
            "ic_5d": d[5],
            "ic_20d": d[20],
            "ic_60d": d[60],
            "half_life": half_life(d),
        }
        r.update(cost_curve(sig_delayed, fwd))
        r["breakeven_bps"] = breakeven_bps(sig_delayed, fwd)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    returns = np.load("data/returns.npy")
    vendors = dict(np.load("data/vendors.npz"))
    key = pd.read_csv("data/answer_key.csv")
    lags = dict(zip(key["vendor"], key["lag_days"])) if "lag_days" in key.columns else {k: 0 for k in vendors}

    df = report(returns, vendors, lags=lags)
    df = df.merge(key, on="vendor", how="left")

    if "lag_days" not in df.columns:
        df["lag_days"] = 0
    if "true_ic" not in df.columns:
        df["true_ic"] = 0.0

    df["kind"] = np.where(df["true_ic"] > 0, "REAL", "fake")
    df = df.sort_values("true_ic", ascending=False)

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/decay.csv", index=False)

    pd.set_option("display.width", 190)
    print("SIGNAL DECAY   (IC at horizon h, cross-sectional)")
    print(df[["vendor", "kind", "true_ic", "lag_days", "ic_1d", "ic_5d", "ic_20d", "ic_60d", "half_life"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nTRANSACTION COSTS & LAG   (net Sharpe of the delayed ensemble book)")
    cols = ["vendor", "kind", "lag_days", "ann_turnover"] + [f"sharpe_{c}bps" for c in COST_BPS] \
        + ["breakeven_bps"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()