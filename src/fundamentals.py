"""AltSignal - the fundamentals layer. Scoring vendors against what they actually measure.

Every version of this project so far built vendors as degraded copies of RETURNS, and that
framing produced conclusion 8: below IC 0.02, a vendor is unverifiable, because t = Sharpe x
sqrt(years) and the years required run to decades. That conclusion is true, but only of
return-based testing. It is not true in general, and this module is the counter-example.

Real vendors do not claim to predict returns. A credit-card panel claims to measure REVENUE.
Revenue is disclosed every quarter, exactly, with almost no measurement noise. So instead of
one low-powered question you get three separate high-powered ones:

    accuracy      does the vendor's nowcast track the revenue the firm later reported?
    incremental   does it beat what sell-side consensus already predicted?
    alpha         does that edge survive into tradeable P&L?

The sample sizes are not comparable. Returns give 1500 noisy daily observations of a signal
buried under market and idiosyncratic noise. Revenue gives 24 quarters x 400 firms = 9600
clean observations of the thing being measured. That is the loophole the industry lives in.

THE TRAP THIS MODULE IS BUILT TO EXPOSE
High accuracy does not imply alpha. A vendor can nowcast revenue almost perfectly and be
worthless, because consensus already knew. So the vendor set here deliberately includes
MIRROR vendors: they track reported revenue beautifully (high accuracy R^2) and carry nothing
consensus did not already have (zero incremental R^2, zero alpha). If a harness cannot tell a
mirror from a genuine forecaster, the accuracy test has replaced one blind spot with another.

    genuine   nowcast built from the latent fundamental, independent of consensus
    mirror    nowcast built from CONSENSUS - accurate, and useless
    junk      pure noise

Run:  python -m src.fundamentals
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.naive import CONFIGS, smooth, top_k, weights

SEED = 42
N_ASSETS = 400
N_DAYS = 1500
N_QUARTERS = 24
QUARTER_LEN = N_DAYS // N_QUARTERS          # ~62 trading days
REPORT_LAG = 15                             # days after quarter end that results are announced

CONSENSUS_ACC = 0.70        # correlation between sell-side consensus and the true fundamental
FUND_PHI = 0.60             # quarter-to-quarter persistence of a firm's growth
JUMP = 0.030                # return move per 1 sd of earnings surprise on announcement day
MKT_VOL, IDIO_VOL = 0.010, 0.012
TRADING_DAYS = 252

# vendor_name -> (kind, parameter)
#   genuine: parameter is measurement accuracy against the latent fundamental
#   mirror:  parameter is how tightly it tracks consensus (accurate, adds nothing)
#   junk:    no information at all
VENDOR_SPEC = [
    ("vendor_00", "genuine", 0.60),
    ("vendor_01", "genuine", 0.40),
    ("vendor_02", "genuine", 0.25),
    ("vendor_03", "genuine", 0.15),
    ("vendor_04", "mirror", 0.95),
    ("vendor_05", "mirror", 0.80),
] + [(f"vendor_{i:02d}", "junk", 0.0) for i in range(6, 14)]


def _std_rows(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


def build(seed=SEED):
    """Generate the fundamental layer, returns driven by earnings surprise, and the vendors."""
    rng = np.random.default_rng(seed)

    # --- latent quarterly fundamental: revenue growth, persistent across quarters ---
    g = np.empty((N_QUARTERS, N_ASSETS))
    g[0] = rng.standard_normal(N_ASSETS)
    for q in range(1, N_QUARTERS):
        g[q] = FUND_PHI * g[q - 1] + np.sqrt(1 - FUND_PHI ** 2) * rng.standard_normal(N_ASSETS)
    g = _std_rows(g)

    # --- sell-side consensus: a good but imperfect forecast, published before the print ---
    c = CONSENSUS_ACC * g + np.sqrt(1 - CONSENSUS_ACC ** 2) * rng.standard_normal(g.shape)
    c = _std_rows(c)

    # The surprise is the ONLY part of the fundamental that can move a price. Whatever
    # consensus already knew is in the price before the announcement.
    surprise = _std_rows(g - c)

    # --- returns: ordinary market and idiosyncratic noise, plus an earnings-day jump ---
    market = MKT_VOL * rng.standard_t(4, N_DAYS) / np.sqrt(2)
    beta = rng.normal(1.0, 0.3, N_ASSETS)
    returns = market[:, None] * beta[None, :] + IDIO_VOL * rng.standard_t(
        4, (N_DAYS, N_ASSETS)) / np.sqrt(2)

    announce = {}
    for q in range(N_QUARTERS):
        day = (q + 1) * QUARTER_LEN + REPORT_LAG
        if day >= N_DAYS:
            continue
        announce[q] = day
        returns[day] += JUMP * surprise[q]

    # --- vendors ---
    # A vendor's tradeable view is its nowcast MINUS consensus: the part of the surprise it
    # thinks the market has not priced. It is published daily through the quarter and settles
    # when the print lands.
    nowcasts, signals, kinds = {}, {}, {}
    for name, kind, param in VENDOR_SPEC:
        if kind == "genuine":
            n = param * g + np.sqrt(1 - param ** 2) * rng.standard_normal(g.shape)
        elif kind == "mirror":
            # tracks consensus, not the fundamental: high accuracy, no new information
            n = param * c + np.sqrt(1 - param ** 2) * rng.standard_normal(g.shape)
        else:
            n = rng.standard_normal(g.shape)
        n = _std_rows(n)
        nowcasts[name] = n
        kinds[name] = kind

        view = _std_rows(n - c)                      # what the vendor thinks is unpriced
        sig = np.zeros((N_DAYS - 1, N_ASSETS))
        for q, day in announce.items():
            start = q * QUARTER_LEN
            sig[start:min(day, N_DAYS - 1)] = view[q]     # held until the print
        signals[name] = sig

    return {"returns": returns, "vendors": signals, "nowcasts": nowcasts, "g": g, "c": c,
            "surprise": surprise, "kinds": kinds, "announce": announce, "seed": seed}


def _r2(y, x):
    """R-squared of a simple pooled regression of y on x."""
    y, x = y.ravel(), x.ravel()
    if x.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def _fit_r2(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1.0 - (y - X @ b).var() / y.var()


def _incremental_r2(g, c, n):
    """How much R-squared the nowcast ADDS to a regression that already contains consensus.

    The obvious version - correlating (actual - consensus) with (nowcast - consensus) - is
    wrong, and wrong in a way that flatters junk. Both differences contain the same -consensus
    term, so a vendor made of pure noise still scores about 0.07 purely from that shared term.
    Nesting the regressions instead removes it: fit consensus alone, then consensus plus the
    nowcast, and take the difference. Junk then scores zero, as it must.
    """
    g, c, n = g.ravel(), c.ravel(), n.ravel()
    ones = np.ones(len(g))
    base = _fit_r2(g, np.column_stack([ones, c]))
    full = _fit_r2(g, np.column_stack([ones, c, n]))
    return float(max(full - base, 0.0))


def _sharpe(pnl):
    sd = pnl.std()
    return 0.0 if sd == 0 else float(pnl.mean() / sd * np.sqrt(TRADING_DAYS))


def report(u):
    fwd = u["returns"][1:]
    g, c, surprise = u["g"], u["c"], u["surprise"]

    rows = []
    for name, sig in u["vendors"].items():
        n = u["nowcasts"][name]
        # Days after the final print carry no signal at all; their weight rows are all zero
        # and normalising them divides by zero, which propagates NaN through the whole Sharpe.
        live = np.abs(sig).sum(axis=1) > 0
        w = np.mean([weights(top_k(smooth(sig[live], s), k)) for _, s, k in CONFIGS], axis=0)
        pnl = (w * fwd[live]).sum(axis=1)
        rows.append({
            "vendor": name,
            "kind": u["kinds"][name],
            # TEST 1: does it track what the firm later reported? 9600 clean observations.
            "accuracy_r2": _r2(g, n),
            # TEST 2: does it beat what consensus already knew? The regression that matters.
            "incremental_r2": _incremental_r2(g, c, n),
            # TEST 3: does the edge survive into P&L? 1500 noisy daily observations.
            "sharpe": _sharpe(pnl),
            "t_stat": float(pnl.mean() / pnl.std() * np.sqrt(len(pnl))) if pnl.std() else 0.0,
        })
    return pd.DataFrame(rows)


def main():
    u = build()
    df = report(u)

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/fundamentals.csv", index=False)

    pd.set_option("display.width", 150)
    print(f"FUNDAMENTALS LAYER   ({N_QUARTERS} quarters x {N_ASSETS} firms = "
          f"{N_QUARTERS * N_ASSETS} firm-quarters, vs {N_DAYS} daily returns)")
    print(f"  consensus accuracy vs truth: R2 = {_r2(u['g'], u['c']):.3f}")
    print(f"  earnings-day move per 1sd surprise: {JUMP:.1%}\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n  accuracy_r2      does the nowcast track REPORTED revenue (9600 obs)")
    print("  incremental_r2   does it beat CONSENSUS - the part that can move a price")
    print("  sharpe / t_stat  does it survive into P&L (1500 noisy daily obs)")

    print("\nWHAT THE THREE TESTS DISAGREE ABOUT")
    for _, r in df[df["kind"] != "junk"].iterrows():
        verdict = ("accurate but adds nothing" if r["kind"] == "mirror"
                   else "genuine forecaster")
        print(f"  {r['vendor']:<11} {r['kind']:<8} accuracy {r['accuracy_r2']:.3f}"
              f"   incremental {r['incremental_r2']:.3f}"
              f"   t {r['t_stat']:+.2f}   -> {verdict}")

    junk = df[df["kind"] == "junk"]
    print(f"\n  8 junk vendors: max accuracy_r2 {junk['accuracy_r2'].max():.4f},"
          f" max |t| {junk['t_stat'].abs().max():.2f}")


if __name__ == "__main__":
    main()