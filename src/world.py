"""AltSignal - the unified world. Fundamentals, backfill and a realistic shortlist in one place.

Until now the project had two separate synthetic worlds. src/generate.py built 30 vendors with
backfill and ran the full gate pipeline, but its vendors were degraded copies of RETURNS.
src/fundamentals.py built a latent quarterly fundamental and showed that scoring against it
separates vendors 55x better - but only on its own 14-vendor toy world, with no backfill and no
pipeline.

Neither is enough to answer the question the project now cares about: does replacing the
significance gate with a fundamentals gate actually move the precision/recall curve, or just
slide along it? That needs one world with everything in it.

WHAT THIS GENERATES

  A latent quarterly fundamental per firm (revenue growth), persistent across quarters. This is
  the thing vendors claim to measure, and it is DISCLOSED at the announcement - which is what
  makes it a usable answer key on real data, where returns give you none.

  A sell-side consensus: a good but imperfect forecast, published before the print. Whatever
  consensus already knows is in the price, so only the SURPRISE can move anything.

  Returns that jump on the announcement day in proportion to the surprise, plus ordinary market
  and idiosyncratic noise the rest of the time.

  Thirty vendors of four kinds:
      genuine  nowcast built from the latent fundamental - real, tradeable information
      mirror   nowcast built from CONSENSUS - highly accurate, adds nothing, worth nothing.
               Without these, accuracy and alpha would be nearly the same thing and the
               fundamentals gate would look far more powerful than it is.
      junk     pure noise
      backfill any of the above may additionally have reconstructed pre-launch history

  A daily trading signal per vendor: its nowcast minus consensus, held through the quarter and
  settled at the print. This is the vendor's view of what is unpriced.

Everything a gate is allowed to see is returned separately from everything it is not. The
answer key (kinds, true accuracy) is for scoring only; live_start is disclosed, because in
reality the vendor tells you when they began collecting.

Run:  python -m src.world
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_ASSETS = 400
N_DAYS = 1500
N_QUARTERS = 24
QUARTER_LEN = N_DAYS // N_QUARTERS
REPORT_LAG = 15

CONSENSUS_ACC = 0.70        # correlation between consensus and the truth
FUND_PHI = 0.60             # quarter-to-quarter persistence of growth
JUMP = 0.030                # return move per 1sd of surprise, on the announcement day
MKT_VOL, IDIO_VOL = 0.010, 0.012

# Five genuine vendors, spanning strong to nearly invisible. Two mirrors - accurate and
# useless. Twenty-three junk. A 1-in-6 base rate, matching a realistic shortlist.
GENUINE_ACC = [0.55, 0.40, 0.30, 0.22, 0.15]
MIRROR_ACC = [0.95, 0.80]
N_JUNK = 23

BACKFILL_ACC = float(os.environ.get("ALTSIGNAL_BACKFILLACC", "0.75"))
N_BACKFILL = int(os.environ.get("ALTSIGNAL_NBACKFILL", "8"))
LIVE_LO, LIVE_HI = 0.45, 0.75


def _std_rows(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


def build(seed=SEED):
    """One complete world. Pure: no files, no printing."""
    rng = np.random.default_rng(seed)

    # --- latent quarterly fundamental ---
    g = np.empty((N_QUARTERS, N_ASSETS))
    g[0] = rng.standard_normal(N_ASSETS)
    for q in range(1, N_QUARTERS):
        g[q] = FUND_PHI * g[q - 1] + np.sqrt(1 - FUND_PHI ** 2) * rng.standard_normal(N_ASSETS)
    g = _std_rows(g)

    # --- consensus, and the surprise that is all a price can react to ---
    c = _std_rows(CONSENSUS_ACC * g
                  + np.sqrt(1 - CONSENSUS_ACC ** 2) * rng.standard_normal(g.shape))
    surprise = _std_rows(g - c)

    # --- returns ---
    market = MKT_VOL * rng.standard_t(4, N_DAYS) / np.sqrt(2)
    beta = rng.normal(1.0, 0.3, N_ASSETS)
    returns = (market[:, None] * beta[None, :]
               + IDIO_VOL * rng.standard_t(4, (N_DAYS, N_ASSETS)) / np.sqrt(2))

    announce = {}
    for q in range(N_QUARTERS):
        day = (q + 1) * QUARTER_LEN + REPORT_LAG
        if day < N_DAYS:
            announce[q] = day
            returns[day] += JUMP * surprise[q]

    # --- vendors ---
    spec = ([("genuine", a) for a in GENUINE_ACC]
            + [("mirror", a) for a in MIRROR_ACC]
            + [("junk", 0.0)] * N_JUNK)

    def nowcast(kind, acc):
        if kind == "genuine":
            base = g
        elif kind == "mirror":
            base = c                       # tracks consensus: accurate, and adds nothing
        else:
            return _std_rows(rng.standard_normal(g.shape))
        return _std_rows(acc * base + np.sqrt(1 - acc ** 2) * rng.standard_normal(g.shape))

    backfilled = set(rng.choice(len(spec), N_BACKFILL, replace=False).tolist())

    def to_daily(n):
        """Vendor view (nowcast minus consensus) held through the quarter, settled at print."""
        view = _std_rows(n - c)
        sig = np.zeros((N_DAYS - 1, N_ASSETS))
        for q, day in announce.items():
            sig[q * QUARTER_LEN:min(day, N_DAYS - 1)] = view[q]
        return sig

    nowcasts, signals, kinds, accs, live_starts = {}, {}, {}, {}, {}
    for i, (kind, acc) in enumerate(spec):
        name = f"vendor_{i:02d}"
        n = nowcast(kind, acc)

        if i in backfilled:
            # reconstructed history: built later, by people who knew how it turned out
            q_live = int(rng.integers(int(LIVE_LO * N_QUARTERS), int(LIVE_HI * N_QUARTERS)))
            fake = _std_rows(BACKFILL_ACC * g
                             + np.sqrt(1 - BACKFILL_ACC ** 2) * rng.standard_normal(g.shape))
            n = n.copy()
            n[:q_live] = fake[:q_live]
            live_starts[name] = q_live * QUARTER_LEN
        else:
            live_starts[name] = -1

        nowcasts[name], signals[name] = n, to_daily(n)
        kinds[name], accs[name] = kind, acc

    return {"returns": returns, "vendors": signals, "nowcasts": nowcasts,
            "g": g, "c": c, "surprise": surprise, "announce": announce,
            "kinds": kinds, "true_acc": accs, "live_starts": live_starts,
            "quarter_len": QUARTER_LEN, "seed": seed}


def main():
    u = build()
    out = Path("data")
    out.mkdir(exist_ok=True)
    np.save(out / "w_returns.npy", u["returns"])
    np.savez_compressed(out / "w_vendors.npz", **u["vendors"])
    np.savez_compressed(out / "w_nowcasts.npz", **u["nowcasts"])
    np.savez_compressed(out / "w_truth.npz", g=u["g"], c=u["c"], surprise=u["surprise"])
    pd.DataFrame({
        "vendor": list(u["vendors"]),
        "kind": [u["kinds"][n] for n in u["vendors"]],
        "true_acc": [u["true_acc"][n] for n in u["vendors"]],
        "live_start": [u["live_starts"][n] for n in u["vendors"]],
    }).to_csv(out / "w_key.csv", index=False)

    n_real = sum(k == "genuine" for k in u["kinds"].values())
    n_mir = sum(k == "mirror" for k in u["kinds"].values())
    print(f"UNIFIED WORLD   (seed {u['seed']}, {len(u['vendors'])} vendors: "
          f"{n_real} genuine, {n_mir} mirror, {len(u['vendors']) - n_real - n_mir} junk)")
    print(f"  {N_QUARTERS} quarters x {N_ASSETS} firms = {N_QUARTERS * N_ASSETS} firm-quarters")
    print(f"  consensus vs truth: R2 = {np.corrcoef(u['g'].ravel(), u['c'].ravel())[0,1]**2:.3f}")
    print(f"  announcement move per 1sd surprise: {JUMP:.1%}")
    print(f"  backfilled vendors: {sum(v > 0 for v in u['live_starts'].values())}")
    print(f"  annualised vol: {u['returns'].std() * np.sqrt(252):.1%}")


if __name__ == "__main__":
    main()