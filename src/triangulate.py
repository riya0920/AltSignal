"""AltSignal - cross-vendor triangulation. Consistency without any ground truth.

The one alt-data validation method that appears consistently in the public record is
triangulation: consolidating multiple sources and cross-checking them against each other for
consistency. The logic is simple. Two vendors measuring the same underlying reality by
different collection methods - one a card panel, one a receipt feed - should agree at the
company level. If they disagree, at least one is measuring its own collection artifacts rather
than the world.

What makes this worth building here is what it does NOT need. Gate 2 needs reported
fundamentals. The significance gate needs returns. Triangulation needs neither: it only needs
the vendors themselves. Genuine vendors correlate with each other because they are all reading
the same underlying fundamental. Junk correlates with nothing, including other junk. So the
shortlist can be ranked before a single external number is fetched.

THE STATISTIC
For each vendor, the median absolute cross-sectional correlation with every OTHER vendor,
pooled across quarters. Median rather than mean so one anomalous pair cannot carry a vendor,
and absolute so a vendor that reads the same reality with an inverted sign still counts as
agreeing.

THE TRAP THIS IS BUILT TO EXPOSE
Triangulation cannot distinguish a genuine forecaster from a mirror. Mirrors track sell-side
consensus, consensus tracks the fundamental, and genuine vendors track the fundamental - so
mirrors agree with the crowd just as strongly as real vendors do, sometimes more. A shortlist
filtered on agreement alone will buy them.

That is not a flaw in the method, it is the method's boundary, and it is the same boundary
accuracy alone has. Triangulation answers "is this vendor measuring something real?" It cannot
answer "is what it measures already priced?" - which needs consensus, and therefore an external
source.

So this belongs BEFORE gate 2 as a free pre-filter, never instead of it. The test below is
whether it separates genuine-plus-mirror from junk, and it is scored on exactly that.

Run:  python -m src.triangulate
      python -m src.triangulate --sweep 20
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.world import build


def _std_rows(x):
    return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)


def agreement_matrix(nowcasts, names, live_q=None, quarter_len=None):
    """Pairwise agreement: median |cross-sectional correlation| across quarters.

    live_q restricts each PAIR to the quarters where BOTH vendors were collecting live.
    Without it the method breaks completely, and the failure is instructive: reconstructed
    histories are all built from the same realised outcome, so backfilled vendors agree with
    each other at ~0.55 regardless of whether either has any real signal. On seed 42 the six
    highest-agreement vendors were all backfilled junk. Triangulation was measuring shared
    hindsight rather than shared reality.

    So this gate MUST run after provenance, on live data only. Agreement between two
    reconstructed histories is not evidence about the world.
    """
    z = {n: _std_rows(nowcasts[n]) for n in names}
    k = len(names)
    n_q, n_assets = z[names[0]].shape
    M = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            lo = 0 if live_q is None else max(live_q[names[i]], live_q[names[j]])
            if n_q - lo < 6:                       # too few shared live quarters to judge
                M[i, j] = M[j, i] = np.nan
                continue
            per_q = np.abs((z[names[i]][lo:] * z[names[j]][lo:]).sum(axis=1) / n_assets)
            M[i, j] = M[j, i] = float(np.median(per_q))
    return M


def analyse(u, top_k=6):
    names = list(u["nowcasts"])
    qlen = u["quarter_len"]
    live_q = {n: max(int(u["live_starts"][n]) // qlen, 0) if u["live_starts"][n] > 0 else 0
              for n in names}
    M = agreement_matrix(u["nowcasts"], names, live_q, qlen)

    rows = []
    for i, name in enumerate(names):
        others = np.delete(M[i], i)
        others = others[~np.isnan(others)]
        # Mean of the TOP-K partners, not the median of all. Only about 7 of 30 vendors carry
        # any signal, so a median over all 29 partners is dominated by junk pairs and measures
        # nothing but the noise floor. A genuine vendor's evidence is that it agrees strongly
        # with a FEW others, not weakly with everyone.
        top = np.sort(others)[-top_k:]
        rows.append({
            "vendor": name,
            "kind": u["kinds"][name],
            "agreement": float(top.mean()),
            # best single partner: a vendor with one strong match and no others is
            # suspicious - it may be measuring the same artifact as one other feed
            "best_partner": float(others.max()),
            "best_partner_name": names[int(np.nanargmax(np.where(
                np.arange(len(names)) == i, -np.inf, M[i])))],
            "n_peers_above_05": int((others > 0.05).sum()),
        })
    df = pd.DataFrame(rows).sort_values("agreement", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df, M


def summarise(df):
    real = df["kind"] == "genuine"
    mirror = df["kind"] == "mirror"
    junk = df["kind"] == "junk"
    n_signal = int((real | mirror).sum())

    # Does the ranking put every non-junk vendor on top? That is the most this method can
    # possibly achieve, since it cannot see consensus and so cannot separate mirrors.
    top = df.head(n_signal)
    return {
        "genuine_median_agreement": float(df.loc[real, "agreement"].median()),
        "mirror_median_agreement": float(df.loc[mirror, "agreement"].median()),
        "junk_median_agreement": float(df.loc[junk, "agreement"].median()),
        "junk_max_agreement": float(df.loc[junk, "agreement"].max()),
        # separation: does the weakest signal vendor beat the luckiest junk vendor?
        "clean_separation": bool(df.loc[real | mirror, "agreement"].min()
                                 > df.loc[junk, "agreement"].max()),
        "precision_at_n": float((top["kind"] != "junk").mean()),
        "genuine_recall_at_n": float((top["kind"] == "genuine").sum() / real.sum()),
        "mirrors_in_top": int((top["kind"] == "mirror").sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=0)
    args = ap.parse_args()

    if args.sweep:
        rows, t0 = [], time.time()
        for i, seed in enumerate(range(args.sweep), 1):
            df, _ = analyse(build(seed))
            s = summarise(df)
            s["seed"] = seed
            rows.append(s)
            if i % 5 == 0 or i == args.sweep:
                print(f"  {i}/{args.sweep} seeds ({(time.time()-t0)/i:.1f}s each)", flush=True)
        d = pd.DataFrame(rows)
        Path("results").mkdir(exist_ok=True)
        d.to_csv("results/triangulate_sweep.csv", index=False)

        print(f"\nCROSS-VENDOR TRIANGULATION   ({args.sweep} seeds, NO ground truth used)")
        print(f"  median agreement - genuine: {d['genuine_median_agreement'].mean():.4f}")
        print(f"                     mirror:  {d['mirror_median_agreement'].mean():.4f}")
        print(f"                     junk:    {d['junk_median_agreement'].mean():.4f}"
              f"   (max seen {d['junk_max_agreement'].mean():.4f})")
        print(f"\n  top-7 precision (non-junk):   {d['precision_at_n'].mean():.1%}")
        print(f"  genuine vendors in the top 7: {d['genuine_recall_at_n'].mean():.1%}")
        print(f"  mirrors in the top 7:         {d['mirrors_in_top'].mean():.2f} of 2")
        print(f"  clean separation from junk:   {d['clean_separation'].mean():.1%} of seeds")
        print("\n  Mirrors ranking high is CORRECT behaviour, not a failure: they do measure")
        print("  something real. Triangulation cannot see consensus, so it cannot know that")
        print("  what they measure is already priced. Gate 2 is what removes them.")
        return

    u = build()
    df, M = analyse(u)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/triangulate.csv", index=False)

    pd.set_option("display.width", 160)
    print("CROSS-VENDOR TRIANGULATION   (seed 42, no fundamentals, no returns, no answer key)")
    print(df.head(12)[["rank", "vendor", "kind", "agreement", "best_partner",
                       "best_partner_name", "n_peers_above_05"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    s = summarise(df)
    print(f"\n  median agreement - genuine {s['genuine_median_agreement']:.4f}"
          f"   mirror {s['mirror_median_agreement']:.4f}"
          f"   junk {s['junk_median_agreement']:.4f}")
    print(f"  top-7 precision {s['precision_at_n']:.0%}"
          f"   genuine recall {s['genuine_recall_at_n']:.0%}"
          f"   mirrors in top 7: {s['mirrors_in_top']}")
    print(f"  every signal vendor above every junk vendor: {s['clean_separation']}")


if __name__ == "__main__":
    main()