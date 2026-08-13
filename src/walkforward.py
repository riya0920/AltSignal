"""AltSignal Step 3 - purged walk-forward validation.

The bug this fixes: Step 2 chose the winning config using all 1500 days and then reported
that config's score on the same 1500 days. Selection and evaluation shared data, so a config
that won by fitting noise was rewarded for having done so.

Walk-forward separates the two. The config is chosen on a training window, then locked and
scored on days strictly after it that the choice never touched. The window rolls forward and
the out-of-sample P&L is stitched together into one series.

There is still no model being fitted anywhere in this file. The only thing "trained" is the
choice of one config label out of nine. That is deliberate: build the scaffolding while the
selection is trivial to reason about, then Step 5 swaps in Ridge and an MLP and the same
fold machinery carries them unchanged.

score_all_wf() is pure - arrays in, dataframe out - so src/robustness.py can call it in a
loop over seeds exactly as it calls naive.score_all().

Run:  python -m src.walkforward      (from the repo root, after python -m src.generate)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.naive import CONFIGS, smooth, top_k, weights

# --- fold geometry ---
# Rolling, not expanding: markets change regime, and a 2018 training block should not still
# be steering a 2024 decision. At n=1499 these values yield 7 folds covering 882 test days.
TRAIN_DAYS = 504        # ~2 years of selection data
TEST_DAYS = 126         # ~6 months of evaluation
PURGE_DAYS = 25         # gap between train end and test start
EMBARGO_DAYS = 25       # gap after a test block before it may re-enter a later training set

MIN_FOLDS = 5


def make_folds(n, train_days=TRAIN_DAYS, test_days=TEST_DAYS,
               purge_days=PURGE_DAYS, embargo_days=EMBARGO_DAYS):
    """Rolling purged/embargoed folds over a signal of length n.

    PURGE: the signal is smoothed with a trailing mean of up to 20 days, so the vendor value
    on the first test day is built partly from the last training days. Without a gap, train
    and test performance share inputs. purge_days must exceed max(SMOOTH_WINDOWS).

    EMBARGO: windows roll forward, so days that were test data in fold i fall inside the
    training window of a later fold. The embargo_days immediately following each test block
    are masked out of every later training slice. It is implemented as an explicit index mask
    rather than an offset, so the excluded days are visible and assertable.
    """
    raw = []
    i = 0
    while True:
        train_start = i * test_days
        train_end = train_start + train_days
        test_start = train_end + purge_days
        test_end = test_start + test_days
        if test_end > n:
            break
        raw.append((train_start, train_end, test_start, test_end))
        i += 1

    if len(raw) < MIN_FOLDS:
        raise ValueError(
            f"only {len(raw)} folds at n={n} with train={train_days}, test={test_days}, "
            f"purge={purge_days}; need >= {MIN_FOLDS}. A near-single-fold run looks fine "
            f"and means nothing."
        )

    folds = []
    for j, (a, b, c, d) in enumerate(raw):
        banned = np.zeros(n, dtype=bool)
        for k, (_, _, tc, td) in enumerate(raw):
            if k < j:                                   # only EARLIER test blocks
                banned[td:td + embargo_days] = True      # the embargo gap after them
        train_idx = np.arange(a, b)[~banned[a:b]]
        folds.append({
            "fold": j,
            "train_idx": train_idx,
            "test_idx": np.arange(c, d),
            "train_span": (a, b),
            "test_span": (c, d),
        })
    return folds


def assert_folds_clean(folds, n, purge_days=PURGE_DAYS, embargo_days=EMBARGO_DAYS):
    """Step 3 acceptance criterion 2, as a hard assert rather than a code review.

    The likeliest bug here is an off-by-one that lets a test day into selection. It does not
    crash, and it makes walk-forward look BETTER than it is - a failure mode that presents as
    a good result. So it is checked directly.
    """
    ends = [f["test_span"][1] for f in folds]
    for j, f in enumerate(folds):
        tr, te = set(f["train_idx"].tolist()), set(f["test_idx"].tolist())
        assert not (tr & te), f"fold {j}: test days leaked into selection"
        assert f["test_span"][0] - f["train_span"][1] >= purge_days, f"fold {j}: purge too small"
        assert max(f["train_idx"]) < min(f["test_idx"]), f"fold {j}: training day after test start"
        for k in range(j):
            gap = set(range(ends[k], min(ends[k] + embargo_days, n)))
            assert not (tr & gap), f"fold {j}: embargo after fold {k} violated"
    return True


def config_pnls(sig, fwd):
    """Daily P&L series for each of the nine configs, computed once over the full history.

    Trailing means are causal, so smoothing the whole series and then slicing introduces no
    look-ahead. Recomputing inside each window would leave the first 20 rows of every test
    block unsmoothed and silently change the signal being evaluated.
    """
    return {label: (weights(top_k(smooth(sig, s), k)) * fwd).sum(axis=1)
            for label, s, k in CONFIGS}


def _sharpe(pnl):
    sd = pnl.std()
    return 0.0 if sd == 0 else pnl.mean() / sd * np.sqrt(252)


def score_vendor_wf(sig, fwd, folds):
    """Walk-forward score for one vendor. Returns a dict of result columns."""
    pnls = config_pnls(sig, fwd)

    oos, picks, fold_sharpes = [], [], []
    for f in folds:
        tr, te = f["train_idx"], f["test_idx"]
        # choose on train only; the training scores are selection output, not results
        pick = max(CONFIGS, key=lambda c: _sharpe(pnls[c[0]][tr]))[0]
        picks.append(pick)
        seg = pnls[pick][te]                      # the locked config, applied out of sample
        oos.append(seg)
        fold_sharpes.append(_sharpe(seg))

    oos = np.concatenate(oos)

    # Fair comparison: the naive number is computed on the SAME days, so the contrast is the
    # protection and not a change of sample.
    test_union = np.concatenate([f["test_idx"] for f in folds])
    naive_sharpe = max(_sharpe(p[test_union]) for p in pnls.values())

    modal = max(set(picks), key=picks.count)
    return {
        "wf_sharpe": _sharpe(oos),
        "naive_sharpe": naive_sharpe,
        "selection_cost": naive_sharpe - _sharpe(oos),
        "config_switches": sum(a != b for a, b in zip(picks, picks[1:])),
        "modal_config": modal,
        "modal_config_frac": picks.count(modal) / len(picks),
        "n_folds": len(folds),
        "wf_sharpe_per_fold": ";".join(f"{x:.3f}" for x in fold_sharpes),
    }


def score_all_wf(returns, vendors):
    """Walk-forward score every vendor. Pure: no files, no printing, no answer key.

    Column contract matches naive.score_all plus the walk-forward columns, so
    robustness.py can call either scorer.
    """
    fwd = returns[1:]
    n = fwd.shape[0]
    folds = make_folds(n)
    assert_folds_clean(folds, n)

    rows = []
    for name, sig in vendors.items():
        r = {"vendor": name}
        r.update(score_vendor_wf(sig, fwd, folds))
        rows.append(r)
    return (pd.DataFrame(rows)
            .sort_values("wf_sharpe", ascending=False)
            .reset_index(drop=True))


def main():
    returns = np.load("data/returns.npy")
    vendors = dict(np.load("data/vendors.npz"))
    key = pd.read_csv("data/answer_key.csv")

    df = score_all_wf(returns, vendors)

    # answer key joined for DISPLAY ONLY, after all scoring is complete
    df = df.merge(key, on="vendor", how="left")
    df["kind"] = np.where(df["true_ic"] > 0, "REAL", "fake")

    out = Path("results")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "step3_walkforward.csv", index=False)

    folds = make_folds(returns.shape[0] - 1)
    cover = sum(len(f["test_idx"]) for f in folds)
    print(f"STEP 3 PURGED WALK-FORWARD   ({len(folds)} folds, {cover} out-of-sample days, "
          f"train={TRAIN_DAYS} test={TEST_DAYS} purge={PURGE_DAYS} embargo={EMBARGO_DAYS})")

    show = ["vendor", "kind", "true_ic", "naive_sharpe", "wf_sharpe",
            "selection_cost", "config_switches", "modal_config", "modal_config_frac"]
    pd.set_option("display.width", 160)
    print(df[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    fakes, reals = df[df["true_ic"] == 0], df[df["true_ic"] > 0]
    inv = int((fakes["wf_sharpe"].values[:, None] > reals["wf_sharpe"].values).sum())
    inv_naive = int((fakes["naive_sharpe"].values[:, None] > reals["naive_sharpe"].values).sum())

    print("\nSUMMARY   (same days for both columns)")
    print(f"  inversions, naive selection:   {inv_naive}   (fake outranking real)")
    print(f"  inversions, walk-forward:      {inv}")
    print(f"  mean selection_cost, fakes:    {fakes['selection_cost'].mean():+.3f}"
          f"   (Sharpe that existed only because the outcome was visible)")
    print(f"  mean selection_cost, reals:    {reals['selection_cost'].mean():+.3f}")
    print(f"  mean config_switches, fakes:   {fakes['config_switches'].mean():.2f}"
          f"   (of {len(folds) - 1} possible)")
    print(f"  mean config_switches, reals:   {reals['config_switches'].mean():.2f}")
    print("\n  NOTE: this is one seed. Run src/robustness.py --scorer walkforward.")


if __name__ == "__main__":
    main()