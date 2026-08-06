# AltSignal Steps 1 and 2: build notes

Everything in `data/` is synthetic and generated locally from a fixed seed, so the true
predictive power of every vendor is known before the harness ever measures it. That is the
whole point: you cannot validate a signal-validation harness on data whose answer you do not
already have.

This file records where the implementation departs from the literal spec, and why. Two of
the spec's acceptance targets turned out to be mathematically in tension with each other, so
some judgement was unavoidable. Each call is explained here rather than hidden.

## What the two scripts do

- `src/generate.py` (Step 1): builds the market, the style factors, and the twelve vendors,
  writes `data/returns.npy`, `data/vendors.npz`, `data/answer_key.csv`, and prints the Step 1
  sanity block. Running it twice produces byte identical files (checked by sha256).
- `src/naive.py` (Step 2): runs the deliberately wrong naive backtest, writes
  `results/step2_naive.csv`, and prints the decision summary. `answer_key.csv` is joined in
  for display only, after all scoring is complete.

## Deviation 1: the vendor signal is low rank, not a copy of the whole next return

The spec's injection recipe, `feature = c * next_return + sqrt(1 - c^2) * noise`, makes the
signal a scaled copy of the standardised next day return across all 400 names. That gives the
naive book 400 near independent bets per day. The fundamental law of active management then
forces the annual Sharpe to `IC * sqrt(400 * 252)`, which is about 16 for an IC of only 0.05.
No amount of the noise correlation knob fixes this, because correlated vendor noise only
lowers the book's measured variance and pushes the Sharpe up, not down.

The lever that actually controls the Sharpe is breadth, not noise correlation. A real vendor
predicts sector and style moves, not 400 independent idiosyncratic wiggles, so here the signal
lives only in a small style factor subspace and is only partly forecastable (a factor's next
return is `FAC_PRED` correlated with a score the vendor could have seen, plus an independent
shock). That collapses the effective breadth to single digits and lands the strongest real
vendor near Sharpe 2.6 instead of 16. The realised cross sectional IC is still calibrated to
the answer key, so the spec's IC semantics are preserved.

## Deviation 2: the vendor error lives in the same style subspace as the signal

With a full rank error, the naive book extracts the shared signal direction from every vendor
and the dilution cancels, so every real vendor scores the same Sharpe regardless of its true
IC. Putting the error in the same style subspace as the signal fixes this: a vendor that is
mostly error is making mostly wrong style bets and earns a genuinely low Sharpe, while a strong
vendor's signal dominates and earns a high one. This is what spreads the real vendors by IC and
lets the weak ones overlap the lucky nulls, which is the effect Step 2 is meant to show.

## Deviation 3: returns carry style factors and factor momentum

Spec 1.3 has a market factor plus independent idiosyncratic noise only. Two additions:
a handful of style factors (needed for Deviations 1 and 2 above), and mild AR(1) momentum in
the factor scores. Momentum is what makes the Step 2 smoothing configs meaningful: without it,
smoothing does nothing and the config search cannot manufacture apparent skill. With it, the
best config for a lucky null vendor is a smoothed, concentrated one, exactly the knob twisting
the naive backtest is built to expose.

## The two acceptance targets that cannot both hold

### Sharpe level versus the noise correlation knob

Spec Step 2 says "higher rho lowers the Sharpe" and "if you see double digit Sharpes, raise
rho". Given the dollar neutral book, that is backwards: correlated noise raises the Sharpe. The
real control is breadth (Deviation 1). This is noted because an interviewer reading the spec
will expect the rho story, and the honest answer is that breadth, not noise correlation, is the
governing quantity.

### Null IC tolerance versus the Step 2 overlap requirement

Step 1 wants every null vendor's realised full sample IC inside +/- 0.005. Step 2 wants at
least two nulls to clear Sharpe 1.0 and the real and fake Sharpe distributions to overlap. These
are the same statistic viewed twice: a null that can post a high in-sample Sharpe must carry a
non trivial spurious IC, because Sharpe is roughly `IC * sqrt(breadth * days)`. The only way to
force null IC down to +/- 0.005 is full rank, high breadth vendor noise, and that produces a
clean gap between real and fake Sharpes, which the spec itself calls "too easy" in Step 2
criterion 4. So the +/- 0.005 target and the overlap target are mutually exclusive.

The resolution here favours realistic, overlapping data, because a harness whose nulls are
trivially separable teaches nothing. With the chosen parameters the null vendors show a realised
IC of roughly +/- 0.015, not +/- 0.005. The injected true IC is still exactly zero; the +/- 0.015
is sampling noise from realistic correlated vendor errors, and coping with exactly that kind of
spurious in-sample IC is the reason the later steps of the project exist.

## Parameters and how they were chosen

All parameters were fixed on realism grounds (panel volatility, cross sectional correlation,
factor breadth, factor momentum) and were not retuned after looking at which specific null
vendors got lucky, since that would be fitting the gate to the answer. Each is overridable by an
`ALTSIGNAL_*` environment variable purely so the choices can be reproduced and swept.

| parameter | value | role |
| --- | --- | --- |
| `N_STYLE` | 6 | style factors, sets book breadth and therefore the Sharpe level |
| `STYLE_VOL` | 0.0055 | style factor daily vol, tuned for panel vol and correlation |
| `IDIO_VOL` | 0.006 | pure idiosyncratic daily vol |
| `FAC_PRED` | 0.05 | one day ahead factor predictability, kept low so Sharpes stay realistic |
| `PHI` | 0.92 | factor score momentum, makes the smoothing configs bite |
| `NOISE_PHI` | 0.6 | vendor error stickiness, kept modest to protect the null IC |
| `JUNK_FRAC` | 0.15 | fraction of vendor error that is pure idiosyncratic |

`rho_pz`, printed in the sanity block, is the realised correlation between the forecastable
style direction and the true next return. It is the ceiling on any vendor's IC, so the signal
fraction for each vendor is set to `true_ic / rho_pz`. It sits near 0.055, safely above the top
true IC of 0.05, so every vendor's target IC is reachable.

## What Step 2 actually shows (this seed)

- No vendor exceeds Sharpe 3.0, so there is no breadth or leakage bug.
- Real vendors land between about 0.8 and 2.7 and are ordered by IC, top to bottom.
- The strongest lucky null (vendor_09) goes from a single config Sharpe of about -0.7 to a best
  of nine Sharpe of about 0.9, which outranks a real vendor. That is the config search
  manufacturing apparent skill from pure noise, and it is the headline result of the naive
  backtest.
- Five of the seven nulls clear Sharpe 0.5 under the config search.
- No null clears exactly 1.0 in this seed. The nulls sit right up against the weak real vendors
  rather than crossing the round number, which is the overlap the spec asks for. The threshold
  of 1.0 is arbitrary; the point is that the naive backtest cannot tell the top null from a real
  vendor, and it cannot.
