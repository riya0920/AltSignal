# AltSignal generator design notes

Everything in `data/` is synthetic and generated locally from a fixed seed, so the true
predictive power of every vendor is known before the harness ever measures it. That is the whole
point: you cannot validate a signal-validation harness on data whose answer you do not already
have.

This file covers the design of the synthetic generators (`src/generate.py` and `src/world.py`)
and the two places where the original spec's acceptance targets turned out to be mathematically
in tension. The rest of the harness (the walk-forward, significance, PBO, reality-check,
economics, provenance and fundamentals gates) is documented in each module's own docstring, and
the way they compose is documented at the top of `src/pipeline.py` and `src/pipeline2.py`.

## The shortlist the generator builds

`src/generate.py` builds the return-driven world: 400 assets, 1500 days, a sticky market factor,
six style factors, and fat-tailed returns. It then builds thirty vendors.

- Five real vendors, with information coefficients compressed into 0.010 to 0.030, and
  twenty-five nulls. That is a 1-in-6 base rate, close to what a real alternative-data shortlist
  looks like, and much harder than an even split: with twenty-five nulls competing, many more of
  them get lucky, so even an accurate test approves a lot of junk.
- Each style factor has its own persistence, from fast mean-reversion to slow trend, and each
  real vendor primarily reads a different factor. Two vendors with identical IC therefore differ
  in decay and turnover, which is the most common way real vendors differ in value.
- Some vendors ship reconstructed ("backfilled") history: a strong injected signal before a
  disclosed live-start date, reverting to true skill after it. One real vendor is backfilled too,
  so that "has backfill" is not just a synonym for "fake". The live-start date is disclosed in
  the answer key, because in reality it is on the vendor's data sheet.

Running `src/generate.py` twice produces byte-identical files (checked by sha256). The answer key
(`true_ic` and `live_start`) is written to `data/` but is only ever joined back in downstream for
scoring, never read while a gate is deciding.

## Deviation 1: the vendor signal is low rank, not a copy of the whole next return

A naive IC-injection recipe, `feature = c * next_return + sqrt(1 - c^2) * noise`, makes the signal
a scaled copy of the standardised next-day return across all 400 names. That gives a cross
sectional book 400 near-independent bets per day. The fundamental law of active management then
forces the annual Sharpe to `IC * sqrt(400 * 252)`, which is about 16 even for an IC of 0.03. No
amount of a noise-correlation knob fixes this, because correlated vendor noise only lowers the
book's measured variance and pushes the Sharpe up, not down.

The lever that actually controls the Sharpe is breadth, not noise correlation. A real vendor
predicts sector and style moves, not 400 independent idiosyncratic wiggles, so here the signal
lives only in a small style-factor subspace and is only partly forecastable: a factor's next
return is `FAC_PRED` correlated with a score the vendor could have seen, plus an independent
shock. That collapses the effective breadth to single digits and lands the strongest real vendor
near a realistic Sharpe. The realised cross-sectional IC is still calibrated to the answer key, so
the IC semantics are preserved.

## Deviation 2: the vendor error lives in the same style subspace as the signal

With a full-rank error, a cross-sectional book extracts the shared signal direction from every
vendor and the dilution cancels, so every real vendor scores roughly the same Sharpe regardless
of its true IC. Putting the error in the same style subspace as the signal fixes this: a vendor
that is mostly error is making mostly wrong style bets and earns a genuinely low Sharpe, while a
strong vendor's signal dominates. This is what spreads the real vendors by IC and lets the weak
ones overlap the lucky nulls, which is the effect the whole harness exists to see through.

## Deviation 3: returns carry style factors and per-factor momentum

A market factor plus independent idiosyncratic noise is not enough. Two additions: a handful of
style factors (needed for Deviations 1 and 2), and AR(1) momentum in the factor scores, with a
different persistence per factor. Momentum is what makes smoothing meaningful downstream: without
it, a trailing average does nothing and the naive config search cannot manufacture apparent skill.
Per-factor persistence is what gives each vendor a distinct decay profile, which is what
`decay.py`, `turnover.py` and `capacity.py` measure.

## Tension 1: Sharpe level versus the noise-correlation knob

The original spec said "higher rho lowers the Sharpe" and "if you see double-digit Sharpes, raise
rho". Given a dollar-neutral book, that is backwards: correlated noise raises the Sharpe. The real
control is breadth (Deviation 1). This is worth stating plainly, because a reader expecting the
rho story will be surprised, and the honest answer is that breadth, not noise correlation, is the
governing quantity.

## Tension 2: null IC tolerance versus the overlap requirement

One acceptance target wants every null vendor's realised full-sample IC inside a tight band around
zero. Another wants the nulls to be genuinely confusable with weak real vendors. These are the
same statistic viewed twice: a null that can look convincing in sample must carry a non-trivial
spurious IC, because Sharpe is roughly `IC * sqrt(breadth * days)`. The only way to force null IC
to near zero is full-rank, high-breadth vendor noise, and that produces a clean gap between real
and fake, which is exactly the "too easy" failure a good harness has to avoid.

The resolution favours realistic, overlapping data, because a harness whose nulls are trivially
separable teaches nothing. The nulls therefore show a small but non-zero realised IC. The injected
true IC is still exactly zero; the residual is sampling noise from realistic correlated vendor
errors, and coping with exactly that kind of spurious in-sample IC is the reason the downstream
gates exist.

## Generator parameters

All parameters were fixed on realism grounds (panel volatility, cross-sectional correlation,
factor breadth, factor momentum, base rate) and were not retuned after looking at which specific
null vendors got lucky, since that would be fitting the gate to the answer. Each is overridable by
an `ALTSIGNAL_*` environment variable only so the choices can be reproduced and swept.

| parameter | value | role |
| --- | --- | --- |
| `REAL_ICS` | 0.030 to 0.010 | five real vendors, compressed into the hard band |
| `N_NULLS` | 25 | null vendors, giving a 1-in-6 base rate |
| `N_STYLE` | 6 | style factors, sets book breadth and the Sharpe level |
| `STYLE_VOL` | 0.0055 | style factor daily vol, tuned for panel vol and correlation |
| `IDIO_VOL` | 0.006 | pure idiosyncratic daily vol |
| `FAC_PRED` | 0.10 | one-day-ahead factor predictability |
| `PHI_LO` / `PHI_HI` | 0.55 / 0.97 | fastest and slowest factor persistence |
| `PRIMARY_W` | 0.75 | how concentrated a real vendor is on its primary factor |
| `NOISE_PHI` | 0.6 | vendor error stickiness, kept modest to protect the null IC |
| `JUNK_FRAC` | 0.15 | fraction of vendor error that is pure idiosyncratic |
| `BACKFILL_IC` | 0.05 | injected IC in the reconstructed portion of a backfilled vendor |
| `N_BACKFILL_NULLS` | 8 | null vendors shipped with reconstructed history |

`rho`, computed per vendor inside the generator, is the realised correlation between the
forecastable style direction and the true next return. It is the ceiling on any vendor's IC, so
each vendor's signal fraction is set to `min(true_ic / rho, 1)`. It sits comfortably above the top
true IC of 0.030, so every vendor's target IC is reachable.
