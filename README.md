# AltSignal

All data in this project is synthetic and generated locally from a fixed seed. Nothing here
touches a real vendor or a licensed feed. That is deliberate: AltSignal is a harness that
decides which alternative-data vendors genuinely predict stock returns and which merely got
lucky, and a harness like that can only be trusted once it has been validated on data whose
true answer is known in advance. The generators build a shortlist of candidate vendors, a
handful with real predictive power and the rest pure noise, and the chosen list of true
information coefficients is the answer key the harness is graded against but never allowed to
read while deciding.

The whole point is the gate a dataset has to pass before a research desk spends money and
attention on it: does it say something new, is it real or just lucky, is its history genuine,
and can it pay for its own trading. Every number below comes out of measured output, and the
ground truth is only ever joined back in afterwards, for scoring.

## The two worlds

The harness runs against two synthetic worlds, each with a fixed seed and a disclosed answer
key.

- **Return-driven world** (`src/generate.py`). 400 assets, 1500 days, a sticky market factor,
  six style factors each with its own persistence (fast mean-reversion through slow trend), and
  fat-tailed returns. Thirty vendors: five real with information coefficients compressed into
  0.010 to 0.030, and twenty-five nulls, a 1-in-6 base rate that mirrors a real shortlist.
  Each real vendor reads a different factor, so vendors with identical IC still differ in decay
  and turnover. Some vendors ship reconstructed ("backfilled") history with a disclosed
  live-start date, including one real vendor so that "has backfill" is not a synonym for "fake".

- **Unified world** (`src/world.py`). The same returns plus a fundamentals layer: quarterly
  revenue, consensus estimates, and per-vendor nowcasts. It adds mirror vendors that track
  consensus almost perfectly, so they are the most accurate vendors in the shortlist and worth
  nothing, because everything they know is already in the price. They exist to break any gate
  that filters on accuracy alone.

## The pipeline

`src/pipeline.py` runs four gates in order, cheapest and most powerful first, each narrowing
what the next may look at. Gate order is enforced structurally, not by convention.

1. **Provenance.** A vendor that discloses a live-start date is scored only on data after it.
   Reconstructed history is not evidence. This runs first because it changes the data every
   later gate sees.
2. **Significance.** A Newey-West p-value on the turnover-controlled ensemble book, then
   Benjamini-Hochberg across vendors. It controls the fraction of the buy list that is junk,
   which is the right framing when you are buying five things, not promising to never be wrong.
3. **Economics.** Breakeven cost and capacity on a turnover-controlled book. It kills vendors
   that are real but cannot pay for their own trading.
4. **Holdout.** The final 250 days are sealed before gate 1 runs and opened once, after the buy
   list is fixed.

The output that matters is not accuracy, since the answer key gives that away. It is whether the
holdout would have told a desk the same thing without an answer key: if the bought vendors
outperform the rejected ones on data nothing touched, a desk running this process gets an honest
read on its own hit rate.

`src/pipeline2.py` runs the same worlds twice, changing only what gate 2 reads: return-based
p-values versus a fundamentals test that requires both accuracy against reported revenue and
incremental value over consensus. That second condition is what removes the mirror vendors.

## Run it

```
pip install numpy pandas scipy
python -m src.generate            # build the return-driven world into data/, print the sanity block
python -m src.pipeline            # the four-gate buy list on seed 42, writes results/pipeline.csv
python -m src.pipeline --sweep 30 # the same across 30 seeds, writes results/pipeline_sweep.csv
python -m src.pipeline2           # returns vs fundamentals gate, writes results/pipeline2.csv
```

Each analysis module is also runnable on its own with `python -m src.<name>` and writes its own
table into `results/`. Most take a `--sweep N` flag for the multi-seed version. Generation is
deterministic: same seed in, byte-identical arrays out.

## Layout

```
src/
  generate.py      return-driven world: the universe and the 30-vendor shortlist  (Step 1)
  world.py         unified world: the above plus the fundamentals and mirror layer
  naive.py         the deliberately wrong baseline: no split, keep the best of nine configs  (Step 2)
  walkforward.py   purged walk-forward validation with an embargo  (Step 3)
  ensemble.py      the alternative to selection: stop picking a config, average all nine  (Step 3)
  significance.py  Newey-West, Benjamini-Hochberg, deflated Sharpe: pay for your searches  (Step 4)
  pbo.py           Probability of Backtest Overfitting, combinatorially symmetric cross-validation
  reality_check.py White's Reality Check, Hansen's SPA, Romano-Wolf stepdown
  gate_sweep.py    what the decision rule costs: sweeping gate 2 instead of assuming it
  turnover.py      turnover control, so the book is not rebuilding itself every three days
  decay.py         signal decay and transaction costs: why statistically real vendors still fail
  capacity.py      capacity: dropping the infinite-liquidity assumption
  backfill.py      backfill detection, the highest-powered test in the harness, and it uses no statistics
  changepoint.py   finding the backfill seam without being told where it is
  holdout.py       the true holdout: what the whole pipeline is worth on data it never saw
  triangulate.py   cross-vendor triangulation: consistency without any ground truth
  fundamentals.py  scoring vendors against what they actually measure
  event_book.py    trading a quarterly signal the way a real earnings desk does
  robustness.py    multi-seed harness that wraps any scorer
  pipeline.py      the four-gate buy list, return-driven world
  pipeline2.py     the same, comparing a returns gate against a fundamentals gate

data/              generated, gitignored (returns.npy, vendors.npz, answer_key.csv)
results/           one CSV per module, plus the *_sweep.csv multi-seed summaries
NOTES.md           design notes for the generator and the two tensions in the original spec
```

## What the pipeline finds (seed 42, return-driven world)

- Provenance discards several thousand vendor-days of reconstructed history before anything is scored.
- Significance leaves a handful of vendors, a clear majority of them real.
- Economics removes the survivors that cannot cover trading cost.
- On the sealed holdout, the bought vendors outperform the rejected ones. That gap, not the
  precision against the answer key, is the number a real desk could actually observe.

Numbers move seed to seed, which is the point: run any module with `--sweep` for the
distribution rather than a single lucky or unlucky draw.

## Read NOTES.md for the generator design

The generator makes a few deliberate departures from a naive IC-injection recipe, and two of the
original spec's acceptance targets turned out to be mathematically in tension. The short version:
the Sharpe level of a backtest is governed by the effective breadth of the book, not by any
noise-correlation knob, and a null vendor that can look good in sample must carry some spurious
in-sample IC, so it cannot both look convincing downstream and have a near-zero realised IC.
`NOTES.md` works through both.
