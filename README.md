# AltSignal

All data in this project is synthetic and generated locally from a fixed seed. Nothing here
touches a real vendor or a licensed feed. That is deliberate: AltSignal is a harness that
decides which alternative data vendors genuinely predict stock returns and which merely got
lucky, and a harness like that can only be trusted once it has been validated on data whose
true answer is known in advance. Twelve candidate vendors are built, five with real predictive
power and seven that are pure noise, and the chosen list of true information coefficients is the
answer key the harness is later graded against.

This repository currently covers Step 1 (the synthetic universe and the twelve vendor datasets)
and Step 2 (a deliberately wrong naive backtest that is supposed to produce false positives).
Steps 3 to 5, which add the statistical guards, are out of scope here.

## Run it

```
pip install numpy pandas
python -m src.generate     # Step 1: writes data/, prints the sanity block
python -m src.naive        # Step 2: writes results/step2_naive.csv, prints the summary
```

`python -m src.generate` is deterministic: run it twice and the files in `data/` are byte
identical.

## Layout

```
src/generate.py     Step 1, the synthetic universe and vendors
src/naive.py        Step 2, the naive backtest baseline
data/               generated, gitignored (returns.npy, vendors.npz, answer_key.csv)
results/            step2_naive.csv, the naive scorecard
NOTES.md            build notes: where this departs from the literal spec and why
```

## Read NOTES.md first

The generator makes a few deliberate departures from the literal spec, and two of the spec's
acceptance targets turned out to be mathematically in tension. `NOTES.md` explains every choice.
The short version: the Sharpe level of the naive backtest is governed by the effective breadth
of the book, not by any noise correlation knob, and a null vendor that can look good in sample
must carry some spurious in-sample IC, so it cannot both look good in Step 2 and have a
near zero realised IC in Step 1.
