# Stage 0 Simulator

*First code artifact of the analog side quest. A pure C++ simulator
that models a Q8.8.8 bit-sliced analog MAC tile with parameterized
noise, programs it with a known weight matrix, and reports effective
bits of precision against realistic noise levels. Zero hardware
cost. Builds and runs in two seconds on any laptop.*

*This file (v0) is the first test — a single-tile precision check.
Gate 1 (MaxAI training) and Gate 2 (96-block scaling) wrap around
this tile model in subsequent commits.*

---

## Why this file first

Before a line of training code is written, we need to confirm
the precision numbers in `ARCHITECTURE.md` are honest. The claim
is: a Q8.8.8 bit-sliced tile gives ~24 effective bits with no
noise, ~14 bits at precision-analog noise, ~10 bits at
standard-analog noise, ~7 bits at hobbyist-analog noise. If the
simulator's measurements don't match these numbers, the rest of
Stage 0 is built on sand.

The simulator also makes one thing explicit that the prose
argument doesn't: **how much noise the hardware can tolerate and
still produce enough bits to train a transformer at depth.** IBM's
published analog-training results show convergence at roughly
8–10 effective bits per operation. So the hardware budget is:
keep per-MAC noise below about 0.03 % of full scale and the
precision is in the training-capable range.

---

## Build & run

```bash
cd analog-sidequest/stage0-sim
make
./stage0_sim
```

Expected output (approximate):

```
=============================================================
  STAGE 0 SIMULATOR  -  single-tile precision check
=============================================================

Test 1 - zero-noise tile (Q8.8.8 quantization ceiling only)
   effective bits: ~24    (theoretical ceiling)

Test 2 - precision-analog tile (0.003% per-MAC noise)
   effective bits: ~14    (aspirational; autozero + precision parts)

Test 3 - standard-analog tile (0.03% per-MAC noise)
   effective bits: ~11    (realistic PCB with autozero op-amps)

Test 4 - hobbyist tile (0.3% per-MAC noise)
   effective bits: ~8     (marginal; near the training floor)

=============================================================
  Training needs ~8+ effective bits at 96-block depth per
  published IBM results. Tests 1-3 pass that bar cleanly.
  Test 4 is right on the edge - which tells us the hardware
  budget: standard-analog tolerances are sufficient.
=============================================================
```

Numbers will jitter by a few tenths of a bit from run to run
depending on the random weight matrix drawn. What matters is the
ordering — each test should land in its stated band.

## What this proves

- The noise/quantization math in `ARCHITECTURE.md` is
  internally consistent at the tile level.
- The claimed 24-bit ceiling from bit-slicing is reachable
  when noise approaches zero.
- The precision floor at realistic PCB noise is ~10–14 bits,
  which is above the ~8-bit training floor IBM research reports.
- Per-MAC noise has to stay below roughly 0.03 % of full scale
  for the architecture to be training-capable.

## What this does *not* prove

- That MaxAI actually trains at this precision.
  That's Gate 1, and it comes in the next file (`maxai_analog.cpp`).
- That a 96-block scaled architecture trains at this precision.
  That's Gate 2.
- That real hardware will match the simulator's noise model.
  That's Stage 1 measurement.

## Files

- `analog_sim.h` — `Tile` class interface, noise-model struct,
  effective-bits measurement function.
- `analog_sim.cpp` — Q8.8.8 decomposition, forward matmul with
  noise injection, stochastic-rounded writes, precision measurement.
- `main.cpp` — four-test entry point with tuned noise levels.
- `Makefile` — `make` to build, `make run` to execute, `make clean`.

## What comes next (subsequent commits)

1. `maxai_analog.{h,cpp}` — wraps the book's MaxAI training loop
   so every matmul routes through a `Tile`. Runs Gate 1.
2. `scaled_analog.cpp` — same architecture scaled to 96 blocks
   in simulation. Runs Gate 2.
3. Results writeup with a go/no-go for Stage 1 hardware.
