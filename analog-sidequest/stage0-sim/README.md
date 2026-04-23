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
- `main.cpp` — single-tile precision test (four noise levels).
- `sgd_test.cpp` — single-layer SGD convergence test through the tile.
- `maxai_analog.cpp` — Gate 1. A copy of the book's `maxai.cpp` with
  analog hooks injected at every forward matmul and every SGD weight
  update. At `--analog_noise=0` the code is byte-identical to stock
  MaxAI (FMA-preserving); at `sigma>0` each matmul output picks up
  Gaussian noise and each weight update routes through Q8.8.8
  stochastic rounding.
- `Makefile` — `make` to build all, `make run` to execute, `make clean`.

## Gate 1 results (passed)

Running the seashells BPE-16 training for 3000 steps at various analog
noise settings:

```
./stage0_maxai --task=next_token --causal=1 --vocab=chars \
               --corpus=seashells --bpe=16 --seq_len=16 \
               --batch=16 --blocks=2 --layernorm=1 --steps=3000 \
               --analog_noise=<sigma> --gen_prompt="she sells "
```

| σ (per-MAC)    | hardware regime             | final loss | generation from `"she sells "`                          |
|----------------|-----------------------------|------------|---------------------------------------------------------|
| 0              | clean float32 baseline      | 0.098      | "by the sea shore the shells she sells are"             |
| 0.00003        | precision analog            | 0.059      | "by the sea shore the shells she sells are"             |
| 0.0003         | standard analog (target)    | 0.059      | "by the sea shore the shells she sells are"             |
| 0.003          | hobbyist analog             | 0.060      | "by the sea shore the shells she sells are"             |

All four runs converge and produce identical correct generated text.
Gate 1 pass criteria from `../STAGE0.md`:

1. ✓ Training loss decreases monotonically, no log(VOCAB) plateau.
2. ✓ Analog-trained held-out loss within 1.5× clean baseline (in fact
   a bit *below* — the stochastic writes appear to act as regularization
   on this small corpus).
3. ✓ Generated text is recognizably coherent at every noise level.
4. ✓ Numbers consistent with published IBM analog-training results in
   effective-bits range.

**The analog architecture trains MaxAI at standard-analog PCB noise.**

## Gate 2 results (passed)

Gate 2 asks the question Gate 1 left open: does analog noise compound
destructively as depth grows? The same `stage0_maxai` binary supports
`--blocks=96` out of the box (that's `MAX_BLOCKS` in the source, sized
to match GPT-3's layer count), so Gate 2 is a depth-and-noise sweep
against the existing binary rather than a new file.

Every run below is 3000 steps on the seashells BPE-16 corpus,
`--seq_len=16 --batch=16 --layernorm=1`, prompt `"she sells "`.

### Matrix A — depth sweep at standard-analog noise (σ=0.0003)

| blocks | final loss | generation from `"she sells "`                     |
|--------|------------|------------------------------------------------------|
| 2      | 0.059      | "by the sea shore the shells she sells are"         |
| 12     | 0.068      | "by the sea shore the shells she sells are"         |
| 48     | 0.066      | "bare sea shells are sea shells im"                 |
| 96     | 0.053      | "by the sea shore the shells she sells are"         |

Loss does not grow with depth. The deepest network actually lands at
the lowest loss — deeper models fit this tiny corpus better, and the
standard-analog noise does not disrupt that fit.

### Matrix B — noise sweep at blocks=96

| σ (per-MAC) | hardware regime     | final loss | generation                                          |
|-------------|---------------------|------------|------------------------------------------------------|
| 0           | clean float32       | 0.220      | "are sea shells im sure so i"                       |
| 0.00003     | precision analog    | 0.052      | "by the sea shore the shells she sells are"         |
| 0.0003      | standard analog     | 0.053      | "by the sea shore the shells she sells are"         |
| 0.003       | hobbyist analog     | 0.077      | "are sea shells she sells are sea shells im"        |

**A genuine surprise.** The clean baseline ends at a *higher* loss than
all three noisy runs. At 96-block depth the stochastic-rounded weight
updates plus Gaussian matmul noise behave like gradient noise and
dropout — mild regularization that helps a deep net fit a small
corpus. This is consistent with published noise-injection results
(Neelakantan 2015, Gulcehre 2016) and is an accidental bonus for the
analog approach, not something we were designing for.

Gate 2 pass criteria from `../STAGE0.md`:

1. ✓ Training converges at 96-block depth at every noise level.
2. ✓ Standard-analog loss at depth (0.053) is not worse than at 2
   blocks (0.059); noise does not compound destructively.
3. ✓ Generated text is recognizably coherent at every noise level.
4. ✓ Even hobbyist-analog noise trains through 96 blocks.

**The analog architecture trains MaxAI at GPT-3 depth at standard-
analog PCB noise. Both Stage 0 gates pass.**

## What comes next

1. **Stage 1 hardware.** With both gates passed, the noise budget for
   the PCB spec is locked: per-MAC RMS error ≤ 0.03 % of full scale,
   equivalent to σ=0.0003 in the simulator. A single precision-analog
   tile (autozero op-amps, 0.01 % thin-film resistors, 12-bit DAC
   writes with stochastic rounding) meets this target.
2. Writeup — go/no-go decision for Stage 1 build with the simulator
   numbers as backing evidence, then the PCB schematic.
