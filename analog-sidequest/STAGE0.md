# Stage 0 — the simulator viability gate

*Before any PCB is sent to fab. Before any parts are ordered.
Before any digipot is soldered. A software-only simulator runs the
entire tile architecture, with realistic noise, write quantization,
and the five training tricks applied, to answer one question:
**does MaxAI actually train to recognizable output on a simulated
version of the analog hardware we plan to build?**

If yes, Stage 1 begins. If no, the project stops here, at zero
hardware cost.*

---

## Why Stage 0 exists

Every commercial analog-AI startup that eventually failed built
hardware first and discovered the training/inference-accuracy
shortfall later. Mythic fabricated real silicon before finding the
market didn't want inference-only cards. Graphcore taped out
multiple generations of chips before the software stack caught up
(it didn't). Lightmatter spent years on photonic meshes before
pivoting away from neural-network compute.

This project explicitly refuses that pattern. **Stage 0 is a
cheap, fast, disposable simulator whose job is to kill the
project if training is not viable at our target precision, before
a dollar is spent on physical hardware.**

The argument: PCBs are cheap but not free (~$100 per tile + a
week of iteration time). Parts are cheap but not free (~$80 per
tile). Each failed physical design costs hundreds of dollars and
days. A simulator costs nothing per run. If the simulator reveals
that MaxAI cannot train under the noise the real hardware will
produce, the right answer is to find that out on a laptop in an
afternoon, not on a breadboard over a month.

---

## What the simulator must model

The simulator is written in C++ (the project's native language)
and models the *full* analog-hardware stack as faithfully as
possible without actually building it. All four hardware
precision amplifiers AND all five software training tricks from
`ARCHITECTURE.md` must be modeled. Removing any one of them and
re-running is a first-class experiment the simulator supports —
we want to know which tricks are load-bearing and which are
decorative.

### Forward pass, per tile (with Q8.8.8 bit-slicing)

```
y = tile_forward_bitsliced(W_A, W_B, W_C, x, noise_params)
```

Where `W_A`, `W_B`, `W_C` are the three 8-bit slices of each
weight, summed with binary weighting 1, 1/256, 1/65536. Behavior:

- Each slice is quantized to 8 bits (digipot tap resolution).
- Per-cell Gaussian noise of ~0.3% full scale at room
  temperature, modeled independently for each of the three slices.
  Noise on the low-order slice dominates the contribution of that
  slice at the summing junction.
- Autozero summing op-amp: offset drift below noise floor
  (effectively negligible), 1/f noise mostly rejected. Modeled as
  Gaussian white noise only.
- Input `x` quantized to DAC resolution (12 bits).
- Temperature drift modeled as a slow bias per cell, **reset by
  the calibration-reference-cell correction loop** at a
  configurable cadence.
- Crosstalk between adjacent output columns modeled from
  published measurements.

### Write channel (bit-sliced, stochastic rounding)

```
commit_weight_update(tile_id, cell_id, desired_delta, noise_params)
```

- The desired delta is decomposed across the three bit slices
  (most → least significant) and each slice's tap change is
  computed.
- For each slice, the partial update is rounded stochastically
  (±1 tap with probability equal to fractional part, else 0 taps).
- Per-write ±1 tap jitter modeled.
- Write endurance tracked: 10⁶–10¹² cycles depending on cell
  model; degraded cells exhibit increased noise.

### Gradient accumulator (32-bit wide)

```
accumulator[cell_id] += gradient_update        // 32-bit precision
while |accumulator[cell_id]| >= finest_slice_step:
    commit_weight_update(..., stochastic_round_delta)
    accumulator[cell_id] -= committed_delta
```

The accumulator is explicitly 32-bit to avoid being the bottleneck
on training precision. The step threshold is the finest bit-slice
step (roughly 1/65,536 of full scale for Q8.8.8), not the coarsest.

### Calibration reference cells

Each tile has 4–8 cells programmed to known reference values. The
simulator tracks their drift over time (matched to the model of
regular cells); a calibration loop reads them periodically and
computes a per-tile correction matrix that is applied to all
other readbacks.

### Readback with oversampling (optional per layer)

```
y_hat = average over N reads of tile_forward_bitsliced(...),
        each with small random dither voltage added
```

Configurable `N` per layer, per direction (forward vs backward).

---

## Two gates to pass — functional, then scaling

The simulator runs **two** tests. The first must pass before the
second is worth running; the second must pass before Stage 1
hardware is cleared to fabricate.

### Gate 1 — Functional (does the architecture train at all?)

Run the book's MaxAI training loop (2 blocks, char-level BPE,
Gettysburg or seashells corpus) through the noise model with the
four hardware tricks (Q8.8.8, autozero, 32-bit accumulator,
calibration refs) and the five software tricks enabled. Pass
criteria:

1. **Training loss decreases** monotonically (allowing for normal
   SGD noise). The simulator should not get stuck at the
   log(VOCAB) ceiling.
2. **Held-out cross-entropy reaches within 1.5× of the clean
   software baseline.** MaxAI on clean float32 hits roughly
   0.2 nats/char on seashells after 5000 steps. The analog
   simulation should reach ~0.3 nats/char or better at the same
   step count.
3. **Generated text is recognizably coherent** — the same
   Gettysburg-completion or seashell-repetition pattern the clean
   MaxAI produces, not garbage bigrams.
4. **The simulator's precision budget matches published analog
   training results.** Compare to IBM's Nandakumar et al. 2020
   phase-change-memory training numbers. Any large divergence is
   a sign the noise model is wrong.

If Gate 1 passes, Gate 2 is worth running.

### Gate 2 — Scaling (does the architecture survive 96-layer depth?)

Take the same MaxAI architecture — same block structure, same
vocabulary, same BPE — and **scale the block count from 2 to 96,
keeping per-layer precision and the nine tricks fixed.** Train
from scratch on the same corpus. Pass criteria:

1. **Training converges.** The deeper model should, if anything,
   converge *faster* per-iteration than the 2-block version (more
   capacity), though wall-time is longer.
2. **Loss does not plateau due to compound analog noise.** This
   is the critical indicator. If loss gets stuck well above the
   clean float32 baseline and diagnostics show it is layer-to-
   layer error accumulation (not optimization failure), the
   architecture does not scale and the GPT-depth claim must be
   retracted.
3. **Output quality is competitive with a clean float32 96-block
   run on the same corpus.** Within 2× on held-out cross-entropy.
   This is a looser bar than Gate 1's 1.5× because deeper models
   are always harder to match precisely — but the point is to
   rule out qualitative failure (generates nothing coherent),
   not to claim we match float32 exactly.

Passing Gate 2 is what makes the hardware architecture
*scalable*, not just a demo. Failing Gate 2 is informative: it
tells us exactly how deep the architecture survives at our
precision budget. If it survives 10 blocks but not 96, that is
still useful information — the hardware is good for medium-sized
models, not frontier. If it survives zero blocks past MaxAI's 2,
we retract the scaling claim entirely and reposition the project
as "small-model only, no path to GPT-scale."

If Gate 2 fails, the diagnostic process is mechanical: identify
which layer the loss plateaus at, isolate the dominant noise
source (compound matmul error? gradient quantization? cell
drift?), and decide whether adding more precision tricks (Q8.8.8.8,
ensemble averaging on critical layers, autozero ADCs on readout)
can close the gap.

### What passing both gates proves

- The architecture is training-viable at MaxAI scale (Gate 1).
- The architecture is scalable to GPT-depth in principle (Gate 2),
  limited only by cell count, not by precision envelope.
- The Stage 1 hardware is a valid miniature of a scalable design,
  not a toy whose lessons don't transfer.

### What only passing Gate 1 proves

- The architecture is training-viable at MaxAI scale.
- Scaling past MaxAI is an open question and should not be
  claimed in presentations or documentation.
- Stage 1 hardware is still worth building for the MaxAI
  demonstration, but the "replaces-GPUs-if-scaled" pitch goes away.

---

## What "failing" looks like

Several possible failure modes, in rough order of severity:

1. **Loss plateaus at the VOCAB ceiling.** Weights cannot be
   updated finely enough to make progress. The gradient
   accumulator is not helping. Solution: larger accumulator,
   smaller learning rate, more aggressive stochastic rounding.
   If none help, the tile noise is simply too high — look at
   lower-noise components or a different storage technology.
2. **Training diverges.** Loss grows unboundedly. The noise
   characteristic is causing gradient explosions. Solution:
   gradient clipping, noise-aware scaling, smaller learning
   rates. If none help, the noise model has fundamental
   instabilities we cannot engineer around.
3. **Training converges but generalization fails.** Held-out
   cross-entropy way worse than training loss. Overfitting to
   the noise pattern itself. Solution: retrain with different
   noise seed per step (noise as regularizer, not as fixed
   corruption).
4. **Training converges but inference is wrong.** Once trained,
   the model produces different outputs in the noisy simulator
   versus a clean test pass. Solution: check that the inference
   noise matches the training noise; quantization-aware training
   assumes deployment conditions match training conditions.

Each failure mode has a specific diagnostic and a specific
remedy. **If none of the remedies work on all failure modes, the
architecture does not train and the project stops.** No hardware
is built.

---

## Estimated effort

The simulator is a reskinning of the book's existing C++
MinAI/MaxAI training loop, with bit-slicing, noise, and
quantization injected at the right points. Most of the training
code already exists; the new code is ~500–800 lines wrapping
calls to forward/backward with the nine tricks above, plus ~200
lines for the Gate 2 96-block scaled architecture.

**Wall time: about two full weeks of focused effort** (up from
one, because Gate 2 is a real scaling experiment that takes time
to train in simulation even on a fast machine). Breakdown:

- Days 1–3: implement the bit-sliced tile + noise model in C++.
- Days 4–5: implement the 32-bit accumulator, calibration cells,
  stochastic rounding, oversampling, and physics-as-lookup
  nonlinearities.
- Day 6: sanity-check against published IBM numbers.
- Days 7–8: run Gate 1 (MaxAI at 2 blocks). Sweep each trick
  on/off to identify which are load-bearing.
- Days 9–12: run Gate 2 (MaxAI architecture scaled to 96 blocks).
  This is the long run — a 96-block network takes hours per
  training epoch even in optimized C++, and you need enough
  epochs to judge convergence.
- Days 13–14: write up results. Go/no-go decision for Stage 1.

Cost: zero dollars. Pure software work on the host you already
have.

---

## What Stage 0 does not have to prove

- **That real hardware will behave exactly like the model.** It
  will not. The simulator's job is to establish that the
  *architecture* is training-viable at our target precision. Real
  hardware will have quirks (unmodeled 1/f noise, specific
  digipot nonlinearities, power-supply coupling) that Stage 1
  measurement will reveal and the noise model will be updated to
  include.
- **That training is competitive with float32 GPU training.** It
  will not be. Quality gap of 1–5% on held-out metrics is
  expected. The bar is "usable" not "equivalent."
- **That the card would win a benchmark.** Benchmarking comes
  much later, on real hardware. Stage 0 is about architectural
  viability, full stop.

---

## Output of Stage 0

Four things, at the end of the two weeks:

1. **A clear Gate-1 pass/fail result** with per-trick sensitivity
   data (which of the nine tricks are load-bearing at MaxAI scale).
2. **A clear Gate-2 pass/fail result** with the maximum block
   depth the architecture survives before accumulated noise
   overwhelms training. Ideally that number is ≥96; acceptable
   outcomes are "scales to 96" (best), "scales to some smaller
   number N" (still useful, reposition the project), or "stays at
   2" (project is small-model-only; scaling story is retracted).
3. **A calibrated noise profile.** Specific numbers for per-MAC
   noise, per-write precision, oversampling requirements,
   accumulator width, and bit-slice count — measured in
   simulation and carried into Stage 1 as the exact hardware
   specification the tile must meet. The Stage 1 PCB design
   becomes a physical implementation of those numbers, not a
   guess.
4. **A `go` or `stop` decision for Stage 1.** If Gate 1 fails,
   stop entirely. If Gate 1 passes but Gate 2 fails, decide
   whether to proceed with a smaller scaling claim or stop. If
   both pass, proceed to Stage 1 with full confidence the
   hardware is a valid miniature of a scalable architecture.

This is the cheapest insurance the project can buy. Use it.
