# Stage 1 — single analog MAC tile, Pi Zero host

*The first physical build. One PCB, one Pi, one demonstrable result.
Target: an afternoon to lay out the board, a day to populate it, an
afternoon to verify it against the NumPy/C++ reference.*

---

## What Stage 1 proves

- That an analog multiply-accumulate, built from off-the-shelf SMD
  parts, matches digital matmul to within measurable tolerance.
- That a Raspberry Pi Zero 2W can load weights, orchestrate
  inference, and read results via SPI in software that fits in
  a few hundred lines of C++ or Python.
- **That one full training step — forward pass, gradient
  computation, stochastic-rounded weight update through the
  digital accumulator — executes cleanly on physical hardware.**
  This is the piece that distinguishes this project from every
  commercial analog-AI attempt.
- That the iteration cycle is genuinely hours: change a part,
  repopulate, measure.

Stage 1 does *not* yet run a full transformer. Its output is
**one row of a matrix multiplication** (one attention head's
projection for one token), run both forward and backward. That is
enough to prove the architecture is *trainable*, not just
inferable — which is the whole point.

Stage 0 (the simulator viability gate, see `STAGE0.md`) passed
both gates — the go/no-go is in `STAGE1_GONOGO.md`. The Stage 1
BOM and tolerances below incorporate what Stage 0 actually
measured, which is often tighter than the numbers originally
predicted and in one case (noise at depth) is unexpectedly
looser.

**Strongly recommended precursor: Stage 0.5** (see `STAGE0_5.md`).
A 4×1 Q8.8.8 dot-product board, ~$85 all-in, that uses the same
MCP4251 / AD8629 parts and the same three-slice / cascaded-combiner
topology Stage 1 will use, in a one-summing-column miniature. Its
job is to surface KiCad-and-fab mistakes on a $25 PCB rather than
a $500 one. Skipping Stage 0.5 is allowed but every error you
would have caught on the small board, you will catch on the big
board, with longer fab cycles and more expensive populated parts.

---

## The tile shape: 16×16

One tile holds 16 inputs, 16 outputs, 256 weights. Why 16:

- Small enough to fit on a 100 mm × 100 mm 4-layer PCB at a
  comfortable density. Every hobbyist fab house can do that board
  for a few dollars per copy.
- Large enough that real matmul behaviour is visible (parallel
  summation across 16 inputs).
- MinAI's `D_MODEL = 16`, so **one tile exactly matches MinAI's
  hidden dimension.** A single tile IS one of MinAI's linear layers
  (Wq, Wk, Wv, or Wout's reverse shape) for one block. This is not
  an accident. It is the main reason this project is tractable.

---

## Bill of materials (Stage-1 tile + Pi + cabling)

All prices approximate, 2026 retail, ordered for a single board's
worth.

### On the tile PCB

Stage 1 implements the Q8.8.8 bit-sliced architecture the
Stage-0 simulator validated: **three digipots per weight**. The
ceiling is ~24 bits effective (zero-noise); the locked Stage 1
target is 11 bits effective (σ ≤ 0.0003 per-MAC, the
standard-analog regime Gate 1/Gate 2 both trained at). The bit-
sliced structure is needed because we cannot guarantee an
8-bit cell's step size sits below the training-relevant
gradient; three slices gives enough headroom for stochastic
rounding to resolve updates much finer than one tap.

| Part | Qty | Role | Unit cost | Notes |
|---|---|---|---|---|
| **MCP4251-103E/P** (or -E/ML QFN) | **384** | **Three slices per weight (Q8.8.8 bit-slicing), two channels per package** | $1.01 | **Dual 8-bit SPI digipots, 10 kΩ, volatile RAM.** 257 taps (8-bit) per channel, 10 MHz SPI, 75 Ω wiper. Datasheet DS22060B. The dual packaging cuts chip count 2× vs single-channel parts, and — critically — the two channels on one die have **15 ppm/°C ratiometric tempco** (matched on-die). Route same-weight slice pairs (e.g., MSB + middle) onto the same package so the tightest-coupled ratio pair benefits from the on-die matching. Organized in three slice planes: MSB, middle, LSB. Binary-weighted feedback at the summing op-amp sums the three planes with coefficients 1, 1/256, 1/65536. Single-channel MCP4131 is an acceptable fallback at ~$0.70 but loses the ratiometric-matching benefit. |
| **AD8629 (dual autozero op-amp)** | **16** | **Summing amplifiers — autozero for drift and 1/f stability** | $2 | Replaces the generic TLV9002. Chosen for its ≤ 1 µV Vos and near-zero drift over temperature. That drift stability — not the 24-bit ceiling — is what keeps per-MAC noise inside the 0.03 % standard-analog budget over a training run. A generic op-amp's thermal Vos drift alone would exceed that budget. |
| MCP4728 (quad 12-bit DAC) | 4 | Input voltage drivers | $3 | I²C. Also used for write-path dither injection during stochastic rounding. |
| ADS1115 (4-ch 16-bit ADC) | 4 | Output readers | $4 | I²C. Oversampling recovers additional bits. |
| **23LC1024 SRAM × 2** | **2** | **32-bit-wide gradient accumulator** | **$3 ea** | **Widened from 16-bit to 32-bit per weight. Two SRAMs chained on SPI, giving 256 KB total accumulator storage.** |
| **AD5270BRMZ (reference)** | **8** | **Calibration reference cells** | $2 | Programmed to known values, read periodically for drift correction. ~3% overhead. |
| 1N4148 (signal diode) | 16 | Per-column rectifier / ReLU | $0.05 | Analog ReLU without host round-trip. |
| 2N3904 + 2N3906 matched pair | 8 | Optional differential-pair nonlinearity | $0.10 | For tanh/sigmoid-like transfer curves. |
| NTCG063JF103 thermistor | 1 | Per-tile temperature reference | $0.50 | Feeds the Pi-side calibration loop. |
| 0.1 µF decoupling caps | ~100 | Every IC's power pins | $0.02 | Scale with BOM. Plus 10 µF bulk near the regulator. |
| AMS1117-3.3 LDO regulator | 1 | Local 3.3 V rail | $0.30 | Fed from Pi's 5 V. |
| 2×20 pin header (Pi hat) | 1 | Mates to Pi Zero 2W | $0.80 | Or JST cable if you don't want a hat. |

Subtotal for tile with the MCP4251 digipot line: **roughly $500–$600**
(digipots ~$390, op-amps ~$32, precision feedback/summing network,
SRAM, DACs, ADCs, reference cells, passives, regulator, PCB fab).
Higher than the original $200–$300 estimate, which was written before
Stage 0 validated the Q8.8.8 bit-sliced architecture as the committed
path. Still well under $1,000 all-in for the first board.

If that cost is the binding constraint in a given iteration, a
Q8.8 (single-cell) variant is a valid "half-bill" prototype —
you can etch the single-slice version in-house first to debug
topology, then move to the Q8.8.8 pro-fab version once the
single-slice topology is working.

A Stage 0 surprise re-opened this fallback. Gate 2 showed that
even hobbyist-grade noise (σ = 0.003, roughly a single-cell Q8
tile's effective tolerance) still trained MaxAI cleanly at
96-block depth — in fact with *lower* loss than the clean
float32 baseline, because analog noise acts as regularization in
deep stacks. This does not promote Q8.8 to the production
architecture, but it does mean a single-cell bring-up board is no
longer categorically "won't train." It is a cheaper learning
vehicle than we previously thought. The Q8.8.8 configuration
remains the committed Stage 1 build because it hits the
standard-analog precision target with margin; Q8.8 hits the
hobbyist regime with almost none.

If you want to reduce component count at the expense of dynamic
range, you can use 8-channel digipots like the AD5204 (4 × 256-step
channels in a single package) and bring the chip count from 256 to
32. The BOM drops but each channel's stability is slightly worse.
Stage 1 is a good place to try both.

### Host side

| Part | Qty | Role | Unit cost |
|---|---|---|---|
| Raspberry Pi Zero 2W | 1 | Host, SPI master | ~$15 |
| 32 GB microSD | 1 | Holds OS + trained weights | ~$10 |
| Power supply (5 V 2 A, micro-USB) | 1 | For Pi + board | ~$10 |

Host subtotal: **~$35**.

### Tools you need but may already own

- A USB-C oscilloscope or any 100 MHz scope. For seeing that the
  summing nodes actually settle and for verifying digipot programming
  pulses.
- A multimeter.
- A small soldering station with a fine tip, or reflow oven and
  stencil if you want to reflow.
- KiCad for schematic + layout. Free.
- A PCB fab — your in-house one.

**All-in cost to stand up Stage 1: under $150 of new parts plus your
PCB fab time.**

---

## Schematic architecture (one column at a time)

A single output column `j` looks like:

```
   V[0] ----[ digipot G[0,j] ]----+
                                   |
   V[1] ----[ digipot G[1,j] ]----+
                                   |
   V[2] ----[ digipot G[2,j] ]----+----> (-) input of op-amp
                                   |     (virtual-ground summing node)
   ...                             |
                                   |
   V[15]----[ digipot G[15,j] ]---+
                                   |
                                  [Rf]   feedback resistor
                                   |
                                   +----> (-) -> op-amp output = Y[j]
   (+) input of op-amp tied to ground through matching resistor
```

The op-amp holds the (-) node at virtual ground. All currents from
the digipots sum at that node by Kirchhoff's law. The op-amp sinks
that total current through `Rf`, producing:

```
Y[j] = -Rf × Σ_i ( V[i] / R[i,j] )
     = -Rf × Σ_i ( V[i] × G[i,j] )
```

Which is exactly the matrix-vector product, up to the sign of the
feedback resistor (easily flipped with a follow-up inverting stage,
or simply compensated for in software).

Sixteen of these columns in parallel, sharing the same row-wire input
layer, is the whole tile.

---

## Routing notes when you lay the board out

A couple of decisions that matter at schematic/layout time and are
easy to miss.

- **Pair slice roles on-die.** Each MCP4251 package holds two wiper
  channels sharing one silicon die — 15 ppm/°C ratiometric tempco.
  Route the **MSB and middle slices of the same weight** onto the
  two channels of one MCP4251. That's the tightest-coupled ratio
  pair in the tile (they sum with a 256:1 weighting, so any drift
  between them shifts the effective weight value directly). The LSB
  slice can live on a separate package — its contribution is already
  scaled by 1/65536, so drift between LSB and the other slices is
  three orders of magnitude less painful.

- **Ask the fab for 6-layer AND 8-layer quotes.** The original STAGE1
  timeline budgeted an 8-layer board with careful stripline routing
  because the single-channel MCP4131 line put ~768 parts on the
  board. MCP4251 halves the package count, which can plausibly fit
  on 6 layers — saving ~30 % of fab cost and 2–3 days of fab lead
  time. Get both quotes from JLCPCB/PCBWay before committing; the
  6-layer number lets you decide if the layout-density savings pay
  for the routing discipline 8 layers buys you.

---

## The weight-loader, in Pi-side software

Pseudocode for the boot flow, in C++ (Pi side):

```cpp
// 1. Read trained weights from disk.
std::vector<float> weights = load_weights("/boot/minai_Wq.bin");

// 2. Quantize each weight to the digipot's tap range (0..255 for an MCP4131).
for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
        float w = weights[i * 16 + j];
        uint8_t tap = scale_weight_to_tap(w);   // maps [-w_max, +w_max] -> [0, 255]
        // 3. Send SPI command to the specific digipot at crosspoint (i, j).
        spi_write_digipot(chip_select(i, j), tap);
    }
}
// All 256 digipots are now programmed. Board is ready.
```

The `scale_weight_to_tap` function is the only numerically interesting
piece: a digipot can only represent positive resistance, so negative
weights need a differential representation (one digipot as the
positive part, one as the negative part, and a differential op-amp
subtracts them). Stage 1 can cheat by adding a bias so all weights
are positive; proper differential pairs come in Stage 1.5.

### Running one inference

```cpp
// Input: 16 values for one token embedding.
float x[16] = { ... };

// 1. Drive the input voltages via the MCP4728 DAC array.
for (int i = 0; i < 16; ++i)
    dac_write(i, scale_input_to_voltage(x[i]));

// 2. Wait for op-amps to settle (a few microseconds).
delay_us(10);

// 3. Read the 16 outputs via the ADS1115 ADC array.
float y[16];
for (int j = 0; j < 16; ++j)
    y[j] = scale_reading_to_float(adc_read(j));

// y is now X @ Wq for this one token.
```

Compare `y` to the output of the same matmul computed in pure C++
against the same weights.

**Stage 1 success bar — tightened from the pre-Stage-0 number.**
Originally the bar was "within 1–2 % RMS." After Stage 0, we know
what the RMS actually has to be to land inside the training
budget Gate 2 validated. With 16 summed MACs per output, the
output RMS is ≈ √16 × σ_perMAC = 4 σ_perMAC. So:

- **Target**: output RMS ≤ **0.1 %** of full scale across 1000
  random inputs. That corresponds to σ_perMAC ≈ 0.0003 — the
  standard-analog budget Gate 2 trained at with margin. This is
  the bar the pro-fab Q8.8.8 build should clear.
- **Minimum viable**: output RMS ≤ **0.3 %** of full scale. That
  corresponds to σ_perMAC ≈ 0.0008 — between standard and
  hobbyist regimes. Training still works (Gate 2 showed σ = 0.003
  trains at 96 blocks), but margin is thin.
- **Original "1–2 % RMS" bar is wrong** in hindsight: that
  corresponds to σ_perMAC ≈ 0.003, the hobbyist regime with no
  margin. Gate 2 said this still trains, but the bench should
  diagnose, not tolerate, any tile that lands there.

---

## Four Stage-1 experiments to run, in order

### Experiment 1: "Does one column work?"

Build a single output column (one op-amp, 16 digipots). Program all
16 digipots to the same resistance. Drive one row at a time with a
step voltage. Observe the column output step proportionally. Varies
like a DAC.

If this works, you have already proven in-hardware analog summation.

### Experiment 2: "Does the column do a dot product?"

Program the 16 digipots to a known weight vector. Drive the 16 rows
with a known input vector. Compare the output to the software
`dot(X, W)`.

### Experiment 3: "Does the full 16×16 tile run MinAI's first matmul?"

Load the trained `Wq` weights from the book's C++ output. Drive the
input embedding of the first token of MinAI's fixed example. Compare
the 16 output values to the `Q[0][:]` vector a software MinAI
produces.

If this matches within 1–2%, **you have run MinAI's first matmul in
analog hardware.**

### Experiment 4: "Does one full training step execute on the tile?"

This is the experiment that distinguishes this project from every
prior analog-AI hardware attempt.

1. Load a set of starting weights (from a mid-training MinAI
   checkpoint, where the gradients are non-trivial but training
   has not converged) into the tile.
2. Run one forward pass: input token → analog matmul → output
   vector.
3. Read the output via ADC; compute the expected gradient
   digitally on the Pi (the book's C++ backward pass, limited to
   one cell for this experiment).
4. For each of the 256 weights, compute the SGD update. Accumulate
   the update in the on-board SRAM (16-bit precision).
5. For any accumulator crossing the digipot step threshold,
   compute a stochastic-rounded write — ±1 tap with the
   appropriate probability — and commit it to the analog cell.
   Subtract the committed step from the accumulator.
6. Run a second forward pass with the updated weights. Compare
   the output vector against the software reference.

**If the output moves in the direction gradient descent predicted,
within the precision expected from the noise model**, Stage 1 is
complete and you have demonstrated **the first training step of
any analog AI system built outside a research lab**. That is the
artifact on which the rest of the project stands.

The quantitative bar is: update magnitude matches the software-
computed SGD step within the same 0.1 % RMS margin the forward
pass hit in Experiment 3. Gate 2 proved 96-block training works
at σ = 0.003 (hobbyist); Stage 1's 0.0003 target has a
comfortable 10× margin, so a Stage 1 tile that runs one correct
training step is nowhere near the cliff. If the step magnitude is
wildly off, the fault is on the bench (SPI, wiper, accumulator
logic) — not at the architectural margin.

If it doesn't move the expected direction, the diagnostic is
straightforward:

- Wrong magnitude but right sign → scale the learning rate; the
  gradient is being accumulated correctly but the commit is
  saturating.
- Right magnitude, wrong sign → sign error somewhere in the
  write path or the gradient sign convention. Easy to find.
- No motion at all → the accumulator is not crossing the threshold
  (too-small gradients) or the write is not actually sticking (a
  hardware fault). Probe the SPI and the digipot wipers.

Reaching Experiment 4 cleanly is the condition for Stage 2 to
begin.

---

## What Stage 1 does not yet handle

- **Negative weights.** Addressed with differential pairs in Stage 1.5.
- **Nonlinearity.** Only done digitally on the Pi for Stage 1. Move
  to analog in Stage 2.
- **Scale of more than one layer.** Stage 2 adds tiles.
- **Temperature drift.** Stage 1 is short-run; calibration comes in
  Stage 2.
- **LayerNorm.** Digital on the Pi. Move to analog much later, if at
  all.

---

## Two-plus weeks of Stage 1, honestly

The Q8.8.8 build has 3× the cell count of a naive Q8.8 design.
Dual-channel MCP4251 packaging cuts the chip count back down to
~1.5× (384 dual chips vs 256 single chips for Q8.8), so the
cadence stretches less than it would with single-channel
digipots. Stage 0 passed both gates; build cadence assumes that.

Day 1–2 — sketch the schematic in KiCad; lay out the 8-layer
          board with careful stripline routing for the summing
          buses; include the 32-bit SRAM, autozero op-amps, and
          calibration reference cells.
Day 3 — order parts, send the board to pro fab (expect 10-day
          turnaround at JLCPCB/PCBWay 8-layer PCBA service).
Day 4–10 — wait for fab; write the Pi-side weight-loader with
          Q8.8.8 decomposition, the 32-bit accumulator manager,
          stochastic-rounded write code per slice, and the
          calibration-reference readout / correction loop.
Day 11 — parts and board arrive; populate one column across all
          three slices (Experiment 1, extended).
Day 12 — populate the rest; run Experiment 2.
Day 13 — run Experiment 3 against MinAI's trained Wq.
Day 14 — run Experiment 4 (the training step).
Day 15 — characterize tile noise across all three slices; update
          the Stage-0 simulator's noise model with measured
          values. Any discrepancy between simulated and measured
          behavior triggers a deeper investigation before Stage 2.

**Two to three focused weeks** for someone with the tooling you
described, assuming you are willing to let a pro fab do the
heavy PCBA work rather than soldering ~400 dual-channel digipot
packages (≈ 3,200 pins) by hand. It is three-plus months for
someone without pro-fab assembly. Dual-channel MCP4251 packaging
roughly halves the joint count vs the single-channel line, but
the density still argues for pro-fab PCBA at Stage 1.
