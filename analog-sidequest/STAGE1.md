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

Stage 1 assumes Stage 0 (the simulator viability gate, see
`STAGE0.md`) has already passed. If it hasn't, do not fabricate
this board. The Stage 1 BOM and time budget below are committed
only after Stage 0 gives a `go`.

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
Stage-0 simulator validated: **three digipots per weight** giving
24-bit effective precision from 8-bit parts. BOM scales
accordingly.

| Part | Qty | Role | Unit cost | Notes |
|---|---|---|---|---|
| AD5270BRMZ-20 (or MCP4131-103) | **768** | **Three per weight (Q8.8.8 bit-slicing)** | $1–$3 | Single-channel digipots. 256-step (MCP) or 1024-step (AD). SPI. Organized in three planes: MSB, middle, LSB. Binary-weighted feedback at the summing op-amp sums the three planes with coefficients 1, 1/256, 1/65536. |
| **AD8629 (dual autozero op-amp)** | **16** | **Summing amplifiers — autozero for 20+ bit path precision** | $2 | Replaces the generic TLV9002. Essential for exploiting the 24-bit weight precision. |
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

Subtotal for tile: **roughly $200–$300**, roughly 3× the cost of
the naive Q8.8 version. Still well under $500 all-in for the
first board.

If that cost is the binding constraint in a given iteration, a
Q8.8 (single-cell) variant is a valid "half-bill" prototype —
you can etch the single-slice version in-house first to debug
topology, then move to the Q8.8.8 pro-fab version once the
single-slice topology is working. But the Gate-0 simulator
passes *only* for the Q8.8.8 configuration; a single-cell build
will not train at 96-block depths and is explicitly not a
miniature of the scalable architecture. It is a stepping stone
for your workflow, not a representative demonstration.

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
against the same weights. Stage-1 success: the two vectors match
within 1–2% RMS across 1000 random test inputs.

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

The Q8.8.8 build has 3× the cell count and 3× the solder joints,
so the cadence stretches accordingly. Assuming Stage 0 has
already passed both gates.

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
heavy PCBA work rather than soldering 800+ joints by hand. It
is three-plus months for someone without pro-fab assembly. The
pro-fab line item is therefore less optional at Stage 1 than it
was in the naive Q8.8 version.
