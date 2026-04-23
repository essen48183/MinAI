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
- That the iteration cycle is genuinely hours: change a part, repopulate,
  measure.

Stage 1 does *not* yet run a full transformer. Its output is
**one row of a matrix multiplication** (one attention head's
projection for one token). That is enough to prove the architecture.

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

| Part | Qty | Role | Unit cost | Notes |
|---|---|---|---|---|
| AD5270BRMZ-20 (or MCP4131-103) | 256 | One weight each | $1–$3 | Single-channel digipots. 256-step (MCP) or 1024-step (AD). SPI. |
| TLV9002 (dual op-amp) | 16 | Summing amplifiers | $0.50 | SOIC-8. Rail-to-rail I/O. |
| MCP4728 (quad 12-bit DAC) | 4 | Input voltage drivers | $3 | I²C. One for every 4 inputs. |
| ADS1115 (4-ch 16-bit ADC) | 4 | Output readers | $4 | I²C. One for every 4 outputs. |
| 1N4148 (signal diode) | 16 | Optional per-column rectifier / ReLU | $0.05 | Only if you want ReLU-style nonlinearity per-column. |
| 2N3904 + 2N3906 matched pair | 8 | Optional differential-pair nonlinearity | $0.10 | For tanh/sigmoid-like transfer curves. |
| 0.1 µF decoupling caps | ~50 | Every IC's power pins | $0.02 | Plus 10 µF bulk near the regulator. |
| AMS1117-3.3 LDO regulator | 1 | Local 3.3 V rail | $0.30 | Fed from Pi's 5 V. |
| 2×20 pin header (Pi hat) | 1 | Mates to Pi Zero 2W | $0.80 | Or JST cable if you don't want a hat. |

Subtotal for tile: **roughly $60–$100**, depending on digipot choice.

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

## Three Stage-1 experiments to run, in order

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
analog hardware.** The rest of the project is scaling, not research.

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

## A week of Stage 1, honestly

Day 1 — sketch the schematic in KiCad; lay out the board.
Day 2 — order parts, send the board to fab.
Day 3–4 — wait for fab; write the Pi-side weight-loader C++.
Day 5 — parts and board arrive; populate one column (Experiment 1).
Day 6 — populate the rest; run Experiment 2.
Day 7 — run Experiment 3 against MinAI's trained Wq.

This is a week for someone with the tooling you described. It is
three months for someone without it. That gap is the whole reason
this side quest is even contemplatable.
