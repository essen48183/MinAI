# Stage 0.5 — first physical board, smallest possible

*A stepping-stone between the Stage 0 simulator (zero hardware) and the
Stage 1 tile (~400 packages, ~$500). Stage 0.5 is a single 4-input
dot product with Q8.8 bit-slicing on a small PCB. The job is to make
the first board mistakes cheap, on a board whose BOM is under $50 and
whose layout is small enough to KiCad in an afternoon.*

---

## Why this exists

The Stage 1 board has ~400 dual-channel digipot packages, 16 op-amps,
8 reference cells, 4 DACs, 4 ADCs, two SRAMs, and a 6-or-8-layer
mixed-signal layout. That is a lot of board-design unknowns to clear
on the first attempt. The first physical board almost always has
errors — bad footprints, missing decoupling, ground-loop traces,
flipped pin assignments, wrong package selection. Catching them at
Stage 1 scale wastes $500 and ten days of fab turnaround per mistake.

Stage 0.5's only job is to make those mistakes on a board that costs
$25 to fab and 4 chips to populate, whose schematic is small enough
to read in one breath, and whose firmware reuses everything Stage 1
will need.

It is *not* a simplified version of the architecture — it is a
miniaturized version. Same parts, same SPI command set, same op-amp
family, same KiCad symbols and footprints. Every error you debug at
Stage 0.5 is an error you don't debug at Stage 1.

---

## What Stage 0.5 proves

- KiCad workflow (schematic → board → fab files → assembly drawings).
- Pi-side SPI orchestration of the **MCP4251 digipot** (the same part
  Stage 1 uses) — addressing, command set, write-and-readback.
- The op-amp summing primitive works in physical silicon: drive a
  known input, observe the predicted output voltage.
- **The Q8.8 bit-slice combiner works.** Two summing stages plus a
  binary-weighted final stage produce the same result as a software
  matmul with reconstructed Q8.8 weights, to within Stage 0's
  measured noise budget.
- (Optional, end of Stage 0.5) one stochastic-rounded weight update
  through the Pi-side firmware lands the digipot taps where the
  software model says it should.

## What Stage 0.5 does *not* prove

- 16-wide parallel summing (Stage 1 territory).
- Q8.8.8 three-slice bit-slicing (Stage 1 territory; Stage 0.5 stops
  at Q8.8 to keep the slice count and combiner topology minimal).
- Calibration reference cells (Stage 1 — too small a tile to need
  them; bench measurements stand in for calibration).
- 32-bit on-board gradient accumulator (still on the Pi for Stage
  0.5; the SRAM line item moves to Stage 1).

---

## The shape: 4×1 Q8.8

Four input voltages, one output voltage. Four weights, each
represented as two 8-bit digipot slices summed with binary weighting
1 : 1/256.

```
                Q8.8 weight = (MSB_tap × 1) + (LSB_tap × 1/256)
                    ↑                              ↑
              MCP4251 ch.A                   MCP4251 ch.B
              (same die as ch.B for 15 ppm/°C ratiometric tempco)
```

Total: 4 weights × 2 slices = 8 digipot channels, packaged as 4
dual-channel MCP4251s. **Pair MSB+LSB of the same weight onto the
same MCP4251 die** so on-die ratiometric matching covers the
binary-weighted ratio.

---

## Bill of materials

| Part | Qty | Role | Unit cost | Subtotal |
|---|---|---|---|---|
| MCP4251-103E/P (or -E/ML QFN) | 4 | Dual 8-bit volatile SPI digipot, 10 kΩ. One package = one Q8.8 weight (MSB+LSB on the same die). | $1.01 | $4 |
| AD8629 (dual autozero op-amp) | 2 | Three op-amp stages: MSB summer, LSB summer, combiner. One AD8629 = 2 op-amps; two packages give us 4 channels with one spare for a debug buffer. | $2.00 | $4 |
| Precision thin-film 1 % feedback resistors (10 kΩ, 2.56 MΩ, plus matched-pair ratios) | ~10 | Sets the unity and 1/256 weights at the combiner stage. 0.01 % thin-film for the binary-weighted pair only; the rest can be 1 % thick-film. | $0.30 | $3 |
| MCP4728 (quad 12-bit DAC) | 1 | Drives the 4 input voltages over I²C. | $3.00 | $3 |
| ADS1115 (4-ch 16-bit ADC) | 1 | Reads the single output voltage; spare channels available for the MSB summer and LSB summer mid-points (instructive for debug). | $4.00 | $4 |
| AMS1117-3.3 LDO regulator | 1 | Local 3.3 V rail from the Pi's 5 V. | $0.30 | $0.30 |
| 0.1 µF ceramic decoupling caps | ~20 | One per IC power pin, plus a 10 µF bulk near the regulator. | $0.02 | $0.40 |
| 2×20 pin header (Pi hat) | 1 | Mates to Pi Zero 2W. | $0.80 | $0.80 |
| 4-layer PCB, ~50 × 50 mm, JLCPCB / PCBWay | 5 | The board itself; fab houses ship 5 minimum. | ~$5 | $25 |

**Tile-side total: ~$45.** With Pi Zero 2W ($15), microSD ($10), and
USB power supply ($10), **all-in stand-up cost: ~$80.**

That is one-tenth of Stage 1's BOM and roughly the cost of dinner.

---

## Schematic (one summing column at Q8.8 depth)

```
   V[0] --[MCP4251 #1 ch.A : MSB slice w0]--+
   V[1] --[MCP4251 #2 ch.A : MSB slice w1]--+
   V[2] --[MCP4251 #3 ch.A : MSB slice w2]--+--> (-) op-amp A  ----> V_msb
   V[3] --[MCP4251 #4 ch.A : MSB slice w3]--+         |
                                                     [Rf_a]  feedback

   V[0] --[MCP4251 #1 ch.B : LSB slice w0]--+
   V[1] --[MCP4251 #2 ch.B : LSB slice w1]--+
   V[2] --[MCP4251 #3 ch.B : LSB slice w2]--+--> (-) op-amp B  ----> V_lsb
   V[3] --[MCP4251 #4 ch.B : LSB slice w3]--+         |
                                                     [Rf_b]  feedback

           V_msb ----[ R = 10 kΩ ]-----+
                                       |---> (-) op-amp C ---> V_out
           V_lsb ----[ R = 2.56 MΩ ]---+              |
                                                    [Rf_c]
```

The 10 kΩ : 2.56 MΩ ratio at the combiner stage is the **binary
weighting**: V_lsb gets divided by 256 before summing with V_msb.
Use **0.01 % thin-film matched-pair resistors** for these two —
ratio precision matters far more than absolute value here, since the
absolute scale is absorbed by software calibration.

The two summing stages (A and B) use the same 10 kΩ feedback
resistor. They are two instances of the same circuit; copy the
schematic block and rename nets.

---

## Pi-side firmware (sketch)

The firmware reuses the Stage 1 codebase entirely. The MCP4251 SPI
interface is documented in Microchip DS22060B; one register write
per channel sets the wiper tap.

```cpp
// Program one Q8.8 weight (8 MSB bits + 8 LSB bits) into chip i.
void program_weight(int i, uint16_t q88_taps) {
    uint8_t msb = (q88_taps >> 8) & 0xFF;   // slice A, range 0..255
    uint8_t lsb = q88_taps & 0xFF;          // slice B, range 0..255
    spi_write_mcp4251(chip_select[i], /*wiper_id=*/0, msb);
    spi_write_mcp4251(chip_select[i], /*wiper_id=*/1, lsb);
}

// Forward pass: drive 4 inputs, read 1 output.
float forward(const float x[4]) {
    for (int i = 0; i < 4; ++i)
        dac_write(i, scale_input_to_voltage(x[i]));
    delay_us(10);                 // op-amp settling
    return scale_reading_to_float(adc_read(/*ch=*/0));
}
```

Stochastic rounding for the optional Experiment 3 below uses the
same code path Stage 1 will use, just with two slices instead of
three. The simulator's `analog_sim.cpp` is the reference
implementation.

---

## Three Stage 0.5 experiments

### Experiment 1 — "does the MSB summing stage work?"

Set all four MSB digipot wipers to mid-scale (tap 128 of 256). Drive
one input at a time with a step from 0 V → 1 V, others held at 0 V.
The MSB op-amp's output should step proportionally — same magnitude
each time within the 0.1 % bench measurement. Step all four inputs
together; output is roughly 4× a single step. **You have just done
analog summation in physics.**

### Experiment 2 — "does the Q8.8 combiner work?"

Program a known set of 4 Q8.8 weights via the MSB + LSB wipers. Drive
4 known inputs. Compare the measured output voltage to the software
dot-product of the same Q8.8 weights and inputs computed against the
Stage 0 simulator's `Tile::forward`. Match should be within the
σ ≈ 0.0003 per-MAC budget × √4 = 0.06 % of full scale.

If it doesn't match, the diagnostic surface area is small: probe
V_msb and V_lsb separately (the spare ADC channels are wired for
exactly this), and check whether the binary-weighted combiner is
giving 1 : 1/256 within 1 %.

### Experiment 3 — "does one stochastic-rounded weight update execute?"

Program a starting weight w. Compute a small weight update Δw
digitally on the Pi. Decompose Δw into integer-tap deltas across MSB
and LSB slices using the simulator's `write_delta` logic.
Stochastically round the fractional residue and commit it via SPI
writes. Read back the wiper register from each MCP4251 and confirm
the taps moved exactly where the firmware predicted.

If they did, **the Stage 0.5 board is done** and the firmware is
Stage-1-ready.

---

## Success criteria

Stage 0.5 passes when all three experiments pass to the precisions
above. The deliverables that move into Stage 1 unchanged:

- KiCad symbols and footprints for MCP4251, AD8629, MCP4728, ADS1115,
  Pi header.
- Pi-side firmware: SPI MCP4251 driver, I²C MCP4728 driver, I²C
  ADS1115 driver, Q8.8 weight decomposition + stochastic rounding.
- Layout patterns: ground pour discipline, decoupling placement,
  star ground point, mixed-signal separation between digital SPI/I²C
  traces and analog summing nodes.
- A working physical primitive whose measured noise can be compared
  to the simulator's prediction at σ = 0.0003.

If Experiment 2 lands at output RMS > 1 % of full scale, the board
has a layout or component-selection problem; do not advance to
Stage 1 until that's resolved on the small board, where iteration is
cheap.

---

## What this does *not* try to do

- Run a full transformer matmul (4×1 is one row of one tile).
- Validate Q8.8.8 (Stage 1's third-slice combiner — same topology
  as the Q8.8 combiner here, just one more stage of binary weighting).
- Train MaxAI through hardware (still software-only at this stage).
- Address negative weights with differential pairs (Stage 1.5).

## Estimated time and cost

- **Day 1**: Schematic in KiCad (one summing column, three op-amps,
  one DAC, one ADC, Pi header). Component selection, BOM finalize.
- **Day 2**: Layout. Should fit on a 50 × 50 mm 4-layer board with
  comfortable density. Generate fab files; submit to JLCPCB or
  PCBWay (3-day turnaround at this size).
- **Day 3–5**: Wait for fab. Write the Pi-side SPI driver.
- **Day 6**: Populate by hand (~20 surface-mount parts, an evening of
  soldering with a fine tip and flux paste).
- **Day 7**: Bring-up. Run Experiments 1–3.
- **Day 8**: Characterize noise; compare to Stage 0 simulator.

**One week from KiCad to a measured first board.** All-in cost
including PCB fab and Pi: ~$80. Iteration cost on a layout
respin: ~$25 + 3 days.

This is the rate at which board-level mistakes should be made.
Stage 1 begins after Stage 0.5 lands a clean Experiment 2.
