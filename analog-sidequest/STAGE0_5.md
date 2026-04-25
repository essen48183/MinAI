# Stage 0.5 — first physical board, full Q8.8.8 in miniature

*A stepping-stone between the Stage 0 simulator (zero hardware) and the
Stage 1 tile (~400 packages, ~$500). Stage 0.5 is a single 4-input
dot product with **the same Q8.8.8 bit-slicing topology Stage 1 will
use**, just one summing column wide instead of sixteen. The job is
to make the first board mistakes cheap, on a board whose BOM is
under $55 and whose layout is small enough to KiCad in an afternoon.*

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
- **The full Q8.8.8 bit-slice combiner works.** Three summing stages
  (MSB, middle, LSB) plus two cascaded binary-weighted combiner
  stages produce the same result as a software matmul with
  reconstructed Q8.8.8 weights, to within Stage 0's measured noise
  budget.
- (Optional, end of Stage 0.5) one stochastic-rounded weight update
  through the Pi-side firmware lands the digipot taps where the
  software model says it should.

## What Stage 0.5 does *not* prove

- 16-wide parallel summing (Stage 1 territory; Stage 0.5 builds one
  summing column. Stage 1 is sixteen copies of the same column
  sharing the input bus).
- Calibration reference cells (Stage 1 — too small a tile to need
  them; bench measurements stand in for calibration).
- 32-bit on-board gradient accumulator (still on the Pi for Stage
  0.5; the SRAM line item moves to Stage 1).

---

## The shape: 4×1 Q8.8.8

Four input voltages, one output voltage. Four weights, each
represented as three 8-bit digipot slices summed with binary
weighting 1 : 1/256 : 1/65536.

```
   Q8.8.8 weight = (MSB_tap × 1) + (MID_tap × 1/256) + (LSB_tap × 1/65536)
```

Total: 4 weights × 3 slices = 12 digipot channels, packaged as 6
dual-channel MCP4251s. **Pair MSB + middle slices of the same
weight on the same MCP4251 die** — that is the tightest-coupled
ratio (256:1) and the on-die 15 ppm/°C ratiometric tempco cancels
the drift between them. The LSB slice can live on a separate
package because its contribution is already scaled by 1/65536, so
drift between LSB and the other slices is three orders of magnitude
less painful. Six chips × two channels = exactly 12 channels, no
spares wasted: chips 1–4 hold the four (MSB, middle) pairs; chips
5–6 hold the four LSB slices (two per package).

---

## Bill of materials

| Part | Qty | Role | Unit cost | Subtotal |
|---|---|---|---|---|
| MCP4251-103E/P (or -E/ML QFN) | 6 | Dual 8-bit volatile SPI digipot, 10 kΩ. Chips 1–4 hold MSB + middle slices of weights w0…w3 (paired on-die for 15 ppm/°C ratiometric tempco). Chips 5–6 hold the four LSB slices (two weights per package). | $1.01 | $6 |
| **74HC138** (or 74HCT138, SN74HC138) | 1 | **3-to-8 decoder for chip select.** Drives 6 of 8 outputs to the MCP4251 CS pins. Same architectural shape Stage 1 will use at scale (cascaded decoders for 384 CS lines). | $0.30 | $0.30 |
| AD8629 (dual autozero op-amp) | 3 | Five op-amp stages: MSB summer, middle summer, LSB summer, first combiner (MSB + middle/256), second combiner ((MSB+mid/256) + LSB/256). Three AD8629 packages give 6 channels — five used, one spare for a debug buffer. | $2.00 | $6 |
| Precision 0.01 % thin-film matched-pair feedback resistors (10 kΩ : 2.56 MΩ, two pairs for the cascaded combiner) | 4 | Set the 1 : 1/256 weighting at each of the two combiner stages. Ratio precision matters far more than absolute value; matched-pair 0.01 % parts make this a non-issue. | $0.30 | $1.20 |
| 1 % thick-film feedback resistors for the three summers (10 kΩ) | ~10 | Set the gain of the three slice-summing op-amps. 1 % is fine; the absolute scale is absorbed by software calibration. | $0.05 | $0.50 |
| MCP4728 (quad 12-bit DAC) | 1 | Drives the 4 input voltages over I²C. | $3.00 | $3 |
| ADS1115 (4-ch 16-bit ADC) | 1 | Reads the final output voltage; spare channels probe V_msb, V_mid, V_lsb individually for debug. | $4.00 | $4 |
| AMS1117-3.3 LDO regulator | 1 | Local 3.3 V rail from the Pi's 5 V. | $0.30 | $0.30 |
| 0.1 µF ceramic decoupling caps | ~25 | One per IC power pin, plus a 10 µF bulk near the regulator. | $0.02 | $0.50 |
| 2×20 pin header (Pi hat) | 1 | Mates to Pi Zero 2W. | $0.80 | $0.80 |
| 4-layer PCB, ~50 × 50 mm, JLCPCB / PCBWay | 5 | The board itself; fab houses ship 5 minimum. | ~$5 | $25 |

**Tile-side total: ~$50.** With Pi Zero 2W ($15), microSD ($10), and
USB power supply ($10), **all-in stand-up cost: ~$85.**

That is one-tenth of Stage 1's BOM and roughly the cost of dinner.

---

## Schematic (one summing column at Q8.8.8 depth)

Three slice summers feed two cascaded binary-weighted combiners:

```
  Slice summers (each is the same inverting-summer circuit, 10 kΩ feedback):

   V[0..3] --[ chips 1-4 ch.A : MSB slices ]-+--> (-) op-amp A  --> V_msb
                                              |        |
                                             [Rf_a = 10 kΩ]

   V[0..3] --[ chips 1-4 ch.B : MID slices ]-+--> (-) op-amp B  --> V_mid
                                              |        |
                                             [Rf_b = 10 kΩ]

   V[0..3] --[ chips 5-6 (8 ch) : LSB slices]-+--> (-) op-amp C  --> V_lsb
                                              |        |
                                             [Rf_c = 10 kΩ]

  First combiner (combine MSB and middle at 1 : 1/256):

       V_msb --[ R = 10 kΩ ]----+
                                 |---> (-) op-amp D ---> V_a = V_msb + V_mid/256
       V_mid --[ R = 2.56 MΩ ]--+              |
                                              [Rf_d = 10 kΩ]

  Second combiner (combine that with LSB at 1 : 1/256):

       V_a   --[ R = 10 kΩ ]----+
                                 |---> (-) op-amp E ---> V_out
       V_lsb --[ R = 2.56 MΩ ]--+              |       = V_msb + V_mid/256 + V_lsb/65536
                                              [Rf_e = 10 kΩ]
```

A single-stage three-input combiner with ratio 1 : 1/256 : 1/65536
is *not* viable here — the LSB input resistor would need to be
~655 MΩ (1/65536 of 10 kΩ × 65536 = practical-impossible). Cascading
two 1 : 1/256 stages gets the same effective LSB weighting using two
pairs of sane 10 kΩ : 2.56 MΩ thin-film resistors. Standard textbook
bit-slice DAC topology.

Use **0.01 % thin-film matched-pair resistors** for each 10 kΩ :
2.56 MΩ pair. Ratio precision matters far more than absolute value;
absolute scale is absorbed by software calibration.

The three slice summers (A, B, C) are three instances of the same
circuit. Copy the schematic block and rename nets.

---

## Pi-side pin routing (Pi Zero 2W → board)

**Why this matters more than it looks like it should.** Stage 1 has
**384 MCP4251 packages**. Direct hardware chip select doesn't reach
that — the Pi has 5 hardware CE lines total across SPI0 (2) and SPI1
(3), and only ~26 free GPIOs even if you drive every CS in software.
**Stage 1 must use a chip-select multiplexer**, almost certainly a
cascade of 74HC138 (3-to-8) or 74HC154 (4-to-16) decoders driven
from a few address-bit GPIOs. Stage 0.5's CS topology is a
1/64-scale rehearsal of that exact pattern: one 74HC138 decoder, six
of its eight outputs in use. Same firmware abstraction, same
schematic shape, just smaller.

**Power**

| Pi pin | Net | Goes to |
|---|---|---|
| 1 | 3.3 V (Pi-regulated) | Digital VDD on all MCP4251s and the 74HC138 decoder |
| 2 *or* 4 | 5 V (Pi raw) | Input to onboard AMS1117 LDO; LDO output is the analog 3.3 V rail for the AD8629 op-amps. Keep analog and digital 3.3 V rails separate; tie grounds at one star point near the header. |
| 6, 14, 20, 25, 30, 34, 39 | GND | Ground plane (use **multiple** for low-impedance return — at least 6, 14, and 30) |

**SPI0 bus — shared by all six MCP4251s (multidrop)**

| Pi pin | GPIO | Function | Net on board |
|---|---|---|---|
| 19 | GPIO 10 | SPI0 MOSI | SDI bus, fanned out to SDI on all 6 chips |
| 21 | GPIO 9  | SPI0 MISO | SDO bus, all 6 SDO pins (tristate when CS high; multidrop is fine) |
| 23 | GPIO 11 | SPI0 SCLK | SCK bus, fanned out to all 6 chips |

We do **not** use SPI0's hardware CE0/CE1 for chip selects, even
though they're available — because they don't extrapolate. The
chip-select strategy below scales to Stage 1 unchanged.

**Chip select via 74HC138 3-to-8 decoder (cost: ~$0.30)**

| Pi pin | GPIO | Function | Decoder pin |
|---|---|---|---|
| 24 | GPIO 8  | A0 (chip-address LSB) | 74HC138 pin A |
| 26 | GPIO 7  | A1 | 74HC138 pin B |
| 29 | GPIO 5  | A2 (chip-address MSB) | 74HC138 pin C |
| 31 | GPIO 6  | !ENABLE (active-low: low → decoder runs, high → all CS deasserted) | 74HC138 pin G2A (or G2B) |

Decoder outputs Y0–Y5 (active-low) drive MCP4251 #1–#6 chip-select
pins. Y6 and Y7 are left unconnected. Tie G1 high (decoder
permanent-enable from this side) and use G2A as the runtime
gate.

To talk to chip *i* (0..5):
1. Drive G2A high → all CS deasserted (no chip is listening).
2. Set A2:A1:A0 = binary i.
3. Drive G2A low → decoder asserts Yi → that one chip's CS goes
   low, the other five stay high.
4. Issue the SPI transaction over MOSI/MISO/SCLK (kernel does this
   over `/dev/spidev0.0`; ignore CE0/CE1 — let `SPI_NO_CS` clear
   the kernel's CS handling).
5. Drive G2A high → release CS, prepare for next chip.

The kernel's `spidev` happily runs without driving CE0/CE1 if you
set the `SPI_NO_CS` flag. The four "address" GPIOs are bog-standard
GPIO writes; toggle them between transactions.

**Why this is the right rehearsal for Stage 1.** Stage 1's 384
chips need a tree of decoders — for example, two cascaded 4-to-16
74HC154s would give 256 outputs from 8 address bits, or three of
them give 384+ from 9 bits. The firmware abstraction is the same:
"set address bits, gate enable, run SPI." Stage 0.5's `select_chip(i)`
function is the literal seed of Stage 1's `select_chip(i)`; only
the address-width constant grows.

**I²C1 — to DAC and ADC**

| Pi pin | GPIO | Function | Connects to |
|---|---|---|---|
| 3 | GPIO 2 | I²C1 SDA | MCP4728 SDA + ADS1115 SDA (shared bus) |
| 5 | GPIO 3 | I²C1 SCL | MCP4728 SCL + ADS1115 SCL (shared bus) |

Default I²C addresses don't collide: MCP4728 at 0x60, ADS1115 at
0x48 (with its ADDR pin tied to GND).

**Settings to verify in `raspi-config` before bring-up**

- Enable SPI (`Interface Options → SPI → Enable`).
- Enable I²C (`Interface Options → I²C → Enable`).
- SPI clock for the MCP4251: 5 MHz is comfortable; chip is rated to
  10 MHz. Set via `ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed)`.
- SPI mode: 0 (CPOL=0, CPHA=0). MCP4251 supports modes 0,0 and 1,1.
- Open `/dev/spidev0.0` with `mode |= SPI_NO_CS` so the kernel
  doesn't try to drive CE0 — the decoder handles CS.

**Notes**

- The Pi's 3.3 V GPIO logic levels match MCP4251's threshold at VDD =
  3.3 V — no level shifters needed. The 74HC138 also runs happily at
  3.3 V (HC family operates 2 V to 6 V).
- Pull-ups on I²C: 4.7 kΩ to 3.3 V on each of SDA and SCL, on the
  board (not relying on the Pi's weak internal pull-ups).
- The MISO bus has six tri-state drivers on it. A 10 kΩ pull-up to
  3.3 V on the bus gives it a defined idle state so readback
  (Experiment 3) doesn't return floating-line garbage.
- 74HC138 propagation delay is ~10 ns at 3.3 V. Negligible vs the
  SPI transaction time, but order matters: gate enable *after*
  address bits settle, gate enable *off before* changing address
  bits. Glitches on Yi outputs during address transitions are the
  classic decoder bug.

## Pi-side firmware (sketch)

The firmware reuses the Stage 1 codebase entirely. The MCP4251 SPI
interface is documented in Microchip DS22060B; one register write
per channel sets the wiper tap.

```cpp
// select_chip(i): drive the 74HC138 address lines and assert decoder enable
// so MCP4251 #i (1..6) is the only one with its CS pulled low. Stage 1 will
// have the same function — wider address bus, same shape.
void select_chip(int i) {
    gpio_write(GPIO_DECODER_ENABLE, 1);          // deassert all CS first
    gpio_write(GPIO_ADDR_A0, (i >> 0) & 1);
    gpio_write(GPIO_ADDR_A1, (i >> 1) & 1);
    gpio_write(GPIO_ADDR_A2, (i >> 2) & 1);
    gpio_write(GPIO_DECODER_ENABLE, 0);          // assert: chip i's CS goes low
}
void deselect_all() { gpio_write(GPIO_DECODER_ENABLE, 1); }

// SPI transaction wrapper: select chip, write wiper, deselect.
void spi_write_mcp4251(int chip, int wiper, uint8_t tap) {
    select_chip(chip);
    uint8_t cmd[2] = { (uint8_t)((wiper << 4) | 0x00), tap };  // write-wiper command
    write(spi_fd, cmd, 2);                       // /dev/spidev0.0 opened with SPI_NO_CS
    deselect_all();
}

// Program one Q8.8.8 weight (24 bits: 8 MSB + 8 middle + 8 LSB).
//   MSB + middle slices live on the same MCP4251 die  (chips 1..4)
//   LSB slice lives on a separate die                  (chips 5..6, 2 weights/chip)
void program_weight(int i, uint32_t q888_taps) {
    uint8_t msb = (q888_taps >> 16) & 0xFF;
    uint8_t mid = (q888_taps >>  8) & 0xFF;
    uint8_t lsb = (q888_taps      ) & 0xFF;

    spi_write_mcp4251(/*chip=*/ 1 + i,       /*wiper=*/ 0, msb);
    spi_write_mcp4251(/*chip=*/ 1 + i,       /*wiper=*/ 1, mid);
    spi_write_mcp4251(/*chip=*/ 5 + (i / 2), /*wiper=*/ i % 2, lsb);
}

// Forward pass: drive 4 inputs, read 1 output.
float forward(const float x[4]) {
    for (int i = 0; i < 4; ++i)
        dac_write(i, scale_input_to_voltage(x[i]));
    delay_us(10);                 // op-amp settling
    return scale_reading_to_float(adc_read(/*ch=*/0));
}
```

Stochastic rounding for the optional training-step experiment uses
the same code path Stage 1 will use — three slices, identical to the
simulator's `analog_sim.cpp` reference implementation. Stage 0.5
exercises the full Q8.8.8 firmware path; nothing about the firmware
changes when scaling to a 16-wide tile in Stage 1.

---

## Three Stage 0.5 experiments

### Experiment 1 — "does each summing stage work?"

For each of the three slice summers (MSB, middle, LSB) in turn:
set all four digipot wipers in that slice to mid-scale (tap 128 of
256). Drive one input at a time with a step from 0 V → 1 V, others
held at 0 V. The slice op-amp's output should step proportionally —
same magnitude each time within the 0.1 % bench measurement. Step
all four inputs together; output is roughly 4× a single step.

The spare ADC channels are wired to V_msb, V_mid, and V_lsb so all
three stages can be probed without lifting a scope. **Doing this
once for each slice is doing analog summation in physics, three
times.**

### Experiment 2 — "do the cascaded combiners produce Q8.8.8 output?"

Program a known set of 4 Q8.8.8 weights across MSB + middle + LSB
slices. Drive 4 known inputs. Compare the measured V_out to the
software dot-product of the same Q8.8.8 weights and inputs computed
against the Stage 0 simulator's `Tile::forward`. Match should be
within the σ ≈ 0.0003 per-MAC budget × √4 = 0.06 % of full scale.

If it doesn't match, the diagnostic surface area is small. Probe
V_msb, V_mid, V_lsb individually on the spare ADC channels, then
probe V_a (output of first combiner) on a scope. One of three
things is going wrong:

- A slice summer is off → that slice's R_AB or wiper programming is
  wrong. Cross-check against Experiment 1's measurements.
- A combiner ratio is off → measure the 10 kΩ : 2.56 MΩ matched-pair
  resistors with a multimeter; they should be within 0.01 % of
  ratio. If not, the matched-pair part substitution failed.
- The middle and LSB slices are reversed → re-check the chip-select
  mapping in firmware. Easy to swap.

### Experiment 3 — "does one stochastic-rounded weight update execute?"

Program a starting weight w. Compute a small weight update Δw
digitally on the Pi. Decompose Δw into integer-tap deltas across all
three slices (MSB, middle, LSB) using the simulator's `write_delta`
logic. Stochastically round the fractional residue and commit it
via SPI writes. Read back the wiper register from each MCP4251 and
confirm the taps moved exactly where the firmware predicted.

If they did, **the Stage 0.5 board is done** and the firmware is
Stage-1-ready — the same `write_delta` code path will run unchanged
on the 16-wide tile.

---

## Success criteria

Stage 0.5 passes when all three experiments pass to the precisions
above. The deliverables that move into Stage 1 unchanged:

- KiCad symbols and footprints for MCP4251, AD8629, MCP4728, ADS1115,
  Pi header.
- Pi-side firmware: SPI MCP4251 driver, I²C MCP4728 driver, I²C
  ADS1115 driver, Q8.8.8 weight decomposition + stochastic rounding.
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
including PCB fab and Pi: ~$85. Iteration cost on a layout
respin: ~$25 + 3 days.

This is the rate at which board-level mistakes should be made.
Stage 1 begins after Stage 0.5 lands a clean Experiment 2.
