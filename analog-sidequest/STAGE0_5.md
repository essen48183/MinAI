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

The "JLC tier" column flags whether the part is on JLCPCB's **Basic
Parts** list (pre-loaded reels, no per-part setup fee on PCBA orders)
or **Extended Parts** (~$3 setup per reel). Stage 0.5 deliberately
keeps the Extended-Parts list short so PCBA assembly tops out around
$15 in setup on top of the parts cost.

### Active components (ICs)

| Ref | Part | Qty | Role | Package | JLC tier | Unit | Subtotal |
|---|---|---|---|---|---|---|---|
| U1–U6 | **MCP4251-103E/SL** | 6 | Dual 8-bit volatile SPI digipot, 10 kΩ. U1–U4 hold MSB + middle slice pairs of weights w0–w3 (on the same die for 15 ppm/°C ratiometric tempco). U5–U6 hold the four LSB slices, two per package. **Used in divider mode** — pin A = V_in, pin B = V_CM (buffered midrail), wiper = scaled output. | SOIC-14 | Basic | $1.01 | $6.06 |
| U11–U13 | **OPA2333AIDGKTG4** (dual autozero op-amp) | 3 | **Six op-amp channels** = 3 slice summers (MSB / middle / LSB) + 2 cascaded combiners + 1 buffered midrail (V_CM) reference. Substitution for AD8629 — same autozero topology, 10 µV Vos max, 0.05 µV/°C drift, single-supply 1.8–5.5 V. | VSSOP-8 | Extended | $2.00 | $6.00 |
| U7 | **SN74HC138PWR** | 1 | 3-to-8 decoder for chip select. Drives 6 of 8 outputs to MCP4251 CS pins. The 1/64-scale rehearsal of Stage 1's cascaded-decoder CS network. | TSSOP-16 | Basic | $0.30 | $0.30 |
| U8 | **MCP4728A0T-E/UN** (quad 12-bit DAC) | 1 | Drives the four input voltages over I²C (default address 0x60). Internal 2.048 V Vref selected at startup; gain 1×. | MSOP-10 | Extended | $3.00 | $3.00 |
| U9 | **ADS1115IDGSR** (4-ch 16-bit ADC) | 1 | Reads V_OUT (ch.0); spare channels read V_msb, V_mid, V_lsb on test points TP2–TP4 for slice-level debug. I²C address 0x48. | VSSOP-10 | Basic | $4.00 | $4.00 |
| U10 | **AMS1117-3.3** LDO regulator | 1 | Generates the on-board 3.3 V from the Pi's 5 V; output then splits into separate digital and analog branches via a ferrite bead. | SOT-223 | Basic | $0.30 | $0.30 |
| FB1 | **BLM18AG601SN1D** ferrite bead | 1 | 600 Ω @ 100 MHz, 1 A. Filters the analog 3.3 V branch from the digital 3.3 V branch. Cleaner than a simple LC filter and small. | 0603 | Basic | $0.10 | $0.10 |

### Resistors

| Ref | Part | Qty | Role | Package | JLC tier | Unit | Subtotal |
|---|---|---|---|---|---|---|---|
| R12–R23 | **PAT0603E4002BST1** (Vishay Dale, 40 kΩ, 0.1 %, 25 ppm/°C) | 12 | Slice-summer input resistors — one per (input × slice) cell. With the digipot in divider mode, this resistor sets the per-input current scale into the summing op-amp: `I = V_wiper / 40 kΩ`. Precision matters here because the matmul accuracy is bounded by how well these match across cells. | 0603 thin-film | Extended | $0.30 | $3.60 |
| R24, R25 | **PAT0603E2564BST1** (Vishay Dale, 2.56 MΩ, 0.1 %, 25 ppm/°C) | 2 | The 1/256 path at each of the two cascaded combiner stages. Pair these with the two PAT0603E1002BST1 (10 kΩ 0.1 %) below to set the binary-weighted ratio. | 0603 thin-film | Extended | $0.50 | $1.00 |
| **(2 of R1–R11)** | **PAT0603E1002BST1** (Vishay Dale, 10 kΩ, 0.1 %, 25 ppm/°C) | 2 | **Combiner-input 10 kΩ — partners to R24/R25.** Specifically: the 10 kΩ on V_msb's path into the first combiner op-amp, and the 10 kΩ on VLOW's path into the second combiner op-amp. **These set the binary-weighted ratio precision and must be 0.1 %**, not the generic 1 % thick-film. Identify by tracing the inverting-input nets of the two combiner op-amps in the schematic editor. | 0603 thin-film | Extended | $0.30 | $0.60 |
| **(remaining R1–R11, R26, R27)** | **RC0603FR-07 series** (Yageo, 1 %, thick-film) | 11 | Mixed bias / feedback / pull-up roles where 1 % is sufficient: slice-summer feedback (×3), combiner-stage feedback (×2), midrail-divider 10k:10k (×2), MISO bus pull-up (×1), I²C pull-ups 4.7 kΩ (×2 = R26, R27), spare. | 0603 | Basic | $0.005 | $0.06 |
| R28 | **RC0603JR-130RL** (Yageo, 13 Ω, 5 %) | 1 | Single AGND-to-DGND star-ground tie. Place ONLY this one across the analog/digital ground split — do not add additional ties anywhere else on the board. | 0603 | Basic | $0.01 | $0.01 |

### Capacitors

| Ref | Part | Qty | Role | Package | JLC tier | Unit | Subtotal |
|---|---|---|---|---|---|---|---|
| C1–C14 | **CL10B104KB8NNNC** (Samsung, 0.1 µF X7R 50 V ±10 %) | 14 | Power-pin decoupling — one per IC power pin (MCP4251 ×6, OPA2333 ×3, 74HC138, MCP4728, ADS1115, AMS1117 input, AMS1117 output). Sets the local high-frequency bypass for each chip. | 0603 | Basic | $0.01 | $0.14 |
| C15–C20 | **C2012X5R1A106M085AB** (TDK, 10 µF X5R 10 V ±20 %) | 6 | Per-rail bulk decoupling: 5 V input, AMS1117 output, A3V3 rail (post-ferrite), D3V3 rail, DAC/ADC supply, V_CM reference output. Six caps so each rail has its own local energy reservoir. **MLCC ceramic** — non-polarized, no orientation to track. | 0805 | Basic | $0.05 | $0.30 |

**Avoid Y5V and Z5U dielectrics** even though they appear in the
same MLCC catalogs — capacitance drops 50–80 % under temperature
extremes and DC bias. They're not real capacitors at the precision
this board needs.

### Connectors, test points, and PCB

| Ref | Part | Qty | Role | Package | JLC tier | Unit | Subtotal |
|---|---|---|---|---|---|---|---|
| J1 | **302-S401** (On Shore, 2×20 0.1″) | 1 | Mates to Pi Zero 2W's 40-pin GPIO header. | THT 2.54 mm | Basic | $0.80 | $0.80 |
| TP1–TP8 | **1.8 mm test pads** | 8 | Bring-up probe points: TP1 = V_CM (buffered midrail), TP2 = V_OUT, TP3 = V_LOW (intermediate combiner), TP4 = V_LSB, TP5 = A3V3, TP6 = D3V3, TP7 = AGND, TP8 = DGND. **Don't skip these** — they cut Experiment-1/Experiment-2 debug time by an order of magnitude. | — | Basic | $0.05 | $0.40 |
| — | 4-layer PCB, ~50 × 50 mm | 5 | JLCPCB or PCBWay (fab-house minimum batch is 5). | — | — | $5 | $25.00 |

### Optional but useful

| Ref | Part | Qty | Role | Package | JLC tier | Unit | Subtotal |
|---|---|---|---|---|---|---|---|
| — | 10 kΩ NTC thermistor (e.g., NTCG063JF103) | 1 | Optional. Adds a fifth ADC channel for tempco characterization during Experiment 2. Not required to pass any experiment; useful for understanding why the noise floor is what it is. | 0603 | Basic | $0.50 | $0.50 |

### Totals

- **Active ICs (incl. ferrite bead):** ~$19.76
- **Resistors:** ~$5.27
- **Caps:** ~$0.44
- **Connectors / test points / PCB:** ~$26.20
- **Optional thermistor:** ~$0.50
- **JLCPCB extended-parts setup fees:** ~$15 (OPA2333, MCP4728, PAT0603 thin-film resistor reels)

**Tile-side total: ~$67 with the optional thermistor and PCB
included.** With Pi Zero 2W ($15), microSD ($10), and USB power
supply ($10), **all-in stand-up cost: ~$102.**

That is one-tenth of Stage 1's BOM and roughly the cost of a nice
dinner for two.

---

## JLCPCB sourcing notes

If you order this board with JLCPCB's PCBA service (board fab +
parts placement in one shot), the path is:

1. Upload Gerber files for the PCB.
2. Upload the BOM and the centroid (CPL) file.
3. JLCPCB cross-references each line item against their parts
   library. Lines that match a Basic Part get assembled at the
   default $0 setup fee. Lines that match an Extended Part get
   assembled at ~$3 setup per reel.
4. Lines that don't match anything in their library either get
   substituted, or you ship them as customer-supplied parts (extra
   handling fee), or you drop them and hand-solder after the board
   ships.

For Stage 0.5 the realistic split is:

- **Basic Parts (no setup fee):** MCP4251-103E/SL digipots,
  SN74HC138PWR decoder, ADS1115IDGSR ADC, AMS1117-3.3 regulator,
  BLM18AG601SN1D ferrite bead, the Yageo 1 % thick-film resistors
  (RC0603FR series), Samsung X7R 0.1 µF caps, TDK X5R 10 µF caps,
  the optional thermistor, the Pi header (302-S401).
- **Extended Parts (setup fee, ~$3/reel each):**
  - **OPA2333AIDGKTG4** op-amp (LCSC C2060187)
  - **MCP4728A0T-E/UN** DAC
  - **PAT0603E4002BST1** (40 kΩ 0.1 %), **PAT0603E2564BST1**
    (2.56 MΩ 0.1 %), and **PAT0603E1002BST1** (10 kΩ 0.1 %) — the
    Vishay Dale precision thin-film line. JLCPCB stocks the PAT0603
    family in their Extended catalog; use it if you can.

The 0.1 % precision thin-film resistors are the ones to verify in
JLCPCB's parts library *before* you place the order — the board's
ratio precision depends on them, and substituting in 1 % thick-film
silently degrades Experiment 2's pass margin from comfortable to
borderline.

**Total setup fees for a typical Stage 0.5 order: ~$15** (five
extended-part reels: OPA2333, MCP4728, plus three PAT0603 resistor
values). That brings PCBA-assembled board cost to ~$82 with
shipping, vs ~$67 for hand-populated.

---

## Schematic (one summing column at Q8.8.8 depth)

Three slice summers feed two cascaded binary-weighted combiners.
Single-supply 3.3 V op-amps mean every signal is biased around a
**buffered midrail reference V_CM ≈ 1.65 V**, generated once by a
divider + buffer op-amp and distributed to all op-amp + inputs and
to the digipots' B pins.

```
  Buffered midrail reference (one-shot setup, used everywhere below):

       3.3 V ─[10 kΩ]─┬─[10 kΩ]─ GND
                      │
                      ├──> (+) op-amp F ──> V_CM (≈ 1.65 V)
                                  │
                                 (-) tied to output (unity follower)
                                 [routed to TP1, op-amp + inputs, digipots' B pins]

  Slice summers (digipots in DIVIDER mode):

       V[i]  ──> A pin of MCP4251 (chips 1-4 ch.A : MSB slice for input i)
       V_CM ──> B pin
                wiper (W) = V_CM + (V[i] - V_CM) × tap/256
                  │
                  └──[40 kΩ R_series]──┐
                                       │
                                       ├──> (-) op-amp A ──> V_MSB
                                       │       (+) tied to V_CM
                                       │
                                  [Rf = 10 kΩ feedback to V_OUT pin of op-amp A]

       (the same shape, repeated for the four MSB inputs into op-amp A,
        for the four middle inputs into op-amp B, and for the four LSB
        inputs into op-amp C — three slice summers total)

  First combiner (V_MSB × 1 + V_MID × 1/256 → V_LOW):

       V_MSB ─[10 kΩ, 0.1 % ★ ]─┐
                                 ├──> (-) op-amp D ──> V_LOW
       V_MID ─[2.56 MΩ, 0.1 %]──┘       (+) tied to V_CM
                                  [Rf = 10 kΩ feedback]

  Second combiner (V_LOW × 1 + V_LSB × 1/256 → V_OUT):

       V_LOW ─[10 kΩ, 0.1 % ★ ]─┐
                                 ├──> (-) op-amp E ──> V_OUT
       V_LSB ─[2.56 MΩ, 0.1 %]──┘       (+) tied to V_CM
                                  [Rf = 10 kΩ feedback]

       V_OUT  =  V_MSB + V_MID/256 + V_LSB/65536  (all referenced to V_CM)
```

The two **★** resistors are the ones flagged in the BOM as
**`PAT0603E1002BST1` (10 kΩ 0.1 % thin-film)**. They sit in the
binary-weighted-ratio path with the 2.56 MΩ R24/R25 partners; if
they're left at 1 % thick-film, the 1 : 1/256 ratio is only
accurate to ~1 % — about 18× larger than the σ ≈ 6 × 10⁻⁴ output
RMS noise budget — and the LSB / middle slice contributions are
systematically distorted in a way the Experiment-2 single-scale
calibration can't absorb.

**Why divider mode and not rheostat mode.** Putting the digipot
between V_in and the summing junction (rheostat mode) means the
weight is `R_f / (R_series + R_wiper)` — a *non-linear* function of
the wiper position, and one that goes to infinity if the wiper
shorts. Driver mode (V_in on pin A, V_CM on pin B, wiper through a
fixed series resistor) makes the wiper voltage **linear in the tap
value**: `V_wiper = V_CM + (V_in − V_CM) × tap/256`. Linear is
what the simulator's Q8.8.8 math assumes; rheostat mode would
require recalibration of the encoding.

**Why a single-stage three-input combiner doesn't work.** The
one-shot V_OUT = V_MSB × 1 + V_MID × 1/256 + V_LSB × 1/65536
combiner would need a 655 MΩ resistor on the LSB input (10 kΩ ×
65536). 655 MΩ in a 0603 thin-film doesn't exist; even if it did,
PCB leakage and op-amp input bias current would dominate.
Cascading two 1 : 1/256 stages gets the same effective weighting
using two pairs of sane 10 kΩ : 2.56 MΩ thin-film resistors.
Standard textbook bit-slice DAC topology.

The three slice summers (op-amps A, B, C) are three instances of
the same inverting-summer-with-V_CM-bias circuit. Copy the
schematic block and rename nets.

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
including PCB fab and Pi: ~$102. Iteration cost on a layout
respin: ~$25 + 3 days.

This is the rate at which board-level mistakes should be made.
Stage 1 begins after Stage 0.5 lands a clean Experiment 2.
