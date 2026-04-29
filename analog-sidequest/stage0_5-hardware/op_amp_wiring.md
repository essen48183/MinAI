# Stage 0.5 wiring reference — every chip, every net, every cap

*A pin-by-pin reference for the entire Stage 0.5 board, organized so
you can lay it side-by-side with your KiCad schematic and find
exactly where things diverge. Uses the canonical R / C / U / J / FB /
TP reference designators throughout.*

*This document does NOT edit your KiCad files. It exists purely to
make the wiring rules explicit, and to call out the places where
**physical placement** matters as much as connectivity.*

---

## How to read the placement notes

Every section below has a **🔧 Placement** callout box. The rule is:

- **"≤ N mm" means trace length** from one pin to another.
- **"Adjacent" means under or directly next to** the named pin —
  ideally on the back side of the board for surface-mount caps so
  the trace is just two vias deep.
- **"Anywhere on the bus" means routing distance doesn't matter** —
  the part is electrically equivalent at any point on its net. Pull-
  ups and bus terminators usually qualify.
- **"Single point" means literally one location** — every other
  spot on the relevant net must NOT have this connection. Star-
  ground ties are the canonical example.

---

## The thing that's confusing you, in one paragraph

There is **exactly one place** on this board where two resistors share
a net on one end and go to different rails on the other. That place is
the **V_CM (midrail) reference divider**: one 10 kΩ from **3.3 V** to a
shared node, and another 10 kΩ from that same shared node down to
**GND**. The shared node is the (+) input of the V_CM buffer op-amp
(U11A). That's it. Every other "resistor with identical-looking nets"
you see on the board is part of the inverting-summer pattern, where
the resistors join at the **(-) pin** of an op-amp (the summing
junction) and their other ends go to **signal nets** — not to GND or
3.3 V.

**Five resistors total touch GND or a power rail:**

- **R_TOP** (V_CM divider top) → 3.3 V on one end
- **R_BOT** (V_CM divider bottom) → GND on one end
- **R26**, **R27** (I²C pull-ups) → D3.3 V on one end
- **R_MISO** (MISO bus pull-up) → D3.3 V on one end
- **R28** (AGND ↔ DGND star tie) → both ends on grounds

Every other resistor on the entire board has both ends on signal
nets. If your schematic has anything else tied to a rail, that's the
bug.

---

## Board-level data flow

```
   +5 V ───► U10 ───► +3.3 V (LDO out)
   (J1)     LDO       │
                      ├──► (D3V3 net) ──► U7, J1 pull-ups, MISO pull-up
                      │
                      └─► FB1 ───► (A3V3 net) ──► U1..U6, U8, U9, U11..U13, V_CM divider top
                          ferrite

   AGND ◄── R28 (single 13Ω star tie) ──► DGND


  Pi GPIO ─┬─► SPI MOSI ─► all 6 MCP4251 SDI pins
   (J1)    ├─► SPI MISO ◄─ all 6 MCP4251 SDO pins   (R_MISO pull-up)
           ├─► SPI SCLK ─► all 6 MCP4251 SCK pins
           │
           ├─► A0 / A1 / A2 ─► U7 (74HC138 decoder)
           ├─► ENABLE       ─► U7 G2A
           │                       │
           │                       └─► Y0..Y5 ─► U1..U6 CS pins
           │
           ├─► I²C SDA ───► U8 (DAC), U9 (ADC)         (R26 pull-up)
           └─► I²C SCL ───► U8 (DAC), U9 (ADC)         (R27 pull-up)


  U8 (DAC) ─► V_in[0..3] ──► U1..U4 pin A, plus split to U5/U6 LSB pin A's
                                         │
                                  (each MCP4251 in DIVIDER mode:
                                   A = V_in, B = V_CM, W = scaled output)
                                         │
                                         ▼
                              wiper voltages, scaled by tap/256
                                         │
                              ┌──────────┼──────────┐
                              ▼          ▼          ▼
                            (MSB)      (MID)      (LSB)
                              │          │          │
                              ▼          ▼          ▼
                          [40k Rs]   [40k Rs]   [40k Rs]
                              │          │          │
                              ▼          ▼          ▼
                            U11B       U12A       U12B    (slice summers)
                              │          │          │
                              ▼          ▼          ▼
                            V_MSB     V_MID      V_LSB
                              │          │          │
                              └──────────┘          │
                                   │                │
                                   ▼                │
                                 U13A               │
                            (MSB + MID/256)         │
                                   │                │
                                   ▼                │
                                 V_LOW              │
                                   │                │
                                   └────────────────┤
                                                    │
                                                    ▼
                                                  U13B
                                            (V_LOW + V_LSB/256)
                                                    │
                                                    ▼
                                                  V_OUT
                                                    │
                                                    ▼
                                                 U9 AIN0
                                                 (ADC ch.0 reads V_OUT)
                                                 also AIN1=V_MSB,
                                                      AIN2=V_MID,
                                                      AIN3=V_LSB
                                                 for slice-level debug
```

---

# Section A — Power, ground, connectors

## A1 — J1 (Pi header, 2×40 0.1″ THT)

This is the only connection between the Stage 0.5 board and the Pi
Zero 2W. Everything else on the board is downstream of pins on this
header.

### Pin map (only the pins we actually use)

| Pi pin | GPIO / function | Goes to |
|---|---|---|
| 1 | Pi 3.3 V | **NOT used** — we generate our own 3.3 V from 5 V |
| 2 | Pi 5 V | U10 V_in (AMS1117 input) |
| 4 | Pi 5 V | (alternate; tie to the same 5 V net as pin 2) |
| 6, 14, 20, 25, 30, 34, 39 | GND | DGND plane (multiple connections for low-impedance return) |
| 3 | GPIO 2 = I²C1 SDA | I²C SDA bus (U8 SDA, U9 SDA) |
| 5 | GPIO 3 = I²C1 SCL | I²C SCL bus (U8 SCL, U9 SCL) |
| 19 | GPIO 10 = SPI0 MOSI | SPI MOSI bus (U1..U6 SDI) |
| 21 | GPIO 9 = SPI0 MISO | SPI MISO bus (U1..U6 SDO) |
| 23 | GPIO 11 = SPI0 SCLK | SPI SCK bus (U1..U6 SCK) |
| 24 | GPIO 8 | U7 pin A (74HC138 address bit 0) |
| 26 | GPIO 7 | U7 pin B (74HC138 address bit 1) |
| 29 | GPIO 5 | U7 pin C (74HC138 address bit 2) |
| 31 | GPIO 6 | U7 pin G2A (74HC138 active-low enable) |

### 🔧 Placement

- **J1 sits at the edge of the board** so the Pi can plug into it
  cleanly. The Pi's GPIO header is on its long edge; orient the
  board so it doesn't mechanically interfere with the Pi's USB
  ports or microSD slot.
- **C15 (5 V bulk cap) within ~5 mm of pin 2.** The 5 V trace from
  the header should hit the bulk cap before traveling to the LDO,
  so any noise on the Pi's 5 V rail is shunted to ground locally.
- **Pin 1 marker on silkscreen** must align with the Pi's pin 1 —
  silkscreen-print a `1` or a triangle next to that pin.

---

## A2 — U10 (AMS1117-3.3 LDO regulator)

Drops the Pi's 5 V to a clean 3.3 V for the rest of the board.

### Pinout (SOT-223)

```
       ┌─────┐
       │     │
 GND ──┤1   3├── V_in (5 V from J1)
       │     │
       │  2  │
       └──┬──┘
          │
          V_out (3.3 V, before FB1 split)
```

(SOT-223 has a tab on the back that's electrically GND on the
AMS1117 — connect it to a copper pour for thermal dissipation.)

### Connections

| Pin | Connects to |
|---|---|
| 1 (GND) | DGND plane (large pour, with the SOT-223 tab also tied here) |
| 2 (V_out) | 3.3 V net (pre-FB1); also drives **C16 bulk cap** |
| 3 (V_in) | 5 V from J1; also receives **C15 bulk cap** |

### 🔧 Placement — this is one of the few critical-placement parts on the board

- **C15 bulk cap (10 µF X5R 0805) ≤ 5 mm from pin 3.** Input bulk
  cap stabilizes the LDO against fast load transients on the 5 V
  side. Datasheet says minimum 10 µF input.
- **C16 bulk cap (10 µF X5R 0805) ≤ 5 mm from pin 2.** Output bulk
  cap is what stabilizes the LDO's internal feedback loop. The
  AMS1117 is **not unconditionally stable** — without an output
  cap of the right value within reach, it can oscillate. 10 µF is
  the safe default.
- **Both caps to GND with short, fat traces** — at least 0.5 mm
  trace, ideally a flood pour. Long thin traces add inductance and
  blow up the loop response.
- **Tab tied to GND pour** (under the SOT-223). The LDO dissipates
  ~33 mW at typical load (5 V → 3.3 V at ~20 mA total current);
  the tab is the heat path. Don't isolate it.
- **Place U10 close to J1's 5 V pin** so the input trace from the
  Pi is short.

---

## A3 — FB1 (BLM18AG601SN1D ferrite bead)

Splits the LDO output into a digital 3.3 V branch (D3V3) and an
analog 3.3 V branch (A3V3). The bead is 600 Ω at 100 MHz, ~0.05 Ω at
DC — passes power but blocks high-frequency noise from crossing
between the digital and analog domains.

### Connections

| Side | Goes to |
|---|---|
| Pin 1 | LDO output net (also where C16 lives) — call this side "D3V3" |
| Pin 2 | A3V3 net (post-bead, clean analog rail) — also drives **C17** |

### 🔧 Placement — the geographic boundary between digital and analog

- **Place FB1 along a clear line that visually separates the
  digital chips from the analog chips on the board layout.** This
  is where you draw the mental map: everything one one side of the
  bead is digital territory, everything on the other side is
  analog. It makes the layout self-documenting.
- **C17 bulk cap (10 µF X5R 0805) ≤ 5 mm from FB1's analog-side
  pin.** This is the local bulk reservoir for the analog rail; it
  has to live on the *analog* side of the bead, not the digital
  side.
- **The digital side of the bead** (where D3V3 lives) is the rail
  that powers U7 (decoder). It's also the rail that the I²C
  pull-ups (R26, R27) and MISO pull-up (R_MISO) tie to.

---

## A4 — R28 (AGND ↔ DGND star ground tie, 13 Ω 5 %)

The single point where the analog ground plane and digital ground
plane meet. **There must be exactly one of these on the board** —
adding a second tie creates a ground loop and the analog noise floor
goes to garbage.

### Connections

| End | Goes to |
|---|---|
| Pin 1 | AGND plane |
| Pin 2 | DGND plane |

### 🔧 Placement — single point, no exceptions

- **Place R28 physically near the LDO (U10) or near the Pi header
  (J1)** — the entry/exit point of the digital current path. The
  return current from the digital rail naturally wants to head
  back toward U10's GND, and you want it crossing into the analog
  domain at exactly one well-defined location.
- **Do NOT also tie AGND and DGND together at any other location**
  — not under U7, not under U8/U9, not via fill stitching, nothing.
  The two ground planes should appear as separate islands on the
  layout, joined only at R28's pads.
- The 13 Ω value is a small impedance that allows the DC return
  path while damping any high-frequency current trying to circulate
  between the planes. (A direct copper bridge would also work but
  loses the damping; many designs use a 0 Ω jumper if the noise
  budget allows.)

---

# Section B — Digital control plane

## B1 — U7 (SN74HC138 3-to-8 decoder)

Takes the four GPIO bits from the Pi (3 address + 1 enable) and
drives one of six digipot chip-select lines. The 1/64-scale
rehearsal of Stage 1's cascaded-decoder CS strategy.

### Pinout (TSSOP-16)

```
       ┌──────┐
   A ─┤1   16├── V_CC (D3V3)
   B ─┤2   15├── !Y0 ──► U1 CS
   C ─┤3   14├── !Y1 ──► U2 CS
 !G2A ─┤4   13├── !Y2 ──► U3 CS
 !G2B ─┤5   12├── !Y3 ──► U4 CS
   G1 ─┤6   11├── !Y4 ──► U5 CS
  !Y7 ─┤7   10├── !Y5 ──► U6 CS
   GND ─┤8    9├── !Y6 (unused, leave floating)
       └──────┘
```

### Connections

| Pin | Connects to |
|---|---|
| 1 (A) | J1 pin 24 (Pi GPIO 8) — chip-address bit 0 |
| 2 (B) | J1 pin 26 (Pi GPIO 7) — chip-address bit 1 |
| 3 (C) | J1 pin 29 (Pi GPIO 5) — chip-address bit 2 |
| 4 (!G2A) | J1 pin 31 (Pi GPIO 6) — runtime enable, active-low |
| 5 (!G2B) | DGND (tie permanently low — second active-low enable) |
| 6 (G1) | D3V3 (tie permanently high — active-high enable) |
| 7 (!Y7) | unconnected (or a debug pad) |
| 8 (GND) | DGND |
| 9 (!Y6) | unconnected (or a debug pad) |
| 10 (!Y5) | U6 pin 1 (CS) |
| 11 (!Y4) | U5 pin 1 (CS) |
| 12 (!Y3) | U4 pin 1 (CS) |
| 13 (!Y2) | U3 pin 1 (CS) |
| 14 (!Y1) | U2 pin 1 (CS) |
| 15 (!Y0) | U1 pin 1 (CS) |
| 16 (V_CC) | D3V3 |

### 🔧 Placement

- **Decoupling cap (0.1 µF X7R 0805, one of C1–C14) adjacent to
  pin 16.** Shortest possible trace from pin 16 to the cap, then
  short trace from cap to pin 8 (GND). Aim for ≤ 3 mm trace
  length on each leg.
- **Place U7 between J1 and the digipot cluster.** The address
  bits come in from J1; the chip-select outputs fan out to U1–U6.
  Putting U7 in the middle keeps both halves of those traces
  reasonable length.
- **The six CS traces (Y0..Y5) can be different lengths.** They
  toggle on the order of microseconds; trace mismatch isn't a
  problem at this speed. Just keep them away from the analog
  signal traces (V_in, V_MSB, V_MID, V_LSB, V_OUT, V_CM) — digital
  edges crosstalk into analog summing junctions if they share a
  via or run parallel for more than ~5 mm.

---

## B2 — U1 through U6 (MCP4251 dual digipots)

Six dual digipot packages = 12 wiper channels = 12 weight slices
(4 weights × 3 slices each). Used in **divider mode**: each
digipot's pin A is tied to the input voltage V_in[i], pin B to V_CM,
and the wiper W produces a scaled voltage `V_CM + (V_in[i] − V_CM) ×
tap/256`.

### Pinout (SOIC-14, dual potentiometer variant)

```
       ┌──────┐
  CS ─┤1   14├── V_DD (A3V3)
  SCK ─┤2   13├── SDO
  SDI ─┤3   12├── SHDN  (tie to V_DD via 10kΩ — see note)
  V_SS ─┤4   11├── WP   (tie to V_DD)
  P1B ─┤5   10├── P0B
  P1W ─┤6    9├── P0W
  P1A ─┤7    8├── P0A
       └──────┘
```

`P0A/P0W/P0B` are channel A's terminals (digipot 0).
`P1A/P1W/P1B` are channel B's terminals (digipot 1).

### Channel/weight assignment (from STAGE0_5.md)

| Chip | Channel A holds | Channel B holds |
|---|---|---|
| **U1** | MSB slice of weight w0 | MID slice of weight w0 |
| **U2** | MSB slice of weight w1 | MID slice of weight w1 |
| **U3** | MSB slice of weight w2 | MID slice of weight w2 |
| **U4** | MSB slice of weight w3 | MID slice of weight w3 |
| **U5** | LSB slice of weight w0 | LSB slice of weight w1 |
| **U6** | LSB slice of weight w2 | LSB slice of weight w3 |

The MSB+MID-on-the-same-die pairing matters: those two slices have
a 256:1 weight ratio, and the on-die ratiometric tempco (15 ppm/°C)
keeps them tracking even as the room temperature drifts.

### Per-chip connections

For each chip Ux (x = 1..6):

| Pin | Net | Notes |
|---|---|---|
| 1 (CS) | One of U7's Y0..Y5 outputs | active-low, asserted only when this chip is being addressed |
| 2 (SCK) | SPI SCLK bus | shared with all 6 chips |
| 3 (SDI) | SPI MOSI bus | shared with all 6 chips |
| 4 (V_SS) | AGND | digital logic ground; ties to AGND not DGND because the chip lives on the analog rail |
| 13 (SDO) | SPI MISO bus | shared with all 6 chips (tristate when CS deasserted) |
| 12 (SHDN) | A3V3 (or pull up via 10 kΩ) | active-low shutdown — keep high to enable |
| 11 (WP) | A3V3 | write-protect — keep high to enable writes |
| 14 (V_DD) | A3V3 | analog rail; **decoupling cap close** |

For each chip, **both digipots** are wired identically as voltage
dividers:

| Channel | A terminal | B terminal | Wiper output |
|---|---|---|---|
| U1..U4 ch.A (MSB) | V_in[i] (from U8 DAC channel i) | V_CM | through 40 kΩ R12..R15 to U11B pin 6 |
| U1..U4 ch.B (MID) | V_in[i] | V_CM | through 40 kΩ R16..R19 to U12A pin 2 |
| U5 ch.A (LSB w0) | V_in[0] | V_CM | through 40 kΩ R20 to U12B pin 6 |
| U5 ch.B (LSB w1) | V_in[1] | V_CM | through 40 kΩ R21 to U12B pin 6 |
| U6 ch.A (LSB w2) | V_in[2] | V_CM | through 40 kΩ R22 to U12B pin 6 |
| U6 ch.B (LSB w3) | V_in[3] | V_CM | through 40 kΩ R23 to U12B pin 6 |

### 🔧 Placement

- **One 0.1 µF X7R decoupling cap (C1–C14) per chip, adjacent to
  pin 14 (V_DD).** ≤ 3 mm trace to V_DD, ≤ 3 mm trace from cap to
  AGND. Six chips, six caps.
- **Wiper-to-40kΩ trace ≤ 10 mm.** The wiper output is high-
  impedance; long traces pick up capacitive noise. Place the 40 kΩ
  series resistor (R12..R23) close to the digipot's wiper pin.
- **40kΩ-to-op-amp -IN trace ≤ 10 mm.** The other side of the 40 kΩ
  is the summing junction at the op-amp's -IN — also high-impedance
  until the op-amp closes the loop. Keep this short.
- **SPI bus (SDI/SDO/SCK)** can run further; these are digital
  signals at 5 MHz. Keep them on the digital side of the FB1
  ferrite where possible, or shielded by ground pours when they
  cross into analog territory.
- **The six chips can be physically arranged in a row** along the
  analog edge of the board, with U1..U4 (the MSB+MID dies) closer
  to U11B/U12A (their summers), and U5/U6 (the LSB dies) closer to
  U12B (the LSB summer). Layout follows data flow.

---

# Section C — Analog input

## C1 — U8 (MCP4728 quad 12-bit DAC)

Generates the four input voltages V_in[0..3]. I²C address 0x60.
Internal 2.048 V Vref selected at startup; gain 1×, so the four
outputs span 0 to 2.048 V.

### Pinout (MSOP-10)

```
       ┌──────┐
 V_OUTA ─┤1  10├── V_DD (A3V3)
   GND ─┤2   9├── SCL
 V_OUTB ─┤3   8├── SDA
 V_OUTC ─┤4   7├── !LDAC
 V_OUTD ─┤5   6├── !RDY
       └──────┘
```

### Connections

| Pin | Net | Notes |
|---|---|---|
| 1 (V_OUTA) | V_in[0] | drives U1 pin 8 (P0A), U5 pin 8 (P0A) |
| 2 (GND) | AGND | analog ground reference |
| 3 (V_OUTB) | V_in[1] | drives U2 pin 8, U5 pin 7 (P1A) |
| 4 (V_OUTC) | V_in[2] | drives U3 pin 8, U6 pin 8 |
| 5 (V_OUTD) | V_in[3] | drives U4 pin 8, U6 pin 7 |
| 6 (!RDY) | unconnected (high-Z output, not used) |
| 7 (!LDAC) | tied to AGND (DAC updates immediately on I²C STOP) |
| 8 (SDA) | I²C SDA bus |
| 9 (SCL) | I²C SCL bus |
| 10 (V_DD) | A3V3 | **decoupling cap close** |

### 🔧 Placement

- **Decoupling cap (one of C1–C14) adjacent to pin 10.** ≤ 3 mm
  to pin 10, ≤ 3 mm to AGND.
- **Place U8 close to the row of digipots** so the four V_OUT
  traces are short. These are *analog signal* traces — they should
  route on the analog side of the board, away from any SPI/I²C/CS
  trace they don't strictly need to cross.
- **C19 (10 µF bulk cap) on the V_DD pin** near U8 if your layout
  has the headroom — gives the DAC a local energy reservoir for
  fast settling. Optional but recommended.
- **The four V_in traces fanning out to digipots can be different
  lengths.** Settling time at 50 µs is generous; trace length
  mismatch is not a precision issue at this scale.

---

# Section D — The matmul (op-amps)

## D0 — OPA2333 dual pinout (reminder)

```
              ┌──────────┐
       OUT_A ─┤1        8├─ V+ (A3V3)
       -IN_A ─┤2        7├─ OUT_B
       +IN_A ─┤3        6├─ -IN_B
          V- ─┤4        5├─ +IN_B
              └──────────┘
                  VSSOP-8
```

Each package gives you TWO independent op-amps: A (pins 1-3) and B
(pins 5-7). V- = AGND. V+ = A3V3 (post-FB1).

### 🔧 Placement (applies to U11, U12, U13)

- **0.1 µF X7R decoupling cap adjacent to pin 8 (V+).** ≤ 3 mm to
  pin 8, ≤ 3 mm from cap to pin 4 (V-). Three chips, three caps.
- **Place each op-amp package in the data-flow path, not off to
  the side.** U11 near the V_CM divider and the MSB summer's
  inputs; U12 between the MID and LSB summer paths; U13 between
  the slice summer outputs (V_MSB / V_MID / V_LSB) and the ADC.

### Channel assignment (from STAGE0_5.md)

| Package | Channel | Role |
|---|---|---|
| U11 | A (pins 1-3) | **V_CM buffer** (midrail reference, unity follower) |
| U11 | B (pins 5-7) | MSB slice summer |
| U12 | A | MID slice summer |
| U12 | B | LSB slice summer |
| U13 | A | First combiner (V_LOW = V_MSB + V_MID/256) |
| U13 | B | Second combiner (V_OUT = V_LOW + V_LSB/256) |

If your schematic has the channels in a different order, the
*topology* below is still right — just relabel the Ux/Y references.

---

## D1 — V_CM buffer (U11A) — the **only** stage with resistors to GND and 3.3 V

Two 10 kΩ resistors form a voltage divider that drops 3.3 V to
~1.65 V; the op-amp follows that midpoint and outputs a low-impedance
V_CM that every other op-amp's (+) input uses as its bias reference.

```
                                                   3.3 V  (analog rail, A3V3)
                                                     │
                                                     │
                                                    R_TOP    ◄── 10 kΩ 1%
                                                     │            (one of R1, R2, R3, R4, R5, R6, R8, R9, R11)
                                                     │
                                                     ├──────────────► to U11A pin 3 (+IN_A)
                                                     │
                                                    R_BOT    ◄── 10 kΩ 1%
                                                     │            (one of R1, R2, R3, R4, R5, R6, R8, R9, R11)
                                                     │
                                                    GND  (AGND)


                                ┌──────────┐
              U11A out ◄────────┤1        8├──── V+ (A3V3)
                          ┌─────┤2        7├──── (U11B)
                          │     │          │
                R_TOP/R_BOT────► 3        6├──── (U11B)
                  midpoint     │          │
                          ┌────┤4        5├──── (U11B)
                          │   AGND
                          │
                       (decoupling C: 0.1 µF
                        between pin 8 and pin 4,
                        physically adjacent)


   Wiring summary for U11A:
     pin 3 (+IN_A)  ← divider midpoint (between R_TOP and R_BOT)
     pin 2 (-IN_A)  ← shorted directly to pin 1 (unity follower; no resistor!)
     pin 1 (OUT_A)  → V_CM net  (which then routes to:
                                  - U11B pin 5 (+IN_B)
                                  - U12A pin 3 (+IN_A)
                                  - U12B pin 5 (+IN_B)
                                  - U13A pin 3 (+IN_A)
                                  - U13B pin 5 (+IN_B)
                                  - all six MCP4251 B pins (digipot dividers)
                                  - test point TP1)
```

### 🔧 Placement

- **R_TOP and R_BOT adjacent to U11A pin 3.** The divider midpoint
  is high-impedance (5 kΩ source) before the buffer closes the
  loop; long trace from divider to pin 3 picks up noise.
- **Pin 2 ↔ Pin 1 short loop.** This is the unity-gain feedback;
  shouldn't be more than ~5 mm of trace between the two pins.
- **C20 (10 µF bulk on V_CM net) close to U11A pin 1 (OUT_A).**
  V_CM is loaded by every op-amp on the board through their +IN
  pins; a local reservoir cap keeps it stable.

**This is the ONLY op-amp in the design where:**
- A resistor (R_TOP) connects from **3.3 V** to one pin
- A resistor (R_BOT) connects from **GND** to that same pin
- That pin (U11A's +IN_A) is the only node fed by both rails

If your schematic has resistors going to GND and/or 3.3 V at any
**other** op-amp's +IN, that's a wiring mistake.

**Important:** U11A pin 2 (-IN_A) is shorted DIRECTLY to pin 1
(OUT_A). There is no resistor in this loop. A unity-gain follower
has no feedback resistor.

---

## D2 — MSB slice summer (U11B)

Inverting summer with **four input resistors** (one per input
voltage), **one feedback resistor**, and the (+) input tied to V_CM
(no resistor).

```
   V_in[0] ──[R12 = 40 kΩ 0.1%]──┐
   V_in[1] ──[R13 = 40 kΩ 0.1%]──┤
   V_in[2] ──[R14 = 40 kΩ 0.1%]──┤
   V_in[3] ──[R15 = 40 kΩ 0.1%]──┤
                                 │
                                 ▼
                           ┌──────────┐
                          ─┤6 -IN_B   │
                           │          ├ pin 7 (OUT_B) ──► V_MSB net
                           │          │                    │
                           │          │                    │
                           │          │                    └──[R_FB_MSB = 10 kΩ 1%]──┐
                           │          │                                                │
                          V_CM ──────┤5 +IN_B  │                                       │
                                     │         │                                       │
                                     └─────────┘                                       │
                                                                                       │
   pin 6 (-IN_B) ◄──────────────────────────────────────────────────────────────────────┘


   Wiring summary for U11B:
     pin 5 (+IN_B)     ← V_CM net (NOT to ground, NOT to 3.3 V)
     pin 6 (-IN_B)     ← summing junction:
                            R12 from V_in[0]   (slice MSB of weight w0 from MCP4251 #1 ch.A wiper)
                            R13 from V_in[1]   (slice MSB of weight w1 from MCP4251 #2 ch.A wiper)
                            R14 from V_in[2]   (slice MSB of weight w2 from MCP4251 #3 ch.A wiper)
                            R15 from V_in[3]   (slice MSB of weight w3 from MCP4251 #4 ch.A wiper)
                            R_FB_MSB from pin 7 (feedback)
     pin 7 (OUT_B)     → V_MSB net  (and to one end of R_FB_MSB, and to TP for debug)


   None of R12, R13, R14, R15, R_FB_MSB go to GND. None go to 3.3 V.
```

### 🔧 Placement

- **R12, R13, R14, R15 adjacent to U11B pin 6.** The summing
  junction is high-impedance before the loop closes; resistors
  hanging off long traces capacitively couple noise.
- **R_FB_MSB short — pin 7 to pin 6 with a small loop.** Feedback
  trace ≤ 10 mm.
- **V_MSB output trace** can run further (it's a low-impedance
  op-amp output) — to U13A's input pin and to the debug test
  point.

`R_FB_MSB` is one of the unallocated 10 kΩ 1 % resistors from the
{R1, R2, R3, R4, R5, R6, R8, R9, R11} pool.

---

## D3 — MID slice summer (U12A)

Identical topology to MSB summer, with the *middle* slice's wiper
voltages as inputs. Different chip channels (each MCP4251's `ch.B`
wiper for chips 1–4).

```
   V_mid[0] (MCP4251 #1 ch.B wiper) ──[R16 = 40 kΩ 0.1%]──┐
   V_mid[1] (MCP4251 #2 ch.B wiper) ──[R17 = 40 kΩ 0.1%]──┤
   V_mid[2] (MCP4251 #3 ch.B wiper) ──[R18 = 40 kΩ 0.1%]──┤
   V_mid[3] (MCP4251 #4 ch.B wiper) ──[R19 = 40 kΩ 0.1%]──┤
                                                          ▼
                                                  ┌──────────┐
                                                 ─┤2 -IN_A   │
                                                  │          ├ pin 1 (OUT_A) ──► V_MID net
                                                  │          │                    │
                                            V_CM ─┤3 +IN_A   │                    │
                                                  └──────────┘                    │
                                                                                  │
                                                                ┌─[R_FB_MID = 10 kΩ 1%]─┘
                                                                │
                                                                └──► back to pin 2

   Wiring summary for U12A:
     pin 3 (+IN_A) ← V_CM
     pin 2 (-IN_A) ← R16, R17, R18, R19 from MID wipers + R_FB_MID from pin 1
     pin 1 (OUT_A) → V_MID net (also to TP for debug)
```

### 🔧 Placement

- **Same rules as U11B**: R16..R19 adjacent to pin 2, R_FB_MID
  short between pin 1 and pin 2.

---

## D4 — LSB slice summer (U12B)

Same again, with the LSB-slice wipers as inputs. The LSB digipots
live on chips U5 and U6 (two LSB slices per chip).

```
   V_lsb[0] (U5 ch.A wiper) ──[R20 = 40 kΩ 0.1%]──┐
   V_lsb[1] (U5 ch.B wiper) ──[R21 = 40 kΩ 0.1%]──┤
   V_lsb[2] (U6 ch.A wiper) ──[R22 = 40 kΩ 0.1%]──┤
   V_lsb[3] (U6 ch.B wiper) ──[R23 = 40 kΩ 0.1%]──┤
                                                  ▼
                                          ┌──────────┐
                                         ─┤6 -IN_B   │
                                          │          ├ pin 7 (OUT_B) ──► V_LSB net
                                    V_CM ─┤5 +IN_B   │                    │
                                          └──────────┘                    │
                                                          ┌─[R_FB_LSB = 10 kΩ 1%]─┘
                                                          │
                                                          └──► back to pin 6

   Wiring summary for U12B:
     pin 5 (+IN_B) ← V_CM
     pin 6 (-IN_B) ← R20, R21, R22, R23 from LSB wipers + R_FB_LSB from pin 7
     pin 7 (OUT_B) → V_LSB net (and to TP4)
```

### 🔧 Placement

- Same rules as U11B / U12A. R20..R23 adjacent to pin 6, R_FB_LSB
  short.

---

## D5 — First combiner (U13A): V_LOW = V_MSB + V_MID/256

Two-input inverting summer. The 1:1/256 ratio comes from the **input**
resistors, not the feedback. **R7 must be 10 kΩ 0.1 %**, **R24 must
be 2.5 MΩ + 60 kΩ in series at 0.1 %**. The feedback (R_FB_C1) is
just 10 kΩ 1 %.

```
   V_MSB ──[R7 = 10 kΩ 0.1% ★]─────────────────┐
                                                │
   V_MID ──[R24A = 2.5 MΩ 0.1%]──[R24B = 60 kΩ 0.1%]──┤    (series pair = 2.56 MΩ effective)
                                                │
                                                ▼
                                        ┌──────────┐
                                       ─┤2 -IN_A   │
                                        │          ├ pin 1 (OUT_A) ──► V_LOW net
                                  V_CM ─┤3 +IN_A   │                    │
                                        └──────────┘                    │
                                                          ┌─[R_FB_C1 = 10 kΩ 1%]─┘
                                                          │
                                                          └──► back to pin 2

   Wiring summary for U13A:
     pin 3 (+IN_A) ← V_CM
     pin 2 (-IN_A) ← R7 from V_MSB + R24A-R24B series from V_MID + R_FB_C1 from pin 1
     pin 1 (OUT_A) → V_LOW net (also to TP3, and into U13B's input as the next stage)


   None of R7, R24A, R24B, R_FB_C1 go to GND or 3.3 V.
   They all join at pin 2 of U13A.
```

### 🔧 Placement

- **R7, R24A, R24B, R_FB_C1 all adjacent to U13A pin 2.** This is
  the binary-weighted summing junction; *both* inputs are sensitive
  and the ratio precision depends on minimal stray capacitance.
- **Place R24A and R24B in series with NO branch trace between
  them.** Two pads, one short trace between them, then into pin 2.
  A bonus pad in the middle would create a stub that can pick up
  capacitive noise.
- **V_LOW output (pin 1) trace to U13B is short** — these two
  combiner stages should sit close together on the layout.

---

## D6 — Second combiner (U13B): V_OUT = V_LOW + V_LSB/256

Identical shape to U13A, with V_LOW and V_LSB as the two inputs.
**R10 must be 10 kΩ 0.1 %**, **R25 must be 2.5 MΩ + 60 kΩ in series
at 0.1 %**.

```
   V_LOW ──[R10 = 10 kΩ 0.1% ★]────────────────┐
                                                │
   V_LSB ──[R25A = 2.5 MΩ 0.1%]──[R25B = 60 kΩ 0.1%]──┤
                                                │
                                                ▼
                                        ┌──────────┐
                                       ─┤6 -IN_B   │
                                        │          ├ pin 7 (OUT_B) ──► V_OUT net
                                  V_CM ─┤5 +IN_B   │                    │
                                        └──────────┘                    │
                                                          ┌─[R_FB_C2 = 10 kΩ 1%]─┘
                                                          │
                                                          └──► back to pin 6

   Wiring summary for U13B:
     pin 5 (+IN_B) ← V_CM
     pin 6 (-IN_B) ← R10 from V_LOW + R25A-R25B series from V_LSB + R_FB_C2 from pin 7
     pin 7 (OUT_B) → V_OUT net (also to TP2, and to ADS1115 ADC channel 0)
```

### 🔧 Placement

- **R10, R25A, R25B, R_FB_C2 all adjacent to U13B pin 6.** Same
  reasoning as U13A.
- **V_OUT trace to U9 (ADS1115 AIN0) is the most precision-
  sensitive analog signal on the board.** Keep it short, on the
  analog side of FB1, away from any SPI/I²C/CS line. If a digital
  trace must cross V_OUT, do it perpendicularly (not parallel) and
  ideally with a ground pour between them.

---

# Section E — Analog readout

## E1 — U9 (ADS1115 16-bit ADC)

Reads V_OUT (and the slice mid-points for debug) via I²C. Address
0x48.

### Pinout (VSSOP-10)

```
       ┌──────┐
ADDR ─┤1   10├── V_DD (A3V3)
ALERT/RDY ─┤2   9├── SCL
   GND ─┤3   8├── SDA
  AIN0 ─┤4   7├── AIN3
  AIN1 ─┤5   6├── AIN2
       └──────┘
```

### Connections

| Pin | Net | Notes |
|---|---|---|
| 1 (ADDR) | AGND | I²C address = 0x48 (with ADDR tied low) |
| 2 (ALERT/RDY) | unconnected | (or to a debug pad / GPIO if you want interrupt-based reads) |
| 3 (GND) | AGND |  |
| 4 (AIN0) | V_OUT | the final dot-product output — most important reading |
| 5 (AIN1) | V_MSB | debug — measures MSB summer alone |
| 6 (AIN2) | V_MID | debug — measures MID summer alone |
| 7 (AIN3) | V_LSB | debug — measures LSB summer alone |
| 8 (SDA) | I²C SDA bus |
| 9 (SCL) | I²C SCL bus |
| 10 (V_DD) | A3V3 | **decoupling cap close** |

### 🔧 Placement

- **Decoupling cap (one of C1–C14) adjacent to pin 10.**
- **Place U9 close to where V_OUT comes out of U13B.** AIN0 is
  the precision-sensitive trace; everything else is debug.
- **The four AIN traces should route on the analog side of the
  board** — they're high-impedance summing-stage outputs (the op-
  amps drive them low-impedance, but any noise picked up between
  the op-amp output and the ADC input shows up directly in your
  measurement).

---

# Resistor budget — what every R does

Pulling everything together so you can grep your schematic:

| Ref (or class) | Value | Role | Connects to |
|---|---|---|---|
| **R_TOP** (one of R1, R2, R3, R4, R5, R6, R8, R9, R11) | 10 kΩ 1 % | V_CM divider top | **A3V3** ↔ U11A pin 3 |
| **R_BOT** (one of the same set) | 10 kΩ 1 % | V_CM divider bottom | U11A pin 3 ↔ **AGND** |
| **R_FB_MSB** (same set) | 10 kΩ 1 % | MSB summer feedback | U11B pin 6 ↔ U11B pin 7 |
| **R_FB_MID** (same set) | 10 kΩ 1 % | MID summer feedback | U12A pin 2 ↔ U12A pin 1 |
| **R_FB_LSB** (same set) | 10 kΩ 1 % | LSB summer feedback | U12B pin 6 ↔ U12B pin 7 |
| **R_FB_C1** (same set) | 10 kΩ 1 % | Combiner 1 feedback | U13A pin 2 ↔ U13A pin 1 |
| **R_FB_C2** (same set) | 10 kΩ 1 % | Combiner 2 feedback | U13B pin 6 ↔ U13B pin 7 |
| **R_MISO** (same set) | 10 kΩ 1 % | MISO bus pull-up | MISO net ↔ **D3V3** |
| **R_SPARE** (last of the 9) | 10 kΩ 1 % | Spare / decoder enable bias | depends on schematic; check yours |
| R7 | 10 kΩ 0.1 % | Combiner 1 V_MSB input | V_MSB ↔ U13A pin 2 |
| R10 | 10 kΩ 0.1 % | Combiner 2 V_LOW input | V_LOW ↔ U13B pin 6 |
| R12, R13, R14, R15 | 40 kΩ 0.1 % | MSB slice summer inputs | each MSB digipot wiper ↔ U11B pin 6 |
| R16, R17, R18, R19 | 40 kΩ 0.1 % | MID slice summer inputs | each MID digipot wiper ↔ U12A pin 2 |
| R20, R21, R22, R23 | 40 kΩ 0.1 % | LSB slice summer inputs | each LSB digipot wiper ↔ U12B pin 6 |
| R24A + R24B | 2.5 MΩ + 60 kΩ 0.1 % | Combiner 1 V_MID input (series pair = 2.56 MΩ) | V_MID ↔ R24A ↔ R24B ↔ U13A pin 2 |
| R25A + R25B | 2.5 MΩ + 60 kΩ 0.1 % | Combiner 2 V_LSB input (series pair = 2.56 MΩ) | V_LSB ↔ R25A ↔ R25B ↔ U13B pin 6 |
| R26 | 4.7 kΩ 1 % | I²C SDA pull-up | SDA ↔ **D3V3** |
| R27 | 4.7 kΩ 1 % | I²C SCL pull-up | SCL ↔ **D3V3** |
| R28 | 13 Ω 5 % | AGND ↔ DGND star tie | **AGND** ↔ **DGND** (single point) |

**Resistors that connect to GND on at least one side:**
R_BOT (one end), R28 (one end). That's it.

**Resistors that connect to a power rail on at least one side:**
R_TOP (one end to A3V3), R_MISO (one end to D3V3), R26 + R27
(one end each to D3V3). Four resistors total.

**Every other resistor on the board has both ends on signal nets** (a
chip pin, an op-amp pin, or a digipot wiper). If your schematic has
any other resistor with a pin tied to GND or to a rail, that's where
the wiring went wrong.

---

# Capacitor placement — every C, where it goes

| Cap | Value | Role | Where to place |
|---|---|---|---|
| C1 | 0.1 µF X7R 0805 | Decoupling for U1 (MCP4251 #1) | Adjacent to U1 pin 14 (V_DD), ≤ 3 mm trace |
| C2 | 0.1 µF | Decoupling for U2 | Adjacent to U2 pin 14 |
| C3 | 0.1 µF | Decoupling for U3 | Adjacent to U3 pin 14 |
| C4 | 0.1 µF | Decoupling for U4 | Adjacent to U4 pin 14 |
| C5 | 0.1 µF | Decoupling for U5 | Adjacent to U5 pin 14 |
| C6 | 0.1 µF | Decoupling for U6 | Adjacent to U6 pin 14 |
| C7 | 0.1 µF | Decoupling for U7 (74HC138) | Adjacent to U7 pin 16 (V_CC) |
| C8 | 0.1 µF | Decoupling for U8 (MCP4728) | Adjacent to U8 pin 10 (V_DD) |
| C9 | 0.1 µF | Decoupling for U9 (ADS1115) | Adjacent to U9 pin 10 (V_DD) |
| C10 | 0.1 µF | Decoupling for U10 V_in side | Adjacent to U10 pin 3 (input filter) |
| C11 | 0.1 µF | Decoupling for U10 V_out side | Adjacent to U10 pin 2 (output filter) |
| C12 | 0.1 µF | Decoupling for U11 (OPA2333) | Adjacent to U11 pin 8 (V+) |
| C13 | 0.1 µF | Decoupling for U12 (OPA2333) | Adjacent to U12 pin 8 |
| C14 | 0.1 µF | Decoupling for U13 (OPA2333) | Adjacent to U13 pin 8 |
| C15 | 10 µF X5R 0805 | 5 V input bulk | ≤ 5 mm from J1's 5 V pin |
| C16 | 10 µF | LDO output bulk (pre-FB1) | ≤ 5 mm from U10 pin 2 |
| C17 | 10 µF | A3V3 rail bulk (post-FB1) | ≤ 5 mm from FB1's analog-side pin |
| C18 | 10 µF | D3V3 rail bulk | Anywhere on the D3V3 net, ideally near U7 |
| C19 | 10 µF | DAC supply bulk | Near U8 pin 10 (optional reservoir) |
| C20 | 10 µF | V_CM buffer output bulk | Near U11A pin 1 (low-impedance V_CM source) |

C1–C14 are the same Samsung 0805 X7R 0.1 µF — functionally
interchangeable. Match each to its position in the schematic by
"closest IC."

---

# Physical placement at-a-glance

When in doubt, follow this layout flow from the J1 edge of the board
to the V_OUT edge:

| Zone | Components | Why this zone |
|---|---|---|
| **Edge (J1 side)** | J1, C15 | The Pi connects here; bulk cap on incoming 5 V |
| **Power-entry zone** | U10, C10, C11, C16 | LDO with input/output decoupling; this is where 5 V becomes 3.3 V |
| **Domain boundary** | FB1, C17, C18, R28 | Ferrite bead splits A3V3 from D3V3; star-ground tie is the only AGND↔DGND link |
| **Digital zone** (D3V3 side of FB1) | U7, R26, R27, R_MISO, C7, C18 | Decoder, I²C/MISO pull-ups; isolated from analog |
| **Digipot row** (A3V3 side, near analog) | U1, U2, U3, U4, U5, U6, C1–C6 | Six digipots in a row, decoupling caps adjacent to each |
| **DAC** | U8, C8, C19 | DAC drives input voltages; place near the digipot row |
| **Op-amp slice summers** | U11 (B half), U12, with R12–R23 inputs and R_FB feedbacks | Slice summers consume digipot wipers and produce V_MSB/V_MID/V_LSB |
| **V_CM buffer** | U11 (A half), R_TOP, R_BOT, C20 | Generates the midrail reference; place close to U11 with the divider adjacent to pin 3 |
| **Combiner stack** | U13, with R7, R10, R24A/B, R25A/B, R_FB_C1, R_FB_C2 | Two-stage cascaded combiner; precision-critical |
| **ADC** | U9, C9 | Reads V_OUT; place close to U13B pin 7 |
| **Test points** | TP1–TP8 | Distributed at the natural probe points (V_CM, V_OUT, V_LOW, V_LSB, A3V3, D3V3, AGND, DGND) |

### One-line layout philosophy

**Every analog signal trace flows in one direction across the
board: J1 → DAC → digipots → slice summers → combiners → ADC. Every
digital signal trace stays on the digital side of FB1. The two
domains meet only at R28.** If your layout matches this, the noise
floor will land where Stage 0's simulator predicted. If digital
traces criss-cross the analog signal path or the AGND/DGND planes
have multiple ties, the noise floor will be worse than the budget
and Experiment 2 will fail in ways that are very hard to diagnose
after the fact.

---

# Diagnostic checklist — find the bug

For each chip, walk your schematic with this checklist:

### Power and ground

- [ ] Every chip's V_DD / V_CC pin has a 0.1 µF cap adjacent and
      ties to the right rail (A3V3 for analog, D3V3 for digital).
- [ ] Every chip's V_SS / GND pin ties to the right ground (AGND or
      DGND — usually the chip's analog or digital nature dictates
      which).
- [ ] AGND and DGND are connected at exactly **one** point — R28.
      Anywhere else they touch, you have a ground loop.
- [ ] The 5 V trace from J1 reaches U10 pin 3 with C15 along the
      way; C16 / C17 sit on each side of FB1 with appropriate ties.

### Op-amp wiring (the V_CM trap)

- [ ] **U11A is the ONLY op-amp** with R_TOP from 3.3 V and R_BOT
      from GND meeting at its (+) pin. Every other op-amp's (+)
      pin connects to **V_CM** (the output of U11A), not directly
      to 3.3 V or GND.
- [ ] Every op-amp's (-) pin has multiple resistors converging
      there: input resistors (going to signal nets, NOT to GND or
      3.3 V) plus the feedback resistor (going to that op-amp's
      own output pin).
- [ ] U11A's pin 2 (-IN) is shorted DIRECTLY to pin 1 (OUT). No
      resistor in this loop. (Unity-gain follower.)
- [ ] Feedback resistor on every other op-amp (U11B, U12A, U12B,
      U13A, U13B) closes back from output to the same op-amp's (-).

### Combiner-precision check

- [ ] R7 (10 kΩ 0.1 %) sits between V_MSB and U13A pin 2.
- [ ] R10 (10 kΩ 0.1 %) sits between V_LOW and U13B pin 6.
- [ ] R24 is implemented as R24A (2.5 MΩ) + R24B (60 kΩ) in series,
      with no branch between them; the series pair sits between
      V_MID and U13A pin 2.
- [ ] R25 is the same arrangement (R25A + R25B = 2.56 MΩ effective)
      between V_LSB and U13B pin 6.

### Digipot wiring

- [ ] Each MCP4251's pin 1 (CS) connects to one of U7's Y0..Y5
      outputs — and **each** Y0..Y5 connects to **exactly one**
      MCP4251.
- [ ] Each MCP4251's pin A (P0A and P1A — pins 8 and 7) connects
      to V_in[i] from U8.
- [ ] Each MCP4251's pin B (P0B and P1B — pins 10 and 5) connects
      to **V_CM**.
- [ ] Each MCP4251's wiper (P0W or P1W — pins 9 and 6) connects
      through a 40 kΩ 0.1 % resistor (one of R12–R23) to the
      appropriate slice summer's (-) input.

### Common silent killers

- [ ] **The V_CM divider midpoint is not also tied to V_CM**
      (sometimes a stray net merge). Check that R_TOP-R_BOT meet
      at U11A pin 3, and that pin 3 is NOT directly connected to
      U11A pin 1 (which is V_CM proper). The buffer transforms
      "noisy divider midpoint" → "low-impedance V_CM"; if you short
      around the buffer, it's still a divider but with ten times
      the noise.
- [ ] **Decoupling caps tied to the wrong rail.** Analog chips
      (U1–U6, U8, U9, U11–U13) need their decoupling caps on the
      A3V3/AGND side. Digital chips (U7) on D3V3/DGND. A cap on
      the wrong rail is a low-impedance path for noise to cross
      domains.
- [ ] **R28 missing or replaced with a 0 Ω jumper at multiple
      locations.** R28 is one specific resistor at one specific
      location.
- [ ] **!G2A (U7 pin 4) wired to 3.3 V instead of to a Pi GPIO.**
      If !G2A is permanently high, the decoder is permanently
      disabled — none of the digipots ever get their CS asserted —
      and none of the SPI writes ever land.
- [ ] **!G2B (U7 pin 5) and G1 (U7 pin 6) flipped.** !G2B should
      be tied LOW (to DGND); G1 should be tied HIGH (to D3V3). If
      reversed, the decoder is permanently disabled.

If you walk the checklist and find a specific pin that doesn't
match, share which one ("U12B pin 5 goes to 3.3 V") and I can
confirm whether to move it to V_CM or whether the underlying issue
is something else.
