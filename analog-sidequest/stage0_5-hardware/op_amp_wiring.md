# Op-amp wiring reference — U11 / U12 / U13

*This is a reference schematic for the three OPA2333 dual op-amp packages
on the Stage 0.5 board. It uses the R and C reference designators from
the canonical BOM so you can lay this side-by-side with your KiCad
schematic and find the place where things diverge.*

*This document does NOT edit your KiCad files. It exists purely to make
the wiring rules explicit.*

---

## The thing that's confusing you, in one paragraph

There is **exactly one place** on this board where two resistors share
a net on one end and go to different rails on the other. That place is
the **V_CM (midrail) reference divider**: one 10 kΩ from **3.3 V** to a
shared node, and another 10 kΩ from that same shared node down to
**GND**. The shared node is the + input of the V_CM buffer op-amp
(U11A in the layout below). That's it. Every other "resistor with
identical-looking nets" you see on the board is part of the inverting-
summer pattern, where the resistors join at the **(-) pin** of an
op-amp (the summing junction) and their other ends go to **signal
nets** — not to GND or 3.3 V.

If you have a resistor that *should* be in the inverting-summer loop
but one of its pins is wired to GND or 3.3 V, **that's the bug**.

---

## OPA2333 dual pinout (reminder)

```
              ┌──────────┐
       OUT_A ─┤1        8├─ V+ (3.3 V analog)
       -IN_A ─┤2        7├─ OUT_B
       +IN_A ─┤3        6├─ -IN_B
          V- ─┤4        5├─ +IN_B
              └──────────┘
                  VSSOP-8

   V- = AGND.   V+ = analog 3.3 V (after the FB1 ferrite bead).
   Each package gives you TWO independent op-amps: A (pins 1-3) and B (pins 5-7).
```

A 0.1 µF decoupling cap (one of C1–C14) sits between **pin 8 (V+)**
and **pin 4 (V-)** **right next to the chip** — no more than a few
millimeters of trace, ideally underneath the package on the back side.
That cap is the FIRST thing to place when you populate.

---

## Channel-by-channel wiring (six op-amp roles, three packages)

Below I assume the canonical channel assignment:

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

### Role 1 — V_CM buffer (U11A) — the **only** stage with resistors to GND and 3.3 V

This is the unique stage on the board. Two 10 kΩ resistors form a
voltage divider that drops 3.3 V to ~1.65 V; the op-amp follows that
midpoint and outputs a low-impedance V_CM that every other op-amp's
**(+) input** uses as its bias reference.

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
              U11A out ◄────────┤1        8├──── V+ (3.3 V analog)
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

**This is the ONLY op-amp in the design where:**
- A resistor (R_TOP) connects from **3.3 V** to one pin
- A resistor (R_BOT) connects from **GND** to that same pin
- That pin (U11A's +IN_A) is the only node fed by both rails

If your schematic has resistors going to GND and/or 3.3 V at any
**other** op-amp's +IN, that's a wiring mistake. Every other op-amp's
+IN connects to the **V_CM net** (the output of U11A), not to a rail
directly.

**Important:** U11A pin 2 (-IN_A) is shorted DIRECTLY to pin 1 (OUT_A).
There is no resistor in this loop. A unity-gain follower has no
feedback resistor.

---

### Role 2 — MSB slice summer (U11B)

Inverting summer with **four input resistors** (one per input voltage),
**one feedback resistor**, and the (+) input tied to V_CM (no resistor).

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
                       (the feedback resistor closes back to the same -IN_B pin)


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

`R_FB_MSB` is one of the unallocated 10 kΩ 1 % resistors from the
{R1, R2, R3, R4, R5, R6, R8, R9, R11} pool.

---

### Role 3 — MID slice summer (U12A)

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

---

### Role 4 — LSB slice summer (U12B)

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

---

### Role 5 — First combiner (U13A): V_LOW = V_MSB + V_MID/256

Two-input inverting summer. The 1:1/256 ratio comes from the **input**
resistors, not the feedback. **R7 must be 10 kΩ 0.1 %**, **R24 must be
2.5 MΩ + 60 kΩ in series at 0.1 %**. The feedback (R_FB1) is just
10 kΩ 1 %.

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

---

### Role 6 — Second combiner (U13B): V_OUT = V_LOW + V_LSB/256

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

---

## The complete resistor budget — what every R does

Pulling the above together so you can grep your schematic:

| Ref (or class) | Value | Role | Connects to |
|---|---|---|---|
| **R_TOP** (one of R1, R2, R3, R4, R5, R6, R8, R9, R11) | 10 kΩ 1 % | V_CM divider top | **3.3 V** ↔ U11A pin 3 |
| **R_BOT** (one of the same set) | 10 kΩ 1 % | V_CM divider bottom | U11A pin 3 ↔ **GND** |
| **R_FB_MSB** (same set) | 10 kΩ 1 % | MSB summer feedback | U11B pin 6 ↔ U11B pin 7 |
| **R_FB_MID** (same set) | 10 kΩ 1 % | MID summer feedback | U12A pin 2 ↔ U12A pin 1 |
| **R_FB_LSB** (same set) | 10 kΩ 1 % | LSB summer feedback | U12B pin 6 ↔ U12B pin 7 |
| **R_FB_C1** (same set) | 10 kΩ 1 % | Combiner 1 feedback | U13A pin 2 ↔ U13A pin 1 |
| **R_FB_C2** (same set) | 10 kΩ 1 % | Combiner 2 feedback | U13B pin 6 ↔ U13B pin 7 |
| **R_MISO** (same set) | 10 kΩ 1 % | MISO bus pull-up | MISO net ↔ **D3.3 V** |
| **R_SPARE** (last of the 9) | 10 kΩ 1 % | Spare / decoder enable bias | depends on schematic; check yours |
| R7 | 10 kΩ 0.1 % | Combiner 1 V_MSB input | V_MSB ↔ U13A pin 2 |
| R10 | 10 kΩ 0.1 % | Combiner 2 V_LOW input | V_LOW ↔ U13B pin 6 |
| R12, R13, R14, R15 | 40 kΩ 0.1 % | MSB slice summer inputs | each MSB digipot wiper ↔ U11B pin 6 |
| R16, R17, R18, R19 | 40 kΩ 0.1 % | MID slice summer inputs | each MID digipot wiper ↔ U12A pin 2 |
| R20, R21, R22, R23 | 40 kΩ 0.1 % | LSB slice summer inputs | each LSB digipot wiper ↔ U12B pin 6 |
| R24A + R24B | 2.5 MΩ + 60 kΩ 0.1 % | Combiner 1 V_MID input (series pair = 2.56 MΩ) | V_MID ↔ R24A ↔ R24B ↔ U13A pin 2 |
| R25A + R25B | 2.5 MΩ + 60 kΩ 0.1 % | Combiner 2 V_LSB input (series pair = 2.56 MΩ) | V_LSB ↔ R25A ↔ R25B ↔ U13B pin 6 |
| R26 | 4.7 kΩ 1 % | I²C SDA pull-up | SDA ↔ **D3.3 V** |
| R27 | 4.7 kΩ 1 % | I²C SCL pull-up | SCL ↔ **D3.3 V** |
| R28 | 13 Ω 5 % | AGND ↔ DGND star tie | **AGND** ↔ **DGND** (single point) |

**Resistors that connect to GND on at least one side:**
R_BOT (one end), R28 (one end). That's it.

**Resistors that connect to 3.3 V (or D3.3 V) on at least one side:**
R_TOP (one end to 3.3 V), R_MISO (one end to D3.3 V), R26 + R27
(one end each to D3.3 V). Four resistors total.

**Every other resistor on the board has both ends on signal nets** (a
chip pin, an op-amp pin, or a digipot wiper). If your schematic has
any other resistor with a pin tied to GND or to a rail, that's where
the wiring went wrong.

---

## Decoupling caps — placement, not topology

Decoupling caps are simpler — they all go from a power pin to ground,
right next to the chip:

| Cap (one of C1–C14) | From | To | Where to place |
|---|---|---|---|
| ~6 caps | each MCP4251 V_DD pin | AGND | adjacent to the chip's V_DD pin, ≤ 3 mm trace |
| 1 cap | 74HC138 V_CC pin | DGND | adjacent to U7 |
| 1 cap | MCP4728 V_DD pin | DGND | adjacent to U8 |
| 1 cap | ADS1115 V_DD pin | DGND | adjacent to U9 |
| 2 caps | AMS1117 V_in and V_out pins | GND | one per side of the LDO |
| 3 caps | each OPA2333 V+ pin (U11, U12, U13) | AGND | one per package, between pin 8 and pin 4 |

C1–C14 are functionally interchangeable — same Samsung 0805 X7R
0.1 µF part. Match each to its position in your schematic by
"closest IC" and you're fine.

The bulk caps (C15–C20) are different — those go on the rails, not
on individual chip power pins:

| Cap | Where |
|---|---|
| C15 | 5 V input rail, near the J1 header |
| C16 | AMS1117 output (post-LDO, pre-ferrite) |
| C17 | Analog 3.3 V rail (post-FB1 ferrite bead) — the "A3V3" net |
| C18 | Digital 3.3 V rail — the "D3V3" net |
| C19 | DAC/ADC supply pin (typically tied to D3V3 anyway, with this cap as local reservoir) |
| C20 | V_CM buffer output — large reservoir on the buffered midrail |

---

## How to use this doc to find the bug

For each op-amp role above, walk your schematic:

1. Locate the op-amp pin.
2. Read every wire leaving that pin — what net is it on?
3. For each net, ask: does it match the table above?

Specifically:

- **Anywhere a +IN pin (pin 3 or pin 5 of any op-amp other than U11A's pin 3) is tied to GND or to 3.3 V instead of V_CM** — that's wrong.
- **Anywhere a -IN pin has a wire going directly to GND or 3.3 V** (with no resistor between the pin and the rail) — that's wrong.
- **Anywhere a feedback resistor's "other end" goes to GND instead of to the op-amp's output pin** — that's wrong.
- **Anywhere R_TOP or R_BOT is missing** (no path from 3.3 V through 10 kΩ to U11A pin 3, or no path from U11A pin 3 through 10 kΩ to GND) — V_CM won't be at 1.65 V and nothing else will work.

If you want, share which Rs you currently have wired to GND or 3.3 V
and which op-amp pins they connect to, and I can tell you whether
each one is correct or which role it should be playing instead.
