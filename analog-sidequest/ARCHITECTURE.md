# Architecture — the whole stack, piece by piece

*This document names every major component in the analog-MinAI system
and explains what it does, why it is there, and what it corresponds to
in a normal computer. The goal is to make the whole system
legible in one sitting.*

---

## The big picture in one diagram

```
  microSD card                       (trained weights, like a hard disk)
        |
        v
  Raspberry Pi Zero 2W               (host: weight loading at boot,
        |                             tokenizer, generation-loop control)
        |
        |   1× DAC at start of each generation step
        v
  +---------------------------------------------------------+
  |  Analog Tile Board — everything below stays at voltage  |
  |                                                         |
  |   Bit-sliced digital-potentiometer crossbars            |
  |   (the weights, programmed once at boot)                |
  |        |                                                |
  |        v                                                |
  |   Analog matmul network    (Ohm's law + Kirchhoff)      |
  |        |                                                |
  |        v                                                |
  |   Analog residual adders   (summing op-amps)            |
  |        |                                                |
  |        v                                                |
  |   Analog LayerNorm / RMSNorm stages                     |
  |        |                                                |
  |        v                                                |
  |   Analog ReLU / GeLU       (diode / BJT)                |
  |        |                                                |
  |        v                                                |
  |   ... repeat per transformer block ...                  |
  |        |                                                |
  |        v                                                |
  |   Analog softmax           (diode-exp + summing node    |
  |        |                    + translinear divider)      |
  |        v                                                |
  |   Analog sampler           (winner-take-all OR ramp-    |
  |        |                    compare temperature/top-k)  |
  +---------------------------------------------------------+
        |
        |   1× small ADC per generated token (5–7 bits — a token ID)
        v
   Raspberry Pi Zero 2W              (receives token ID, appends to
                                      prompt, loops back to DAC)
```

**One DAC at the start of each generation step. One ADC at the end.
Nothing between them leaves the analog domain.** Softmax does not
round-trip to the host. Neither does sampling, neither does
LayerNorm, neither does the nonlinearity. The host is a typewriter
that types one token in and reads one token out per cycle; every
piece of vector arithmetic happens on the board.

---

## The mapping to a normal computer

| Normal computer | This system | Why it's there |
|---|---|---|
| Hard disk / SSD | **microSD on the Pi** | Holds trained weights at rest. Just a file. |
| System RAM | **Pi's 512 MB DRAM** | Stages weights during boot. Runs the host's Python/C++ glue code. |
| CPU | **Pi's ARM cores** | Tokenizes prompts, runs the generation loop, does EOS check. **Does not run softmax. Does not run sampling. Does not run any vector arithmetic in the inference path.** |
| GPU | **The analog tile board (matmul + nonlinearity + LayerNorm + softmax + sampler)** | All of the inference work, from input embedding voltage to output token ID. The only new thing in this project. |
| PCIe bus | **SPI/I²C link from Pi to board** | How the CPU talks to the accelerator. (In Stage 3, this really is PCIe.) |
| GPU VRAM / HBM | **The digipot wiper state on the tile** | Holds weights during inference. Volatile — vanishes on power-off. |
| GPU kernel upload | **Pi's boot script walks the weight file and clocks each value into the digipot register over SPI** | Same pattern as `cudaMemcpy`, just ending at a resistance instead of a RAM cell. |

The one-line summary: **every piece of this system has a direct
counterpart in a normal GPU-based AI computer, except the matmul tile
itself, which replaces a stack of floating-point multipliers with
Ohm's law.**

---

## Architectural rule #1 — stay on the board

Stated up front because it shapes every other decision in this
document and overrides any per-stage convenience argument:

> **Inside the inference path, every piece of arithmetic that *can*
> happen in analog *does* happen in analog. The signal enters the
> board once via DAC, leaves once via ADC, and never round-trips in
> between.**

Concretely: matmuls, residual adds, ReLU/GeLU/tanh, LayerNorm /
RMSNorm, softmax, and sampling all live on the board. The host's
work per generated token is reduced to the four irreducibly-digital
operations: tokenize the prompt at start, drive one DAC at the
start of each step, read one ADC token-ID at the end of each step,
decide whether to keep generating.

Why this rule and not "softmax is cheap, just do it on the host":

1. **Energy.** Each ADC-DAC round-trip costs more energy than the
   matmul that sits between them on small models. The whole pitch
   of analog AI — 50–100× lower energy than GPU inference — only
   holds if conversions are rare. One per token is rare; one per
   layer is not. One per softmax is catastrophic on a long
   generation.
2. **Latency.** ADC settling at the precision we need (24+ bits,
   delta-sigma) takes microseconds. The matmul itself takes tens
   of nanoseconds. A round-trip per layer makes the analog speed
   advantage disappear into the converters.
3. **Precision.** Every ADC-DAC pair quantizes the signal twice. At
   24 effective bits per pass, eight round-trips through a
   transformer block stack drop the effective precision into the
   teens — exactly the territory where digital quantization
   research says training and inference start hurting. By keeping
   the signal continuous through the block, the only quantization
   is the boundary one at the output of the whole pipeline.
4. **Architectural identity.** Once you go to the host for
   softmax, you might as well go to the host for LayerNorm. Once
   you do that, you might as well go for residual adds. The
   product collapses into "a digital transformer with some analog
   inside it" — which is exactly the failure mode this project
   exists to avoid. The discipline of "everything on the board"
   keeps the architecture honest.

The 1970s-1990s analog-computing literature has stock primitives for
every operation in a transformer's inference path. Diode-exp,
translinear divide, BJT differential-pair tanh, summing-node mean,
squaring multiplier, square-root cell. All of them are two- to
four-transistor circuits. None of them is novel. The only reason
to do any of these on the host is reflex — the reflex of an
engineer who learned to think of the host as the place where
"messy" math happens. We are unlearning that reflex.

---

## The tile, in more detail

The analog MAC tile is the heart of the whole project. Everything else
exists to load weights into it and to consume its outputs.

### What one tile does

Given N input voltages `V[0..N-1]` and an N×M matrix of conductances
`G[i,j]`, the tile produces M output currents `I[0..M-1]` such that:

```
I[j]  =  Σ_i   V[i] × G[i,j]
```

This is exactly the matrix-vector product a transformer's `Q = X @ Wq`
step computes. It happens in **one pass of electrons through a
resistor grid**, at the speed of op-amp settling (tens of nanoseconds).
No clock. No instructions. No floating-point rounding. The matmul is
the physics of the circuit.

### What lives at each crosspoint

For a digital-potentiometer-based tile (the Stage-1 choice):

```
  row i  -------+-------+-------+-------+---
                |       |       |       |
               [R]     [R]     [R]     [R]     each R is one digipot
                |       |       |       |      programmed to the weight
           col 0 col 1 col 2 col 3              value G[i,j]
```

Each column is tied to the virtual-ground input of a summing op-amp.
The op-amp outputs a voltage proportional to the total current on its
column — that is `I[j]`, scaled. That is the output of one row of the
matmul, for all columns in parallel.

### What a digipot actually is

A **digital potentiometer** is a resistor chain with a digitally
controlled tap. Typical parts (AD5270, MCP4131) have 256 or 1,024 taps
across a total resistance of 10–100 kΩ. You set the tap via SPI or
I²C; the tap position determines the analog resistance presented at
the pins. From the matmul's point of view, the tap setting *is* the
weight, stored as a digital register inside the chip. The chip holds
the register as long as it is powered. On power-off, the register
resets to its default (usually midscale). That is the "volatile
analog weight" model in one sentence.

### Why not memristors yet?

For Stage 1 we want to iterate in hours, not months, using tools that
fit on a workbench. A digipot is an off-the-shelf SMD part that costs
$1–$5, solders in minutes, and never requires a cleanroom. A
memristor requires oxide deposition and a university fab partner. The
*architecture* we are building is memristor-compatible — if we ever
do a silicon tapeout, the digipots become RRAM crosspoints and
nothing else in the design changes. The digipot is the
development-time stand-in for the eventual silicon crosspoint.

---

## The boot sequence

Every session of this machine goes through the same boot flow. This
is worth spelling out because it is the answer to the question
"where does the memory of states come from at boot?"

```
1. Power on.
   Pi boots Linux from microSD. Analog board powers up with digipots
   at their default tap positions (not the trained weights yet —
   random-ish initial state).

2. Pi reads the trained weight file from microSD into DRAM.
   The weight file is MinAI's trained weights, in the same format the
   book's C++ program produces. A few kilobytes.

3. Pi walks the weight array, scaling each float weight to the
   digipot's integer tap range and writing it over SPI. For 256
   digipots this takes roughly a millisecond.

4. The board is now "programmed." Each digipot wiper sits at the
   position that corresponds to one trained weight.

5. Pi looks up the embedding row for the input token in its DRAM
   weight table; clocks the row's values out via the input DAC as
   N analog voltages on the board.

6. The signal traverses the entire on-board transformer pipeline at
   analog voltage levels: per block — Wq/Wk/Wv tiles, Q·Kᵀ, attn·V,
   residual adders, LayerNorm stage, W1, ReLU, W2, residual.
   Then the final LN, the Wout tile, the analog softmax (diode-exp +
   translinear divider), and the analog sampler. **No ADC, no DAC,
   no digital arithmetic inside this pipeline.** Op-amps settle in
   tens of nanoseconds per stage.

7. The on-board sampler emits a token ID (a small integer between 0
   and the vocabulary size). One small ADC (5–7 bits) reads that
   integer back to the Pi.

8. Pi appends the received token to the prompt; loops back to step 5
   for the next generation step.

9. Power off. Digipot state vanishes. Trained weights are still safe
   on the microSD card, ready for the next boot.
```

The boot is identical in spirit to loading a CUDA kernel from disk to
VRAM. The only change is that the final destination is an analog
resistance rather than a digital register. The generation loop is
much smaller than a GPU equivalent: the Pi's per-token work is "look
up an embedding, drive a DAC, read a token ID, decide whether to
stop." Everything else lives in physics.

---

## What the card offloads vs. what stays digital

The honest menu for a transformer forward pass. The goal is
**everything stays analog inside the generation loop; digital only
at the boundaries of the system.** Boundaries means: weights
arriving from flash at boot, prompt tokens arriving from the
user, token IDs leaving for the user's terminal, and the loop
control that says "generate the next token." Nothing in between.

| Step | Where it runs | Why |
|---|---|---|
| Tokenize prompt | Host | Trivial, rare, needs BPE merge tables in RAM. Digital wins. |
| Embedding lookup | Host DAC | A table read, not a matmul. The embedding table lives in digital memory; the selected row gets DAC'd into the analog path. |
| **Linear projections (Q, K, V, Wout, W1, W2)** | **Analog tiles** | 80–90% of total compute. Exact Kirchhoff + Ohm physics. |
| **Attention scores (Q · Kᵀ) and attention-weighted values (attn · V)** | **Analog tiles** | More matmul. Same argument. |
| **Residual add** | **Analog — summing op-amps on the board** | One op-amp per channel. Kirchhoff exact. |
| **ReLU / GeLU / other pointwise nonlinearities** | **Analog — diode or BJT patch** | Analog physics; *more accurate than digital approximations.* |
| **LayerNorm / RMSNorm** | **Analog — on-board LN stage** | See its own section. Keeps the block signal path fully analog. |
| **Softmax** | **Analog — diode `exp` + summing node + translinear divider** | Classical analog primitive. Physics is exact; noise limits it. No reason to round-trip to host. |
| **Sampling (argmax / top-k / top-p / temperature)** | **Analog — winner-take-all or ramp-compare sampler** | Stock analog circuits from the 1980s. Random bits from a thermal noise source on-chip. |
| KV cache storage | Analog at MaxAI scale; digital DRAM at GPT scale | Small caches fit in analog cells. Large caches (hundreds of GB at frontier scale) need DRAM. The book's scale keeps KV analog. |
| Output token ID readout | Host ADC | 5–7 bits per token (enough for a 32–128-symbol vocabulary). One tiny ADC per generated token. |
| Generation loop, EOS check | Host | Boolean control flow. |

**The architectural rule, in one line: all matmuls, all
nonlinearities, softmax, and sampling stay in analog. The host
does weight-loading at boot, prompt-input at start, token-readout
at end, and loop-control in between. That is the entire digital
footprint.**

Which means the number of ADC-DAC round-trips per generated token
is exactly ONE (reading the chosen token's ID out at the end).
Not one per layer, not one per softmax, not one per sampling step.
One total. Every gain in analog efficiency compounds; nothing is
lost to conversion energy.

---

## LayerNorm as an on-board analog stage

The design temptation is to keep LayerNorm digital "because it's
messy." That temptation costs you the architecture. A Pre-LN
transformer has *two* LayerNorms per block (before attention, before
FFN) plus one final LN before the output projection. If each LN
pushes the signal off the analog tiles, into an ADC, through a
host-side mean/variance/sqrt/divide, back out through a DAC, and
onto the next tile, every block costs two round-trips. The analog
advantage survives the first round-trip cleanly; by the eighth it
has been sawtoothed into the digital SNR floor. You end up with a
"digital transformer with some analog inside it" — which is a
pointless product.

The fix is to put LayerNorm on the board as its own analog block,
sitting in-line between matmul tiles. Cost: roughly 50–80 extra
components per LN stage. Benefit: the signal stays at analog
precision through the entire block, and the overall forward pass
does one round-trip *per generated token* instead of one *per
layer*.

### What LayerNorm decomposes into, in circuit form

```
y_i = (x_i - mean(x)) / std(x) * gain_i + bias_i
```

Each step maps onto a well-understood analog primitive from the
1970s–1980s analog-computing literature:

1. **Mean across the D_MODEL channels.** Sum all D channels into a
   virtual-ground summing node with equal weights; scale by `1/D`
   via a feedback resistor. One op-amp. Output is a voltage
   representing `mean(x)`.
2. **Subtract the mean from each channel.** One difference op-amp
   per channel. D op-amps total. Output is `x_i − mean(x)`.
3. **Squared deviation per channel.** An analog multiplier
   (Gilbert-cell IC, or four matched BJTs arranged as a squarer)
   computes `(x_i − mean)²`. D multipliers total. Biggest single
   chunk of added component count; still small in absolute terms.
4. **Variance.** Sum the squared deviations into another summing
   node, scale by `1/D`. Same topology as step 1.
5. **Square root (= std).** A translinear square-root cell — four
   BJTs and a few resistors, a stock primitive from Gilbert (1975).
   One cell, shared across all channels of that LN stage.
6. **Divide centered values by std.** A translinear divider (another
   four-BJT cell). One divider per channel, fully parallel.
7. **Per-channel gain and bias.** Gain is one digipot per channel
   (trainable weight); bias is a DAC-driven voltage added at the
   output stage. Standard.

Total per LN stage, at `D_MODEL = 16`: roughly **50–80 extra
components** (op-amps, multipliers, BJTs, digipots, DACs). Fits
comfortably alongside the matmul tiles it sits between — an LN
stage occupies maybe 20% of the PCB area of one tile.

### Two simplifications worth A/B testing

- **RMSNorm instead of LayerNorm.** Llama 2+, Mistral, DeepSeek, and
  most 2024+ models use RMSNorm: `y = x / RMS(x) × gain`. It skips
  the mean subtraction entirely — no step 2 in the list above, no
  bias term — and empirically matches LayerNorm on every published
  benchmark. **Recommendation: train the analog-deployed variant
  with RMSNorm rather than LayerNorm.** The circuit simplification
  is significant (cuts component count per LN stage by roughly a
  third) and the accuracy cost is zero.
- **Approximate std via absolute-value-mean.** A classic analog-
  computing shortcut: `std(x) ≈ c × mean(|x|)` for a
  distribution-dependent constant `c`. Replaces the square → sum →
  sqrt chain with a single-diode rectifier and a summing stage. A
  few percent accuracy cost; cuts LN component count in half. Worth
  testing against both LayerNorm and RMSNorm once Stage 2 tiles are
  working.

Neither is a design-time decision that needs to be locked in at
Stage 1. Both are A/B-able once there are tiles to measure on.

---

## The stack pattern — all-analog from input to output

Once LN is an on-board block, the full card layout is a stack of
alternating analog stages — literally a chain of mixed-signal
modules soldered next to each other, each feeding the next at
analog voltage levels, with no digital interruption inside a block.

```
(DIGITAL SIDE — host)
  |
  prompt tokens → embedding lookup (table) → DAC
  |
  v  (DAC-to-analog boundary, entering the card)

(ANALOG SIDE — the card, everything below stays at voltage level)

  [LN stage]                           -+
      |                                 |
      v                                 |
  [Wq tile]  [Wk tile]  [Wv tile]       |
                |                       |
                v                       |
  [Q · Kᵀ tile]                         |
                |                       |   per block
                v                       |
  [attn · V tile]                       |
                |                       |
                v                       |
  [residual-add op-amps]                |
                |                       |
                v                       |
  [LN stage]                            |
                |                       |
                v                       |
  [W1 tile]                             |
                |                       |
                v                       |
  [ReLU patch (diode/BJT)]              |
                |                       |
                v                       |
  [W2 tile]                             |
                |                       |
                v                       |
  [residual-add op-amps]               -+  <- end of block; feed to next
      |
      v
  ... next block, identical shape ...
      |
      v
  [final LN stage]
      |
      v
  [Wout tile]
      |
      v
  [analog softmax: diode-exp + summing node + translinear divide]
      |
      v
  [analog sampler: winner-take-all (greedy) OR
                   ramp-compare stochastic sampler (T / top-k / top-p)]
      |
      v
  (ANALOG-TO-DIGITAL BOUNDARY, leaving the card)
      |
      v  small ADC: 5–7 bits of output token ID

(DIGITAL SIDE — host)
  |
  receive token ID → append to prompt → loop back to DAC → repeat
  (no softmax, no sampling, no vector-wide digital arithmetic;
   the host's role is a typewriter that feeds the card its input
   and receives its output, one token at a time.)
```

One ADC-to-DAC round-trip per *generated token*, not per layer
and not per softmax. The host's total digital work per token is:
embed a scalar index into a table, DAC the selected row,
eventually ADC a scalar token ID, decide whether to keep going.
Everything else — matmul, attention, nonlinearity, LayerNorm,
softmax, sampling — happens in analog at analog speed.

Physical layout on a PCIe card: each stage is a sub-module on the
board, connected by short analog traces with careful impedance
matching. For a two-block MaxAI at `D_MODEL = 16`: roughly twenty-
four matmul tiles (at Q8.8.8 that's ~72 digipot clusters), five
LN stages, eight residual/ReLU patches, one softmax block, one
sampler block, an FPGA host, and DRAM for weight staging. All on
a half-length PCIe card with room to breathe.

---

## Analog training — yes, actually, we do it

Every commercial analog-AI startup from 2012 onward has followed
the same pattern: **train on NVIDIA, deploy on our chip.** Every
one that made it to market has then failed in the same way. The
customer arithmetic is simple and unavoidable: *if I need a GPU
for training anyway, I will just do the inference on it too and
skip the second vendor relationship.* Mythic died on this.
Lightmatter pivoted away from it. Graphcore was absorbed because
of it. It is the single most consistent failure mode in fifteen
years of analog-AI commercial history.

This project treats **training on the analog hardware as a
first-class requirement**, not something deferred to silicon. The
thesis is that training *can* run on analog if you accept the
precision compromises and apply five old tricks — none of them
new, all of them from the 1970s–1990s analog-computing literature,
most abandoned when digital got fast enough to make them
unnecessary. Together they move training from "intractable" to
"hard engineering problem with known solutions."

### The core framing: noise is the enemy, not representation

The right mental model for this project is:

> **Digital precision is bounded by representation. Analog
> precision is bounded by noise.**

Those are two different kinds of limits, and the engineering that
follows from each is different.

Digital's precision ceiling is baked into the number format. A
float32 multiplication can never be more accurate than its 23-bit
mantissa allows. Summing a thousand float32 values accumulates
rounding error because every intermediate sum rounds to fit.
Moving past that ceiling requires changing the number type —
float64, fp8, int4, whatever — and each type brings its own
hardware cost and software ecosystem baggage.

Analog has no representation ceiling. The voltage across a
resistor expresses any real number. Two currents at a junction
add with zero rounding. A diode's I-V curve *is* `exp()`, not an
approximation of it. The operations are exact physics. **The only
thing that limits analog precision is that we have to read those
exact values through a measurement channel, and the channel has
noise.** Drive the noise to zero and analog precision goes to
infinity. We can't drive it to zero, but we can chase it very
far, because every noise source has a specific engineering
countermeasure that has been known since the 1970s instrumentation
era.

This is why the project's engineering philosophy is **"catalog
every noise source, apply every mitigation that fits the
budget."** Each noise source on the list below has a specific
counter that peels bits off the floor:

- **Thermal (Johnson-Nyquist) noise** — fundamental. Counter:
  lower-R components, narrower readout bandwidth, active
  temperature stabilization, cryo at the extreme.
- **1/f (flicker) noise** — from semiconductor trap states.
  Counter: autozero/chopper-stabilized op-amps (commercial
  $2 parts like AD8629). Reduces this source to the thermal floor.
- **Shot noise** — quantization of electron flow. Counter: run at
  modestly higher currents where shot noise is dominated by
  thermal. Small trade.
- **Component tolerance** — nominal value vs. actual.
  Counter: precision thin-film (Vishay Z-foil at 0.001%,
  0.05 ppm/°C) and/or the calibration-reference-cell loop.
- **Thermal drift** — resistance changes with temperature.
  Counter: low-tempco parts + live calibration against the
  reference cells.
- **Power supply noise** — from switching regulators and shared
  rails. Counter: LDO regulators per stage, bandgap voltage
  references, separate analog and digital supplies and grounds.
- **Crosstalk** — capacitive and inductive coupling between cells.
  Counter: 8-layer stripline routing, guard traces, differential
  signaling on sensitive buses.
- **EMI** — from environment. Counter: shielded enclosure, input
  filtering. Standard instrumentation practice.
- **Readout quantization** — the ADC at the boundary can't exceed
  its bit depth. Counter: delta-sigma ADCs (ADS1262 is 32 bits);
  28–30 effective bits trivially at modest sample rates.

Stack every mitigation that fits the budget, and the achievable
effective precision at room-temperature PCB scale is **24–28
bits per operation** — which is *better than float32 digital*
(23-bit mantissa) on operations where analog is naturally exact.
Cryogenic operation would push to 30+ bits; we do not need that.

### Where analog actually beats digital (not just matches it)

For the operations that matter in a transformer, here is the
honest ranking of where analog has a true physics advantage over
digital, not merely a competitive trade-off:

| Operation | Digital | Analog | Winner |
|---|---|---|---|
| Summation of many terms | rounds every intermediate sum | Kirchhoff's current law, zero rounding | **Analog, often by several bits on long sums** |
| Multiplication | float32 mantissa bound at 23 bits | Ohm's law, limited only by noise | Tie at fp32; analog wins on energy |
| `exp(x)` | polynomial approximation, bounded error | diode I-V curve, exact physics | **Analog** |
| `log(x)` | same as above | diode in feedback path, exact | **Analog** |
| `tanh(x)` | polynomial, bounded error | BJT differential pair, exact | **Analog** |
| `sqrt(x)` | Newton-Raphson or hardware trick | translinear cell, exact | **Analog** |
| Division | hardware divider, fp32 mantissa | translinear cell, exact up to noise | Tie at fp32 |
| Boolean control flow | native | painful | Digital, always |
| Random access storage at scale | native (DRAM) | limited (cells don't scale) | Digital |

The matmul pipeline of a transformer is almost entirely in the
top half of that table. **Analog doesn't merely match digital at
our workload — it is physically more accurate than digital on
every operation except the three at the bottom, once noise is
adequately engineered down.**

### The hardware precision amplifiers

#### Hardware trick 1 — Bit-sliced cells (Q8.8.8 and beyond)

Your Q8.8 origin in the book's PDP-11 lineage has an obvious
extension to Q8.8.8 (and beyond). Each weight is stored across
**three** analog cells instead of one:

- Cell A holds the most-significant 8 bits of the weight.
- Cell B holds the next 8 bits, scaled by 1/256 at the summing
  junction.
- Cell C holds the lowest 8 bits, scaled by 1/65,536.

At readout, the three cells' currents sum with their appropriate
binary-weighted coefficients into the same op-amp. Effective
weight precision is **24 bits from 8-bit components**. Going
further to Q8.8.8.8 adds another 8 bits of nominal precision, but
device noise on the lowest-significance cell starts dominating
its contribution around the third slice — diminishing returns set
in, and three slices is the practical sweet spot.

Hardware cost: **3× cells per weight.** Same topology, same
layout, same fabrication; the tile simply has three digipot
groups instead of one, and the op-amp at each column has
binary-weighted feedback. At silicon scale, three 1T1R
crosspoints per weight instead of one — linear cost for quadratic
precision gain.

Published analog-AI literature refers to this as "bit-slicing" or
"multi-cell encoding." IBM has published extensively on it with
phase-change memory arrays. It is the single highest-leverage
trick on this list.

#### Hardware trick 2 — Autozero / chopper-stabilized op-amps on summing nodes

Standard op-amps have DC offset drift and 1/f noise that limit
effective precision at the summing junction to maybe 12 bits.
**Commercial autozero ("zero-drift") op-amps** (Analog Devices
AD8629, LTC2054, TI OPA387) self-sample and subtract their own
offset periodically, giving 130+ dB of DC drift rejection —
effectively **20+ bits of op-amp-path precision**. They cost ~$2
instead of ~$0.50 for a standard TLV9002.

Hardware cost: a few extra dollars per tile. Leverage is
extraordinary for the BOM impact. **Default choice for the
summing stage.**

#### Hardware trick 3 — Wide gradient accumulator (32 bits per weight)

The original plan was a 16-bit digital SRAM next to each tile for
the gradient accumulator (software trick 1 below). Widening that
to 32 bits per weight is essentially free — doubles the SRAM size
from 4 Kbit per tile to 8 Kbit; still a single inexpensive chip —
and gives the training loop 32-bit precision on gradient
accumulation even when the analog cell can only hold 24.

Hardware cost: negligible. **Default.**

#### Hardware trick 4 — Calibration reference cells

Dedicate 4–8 cells per 256-weight tile to programmed, known
values. Continuously measure their outputs during operation; the
deviation from expected gives a live-updating per-tile drift
correction that gets applied to all other cells' readouts.

Hardware cost: ~3% extra cells. **Default.**

### The software training tricks

With the hardware envelope widened by the four tricks above, the
software side still needs to manage training under quantization.
Five more classical tricks do that job, all in the order they
appear in a forward-backward cycle:

### Trick 1 — Mixed-precision updates with a digital gradient accumulator

Analog cells have ~8 bits of usable write precision. A raw SGD
update `w_new = w_old − lr × gradient` is usually smaller than one
analog step and would be rounded to zero.

Fix: place a small digital SRAM next to every tile — 16 bits per
weight, ~4 Kbit per 16×16 tile. Gradient updates accumulate there
at full 16-bit precision. When the accumulator for a specific
weight crosses the next analog-writable step (≥ 1/256 of full
scale), commit exactly one step to the analog cell and subtract
that step from the accumulator.

Cost: one small SRAM per tile. Tens of cents. Standard CMOS part,
no custom anything.

This is the approach IBM demonstrated on phase-change-memory arrays
(Nandakumar et al. 2020; Le Gallo et al. 2023). Training quality
within 1–2% of software baselines on networks of a few million
parameters. Published, peer-reviewed, replicated.

### Trick 2 — Stochastic rounding on every write

When the accumulator wants to commit a partial step (say 0.37 taps
of change), do not round to zero. Write +1 tap with probability
0.37, and 0 taps with probability 0.63. Over many updates the
weight converges to its true value despite 1-tap quantization per
write.

Cost: one random-number call per write. Free on the host FPGA.

Höhfeld & Fahlman introduced this in 1992; revived by Gupta,
Agrawal, and others at IBM in 2015. It is the defining trick of
every modern int4 training proposal and applies identically to
analog tile writes.

### Trick 3 — Temporal oversampling on reads (the sigma-delta trick)

A single tile readout gives 6–8 effective bits. Training's forward
and backward passes often need more, because layer-to-layer noise
accumulates and corrupts the gradient.

Fix: run the same matmul `N` times with a small dither voltage
added on each read; average the outputs digitally. `N = 16` buys
~2 extra bits; `N = 256` buys ~4. You trade inference latency for
precision — and you can make the trade per-layer, paying
oversampling cost only where gradient sensitivity demands it.

This is the classic sigma-delta ADC trick applied to analog
matmul. The math is identical.

### Trick 4 — Delta encoding of weight updates

Store and transmit weight changes as small deltas (easily
represented at 8 bits) rather than absolute weight values (large,
poorly represented at 8 bits). Apply additively.

Same insight that makes residual connections work for deep
networks: modeling a small correction is easier than modeling the
absolute answer. Applies to the tile's write channel with no
circuit changes.

### Trick 5 — Analog physics as a free lookup table

The original PDP-11 code used a 256-entry lookup table for `exp()`
because the machine could not compute it. **The analog tile does
not need the lookup.** A diode's I-V curve *is* `exp()`. A BJT's
base-emitter junction *is* `exp()`. The 1970s analog-IC handbooks
list transfer curves for `log`, `tanh`, `sqrt`, multiplication,
and division — each implementable in two to four transistors.

For training specifically, this matters at the softmax backward
(which needs exp) and the LayerNorm backward (which needs sqrt
and divide). In analog, those are not numerical approximations.
They are the physics of the circuit, exact to the tolerance of
the components.

### What precision the noise budget actually delivers

The architecture has two precision numbers, and conflating them
hides the engineering trade. Both are now measured by Stage 0
rather than estimated.

**Quantization ceiling** (Q8.8.8 representation, zero noise):
**23.3 effective bits.** This is the upper bound — what the
architecture would deliver if every noise source were driven to
zero. It is set by the bit-sliced cell count.

**Effective precision at operating noise** depends on how much
noise the BOM allows:

| Noise regime          | σ per-MAC | Effective bits | Use         |
|-----------------------|-----------|----------------|-------------|
| Zero noise (ceiling)  | 0         | **23.3**       | theoretical |
| Precision analog      | 3 × 10⁻⁵  | **14.3**       | aspirational|
| Standard analog (target) | 3 × 10⁻⁴ | **11.0**     | Stage 1 BOM |
| Hobbyist analog       | 3 × 10⁻³  | **7.7**        | bottom edge |

The Stage 1 BOM commits to the standard-analog regime: 11 effective
bits per MAC. That is **above** the 8-bit training floor IBM's
phase-change memory work hit (Nandakumar 2020, Le Gallo 2023), and
**well above** the int8 inference quality bar modern LLMs already
serve at. It is below float32's 23-bit mantissa, which is why
models retrain with measured tile noise injected — same approach
as int8 quantization-aware training, just with a different noise
distribution.

Component-by-component contributions to the noise budget the
11 bits comes out of:

- **Forward matmul (Kirchhoff + Ohm)**: exact physics, no
  representation error. The 11-bit limit is the noise on the
  summing node, not the matmul itself.
- **Weight writes (Q8.8.8 with stochastic rounding)**: 24-bit
  nominal, ~20-bit effective once stochastic rounding averages out
  the quantization across many updates.
- **Gradient accumulation (32-bit digital SRAM)**: no ceiling. The
  one place we deliberately use digital, because bookkeeping is
  digital's strong suit.
- **Nonlinearities (diode `exp`, translinear `sqrt` / divide, BJT
  differential pair `tanh`)**: exact analog physics, no
  representation error. The only limit is component noise. These
  are the operations where analog is *more* accurate than
  float32 digital.
- **Op-amp summing path (autozero AD8629)**: > 20 effective bits.
  This is the headroom that makes the 11-bit overall noise budget
  achievable; without autozero it would be ~12 bits and the rest
  of the budget would have nowhere to go.
- **ADC readout at the boundary (delta-sigma)**: 28–30 effective
  bits. Far above what we need; the boundary ADC is not the
  bottleneck.

**Bottom line:** 11 bits per MAC is what the Stage 1 hardware
delivers, and Stage 0 verified that's enough for MaxAI to train
end-to-end and for the same architecture to train at 96-block
depth. The noise budget is set by the summing-node noise floor,
not the bit-slice ceiling — which means going from Q8.8.8 to
Q8.8.8.8 (more slices) buys nothing as long as we're at this
noise level. Reducing noise (precision analog parts → 14 bits)
is the path to more bits, not adding slices.

The Stage 0 simulator validates the claim before any silicon is
committed. Gate 1 verifies MaxAI (2 blocks) trains. Gate 2
verifies MaxAI architecturally scaled to 96 blocks also trains,
which proves the architecture is GPT-scalable in principle — even
though the physical hardware we build carries only 2 blocks.

### Why this has not been done before at scale

The five tricks have all been individually demonstrated in
research. Stacking them into a trainable analog card at commercial
scale has not happened, for specific reasons repeated across the
failed attempts:

- **Mythic targeted inference-only** because their flash cells had
  limited write endurance (~10⁵ cycles). They literally could not
  train on their hardware even if they wanted to. A digipot-based
  or RRAM-based tile has write endurance orders of magnitude
  higher (10⁶–10¹²). The write-endurance wall that killed Mythic's
  training story does not apply to our substrate.
- **Lightmatter's photonic meshes** can be reconfigured, but the
  thermo-optic phase shifters have millisecond write times and
  thermal crosstalk between neighbors. Training works in
  simulation; in hardware, per-update settling time dominates.
- **IBM's research chips do train** — and produce the precision
  numbers cited above — but IBM publishes papers, not products.
  Their chips have not scaled to commercial deployment, not
  because training fails, but because the rest of the stack
  (driver software, compiler, sales channel) was not the point.

The project's leverage: **MinAI and MaxAI are small enough that
accumulated error does not kill training** even at naively
calibrated tile precision. The same tricks that fail at 96-layer
GPT-3 depth (where each layer's noise compounds) work fine at
MaxAI's 2-block depth. Smallness, again, is the superpower.

### The viability gate — both gates passed (Stage 0 is done)

**This architecture had a hard pre-hardware gate.** Stage 0 was a
C++ simulator of the analog tile with realistic noise (Q8.8.8 bit-
slicing, autozero summing, 32-bit accumulator, calibration cells)
and all nine tricks implemented. Both gates have now passed; the
architecture is cleared to fabricate. See `STAGE1_GONOGO.md` for
the formal verdict and full results.

1. **Functional gate (passed):** Full MaxAI transformer (2 blocks)
   trained end-to-end at four noise levels — clean float32, σ =
   3 × 10⁻⁵ (precision analog), σ = 3 × 10⁻⁴ (standard analog,
   PCB target), σ = 3 × 10⁻³ (hobbyist analog). All four runs
   converged to a final loss in the 0.06–0.10 range and produced
   the identical correct generation `"by the sea shore the shells
   she sells are"` from the prompt `"she sells "`.
2. **Scaling gate (passed):** Same architecture at 96-block depth
   (GPT-3 layer count) trains cleanly at the standard-analog noise
   target. Loss stays in the 0.05–0.07 band from 2 blocks through
   96 blocks — no destructive compounding with depth.

The Stage 0 surprise: at 96 blocks, the clean float32 baseline
ends at loss 0.22 while all three noisy runs land at 0.05–0.08.
The stochastic-rounded weight writes plus Gaussian matmul noise
behave like gradient noise / dropout — analog noise is a
*regularizer* at depth, not a tax. This was not designed for; it
fell out of the simulator. It implies the architecture is more
forgiving than the noise budget assumes.

Passing both gates means: **the 2-block MaxAI hardware Stage 1
will build is a direct miniature of a GPT-depth architecture that
trains at the same per-layer precision, differing only in cell
count.** The hardware demo is not a toy; it is a concrete slice
of a measured-scalable design.

The Stage 1 noise budget is now locked at σ ≤ 3 × 10⁻⁴ per-MAC
(0.03 % of full scale), which the BOM hits with comfortable
margin. See `STAGE0.md` for the simulator scope, `STAGE0_5.md`
for the bring-up board, `STAGE1.md` for the BOM, and
`STAGE1_GONOGO.md` for the gate-pass writeup.

---

## How this scales if we ever took it to silicon

The key architectural property to preserve across stages:

> **One logical tile is one physical resistor crossbar.**

At PCB scale, the crosspoint is a digipot chip. At silicon scale, the
crosspoint is a 1-transistor + 1-memristor (1T1R) cell in a
manufactured crossbar. Same topology, different device. Same
wiring pattern, different feature size. Same matmul operation, same
Ohm's-law physics, just miniaturized.

At 22 nm RRAM process density (~50 billion memristors per cm²), the
equivalent of one Stage-2 PCB worth of analog compute fits in about
0.01 mm² of silicon. A whole GPT-3-layer's worth of matmuls fits on
a die the size of a Tic Tac. That is what the "if it were scaled up
to millions on a chip" phrase means in practice.

The important thing is that the **design we iterate on at PCB scale
is the same design a foundry would produce.** We are not prototyping
a throwaway artifact; we are prototyping the real thing, in a form
where we can hold it in our hands and debug it with a scope.

---

## Constraints, ranked honestly

Since the card is a real product candidate, it is worth naming which
constraints actually bite, and in what order. Most of the intuitive
worries ("won't analog be slow?" "won't the heat be bad?") are
small. One non-obvious worry dominates everything else.

1. **Software ecosystem.** Larger than any hardware limit. GPUs
   come with PyTorch, CUDA, thirty thousand pretrained models, every
   debugger and profiler ever shipped, and every ML engineer already
   trained on them. An analog card ships with a custom driver and
   a partial ONNX compiler. The hardware is the easier half of the
   product; the software decides whether anyone adopts it. The gap
   *has* been shrinking through 2026 — LLM-assisted toolchain
   development, compiler frameworks like TVM, MLC, and Triton, and
   hardware-agnostic runtimes targeting small-model inference have
   narrowed it substantially — but it remains the single biggest
   barrier to adoption.
2. **Model size at silicon scale.** A PCIe card at PCB scale tops
   out around a million weights in analog (hundreds of tiles,
   limited by board area). A 22 nm RRAM ASIC fits about a billion
   weights per die. A multi-chip package stacks to tens of
   billions. Beyond that, multi-card scaling applies, same as GPU
   clusters — each card still doing the same low-power inference
   trick internally. There is no physics that stops the architecture
   from reaching frontier-scale model sizes; there is only tapeout
   cost and manufacturing complexity, which are the same constraints
   the GPU industry has already solved a dozen times.
3. **Precision per MAC** — measured, not modeled. Stage 0's 10,000-
   trial sweep landed at **23.3 effective bits at the quantization
   ceiling, 14.3 bits at precision-analog noise, 11.0 bits at the
   standard-analog PCB target, and 7.7 bits at hobbyist-analog
   noise** (`STAGE0_5.md`'s Experiment 2 will confirm those numbers
   on actual silicon). The committed Stage 1 target is the 11-bit
   standard-analog regime, which is well above the ~8-bit training
   floor IBM published for analog training and well above the int8
   inference quality bar modern LLMs already serve at.
4. **Speed per matmul** — *not* a limit. Analog MACs settle in tens
   of nanoseconds (op-amp slew-rate limited). Digital GPUs do the
   equivalent in comparable wall time but burn 10–100× the energy.
   At the tile level analog wins on latency *and* energy; at the
   card level throughput tracks tile count, which scales with area.
5. **Heat** — *not* a limit. Analog dissipates 10–100× less power
   per MAC than digital. The card runs warm but not hot; a desktop
   case fan is sufficient. A silicon version runs cool enough that
   thermal design is dominated by the digital glue logic, not the
   analog tiles.
6. **Errors: thermal drift, component mismatch, power-supply noise.**
   All real, all well-understood, all mitigated with known techniques
   — periodic calibration loops, per-tile correction tables,
   ground-plane discipline, local regulation per stage. Every
   shipping analog AI chip in 2026 solved these. Cost: a few percent
   overhead, not a design generation.

The short version: **if the card fails commercially, it fails on
ecosystem adoption, not on physics.** The physics is fine. The
physics is, in fact, better than digital for this workload. The
open question is whether the software stack reaches "drop-in with
PyTorch" before market interest moves on.

---

## What the product actually looks like

Pulling the prior sections together into the product-shape that
falls out of them.

**Physical.** Half-length PCIe gen 3 ×4 card. Eight to sixteen
analog tile clusters on the board, each cluster holding one
transformer-block-worth of tiles + LN stages + residual/ReLU
patches. One FPGA as the digital host. Onboard DDR for weight
staging; onboard flash for the bootloader and a default weight set.

**API.** Presents to the host OS as a specialized inference
accelerator. Driver exposes a minimal interface: `load_weights`,
`run_inference`, `read_tokens`. PyTorch-compatible middleware
(ONNX-subset compiler) plus a small C++ inference library for
direct integration.

**Workload fit.** Edge-scale language models (up to ~100M parameters
at PCB-scale Stage 4a; up to ~1B at silicon-scale Stage 4b), vision
models, voice models, real-time audio processing, embedded
assistants, anywhere a small-to-medium inference workload runs at
high duty cycle and power matters.

**Not a fit for.** Frontier-scale LLM inference on a single card
(use a cluster, same as GPUs). Training (the card is an inference
device; train on GPUs, deploy here). Highly-dynamic workloads with
frequent weight changes (weights are effectively baked at boot).
Scientific ML that needs FP16+ numerics (the precision ceiling
bites there).

**Pitch.** *Inference for small-to-medium models at 1/50 the power
of a consumer GPU, in a form factor that drops into any desktop.*
Every number in that sentence is defensible on the physics. What
makes it a product is whether the driver and compiler tell a
compelling enough story for a developer to choose it over spinning
up a cloud GPU. That is the bet.

---

## What could go wrong (honest list, post-Stage-0)

1. **Noise accumulation at depth.** This was the worry going in.
   Stage 0's Gate 2 disproved it — at 96 blocks the noisy runs
   actually outperform the clean float32 baseline because analog
   noise regularizes deep training. So this is no longer the top
   risk. **The residual risk is hardware that doesn't match the
   sim's Gaussian noise model**: real digipots, real op-amps, real
   PCB layout might have a non-Gaussian or correlated noise term
   the simulator didn't capture. Stage 0.5 measures this on a $25
   board before Stage 1 commits.
2. **Digipot write speed during training.** Volatile MCP4251s rewrite
   at SPI rates (5–10 MHz, ~1 µs per wiper). Fine for SGD steps at
   the cadence Stage 1 needs (training step on the order of
   milliseconds). NV digipots would be slower per write *and* have
   limited write endurance — that's why we're committed to volatile
   for the training board.
3. **Op-amp saturation at intermediate stages.** If any matmul or
   summing stage overshoots the supply rails, the chain clips and
   the tail of the pipeline goes nonsensical. Weight initialization
   and per-layer scaling discipline matter. Software trick: train
   with output-magnitude-aware regularization on the weights (the
   simulator already does this implicitly).
4. **Nonlinearity transfer-function drift.** A diode ReLU's transfer
   curve shifts ~2 mV/°C with temperature. Training that bakes in
   a fixed transfer function will misbehave on a board running 10°C
   off the bench condition. Per-tile thermistor + calibration loop
   in Stage 2 onwards; for Stage 1, hold the bench at room temp.
5. **Yield and component matching.** A hand-built tile has
   0.01–0.1 % component-tolerance-level dispersion. Calibration
   reference cells absorb most of it. The MCP4251 dual-channel
   choice (15 ppm/°C ratiometric on-die tempco) absorbs more.
   Per-tile calibration after assembly remains required.
6. **PCB layout discipline.** Mixed-signal layout is its own skill
   and the first board will have layout mistakes. **This is what
   Stage 0.5 exists to surface on a $25 board** rather than a $500
   one. See `STAGE0_5.md`.

None of these is a reason not to build. All of them are reasons the
Stage 0.5 and Stage 1 experiments are designed to measure them
before committing to Stage 2.

---

## What we do not do, by design

- **Dynamic topology.** The tile size is fixed. Larger models tile
  up; they do not reconfigure the cells.
- **Full float-32 numerics on every operation.** Stage 0 measured
  11.0 effective bits at the standard-analog noise target — well
  above the 8-bit training floor and the int8 inference quality
  bar, but not float32. Models retrain with injected tile noise to
  match. We do not pretend to reach fp32 across the whole pipeline.
- **Round-trip the signal to digital mid-pipeline.** This is the
  rule from "Architectural rule #1" above, repeated here as a
  non-design: no digital softmax, no digital LayerNorm, no digital
  residual add, no digital nonlinearity in the inference loop.
  One DAC at start, one ADC at end, no exceptions for "convenience."

### What we explicitly do do — corrections to earlier drafts

- **Training on the analog hardware IS in scope.** Stage 0's Gate 1
  proved it works at standard-analog noise; the analog-architecture
  trains MaxAI to identical correct output across four orders of
  magnitude of per-MAC noise. The prior framing of "train digital,
  deploy analog" was the Mythic framing, and it is the failure
  mode this project exists to avoid.
- **GPT-depth scaling IS in scope, in principle.** Stage 0's Gate 2
  trained the same architecture at 96 blocks (GPT-3 layer count) at
  standard-analog noise without accuracy degradation — and the
  noise actually helps at depth via regularization. The earlier
  hedge that "accumulated analog noise across hundreds of layers
  is fatal" was wrong; the simulator empirically refuted it. The
  physical hardware we build is still 2 blocks for cost reasons,
  but the architecture is provably scalable.
- **Temperature compensation IS on the board** from Stage 2
  onward. A thermistor per tile plus a host-side calibration loop
  retrains the per-cell correction tables every few seconds during
  inference, faster during training.

---

## The one-sentence architecture summary

**Boot the host; load the weights into the bit-sliced analog
tile array; feed prompt tokens in one side of the board as
voltages and read generated token IDs out the other through a
small ADC; everything in between — matmul, residual, ReLU,
LayerNorm, softmax, and sampling — happens in analog, where the
operations are exact physics bounded only by the noise budget
we have deliberately engineered down.**
