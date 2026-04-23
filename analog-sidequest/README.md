# Analog MinAI — an analog compute side quest

*A parallel project that runs MinAI's math on discrete analog hardware
instead of digital CPUs/GPUs. PCB-fabbable, home-iterable in hours,
and the only reason it is tractable is that MinAI itself is small.*

---

## Why this project exists

Modern AI is a 2017 neural-network architecture running on 1990s-era
graphics hardware, burning megawatts of electricity to simulate physics
that any properly-built analog circuit does for free. Every transistor
in a GPU spends clock cycles computing `a × b + c` with floating-point
arithmetic, when Ohm's law and Kirchhoff's current law will produce the
same answer at the speed of electricity in a resistor grid.

This side project builds a physical demonstration of that argument at a
scale a single engineer can actually fabricate. It uses **MinAI's 2,240
trained weights** as its reference model, because that is a size a
handful of PCBs can hold. A GPT-3-scale analog build would need millions
of tiles and a foundry. A MinAI-scale analog build fits on a desktop.

**The book's small size is the point.** The project does not exist
despite MinAI being tiny; it exists *because* MinAI is tiny. That
smallness is a superpower here — it makes the entire "is this possible
outside a foundry?" question answerable.

---

## What a final working result looks like

A PCIe card that drops into any motherboard. On the card: a host-side
FPGA, a small onboard flash holding trained weights, and an array of
analog matmul tiles. Running MinAI's forward pass on that card should
produce the same tokens the software version produces, using
**picojoules per multiply-accumulate** (versus nanojoules for a digital
GPU) and with **no clocked computation** — the matmul settles at
analog speed.

The card is a proof of concept, not a product. What it proves is:

1. Analog compute really does match the math of a transformer.
2. The energy and speed advantages predicted by physics are measurable
   in hardware, not just simulation.
3. The build is technically within reach of a small team rather than
   an industrial fab.
4. The architecture scales — the same tile miniaturized to silicon
   with RRAM/memristor crosspoints is the known path to million-weight
   single-chip accelerators.

---

## The critical difference from prior attempts

Every commercial analog-AI attempt from 2012–2024 took the same
shortcut: *train the model on NVIDIA, deploy on our analog chip*.
That shortcut has killed every one of them commercially. The
customer math is unavoidable — if the customer already needs a
GPU for training, they just use it for inference too rather than
bring in a second vendor.

**This project treats training on the analog hardware as a
first-class, non-negotiable requirement**, addressed as Stage 0
(in simulation) and Experiment 4 of Stage 1 (on real hardware).

The core framing — different from every prior analog-AI project —
is **noise is the enemy, not representation.** Digital precision
is bounded by the number format (float32 can never exceed its
23-bit mantissa). Analog precision is bounded only by the noise
of the measurement channel, and every noise source (thermal,
flicker, shot, drift, crosstalk, EMI) has a known engineering
countermeasure. Aggressively applying those countermeasures —
precision thin-film components, autozero op-amps, delta-sigma
ADCs, 8-layer stripline PCBs, calibration reference cells — pushes
analog effective precision to **24–28 bits per operation**, which
is *better* than float32 digital on the operations where analog is
naturally exact (Kirchhoff summation, Ohm's-law multiplication,
exp/log/sqrt/tanh through diode and transistor I-V curves).

The architecture combines **three hardware precision amplifiers**
— Q8.8.8 bit-slicing (three analog cells per weight for 24-bit
nominal precision), autozero summing op-amps (20+ bit op-amp
paths), a 32-bit digital gradient accumulator, plus continuous
calibration via dedicated reference cells — with **five software
training tricks** from the 1970s–1990s analog-computing literature
— mixed-precision updates, stochastic rounding, temporal
oversampling, delta encoding, and analog physics as the native
lookup table. Together they clear the precision bar where IBM
research has demonstrated analog training of **96-layer
transformers**, not just shallow networks.

Softmax, sampling, and all nonlinearities stay analog too — the
same physics (diode `exp`, translinear divide, winner-take-all or
ramp-compare samplers) that makes matmul analog-native also makes
these operations analog-native. The host's digital footprint
reduces to: weight-load at boot, prompt-token DAC in, output-token
ID ADC out, and loop control. **One ADC-DAC round-trip per
generated token, not per layer.**

The project refuses to fabricate any hardware until a pure-
software Stage 0 simulator passes **two** gates:

- **Gate 1 (functional):** MaxAI (2 blocks) trains to recognizable
  output at simulated hardware precision.
- **Gate 2 (scaling):** MaxAI scaled to 96 blocks, same per-layer
  precision, also trains to recognizable output — proving the
  architecture scales to GPT depth in principle.

If Gate 1 fails, the project stops. If Gate 2 fails, the
GPT-depth scaling claim is retracted and the project is rescoped
as small-model-only (still valid, but a narrower pitch). Only
passing both gates clears Stage 1 to fabricate hardware.

See `ARCHITECTURE.md` for the five-trick stack and `STAGE0.md` for
the viability gate.

---

## The five stages

Each stage produces a complete, measurable artifact. Each stage's
output is the foundation for the next. Nobody has to commit to
Stage 4 before finishing Stage 1. The project has a clear inflection
at Stage 3: before it, you are proving the hardware *works*; after
it, you are proving it *matters*.

### Stage 0 — the simulator viability gate *(software only, kills the project cheaply if training doesn't work)*

- A C++ simulator of the tile architecture with Q8.8.8 bit-slicing,
  autozero op-amp paths, a 32-bit gradient accumulator,
  calibration reference cells, realistic device noise, write
  quantization, and all nine training tricks implemented.
- Runs the book's MaxAI training loop through the simulator at
  both 2-block depth (Gate 1) and 96-block depth (Gate 2).
- **Pass both gates:** architecture trains at MaxAI scale AND the
  same architecture scales to GPT-depth in principle. Proceed
  to Stage 1 with full scaling claim intact.
- **Pass Gate 1 only:** architecture trains at small scale but
  compound noise kills deeper networks. Project may still
  proceed, but the scaling claim gets retracted and the pitch
  is rescoped to "edge / small-model only."
- **Fail Gate 1:** the architecture does not train at all at this
  precision. Project stops. Zero hardware cost.
- Wall-time estimate: ~two focused weeks. Budget: $0.
- **What Stage 0 buys:** insurance against repeating Mythic's mistake
  of fabricating hardware before knowing if the architecture
  trains, *and* against the Lightmatter mistake of shipping
  hardware whose scaling ceiling wasn't known until silicon was
  in hand.

### Stage 1 — a single Q8.8.8 analog matmul tile *(validation, including one training step on real hardware)*

- Raspberry Pi Zero 2W as host.
- One small PCB holding a 16×16 analog MAC tile (256 weights).
- Performs MinAI's `Q = X @ Wq` projection entirely in analog.
- Host orchestrates: loads weights from microSD, drives inputs, reads
  outputs, compares against a NumPy/C++ reference from the book.
- Budget: **under $100 in parts**, plus your PCB fab.
- Iteration cycle: hours. Swap digipots, rework the nonlinearity, try
  different op-amps, measure on a scope.
- **What Stage 1 buys:** hardware-proven analog matmul, measurable
  against bit-exact software output. Goes no further than "the
  physics is real."

### Stage 2 — a full MinAI block in analog *(architecture proof)*

- Same Pi host, but now ~10 tiles wired together (sharing a
  backplane PCB).
- Holds MinAI's complete weight set (Q, K, V, Wout, W1, W2 of one
  block = ~2,000 weights across roughly 10 tiles).
- Pi does softmax, sampling, EOS, and tile-to-tile coordination.
  Everything else is analog.
- **What Stage 2 buys:** proof that tiling works and that a whole
  transformer block fits. The output still isn't a wow demo — it's
  MinAI still reversing digits — but the multi-tile architecture
  is proven. Everything beyond this is scaling.
- Budget: **a few hundred dollars** in parts.

### Stage 3 — MaxAI in analog *(the raised-eyebrow demo)*

- **The first demo a stranger will care about.** A Raspberry Pi
  driving a larger stack of analog tiles that holds MaxAI's weights
  (~5,000+ parameters, BPE-tokenized character vocabulary).
- User types a prompt. Pi runs the BPE tokenizer, loads token
  embeddings, streams input vectors into the analog tile array.
  Tiles compute matmuls; Pi manages the KV cache and softmax; the
  combined system generates text, token by token, at interactive
  speeds.
- Expected demo: prompt `"four score "` → analog card generates
  `"and seven years ago"`. Shown on a screen next to a laptop
  running the same model in software, producing identical tokens.
- Power budget: roughly a **watt or two** for the entire analog
  stack. The laptop next to it is pulling thirty.
- Budget: **low four figures** in parts, plus a custom backplane PCB.
- **What Stage 3 buys:** a physical artifact that demonstrates
  analog generative AI at a scale anyone can understand. This is
  the demo you show at Hackaday, Supercon, a VC meeting, a research
  group. It answers the question "could this ever matter?"

### Stage 4 — productize *or* miniaturize *(scaling path, pick one or both)*

Two branches, not necessarily exclusive. Stage 4 is the
answer to the "so what comes next?" question after the Stage-3 demo.

**Stage 4a — productize as a PCIe card.**
- Drop the Raspberry Pi. An FPGA on-card becomes the host interface.
- Card exposes a driver interface to the host PC: load weights,
  run inference, return tokens. Looks to the host like any other
  accelerator (TPU, NPU, etc.).
- Physical form: standard PCIe gen 3 ×4 slot card.
- Budget: five figures, six-month design cycle.
- **What Stage 4a buys:** a tangible product — an analog AI card
  someone can buy and plug into their desktop. Market case: edge
  inference for small language models at one-hundredth the power of
  a GPU.

**Stage 4b — miniaturize to silicon.**
- The same tile architecture, shrunk to a CMOS + RRAM process.
- Each digital potentiometer becomes a 1-transistor + 1-memristor
  crosspoint. The op-amps become integrated summing amplifiers.
  Everything else stays.
- A 256×256 tile in 22 nm fits in roughly 0.5 mm². A million-weight
  chip fits on a die smaller than a fingernail.
- Budget: $30k–$100k for a shared MPW run at a foundry that
  supports RRAM (TSMC, SMIC, IMEC).
- **What Stage 4b buys:** the definitive answer to "could this
  scale to GPT?" A single-chip version of MaxAI at a million or
  tens-of-millions of weights, at milliwatts, validates the whole
  architectural thesis. From there, larger tapeouts are engineering
  scaling, not research.

Stages 4a and 4b are not in competition. A small team that reaches
Stage 3 might do 4a first (to generate revenue) and 4b later (once
the architecture is validated enough to spend tapeout money on).

---

## The arc, in one sentence per stage

- **Stage 0:** Prove in software that the architecture can train at all, before any hardware is built.
- **Stage 1:** Prove that analog matmul AND one training step work on real hardware.
- **Stage 2:** Prove the tile architecture scales to a full transformer block, trainable.
- **Stage 3:** Prove the architecture trains and generates real text — the demo that raises eyebrows, with no GPU in the loop after initialization.
- **Stage 4:** Prove the concept scales — either to a consumer-product card or to a silicon die, both carrying training support, not inference-only.

---

## Folder contents

- `README.md` — this file.
- `ARCHITECTURE.md` — the full system architecture, piece by piece,
  with the "normal computer → analog card" mapping spelled out,
  the on-board LayerNorm stage, the stack pattern, the five
  training tricks, and the ranked constraints.
- `STAGE0.md` — **the simulator viability gate. Read first.**
  Software-only. Determines whether hardware is worth building.
- `STAGE1.md` — single-tile validation, now including the training
  experiment. Schematic, parts list, four measurement experiments.
- `STAGE2.md` — full MinAI block across multiple tiles.
- `STAGE3.md` — the MaxAI demo. Text generation from a prompt,
  on an analog-hardware stack that sits next to your laptop.
- `STAGE4.md` — the productize-or-miniaturize fork. PCIe card vs.
  silicon tapeout, both paths laid out, both carrying training
  support (not inference-only).

---

## How this relates to the book

Everything in the *Arithmetic of Intelligence* describes the
mathematics. This side quest asks whether the same mathematics can be
carried out by something other than a stack of floating-point
multipliers in a cloud-hosted GPU. The answer, apparently, is yes —
and small enough to fit on a desk.

If MinAI were larger, this project would be out of reach. It is not.
