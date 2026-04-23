# Stage 4 — productize or miniaturize

*The answer to "so what comes next?" after someone sees the Stage-3
demo. Two branches. Neither forecloses the other. Most serious
projects do 4a first to generate revenue and 4b later once the
architecture is validated enough to justify tapeout budget.*

---

## Branch 4a — PCIe accelerator card

A drop-in expansion card for a consumer desktop. Same analog tiles
as Stage 3, same digital glue logic, but consolidated onto a single
half-length PCIe card with the Raspberry Pi replaced by an FPGA.

### What 4a buys

- **A product.** The demo becomes a thing you ship. Anyone with a
  PC can plug it in and run low-power inference on a small language
  model.
- **A market.** The "edge AI" space — assistive devices, offline
  translation, embedded automotive, private local-only models —
  pays real money for watts-not-kilowatts inference on small models.
- **A credible path to a business.** An analog card that runs a 6000-
  parameter model at 2 W has an obvious scaling story: "tile count
  is how we get to a billion-parameter model; come talk to us about
  the roadmap."

### What changes from Stage 3

- **The Pi is gone.** An on-card FPGA (Lattice ECP5, AMD Spartan-7,
  or similar — $20–$100 part) becomes the host interface. It speaks
  PCIe to the motherboard, controls weight loading, sequences tile
  operations, does softmax and sampling in digital glue.
- **A single PCB, not a stack.** Tiles are reflowed onto a single
  card instead of stacked daughterboards. Denser, cheaper, shippable.
- **A real driver.** A Linux kernel module that presents the card
  as a character device or a custom accelerator. Host software
  (PyTorch, ONNX Runtime, or a custom C++ inference library) calls
  the card's matmul primitives.
- **Real firmware.** The FPGA runs a small soft-CPU (picoRV32 or
  similar) that handles the control plane; dedicated FPGA logic
  drives the tile SPI buses and the ADC readback.

### Engineering cadence

| Phase | Duration | Deliverable |
|---|---|---|
| Schematic + FPGA design | 1–2 months | Simulated card |
| First PCB fab | 1 month | Populated rev-A card |
| Bring-up + driver | 2–3 months | Card runs MaxAI in a host OS |
| Reliability + documentation | 1–2 months | Shippable rev-B |

Call it **six to nine months** to a sellable product from the end of
Stage 3.

### Budget

- NRE (design + tooling): **$20k–$60k** including FPGA boards,
  prototype PCB fab, test equipment upgrades.
- Per-unit BOM at low volumes: **~$200–$400**.
- Per-unit price point: you decide. Comparable SKUs (Coral USB
  accelerator, Hailo-8) sit at $100–$800.

### Market positioning

Not "we compete with NVIDIA H100." That fight is lost before it
starts. The pitch is: *"We do small-model inference at 1/50th the
energy and 1/30th the price of even a consumer GPU, for the growing
edge-AI market."* Local-only language models for privacy, voice
interfaces for low-power devices, translation for offline
environments, specialized assistants where a 1–5 billion parameter
model is enough. That market is real and growing in 2026.

### The honest limit of 4a

At PCB scale, tile density is hard-capped by board area. A full
PCIe card holds maybe 500–1,000 tiles — call it **250,000 weights
in analog**, versus **billions** on a real GPU. 4a wins at watts-per-
token; it does not win at tokens-per-second at big-model sizes. If
you want to compete on model size, you need 4b.

---

## Branch 4b — silicon tapeout with RRAM crosspoints

The same tile, but the 256 digipots become 256 memristor
crosspoints in a 1T1R (one-transistor + one-memristor) layout on a
real silicon die. This is the answer to "could this ever scale to
GPT?"

### What 4b buys

- **A thousand times the density, with GPT-depth training.** At
  Q8.8.8 in 22 nm RRAM, each tile is ~1.5 mm². An H100-sized die
  holds hundreds of millions to low billions of weights — and
  because the architecture (per the Stage 0 Gate 2 simulator)
  trains at 96-block depth natively, the silicon version is not
  just a scaled-up inference chip but a **trainable analog AI
  processor at frontier-model depths.** Every prior commercial
  analog AI attempt was inference-only; this one carries training
  through to silicon because the architecture was designed for it
  from Stage 0.
- **The definitive architectural answer.** A working silicon
  version of the architecture is what converts "cool side project"
  into "new hardware platform." Every question that began "but
  could this really scale?" gets an on-chip measurement instead of
  a spreadsheet.
- **IP.** A working silicon tapeout is publishable, patentable, and
  investable in ways that a PCB demo is not. Serious money arrives
  here, not earlier.

### What changes from 4a

- **No more digipots.** Each 1T1R crosspoint is a single transistor
  + a single metal-oxide memristor cell. Programmed via row/column
  addressing, read via the same current-summation trick.
- **No more discrete op-amps.** Integrated summing amplifiers at
  the column outputs, sized for the chip's voltage rails.
- **On-chip ADCs.** Instead of ADS1115 parts, each column bank has a
  column-parallel ADC block.
- **No more SPI to digipots.** Weight loading becomes a memory-
  mapped programming sequence. Tens of milliseconds total to
  program a million weights.
- **Everything smaller, faster, less power.** Per-MAC energy
  drops from picojoules (on PCB) to femtojoules (on-chip). Tile
  throughput rises from MHz to GHz.

### Engineering cadence

| Phase | Duration | Deliverable |
|---|---|---|
| Choose fab partner + PDK | 1–3 months | NDA, PDK access, process variant |
| RTL + mixed-signal design | 6–9 months | Tapeout-ready GDS |
| Tapeout + fab | 4–8 months | First wafer back |
| Bring-up + characterization | 3–6 months | Working chip on an eval board |
| Second iteration | 6–12 months | Production-worthy silicon |

Call it **two to three years** from the end of Stage 3 to a
production silicon product. Plus another year for commercial
packaging if the goal is a shippable chip.

### Budget

- **MPW (shared wafer) tapeout**: $30k–$150k depending on process
  and die area. Practical for a 1-million-weight proof-of-concept.
- **Dedicated tapeout**: $500k–$5M at modern RRAM-capable nodes
  (TSMC 22 nm ULP, SMIC equivalents, or specialty foundries like
  Tower Semiconductor). Required for a production-scale chip.
- **Design tools and EDA licenses**: $50k–$500k/year, though
  open-source flows (OpenROAD, Magic, KLayout) are increasingly
  viable and can cut this substantially.
- **Team**: 3–10 engineers, depending on scope and ambition.

Not a solo project. Stage 4b is where the side quest becomes a
company, or joins one, or licenses to one.

### Fab partners with RRAM in 2026

Shortlist, roughly cheapest-to-priciest:

- **CUMEC** (China) and **SMIC** RRAM-capable MPW programs —
  cheapest RRAM access, 6–12 month turnaround.
- **TSMC N22 eMRAM / RRAM** via shuttles — more expensive, more
  mature, export-controlled on some parts.
- **GlobalFoundries 22FDX ReRAM** — mature, widely characterized
  cells, available on their MPW programs.
- **IMEC / LETI research lines** — for academic collaboration; very
  flexible but slow.
- **AIM Photonics** — if you want to add photonic I/O later, they
  integrate.

### Honest prerequisites

- **Stage 3 working.** You will not raise money or attract a fab
  partner without a functioning analog proof on the bench.
- **A team.** At least one senior analog/mixed-signal designer and
  one digital architect. Preferably with RRAM experience (~50
  people in the world as of 2026, most at large companies; the
  academic pool is small but growing).
- **A paper.** Publishing the Stage-3 results in IEEE Solid-State
  Circuits, ISSCC, or a machine-learning hardware venue (ISCA, MLSys,
  HPCA) dramatically opens doors to fab partnerships and funding.

---

## Which branch first?

Almost always **4a before 4b**. Here's the shape of it:

- 4a produces **revenue**, proves **commercial viability**, and
  grows the **team and brand** without the time and capital risk of
  a full tapeout.
- Revenue from 4a (even modestly priced at $500/card, $5M of gross
  revenue buys a decent team for a year, two tapeout shots, and EDA
  licenses) **funds** 4b.
- 4b produces a **moat** — a real silicon platform competitors
  cannot copy in a weekend.

The order that routinely works: side-project → Stage 3 demo →
Stage 4a card → small team + small revenue → Stage 4b silicon →
bigger team + real moat → scale.

The order that routinely fails: side-project → Stage 3 demo → jump
straight to Stage 4b tapeout → burn two years and most of the
capital before the first real-world workload touches the chip.

---

## What if you skip 4 entirely?

Perfectly legitimate. Stage 3 is a complete artifact. It demonstrates
the entire argument: that modern AI is a digital accident and that
the purpose-built analog alternative is within a weekend engineer's
reach at small model sizes. Published as a blog post, a talk, a
paper, or a book chapter, it influences the field without asking
anyone for investment.

Not every side quest needs a company at the end. The more important
thing is whether the argument travels far enough to change minds.
A good Stage-3 demo on a workbench, filmed and posted, might be
enough to tilt the field toward analog hardware by itself. The
people reading it will build the Stage-4 products.
