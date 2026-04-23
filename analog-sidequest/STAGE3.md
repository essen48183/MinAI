# Stage 3 — MaxAI in analog (the raised-eyebrow demo)

*The first time this project produces output a stranger cares about.
Same architecture as Stages 1–2, scaled up enough to hold MaxAI's
trained weights, driven by the same Raspberry Pi. Plug it in, type a
prompt, watch English come out. A few watts of power, a few seconds
to boot, a real physical artifact that generates text.*

---

## What Stage 3 proves

- That **generative language modeling** — not just matmul — runs on
  analog hardware. The Stage-3 box accepts `"four score "` and
  produces `"and seven years ago"`.
- That the architectural choices locked in at Stages 1–2 (volatile
  weights loaded at boot, tile-based matmul, digital softmax on the
  host) compose into a *product-shaped* demonstration, not an
  academic one.
- That the energy budget of "real" generative AI is not inherent to
  the task — it is inherent to the current digital substrate. A
  laptop running MaxAI burns 30 W. The Stage-3 analog stack next to
  it burns ~2 W and produces the same tokens.

This is the demo you bring to someone who does not care about
transformers, Ohm's law, or any of the other good ideas in this
folder. You plug it in. It prints English. That is the pitch.

---

## What sits on the table at Stage 3

A roughly business-card-sized analog board (or small stack of
boards) connected to a Raspberry Pi Zero 2W. Screen or terminal shows
a prompt cursor.

Physical power draw, measured: **1–3 watts at the wall, including
the Pi.** Compare: a laptop idling is 10 W; a laptop generating
text with a software model is 30–60 W; a cloud GPU running a real
LLM is hundreds of watts. You are demonstrating the energy argument
in person.

Boot time: **a few seconds.** The Pi boots Linux, reads MaxAI's
weight file, walks the weights into the tile array over SPI. Call it
the same boot feel as a small embedded system starting a game.

Latency per generated token: **milliseconds.** The analog matmul is
nanoseconds; the Pi's softmax and RNG loop dominate. Streaming
characters to the screen looks like a normal terminal.

---

## What MaxAI brings that Stage 2 did not need

### A larger vocabulary

MaxAI's chars vocabulary is 28 symbols (26 letters + space + EOS),
expanded to 44 after 16 BPE merges on the Gettysburg corpus. Tile
shapes change accordingly:

- Token embedding: 44 × 16 = 704 weights  (3 tiles)
- Output projection: 16 × 44 = 704 weights  (3 tiles)

Not a huge increase. The tile architecture accommodates it
natively; it just takes a few more boards.

### A BPE tokenizer

Runs on the Pi, same as the book's `bpe_encode_text()`. Input
prompt → token sequence → DAC voltages for the embedding stage.
Zero hardware change; a few hundred lines of C++ on the host.

### A KV cache

Also on the Pi. Cached K and V vectors stay in Pi DRAM between
token generations; only the new token's K and V are computed on the
analog tiles per step. This is exactly the Chapter-22 incremental
forward pass, but split across analog hardware and Pi software.

The **K/V cache is the only reason per-token latency stays in
milliseconds rather than multiplying by the sequence length.** If
Stage 3 skipped the KV cache and re-ran the full prefix on every
generated token, latency would rise linearly with prompt length —
tolerable for a demo, but wasteful.

### Sampling

Temperature / top-k / top-p — all on the Pi, same code as the book
describes. Pi reads the analog score vector through the ADC, applies
the sampler, feeds the chosen token back as the next input vector.

---

## Tile count, for the full MaxAI

MaxAI with `--blocks=2 --layernorm=1 --bpe=16`:

| Component | Approximate weights | Tiles at 16×16 |
|---|---|---|
| Token embedding (44 × 16) | 704 | 3 |
| Position embedding (32 × 16) | 512 | 2 |
| Block 0: Q, K, V, Wout | ~1,024 | 4 |
| Block 0: FFN W1, W2 | ~1,024 | 4 |
| Block 1: Q, K, V, Wout | ~1,024 | 4 |
| Block 1: FFN W1, W2 | ~1,024 | 4 |
| Output projection (16 × 44) | 704 | 3 |
| LayerNorm params | negligible | — |
| **Total** | ~6,000 | **~24 tiles** |

Twenty-four tiles. Each tile is the Stage-1 board. On a 100 mm ×
100 mm backplane panel, four tiles fit comfortably. Six panels or
one larger carrier holds the full set. The entire compute engine is
roughly the footprint of a paperback book.

---

## The demo script

```
[ Hardware set up on a desk. ]
[ Pi boots. Terminal shows: ]

   MaxAI analog accelerator v0.1
   Tiles detected: 24/24 healthy
   Loading MaxAI weights (corpus=gettysburg, bpe=16)...
   Done in 418 ms.
   Power: 1.8 W at wall (Pi + analog board).

   prompt> four score

[ user hits enter ]

   four score and seven years ago our fathers brought forth on
   this continent a new nation

   49 tokens generated in 382 ms.
   Energy: 0.42 J total (~9 mJ per token).
   Analog MACs executed: ~140,000. Clock cycles used: 0.

   prompt>
```

The output may be memorized (Chapter 20: the corpus is short, the
model overfits) — but the **generation path is analog**, and every
predicted character travels through 24 tiles of Ohm's-law matmul
before the Pi picks it. Printing "memorized" English from a physical
circuit you built is still the most compelling thing in the room.

---

## Expected measurements to show

Three numbers you put on the side of the enclosure. These are the
numbers that make the engineering-minded person in the room stop
and ask follow-ups.

1. **Total wall power.** A laptop running the same model in C++
   software vs. the analog stack. Shows ~10–20× improvement.
2. **Energy per generated token.** Millijoules for the analog box;
   tens of millijoules for the laptop. Measurable with a USB power
   meter.
3. **Token output bit-identical to software** (seed-matched
   sampling). Proves the analog hardware is not cheating.

For conference or podcast demos, a live oscilloscope showing a
summing node settling in tens of nanoseconds while a character
arrives on the terminal is, pedagogically, the best single frame.

---

## Tile reuse (the economic argument to show next to the demo)

The Stage-3 rig contains ~24 16×16 tiles. All twenty-four are
**identical** Stage-1 boards. Each is ~$80 in parts; the total
hardware cost is around **$2,000**. The Pi, backplane, and
enclosure add another few hundred. A finished Stage-3 demo costs
**roughly $2,500 in parts.**

This matters for the "so what?" conversation. A GPU that can
generate the same tokens is a $30,000 H100. Even a cheap
consumer GPU is several hundred dollars *and* burns 200 W. The
analog box is at worst 1/10 the price and 1/20 the power for a
task of this size. That ratio gets better at scale, not worse.

---

## Honest limits at Stage 3

- **MaxAI is a toy model.** The generated text is memorized
  Gettysburg, not original thought. This is a hardware demo, not an
  AI demo. Anyone who thinks it is a small GPT will be disappointed.
- **Weights are digitally trained.** Training is still done in
  software; the analog hardware loads the trained weights. Same as
  every production analog AI chip.
- **Noise per tile is real.** Short-term drift is still managed
  with per-tile calibration and periodic reference reads. What
  changed between pre-Stage-0 and now: we no longer need to
  deliberately inject training-time noise as a generalization aid.
  Gate 2 showed that the analog stack's own stochastic-rounded
  writes plus Gaussian matmul noise already act as regularization
  at depth (at 96 blocks, clean loss 0.22 vs noisy loss 0.05–0.08).
  The architecture is self-regularizing; what's left for Stage 3 is
  keeping drift bounded, not adding noise.
- **Scaling to GPT is still a cost question, not a physics
  question.** The Stage-3 box is about 2,000× too few tiles to run
  GPT-3. Fitting 2,000× more tiles on a card is what Stage 4b
  (silicon) is for.

---

## Budget and cadence

- **Parts:** roughly **$2,000–$3,000** for 24 tiles + backplane +
  Pi + enclosure + cables + USB power meter.
- **Time:** two to four months once Stage 2 is complete, the bulk
  of which is careful backplane design and per-tile calibration
  scripts.
- **Iteration:** individual tiles remain day-level swappable;
  backplane changes are still fab-turn-bound; the calibration
  pipeline (Pi software) is rewrite-anywhere.

---

## Why Stage 3 is the point

MinAI in software is small enough to fit on a PDP-11. MinAI in
analog hardware (Stage 2) is a proof that our Ohm's-law architecture
works. **MaxAI in analog hardware is the smallest possible thing
that actually lets a human in a room feel how radically different
this computing substrate is.**

Before Stage 3, the project is a novelty. After Stage 3, the
conversation changes. "Can you scale it?" is suddenly the right
question to answer. That answer is Stage 4.
