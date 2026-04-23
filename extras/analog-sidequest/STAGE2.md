# Stage 2 — full MinAI block in analog

*Stage 1 proved one tile works. Stage 2 proves ten tiles work
together. Not a crowd-pleaser yet — MinAI still reverses digits —
but it closes out every architectural question before MaxAI lands
on top of the same hardware in Stage 3.*

---

## What Stage 2 proves

- Tiling works. You can route analog signals between PCBs without
  losing the matmul's accuracy.
- Digital orchestration (the Pi) can sequence inputs, weights, and
  readbacks across multiple tiles in the correct order for one
  forward pass of a transformer block.
- Negative weights are representable (differential-pair tiles).
- The combined system produces MinAI's trained output tokens, bit-
  for-bit, using the same weights the software version uses.

Stage 2 is the last stage where you can fully verify against a
known-correct reference. From Stage 3 onward, MaxAI's sampling is
stochastic and bit-for-bit verification gets softer (identical only
with identical RNG seed).

---

## Tile count

MinAI's default configuration has these weight matrices per block:

| Matrix | Shape | Weight count |
|---|---|---|
| `Wq` | 16 × 16 | 256 |
| `Wk` | 16 × 16 | 256 |
| `Wv` | 16 × 16 | 256 |
| `W1` (FFN in) | 16 × 32 | 512 |
| `W2` (FFN out) | 32 × 16 | 512 |
| `Wout` | 16 × 10 | 160 |
| `token_emb` | 10 × 16 | 160 |
| `pos_emb` | 8 × 16 | 128 |

Total: roughly **2,240 weights**.

With 16×16 tiles (256 weights each), Stage 2 needs **approximately
10 tiles** — plus a little flexibility for split-tile shapes like
the 16×32 FFN matrices (which become two stacked tiles each).

A common layout:

```
  [  Tile 1: Wq  ]   [  Tile 2: Wk  ]   [  Tile 3: Wv  ]
  [  Tile 4: W1-left  ]   [  Tile 5: W1-right  ]
  [  Tile 6: W2-top  ]   [  Tile 7: W2-bottom  ]
  [  Tile 8: Wout (truncated 16x10) ]
  [  Tile 9: embeddings (token + position combined) ]
  [  Tile 10: spare / calibration  ]
```

Ten tiles, each its own small daughterboard, plug into a common
backplane. The backplane carries power, SPI for weight loading,
and analog signal routing between tiles.

---

## What changes from Stage 1

### Negative weights (differential pair representation)

A single digipot can only present positive resistance. Neural-network
weights are signed. Stage 2's tiles use **paired digipots** — one
for the positive part of each weight, one for the negative. A
differential op-amp subtracts the two column currents, producing a
signed output.

Cost: doubles the digipot count per tile (256 → 512). The weight
loader on the Pi side adds the scaling step:

```
if (w >= 0) { tap_pos = scale(w); tap_neg = 0; }
else        { tap_pos = 0;         tap_neg = scale(-w); }
```

### Analog nonlinearity

Stage 1 let the Pi do the nonlinearity digitally. Stage 2 moves it
onto the tile. Each output column gets a small patch:

- For **ReLU**: a single signal diode forward-biased from the
  summing node's output to the next-stage input. Passes positive
  swings, blocks negatives. Good enough for the FFN's ReLU.
- For **approximate softmax (via squared nonlinearity)**: a Gilbert
  cell multiplier (discrete AD633, or a matched-pair BJT cell)
  squares the score. The Pi still handles the final normalize-and-
  sample step — see the Chapter-22 discussion of why full softmax
  stays digital.

### Analog residual connections

The residual `H1 = X + attn_o` is trivially analog: two op-amp
outputs driving a summing node is exactly a vector add. No new
chips required.

### Tile-to-tile communication

The awkward piece. Options:

1. **Analog signal chain.** Route the output of Tile 1 directly into
   the input of Tile 2 as voltages. No ADC/DAC round-trip. Preserves
   precision but requires careful matching of output impedances.
2. **Digital round-trip.** ADC the output of Tile 1, feed it through
   the Pi, DAC into the input of Tile 2. Simpler to debug, but costs
   precision and latency. Acceptable at Stage 2; swap to option 1 in
   Stage 3 if profiling shows the round-trip matters.

Stage 2 can start with (2) for ease and move to (1) as
measurements demand.

---

## The boot sequence, now

Same shape as Stage 1, just longer:

```
1. Pi reads minai_weights.bin from microSD.
2. Pi assigns each weight matrix to a specific tile.
3. For each tile, for each weight, scale to digipot tap and send
   over SPI. All ten tiles loaded in parallel (separate SPI chip-
   selects). Total load time: a few hundred milliseconds.
4. All tiles now programmed.
5. Pi sends the initial input tokens. Backplane routes signals in
   the correct sequence: tokens -> embedding tile -> Wq/Wk/Wv tiles
   (in parallel) -> attention/score tiles -> residual -> FFN tiles
   -> residual -> Wout tile -> ADC back to the Pi.
6. Pi applies softmax, argmax, produces the predicted token.
7. Repeat for every position.
```

For MinAI's fixed digit-reversal task, this produces the familiar
`0 1 2 3 4 5 6 7 -> 7 6 5 4 3 2 1 0` pattern. Not wowing anyone, but
it's the analog version of that pattern running on 10 small PCBs on
your desk.

---

## Validation

The target for Stage 2: **identical token output to the book's
software MinAI** on the full set of 64 held-out examples (when run
with `--random=1`). The analog hardware's per-position accuracy
should be within 1–2 percentage points of the software accuracy —
the rest is analog noise and quantization of the digipot taps.

When that gap is narrow enough to not change any argmax decision,
Stage 2 is complete.

---

## What Stage 2 does not yet handle

- **Variable vocabulary.** MinAI is stuck at `VOCAB=10`. MaxAI's
  28-symbol character vocab (or larger BPE) is Stage 3.
- **Generation loop.** MinAI doesn't generate — it produces a
  reversal. Autoregressive loop is a Stage-3 concern.
- **KV cache.** Not relevant at this scale; MinAI's seq_len is 8.
  Stage 3 introduces it.

Stage 2's job is narrow: prove ten tiles can act as one transformer
block, loaded from the same trained weights the book produces,
giving the same answer. Everything interesting comes next.

---

## Budget and cadence

- **Parts:** roughly $400–$600 total (ten Stage-1 tiles + backplane +
  enclosure).
- **Time:** two to four weeks of evening work, assuming Stage 1 is
  already done and the tile design is nailed down.
- **Iteration:** any individual tile remains swap-in-a-day; the
  backplane is the new long-cycle item (one fab turn per redesign).
