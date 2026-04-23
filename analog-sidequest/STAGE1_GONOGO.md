# Stage 1 go / no-go

**Decision: GO.** Stage 0 passed cleanly. Fabricate the tile.

---

## What Stage 0 had to prove

Stage 0's whole job was to kill the project, on a laptop, for free, if
the analog architecture wasn't training-viable at the noise levels real
hardware will produce. Three questions, three gates:

1. **Tile precision.** Does a Q8.8.8 bit-sliced tile with realistic noise
   actually deliver the effective-bit counts `ARCHITECTURE.md` claims?
2. **Gate 1 (training viability).** Does a full transformer — not a toy
   matmul, not one layer, but the book's `maxai.cpp` end-to-end — train
   through the sim at standard-analog PCB noise?
3. **Gate 2 (depth scaling).** Does the same architecture still train
   at 96-block depth, or does noise compound destructively as the stack
   gets deep?

Stage 1 was gated on all three passing. All three passed.

---

## What the numbers said

### Tile precision (four-setting sweep, 10,000 matmul trials each)

| Noise regime        | Per-MAC σ | Effective bits |
|---------------------|-----------|----------------|
| Zero noise (ceiling)| 0         | **23.3 bits**  |
| Precision analog    | 0.00003   | **14.3 bits**  |
| Standard analog     | 0.0003    | **11.0 bits**  |
| Hobbyist analog     | 0.003     | **7.7 bits**   |

IBM's published analog-training results (Nandakumar 2020, Le Gallo 2023)
put the training floor at ~8 effective bits per operation. Three of
four settings clear that bar; the fourth is right on the edge.

### Gate 1 — MaxAI end-to-end, 2 blocks, 3000 steps

All four noise levels converge. All four generate the identical correct
text from the prompt `"she sells "`:

> *"by the sea shore the shells she sells are"*

Final loss: 0.098 (clean), 0.059, 0.059, 0.060 (noisy). **A full
transformer trains through the sim at every realistic PCB noise level.**

### Gate 2 — 96-block depth

Matrix A (depth sweep at σ = 0.0003): 2-block loss 0.059, 12-block
0.068, 48-block 0.066, 96-block **0.053**. Loss does not grow with
depth; the deepest run has the lowest loss.

Matrix B (noise sweep at blocks = 96): clean baseline ends at loss
**0.220**. All three noisy runs land between **0.052 and 0.077**. At
GPT-3 depth, analog noise isn't a tax — it's a regularizer. This
matches Neelakantan 2015 and Gulcehre 2016 on noise injection in deep
nets. We didn't design for it. It came for free.

---

## What this locks in for Stage 1

Numbers that were speculative before Stage 0 are now committed. The
Stage 1 BOM (`STAGE1.md`) treats these as hard requirements, not
aspirations.

| Locked spec                              | Value from Stage 0                              |
|------------------------------------------|-------------------------------------------------|
| Per-MAC weight RMS error budget          | ≤ 0.03 % of full scale (σ = 0.0003)             |
| Weight slice depth                       | Q8.8.8 (three 8-bit digipots per weight)        |
| Gradient accumulator width               | 32 bits (digital, off-tile)                     |
| Weight write path                        | Stochastic rounding across slices, every step   |
| Summing-amplifier grade                  | Autozero (AD8629 or equivalent, ≤ 1 µV Vos)     |
| Resistor feedback grade                  | 0.01 % thin-film, matched set                   |
| Calibration reference cells              | ~3 % tile overhead, re-read each step           |
| Host-side FP path at tile boundaries     | Required (LayerNorm, softmax stay on the Pi)    |

The first row is the critical one. Everything else in the BOM exists to
hit it.

---

## Residual risks the sim cannot tell us

The sim is faithful to first-order noise physics, but four things only
hardware can confirm:

1. **Drift.** The sim models drift as a slow Gaussian walk. Real digipots
   drift with temperature, humidity, and time in ways that aren't
   Gaussian. Calibration reference cells and periodic recal are the plan;
   frequency of recal is a hardware-measurement question.
2. **Crosstalk.** The sim assumes per-cell noise is independent. On a
   real PCB, neighbouring traces couple. Layout discipline (ground pour,
   differential routing, 0.1 µF decoupling everywhere) is the plan;
   how much it matters is a hardware-measurement question.
3. **Temperature.** The sim runs at fixed temperature. Thin-film
   resistors have ~25 ppm/°C; digipots have wider specs. A thermistor
   per tile plus Pi-side trim is the plan; real effect is measured on
   the bench.
4. **Yield.** The sim has one tile. A stack has dozens. Part-to-part
   variation of 0.01 % resistors is small but non-zero; a few outlier
   weights will exceed the budget. Per-cell calibration absorbs this;
   how much budget it consumes is a yield question.

None of these threaten the decision — they threaten the timeline. If
drift is worse than expected, recal cadence goes up. If crosstalk is
worse than expected, the tile respins with better layout. None of them
invalidate what Stage 0 proved.

---

## Stage 1 scope (unchanged by this decision)

One 16×16 Q8.8.8 tile. One Raspberry Pi Zero 2W host. One bench. Goal:
reproduce one row of a matmul in analog within measured tolerance of
the NumPy reference, then execute one full training step (forward,
gradient, stochastic-rounded write) on physical hardware. See
`STAGE1.md` for BOM and layout.

Budget, from `STAGE1.md`: ~$200–300 parts, 4-layer PCB from a hobbyist
fab, one afternoon to place parts, one day to populate, one afternoon
to validate. If the first tile does not match the NumPy reference to
within 11 bits effective, the failure is on the bench, not in the
architecture — which is exactly the position we want to be in.

---

## What this unblocks

- Order Stage 1 BOM.
- Lay out the 16×16 tile PCB.
- Write the Pi-side orchestration code (SPI weight load, ADC read,
  32-bit gradient accumulator, stochastic-rounded writeback).
- Cross-check first hardware matmul against the sim's float reference
  for the same weights.

## What this explicitly does not claim

- That the Stage 1 tile will hit the 14.3-bit *precision-analog* number.
  Stage 1 is a *standard-analog* build; 11.0 bits is the target.
- That real hardware noise will match the sim's Gaussian model to within
  a tenth of a bit. Stage 1's first afternoon is measuring exactly that.
- That Stage 2 (block), Stage 3 (demo), and Stage 4 (card/silicon) are
  go. Each of those has its own gate.

---

**Signed off: both Stage 0 gates pass; Stage 1 is authorised to build.**
