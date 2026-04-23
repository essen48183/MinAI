# Architecture — the whole stack, piece by piece

*This document names every major component in the analog-MinAI system
and explains what it does, why it is there, and what it corresponds to
in a normal computer. The goal is to make the whole system
legible in one sitting.*

---

## The big picture in one diagram

```
  microSD card              (trained weights, like a hard disk)
        |
        v
  Raspberry Pi Zero 2W      (host: tokenization, softmax, sampling)
        |   (SPI / I2C / parallel GPIO)
        v
  +-----------------------------------------------+
  |  Analog Tile Board                            |
  |                                               |
  |   DAC array (weight loader)                   |
  |        |                                      |
  |        v                                      |
  |   Digital potentiometers  <-- the weights     |
  |        |                                      |
  |        v                                      |
  |   Analog MAC network  (Ohm's law + Kirchhoff) |
  |        |                                      |
  |        v                                      |
  |   Nonlinearity patch (diode / BJT / AD633)    |
  |        |                                      |
  |        v                                      |
  |   ADC array (readout)                         |
  |                                               |
  +-----------------------------------------------+
        |   (readings)
        v
   Raspberry Pi Zero 2W      (applies softmax, samples token,
                              feeds next prompt token back in)
```

---

## The mapping to a normal computer

| Normal computer | This system | Why it's there |
|---|---|---|
| Hard disk / SSD | **microSD on the Pi** | Holds trained weights at rest. Just a file. |
| System RAM | **Pi's 512 MB DRAM** | Stages weights during boot. Runs the host's Python/C++ glue code. |
| CPU | **Pi's ARM cores** | Tokenizes prompts, runs softmax, samples tokens, runs the generation loop. |
| GPU | **The analog MAC tile board** | Does the expensive matmuls. The only new thing in this project. |
| PCIe bus | **SPI/I²C link from Pi to board** | How the CPU talks to the accelerator. (In Stage 3, this really is PCIe.) |
| GPU VRAM / HBM | **The digipot wiper state on the tile** | Holds weights during inference. Volatile — vanishes on power-off. |
| GPU kernel upload | **Pi's boot script walks the weight file and clocks each value into the digipot register over SPI** | Same pattern as `cudaMemcpy`, just ending at a resistance instead of a RAM cell. |

The one-line summary: **every piece of this system has a direct
counterpart in a normal GPU-based AI computer, except the matmul tile
itself, which replaces a stack of floating-point multipliers with
Ohm's law.**

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

5. Pi sends an input token as a vector of N analog voltages (via its
   onboard DAC or via one of the DAC chips on the tile).

6. The tile does the matmul in hardware. Op-amp outputs settle within
   a few hundred nanoseconds.

7. Pi reads the output currents through the ADC. Applies softmax
   digitally (cheap). Samples the next token.

8. Pi sends the new token back as the next input vector. GOTO 6.

9. Power off. Digipot state vanishes. Trained weights are still safe
   on the microSD card, ready for the next boot.
```

The boot is identical in spirit to loading a CUDA kernel from disk to
VRAM. The only change is that the final destination is an analog
resistance rather than a digital register.

---

## What is analog, what is digital

Not everything has to be analog. The goal is to put the **matmul**
(which is 90%+ of the compute) in analog, and leave the cheap discrete
operations — softmax, token sampling, loop control — in digital,
because those are rare, numerically fussy, and trivially fast on a
dollar's worth of microcontroller.

| Step | Where it runs | Why |
|---|---|---|
| Tokenize prompt | Pi | Trivial, rare. |
| Embedding lookup | Pi | It's a table read, not a matmul. |
| Linear projections (Q, K, V, W1, W2, Wout) | **Analog tiles** | These are 90% of the compute. This is the whole point. |
| Attention dot products (Q·Kᵀ) | **Analog tiles** | More matmul. |
| Softmax | Pi | Requires exp and normalization. Optional: replace with ReLU-like analog nonlinearity and retrain. |
| Sampling / argmax / top-k | Pi | Rare, cheap, needs random numbers. |
| Generation loop | Pi | Control flow. |
| Residual add | Can be on the board (beam-splitter analog) or on the Pi. Cheap either way. | |
| LayerNorm | Pi (Stage 1). Could move to analog later. | |

At Stage 1 we are content with "matmul is analog, everything else is
digital." That captures most of the energy-efficiency story already.

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

## What could go wrong (honest list)

1. **Noise accumulation.** Every analog stage adds its own thermal
   noise. A deep network may not survive without analog calibration
   or per-layer ADC/DAC reset.
2. **Digipot bandwidth.** Classic digipots are slow to rewrite
   (kilohertz range). Fine for "load at boot, hold for inference" but
   a bottleneck if you want fast weight updates. Matches our model.
3. **Op-amp saturation.** If any intermediate matmul overshoots the
   supply rails, the whole chain clips. Weight normalization during
   training matters.
4. **Nonlinearity mismatch.** A diode ReLU is not a perfect ReLU.
   Training has to bake in the measured transfer function.
5. **Yield.** A hand-built tile has 0.1% component tolerance-level
   issues. Per-tile calibration will be needed after every assembly.

None of these is a reason not to build. All of them are reasons the
Stage-1 experiments are to measure them before committing to Stage 2.

---

## What we do not do, by design

- **Training on the analog hardware.** Training requires precise
  weight updates under a precise loss gradient. Analog hardware
  is bad at precise writes today. We train on the digital version
  of MinAI (in C++, as the book describes), freeze the weights, and
  load them onto the analog card. This is exactly the pattern every
  production analog-AI chip follows (Mythic, Lightmatter, IBM
  NorthPole) as of 2026.
- **Dynamic topology.** The tile size is fixed. Larger models tile
  up; they do not reconfigure the cells.
- **Temperature compensation.** Cell conductance drifts slightly with
  ambient temperature. At Stage 1 we measure. At Stage 2 we add a
  thermistor and a Pi-side calibration loop. At Stage 3 the
  compensation is on-card.

---

## The one-sentence architecture summary

**Boot the Pi; read the model from microSD; walk the weights into the
digipot array; send input voltages; read output currents; apply a
digital softmax; loop. The matmul happens in physics.**
