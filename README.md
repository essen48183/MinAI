# MinAI

**A single-file C++ transformer, built to be read.** Every mechanism of a modern LLM — attention, softmax, backprop, residual connections, causal masking — in ~780 lines of heavily commented code. Learns to reverse a list of 8 digits (and sort, shift, or mod-sum), demonstrating in miniature every idea that powers GPT-4.

Descends from Damien Boureille's [ATTN-11](https://github.com/dbrll/ATTN-11) (PDP-11 MACRO-11 assembly) via Dave Plummer's [2.11BSD port](https://github.com/davepl/pdpsrc/tree/main/bsd/attn). Portable C++17 — builds on macOS, Linux, and Windows.

## Why this exists

It's easy to lose the plot on what a large language model does when every explanation starts at "96 transformer blocks". This program is the plot — the irreducible version that still contains every mechanism. If you can read ~780 lines of commented C++, you understand the skeleton of every frontier model. The only difference between this and GPT-4 is quantity: more layers, more heads, more width, more data. Not one new idea.

## Quick start — two paths

### If you just want to run it (no compiler, no programming required)

Prebuilt binaries for common platforms live in [`bin/`](bin/). Grab the one for your platform and run it from a terminal:

| Platform | File | How to run |
|---|---|---|
| **macOS** (Apple Silicon or Intel) | [`bin/macos-universal/minai`](bin/macos-universal/minai) | `./minai` from a terminal |
| **Linux** (x86_64) | [`bin/linux-x86_64/minai`](bin/linux-x86_64/minai) | `./minai` from a terminal |
| **Windows** (x86_64) | [`bin/windows-x86_64/minai.exe`](bin/windows-x86_64/minai.exe) | double-click or run from `cmd`/PowerShell |

See [`bin/README.md`](bin/README.md) for details (including the macOS Gatekeeper first-run workaround). After it runs, `./minai --help` shows every flag.

### If you want to read and modify the source

**macOS / Linux** (any C++17 compiler — clang++ or g++):

```bash
make              # builds ./minai with your default C++ compiler
./minai           # trains the default config and runs a demo
./minai --help    # see all flags
```

**Windows** (and an all-platforms fallback) via CMake:

```bash
cmake -B build
cmake --build build --config Release
./build/minai --help          # macOS/Linux
./build/Release/minai.exe     # Windows MSVC
```

No external dependencies — everything needed is in the standard library. Colored sparkline output uses Unicode block characters; works in macOS Terminal, Linux terminals, and Windows Terminal (the default on Windows 11).

Abridged output:

```
MinAI — fixed example, task=reverse, seq_len=8, batch=1, blocks=1, ffn=on, causal=off
Parameters: 2240    Training steps: 800

step    loss     correct/8
-----   ------   ----------
    1   2.2802   0/8
  ...
  800   0.0078   8/8

Loss curve (800 steps, 60 columns, higher bars = higher loss):
   2.280 |▇▇▇▇▆▆▆▅▅▄▄▃▃▂▂▁▁                                        | 0.008
           step 1                                           step 800

input  : 0 1 2 3 4 5 6 7
output : 7 6 5 4 3 2 1 0
```

## Read the docs

Two companion documents live next to the source. **Read the one that matches you:**

- **[TRAINER.md](TRAINER.md)** / [TRAINER.pdf](TRAINER.pdf) — the terse operator's manual. Flag reference, task zoo, worked examples. If you already know what a transformer is and just want to drive this one, start here.

- **[ARITHMETICOFINTELLIGENCE.md](ARITHMETICOFINTELLIGENCE.md)** / [ARITHMETICOFINTELLIGENCE.pdf](ARITHMETICOFINTELLIGENCE.pdf) — *The Arithmetic of Intelligence*, a three-section textbook that builds every concept from scratch: what a model is, what attention does, why GPUs are suddenly the bottleneck of the AI industry. Section 1 (Chs 1–15) is the simple introduction, grounded in `minai.cpp` and including a hands-on walkthrough of every flag. Section 2 (Chs 16–22) grows the program into `maxai.cpp`, a toy GPT that accepts prompts, tokenizes with BPE, samples with temperature/top-k/top-p, and runs under a KV cache. Section 3 (Chs 23–28) is a career/field essay on where AI is going and why software engineering is not ending. Written for a reader who has taken some math but has never seen *why* any of it mattered; a curious adult with high-school algebra can follow it end to end.

Read either document with `minai.cpp` open alongside. The comments in the source are in the same tone and will not contradict the prose.

## Flags at a glance

```
--blocks=N      stack N transformer layers (1..96, default 1 — 96 matches GPT-3 depth)
--ffn=0|1       include the feed-forward sub-layer (default on)
--causal=0|1    causal attention mask (default off)
--layernorm=0|1 Pre-LN normalization before each sub-layer; makes deep stacks trainable
--random=0|1    train on random sequences + measure held-out accuracy (default off)
--seq_len=N     sequence length 1..32 (attention is O(N^2) in this)
--batch=B       examples averaged per step 1..128 (default 1)
--task=NAME     reverse | sort | shift | mod_sum (default reverse)
--steps=N       training iterations (default 800)
--extra_demos=0|1  run three bonus demos after training (hierarchical weight quantization,
                   speculative decoding, KV cache tiering). Each is a real LLM inference
                   technique, shown in miniature. Adds a few seconds.
```

Try `./minai --random=1 --blocks=2 --batch=16 --layernorm=1 --steps=3000` for a taste of how a real LLM trains in miniature — the model has to generalize, not memorize, and with LN + batch the grokking moment should land well before 3000 steps.

## Repo layout

```
minai.cpp            the program — 19 numbered parts, every line commented
Makefile             one-line clang++ build
TRAINER.md  .pdf     operator's manual
ARITHMETICOFINTELLIGENCE.md .pdf .docx  the book — textbook, toy-GPT growth, career essay
maxai.cpp            the Section-2 companion program (grows from minai.cpp)
README.md            this file
CLAUDE.md            instructions for the AI assistant that maintains this project
md2pdf.sh            regenerates PDFs from the .md sources
LICENSE              MIT
```

## Credit

Damien Boureille wrote the [ATTN-11](https://github.com/dbrll/ATTN-11) original in PDP-11 MACRO-11 assembly. Dave Plummer ported it to 2.11BSD `as`. This repo is the C++/Apple-Silicon rewrite with extensive annotation, a textbook companion, and runtime flags for blocks / FFN / causal / random / seq_len / batch / task. The core idea — "tiny transformer, every mechanism present, no magic" — is entirely theirs.
