// ============================================================================
//  MinAI — a minimalist transformer that learns little sequence puzzles
//  ----------------------------------------------------------------------------
//  Apple-Silicon C++ port of Damien Boureille's ATTN-11 (PDP-11 MACRO-11)
//  via Dave Plummer's 2.11BSD port (github.com/davepl/pdpsrc/tree/main/bsd/attn).
//
//  THE TASK (default)
//    Input:  [0, 1, 2, 3, 4, 5, 6, 7]
//    Output: [7, 6, 5, 4, 3, 2, 1, 0]
//  A child solves it instantly. A transformer has to *learn* it from gradient
//  signal alone, and in the process demonstrates every mechanism that powers
//  GPT-4.
//
//  WHAT THIS VERSION ADDS OVER THE MINIMUM
//    Architecture:
//      --blocks=N     stack N transformer layers (1..MAX_BLOCKS)
//      --ffn=0|1      enable the feed-forward sub-layer in each block
//      --causal=0|1   causal attention mask (a la GPT)
//    Data:
//      --random=0|1   train on random sequences (forces the model to
//                     generalize instead of memorize the single fixed example)
//      --seq_len=N    sequence length 1..MAX_SEQ_LEN (watch attention's
//                     O(N^2) cost grow with this)
//      --batch=B      examples per training step, averaged into one gradient
//                     update (smoother learning; mirrors how real LLMs train)
//      --task=NAME    reverse | sort | shift | mod_sum
//    Pacing:
//      --steps=N      total training steps (default 800)
//  An ASCII loss curve and (when --random=1) a held-out accuracy curve are
//  printed after training so the reader can see the gradient-descent story
//  instead of having to imagine it.
//  See TRAINER.md for a flag reference and GUIDED_TOUR.md for the long
//  explanation + a hands-on walkthrough of the flags.
//
//  HOW THE ORIGINAL DID IT ON A PDP-11 (and what we kept vs. simplified)
//    The PDP-11 has NO floating-point unit by default. Boureille used Q8.8
//    fixed-point for the forward pass (int16 where the low 8 bits are
//    fractional; "1.5" = 0x0180 = 384) and Q15 for gradients. We use plain
//    `float` here because Apple Silicon has a vector FPU and because backprop
//    is hard enough to see without integer bookkeeping over it. At the bottom
//    of the file is a working Q8 softmax-by-lookup-table demo so you can watch
//    the real PDP-11 trick execute.
//
//  Compile: clang++ -std=c++17 -O2 -o minai minai.cpp
//  Run    : ./minai --help
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

// ============================================================================
//  PART 1 — HYPERPARAMETERS
// ============================================================================
//  These are the knobs that define the shape of the model. In big models
//  they become the thing papers are titled about ("a 70B-parameter model").
//  Here they are tiny — that's the whole point.
//
//  Compile-time upper bounds are declared first. Runtime knobs bounded by
//  these live in `cfg` in Part 3. Global buffers are sized to the MAX values
//  so that changing `--seq_len`, `--blocks` etc. at runtime requires no
//  reallocation; only the relevant *prefix* of each buffer is used.
// ----------------------------------------------------------------------------

constexpr int MAX_BLOCKS  = 4;    // upper bound on --blocks
constexpr int MAX_SEQ_LEN = 32;   // upper bound on --seq_len
constexpr int MAX_BATCH   = 128;  // upper bound on --batch
constexpr int NUM_EVAL    = 64;   // size of the held-out eval set (when --random=1)

constexpr int VOCAB       = 10;   // digits 0..9. GPT-4's vocab is ~100k.
constexpr int D_MODEL     = 16;   // per-token hidden dim. GPT-3 uses 12,288.
constexpr int D_FF        = 32;   // feed-forward inner dim (2 * D_MODEL).
                                  // GPT-3 uses 4 * D_MODEL; wider = more capacity.

// Per-group learning rates. Adam would adapt these automatically at 3x memory.
constexpr float LR_EMB  = 0.05f;
constexpr float LR_ATTN = 0.10f;
constexpr float LR_FFN  = 0.05f;
constexpr float LR_OUT  = 0.02f;

// Parameter-count table across flag combos (for the default SEQ_LEN=8):
//    base           = VOCAB*D + SEQ_LEN*D + D*VOCAB             = 448
//    per-block FFN  = 3*D*D (QKV) + 2*D*D_FF (FFN)              = 1,792
//    per-block noFFN= 3*D*D (QKV)                               =   768
//
//    --blocks=1 --ffn=0   (classic)     |  1,216
//    --blocks=1 --ffn=1   (default)     |  2,240
//    --blocks=2 --ffn=1                 |  4,032
//    --blocks=4 --ffn=1                 |  7,616
//
//  Bumping --seq_len=16 adds another 8*D = 128 params (more position rows).
//  Bumping --batch does NOT change the parameter count (it only affects how
//  gradients are averaged per step).
//  --causal also does not change the parameter count.
//  For scale: GPT-3 is 175,000,000,000 parameters. The default MinAI config
//  is therefore about 78 million times smaller. Same architecture skeleton.

// ============================================================================
//  PART 2 — A TINY REPRODUCIBLE RNG
// ============================================================================
//  xorshift32 — 3 lines, deterministic. Two separate streams:
//    rng_state       — weight init + random training sequences
//    eval_rng_state  — held-out eval set (always the same so scores compare)
// ----------------------------------------------------------------------------

static uint32_t rng_state      = 0xCAFEBABEu;
static uint32_t eval_rng_state = 0x12345678u;

static uint32_t xorshift32(uint32_t& s) {
    uint32_t x = s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return s = x;
}
static float rand_uniform(float scale) {
    // signed 32-bit divided by 2^31 -> float in (-1, 1).
    return scale * ((int32_t)xorshift32(rng_state) / 2147483648.0f);
}

// ============================================================================
//  PART 3 — RUNTIME CONFIG (command-line flags)
// ============================================================================
//  TERMINOLOGY — READ THIS FIRST. The rest of the comments use it freely.
//
//    "block" and "layer" are the SAME THING. Different papers pick different
//    words for the same unit. GPT-3 has "96 layers" = 96 blocks. MinAI with
//    --blocks=N has N of them.
//
//    A CANONICAL transformer block is TWO sub-layers stacked, each with a
//    residual connection around it:
//
//        block(X):
//          H1 = X  + attention(X)      <-- sub-layer 1: attention
//          H2 = H1 + FFN(H1)           <-- sub-layer 2: feed-forward network
//          return H2
//
//    Attention moves information ACROSS positions (how position 6 sees
//    position 1). FFN is a 2-layer MLP (a Multi-Layer Perceptron — a simple
//    neural network made of two matrix multiplies with a non-linearity in
//    between; think of it as two "linear layers" with a ReLU sandwich)
//    applied to each token INDEPENDENTLY. In other words, each position
//    runs its own little computation in parallel, thinking about what it
//    just received from attention.
//
//    --ffn=0 DROPS THE SECOND SUB-LAYER. Each block becomes just attention +
//    residual. Still trains, but it is "half a canonical layer". When a
//    paper says "GPT-3 is 96 layers", they mean 96 of the --ffn=1 kind.
//
//  FLAGS:
//    --blocks=N      stack N transformer layers (1..MAX_BLOCKS, default 1)
//    --ffn=0|1       include the FFN sub-layer (default 1)
//    --causal=0|1    mask attention so position t only sees positions <= t
//                    (default 0; required for generative LMs)
//    --random=0|1    0 = fixed training example (the classic MinAI);
//                    1 = fresh random sequence every step, plus a held-out
//                    eval set so you can watch GENERALIZATION vs memorization
//                    (default 0)
//    --seq_len=N     1..MAX_SEQ_LEN (default 8). Attention is O(N^2) in this.
//    --batch=B       examples per training step, averaged into one update
//                    (1..MAX_BATCH, default 1). This is how real models train.
//    --task=NAME     reverse | sort | shift | mod_sum   (default reverse)
//    --steps=N       total training steps (default 800)
//    --help / -h     print this list
// ----------------------------------------------------------------------------

enum class Task { Reverse, Sort, Shift, ModSum };

struct Config {
    int  num_blocks = 1;
    bool use_ffn    = true;
    bool use_causal = false;
    bool random     = false;
    int  seq_len    = 8;
    int  batch      = 1;
    Task task       = Task::Reverse;
    int  num_steps  = 800;
};
static Config cfg;

static void die(const char* msg) { std::fprintf(stderr, "error: %s\n", msg); std::exit(1); }

static const char* task_name(Task t) {
    switch (t) {
        case Task::Reverse: return "reverse";
        case Task::Sort:    return "sort";
        case Task::Shift:   return "shift";
        case Task::ModSum:  return "mod_sum";
    }
    return "?";
}

static Task parse_task(const char* s) {
    if      (std::strcmp(s, "reverse") == 0) return Task::Reverse;
    else if (std::strcmp(s, "sort")    == 0) return Task::Sort;
    else if (std::strcmp(s, "shift")   == 0) return Task::Shift;
    else if (std::strcmp(s, "mod_sum") == 0) return Task::ModSum;
    die("unknown --task (use reverse, sort, shift, or mod_sum)");
    return Task::Reverse;
}

static void print_usage(const char* argv0) {
    std::printf(
        "usage: %s [flags]\n"
        "  --blocks=N      stack N transformer layers (1..%d, default 1)\n"
        "  --ffn=0|1       feed-forward sub-layer on/off (default 1)\n"
        "  --causal=0|1    causal mask on/off (default 0)\n"
        "  --random=0|1    random training sequences (default 0 = fixed)\n"
        "  --seq_len=N     sequence length 1..%d (default 8)\n"
        "  --batch=B       examples per step 1..%d (default 1)\n"
        "  --task=NAME     reverse | sort | shift | mod_sum (default reverse)\n"
        "  --steps=N       training steps (default 800)\n"
        "  --help          this help\n",
        argv0, MAX_BLOCKS, MAX_SEQ_LEN, MAX_BATCH);
}

static void parse_args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if      (std::strncmp(a, "--blocks=",  9) == 0) cfg.num_blocks = std::atoi(a + 9);
        else if (std::strncmp(a, "--ffn=",     6) == 0) cfg.use_ffn    = std::atoi(a + 6) != 0;
        else if (std::strncmp(a, "--causal=",  9) == 0) cfg.use_causal = std::atoi(a + 9) != 0;
        else if (std::strncmp(a, "--random=",  9) == 0) cfg.random     = std::atoi(a + 9) != 0;
        else if (std::strncmp(a, "--seq_len=",10) == 0) cfg.seq_len    = std::atoi(a + 10);
        else if (std::strncmp(a, "--batch=",   8) == 0) cfg.batch      = std::atoi(a + 8);
        else if (std::strncmp(a, "--task=",    7) == 0) cfg.task       = parse_task(a + 7);
        else if (std::strncmp(a, "--steps=",   8) == 0) cfg.num_steps  = std::atoi(a + 8);
        else if (std::strcmp (a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
            print_usage(argv[0]); std::exit(0);
        }
        else { std::fprintf(stderr, "unknown flag: %s\n", a); print_usage(argv[0]); std::exit(1); }
    }
    if (cfg.num_blocks < 1 || cfg.num_blocks > MAX_BLOCKS)   die("--blocks out of range");
    if (cfg.seq_len   < 1 || cfg.seq_len   > MAX_SEQ_LEN)    die("--seq_len out of range");
    if (cfg.batch     < 1 || cfg.batch     > MAX_BATCH)      die("--batch out of range");
    if (cfg.num_steps < 1)                                    die("--steps must be positive");
}

// ============================================================================
//  PART 4 — MODEL PARAMETERS
// ============================================================================
//  "Parameters" are the numbers the model learns. Everything else in the
//  program is *arithmetic on these numbers*. We declare them all at MAX size
//  (so arrays are static) but only the first cfg.seq_len rows of pos_emb, and
//  the first cfg.num_blocks entries of each per-block array, are used on any
//  given run.
// ----------------------------------------------------------------------------

// Embeddings. Tokens -> vectors. The network cannot do math on the symbol "3",
// so we keep a learned D_MODEL-wide vector for each of the 10 digits. Position
// embeddings add a per-slot vector so position 0 and position 5 look different
// (attention would otherwise be order-blind).
float token_emb[VOCAB      ][D_MODEL];
float pos_emb  [MAX_SEQ_LEN][D_MODEL];

// Per-block attention + FFN weights.
//   Sub-layer 1 (attention) uses Wq, Wk, Wv. Moves information ACROSS
//   positions. For position t: I (Q[t]) advertise what I look for; others
//   advertise their K[u] and would contribute V[u] if chosen. My new vector
//   is the attention-weighted sum of the V's; softmax makes it soft.
//
//   Sub-layer 2 (FFN) uses W1, W2. A per-token MLP (multi-layer perceptron —
//   two linear layers with a ReLU between them): expand D_MODEL -> D_FF,
//   apply ReLU, project back to D_MODEL. Omitted when --ffn=0.
//
// Each block index b has its OWN copies of all five matrices; they are
// learned independently. Stacking (--blocks=N) just means feeding block b's
// output into block b+1's input.
float Wq[MAX_BLOCKS][D_MODEL][D_MODEL];
float Wk[MAX_BLOCKS][D_MODEL][D_MODEL];
float Wv[MAX_BLOCKS][D_MODEL][D_MODEL];
float W1[MAX_BLOCKS][D_MODEL][D_FF   ];
float W2[MAX_BLOCKS][D_FF   ][D_MODEL];

// Output projection. Turns the final per-token D_MODEL vector into VOCAB-wide
// logits. The softmax of logits is a probability distribution over digits;
// argmax is the prediction.
float Wout[D_MODEL][VOCAB];

// ============================================================================
//  PART 5 — ACTIVATION BUFFERS
// ============================================================================
//  Backprop needs the forward-pass activations, so we keep them. Globals,
//  sized to MAX_SEQ_LEN / MAX_BLOCKS. Within a given run we only touch the
//  relevant prefix.
//
//  Naming:   X_block[b]           = INPUT to block b
//            X_block[cfg.num_blocks] = OUTPUT of the whole stack
//                                      (the input to Wout)
// ----------------------------------------------------------------------------

float X_block[MAX_BLOCKS + 1][MAX_SEQ_LEN][D_MODEL];

float Q     [MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float K     [MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float V     [MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float scores[MAX_BLOCKS][MAX_SEQ_LEN][MAX_SEQ_LEN];
float attn  [MAX_BLOCKS][MAX_SEQ_LEN][MAX_SEQ_LEN];
float attn_o[MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float H1    [MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float ff_pre[MAX_BLOCKS][MAX_SEQ_LEN][D_FF   ];
float ff_act[MAX_BLOCKS][MAX_SEQ_LEN][D_FF   ];
float ff_out[MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];

float logits[MAX_SEQ_LEN][VOCAB];
float probs [MAX_SEQ_LEN][VOCAB];

// Gradient buffers. For each parameter: dL/dparam. For each activation:
// dL/dactivation. The weight-grad arrays are accumulated across a batch
// (zeroed at the start of each training step, += for each batch member).
float g_token_emb[VOCAB      ][D_MODEL];
float g_pos_emb  [MAX_SEQ_LEN][D_MODEL];
float g_Wq[MAX_BLOCKS][D_MODEL][D_MODEL];
float g_Wk[MAX_BLOCKS][D_MODEL][D_MODEL];
float g_Wv[MAX_BLOCKS][D_MODEL][D_MODEL];
float g_W1[MAX_BLOCKS][D_MODEL][D_FF   ];
float g_W2[MAX_BLOCKS][D_FF   ][D_MODEL];
float g_Wout[D_MODEL][VOCAB];

float g_X_block[MAX_BLOCKS + 1][MAX_SEQ_LEN][D_MODEL];
float g_Q     [MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float g_K     [MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float g_V     [MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float g_scores[MAX_BLOCKS][MAX_SEQ_LEN][MAX_SEQ_LEN];
float g_attn  [MAX_BLOCKS][MAX_SEQ_LEN][MAX_SEQ_LEN];
float g_attn_o[MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float g_H1    [MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float g_ff_pre[MAX_BLOCKS][MAX_SEQ_LEN][D_FF   ];
float g_ff_act[MAX_BLOCKS][MAX_SEQ_LEN][D_FF   ];
float g_ff_out[MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float g_logits[MAX_SEQ_LEN][VOCAB];

// ============================================================================
//  PART 6 — WEIGHT INITIALIZATION
// ============================================================================
//  Weights cannot all start at zero (neurons identical -> symmetry never
//  breaks). They cannot be too large either (activations explode, softmax
//  saturates, gradients vanish). Classic answer: small random numbers.
// ----------------------------------------------------------------------------

void init_weights() {
    for (int v = 0; v < VOCAB;        ++v) for (int d = 0; d < D_MODEL; ++d) token_emb[v][d] = rand_uniform(0.1f);
    // Initialize ALL pos_emb rows even though we may only use cfg.seq_len.
    // It costs nothing and keeps the results deterministic across --seq_len.
    for (int t = 0; t < MAX_SEQ_LEN;  ++t) for (int d = 0; d < D_MODEL; ++d) pos_emb  [t][d] = rand_uniform(0.1f);

    for (int b = 0; b < cfg.num_blocks; ++b) {
        for (int i = 0; i < D_MODEL; ++i)
            for (int j = 0; j < D_MODEL; ++j) {
                Wq[b][i][j] = rand_uniform(0.25f);
                Wk[b][i][j] = rand_uniform(0.25f);
                Wv[b][i][j] = rand_uniform(0.25f);
            }
        for (int i = 0; i < D_MODEL; ++i) for (int j = 0; j < D_FF;    ++j) W1[b][i][j] = rand_uniform(0.20f);
        for (int i = 0; i < D_FF;    ++i) for (int j = 0; j < D_MODEL; ++j) W2[b][i][j] = rand_uniform(0.20f);
    }

    for (int d = 0; d < D_MODEL; ++d) for (int v = 0; v < VOCAB; ++v) Wout[d][v] = rand_uniform(0.25f);
}

// ============================================================================
//  PART 7 — SOFTMAX (floats)
// ============================================================================
//  softmax(x)_i = exp(x_i) / sum_j exp(x_j). Turns a vector of reals into a
//  probability distribution. Subtract-the-max is a translation trick so
//  exp() never overflows; softmax is translation-invariant so the answer is
//  identical. [PDP-11] See Part 18 for the real trick (lookup table, no exp).
// ----------------------------------------------------------------------------

void softmax_inplace(float* row, int n) {
    float m = row[0];
    for (int i = 1; i < n; ++i) if (row[i] > m) m = row[i];
    float s = 0.0f;
    for (int i = 0; i < n; ++i) { row[i] = std::exp(row[i] - m); s += row[i]; }
    float inv = 1.0f / s;
    for (int i = 0; i < n; ++i) row[i] *= inv;
}

// ============================================================================
//  PART 8 — FORWARD PASS
// ============================================================================
//  Compute every activation up through `probs` (one probability distribution
//  per sequence position). Every loop uses T = cfg.seq_len, so the same code
//  handles any sequence length up to MAX_SEQ_LEN.
// ----------------------------------------------------------------------------

// One block = one transformer layer: attention sub-layer, then (if --ffn=1)
// an FFN sub-layer, each wrapped in a residual. Reads X_block[b] (this
// block's input), writes X_block[b+1] (this block's output).
static void forward_block(int b) {
    const int T = cfg.seq_len;
    auto& X_in = X_block[b];
    auto& X_ot = X_block[b+1];

    // Linear projections Q, K, V = X_in @ (Wq, Wk, Wv).
    // Three matrix multiplies. Each gives a different learned "view" of the
    // same input. [PDP-11] Q8*Q8 products in a 32-bit accumulator, >>8 at end.
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < D_MODEL; ++j) {
            float q = 0, k = 0, v = 0;
            for (int i = 0; i < D_MODEL; ++i) {
                q += X_in[t][i] * Wq[b][i][j];
                k += X_in[t][i] * Wk[b][i][j];
                v += X_in[t][i] * Wv[b][i][j];
            }
            Q[b][t][j] = q; K[b][t][j] = k; V[b][t][j] = v;
        }
    }

    // Attention scores = (Q @ K^T) / sqrt(D_MODEL). The division keeps
    // variance ~1 regardless of D; otherwise softmax saturates at large D
    // and gradients vanish.
    const float scale = 1.0f / std::sqrt((float)D_MODEL);
    for (int t = 0; t < T; ++t)
        for (int u = 0; u < T; ++u) {
            float s = 0;
            for (int d = 0; d < D_MODEL; ++d) s += Q[b][t][d] * K[b][u][d];
            scores[b][t][u] = s * scale;
        }

    // Causal mask: a position can only attend to itself and earlier positions.
    // Setting scores to -inf makes softmax produce exactly 0 in those slots.
    // This is what makes a transformer generative (autoregressive).
    if (cfg.use_causal) {
        const float NEG_INF = -1e30f;
        for (int t = 0; t < T; ++t)
            for (int u = t + 1; u < T; ++u)
                scores[b][t][u] = NEG_INF;
    }

    // Softmax over the key axis. Each row becomes a probability distribution
    // over positions: attn[t][u] = "how much t pays attention to u".
    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < T; ++u) attn[b][t][u] = scores[b][t][u];
        softmax_inplace(attn[b][t], T);
    }

    // Weighted sum of values: attn_o = attn @ V.
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D_MODEL; ++d) {
            float s = 0;
            for (int u = 0; u < T; ++u) s += attn[b][t][u] * V[b][u][d];
            attn_o[b][t][d] = s;
        }

    // Residual: H1 = X_in + attn_o. Attention outputs a delta on top of its
    // input. This +1 path is what keeps gradients alive in deep stacks.
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D_MODEL; ++d)
            H1[b][t][d] = X_in[t][d] + attn_o[b][t][d];

    // Feed-forward sub-layer (optional). Per-token MLP (multi-layer
    // perceptron: two linear layers with a ReLU between them). Expand
    // D_MODEL -> D_FF, apply ReLU, project back to D_MODEL. The ReLU is what
    // prevents the two matrix multiplies from collapsing into one
    // (linear * linear = linear, which would waste the extra weights).
    if (cfg.use_ffn) {
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < D_FF; ++j) {
                float s = 0;
                for (int i = 0; i < D_MODEL; ++i) s += H1[b][t][i] * W1[b][i][j];
                ff_pre[b][t][j] = s;
                ff_act[b][t][j] = s > 0 ? s : 0;
            }
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D_MODEL; ++d) {
                float s = 0;
                for (int j = 0; j < D_FF; ++j) s += ff_act[b][t][j] * W2[b][j][d];
                ff_out[b][t][d] = s;
                X_ot[t][d] = H1[b][t][d] + s;   // second residual
            }
    } else {
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D_MODEL; ++d)
                X_ot[t][d] = H1[b][t][d];
    }
}

void forward(const int tokens[]) {
    const int T = cfg.seq_len;

    // Input embedding: X_block[0] = token_emb + pos_emb. The one and only
    // line where integer token IDs enter the network. Past this, all floats.
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D_MODEL; ++d)
            X_block[0][t][d] = token_emb[tokens[t]][d] + pos_emb[t][d];

    for (int b = 0; b < cfg.num_blocks; ++b) forward_block(b);

    // Output projection + softmax to probability distribution over VOCAB.
    auto& X_final = X_block[cfg.num_blocks];
    for (int t = 0; t < T; ++t)
        for (int v = 0; v < VOCAB; ++v) {
            float s = 0;
            for (int d = 0; d < D_MODEL; ++d) s += X_final[t][d] * Wout[d][v];
            logits[t][v] = s;
        }
    for (int t = 0; t < T; ++t) {
        for (int v = 0; v < VOCAB; ++v) probs[t][v] = logits[t][v];
        softmax_inplace(probs[t], VOCAB);
    }
}

// ============================================================================
//  PART 9 — LOSS (cross-entropy)
// ============================================================================
//  L_t = -log(probs[t][target[t]]). Averaged across positions.
//  The algebraic miracle: d L(softmax(logits)) / d logits = probs - onehot.
//  No exp, no log in backward. Used in backward() below.
// ----------------------------------------------------------------------------

float compute_loss(const int target[]) {
    const int T = cfg.seq_len;
    float total = 0.0f;
    for (int t = 0; t < T; ++t) {
        float p = probs[t][target[t]];
        if (p < 1e-12f) p = 1e-12f;
        total += -std::log(p);
    }
    return total / T;
}

// ============================================================================
//  PART 10 — BACKWARD PASS
// ============================================================================
//  Backprop is the chain rule applied mechanically. For every forward op,
//  given dL/doutput, compute dL/dinputs and dL/dweights.
//
//  Linear layer Y = X @ W:   dX = dY @ W^T,   dW = X^T @ dY.
//
//  Weight-grad writes use +=, so a call to backward() ACCUMULATES into the
//  weight grads. That's exactly what we want for batch training: call
//  backward() once per batch member and the grads sum. Divide by batch size
//  as part of the initial d_logits factor (Part 10 step 1) so everything
//  downstream is already the mean.
//
//  zero_param_grads() is called ONCE PER TRAINING STEP (before the batch
//  loop), NOT once per example. Activation grads (g_X, g_Q, g_V, ...) are
//  written with = (not +=) inside backward_block, so they naturally overwrite
//  between examples and need no zeroing.
// ----------------------------------------------------------------------------

static void zero_param_grads() {
    std::fill_n(&g_token_emb[0][0], VOCAB       * D_MODEL, 0.0f);
    std::fill_n(&g_pos_emb  [0][0], MAX_SEQ_LEN * D_MODEL, 0.0f);
    std::fill_n(&g_Wq  [0][0][0], MAX_BLOCKS * D_MODEL * D_MODEL, 0.0f);
    std::fill_n(&g_Wk  [0][0][0], MAX_BLOCKS * D_MODEL * D_MODEL, 0.0f);
    std::fill_n(&g_Wv  [0][0][0], MAX_BLOCKS * D_MODEL * D_MODEL, 0.0f);
    std::fill_n(&g_W1  [0][0][0], MAX_BLOCKS * D_MODEL * D_FF,    0.0f);
    std::fill_n(&g_W2  [0][0][0], MAX_BLOCKS * D_FF    * D_MODEL, 0.0f);
    std::fill_n(&g_Wout[0][0],     D_MODEL * VOCAB,               0.0f);
}

static void backward_block(int b) {
    const int T = cfg.seq_len;
    auto& X_in   = X_block[b];
    auto& dX_in  = g_X_block[b];
    auto& dX_out = g_X_block[b+1];

    // FFN backward (if enabled): X_out = H1 + ff_out.
    float dH1[MAX_SEQ_LEN][D_MODEL];
    if (cfg.use_ffn) {
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D_MODEL; ++d) {
                g_ff_out[b][t][d] = dX_out[t][d];
                dH1[t][d]         = dX_out[t][d];          // residual direct path
            }
        // ff_out = ff_act @ W2  =>  dff_act = dff_out @ W2^T, dW2 += ff_act^T @ dff_out
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < D_FF; ++j) {
                float s = 0;
                for (int d = 0; d < D_MODEL; ++d) s += g_ff_out[b][t][d] * W2[b][j][d];
                g_ff_act[b][t][j] = s;
            }
        for (int j = 0; j < D_FF; ++j)
            for (int d = 0; d < D_MODEL; ++d) {
                float s = 0;
                for (int t = 0; t < T; ++t) s += ff_act[b][t][j] * g_ff_out[b][t][d];
                g_W2[b][j][d] += s;    // += for batch accumulation
            }
        // ReLU backward
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < D_FF; ++j)
                g_ff_pre[b][t][j] = (ff_pre[b][t][j] > 0) ? g_ff_act[b][t][j] : 0.0f;
        // ff_pre = H1 @ W1  =>  dH1 += dff_pre @ W1^T, dW1 += H1^T @ dff_pre
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D_MODEL; ++d) {
                float s = 0;
                for (int j = 0; j < D_FF; ++j) s += g_ff_pre[b][t][j] * W1[b][d][j];
                dH1[t][d] += s;
            }
        for (int d = 0; d < D_MODEL; ++d)
            for (int j = 0; j < D_FF; ++j) {
                float s = 0;
                for (int t = 0; t < T; ++t) s += H1[b][t][d] * g_ff_pre[b][t][j];
                g_W1[b][d][j] += s;
            }
    } else {
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D_MODEL; ++d) dH1[t][d] = dX_out[t][d];
    }
    // Stash dH1 for debugging / inspection.
    for (int t = 0; t < T; ++t) for (int d = 0; d < D_MODEL; ++d) g_H1[b][t][d] = dH1[t][d];

    // Attention residual: H1 = X_in + attn_o. Gradient flows to both.
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D_MODEL; ++d) {
            dX_in    [t][d] = dH1[t][d];
            g_attn_o[b][t][d] = dH1[t][d];
        }

    // attn_o = attn @ V  =>  dattn = dattn_o @ V^T, dV = attn^T @ dattn_o.
    for (int t = 0; t < T; ++t)
        for (int u = 0; u < T; ++u) {
            float s = 0;
            for (int d = 0; d < D_MODEL; ++d) s += g_attn_o[b][t][d] * V[b][u][d];
            g_attn[b][t][u] = s;
        }
    for (int u = 0; u < T; ++u)
        for (int d = 0; d < D_MODEL; ++d) {
            float s = 0;
            for (int t = 0; t < T; ++t) s += attn[b][t][u] * g_attn_o[b][t][d];
            g_V[b][u][d] = s;
        }

    // Softmax backward: dx_i = y_i * (dy_i - sum_j y_j * dy_j).
    for (int t = 0; t < T; ++t) {
        float dot = 0.0f;
        for (int u = 0; u < T; ++u) dot += attn[b][t][u] * g_attn[b][t][u];
        for (int u = 0; u < T; ++u)
            g_scores[b][t][u] = attn[b][t][u] * (g_attn[b][t][u] - dot);
    }

    // Causal mask backward: zero the gradient where the score was -inf.
    if (cfg.use_causal) {
        for (int t = 0; t < T; ++t)
            for (int u = t + 1; u < T; ++u)
                g_scores[b][t][u] = 0.0f;
    }

    // Scale backward for 1/sqrt(D_MODEL).
    const float scale = 1.0f / std::sqrt((float)D_MODEL);
    for (int t = 0; t < T; ++t)
        for (int u = 0; u < T; ++u)
            g_scores[b][t][u] *= scale;

    // scores = Q @ K^T  =>  dQ = g_scores @ K, dK = g_scores^T @ Q.
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D_MODEL; ++d) {
            float s = 0;
            for (int u = 0; u < T; ++u) s += g_scores[b][t][u] * K[b][u][d];
            g_Q[b][t][d] = s;
        }
    for (int u = 0; u < T; ++u)
        for (int d = 0; d < D_MODEL; ++d) {
            float s = 0;
            for (int t = 0; t < T; ++t) s += g_scores[b][t][u] * Q[b][t][d];
            g_K[b][u][d] = s;
        }

    // Q, K, V = X_in @ (Wq, Wk, Wv). dX_in gets contributions from all three.
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D_MODEL; ++d) {
            float s = 0;
            for (int j = 0; j < D_MODEL; ++j) {
                s += g_Q[b][t][j] * Wq[b][d][j]
                   + g_K[b][t][j] * Wk[b][d][j]
                   + g_V[b][t][j] * Wv[b][d][j];
            }
            dX_in[t][d] += s;   // += because residual path already wrote it
        }
    for (int i = 0; i < D_MODEL; ++i)
        for (int j = 0; j < D_MODEL; ++j) {
            float sq = 0, sk = 0, sv = 0;
            for (int t = 0; t < T; ++t) {
                sq += X_in[t][i] * g_Q[b][t][j];
                sk += X_in[t][i] * g_K[b][t][j];
                sv += X_in[t][i] * g_V[b][t][j];
            }
            g_Wq[b][i][j] += sq;   // += for batch accumulation
            g_Wk[b][i][j] += sk;
            g_Wv[b][i][j] += sv;
        }
}

void backward(const int tokens[], const int target[]) {
    const int T = cfg.seq_len;

    // dL/dlogits = (probs - onehot(target)) / T / batch. Division by `batch`
    // here propagates through the chain rule so all downstream weight grads
    // end up representing the MEAN gradient across batch members. This is
    // exactly what we want for SGD with batching.
    const float inv_batch = 1.0f / (float)cfg.batch;
    for (int t = 0; t < T; ++t)
        for (int v = 0; v < VOCAB; ++v)
            g_logits[t][v] = (probs[t][v] - (v == target[t] ? 1.0f : 0.0f))
                             / (float)T * inv_batch;

    // logits = X_final @ Wout  =>  dX_final = dlogits @ Wout^T, dWout += X_final^T @ dlogits
    auto& X_final  = X_block[cfg.num_blocks];
    auto& dX_final = g_X_block[cfg.num_blocks];
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D_MODEL; ++d) {
            float s = 0;
            for (int v = 0; v < VOCAB; ++v) s += g_logits[t][v] * Wout[d][v];
            dX_final[t][d] = s;
        }
    for (int d = 0; d < D_MODEL; ++d)
        for (int v = 0; v < VOCAB; ++v) {
            float s = 0;
            for (int t = 0; t < T; ++t) s += X_final[t][d] * g_logits[t][v];
            g_Wout[d][v] += s;      // += for batch accumulation
        }

    // Walk blocks in REVERSE. Each block consumes g_X_block[b+1] and produces
    // g_X_block[b]. Weight grads inside backward_block use +=.
    for (int b = cfg.num_blocks - 1; b >= 0; --b) backward_block(b);

    // Embedding backward: X_block[0] = token_emb + pos_emb. Both receive the
    // same gradient. += because the same vocab slot may be looked up multiple
    // times in one sequence (random data especially).
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D_MODEL; ++d) {
            g_token_emb[tokens[t]][d] += g_X_block[0][t][d];
            g_pos_emb  [t]        [d] += g_X_block[0][t][d];
        }
}

// ============================================================================
//  PART 11 — OPTIMIZER: vanilla SGD with per-group learning rates
// ============================================================================
//  w := w - lr * dL/dw. The grads already reflect the batch mean (see Part
//  10), so nothing more to do here than multiply by lr and subtract.
// ----------------------------------------------------------------------------

void sgd_step() {
    for (int v = 0; v < VOCAB;         ++v) for (int d = 0; d < D_MODEL; ++d) token_emb[v][d] -= LR_EMB * g_token_emb[v][d];
    for (int t = 0; t < cfg.seq_len;   ++t) for (int d = 0; d < D_MODEL; ++d) pos_emb  [t][d] -= LR_EMB * g_pos_emb  [t][d];

    for (int b = 0; b < cfg.num_blocks; ++b) {
        for (int i = 0; i < D_MODEL; ++i)
            for (int j = 0; j < D_MODEL; ++j) {
                Wq[b][i][j] -= LR_ATTN * g_Wq[b][i][j];
                Wk[b][i][j] -= LR_ATTN * g_Wk[b][i][j];
                Wv[b][i][j] -= LR_ATTN * g_Wv[b][i][j];
            }
        if (cfg.use_ffn) {
            for (int i = 0; i < D_MODEL; ++i) for (int j = 0; j < D_FF;    ++j) W1[b][i][j] -= LR_FFN * g_W1[b][i][j];
            for (int i = 0; i < D_FF;    ++i) for (int j = 0; j < D_MODEL; ++j) W2[b][i][j] -= LR_FFN * g_W2[b][i][j];
        }
    }

    for (int d = 0; d < D_MODEL; ++d) for (int v = 0; v < VOCAB; ++v) Wout[d][v] -= LR_OUT * g_Wout[d][v];
}

// ============================================================================
//  PART 12 — TASKS: the target-generating function
// ============================================================================
//  Four puzzles the model can be trained on. Each is a simple rule from an
//  input sequence to an output sequence; gradient descent has to discover it.
//
//    reverse: target[t] = input[T-1-t]
//             Needs attention: each output t has to retrieve input T-1-t.
//    sort:    target = ascending-sorted(input)
//             Much harder: each output needs to know all inputs to decide rank.
//    shift:   target[t] = input[(t+1) % T]
//             Trivial positional rotation; doesn't really need attention.
//    mod_sum: target[t] = (input[t] + input[(t+1) % T]) mod VOCAB
//             Local computation; FFN does most of the work.
// ----------------------------------------------------------------------------

static void make_target(const int* tokens, int* target) {
    const int T = cfg.seq_len;
    switch (cfg.task) {
        case Task::Reverse:
            for (int t = 0; t < T; ++t) target[t] = tokens[T - 1 - t];
            break;
        case Task::Sort: {
            int tmp[MAX_SEQ_LEN];
            for (int t = 0; t < T; ++t) tmp[t] = tokens[t];
            std::sort(tmp, tmp + T);
            for (int t = 0; t < T; ++t) target[t] = tmp[t];
            break;
        }
        case Task::Shift:
            for (int t = 0; t < T; ++t) target[t] = tokens[(t + 1) % T];
            break;
        case Task::ModSum:
            for (int t = 0; t < T; ++t) target[t] = (tokens[t] + tokens[(t + 1) % T]) % VOCAB;
            break;
    }
}

// ============================================================================
//  PART 13 — DATA: training-example generator + held-out eval set
// ============================================================================
//  Two data regimes:
//    --random=0  (default): a single FIXED example, tokens[t] = t % VOCAB.
//                No held-out set because there is only one example.
//    --random=1: each training step samples B fresh random sequences. A
//                frozen held-out set of NUM_EVAL random sequences (sampled
//                once at startup from a dedicated RNG stream) is used to
//                measure GENERALIZATION — the model has never trained on
//                those specific sequences.
//
//  With --random=0 the model can memorize. With --random=1 it has to learn
//  the underlying RULE to do well on held-out. That distinction is the
//  whole point of measuring "test accuracy" in machine learning.
// ----------------------------------------------------------------------------

static int held_out_tokens[NUM_EVAL][MAX_SEQ_LEN];
static int held_out_target[NUM_EVAL][MAX_SEQ_LEN];

static void build_heldout() {
    const int T = cfg.seq_len;
    for (int i = 0; i < NUM_EVAL; ++i) {
        for (int t = 0; t < T; ++t)
            held_out_tokens[i][t] = (int)(xorshift32(eval_rng_state) % (uint32_t)VOCAB);
        make_target(held_out_tokens[i], held_out_target[i]);
    }
}

// Produce one training example. With --random=0 always returns the fixed
// sequence; with --random=1 draws from the training RNG stream.
static void sample_example(int* out_tokens, int* out_target) {
    const int T = cfg.seq_len;
    if (!cfg.random) {
        for (int t = 0; t < T; ++t) out_tokens[t] = t % VOCAB;
    } else {
        for (int t = 0; t < T; ++t)
            out_tokens[t] = (int)(xorshift32(rng_state) % (uint32_t)VOCAB);
    }
    make_target(out_tokens, out_target);
}

// ============================================================================
//  PART 14 — PREDICTION / ACCURACY
// ============================================================================

int argmax_row(const float* row, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (row[i] > row[best]) best = i;
    return best;
}
int count_correct(const int target[]) {
    const int T = cfg.seq_len;
    int ok = 0;
    for (int t = 0; t < T; ++t) if (argmax_row(probs[t], VOCAB) == target[t]) ++ok;
    return ok;
}

// ============================================================================
//  PART 15 — EVALUATION (held-out accuracy)
// ============================================================================
//  Run the model over the held-out set and report fraction of tokens
//  predicted correctly. Only meaningful when --random=1.
// ----------------------------------------------------------------------------

static float eval_heldout_accuracy() {
    const int T = cfg.seq_len;
    int correct = 0;
    for (int i = 0; i < NUM_EVAL; ++i) {
        forward(held_out_tokens[i]);
        for (int t = 0; t < T; ++t)
            if (argmax_row(probs[t], VOCAB) == held_out_target[i][t]) ++correct;
    }
    return (float)correct / (float)(NUM_EVAL * T);
}

// ============================================================================
//  PART 16 — TRAINING LOOP
// ============================================================================
//  Per step:
//    zero_param_grads();
//    for b in 1..cfg.batch:
//      sample_example(tokens, target);
//      forward(tokens);
//      accumulate loss;
//      backward(tokens, target);      // accumulates weight grads
//    sgd_step();                      // applies averaged gradient
//
//  Reporting cadence is dense early (every step for 1..20), thinning out
//  later, so the reader actually sees the rapid early descent in raw numbers.
//
//  All per-step losses are stored in loss_history for the ASCII plot.
// ----------------------------------------------------------------------------

static std::vector<float> loss_history;     // one entry per training step
static std::vector<int>   eval_step_index;  // steps at which we measured eval
static std::vector<float> eval_heldout_acc; // corresponding held-out accuracy

static bool should_report(int step, int total) {
    if (step == 1 || step == total) return true;
    if (step <= 20)   return true;
    if (step <= 50)   return (step % 5) == 0;
    if (step <= 200)  return (step % 10) == 0;
    return (step % 50) == 0;
}

void train() {
    int tokens[MAX_SEQ_LEN], target[MAX_SEQ_LEN];

    loss_history.reserve(cfg.num_steps);
    eval_step_index.clear();
    eval_heldout_acc.clear();

    // Header row depends on whether we're showing held-out.
    if (cfg.random) {
        std::printf("step    loss     train%%    heldout%%\n");
        std::printf("-----   ------   ------    --------\n");
    } else {
        std::printf("step    loss     correct/%d\n", cfg.seq_len);
        std::printf("-----   ------   ----------\n");
    }

    for (int step = 1; step <= cfg.num_steps; ++step) {
        zero_param_grads();
        float step_loss = 0.0f;
        int   step_correct = 0;

        // Batch loop: accumulate gradients across cfg.batch examples.
        for (int b = 0; b < cfg.batch; ++b) {
            sample_example(tokens, target);
            forward(tokens);
            step_loss    += compute_loss(target);
            step_correct += count_correct(target);
            backward(tokens, target);   // += into weight grads; /batch in d_logits
        }
        sgd_step();

        float avg_loss     = step_loss / cfg.batch;
        float train_acc    = (float)step_correct / (float)(cfg.batch * cfg.seq_len);
        loss_history.push_back(avg_loss);

        if (should_report(step, cfg.num_steps)) {
            if (cfg.random) {
                float heldout = eval_heldout_accuracy();
                eval_step_index.push_back(step);
                eval_heldout_acc.push_back(heldout);
                std::printf("%5d   %6.4f   %5.1f%%    %5.1f%%\n",
                            step, avg_loss, train_acc * 100.0f, heldout * 100.0f);
            } else {
                // Train correct in "correct out of 8" is clearer than a %.
                std::printf("%5d   %6.4f   %d/%d\n",
                            step, avg_loss, step_correct / cfg.batch, cfg.seq_len);
            }
        }
    }
}

// ============================================================================
//  PART 17 — ASCII PLOT
// ============================================================================
//  Tiny Unicode sparkline. Buckets the data into COLS columns, prints each
//  bucket as one of 8 block characters whose height encodes the value.
//  This keeps the "did the loss actually go down?" question visual instead
//  of making the reader imagine a chart.
// ----------------------------------------------------------------------------

static const char* const BLOCK_GLYPHS[9] = {
    " ", "\xe2\x96\x81", "\xe2\x96\x82", "\xe2\x96\x83", "\xe2\x96\x84",
    "\xe2\x96\x85", "\xe2\x96\x86", "\xe2\x96\x87", "\xe2\x96\x88"
};

static void print_sparkline(const std::vector<float>& data, int cols) {
    if (data.empty()) return;
    float vmin = data[0], vmax = data[0];
    for (float v : data) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
    float range = vmax - vmin;
    if (range < 1e-9f) range = 1.0f;

    const int N = (int)data.size();
    for (int c = 0; c < cols; ++c) {
        int lo = (int)((long long)c * N / cols);
        int hi = (int)((long long)(c + 1) * N / cols);
        if (hi <= lo) hi = lo + 1;
        if (hi > N) hi = N;
        float sum = 0; int count = 0;
        for (int i = lo; i < hi; ++i) { sum += data[i]; ++count; }
        float avg = sum / count;
        int level = (int)std::floor((avg - vmin) / range * 8.0f);
        if (level < 0) level = 0;
        if (level > 8) level = 8;
        std::printf("%s", BLOCK_GLYPHS[level]);
    }
}

static void plot_curves() {
    const int COLS = 60;
    std::printf("\nLoss curve (%d steps, %d columns, higher bars = higher loss):\n",
                (int)loss_history.size(), COLS);
    std::printf("  %6.3f |", loss_history.front());
    print_sparkline(loss_history, COLS);
    std::printf("| %.3f\n", loss_history.back());
    std::printf("           step 1");
    for (int i = 0; i < COLS - 7 - 10; ++i) std::putchar(' ');
    std::printf("step %d\n", (int)loss_history.size());

    if (!eval_heldout_acc.empty()) {
        std::printf("\nHeld-out accuracy (%zu eval points; higher bars = more correct):\n",
                    eval_heldout_acc.size());
        std::printf("  %5.1f%% |", eval_heldout_acc.front() * 100.0f);
        print_sparkline(eval_heldout_acc, COLS);
        std::printf("| %.1f%%\n", eval_heldout_acc.back() * 100.0f);
    }
}

// ============================================================================
//  PART 18 — DEMO / INFERENCE
// ============================================================================
//  Show one worked example (either the fixed sequence or a fresh random one)
//  plus the attention matrix of the LAST block.
// ----------------------------------------------------------------------------

void demo() {
    int tokens[MAX_SEQ_LEN], target[MAX_SEQ_LEN];
    const int T = cfg.seq_len;

    // For random mode, sample a specific demo sequence so the reader sees
    // the model at work on something. Seed it deterministically.
    if (cfg.random) {
        uint32_t demo_rng = 0x0DEADBEEu;
        for (int t = 0; t < T; ++t)
            tokens[t] = (int)(xorshift32(demo_rng) % (uint32_t)VOCAB);
    } else {
        for (int t = 0; t < T; ++t) tokens[t] = t % VOCAB;
    }
    make_target(tokens, target);
    forward(tokens);

    std::printf("\ntask   : %s\n", task_name(cfg.task));
    std::printf("input  : ");
    for (int t = 0; t < T; ++t) std::printf("%d ", tokens[t]);
    std::printf("\ntarget : ");
    for (int t = 0; t < T; ++t) std::printf("%d ", target[t]);
    std::printf("\noutput : ");
    for (int t = 0; t < T; ++t) std::printf("%d ", argmax_row(probs[t], VOCAB));
    std::printf("\n\n");

    // Show the attention matrix of the last block — usually the most
    // interpretable.
    int b = cfg.num_blocks - 1;
    std::printf("attention matrix of block %d (rows = output pos, cols = input pos):\n", b);
    std::printf("       ");
    for (int u = 0; u < T; ++u) std::printf("  in=%d", u);
    std::printf("\n");
    for (int t = 0; t < T; ++t) {
        std::printf("out=%d ", t);
        for (int u = 0; u < T; ++u) std::printf("  %.2f", attn[b][t][u]);
        std::printf("\n");
    }
    if (cfg.use_causal) std::printf("(causal mask: upper triangle is zero.)\n");
}

// ============================================================================
//  PART 19 — [PDP-11 BONUS] SOFTMAX WITH A 256-ENTRY FIXED-POINT EXP TABLE
// ============================================================================
//  The real trick from the PDP-11 version. No FPU, no std::exp, no float at
//  all — just integer arithmetic on a lookup table.
// ----------------------------------------------------------------------------

constexpr int EXPTBL_N = 256;
static int16_t EXPTBL[EXPTBL_N];

static void build_exptbl() {
    for (int k = 0; k < EXPTBL_N; ++k) {
        float v = std::exp(-(float)k / 8.0f);
        EXPTBL[k] = (int16_t)std::lround(v * 256.0f);
    }
}

static void softmax_q8(const int16_t* x, int16_t* out, int n) {
    int16_t max_v = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > max_v) max_v = x[i];
    int32_t sum = 0;
    for (int i = 0; i < n; ++i) {
        int32_t diff = (int32_t)max_v - x[i];
        int idx = diff >> 5;
        int16_t e = (idx < EXPTBL_N) ? EXPTBL[idx] : (int16_t)0;
        out[i] = e;
        sum += e;
    }
    if (sum == 0) sum = 1;
    for (int i = 0; i < n; ++i)
        out[i] = (int16_t)(((int32_t)out[i] << 8) / sum);
}

void demo_fixed_point_softmax() {
    float fx[5] = {3.0f, 1.5f, 0.5f, -1.0f, 2.0f};
    int16_t qx[5], qy[5];
    for (int i = 0; i < 5; ++i) qx[i] = (int16_t)std::lround(fx[i] * 256.0f);
    softmax_inplace(fx, 5);
    softmax_q8(qx, qy, 5);
    std::printf("\nQ8 softmax demo (the actual PDP-11 trick, no std::exp):\n");
    std::printf("           ");
    for (int i = 0; i < 5; ++i) std::printf("   [%d]", i);
    std::printf("\n  float:   ");
    for (int i = 0; i < 5; ++i) std::printf("  %.3f", fx[i]);
    std::printf("\n  Q8 int:  ");
    for (int i = 0; i < 5; ++i) std::printf("  %.3f", qy[i] / 256.0f);
    std::printf("\n(Q8 output = float output rounded to 1/256. No FPU used.)\n");
}

// ============================================================================
//  MAIN
// ============================================================================

static int param_count() {
    int p = VOCAB * D_MODEL               // token_emb
          + cfg.seq_len * D_MODEL         // pos_emb (only rows actually used)
          + D_MODEL * VOCAB;              // Wout
    int per_block = 3 * D_MODEL * D_MODEL + (cfg.use_ffn ? 2 * D_MODEL * D_FF : 0);
    return p + cfg.num_blocks * per_block;
}

int main(int argc, char** argv) {
    parse_args(argc, argv);
    build_exptbl();
    init_weights();
    build_heldout();

    std::printf("MinAI — ");
    if (!cfg.random) std::printf("fixed example, ");
    std::printf("task=%s, seq_len=%d, batch=%d, blocks=%d, ffn=%s, causal=%s\n",
                task_name(cfg.task), cfg.seq_len, cfg.batch,
                cfg.num_blocks, cfg.use_ffn ? "on" : "off", cfg.use_causal ? "on" : "off");
    std::printf("Parameters: %d    Training steps: %d\n\n", param_count(), cfg.num_steps);

    train();
    plot_curves();
    demo();
    demo_fixed_point_softmax();
    return 0;
}
