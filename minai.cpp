// ============================================================================
//  MinAI — a minimalist transformer that learns little sequence puzzles
//  ----------------------------------------------------------------------------
//  Portable C++17 port (developed on Apple Silicon, builds on macOS, Linux,
//  and Windows) of Damien Boureille's ATTN-11 (PDP-11 MACRO-11) via Dave
//  Plummer's 2.11BSD port (github.com/davepl/pdpsrc/tree/main/bsd/attn).
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
//      --blocks=N      stack N transformer layers (1..MAX_BLOCKS)
//      --ffn=0|1       enable the feed-forward sub-layer in each block
//      --causal=0|1    causal attention mask (a la GPT)
//      --layernorm=0|1 Pre-LN normalization before each sub-layer; required
//                      to train deep stacks (see Part 7b)
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
//  See TRAINER.md for a flag reference and ARITHMETICOFINTELLIGENCE.md for the long
//  explanation + a hands-on walkthrough of the flags.
//
//  HOW THE ORIGINAL DID IT ON A PDP-11 (and what we kept vs. simplified)
//    The PDP-11 has NO floating-point unit by default. Boureille used Q8.8
//    fixed-point for the forward pass (int16 where the low 8 bits are
//    fractional; "1.5" = 0x0180 = 384) and Q15 for gradients. We use plain
//    `float` here because every modern CPU has a vector FPU and because
//    backprop is hard enough to see without integer bookkeeping over it. At
//    the bottom of the file is a working Q8 softmax-by-lookup-table demo so
//    you can watch the real PDP-11 trick execute.
//
//  Compile (Mac/Linux):   make
//          (any platform): cmake -B build && cmake --build build
//          (direct):      clang++ -std=c++17 -O2 -o minai minai.cpp
//  Run     : ./minai --help
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

constexpr int MAX_BLOCKS  = 96;   // upper bound on --blocks. 96 matches GPT-3's
                                  // layer count so you can write --blocks=96 and
                                  // literally run a model of GPT-3 depth (with
                                  // ridiculously tiny width). Memory cost is about
                                  // 75 KB of globals per block = ~7 MB at the max,
                                  // which is nothing on modern hardware.
                                  // NOTE: without LayerNorm, stacks deeper than
                                  // ~16 often fail to train — gradients vanish or
                                  // explode. That failure mode is itself a useful
                                  // lesson about why real transformers include
                                  // layer normalization between every sub-block.
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
//    --blocks=1 --ffn=0   (classic)     |   1,216
//    --blocks=1 --ffn=1   (default)     |   2,240
//    --blocks=2 --ffn=1                 |   4,032
//    --blocks=4 --ffn=1                 |   7,616
//    --blocks=16 --ffn=1                |  29,120
//    --blocks=32 --ffn=1                |  57,792
//    --blocks=96 --ffn=1  (GPT-3 depth) | 172,480
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
    int  num_blocks    = 1;
    bool use_ffn       = true;
    bool use_causal    = false;
    bool use_layernorm = false;   // Pre-LN around each sub-layer (see Part 7b)
    bool random        = false;
    int  seq_len       = 8;
    int  batch         = 1;
    Task task          = Task::Reverse;
    int  num_steps     = 800;
    bool extra_demos   = false;   // run the three bonus demos (Parts 20-22)
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
        "  --layernorm=0|1 Pre-LN normalization before each sub-layer; makes\n"
        "                  deep stacks trainable (default 0)\n"
        "  --random=0|1    random training sequences (default 0 = fixed)\n"
        "  --seq_len=N     sequence length 1..%d (default 8)\n"
        "  --batch=B       examples per step 1..%d (default 1)\n"
        "  --task=NAME     reverse | sort | shift | mod_sum (default reverse)\n"
        "  --steps=N       training steps (default 800)\n"
        "  --extra_demos=0|1  run the three bonus demos:\n"
        "                     hierarchical quantization / speculative decoding / KV tiering\n"
        "                     (default 0; adds a few seconds)\n"
        "  --help          this help\n",
        argv0, MAX_BLOCKS, MAX_SEQ_LEN, MAX_BATCH);
}

static void parse_args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if      (std::strncmp(a, "--blocks=",  9) == 0) cfg.num_blocks = std::atoi(a + 9);
        else if (std::strncmp(a, "--ffn=",     6) == 0) cfg.use_ffn    = std::atoi(a + 6) != 0;
        else if (std::strncmp(a, "--causal=",  9) == 0) cfg.use_causal    = std::atoi(a + 9) != 0;
        else if (std::strncmp(a, "--layernorm=",12) == 0) cfg.use_layernorm = std::atoi(a + 12) != 0;
        else if (std::strncmp(a, "--random=",  9) == 0) cfg.random        = std::atoi(a + 9) != 0;
        else if (std::strncmp(a, "--seq_len=",10) == 0) cfg.seq_len    = std::atoi(a + 10);
        else if (std::strncmp(a, "--batch=",   8) == 0) cfg.batch      = std::atoi(a + 8);
        else if (std::strncmp(a, "--task=",    7) == 0) cfg.task       = parse_task(a + 7);
        else if (std::strncmp(a, "--steps=",   8) == 0) cfg.num_steps  = std::atoi(a + 8);
        else if (std::strncmp(a, "--extra_demos=",14) == 0) cfg.extra_demos = std::atoi(a + 14) != 0;
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

// ---- LayerNorm parameters (used only when --layernorm=1) -------------------
// LayerNorm has two learned per-channel vectors: a "gain" (multiplier) and a
// "bias" (offset). See Part 7b for the math. We keep:
//   - Two LNs per block (one before attention, one before FFN), Pre-LN style.
//   - One final LN just before the output projection.
// gain initialized to 1, bias initialized to 0, which makes LN an identity
// at the start of training; the model can learn to move away from that.
float gain1[MAX_BLOCKS][D_MODEL],  bias1[MAX_BLOCKS][D_MODEL];   // before attention
float gain2[MAX_BLOCKS][D_MODEL],  bias2[MAX_BLOCKS][D_MODEL];   // before FFN
float gain_f[D_MODEL],             bias_f[D_MODEL];              // before Wout

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

// LayerNorm forward caches. Needed by the LN backward pass. Only populated
// when cfg.use_layernorm is true.
//   ln*_out    — the LN output (normalized, then gain-scaled and bias-shifted).
//                This is what feeds into Q/K/V (for ln1) or W1 (for ln2).
//   ln*_xhat   — the normalized-but-not-yet-gain-scaled value (x - mean)/std.
//                Stored because the backward needs both xhat and gain.
//   ln*_invstd — 1/std per token. Saves a reciprocal + sqrt in the backward.
float ln1_out   [MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float ln1_xhat  [MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float ln1_invstd[MAX_BLOCKS][MAX_SEQ_LEN];
float ln2_out   [MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float ln2_xhat  [MAX_BLOCKS][MAX_SEQ_LEN][D_MODEL];
float ln2_invstd[MAX_BLOCKS][MAX_SEQ_LEN];
float ln_f_out   [MAX_SEQ_LEN][D_MODEL];
float ln_f_xhat  [MAX_SEQ_LEN][D_MODEL];
float ln_f_invstd[MAX_SEQ_LEN];

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
// LN gradients (accumulated across a batch like the other weight grads).
float g_gain1[MAX_BLOCKS][D_MODEL], g_bias1[MAX_BLOCKS][D_MODEL];
float g_gain2[MAX_BLOCKS][D_MODEL], g_bias2[MAX_BLOCKS][D_MODEL];
float g_gain_f[D_MODEL],            g_bias_f[D_MODEL];

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

    // LayerNorm parameters. gain=1, bias=0 => LN starts life as the identity,
    // so the model behaves as if LN isn't there at step 0 and can learn to
    // move away. Same init as every real transformer.
    for (int b = 0; b < cfg.num_blocks; ++b)
        for (int d = 0; d < D_MODEL; ++d) {
            gain1[b][d] = 1.0f; bias1[b][d] = 0.0f;
            gain2[b][d] = 1.0f; bias2[b][d] = 0.0f;
        }
    for (int d = 0; d < D_MODEL; ++d) { gain_f[d] = 1.0f; bias_f[d] = 0.0f; }
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
//  PART 7b — LAYER NORMALIZATION
// ============================================================================
//  Problem: as you stack more transformer layers, the scale of the activation
//  vectors drifts. A deep stack without normalization either shrinks them to
//  zero (vanishing gradients, nothing trains) or blows them up (NaN). The
//  residual connection alone is not enough past ~10-20 layers.
//
//  Solution: after every sub-layer input, re-center and re-scale each token's
//  vector to have zero mean and unit variance across its D_MODEL components.
//  Then allow two learned per-channel knobs (gain and bias) so the model can
//  scale/shift the normalized value back out if it wants to:
//
//      y = ((x - mean(x)) / std(x)) * gain + bias
//
//  mean and std are taken across the D_MODEL axis of a SINGLE TOKEN. Each
//  token is normalized independently. Hence "layer normalization" — it
//  normalizes across the layer dimension, not across the batch.
//
//  We use "Pre-LN": LN is applied BEFORE each sub-layer (attention, FFN),
//  and the residual connection wraps the ORIGINAL (un-normalized) input:
//
//      H1 = X + Attn(LN(X))
//      H2 = H1 + FFN(LN(H1))
//
//  Pre-LN is what GPT-2+ uses; the original 2017 transformer used Post-LN
//  (LN applied AFTER each residual sum), which is much less stable in deep
//  networks. Pre-LN lets you train 100+ layers without losses blowing up.
//
//  There is also one final LayerNorm right before the output projection,
//  so the vector feeding into Wout has a predictable scale.
//
//  [PDP-11 sidebar] LayerNorm needs a square root and a reciprocal per
//  token — neither of which the PDP-11 had as instructions. A faithful
//  fixed-point port would use a Newton-Raphson-style approximate sqrt,
//  or a lookup table for 1/sqrt(x+eps). We just use std::sqrt here.
// ----------------------------------------------------------------------------

// Forward LN on one token's D-vector. Writes y[D] and caches xhat[D] + invstd
// so the backward pass doesn't have to recompute mean and std.
static void layernorm_forward_row(
    const float* x,              // input  [D_MODEL]
    float*       y,              // output [D_MODEL]
    const float* gain,           // [D_MODEL]
    const float* bias,           // [D_MODEL]
    float*       xhat_out,       // cache  [D_MODEL]  = (x - mean)/std
    float*       invstd_out      // cache  scalar     = 1/std
) {
    const int D = D_MODEL;
    const float eps = 1e-5f;
    float mean = 0.0f;
    for (int d = 0; d < D; ++d) mean += x[d];
    mean /= (float)D;
    float varsum = 0.0f;
    for (int d = 0; d < D; ++d) { float diff = x[d] - mean; varsum += diff * diff; }
    float invstd = 1.0f / std::sqrt(varsum / (float)D + eps);
    *invstd_out = invstd;
    for (int d = 0; d < D; ++d) {
        float xh = (x[d] - mean) * invstd;
        xhat_out[d] = xh;
        y[d] = xh * gain[d] + bias[d];
    }
}

// Backward LN on one token's D-vector.
//   dy       : gradient at the LN output (after gain+bias)
//   xhat     : cached (x - mean)/std from forward
//   invstd   : cached 1/std
//   gain     : the LN gain parameter
//   g_gain   : [D] accumulator for gain gradient (+=)
//   g_bias   : [D] accumulator for bias gradient (+=)
//   dx_out   : [D] gradient at the LN input (written; caller decides +=/=)
//
// Math — detailed derivation in Chapter 9 of ARITHMETICOFINTELLIGENCE.md. Short form:
//   Let dxhat_d = dy_d * gain_d.
//   Let R = Σ dxhat_d, S = Σ (xhat_d * dxhat_d).
//   Then dx_d = invstd * (dxhat_d - R/D - xhat_d * S/D).
// All sums are across the D_MODEL axis of the single token.
static void layernorm_backward_row(
    const float* dy,
    const float* xhat,
    float        invstd,
    const float* gain,
    float*       g_gain,
    float*       g_bias,
    float*       dx_out
) {
    const int D = D_MODEL;
    float dxhat[D_MODEL];
    for (int d = 0; d < D; ++d) {
        g_gain[d] += dy[d] * xhat[d];
        g_bias[d] += dy[d];
        dxhat[d]   = dy[d] * gain[d];
    }
    float R = 0.0f, S = 0.0f;
    for (int d = 0; d < D; ++d) { R += dxhat[d]; S += xhat[d] * dxhat[d]; }
    const float invD = 1.0f / (float)D;
    for (int d = 0; d < D; ++d)
        dx_out[d] = invstd * (dxhat[d] - R * invD - xhat[d] * S * invD);
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

    // Pre-LN: LayerNorm the block input before sending it into Q/K/V.
    // The residual path (below) wraps the ORIGINAL X_in, not the LN output,
    // which is what makes deep stacks trainable.
    float (*attn_in)[D_MODEL] = X_in;
    if (cfg.use_layernorm) {
        for (int t = 0; t < T; ++t)
            layernorm_forward_row(
                X_in[t], ln1_out[b][t],
                gain1[b], bias1[b],
                ln1_xhat[b][t], &ln1_invstd[b][t]);
        attn_in = ln1_out[b];
    }

    // Linear projections Q, K, V = attn_in @ (Wq, Wk, Wv).
    // Three matrix multiplies. Each gives a different learned "view" of the
    // same input. [PDP-11] Q8*Q8 products in a 32-bit accumulator, >>8 at end.
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < D_MODEL; ++j) {
            float q = 0, k = 0, v = 0;
            for (int i = 0; i < D_MODEL; ++i) {
                q += attn_in[t][i] * Wq[b][i][j];
                k += attn_in[t][i] * Wk[b][i][j];
                v += attn_in[t][i] * Wv[b][i][j];
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
    //
    // Pre-LN: if LN is enabled, normalize H1 before feeding it into W1. The
    // residual below still wraps the ORIGINAL H1 (un-normalized).
    if (cfg.use_ffn) {
        float (*ffn_in)[D_MODEL] = H1[b];
        if (cfg.use_layernorm) {
            for (int t = 0; t < T; ++t)
                layernorm_forward_row(
                    H1[b][t], ln2_out[b][t],
                    gain2[b], bias2[b],
                    ln2_xhat[b][t], &ln2_invstd[b][t]);
            ffn_in = ln2_out[b];
        }
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < D_FF; ++j) {
                float s = 0;
                for (int i = 0; i < D_MODEL; ++i) s += ffn_in[t][i] * W1[b][i][j];
                ff_pre[b][t][j] = s;
                ff_act[b][t][j] = s > 0 ? s : 0;
            }
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D_MODEL; ++d) {
                float s = 0;
                for (int j = 0; j < D_FF; ++j) s += ff_act[b][t][j] * W2[b][j][d];
                ff_out[b][t][d] = s;
                X_ot[t][d] = H1[b][t][d] + s;   // second residual (wraps H1, not ffn_in)
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
    // Pre-LN: one more LayerNorm right before Wout so the final vector feeds
    // in at a predictable scale.
    auto& X_final = X_block[cfg.num_blocks];
    float (*final_in)[D_MODEL] = X_final;
    if (cfg.use_layernorm) {
        for (int t = 0; t < T; ++t)
            layernorm_forward_row(
                X_final[t], ln_f_out[t],
                gain_f, bias_f,
                ln_f_xhat[t], &ln_f_invstd[t]);
        final_in = ln_f_out;
    }
    for (int t = 0; t < T; ++t)
        for (int v = 0; v < VOCAB; ++v) {
            float s = 0;
            for (int d = 0; d < D_MODEL; ++d) s += final_in[t][d] * Wout[d][v];
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
    std::fill_n(&g_gain1[0][0], MAX_BLOCKS * D_MODEL, 0.0f);
    std::fill_n(&g_bias1[0][0], MAX_BLOCKS * D_MODEL, 0.0f);
    std::fill_n(&g_gain2[0][0], MAX_BLOCKS * D_MODEL, 0.0f);
    std::fill_n(&g_bias2[0][0], MAX_BLOCKS * D_MODEL, 0.0f);
    std::fill_n(&g_gain_f[0], D_MODEL, 0.0f);
    std::fill_n(&g_bias_f[0], D_MODEL, 0.0f);
}

static void backward_block(int b) {
    const int T = cfg.seq_len;
    auto& X_in   = X_block[b];
    auto& dX_in  = g_X_block[b];
    auto& dX_out = g_X_block[b+1];

    // FFN backward (if enabled): X_out = H1 + ff_out (residual); with LN on,
    // ff_out was computed from ffn_in = LN2(H1), not from H1 directly.
    //
    // Strategy:
    //   1. Direct residual path: dH1 starts as dX_out.
    //   2. Backprop through W2, ReLU, W1 to get dffn_in (gradient at the
    //      INPUT to the FFN — which is either H1 or LN2(H1)).
    //   3. If LN on: route dffn_in through LN2 backward, add to dH1,
    //      accumulate gain2/bias2 grads. Weight grad for W1 uses ln2_out.
    //      If LN off: ffn_in IS H1, so just add dffn_in to dH1. W1 uses H1.
    float dH1[MAX_SEQ_LEN][D_MODEL];
    if (cfg.use_ffn) {
        float (*ffn_in_forward)[D_MODEL] = cfg.use_layernorm ? ln2_out[b] : H1[b];

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
        // ff_pre = ffn_in @ W1  =>  dffn_in = dff_pre @ W1^T, dW1 += ffn_in^T @ dff_pre
        float dffn_in[MAX_SEQ_LEN][D_MODEL];
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D_MODEL; ++d) {
                float s = 0;
                for (int j = 0; j < D_FF; ++j) s += g_ff_pre[b][t][j] * W1[b][d][j];
                dffn_in[t][d] = s;
            }
        for (int d = 0; d < D_MODEL; ++d)
            for (int j = 0; j < D_FF; ++j) {
                float s = 0;
                for (int t = 0; t < T; ++t) s += ffn_in_forward[t][d] * g_ff_pre[b][t][j];
                g_W1[b][d][j] += s;
            }
        // Route dffn_in back into dH1. With LN on, this goes through LN2
        // backward; without LN, it's the identity (just add).
        if (cfg.use_layernorm) {
            float dH1_via_ln[MAX_SEQ_LEN][D_MODEL];
            for (int t = 0; t < T; ++t)
                layernorm_backward_row(
                    dffn_in[t], ln2_xhat[b][t], ln2_invstd[b][t],
                    gain2[b], g_gain2[b], g_bias2[b],
                    dH1_via_ln[t]);
            for (int t = 0; t < T; ++t)
                for (int d = 0; d < D_MODEL; ++d) dH1[t][d] += dH1_via_ln[t][d];
        } else {
            for (int t = 0; t < T; ++t)
                for (int d = 0; d < D_MODEL; ++d) dH1[t][d] += dffn_in[t][d];
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

    // Q, K, V = attn_in @ (Wq, Wk, Wv). With LN on, attn_in is ln1_out[b];
    // with LN off, attn_in is X_in. Weight grads for Wq/Wk/Wv use attn_in;
    // the gradient dattn_in then either (LN off) goes straight to dX_in, or
    // (LN on) goes through LN1 backward to get an extra contribution to dX_in
    // and accumulate gain1/bias1 grads.
    float (*attn_in_forward)[D_MODEL] = cfg.use_layernorm ? ln1_out[b] : X_in;

    float dattn_in[MAX_SEQ_LEN][D_MODEL];
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D_MODEL; ++d) {
            float s = 0;
            for (int j = 0; j < D_MODEL; ++j) {
                s += g_Q[b][t][j] * Wq[b][d][j]
                   + g_K[b][t][j] * Wk[b][d][j]
                   + g_V[b][t][j] * Wv[b][d][j];
            }
            dattn_in[t][d] = s;
        }
    for (int i = 0; i < D_MODEL; ++i)
        for (int j = 0; j < D_MODEL; ++j) {
            float sq = 0, sk = 0, sv = 0;
            for (int t = 0; t < T; ++t) {
                sq += attn_in_forward[t][i] * g_Q[b][t][j];
                sk += attn_in_forward[t][i] * g_K[b][t][j];
                sv += attn_in_forward[t][i] * g_V[b][t][j];
            }
            g_Wq[b][i][j] += sq;
            g_Wk[b][i][j] += sk;
            g_Wv[b][i][j] += sv;
        }

    // Route dattn_in back into dX_in (where dX_in already holds the residual
    // contribution from dH1 via `dX_in[t][d] = dH1[t][d]` earlier).
    if (cfg.use_layernorm) {
        float dX_via_ln[MAX_SEQ_LEN][D_MODEL];
        for (int t = 0; t < T; ++t)
            layernorm_backward_row(
                dattn_in[t], ln1_xhat[b][t], ln1_invstd[b][t],
                gain1[b], g_gain1[b], g_bias1[b],
                dX_via_ln[t]);
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D_MODEL; ++d) dX_in[t][d] += dX_via_ln[t][d];
    } else {
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D_MODEL; ++d) dX_in[t][d] += dattn_in[t][d];
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

    // logits = final_in @ Wout, where final_in is either X_final (LN off) or
    // the final-LN output (LN on). Backward:
    //   d(final_in) = dlogits @ Wout^T
    //   dWout += final_in^T @ dlogits
    // Then if LN is on, backprop d(final_in) through the final LN to get
    // dX_final (and accumulate gain_f / bias_f gradients).
    auto& X_final  = X_block[cfg.num_blocks];
    auto& dX_final = g_X_block[cfg.num_blocks];

    float (*final_in)[D_MODEL] = cfg.use_layernorm ? ln_f_out : X_final;

    // dfinal is the gradient at the input to Wout — i.e., at ln_f_out or X_final.
    float dfinal[MAX_SEQ_LEN][D_MODEL];
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D_MODEL; ++d) {
            float s = 0;
            for (int v = 0; v < VOCAB; ++v) s += g_logits[t][v] * Wout[d][v];
            dfinal[t][d] = s;
        }
    for (int d = 0; d < D_MODEL; ++d)
        for (int v = 0; v < VOCAB; ++v) {
            float s = 0;
            for (int t = 0; t < T; ++t) s += final_in[t][d] * g_logits[t][v];
            g_Wout[d][v] += s;
        }

    if (cfg.use_layernorm) {
        for (int t = 0; t < T; ++t)
            layernorm_backward_row(
                dfinal[t], ln_f_xhat[t], ln_f_invstd[t],
                gain_f, g_gain_f, g_bias_f,
                dX_final[t]);
    } else {
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D_MODEL; ++d) dX_final[t][d] = dfinal[t][d];
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
        if (cfg.use_layernorm) {
            // LN params use the same LR as the sub-layer they precede.
            for (int d = 0; d < D_MODEL; ++d) {
                gain1[b][d] -= LR_ATTN * g_gain1[b][d];
                bias1[b][d] -= LR_ATTN * g_bias1[b][d];
                gain2[b][d] -= LR_FFN  * g_gain2[b][d];
                bias2[b][d] -= LR_FFN  * g_bias2[b][d];
            }
        }
    }

    for (int d = 0; d < D_MODEL; ++d) for (int v = 0; v < VOCAB; ++v) Wout[d][v] -= LR_OUT * g_Wout[d][v];
    if (cfg.use_layernorm) {
        for (int d = 0; d < D_MODEL; ++d) {
            gain_f[d] -= LR_OUT * g_gain_f[d];
            bias_f[d] -= LR_OUT * g_bias_f[d];
        }
    }
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

// Logarithmic-ish reporting cadence. Dense early so you can watch the
// first descent step by step, thinning out as the loss stabilizes and
// step-by-step detail stops being informative.
//
//   steps 1..25       -> every step           (25 lines)
//   steps 26..50      -> every 5              (5 lines)
//   steps 51..200     -> every 25             (6 lines)
//   steps 201..1000   -> every 100            (8 lines)
//   steps 1001..10k   -> every 500            (18 lines)
//   steps 10k..50k    -> every 1000           (40 lines)
//   steps 50k+        -> every 5000           (N/5000 lines)
//
// Plus step 1 and the final step always.
static bool should_report(int step, int total) {
    if (step == 1 || step == total) return true;
    if (step <= 25)    return true;
    if (step <= 50)    return (step % 5)    == 0;
    if (step <= 200)   return (step % 25)   == 0;
    if (step <= 1000)  return (step % 100)  == 0;
    if (step <= 10000) return (step % 500)  == 0;
    if (step <= 50000) return (step % 1000) == 0;
    return (step % 5000) == 0;
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
//  PART 20 — HIERARCHICAL QUANTIZATION (the "organist" demo)
// ============================================================================
//  Why real LLMs don't store weights as 32-bit floats:
//    * GPT-3 at FP32 = 700 GB. At FP16 = 350 GB. At int4 with hierarchical
//      scales (what llama.cpp does) = ~90 GB, which just about fits on a
//      couple of high-end consumer GPUs.
//    * Inference is memory-bandwidth-bound (see Chapter 12). Fitting 4x more
//      weights per byte feeding the FMAs means 4x throughput.
//
//  Trick: don't use a single 1/256 step (Q8 flat). Use a HIERARCHY of scales.
//  Using the organist metaphor:
//
//      stored_weight = feet_scale * group_scale * int4_value
//
//      feet_scale  = one global number for the whole matrix           (coarse)
//      group_scale = one number per row of Wout (one per output class) (medium)
//      int4_value  = per-weight 4-bit integer in [-7, +7]             (fine)
//
//  The feet put you in the ballpark, the left hand refines to the right
//  chord, the right hand lands on the exact note. This is almost exactly
//  how llama.cpp's Q4_K_M format works on real LLM weights.
//
//  This demo quantizes the TRAINED Wout matrix five different ways and
//  measures (a) max absolute reconstruction error, (b) task accuracy when
//  the quantized Wout is used in place of the real one. It shows that
//  hierarchical Q4 achieves ~5x compression with accuracy-preserving error,
//  while flat Q4 (same bit-budget, no hierarchy) falls apart.
// ----------------------------------------------------------------------------

// Helper: compute max absolute error between two arrays.
static float max_abs_err(const float* a, const float* b, int n) {
    float m = 0;
    for (int i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}
// Helper: compute RMS error between two arrays (average hit, not worst-case).
static float rms_err(const float* a, const float* b, int n) {
    double sum_sq = 0;
    for (int i = 0; i < n; ++i) {
        double d = (double)a[i] - b[i];
        sum_sq += d * d;
    }
    return (float)std::sqrt(sum_sq / n);
}

// Temporarily swap in a quantized Wout, run the demo example, measure
// accuracy, then restore the original. Returns correct/total ratio.
static float accuracy_with_wout(float quantized_wout[D_MODEL][VOCAB], const int* tokens, const int* target) {
    float saved[D_MODEL][VOCAB];
    std::memcpy(saved, Wout, sizeof(Wout));
    std::memcpy(Wout, quantized_wout, sizeof(Wout));
    forward(tokens);
    int correct = 0;
    for (int t = 0; t < cfg.seq_len; ++t)
        if (argmax_row(probs[t], VOCAB) == target[t]) ++correct;
    std::memcpy(Wout, saved, sizeof(Wout));
    return (float)correct / (float)cfg.seq_len;
}

// Flat Q8 quantization: one global scale covers the whole range.
static void quantize_flat_q8(float out[D_MODEL][VOCAB]) {
    float absmax = 0;
    for (int i = 0; i < D_MODEL; ++i)
        for (int j = 0; j < VOCAB; ++j)
            absmax = std::fmax(absmax, std::fabs(Wout[i][j]));
    float scale = absmax / 127.0f;
    if (scale == 0) scale = 1e-9f;
    for (int i = 0; i < D_MODEL; ++i)
        for (int j = 0; j < VOCAB; ++j) {
            int q = (int)std::lround(Wout[i][j] / scale);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            out[i][j] = q * scale;
        }
}

// Flat Q4 quantization: one global scale, 4 bits per weight.
// 4 bits signed = [-7, +7] (we drop -8 for symmetry).
static void quantize_flat_q4(float out[D_MODEL][VOCAB]) {
    float absmax = 0;
    for (int i = 0; i < D_MODEL; ++i)
        for (int j = 0; j < VOCAB; ++j)
            absmax = std::fmax(absmax, std::fabs(Wout[i][j]));
    float scale = absmax / 7.0f;
    if (scale == 0) scale = 1e-9f;
    for (int i = 0; i < D_MODEL; ++i)
        for (int j = 0; j < VOCAB; ++j) {
            int q = (int)std::lround(Wout[i][j] / scale);
            if (q >  7) q =  7;
            if (q < -7) q = -7;
            out[i][j] = q * scale;
        }
}

// Hierarchical Q4 ("K-quant lite"): feet_scale (global FP32) *
// group_scale (per column, FP16-equivalent) * int4_value.
// The group is "per output class" — each of the 10 vocab columns has its
// own scale. Like a per-instrument tuning.
static void quantize_hierarchical_q4(float out[D_MODEL][VOCAB]) {
    // First find the global ("feet") scale: the max group-scale we'll see.
    float col_absmax[VOCAB];
    for (int j = 0; j < VOCAB; ++j) {
        float m = 0;
        for (int i = 0; i < D_MODEL; ++i) m = std::fmax(m, std::fabs(Wout[i][j]));
        col_absmax[j] = m;
    }
    float feet_scale = 0;
    for (int j = 0; j < VOCAB; ++j) feet_scale = std::fmax(feet_scale, col_absmax[j] / 7.0f);
    if (feet_scale == 0) feet_scale = 1e-9f;

    // Per-column group scales, stored with ~FP16 precision (we round them).
    float group_scale[VOCAB];
    for (int j = 0; j < VOCAB; ++j) {
        // ratio in [0, 1]; quantize to 6 bits for the "left hand".
        float ratio = col_absmax[j] / (feet_scale * 7.0f);
        int q6 = (int)std::lround(ratio * 63.0f);
        if (q6 < 1)  q6 = 1;
        if (q6 > 63) q6 = 63;
        group_scale[j] = q6 / 63.0f;
    }

    for (int i = 0; i < D_MODEL; ++i)
        for (int j = 0; j < VOCAB; ++j) {
            float effective_scale = feet_scale * group_scale[j];
            if (effective_scale == 0) effective_scale = 1e-9f;
            int q = (int)std::lround(Wout[i][j] / effective_scale);
            if (q >  7) q =  7;
            if (q < -7) q = -7;
            out[i][j] = q * effective_scale;
        }
}

void demo_quantization() {
    std::printf("\n================================================================\n");
    std::printf(" Demo: Hierarchical weight quantization on Wout\n");
    std::printf("   'feet + left hand + right hand' compression, real LLM style.\n");
    std::printf("================================================================\n");

    // Evaluate each scheme on the default demo example (fixed [0..seq_len-1]).
    int tokens[MAX_SEQ_LEN], target[MAX_SEQ_LEN];
    for (int t = 0; t < cfg.seq_len; ++t) tokens[t] = t % VOCAB;
    make_target(tokens, target);

    float q_flat_q8[D_MODEL][VOCAB];
    float q_flat_q4[D_MODEL][VOCAB];
    float q_hier_q4[D_MODEL][VOCAB];

    quantize_flat_q8          (q_flat_q8);
    quantize_flat_q4          (q_flat_q4);
    quantize_hierarchical_q4  (q_hier_q4);

    const int N = D_MODEL * VOCAB;
    float max_q8    = max_abs_err(&Wout[0][0], &q_flat_q8[0][0], N);
    float max_q4    = max_abs_err(&Wout[0][0], &q_flat_q4[0][0], N);
    float max_hq4   = max_abs_err(&Wout[0][0], &q_hier_q4[0][0], N);
    float rms_q8    = rms_err    (&Wout[0][0], &q_flat_q8[0][0], N);
    float rms_q4    = rms_err    (&Wout[0][0], &q_flat_q4[0][0], N);
    float rms_hq4   = rms_err    (&Wout[0][0], &q_hier_q4[0][0], N);
    float acc_q8    = accuracy_with_wout(q_flat_q8,  tokens, target);
    float acc_q4    = accuracy_with_wout(q_flat_q4,  tokens, target);
    float acc_hq4   = accuracy_with_wout(q_hier_q4,  tokens, target);

    std::printf("\n scheme                    bits/weight   max error   RMS error   accuracy\n");
    std::printf(" -----------------------    -----------   ---------   ---------   --------\n");
    std::printf(" FP32 baseline                32           0.000000    0.000000    8/8 (reference)\n");
    std::printf(" Q8 flat (1 scale)             8           %.6f    %.6f    %d/%d\n", max_q8,  rms_q8,  (int)(acc_q8  * cfg.seq_len), cfg.seq_len);
    std::printf(" Q4 flat (1 scale)             4           %.6f    %.6f    %d/%d\n", max_q4,  rms_q4,  (int)(acc_q4  * cfg.seq_len), cfg.seq_len);
    std::printf(" Q4 hierarchical (feet+hand)  ~5           %.6f    %.6f    %d/%d\n", max_hq4, rms_hq4, (int)(acc_hq4 * cfg.seq_len), cfg.seq_len);

    std::printf("\n The hierarchical scheme stores a global 32-bit 'feet' scale plus a\n");
    std::printf(" 6-bit per-column 'left hand' scale plus a 4-bit 'right hand' index per\n");
    std::printf(" weight. Total: ~5 bits per weight on average, yet RMS error typically\n");
    std::printf(" lands between Q8 and flat Q4 — i.e., you get half the bits of Q8 with\n");
    std::printf(" most of its precision. On MinAI's 16x10 Wout the gap is modest because\n");
    std::printf(" column magnitudes are similar. On a 12288x50000 GPT output matrix the\n");
    std::printf(" per-column scales vary by orders of magnitude, and hierarchical Q4\n");
    std::printf(" beats flat Q4 by a dramatic margin. This is the idea behind llama.cpp's\n");
    std::printf(" Q4_K_M — how 70B-parameter LLMs fit on a MacBook.\n");
}

// ============================================================================
//  PART 21 — SPECULATIVE DECODING (the "fast cortex proposes, slow cortex
//  verifies" demo)
// ============================================================================
//  In a real LLM inference stack this is the dominant latency optimization.
//  A SMALL, fast, heavily-quantized DRAFT model guesses several tokens ahead.
//  The LARGE, slow, high-precision VERIFIER model checks all of the draft's
//  guesses in ONE parallel forward pass. Tokens the big model agrees with are
//  accepted immediately; the first disagreement is overruled by the big
//  model, and the rest of the draft's predictions are discarded.
//
//  Why it speeds things up: the big model's forward pass is dominated by
//  loading its weights from memory (Chapter 12 again). Loading those weights
//  ONCE to verify 4 token-positions costs almost the same as loading them
//  ONCE to produce 1 token. So if the draft is right 70% of the time, the
//  big model effectively produces ~3 tokens per forward pass instead of 1.
//
//  Our demo here is the agreement-rate half of this story. We train a tiny
//  DRAFT model (1 block, no FFN, no LayerNorm) from scratch, then on the
//  held-out set we count the fraction of positions where the draft's top
//  prediction matches the big model's top prediction. That fraction is the
//  EXPECTED ACCEPTANCE RATE of speculative decoding. Given the rate, we can
//  compute the implied speedup for any draft:big cost ratio.
// ----------------------------------------------------------------------------

// A snapshot of every learnable parameter, so we can save big, train draft,
// then restore big for the final demo/plot.
struct WeightSnapshot {
    float token_emb[VOCAB      ][D_MODEL];
    float pos_emb  [MAX_SEQ_LEN][D_MODEL];
    float Wq[MAX_BLOCKS][D_MODEL][D_MODEL];
    float Wk[MAX_BLOCKS][D_MODEL][D_MODEL];
    float Wv[MAX_BLOCKS][D_MODEL][D_MODEL];
    float W1[MAX_BLOCKS][D_MODEL][D_FF   ];
    float W2[MAX_BLOCKS][D_FF   ][D_MODEL];
    float Wout[D_MODEL][VOCAB];
    float gain1 [MAX_BLOCKS][D_MODEL], bias1 [MAX_BLOCKS][D_MODEL];
    float gain2 [MAX_BLOCKS][D_MODEL], bias2 [MAX_BLOCKS][D_MODEL];
    float gain_f[D_MODEL],             bias_f[D_MODEL];
    Config cfg_at_save;
};

static void snapshot_save(WeightSnapshot& s) {
    std::memcpy(s.token_emb, token_emb, sizeof(token_emb));
    std::memcpy(s.pos_emb,   pos_emb,   sizeof(pos_emb));
    std::memcpy(s.Wq, Wq, sizeof(Wq));
    std::memcpy(s.Wk, Wk, sizeof(Wk));
    std::memcpy(s.Wv, Wv, sizeof(Wv));
    std::memcpy(s.W1, W1, sizeof(W1));
    std::memcpy(s.W2, W2, sizeof(W2));
    std::memcpy(s.Wout, Wout, sizeof(Wout));
    std::memcpy(s.gain1, gain1, sizeof(gain1));
    std::memcpy(s.bias1, bias1, sizeof(bias1));
    std::memcpy(s.gain2, gain2, sizeof(gain2));
    std::memcpy(s.bias2, bias2, sizeof(bias2));
    std::memcpy(s.gain_f, gain_f, sizeof(gain_f));
    std::memcpy(s.bias_f, bias_f, sizeof(bias_f));
    s.cfg_at_save = cfg;
}

static void snapshot_load(const WeightSnapshot& s) {
    std::memcpy(token_emb, s.token_emb, sizeof(token_emb));
    std::memcpy(pos_emb,   s.pos_emb,   sizeof(pos_emb));
    std::memcpy(Wq, s.Wq, sizeof(Wq));
    std::memcpy(Wk, s.Wk, sizeof(Wk));
    std::memcpy(Wv, s.Wv, sizeof(Wv));
    std::memcpy(W1, s.W1, sizeof(W1));
    std::memcpy(W2, s.W2, sizeof(W2));
    std::memcpy(Wout, s.Wout, sizeof(Wout));
    std::memcpy(gain1, s.gain1, sizeof(gain1));
    std::memcpy(bias1, s.bias1, sizeof(bias1));
    std::memcpy(gain2, s.gain2, sizeof(gain2));
    std::memcpy(bias2, s.bias2, sizeof(bias2));
    std::memcpy(gain_f, s.gain_f, sizeof(gain_f));
    std::memcpy(bias_f, s.bias_f, sizeof(bias_f));
    cfg = s.cfg_at_save;
}

// Silent training: no printing, no plot history. Used to quickly train the
// draft model in the demo without polluting the main run's output.
static void train_silent(int steps) {
    int tokens[MAX_SEQ_LEN], target[MAX_SEQ_LEN];
    for (int step = 1; step <= steps; ++step) {
        zero_param_grads();
        for (int b = 0; b < cfg.batch; ++b) {
            sample_example(tokens, target);
            forward(tokens);
            backward(tokens, target);
        }
        sgd_step();
    }
}

// Small helper: parameter count for a given config (same math as in main()).
static int params_for_config(const Config& c) {
    int base = VOCAB * D_MODEL + c.seq_len * D_MODEL + D_MODEL * VOCAB;
    int per_block = 3 * D_MODEL * D_MODEL + (c.use_ffn ? 2 * D_MODEL * D_FF : 0);
    if (c.use_layernorm) {
        per_block += 2 * (2 * D_MODEL);
        base      += 2 * D_MODEL;
    }
    return base + c.num_blocks * per_block;
}

// Expected consecutive-matches in a window of K, assuming independent per-
// position match probability p. This is the formula for the expected number
// of tokens a speculative-decoding draft gets accepted per verification
// window: sum_{k=1..K} P(first k match) = sum p^k = p*(1-p^K)/(1-p).
static float expected_accepted(float p, int K) {
    if (p >= 0.9999f) return (float)K;
    return p * (1.0f - std::pow(p, K)) / (1.0f - p);
}

void demo_speculative() {
    std::printf("\n================================================================\n");
    std::printf(" Demo: Speculative decoding (fast-cortex proposes, slow-cortex verifies)\n");
    std::printf("================================================================\n");

    // Save the big model first so we can always restore it.
    WeightSnapshot big;
    snapshot_save(big);

    // Train a DRAFT model. Real-world speculative decoding pairs drastically
    // different sizes: a ~1B draft proposing for a ~70B target (~70x ratio).
    // We emulate that spirit by using 1 block + no FFN + no LayerNorm — the
    // "classic MinAI" architecture — as our draft, regardless of how big the
    // big model is. On the default --blocks=4 --ffn=1 --layernorm=1 big, that
    // gives roughly a 9x compute ratio, which is close to realistic.
    Config saved_cfg = cfg;
    cfg.num_blocks    = 1;
    cfg.use_ffn       = false;
    cfg.use_layernorm = false;
    rng_state = 0xDEADC0DEu;   // reproducible draft init, distinct from big
    init_weights();
    int draft_steps = std::max(1000, saved_cfg.num_steps);
    std::printf(" Training draft (1 block, ffn=off, ln=off) for %d steps...\n", draft_steps);
    train_silent(draft_steps);

    // Snapshot the draft model and restore the big config for eval.
    WeightSnapshot draft;
    snapshot_save(draft);
    cfg = saved_cfg;

    // Parameter counts and a crude compute-cost proxy.
    int big_params   = params_for_config(saved_cfg);
    int draft_params = params_for_config(draft.cfg_at_save);
    // Cost proxy per block: attention ~ 3 units, FFN ~ 4 units.
    auto block_cost = [](const Config& c) -> float {
        return (float)c.num_blocks * (3.0f + (c.use_ffn ? 4.0f : 0.0f));
    };
    float big_cost   = block_cost(saved_cfg);
    float draft_cost = block_cost(draft.cfg_at_save);
    float cost_ratio = draft_cost / big_cost;    // C_draft / C_big

    // Run big and draft on every held-out example, storing per-position
    // predictions so we can both visualize and aggregate.
    const int T = cfg.seq_len;
    const int K = std::min(4, T);
    const int num_eval = cfg.random ? NUM_EVAL : 1;

    std::vector<std::vector<int>> inputs     (num_eval, std::vector<int>(T));
    std::vector<std::vector<int>> targets    (num_eval, std::vector<int>(T));
    std::vector<std::vector<int>> big_preds  (num_eval, std::vector<int>(T));
    std::vector<std::vector<int>> draft_preds(num_eval, std::vector<int>(T));

    for (int e = 0; e < num_eval; ++e) {
        if (cfg.random) { for (int t = 0; t < T; ++t) inputs[e][t] = held_out_tokens[e][t]; }
        else            { for (int t = 0; t < T; ++t) inputs[e][t] = t % VOCAB; }
        int tgt[MAX_SEQ_LEN];
        make_target(inputs[e].data(), tgt);
        for (int t = 0; t < T; ++t) targets[e][t] = tgt[t];

        snapshot_load(big);
        forward(inputs[e].data());
        for (int t = 0; t < T; ++t) big_preds[e][t] = argmax_row(probs[t], VOCAB);

        snapshot_load(draft);
        forward(inputs[e].data());
        for (int t = 0; t < T; ++t) draft_preds[e][t] = argmax_row(probs[t], VOCAB);
    }

    // Aggregate: per-position agreement plus per-window accepted-count histogram.
    // Windows are non-overlapping chunks of K starting at position 0.
    int total_positions = num_eval * T;
    int agreed_positions = 0;
    std::vector<int> hist(K + 1, 0);    // hist[i] = windows where i tokens were accepted
    int total_windows = 0;
    long long sum_accepted = 0;
    for (int e = 0; e < num_eval; ++e) {
        for (int t = 0; t < T; ++t)
            if (big_preds[e][t] == draft_preds[e][t]) ++agreed_positions;
        for (int ws = 0; ws + K <= T; ws += K) {
            int accepted = 0;
            for (int k = 0; k < K; ++k) {
                if (draft_preds[e][ws + k] == big_preds[e][ws + k]) ++accepted;
                else break;
            }
            hist[accepted]++;
            sum_accepted += accepted;
            total_windows++;
        }
    }
    float agreement    = (float)agreed_positions / (float)total_positions;
    float mean_accept  = total_windows ? (float)sum_accepted / (float)total_windows : 0;
    float speedup_meas = (mean_accept + 1.0f) / (K * cost_ratio + 1.0f);

    // --- Report ----------------------------------------------------------
    std::printf("\n Task        : %s (%s, seq_len=%d, batch=%d)\n",
                task_name(saved_cfg.task),
                saved_cfg.random ? "random" : "fixed", T, saved_cfg.batch);
    std::printf(" Big   model : %d blocks, ffn=%s, ln=%s  (%d params, cost %.0f units)\n",
                saved_cfg.num_blocks,
                saved_cfg.use_ffn ? "on" : "off",
                saved_cfg.use_layernorm ? "on" : "off",
                big_params, big_cost);
    std::printf(" Draft model : 1 block, ffn=off, ln=off  (%d params, cost %.0f units)\n",
                draft_params, draft_cost);
    std::printf(" Cost ratio  : C_draft / C_big = %.3f   (draft ~%.1fx cheaper per forward)\n",
                cost_ratio, 1.0f / cost_ratio);

    // Per-example rollout: pick the first eval example and print the
    // predictions side by side so the concept is visible, not just numerical.
    std::printf("\n --- Rollout on held-out[0] (K=%d drafts per big verify pass) ---\n", K);
    std::printf(" input  :"); for (int t = 0; t < T; ++t) std::printf(" %d", inputs     [0][t]); std::printf("\n");
    std::printf(" target :"); for (int t = 0; t < T; ++t) std::printf(" %d", targets    [0][t]); std::printf("\n");
    std::printf(" big    :"); for (int t = 0; t < T; ++t) std::printf(" %d", big_preds  [0][t]); std::printf("\n");
    std::printf(" draft  :"); for (int t = 0; t < T; ++t) std::printf(" %d", draft_preds[0][t]); std::printf("\n");
    std::printf(" match  :"); for (int t = 0; t < T; ++t) std::printf(" %c", big_preds[0][t] == draft_preds[0][t] ? 'Y' : '.'); std::printf("\n");

    // Aggregate stats.
    std::printf("\n --- Aggregate over %d held-out sequences = %d windows of K=%d ---\n",
                num_eval, total_windows, K);
    std::printf(" Per-position agreement      : %d / %d = %.1f%%\n",
                agreed_positions, total_positions, agreement * 100.0f);
    std::printf(" Mean tokens accepted/window : %.2f (max K=%d)\n", mean_accept, K);

    // Histogram. hist[i] = number of windows where exactly i tokens were accepted.
    std::printf("\n Accepted-tokens histogram:\n");
    int hist_max = 0;
    for (int i = 0; i <= K; ++i) if (hist[i] > hist_max) hist_max = hist[i];
    for (int i = 0; i <= K; ++i) {
        int bar_len = hist_max > 0 ? (hist[i] * 50) / hist_max : 0;
        std::printf("   %d accepted : %5d windows (%5.1f%%) ",
                    i, hist[i], total_windows ? 100.0f * hist[i] / total_windows : 0.0f);
        for (int b = 0; b < bar_len; ++b) std::printf("\xe2\x96\x88"); // full block
        std::printf("\n");
    }

    // Honest speedup, with a reference table illustrating the trade-off.
    std::printf("\n --- Speedup analysis (honest: accounts for draft cost) ---\n");
    std::printf(" formula : speedup = (E[accepted] + 1) / (K * C_draft/C_big + 1)\n");
    std::printf(" measured: speedup = (%.2f + 1) / (%d * %.3f + 1) = %.2fx\n",
                mean_accept, K, cost_ratio, speedup_meas);

    std::printf("\n Reference: speedup at varying agreement rates and cost ratios (K=%d)\n", K);
    std::printf("                       | agree 60%% | agree 80%% | agree 90%%\n");
    std::printf("                       +-----------+-----------+----------\n");
    struct Row { const char* label; float ratio; } rows[] = {
        { " draft 10x cheaper    ", 0.10f },
        { " draft  5x cheaper    ", 0.20f },
        { " draft  2x cheaper    ", 0.50f },
    };
    for (const Row& r : rows) {
        float s60 = (expected_accepted(0.60f, K) + 1) / (K * r.ratio + 1);
        float s80 = (expected_accepted(0.80f, K) + 1) / (K * r.ratio + 1);
        float s90 = (expected_accepted(0.90f, K) + 1) / (K * r.ratio + 1);
        std::printf(" %s |    %5.2fx |    %5.2fx |    %5.2fx\n", r.label, s60, s80, s90);
    }

    std::printf("\n Key insight: the draft must be both MUCH cheaper AND frequently right.\n");
    std::printf(" If draft and big have similar cost and agreement is modest, speculative\n");
    std::printf(" decoding can actually *slow down* inference. Production systems pair a\n");
    std::printf(" tiny (1-7B) draft with a huge (70B+) verifier to get both properties at\n");
    std::printf(" once — small enough to run fast, big enough to be mostly right.\n");

    // Restore the big model so the rest of the program uses trained weights.
    snapshot_load(big);
    cfg = saved_cfg;
}

// ============================================================================
//  PART 22 — KV CACHE TIERING (hot working memory vs cold long-term memory)
// ============================================================================
//  The user's insight: human working memory holds about 7 items (Miller 1956).
//  Beyond that, memory gets compressed into longer-term representations. This
//  turns out to be almost exactly the right engineering hack for LLMs too.
//
//  During inference, an LLM re-uses the K and V vectors of every previously
//  seen token. These get stored in the "KV cache". For a 1M-token context,
//  that's hundreds of gigabytes of KV — more than any GPU has.
//
//  Solution: keep the most recent W tokens ("working memory") at full
//  precision, and quantize older tokens' K,V down to Q8 or Q4. Attention's
//  weight on old tokens is small and noisy anyway; the precision loss is
//  largely invisible. Real techniques: StreamingLLM, H2O, KIVI.
//
//  This demo sweeps W from 0 to cfg.seq_len and measures the task accuracy
//  when K,V of positions OLDER than W are quantized to Q8. The expectation
//  is that accuracy stays high until W is small (only a few recent tokens
//  in full precision), then drops off. Where it drops is *empirically the
//  working-memory capacity of our trained model*.
// ----------------------------------------------------------------------------

// Round-trip a float through Q8 (1/256 precision) and back. This is exactly
// the precision loss you'd get from storing a float as an int16 in Q8.8 --
// the realistic compression used for the "warm" tier of a real KV cache.
// NOTE: Q8 is fine-grained enough that on MinAI's tiny activations the
// round-trip error is often below the softmax's sensitivity, so the effect
// on accuracy may not show up until much more of the sequence is cold. See
// the note at the end of the demo output.
static float q8_roundtrip(float x) {
    int16_t q = (int16_t)std::lround(x * 256.0f);
    return q / 256.0f;
}

// Run a forward pass, but quantize K[b][u][d] and V[b][u][d] to Q8 for every
// position u that is more than `hot_size` positions before the end of the
// sequence. Then re-run attention and the downstream layers from there.
// Returns the accuracy on the held-out / fixed example.
static float forward_with_quantized_kv(const int* tokens, int hot_size) {
    // First run the normal forward to populate K, V, and all later activations.
    forward(tokens);

    const int T = cfg.seq_len;

    // Quantize K and V of old positions. "Old" = distance from the last
    // position >= hot_size. In other words: the most recent hot_size tokens
    // stay at full precision.
    for (int b = 0; b < cfg.num_blocks; ++b) {
        for (int u = 0; u < T; ++u) {
            int age = (T - 1) - u;
            if (age >= hot_size) {
                for (int d = 0; d < D_MODEL; ++d) {
                    K[b][u][d] = q8_roundtrip(K[b][u][d]);
                    V[b][u][d] = q8_roundtrip(V[b][u][d]);
                }
            }
        }
    }

    // Re-run attention and downstream layers for every block using the
    // quantized K/V. The Q projections are unchanged (they came from the
    // CURRENT token, which is hot). Easiest: just rerun forward_block from
    // the scores step onward. Since we already have Q, K, V, we only need
    // to recompute scores, attn, attn_o, H1, ff_out, X_block[b+1].
    //
    // For simplicity we rerun the whole forward_block but force it to keep
    // our already-quantized K, V and already-computed Q. We do this by
    // calling the attention pipeline inline here.
    for (int b = 0; b < cfg.num_blocks; ++b) {
        // scores = Q @ K^T / sqrt(D)
        const float scale = 1.0f / std::sqrt((float)D_MODEL);
        for (int t = 0; t < T; ++t)
            for (int u = 0; u < T; ++u) {
                float s = 0;
                for (int d = 0; d < D_MODEL; ++d) s += Q[b][t][d] * K[b][u][d];
                scores[b][t][u] = s * scale;
            }
        if (cfg.use_causal) {
            const float NEG_INF = -1e30f;
            for (int t = 0; t < T; ++t)
                for (int u = t + 1; u < T; ++u) scores[b][t][u] = NEG_INF;
        }
        for (int t = 0; t < T; ++t) {
            for (int u = 0; u < T; ++u) attn[b][t][u] = scores[b][t][u];
            softmax_inplace(attn[b][t], T);
        }
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D_MODEL; ++d) {
                float s = 0;
                for (int u = 0; u < T; ++u) s += attn[b][t][u] * V[b][u][d];
                attn_o[b][t][d] = s;
            }
        // Residual + (optional) FFN, feeding into X_block[b+1].
        for (int t = 0; t < T; ++t)
            for (int d = 0; d < D_MODEL; ++d)
                H1[b][t][d] = X_block[b][t][d] + attn_o[b][t][d];
        if (cfg.use_ffn) {
            float (*ffn_in)[D_MODEL] = H1[b];
            if (cfg.use_layernorm) {
                for (int t = 0; t < T; ++t)
                    layernorm_forward_row(H1[b][t], ln2_out[b][t],
                                          gain2[b], bias2[b],
                                          ln2_xhat[b][t], &ln2_invstd[b][t]);
                ffn_in = ln2_out[b];
            }
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < D_FF; ++j) {
                    float s = 0;
                    for (int i = 0; i < D_MODEL; ++i) s += ffn_in[t][i] * W1[b][i][j];
                    ff_pre[b][t][j] = s;
                    ff_act[b][t][j] = s > 0 ? s : 0;
                }
            for (int t = 0; t < T; ++t)
                for (int d = 0; d < D_MODEL; ++d) {
                    float s = 0;
                    for (int j = 0; j < D_FF; ++j) s += ff_act[b][t][j] * W2[b][j][d];
                    ff_out[b][t][d] = s;
                    X_block[b+1][t][d] = H1[b][t][d] + s;
                }
        } else {
            for (int t = 0; t < T; ++t)
                for (int d = 0; d < D_MODEL; ++d)
                    X_block[b+1][t][d] = H1[b][t][d];
        }
    }

    // Final LN (if on) + output projection + softmax, same as forward().
    auto& X_final = X_block[cfg.num_blocks];
    float (*final_in)[D_MODEL] = X_final;
    if (cfg.use_layernorm) {
        for (int t = 0; t < T; ++t)
            layernorm_forward_row(X_final[t], ln_f_out[t],
                                  gain_f, bias_f,
                                  ln_f_xhat[t], &ln_f_invstd[t]);
        final_in = ln_f_out;
    }
    for (int t = 0; t < T; ++t)
        for (int v = 0; v < VOCAB; ++v) {
            float s = 0;
            for (int d = 0; d < D_MODEL; ++d) s += final_in[t][d] * Wout[d][v];
            logits[t][v] = s;
        }
    for (int t = 0; t < T; ++t) {
        for (int v = 0; v < VOCAB; ++v) probs[t][v] = logits[t][v];
        softmax_inplace(probs[t], VOCAB);
    }

    // Accuracy against target
    int target[MAX_SEQ_LEN];
    make_target(tokens, target);
    int correct = 0;
    for (int t = 0; t < T; ++t)
        if (argmax_row(probs[t], VOCAB) == target[t]) ++correct;
    return (float)correct / (float)T;
}

void demo_kv_tiering() {
    std::printf("\n================================================================\n");
    std::printf(" Demo: KV cache tiering (hot working memory vs cold long-term)\n");
    std::printf("   Newer tokens stay FP32; older tokens get quantized to Q8.\n");
    std::printf("   Sweep the 'hot working memory' size W from 0 to seq_len.\n");
    std::printf("================================================================\n");

    int tokens[MAX_SEQ_LEN];

    // Average accuracy across eval examples.
    const int N = cfg.random ? std::min(NUM_EVAL, 32) : 1;

    std::printf("\n W (hot tokens)   accuracy\n");
    std::printf(" --------------   --------\n");
    for (int W = 0; W <= cfg.seq_len; ++W) {
        float acc_sum = 0;
        for (int e = 0; e < N; ++e) {
            if (cfg.random) for (int t = 0; t < cfg.seq_len; ++t) tokens[t] = held_out_tokens[e][t];
            else            for (int t = 0; t < cfg.seq_len; ++t) tokens[t] = t % VOCAB;
            acc_sum += forward_with_quantized_kv(tokens, W);
        }
        float acc = acc_sum / (float)N;
        const char* marker = (W == 7) ? "   <-- Miller's 7 (human working memory)" : "";
        std::printf(" %3d             %6.1f%%%s\n", W, acc * 100.0f, marker);
    }

    std::printf("\n W = 0 means every K,V is quantized (fully 'cold' memory). W = seq_len\n");
    std::printf(" means no quantization (everything 'hot'). In real LLMs, the gap\n");
    std::printf(" between the two determines where you can put the working-memory\n");
    std::printf(" boundary W. Miller's 7 is the brain's answer; production LLMs usually\n");
    std::printf(" pick W somewhere in the same ballpark (for recent-token attention\n");
    std::printf(" sinks) — not because they're imitating the brain, but because they\n");
    std::printf(" are solving the same fast-vs-slow storage problem and arriving at\n");
    std::printf(" similar numbers.\n");
    std::printf("\n Note: on MinAI specifically, Q8 round-trip is so fine-grained that the\n");
    std::printf(" accuracy almost never moves with W. The production-scale version of\n");
    std::printf(" this same trick uses Q4 or even Q2 for cold tokens on top of much\n");
    std::printf(" larger models where softmax operates in saturation, and the accuracy\n");
    std::printf(" cost becomes visible. We keep Q8 here to stay faithful to the original\n");
    std::printf(" PDP-11 precision.\n");
}

// ============================================================================
//  MAIN
// ============================================================================

static int param_count() {
    int p = VOCAB * D_MODEL               // token_emb
          + cfg.seq_len * D_MODEL         // pos_emb (only rows actually used)
          + D_MODEL * VOCAB;              // Wout
    int per_block = 3 * D_MODEL * D_MODEL + (cfg.use_ffn ? 2 * D_MODEL * D_FF : 0);
    if (cfg.use_layernorm) {
        // 2 LNs per block (each = 2 * D_MODEL: gain + bias) + 1 final LN
        per_block += 2 * (2 * D_MODEL);
        p         += 2 * D_MODEL;
    }
    return p + cfg.num_blocks * per_block;
}

int main(int argc, char** argv) {
    parse_args(argc, argv);
    build_exptbl();
    init_weights();
    build_heldout();

    std::printf("MinAI — ");
    if (!cfg.random) std::printf("fixed example, ");
    std::printf("task=%s, seq_len=%d, batch=%d, blocks=%d, ffn=%s, causal=%s, layernorm=%s\n",
                task_name(cfg.task), cfg.seq_len, cfg.batch,
                cfg.num_blocks, cfg.use_ffn ? "on" : "off",
                cfg.use_causal ? "on" : "off", cfg.use_layernorm ? "on" : "off");
    std::printf("Parameters: %d    Training steps: %d\n\n", param_count(), cfg.num_steps);

    train();
    plot_curves();
    demo();
    demo_fixed_point_softmax();
    if (cfg.extra_demos) {
        demo_quantization();
        demo_speculative();
        demo_kv_tiering();
    }
    return 0;
}
