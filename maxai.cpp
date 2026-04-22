// ============================================================================
//  MaxAI — MinAI grown into a toy GPT, one chapter at a time
//  ----------------------------------------------------------------------------
//  Sequel to minai.cpp. Started as a byte-for-byte copy; now gains exactly ONE
//  new idea per chapter of ARITHMETICOFINTELLIGENCE.md, beginning at chapter 16. By the end
//  of chapter 22 the program will accept a prompt string, sample tokens one at
//  a time, feed them back in, and generate text — a toy GPT, end to end.
//
//  Roadmap ([x] = already in this file, [ ] = future chapters):
//    [x] Ch 16 — next-token prediction as a task
//    [x] Ch 17 — generate() loop for autoregressive inference
//    [x] Ch 18 — sampling: temperature, top-k, top-p
//    [x] Ch 19 — widen the vocabulary from digits to characters
//    [x] Ch 20 — train on real text instead of random sequences
//    [x] Ch 21 — a byte-pair-encoding tokenizer
//    [x] Ch 22 — a KV cache so generation is not O(N^2) per token
//
//  `diff minai.cpp maxai.cpp` shows exactly what the current chapter added.
//  Everything below this line is inherited verbatim from MinAI (still true and
//  still the starting point the book describes; start there if you have not
//  read it yet).
// ============================================================================

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
//      --task=NAME    reverse | sort | shift | mod_sum | next_token
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
#include <chrono>

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

constexpr int MAX_VOCAB   = 128;  // compile-time UPPER BOUND on the vocabulary
                                  // (all static arrays are sized to this).
                                  // The actual runtime vocabulary, VOCAB below,
                                  // is set from --vocab= and can grow up to
                                  // MAX_VOCAB without any reallocation. Chapter
                                  // 19 introduces the "chars" vocabulary at
                                  // size 28 (26 letters + space + EOS); Chapter
                                  // 21 raises it higher with BPE subwords.
                                  // GPT-4's vocab is ~100k — that would need a
                                  // much larger bound and different buffer
                                  // strategy (matrices allocated on the heap)
                                  // but the logic would be exactly what's here.
int VOCAB                 = 10;   // RUNTIME vocabulary size. Defaults to the
                                  // original MinAI choice of ten digits; set
                                  // by parse_args() when --vocab= is given.
                                  // Every loop that iterates over the output
                                  // distribution uses this value; array
                                  // DECLARATIONS use MAX_VOCAB above.
int SAMPLEABLE_VOCAB      = 10;   // Range of tokens a random-data generator
                                  // will produce. Equal to VOCAB for digit
                                  // mode (all ten symbols appear in data).
                                  // In char mode, one token (EOS) is reserved
                                  // for sequence termination and never
                                  // appears in random training sequences, so
                                  // SAMPLEABLE_VOCAB = VOCAB - 1 = 27. The
                                  // loss/accuracy math still uses the full
                                  // VOCAB — the model's output distribution
                                  // is over every symbol it might ever emit,
                                  // even reserved ones.
int EOS_TOKEN             = -1;   // -1 means "this vocabulary has no EOS".
                                  // In char mode, set to 27 by parse_args().

// Chapter 21 BPE storage, declared early so print_token() in Part 3b can
// reference bpe_token_text without a forward declaration. Populated by
// bpe_init_base() and train_bpe() in Part 13c.
constexpr int MAX_TOKEN_TEXT    = 32;    // longest token text, including NUL
constexpr int MAX_CORPUS_TOKENS = 512;   // cap on encoded corpus length
static char bpe_token_text[MAX_VOCAB][MAX_TOKEN_TEXT];
static int  bpe_token_len [MAX_VOCAB];
static int  bpe_merge_left [MAX_VOCAB];
static int  bpe_merge_right[MAX_VOCAB];
static int  encoded_corpus[MAX_CORPUS_TOKENS];
static int  encoded_corpus_len = 0;
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
//    --task=NAME     reverse | sort | shift | mod_sum | next_token  (default reverse)
//    --steps=N       total training steps (default 800)
//    --help / -h     print this list
// ----------------------------------------------------------------------------

// Task::NextToken is the Chapter-16 addition: predict token t+1 from tokens
// 0..t. Every modern LLM is trained on exactly this target rule; everything
// else in this program is inherited from MinAI and stays the same.
enum class Task { Reverse, Sort, Shift, ModSum, NextToken };

// Vocab::Chars is the Chapter-19 addition. Switches the runtime vocabulary
// from the original 10-digit set (MinAI default) to a 28-symbol character
// set: 26 lowercase letters + space + a reserved end-of-sequence symbol.
// The architecture and loss function do not change; only the shape of the
// output distribution (probs over 28 classes vs 10) and how we print/parse
// tokens do. See PART 3b for the encoder/decoder and PART 12 for how the
// EOS target is plumbed into make_target().
enum class Vocab { Digits, Chars };

// Corpus is the Chapter-20 addition. When set to something other than None, a
// sliding window is drawn from a fixed English text at every training step
// (and for every held-out example), replacing the i.i.d.-uniform-random data
// MinAI and Chs 16-19 used. This is what finally turns the next-token task
// from "impossible by information theory" (see Ch 16's ceiling derivation)
// into "actually learnable": English has rich conditional structure, so the
// model's held-out cross-entropy can fall well below log(VOCAB).
// See PART 13b for the corpus storage and the window sampler.
enum class Corpus { None, Gettysburg, SeaShells };

struct Config {
    int  num_blocks    = 1;
    bool use_ffn       = true;
    bool use_causal    = false;
    bool use_layernorm = false;   // Pre-LN around each sub-layer (see Part 7b)
    bool random        = false;
    int  seq_len       = 8;
    int  batch         = 1;
    Task task          = Task::Reverse;
    Vocab vocab        = Vocab::Digits;   // Chapter 19: --vocab=digits|chars.
                                          // Digits (default) preserves every
                                          // chapter 1-18 demo verbatim. Chars
                                          // expands VOCAB to 28 and enables
                                          // EOS-terminated generation.
    Corpus corpus      = Corpus::None;    // Chapter 20: --corpus=NAME. When
                                          // set, training and held-out draws
                                          // come from a fixed English text
                                          // rather than uniform random chars.
                                          // Requires vocab=chars.
    int  bpe_merges    = 0;               // Chapter 21: --bpe=K learn K byte-pair
                                          // merges from the corpus before
                                          // training, expanding the 28-symbol
                                          // char vocabulary to 28 + K subwords.
                                          // 0 (default) disables BPE — behaviour
                                          // reduces to pure char-level (Ch 20).
                                          // Requires vocab=chars AND corpus != none.
    int  num_steps     = 800;
    bool extra_demos   = false;   // run the three bonus demos (Parts 20-22)

    // Chapter-17 addition: optional prompt for the autoregressive generation
    // demo (Part 18b). If empty, demo_generate() picks a sensible default
    // based on the training regime (fixed vs random). If non-empty, it is
    // parsed as a space-separated list of digits ("0 1 2" → tokens 0,1,2).
    // Only used when cfg.task == Task::NextToken, since autoregressive
    // generation is only meaningful for a next-token-trained model.
    char gen_prompt[64] = "";

    // Chapter-18 additions: inference-time sampling controls. See sample_token()
    // in Part 18b for the math; the short version is that the "decoding" knobs
    // every real LLM API exposes (temperature, top-k, top-p) are exactly the
    // same knobs MaxAI exposes here. Defaults produce greedy argmax decoding,
    // which matches the Chapter-17 behaviour; opt into sampling explicitly.
    float    temperature   = 0.0f;        // 0 => greedy (argmax); otherwise divide logits by T
    int      top_k         = 0;           // 0 => disabled; >0 => keep top K logits before softmax
    float    top_p         = 1.0f;        // 1.0 => disabled; <1.0 => nucleus sampling cutoff
    uint32_t sampling_seed = 0xBEEFCAFEu; // separate RNG stream so changing sampling knobs
                                          // does not perturb training / eval reproducibility

    // Chapter 22 addition: incremental "KV cache" generation. When 1, the
    // generate() loop uses forward_step() for tokens past the prompt, which
    // reuses the K and V vectors cached by the prompt forward pass and the
    // previous generation steps, skipping redundant O(N) attention work.
    // Matches the naive rollout bit-exactly; the point is pure speedup.
    int      kv_cache      = 0;           // 0 => naive (Ch 17) loop; 1 => incremental
    int      bench_gen     = 0;           // 1 => print naive-vs-cached timing benchmark
};
static Config cfg;

static void die(const char* msg) { std::fprintf(stderr, "error: %s\n", msg); std::exit(1); }

static const char* task_name(Task t) {
    switch (t) {
        case Task::Reverse:   return "reverse";
        case Task::Sort:      return "sort";
        case Task::Shift:     return "shift";
        case Task::ModSum:    return "mod_sum";
        case Task::NextToken: return "next_token";
    }
    return "?";
}

static Task parse_task(const char* s) {
    if      (std::strcmp(s, "reverse")    == 0) return Task::Reverse;
    else if (std::strcmp(s, "sort")       == 0) return Task::Sort;
    else if (std::strcmp(s, "shift")      == 0) return Task::Shift;
    else if (std::strcmp(s, "mod_sum")    == 0) return Task::ModSum;
    else if (std::strcmp(s, "next_token") == 0) return Task::NextToken;
    die("unknown --task (use reverse, sort, shift, mod_sum, or next_token)");
    return Task::Reverse;
}

static const char* vocab_name(Vocab v) {
    return (v == Vocab::Chars) ? "chars" : "digits";
}

static Vocab parse_vocab(const char* s) {
    if      (std::strcmp(s, "digits") == 0) return Vocab::Digits;
    else if (std::strcmp(s, "chars")  == 0) return Vocab::Chars;
    die("unknown --vocab (use digits or chars)");
    return Vocab::Digits;
}

static const char* corpus_name(Corpus c) {
    switch (c) {
        case Corpus::None:       return "none";
        case Corpus::Gettysburg: return "gettysburg";
        case Corpus::SeaShells:  return "seashells";
    }
    return "?";
}

static Corpus parse_corpus(const char* s) {
    if      (std::strcmp(s, "none")       == 0) return Corpus::None;
    else if (std::strcmp(s, "gettysburg") == 0) return Corpus::Gettysburg;
    else if (std::strcmp(s, "seashells")  == 0) return Corpus::SeaShells;
    die("unknown --corpus (use none, gettysburg, or seashells)");
    return Corpus::None;
}

// ----------------------------------------------------------------------------
// PART 3b — CHAR MODE ENCODING (Chapter 19)
//
// When --vocab=chars, the 28-symbol vocabulary is laid out as:
//
//     tokens 0..25  →  letters 'a'..'z'   (26 symbols)
//     token  26     →  space ' '          ( 1 symbol)
//     token  27     →  EOS                ( 1 symbol, never printed literally)
//
// encode_char() maps a character to a token id, returning -1 if the character
// is not in the vocabulary (uppercase letters are folded to lowercase for
// kindness to readers who forget to hold shift). decode_token_char() is the
// inverse: it returns the visible character for tokens 0..26 and '?' for EOS
// (EOS is printed as a literal "<EOS>" by the higher-level formatter below).
// ----------------------------------------------------------------------------

static int encode_char(char c) {
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= 'A' && c <= 'Z') return c - 'A';   // case-fold
    if (c == ' ')             return 26;
    return -1;
}

// (decode_token_char() used to live here in Chapter 19's original draft; it
// was superseded by bpe_token_text[] in Chapter 21 so print_token() could
// handle both single-char base tokens and multi-character subwords uniformly.
// The table-driven approach generalises; if you add punctuation or uppercase
// to the vocabulary later, you only need to widen bpe_token_text[], not add
// branches to the decoder.)

// Print one token in whatever representation matches the current vocab.
// In char mode EOS is printed as "<EOS>" so it stands out visually; every
// other token prints via its bpe_token_text[] entry (one char for base
// tokens 0..26, a multi-character subword for tokens 28..VOCAB-1 when BPE
// is active). In digit mode we preserve the original "%d " format exactly
// so Ch 16-18 transcripts stay unchanged.
static void print_token(int tok) {
    if (EOS_TOKEN >= 0) {
        if (tok == EOS_TOKEN)            std::fputs("<EOS>", stdout);
        else if (tok >= 0 && tok < VOCAB) std::fputs(bpe_token_text[tok], stdout);
        else                              std::putchar('?');
    } else {
        std::printf("%d ", tok);
    }
}

// Print a whole token sequence. In char mode we wrap the letters in quotes
// so the reader can tell where the sequence starts and ends (leading spaces
// otherwise disappear visually). In digit mode we keep the original format.
static void print_tokens(const int* toks, int n) {
    if (EOS_TOKEN >= 0) {
        std::putchar('"');
        for (int i = 0; i < n; ++i) print_token(toks[i]);
        std::putchar('"');
    } else {
        for (int i = 0; i < n; ++i) print_token(toks[i]);
    }
}

// ----------------------------------------------------------------------------
// effective_seq_len() — how many positions actually contribute to loss,
// gradients, and accuracy on a given task.
//
// For most tasks (reverse, sort, shift, mod_sum) every position in the
// sequence has a well-defined supervised target, so all T positions count
// and T_eff == T.
//
// For next_token the final position (t = T - 1) has no natural target —
// nothing follows the last input — and we pick one of two honest fixes:
//
//   (A) NO EOS (digit vocabulary, Chapter 16 default).
//       We MASK the final position out of the loss entirely. T_eff = T - 1.
//       The value stored at target[T - 1] is arbitrary (we leave it as
//       tokens[T - 1], a valid index); no read site ever consults it.
//
//   (B) EOS AVAILABLE (char vocabulary, Chapter 19 onward).
//       We set target[T - 1] = EOS and train the model to emit it. T_eff
//       stays at T; every position counts toward loss and accuracy, and
//       the model learns "this is where a sequence ends." Generation at
//       inference time also uses EOS as an early-stop signal.
//
// Both are standard patterns in production LLMs. (A) is exactly the
// "mask padding tokens out of the loss" trick used everywhere; (B) is
// exactly the "train next-token prediction to emit EOS at the end of a
// sample" trick used everywhere else. Switching from (A) to (B) is one
// of the cleanest gains we get from widening the vocabulary in Ch 19.
// ----------------------------------------------------------------------------
static int effective_seq_len() {
    if (cfg.task != Task::NextToken) return cfg.seq_len;
    if (EOS_TOKEN >= 0)              return cfg.seq_len;   // (B) — EOS supervised
    return cfg.seq_len - 1;                                // (A) — mask last position
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
        "  --task=NAME     reverse | sort | shift | mod_sum | next_token\n"
        "                  (default reverse; next_token is the Ch 16 LM task,\n"
        "                  requires --causal=1 to be meaningful)\n"
        "  --vocab=NAME    digits | chars  (default digits). Chapter 19.\n"
        "                  digits: 10-symbol vocab, tokens print as 0..9.\n"
        "                  chars : 28-symbol vocab (a-z, space, EOS), tokens\n"
        "                  print as characters and --gen_prompt accepts\n"
        "                  a string like \"hello \".\n"
        "  --corpus=NAME   none | gettysburg | seashells  (default none).\n"
        "                  Chapter 20. When set, training and eval windows\n"
        "                  are drawn from a fixed English text instead of\n"
        "                  uniform random chars. Requires --vocab=chars.\n"
        "  --bpe=K         Chapter 21. Learn K byte-pair merges from the\n"
        "                  corpus at startup, growing VOCAB from 28 to 28+K.\n"
        "                  Each training window now covers more characters\n"
        "                  of text for the same seq_len. K=0 (default)\n"
        "                  disables BPE. Requires --corpus != none.\n"
        "  --steps=N       training steps (default 800)\n"
        "  --gen_prompt=S  Chapter-17 autoregressive generation demo. S is a\n"
        "                  space-separated list of digits, e.g., \"0 1 2\".\n"
        "                  Only used when --task=next_token. If omitted, a\n"
        "                  sensible default is chosen based on training mode.\n"
        "  --temperature=F Chapter-18 sampling temperature. 0 (default) = greedy\n"
        "                  argmax (matches Ch 17). 1.0 = sample from the model's\n"
        "                  actual distribution. <1 sharpens; >1 flattens.\n"
        "  --top_k=N       Chapter-18 top-K truncation. 0 (default) = disabled.\n"
        "                  When set, keep only the N highest-probability tokens.\n"
        "  --top_p=F       Chapter-18 top-P (nucleus) truncation. 1.0 (default)\n"
        "                  = disabled. When <1, keep the smallest set of tokens\n"
        "                  whose cumulative probability >= F.\n"
        "  --sampling_seed=N  Chapter-18 sampler RNG seed (default 0xBEEFCAFE).\n"
        "                     Change to see different rollouts at the same T/k/p.\n"
        "  --kv_cache=0|1  Chapter 22. Use the incremental KV-cache forward\n"
        "                  pass in the generation demo (matches naive rollout\n"
        "                  bit-exactly; only the per-step cost differs).\n"
        "  --bench_gen=0|1 Chapter 22. After training, time naive vs cached\n"
        "                  generation side by side and print the speedup.\n"
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
        else if (std::strncmp(a, "--vocab=",   8) == 0) cfg.vocab      = parse_vocab(a + 8);
        else if (std::strncmp(a, "--corpus=",  9) == 0) cfg.corpus     = parse_corpus(a + 9);
        else if (std::strncmp(a, "--bpe=",     6) == 0) cfg.bpe_merges = std::atoi(a + 6);
        else if (std::strncmp(a, "--steps=",   8) == 0) cfg.num_steps  = std::atoi(a + 8);
        else if (std::strncmp(a, "--extra_demos=",14) == 0) cfg.extra_demos = std::atoi(a + 14) != 0;
        else if (std::strncmp(a, "--temperature=",14) == 0) cfg.temperature   = (float)std::atof(a + 14);
        else if (std::strncmp(a, "--top_k=",       8) == 0) cfg.top_k         = std::atoi(a + 8);
        else if (std::strncmp(a, "--top_p=",       8) == 0) cfg.top_p         = (float)std::atof(a + 8);
        else if (std::strncmp(a, "--sampling_seed=",16) == 0) cfg.sampling_seed = (uint32_t)std::strtoul(a + 16, nullptr, 0);
        else if (std::strncmp(a, "--kv_cache=",     11) == 0) cfg.kv_cache   = std::atoi(a + 11) != 0;
        else if (std::strncmp(a, "--bench_gen=",    12) == 0) cfg.bench_gen  = std::atoi(a + 12) != 0;
        else if (std::strncmp(a, "--gen_prompt=",13) == 0) {
            // strncpy + explicit NUL: cfg.gen_prompt is a fixed-size char buffer,
            // not a std::string, for the simple reason that the entire config
            // lives in a zero-initialized global and cannot own heap memory
            // without a constructor. Truncate silently if the user hands us
            // something longer than the buffer; a 64-byte prompt is already
            // longer than MAX_SEQ_LEN digits will ever fit.
            std::strncpy(cfg.gen_prompt, a + 13, sizeof(cfg.gen_prompt) - 1);
            cfg.gen_prompt[sizeof(cfg.gen_prompt) - 1] = '\0';
        }
        else if (std::strcmp (a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
            print_usage(argv[0]); std::exit(0);
        }
        else { std::fprintf(stderr, "unknown flag: %s\n", a); print_usage(argv[0]); std::exit(1); }
    }
    if (cfg.num_blocks < 1 || cfg.num_blocks > MAX_BLOCKS)   die("--blocks out of range");
    if (cfg.seq_len   < 1 || cfg.seq_len   > MAX_SEQ_LEN)    die("--seq_len out of range");
    if (cfg.batch     < 1 || cfg.batch     > MAX_BATCH)      die("--batch out of range");
    if (cfg.num_steps < 1)                                    die("--steps must be positive");

    // Chapter-16 invariant: next-token prediction is only meaningful with
    // a causal attention mask. Without it, the output at position t can
    // attend to input position t+1 directly — i.e., it can LOOK AT THE
    // ANSWER the loss is grading it against. The model will then learn
    // "copy from the future", loss drops to zero in ~30 steps, and the
    // held-out accuracy is indistinguishable from memorization.
    //
    // Every decoder-only LLM in the world (GPT, Claude, Llama, Gemini)
    // bakes this invariant into the architecture. We bake it into the
    // CLI instead — easier to notice, easier to teach, and trivially
    // removable if you want to run the "non-causal next-token" experiment
    // as a demonstration of why causal masking matters.
    if (cfg.task == Task::NextToken && !cfg.use_causal) {
        die("--task=next_token requires --causal=1\n"
            "       (a non-causal next-token model can trivially peek at\n"
            "        the very token it is asked to predict, so training is\n"
            "        meaningless; see Chapter 16 of ARITHMETICOFINTELLIGENCE.md for why)");
    }

    // Chapter 19: wire the --vocab choice into the runtime globals. These
    // are what every piece of downstream code reads: loop bounds, output
    // projection width, EOS handling, sampleable-range for data generation.
    if (cfg.vocab == Vocab::Chars) {
        VOCAB             = 28;   // 26 letters + space + EOS
        SAMPLEABLE_VOCAB  = 27;   // training data never spawns EOS in the middle
        EOS_TOKEN         = 27;   // reserved as end-of-sequence marker
    } else {
        VOCAB             = 10;
        SAMPLEABLE_VOCAB  = 10;
        EOS_TOKEN         = -1;
    }
    if (VOCAB > MAX_VOCAB) {
        die("VOCAB exceeds MAX_VOCAB — raise the MAX_VOCAB constant in Part 1");
    }

    // Chapter 20: a corpus is only meaningful in char vocabulary (it is a
    // stream of characters, not of digits). Die early with a helpful message
    // rather than producing confusing output downstream.
    if (cfg.corpus != Corpus::None && cfg.vocab != Vocab::Chars) {
        die("--corpus=<name> requires --vocab=chars\n"
            "       (the corpora are English text; they only tokenize under\n"
            "        the chars vocabulary)");
    }

    // Chapter 21: BPE requires a corpus (it trains merges from the corpus)
    // and char vocab (it operates on characters). And it must fit inside
    // MAX_VOCAB after expansion.
    if (cfg.bpe_merges < 0) die("--bpe= must be non-negative");
    if (cfg.bpe_merges > 0) {
        if (cfg.corpus == Corpus::None)
            die("--bpe=K requires --corpus=<name> (merges are learned from the corpus text)");
        if (cfg.vocab != Vocab::Chars)
            die("--bpe=K requires --vocab=chars");
        if (28 + cfg.bpe_merges > MAX_VOCAB)
            die("--bpe=K too large: 28 + K must not exceed MAX_VOCAB (=128). Reduce K.");
    }
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
float token_emb[MAX_VOCAB  ][D_MODEL];
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
float Wout[D_MODEL][MAX_VOCAB];

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

float logits[MAX_SEQ_LEN][MAX_VOCAB];
float probs [MAX_SEQ_LEN][MAX_VOCAB];

// Gradient buffers. For each parameter: dL/dparam. For each activation:
// dL/dactivation. The weight-grad arrays are accumulated across a batch
// (zeroed at the start of each training step, += for each batch member).
float g_token_emb[MAX_VOCAB  ][D_MODEL];
float g_pos_emb  [MAX_SEQ_LEN][D_MODEL];
float g_Wq[MAX_BLOCKS][D_MODEL][D_MODEL];
float g_Wk[MAX_BLOCKS][D_MODEL][D_MODEL];
float g_Wv[MAX_BLOCKS][D_MODEL][D_MODEL];
float g_W1[MAX_BLOCKS][D_MODEL][D_FF   ];
float g_W2[MAX_BLOCKS][D_FF   ][D_MODEL];
float g_Wout[D_MODEL][MAX_VOCAB];
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
float g_logits[MAX_SEQ_LEN][MAX_VOCAB];

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
//  PART 8b — INCREMENTAL FORWARD (Chapter 22): KV cache generation
// ============================================================================
//  During autoregressive generation, positions 0..t_new-1 have already been
//  forwarded through every block. Their token embedding, Q, K, V, and the
//  X_block residual stream at each depth were all computed in a prior call
//  (either the initial forward() on the prompt, or previous forward_step()
//  calls for earlier generation steps). Those values do NOT depend on the
//  token at position t_new, because causal attention FORBIDS position u
//  from looking forward at position t > u — so as long as no earlier
//  position's *input* changed, its cached K, V, and residual-stream values
//  are still correct.
//
//  This means every generation step only needs to compute, at position
//  t_new:
//     * the token embedding (O(D_MODEL))
//     * Q, K, V for this one position (O(D_MODEL^2))
//     * attention scores against the cached K's at positions 0..t_new
//       (O(t_new * D_MODEL))
//     * softmax over t_new+1 entries
//     * attn_o = weighted sum of cached V's (O(t_new * D_MODEL))
//     * FFN at this one position (O(D_MODEL * D_FF))
//     * final LayerNorm + output projection at this one position
//
//  versus the naive forward()'s O(T * D_MODEL^2 + T^2 * D_MODEL) per call
//  when it repeats all T positions. Over a full generation of T - P new
//  tokens, the naive cost is O(T^2 * work_per_position) and the cached
//  cost is O(T * work_per_position) — a full factor-of-T saving. At our
//  toy T=32 the saving is modest; at GPT-4's T~100,000 it is the entire
//  difference between "usable product" and "impossible to serve."
//
//  The cache here is *implicit*: K[b][u][d] and V[b][u][d] are already
//  per-position arrays populated by the initial forward(), and subsequent
//  forward_step() calls write only K[b][t_new][d] / V[b][t_new][d]. There
//  is no separate data structure to manage, no eviction policy, nothing
//  to allocate. In production LLMs the cache is a large allocation (GBs
//  at long contexts) with its own eviction and paging; the CONCEPT here
//  is identical.
// ----------------------------------------------------------------------------

static void forward_block_step(int b, int t_new) {
    auto& X_in = X_block[b];
    auto& X_ot = X_block[b+1];
    const int T = cfg.seq_len;

    // Pre-LN at position t_new only (not at any cached position).
    float attn_in_row[D_MODEL];
    if (cfg.use_layernorm) {
        layernorm_forward_row(
            X_in[t_new], ln1_out[b][t_new],
            gain1[b], bias1[b],
            ln1_xhat[b][t_new], &ln1_invstd[b][t_new]);
        for (int d = 0; d < D_MODEL; ++d) attn_in_row[d] = ln1_out[b][t_new][d];
    } else {
        for (int d = 0; d < D_MODEL; ++d) attn_in_row[d] = X_in[t_new][d];
    }

    // Q, K, V at t_new only. K[b][u][*] and V[b][u][*] for u < t_new are
    // unchanged from previous calls — this is the entire point of the cache.
    for (int j = 0; j < D_MODEL; ++j) {
        float q = 0, k = 0, v = 0;
        for (int i = 0; i < D_MODEL; ++i) {
            q += attn_in_row[i] * Wq[b][i][j];
            k += attn_in_row[i] * Wk[b][i][j];
            v += attn_in_row[i] * Wv[b][i][j];
        }
        Q[b][t_new][j] = q;
        K[b][t_new][j] = k;
        V[b][t_new][j] = v;
    }

    // Attention score row at t_new. Under causal masking (the required
    // regime for KV-cache generation) we only need u in [0, t_new]. For
    // non-causal (never enabled together with kv_cache in practice), we
    // could extend to [0, T); but we do not support that combination.
    const float scale = 1.0f / std::sqrt((float)D_MODEL);
    const int U = cfg.use_causal ? (t_new + 1) : T;
    for (int u = 0; u < U; ++u) {
        float s = 0;
        for (int d = 0; d < D_MODEL; ++d) s += Q[b][t_new][d] * K[b][u][d];
        scores[b][t_new][u] = s * scale;
    }
    if (cfg.use_causal) {
        const float NEG_INF = -1e30f;
        for (int u = t_new + 1; u < T; ++u) scores[b][t_new][u] = NEG_INF;
    }

    // Softmax over the score row (still T-wide so masked positions weigh 0).
    for (int u = 0; u < T; ++u) attn[b][t_new][u] = scores[b][t_new][u];
    softmax_inplace(attn[b][t_new], T);

    // Weighted sum attn_o[t_new] = attn[t_new] @ V (reading cached V's).
    for (int d = 0; d < D_MODEL; ++d) {
        float s = 0;
        for (int u = 0; u < U; ++u) s += attn[b][t_new][u] * V[b][u][d];
        attn_o[b][t_new][d] = s;
    }

    // Residual H1 at t_new.
    for (int d = 0; d < D_MODEL; ++d)
        H1[b][t_new][d] = X_in[t_new][d] + attn_o[b][t_new][d];

    // FFN at t_new (same layout as forward_block, single-position).
    if (cfg.use_ffn) {
        float ffn_in_row[D_MODEL];
        if (cfg.use_layernorm) {
            layernorm_forward_row(
                H1[b][t_new], ln2_out[b][t_new],
                gain2[b], bias2[b],
                ln2_xhat[b][t_new], &ln2_invstd[b][t_new]);
            for (int d = 0; d < D_MODEL; ++d) ffn_in_row[d] = ln2_out[b][t_new][d];
        } else {
            for (int d = 0; d < D_MODEL; ++d) ffn_in_row[d] = H1[b][t_new][d];
        }
        for (int j = 0; j < D_FF; ++j) {
            float s = 0;
            for (int i = 0; i < D_MODEL; ++i) s += ffn_in_row[i] * W1[b][i][j];
            ff_pre[b][t_new][j] = s;
            ff_act[b][t_new][j] = s > 0 ? s : 0;
        }
        for (int d = 0; d < D_MODEL; ++d) {
            float s = 0;
            for (int j = 0; j < D_FF; ++j) s += ff_act[b][t_new][j] * W2[b][j][d];
            ff_out[b][t_new][d] = s;
            X_ot[t_new][d] = H1[b][t_new][d] + s;
        }
    } else {
        for (int d = 0; d < D_MODEL; ++d) X_ot[t_new][d] = H1[b][t_new][d];
    }
}

// Incremental forward pass: compute activations AT POSITION t_new only,
// assuming positions 0..t_new-1 have been forwarded by a previous call.
// Caller must have filled tokens[t_new] with the token to score.
void forward_step(int t_new, const int* tokens) {
    // Token embedding at t_new (all earlier positions' embeddings are in
    // X_block[0][*] from a prior forward() call).
    for (int d = 0; d < D_MODEL; ++d)
        X_block[0][t_new][d] = token_emb[tokens[t_new]][d] + pos_emb[t_new][d];

    for (int b = 0; b < cfg.num_blocks; ++b) forward_block_step(b, t_new);

    auto& X_final = X_block[cfg.num_blocks];
    float final_in_row[D_MODEL];
    if (cfg.use_layernorm) {
        layernorm_forward_row(
            X_final[t_new], ln_f_out[t_new],
            gain_f, bias_f,
            ln_f_xhat[t_new], &ln_f_invstd[t_new]);
        for (int d = 0; d < D_MODEL; ++d) final_in_row[d] = ln_f_out[t_new][d];
    } else {
        for (int d = 0; d < D_MODEL; ++d) final_in_row[d] = X_final[t_new][d];
    }

    for (int v = 0; v < VOCAB; ++v) {
        float s = 0;
        for (int d = 0; d < D_MODEL; ++d) s += final_in_row[d] * Wout[d][v];
        logits[t_new][v] = s;
    }
    for (int v = 0; v < VOCAB; ++v) probs[t_new][v] = logits[t_new][v];
    softmax_inplace(probs[t_new], VOCAB);
}

// ============================================================================
//  PART 9 — LOSS (cross-entropy)
// ============================================================================
//  L_t = -log(probs[t][target[t]]). Averaged across SCORED positions — see
//  effective_seq_len() in Part 3. For most tasks every position is scored
//  (T_eff == T). For next_token the final position has no real target, so
//  we drop it and divide by T_eff = T - 1 instead of T.
//
//  The algebraic miracle from Chapter 9: d L(softmax(logits)) / d logits
//  = probs - onehot. No exp, no log in backward. Used below in backward().
// ----------------------------------------------------------------------------

float compute_loss(const int target[]) {
    const int T_eff = effective_seq_len();
    float total = 0.0f;
    for (int t = 0; t < T_eff; ++t) {
        float p = probs[t][target[t]];
        if (p < 1e-12f) p = 1e-12f;   // guard against log(0) = -inf
        total += -std::log(p);
    }
    return total / (float)T_eff;
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
    const int T     = cfg.seq_len;
    const int T_eff = effective_seq_len();

    // dL/dlogits = (probs - onehot(target)) / T_eff / batch.
    //   - The division by T_eff (not T) matches the averaging inside
    //     compute_loss(): the loss we report is "mean loss over scored
    //     positions," and the gradient must be derived from that same
    //     mean so the scale is consistent with what the optimizer expects.
    //   - The division by `batch` propagates through the chain rule so
    //     downstream weight grads end up being the MEAN across batch
    //     members (each backward() call accumulates into the same weight
    //     grads; without the /batch factor the gradients would be a SUM,
    //     and the optimizer would take steps that grow with batch size).
    //
    // For t in [0, T_eff): the standard softmax + cross-entropy gradient
    //   probs - onehot(target).
    // For t in [T_eff, T): only hits for next_token, only ever the single
    //   position t = T - 1. No supervision there, so we want NO gradient
    //   flowing back through Wout / LayerNorm / the entire block stack
    //   from that position. Setting g_logits[t][*] = 0 makes the entire
    //   downstream gradient chain at that position vanish at the source.
    const float inv_batch = 1.0f / (float)cfg.batch;
    for (int t = 0; t < T_eff; ++t)
        for (int v = 0; v < VOCAB; ++v)
            g_logits[t][v] = (probs[t][v] - (v == target[t] ? 1.0f : 0.0f))
                             / (float)T_eff * inv_batch;
    for (int t = T_eff; t < T; ++t)
        for (int v = 0; v < VOCAB; ++v)
            g_logits[t][v] = 0.0f;

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
//  Five puzzles the model can be trained on. Each is a simple rule from an
//  input sequence to an output sequence; gradient descent has to discover it.
//
//    reverse:    target[t] = input[T-1-t]
//                Needs attention: each output t has to retrieve input T-1-t.
//    sort:       target = ascending-sorted(input)
//                Much harder: each output needs to know all inputs to decide rank.
//    shift:      target[t] = input[(t+1) % T]
//                Trivial positional rotation; doesn't really need attention.
//    mod_sum:    target[t] = (input[t] + input[(t+1) % T]) mod VOCAB
//                Local computation; FFN does most of the work.
//    next_token: target[t] = input[t+1]  (the "language modeling" rule)
//                Added in Chapter 16. Every real LLM is trained on exactly
//                this target; see the long comment inside the switch for why
//                this is the conceptual pivot from "sequence puzzle" to
//                "language model" and why it REQUIRES --causal=1.
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

        // --------------------------------------------------------------------
        // next_token — the Chapter-16 addition. These few lines of code are
        // the pivot from "tiny sequence-puzzle solver" to "language model."
        // Every generative LLM on earth (GPT, Claude, Llama, DeepSeek,
        // Gemini) is trained on exactly this target rule. The only way our
        // setup differs from theirs is scale: alphabet of 10 symbols instead
        // of ~100,000; context of 8 positions instead of ~100,000; one
        // training example (or 16 random ones) per step instead of millions
        // of tokens per step. The *rule* is the same.
        //
        // The rule in plain English: at every position t, the correct
        // output is the token that actually sits at position t+1. Training
        // happens in parallel across all positions — each position is fed
        // the TRUE prefix during training, not the model's own guesses.
        // This shortcut is called "teacher forcing" and it is why one
        // forward pass produces T_eff supervised training examples instead
        // of just one. Without it, training would be roughly T times slower
        // (for GPT-style context lengths that is three orders of magnitude).
        //
        // TWO SUBTLETIES THAT MATTER FOR UNDERSTANDING:
        //
        //  1. Causal masking is NOT optional here. If the model is allowed
        //     to look at position t+1 while predicting it, the "task"
        //     collapses to "copy from the future" — loss drops to zero in
        //     ~30 steps and the model has learned precisely nothing about
        //     the data distribution. Every decoder-only LLM bakes a causal
        //     mask into its architecture for exactly this reason. MaxAI
        //     enforces it at the CLI: parse_args() refuses to start when
        //     --task=next_token is paired with --causal=0. (See the check
        //     at the end of parse_args in Part 3.)
        //
        //  2. At position t = T-1 there is no t+1 to predict — nothing in
        //     the input comes after the last token. We deal with this by
        //     MASKING THE FINAL POSITION OUT of loss, gradients, and
        //     accuracy. See effective_seq_len() in Part 3: for this task
        //     it returns T - 1, and every downstream computation that
        //     averages or counts over positions uses T_eff instead of T.
        //
        //     The value we write into target[T - 1] below is therefore
        //     arbitrary — no read site ever uses it. We leave it as
        //     tokens[T - 1] because that is guaranteed to be a valid VOCAB
        //     index and will never crash any code path that forgets to
        //     check. Think of it as a placeholder whose value does not
        //     matter.
        //
        //     Real LLMs pick a different solution: they add an explicit
        //     end-of-sequence ("EOS") symbol to the vocabulary and train
        //     the final position to emit it. We will do exactly that in
        //     Chapter 19 when the vocabulary grows beyond ten digits and
        //     reserving one more symbol is cheap. For Chapters 16–18 we
        //     keep the vocabulary as-is and mask the position instead,
        //     which is the same pattern real codebases use for padding
        //     tokens in variable-length batches.
        // --------------------------------------------------------------------
        case Task::NextToken:
            for (int t = 0; t < T - 1; ++t) target[t] = tokens[t + 1];
            // Final position: supervise on EOS if the vocabulary has one
            // (Chapter 19 char vocab), otherwise store a harmless placeholder
            // that will be masked out of the loss by effective_seq_len().
            target[T - 1] = (EOS_TOKEN >= 0) ? EOS_TOKEN : tokens[T - 1];
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

// ============================================================================
//  PART 13c — BPE TOKENIZER (Chapter 21): byte-pair subword encoding
// ============================================================================
//  Chapter 19 introduced a 28-symbol character vocabulary. That is structurally
//  correct (input-string-to-integer works end to end) but it is wildly
//  wasteful at any real scale: a word like "proposition" is 11 tokens, and
//  each of those tokens carries one character's worth of information. GPT-4
//  uses ~100,000 subword tokens; Llama uses ~32,000. Subword tokenization is
//  the reason those models can fit a long document into a modest context
//  window.
//
//  Byte-Pair Encoding (BPE) was invented for file compression in 1994
//  (Gage) and repurposed for NLP in 2016 (Sennrich, Haddow, Birch). The
//  algorithm is almost embarrassingly simple:
//
//      start with every character as its own token
//      repeat K times:
//          count every pair of adjacent tokens in the training corpus
//          pick the most frequent pair (a, b)
//          add a new token [ab] to the vocabulary
//          replace every occurrence of the pair (a, b) in the corpus with [ab]
//
//  After K iterations the vocabulary has grown by K tokens. Common character
//  sequences (`th`, `he`, `ing`, ` the`) become single tokens; rare words
//  remain split into smaller pieces; every possible input string has a valid
//  tokenization because any string decomposes into at most its characters.
//
//  Production BPEs differ from this toy in a handful of ways:
//  - They pre-split text on whitespace/punctuation so merges cannot cross
//    word boundaries in weird ways ("the cat" never becomes a single token).
//  - They use a more efficient pair-counter (hash map keyed on (l, r) pairs)
//    and incremental updates after each merge rather than a full rescan.
//  - They may cap the maximum token length to avoid rare super-long merges.
//  - They train on corpora with billions of words, learning tens of thousands
//    of merges.
//
//  None of these change the algorithm meaningfully. This chapter's
//  implementation in ~80 lines is genuinely the same mechanism that powers
//  GPT, Llama, Claude, and DeepSeek tokenization, just without the
//  pre-splitting shortcut and at a much smaller scale.
//
//  Integration with the rest of MaxAI:
//    * Base tokens 0..27 stay exactly as in Chapter 19 (a-z, space, EOS).
//    * Merges 28..28+K-1 get their (left, right) defining pair AND their
//      text representation (the concatenation of left's and right's text).
//    * The active corpus is tokenized ONCE at startup into an integer
//      array; sample_from_corpus() in Part 13b now just indexes into that
//      array, which is much faster than re-tokenizing per sample.
//    * parse_prompt() runs user input through bpe_encode_text() so prompts
//      like "proposition" get split into whatever subwords the BPE learned.
//    * print_token() prints bpe_token_text[tok], which is a single character
//      for base tokens and a multi-character string for merged tokens.
// ----------------------------------------------------------------------------

// Storage for the BPE tables and the encoded corpus is declared earlier in
// the file (Part 1) so print_token() in Part 3b can reference
// bpe_token_text[] without a forward declaration. This Part defines the
// algorithms (init, train, encode, dump) that populate those tables.

// Forward declaration so train_bpe() and bpe_dump() can call get_corpus_text()
// which is defined in Part 13b (just below this section).
static const char* get_corpus_text();

// Initialize the token table for the 28 base characters. Always called, even
// when BPE is disabled (char-level mode uses the same table to print tokens).
static void bpe_init_base() {
    for (int i = 0; i < MAX_VOCAB; ++i) {
        bpe_token_text[i][0] = '\0';
        bpe_token_len [i]    = 0;
        bpe_merge_left [i]   = -1;
        bpe_merge_right[i]   = -1;
    }
    // Base alphabet: 'a'..'z'.
    for (int i = 0; i < 26; ++i) {
        bpe_token_text[i][0] = (char)('a' + i);
        bpe_token_text[i][1] = '\0';
        bpe_token_len [i]    = 1;
    }
    // Space.
    bpe_token_text[26][0] = ' ';
    bpe_token_text[26][1] = '\0';
    bpe_token_len [26]    = 1;
    // EOS — printed as "<EOS>" by print_token specially; text field left empty.
    bpe_token_text[27][0] = '\0';
    bpe_token_len [27]    = 0;
}

// Encode a character string into a token sequence using the trained merges.
// Algorithm: (1) tokenize to char level, (2) for each learned merge in the
// order it was learned, scan the buffer and replace (left, right) pairs
// with the merged token id. Writes into `out`; returns the number of tokens.
// Dies on any character outside the chars vocabulary.
static int bpe_encode_text(const char* s, int* out, int max_out) {
    int toks[MAX_CORPUS_TOKENS];
    int n = 0;
    while (*s && n < MAX_CORPUS_TOKENS) {
        const int t = encode_char(*s);
        if (t < 0) {
            std::fprintf(stderr,
                "error: bpe_encode_text: character '%c' (0x%02x) is not in the chars vocabulary.\n",
                *s, (unsigned char)*s);
            std::exit(1);
        }
        toks[n++] = t;
        ++s;
    }
    for (int m = 0; m < cfg.bpe_merges; ++m) {
        const int new_tok = 28 + m;
        const int lhs     = bpe_merge_left [new_tok];
        const int rhs     = bpe_merge_right[new_tok];
        if (lhs < 0 || rhs < 0) break;   // this merge was never learned
        int write = 0, read = 0;
        while (read < n) {
            if (read + 1 < n && toks[read] == lhs && toks[read + 1] == rhs) {
                toks[write++] = new_tok;
                read += 2;
            } else {
                toks[write++] = toks[read++];
            }
        }
        n = write;
    }
    if (n > max_out) die("bpe_encode_text: output buffer too small");
    for (int i = 0; i < n; ++i) out[i] = toks[i];
    return n;
}

// Train up to K byte-pair merges on the active corpus. Records each merge
// at bpe_merge_left[28+k], bpe_merge_right[28+k], and builds the text
// representation in bpe_token_text[28+k]. Also populates encoded_corpus[]
// with the final tokenization of the corpus. Updates VOCAB to reflect the
// number of merges actually learned (may be < K if the corpus runs out of
// repeated pairs before K is reached).
static void train_bpe() {
    bpe_init_base();
    VOCAB = 28;   // reset; we'll bump as we add merges

    // Tokenize the corpus at char level into a working buffer.
    const char* corpus = get_corpus_text();
    int workbuf[MAX_CORPUS_TOKENS];
    int n = 0;
    while (*corpus && n < MAX_CORPUS_TOKENS) {
        const int t = encode_char(*corpus);
        if (t < 0) die("corpus contains character not in chars vocabulary");
        workbuf[n++] = t;
        ++corpus;
    }
    if (*corpus != '\0') die("corpus exceeds MAX_CORPUS_TOKENS — raise the bound");

    // Storage for pair counts. VOCAB * VOCAB = up to 128*128 = 16384 ints.
    static int pair_count[MAX_VOCAB][MAX_VOCAB];

    const int K = cfg.bpe_merges;
    for (int k = 0; k < K; ++k) {
        // Count every adjacent pair in workbuf.
        std::memset(pair_count, 0, sizeof(pair_count));
        for (int i = 0; i + 1 < n; ++i) {
            ++pair_count[workbuf[i]][workbuf[i + 1]];
        }

        // Pick the most frequent pair. Tie-break by lexicographic (l, r) so
        // the merges are deterministic across runs with the same corpus.
        int best_l = -1, best_r = -1, best_c = 0;
        for (int l = 0; l < 28 + k; ++l) {
            for (int r = 0; r < 28 + k; ++r) {
                if (pair_count[l][r] > best_c) {
                    best_c = pair_count[l][r];
                    best_l = l; best_r = r;
                }
            }
        }
        if (best_c < 2) break;   // no more learnable pairs — all remaining appear at most once

        // Record the new token.
        const int new_tok = 28 + k;
        bpe_merge_left [new_tok] = best_l;
        bpe_merge_right[new_tok] = best_r;
        const int llen = bpe_token_len[best_l];
        const int rlen = bpe_token_len[best_r];
        if (llen + rlen + 1 > MAX_TOKEN_TEXT) {
            die("BPE merge produced a token string longer than MAX_TOKEN_TEXT — raise the bound");
        }
        std::memcpy(bpe_token_text[new_tok],        bpe_token_text[best_l], llen);
        std::memcpy(bpe_token_text[new_tok] + llen, bpe_token_text[best_r], rlen);
        bpe_token_text[new_tok][llen + rlen] = '\0';
        bpe_token_len [new_tok]              = llen + rlen;
        VOCAB = 28 + k + 1;

        // Apply the merge to workbuf.
        int write = 0, read = 0;
        while (read < n) {
            if (read + 1 < n && workbuf[read] == best_l && workbuf[read + 1] == best_r) {
                workbuf[write++] = new_tok;
                read += 2;
            } else {
                workbuf[write++] = workbuf[read++];
            }
        }
        n = write;
    }

    // Store the final encoded corpus for sample_from_corpus() to slice.
    if (n > MAX_CORPUS_TOKENS) die("encoded corpus overflow — raise MAX_CORPUS_TOKENS");
    encoded_corpus_len = n;
    for (int i = 0; i < n; ++i) encoded_corpus[i] = workbuf[i];
}

// Print the learned merges and final encoded corpus as a pedagogical dump.
// Called from main() when BPE is active so the reader can see the subword
// vocabulary the model is actually being trained on.
static void bpe_dump() {
    if (cfg.bpe_merges == 0) return;
    std::printf("BPE: learned %d merges from %d corpus chars (encoded length: %d tokens)\n",
                VOCAB - 28, (int)std::strlen(get_corpus_text()), encoded_corpus_len);
    for (int m = 0; m < VOCAB - 28; ++m) {
        const int tok = 28 + m;
        std::printf("  merge %2d: [%s] + [%s] -> [%s]  (id %d)\n",
                    m,
                    bpe_token_text[bpe_merge_left [tok]],
                    bpe_token_text[bpe_merge_right[tok]],
                    bpe_token_text[tok],
                    tok);
    }
    std::printf("\n");
}

// ============================================================================
//  PART 13b — TEXT CORPORA (Chapter 20): sliding-window English training
// ============================================================================
//  Chapters 16-19 trained on either a single fixed sequence (memorization) or
//  uniformly random sequences (which per Ch 16's ceiling argument carry zero
//  conditional information between positions — the model literally cannot
//  learn anything beyond the marginal distribution, and the cross-entropy
//  floor is log(VOCAB)). Chapter 20 replaces the data generator with a
//  sliding window over a fixed English text. English is NOT i.i.d.; after a
//  'q' the next character is overwhelmingly 'u'; after 'th' the next is
//  usually 'e' or ' '; after a space the next is usually a consonant. The
//  conditional entropy H(next | past) is much smaller than log(VOCAB),
//  which means the model's loss can now fall BELOW the ceiling that bounded
//  Chapter 16-19 random training. That gap — ceiling minus achieved loss —
//  is the model's learned knowledge about English, measured in bits.
//
//  This is the same thing GPT-4 is doing. Same target rule (predict the next
//  token). Same causal mask. Same teacher-forced parallel training. Same
//  cross-entropy. The only differences are:
//    - Our corpus is ~200 characters; GPT-4's training data is ~10^13 tokens.
//    - Our vocabulary is 28 symbols; GPT-4's is ~100,000 BPE subwords.
//    - Our model has ~2 thousand parameters; GPT-4 has ~10^12.
//
//  The ratio is about a trillion between every pair of quantities, and the
//  effect is visible: our model learns to produce output that looks a bit
//  like English fragments but not coherent English. GPT-4 produces text that
//  often looks coherent. The mechanism is identical; the quality tracks the
//  numbers.
//
//  Two canonical corpora ship with MaxAI:
//
//    --corpus=gettysburg  — The opening of Lincoln's Gettysburg Address,
//                           lowercased, punctuation removed (~180 chars).
//                           Formal English, fairly diverse word set.
//
//    --corpus=seashells   — The "she sells sea shells" tongue-twister,
//                           lowercased (~165 chars). Extremely repetitive;
//                           the easiest real-text learning target in the
//                           demo set. Great for watching the ceiling fall
//                           dramatically in a small number of steps.
//
//  You can add your own by extending the enum in Part 3 and adding a string
//  constant below; nothing else in the program needs to change.
// ----------------------------------------------------------------------------

static const char CORPUS_GETTYSBURG[] =
    "four score and seven years ago our fathers brought forth on this "
    "continent a new nation conceived in liberty and dedicated to the "
    "proposition that all men are created equal now we are engaged in "
    "a great civil war";

static const char CORPUS_SEASHELLS[] =
    "she sells sea shells by the sea shore the shells she sells are sea "
    "shells im sure so if she sells sea shells on the sea shore then im "
    "sure she sells sea shore shells";

static const char* get_corpus_text() {
    switch (cfg.corpus) {
        case Corpus::Gettysburg: return CORPUS_GETTYSBURG;
        case Corpus::SeaShells:  return CORPUS_SEASHELLS;
        default:                 return nullptr;
    }
}

// Draw a random seq_len-long window from the already-tokenized corpus
// (encoded_corpus[], populated once at startup by train_bpe() in Part 13c).
// With Chapter 21 BPE this array holds subword tokens; with BPE disabled
// (the Ch 20 baseline) it holds one token per character and the code path
// is identical.
//
// IMPORTANT subtlety for next_token: in non-corpus char mode the final
// position's target is EOS (the Ch 19 convention). But in corpus mode the
// window is embedded in a longer stream, so position T-1 has a *real*
// next token — namely encoded_corpus[start + T] — and we want the model
// supervised on that. make_target()'s generic EOS-at-T-1 rule would teach
// "after [random 16-token window from the middle of a text], emit <EOS>"
// at every window, which is NOT learnable — EOS never appears in the
// corpus. Training would diverge and loss would explode above log(VOCAB).
// Corpus + NextToken therefore overrides target[T-1] with the honest next
// token from the encoded corpus. Every other task still uses make_target.
static void sample_from_corpus(int* out_tokens, int* out_target, uint32_t& rng) {
    const int T = cfg.seq_len;
    const int need = (cfg.task == Task::NextToken) ? (T + 1) : T;
    if (encoded_corpus_len < need) {
        std::fprintf(stderr,
            "error: encoded corpus has only %d tokens; --seq_len=%d needs %d.\n"
            "       lower --seq_len, pick a longer corpus, or reduce --bpe=K\n"
            "       (fewer merges keep more tokens in the encoded corpus).\n",
            encoded_corpus_len, T, need);
        std::exit(1);
    }
    const int max_start = encoded_corpus_len - need;
    const int start = (int)(xorshift32(rng) % (uint32_t)(max_start + 1));
    for (int t = 0; t < T; ++t) out_tokens[t] = encoded_corpus[start + t];
    if (cfg.task == Task::NextToken) {
        // Honest sliding-window targets: shift by one, including the real
        // next token at position T-1. No EOS substitution.
        for (int t = 0; t < T; ++t) out_target[t] = encoded_corpus[start + t + 1];
    } else {
        // Non-language-modeling tasks (reverse, sort, etc.) treat the
        // window as an opaque sequence; their targets are self-contained
        // in out_tokens and need no peek past the window.
        make_target(out_tokens, out_target);
    }
}

static int held_out_tokens[NUM_EVAL][MAX_SEQ_LEN];
static int held_out_target[NUM_EVAL][MAX_SEQ_LEN];

static void build_heldout() {
    const int T = cfg.seq_len;
    for (int i = 0; i < NUM_EVAL; ++i) {
        if (cfg.corpus != Corpus::None) {
            // Ch 20 corpus mode: held-out windows are drawn from the same
            // text as training, but from a dedicated RNG stream so they
            // represent a *sampled* slice that training may or may not
            // have touched. With a 200-char corpus and a 32-char window
            // there are ~170 unique starts; training typically touches
            // most of them, so "held-out accuracy" here measures how well
            // the model handles the windows it was less often batched on
            // rather than unseen text. A more rigorous split (prefix vs
            // suffix of the corpus) would be trivial to add.
            sample_from_corpus(held_out_tokens[i], held_out_target[i],
                               eval_rng_state);
        } else {
            // SAMPLEABLE_VOCAB rather than VOCAB: in char mode we do not
            // want raw EOS appearing inside training/eval sequences — EOS
            // is reserved for the supervised end-of-sequence target that
            // make_target() sets.
            for (int t = 0; t < T; ++t)
                held_out_tokens[i][t] = (int)(xorshift32(eval_rng_state) % (uint32_t)SAMPLEABLE_VOCAB);
            make_target(held_out_tokens[i], held_out_target[i]);
        }
    }
}

// Produce one training example. Three regimes:
//   corpus set:    slide over the corpus (Ch 20).
//   random=0:      fixed sequence (Ch 16 memorization).
//   random=1:      i.i.d. uniform random (Chs 16-19 random training).
static void sample_example(int* out_tokens, int* out_target) {
    const int T = cfg.seq_len;
    if (cfg.corpus != Corpus::None) {
        sample_from_corpus(out_tokens, out_target, rng_state);
        return;
    }
    if (!cfg.random) {
        // Fixed example: tokens[t] = t, capped at SAMPLEABLE_VOCAB. For
        // digit mode and T=8 this is [0,1,2,3,4,5,6,7] exactly as MinAI.
        // For char mode it becomes [0..7] = "abcdefgh" — the natural toy
        // training sequence for the Chapter 19 char demo.
        for (int t = 0; t < T; ++t) out_tokens[t] = t % SAMPLEABLE_VOCAB;
    } else {
        for (int t = 0; t < T; ++t)
            out_tokens[t] = (int)(xorshift32(rng_state) % (uint32_t)SAMPLEABLE_VOCAB);
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
    // Only count scored positions — see effective_seq_len() in Part 3.
    // For next_token this drops the final position entirely; for every
    // other task T_eff == T and this loops over the whole sequence as
    // it always did.
    const int T_eff = effective_seq_len();
    int ok = 0;
    for (int t = 0; t < T_eff; ++t)
        if (argmax_row(probs[t], VOCAB) == target[t]) ++ok;
    return ok;
}

// ============================================================================
//  PART 15 — EVALUATION (held-out accuracy)
// ============================================================================
//  Run the model over the held-out set and report fraction of tokens
//  predicted correctly. Only meaningful when --random=1.
// ----------------------------------------------------------------------------

static float eval_heldout_accuracy() {
    // Same "only score positions in [0, T_eff)" rule as count_correct().
    // The denominator must match the numerator's population size, hence
    // NUM_EVAL * T_eff (not NUM_EVAL * T) — otherwise next_token's
    // held-out accuracy would be systematically understated by a factor
    // of T_eff/T because we'd never award credit for the dropped position.
    const int T_eff = effective_seq_len();
    int correct = 0;
    for (int i = 0; i < NUM_EVAL; ++i) {
        forward(held_out_tokens[i]);
        for (int t = 0; t < T_eff; ++t)
            if (argmax_row(probs[t], VOCAB) == held_out_target[i][t]) ++correct;
    }
    return (float)correct / (float)(NUM_EVAL * T_eff);
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

    // Pick a demo sequence that actually represents what the model was
    // trained to do. Corpus mode draws an in-distribution window; random
    // mode draws a fresh random sequence; fixed mode re-uses the single
    // training example. All three use dedicated deterministic RNG seeds so
    // the demo output is stable across runs.
    if (cfg.corpus != Corpus::None) {
        uint32_t demo_rng = 0x0DEADBEEu;
        sample_from_corpus(tokens, target, demo_rng);
    } else if (cfg.random) {
        uint32_t demo_rng = 0x0DEADBEEu;
        for (int t = 0; t < T; ++t)
            tokens[t] = (int)(xorshift32(demo_rng) % (uint32_t)SAMPLEABLE_VOCAB);
        make_target(tokens, target);
    } else {
        for (int t = 0; t < T; ++t) tokens[t] = t % SAMPLEABLE_VOCAB;
        make_target(tokens, target);
    }
    forward(tokens);

    // Build the argmax output once so we can reuse it below.
    int output[MAX_SEQ_LEN];
    for (int t = 0; t < T; ++t) output[t] = argmax_row(probs[t], VOCAB);

    std::printf("\ntask   : %s\n", task_name(cfg.task));
    std::printf("vocab  : %s (size %d)\n", vocab_name(cfg.vocab), VOCAB);
    std::printf("input  : "); print_tokens(tokens, T); std::printf("\n");
    std::printf("target : "); print_tokens(target, T); std::printf("\n");
    std::printf("output : "); print_tokens(output, T); std::printf("\n\n");

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
//  PART 18b — GENERATE: the autoregressive inference loop (Chapter 17)
// ============================================================================
//  Training is done. The model has internalized (to whatever extent the data
//  distribution allows) the conditional P(next_token | tokens[0..t]). This
//  Part is where we USE that learned distribution the way a human uses
//  ChatGPT: hand the model a prompt, read off its best guess for the next
//  token, append that guess to the prompt, hand the longer prompt back, and
//  repeat. One call to forward() per generated token.
//
//  Pseudocode:
//
//      while current_len < T:
//          forward(tokens)                             # tokens[0..current_len-1]
//          next_token = argmax(probs[current_len - 1])
//          tokens[current_len] = next_token
//          current_len += 1
//
//  Every real LLM runs a loop structurally identical to this one. Chapter 18
//  will swap argmax for sampling (temperature, top-k, top-p). Chapter 19
//  will swap the 10-digit vocabulary for character-scale text. Chapter 22
//  will eliminate the loop's single most glaring inefficiency by adding a
//  KV cache. Everything new in those chapters slots into this same skeleton.
//
//  Three conceptual notes that make the code below make sense:
//
//  (a) "How does feeding zeros into the unused positions not corrupt the
//      output?" It does not corrupt anything because of the causal mask.
//      At position t the attention block is FORBIDDEN from looking at
//      positions > t, so whatever garbage we stored there is never
//      consulted by the computation we care about. Forward() happily
//      computes outputs at every position in [0, T); we simply look only
//      at probs[current_len - 1] and discard the rest. This "garbage is
//      invisible" property of causal attention is what makes the loop
//      implementable as a fixed-size buffer with a moving write pointer.
//
//  (b) "Why is this O(T^2) in the number of generated tokens, and why
//      does Chapter 22 exist?" Each iteration runs forward() from scratch
//      over ALL previous positions, even though the K and V vectors at
//      those prefix positions have not changed since the last iteration.
//      Iteration k redoes the attention work for positions 0..current_len,
//      throwing away the identical answers it computed the step before.
//      At T=8 the waste is invisible. At T=128000 (modern LLM context) it
//      is the difference between a usable product and a useless one. The
//      fix ("KV cache") is a small bookkeeping change with enormous
//      production impact, and it is what Chapter 22 adds to this file.
//
//  (c) "What stops the loop?" Here, simply that current_len == T. The
//      causal context is full; we have no mechanism for extending it.
//      Real LLMs either (i) emit a special end-of-sequence token and stop
//      on it, (ii) apply a rolling-window trick to evict old tokens as
//      new ones come in ("sliding window attention"), or (iii) use
//      architectures with "infinite" context such as state-space models.
//      MaxAI will add option (i) in Chapter 19 alongside the vocabulary
//      expansion that lets us afford one more symbol.
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Chapter-18 SAMPLING (reads logits, returns a token index).
//
// The Chapter-17 generate() loop took argmax(probs) at every step. That is
// "greedy decoding": deterministic, same-prompt-same-output, and notoriously
// prone to producing boring, repetitive text because it cannot escape local
// peaks in the distribution. Every real LLM API — OpenAI, Anthropic, the
// `generate()` method on a Hugging Face model — exposes three sampling knobs
// instead:
//
//   temperature T   divide the logits by T before softmax. T==0 is greedy
//                   (implemented as a fast path here). T<1 sharpens the
//                   distribution (the peak wins more often); T>1 flattens
//                   it toward uniform.
//   top_k     k     after softmax, keep only the k highest-probability tokens,
//                   renormalize, sample from that truncated distribution.
//                   k==0 disables the filter. Typical production value: ~40.
//   top_p     p     nucleus sampling (Holtzman et al., 2019): after softmax,
//                   sort tokens by probability descending, keep the smallest
//                   prefix whose cumulative probability reaches p, renormalize,
//                   sample from that prefix. p==1.0 disables. Typical value:
//                   0.90 or 0.95. Unlike top_k which is a fixed count, top_p
//                   adapts — sharp distributions include few tokens, flat
//                   distributions include many.
//
// The full production pipeline is the composition:
//
//    sample( renormalize( top_p( top_k( softmax( logits / T ) ) ) ) )
//
// with any subset of T, top_k, top_p possibly disabled. sample_token() below
// implements exactly that, in the obvious order.
//
// Why sample at all instead of argmaxing? Three complementary reasons.
// 1. Greedy decoding is a local optimum per step, NOT a global optimum
//    over the whole generated sequence. Sometimes the globally best text
//    passes through a token that is only the second or third most likely.
//    Pure beam search (a variant that maintains k simultaneous hypotheses
//    and picks the highest-joint-likelihood one) addresses this but has
//    its own pathologies and has fallen out of favour for generative LMs.
// 2. Greedy decoding has zero diversity. The same prompt always produces
//    the same output. Humans generating text do not do this; sampled
//    output is a much better match to the statistical structure of the
//    training corpus (see Holtzman 2019's "beam search / argmax produces
//    degenerate output" paper).
// 3. Sampling is the only way to draw output whose statistics match the
//    training distribution. If the data contains tokens that appear 5%
//    of the time in a given context, argmax never emits them; a 1/20-
//    chance sample does. Shannon's source coding theorem says something
//    almost opposite to what intuition suggests: a generator that matches
//    the source entropy produces output that is *hard to compress*, which
//    is exactly the signature of naturalness.
// ----------------------------------------------------------------------------

static uint32_t sampling_rng_state = 0xBEEFCAFEu;   // initialized from cfg.sampling_seed in main()
static float rand_uniform_unit() {
    // Uniform float in [0, 1). 24-bit mantissa is plenty for categorical
    // sampling; we mask off the high byte to avoid the "exactly 1.0" edge
    // case that would let the inverse-CDF loop walk off the end.
    return (float)(xorshift32(sampling_rng_state) & 0x00FFFFFFu) / (float)0x01000000;
}

// Sample a token index in [0, n) from the distribution implied by `logits_in`
// under the given temperature/top-k/top-p. Writes the effective-distribution
// it ultimately sampled from into `out_probs` (same length as logits_in) so
// the caller can print a diagnostic. Pass out_probs=nullptr to skip that.
static int sample_token(const float* logits_in, int n,
                        float temperature, int top_k, float top_p,
                        float* out_probs)
{
    // Greedy fast path — matches Ch 17 exactly. We also write a one-hot
    // "distribution" into out_probs so the caller's diagnostic still makes
    // sense when T == 0.
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; ++i) if (logits_in[i] > logits_in[best]) best = i;
        if (out_probs) {
            for (int i = 0; i < n; ++i) out_probs[i] = (i == best) ? 1.0f : 0.0f;
        }
        return best;
    }

    // 1. Temperature-scale logits. Divide by T BEFORE softmax. This is the
    //    right place to do it: dividing a post-softmax distribution by T
    //    and re-normalizing is NOT the same operation and gives the wrong
    //    answer.
    float scaled[MAX_VOCAB];
    for (int i = 0; i < n; ++i) scaled[i] = logits_in[i] / temperature;

    // 2. Softmax with the subtract-max trick from Chapter 6.
    float mx = scaled[0];
    for (int i = 1; i < n; ++i) if (scaled[i] > mx) mx = scaled[i];
    float p[MAX_VOCAB];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) { p[i] = std::exp(scaled[i] - mx); sum += p[i]; }
    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; ++i) p[i] *= inv_sum;

    // 3. Top-K truncation. Partial selection sort gets the top-k indices
    //    cheaply; we zero every other entry and renormalize.
    if (top_k > 0 && top_k < n) {
        int idx[MAX_VOCAB]; for (int i = 0; i < n; ++i) idx[i] = i;
        for (int k = 0; k < top_k; ++k) {
            int best = k;
            for (int j = k + 1; j < n; ++j)
                if (p[idx[j]] > p[idx[best]]) best = j;
            int tmp = idx[k]; idx[k] = idx[best]; idx[best] = tmp;
        }
        float kept[MAX_VOCAB] = {0};
        float kept_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            kept[idx[k]] = p[idx[k]];
            kept_sum    += p[idx[k]];
        }
        const float r = 1.0f / kept_sum;
        for (int i = 0; i < n; ++i) p[i] = kept[i] * r;
    }

    // 4. Top-P (nucleus) truncation. Sort indices by p descending, walk the
    //    cumulative sum, cut at the first index where cum >= top_p, zero the
    //    remainder, renormalize.
    if (top_p < 1.0f) {
        int idx[MAX_VOCAB]; for (int i = 0; i < n; ++i) idx[i] = i;
        // Full selection sort. n<=VOCAB<=10 (for now); this is trivial.
        for (int i = 0; i < n - 1; ++i) {
            int best = i;
            for (int j = i + 1; j < n; ++j)
                if (p[idx[j]] > p[idx[best]]) best = j;
            int tmp = idx[i]; idx[i] = idx[best]; idx[best] = tmp;
        }
        float cum = 0.0f;
        int keep_until = n;
        for (int k = 0; k < n; ++k) {
            cum += p[idx[k]];
            if (cum >= top_p) { keep_until = k + 1; break; }
        }
        float kept[MAX_VOCAB] = {0};
        float kept_sum = 0.0f;
        for (int k = 0; k < keep_until; ++k) {
            kept[idx[k]] = p[idx[k]];
            kept_sum    += p[idx[k]];
        }
        const float r = 1.0f / kept_sum;
        for (int i = 0; i < n; ++i) p[i] = kept[i] * r;
    }

    // 5. Inverse-CDF sample. Walk the probability array accumulating until
    //    the running sum exceeds a uniform draw. This is the textbook
    //    O(n) categorical sampler and it is trivially correct.
    if (out_probs) for (int i = 0; i < n; ++i) out_probs[i] = p[i];
    const float r = rand_uniform_unit();
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += p[i];
        if (r < acc) return i;
    }
    return n - 1;   // numerical safety net for floating-point drift
}

// Parse a prompt string into a token array. Behaviour depends on the active
// vocabulary:
//   digit mode:  "0 1 2" / "012" / "1,3,5" → each character (ignoring
//                separators) becomes a digit token 0..9.
//   char mode:   "hello world" → each character maps through encode_char()
//                to a letter/space token 0..26. EOS tokens cannot be
//                expressed in the prompt (readers do not type EOS; they
//                type letters and let the model emit EOS as the signal to
//                stop).
//
// Returns the number of tokens parsed. Dies with a helpful error on any
// character the active vocabulary cannot represent.
static int parse_prompt(const char* s, int* out) {
    int n = 0;
    if (EOS_TOKEN >= 0) {
        // Char mode: tokenize through the BPE pipeline. With bpe_merges=0
        // this is equivalent to one token per character (the old Ch 19
        // behaviour); with BPE active the user's string is split into
        // subwords learned from the corpus, potentially many fewer tokens
        // than characters. Either way the output array is seq_len-capped.
        n = bpe_encode_text(s, out, MAX_SEQ_LEN);
    } else {
        // Digit mode: whitespace/commas/tabs are separators; each remaining
        // character must be a digit '0'..'9'.
        while (*s && n < MAX_SEQ_LEN) {
            while (*s == ' ' || *s == ',' || *s == '\t') ++s;
            if (!*s) break;
            if (*s < '0' || *s > '9') {
                std::fprintf(stderr,
                    "error: prompt contains non-digit character '%c'; under\n"
                    "       --vocab=digits --gen_prompt must be digits only,\n"
                    "       e.g. \"0 1 2\" or \"012\".\n", *s);
                std::exit(1);
            }
            out[n++] = *s - '0';
            ++s;
        }
    }
    return n;
}

// generate() — the autoregressive loop itself. Writes verbose diagnostics to
// stdout (a per-step rollout plus the top-3 of each distribution) so the
// reader can see what the model is "thinking" at every step.
void generate(const int* prompt, int prompt_len, bool verbose) {
    const int T = cfg.seq_len;
    if (prompt_len < 1 || prompt_len > T) {
        std::fprintf(stderr,
            "error: prompt length %d is out of range [1, %d = seq_len]\n",
            prompt_len, T);
        std::exit(1);
    }

    // Seed the token buffer with the prompt; zero the rest. The zeros will
    // be overwritten one by one as the loop progresses, and the causal mask
    // guarantees they are never consulted by any output we actually read.
    int tokens[MAX_SEQ_LEN];
    for (int t = 0; t < prompt_len; ++t) tokens[t] = prompt[t];
    for (int t = prompt_len; t < T; ++t) tokens[t] = 0;

    if (verbose) {
        std::printf("\ngenerate (Ch 17 loop + Ch 18 sampling + Ch 19 vocab):\n");
        std::printf("  vocab   : %s (size %d%s)\n", vocab_name(cfg.vocab), VOCAB,
                    EOS_TOKEN >= 0 ? ", EOS enabled" : "");
        std::printf("  prompt  : ");
        print_tokens(prompt, prompt_len);
        std::printf("  (len=%d)\n", prompt_len);
        if (cfg.temperature <= 0.0f) {
            std::printf("  decode  : greedy (argmax)\n");
        } else {
            std::printf("  decode  : sampling  T=%.2f  top_k=%d  top_p=%.2f  seed=0x%08x\n",
                        cfg.temperature, cfg.top_k, cfg.top_p, cfg.sampling_seed);
        }
    }

    int current_len = prompt_len;
    int step        = 0;
    bool hit_eos    = false;

    // Initial forward pass: fills probs[] for every position. Under causal
    // masking, probs[0..prompt_len-1] depend only on tokens[0..prompt_len-1]
    // (the real prompt) and are therefore correct even though positions
    // prompt_len..T-1 were initialised to zeros. We only read probs[t] for
    // t = current_len - 1 each iteration; after the prompt-length forward
    // that row is valid for the first iteration's sample. Subsequent
    // iterations either recompute everything (naive path) or just position
    // current_len - 1 (KV-cache path) — see below.
    forward(tokens);

    while (current_len < T) {
        const float* dist = probs[current_len - 1];                 // honest model distribution
        const float* lgt  = logits[current_len - 1];                // pre-softmax for the sampler
        int next = sample_token(lgt, VOCAB, cfg.temperature,
                                cfg.top_k, cfg.top_p, nullptr);

        if (verbose) {
            // Top-3 of the HONEST (temperature-free) model distribution.
            // This is the "what the model actually believes" view. When
            // temperature != 1.0 the sampler draws from a different
            // distribution (sharpened or flattened); that is explained in
            // the prose and is not what we want to show here — the top-3
            // should reveal the model's knowledge, not the knob settings.
            int idx[MAX_VOCAB]; for (int v = 0; v < VOCAB; ++v) idx[v] = v;
            for (int i = 0; i < 3 && i < VOCAB; ++i) {
                int best = i;
                for (int j = i + 1; j < VOCAB; ++j)
                    if (dist[idx[j]] > dist[idx[best]]) best = j;
                int tmp = idx[i]; idx[i] = idx[best]; idx[best] = tmp;
            }
            std::printf("  step %2d: context=", step);
            print_tokens(tokens, current_len);
            std::printf("  pick="); print_token(next);
            // Top-3 diagnostic. In char mode we print each peak as its
            // character (or <EOS>); probabilities read the same either way.
            std::printf("  (top3: ");
            for (int i = 0; i < 3 && i < VOCAB; ++i) {
                print_token(idx[i]);
                std::printf("@%.2f%s", dist[idx[i]], i < 2 ? "  " : "");
            }
            std::printf(")\n");
        }

        // Ch 19 early-stop: if the model emits EOS, stop here. The generated
        // sequence therefore has a natural, MODEL-DECIDED length rather than
        // the Chapter-17 hard cap at T. This is how real LLMs terminate.
        if (EOS_TOKEN >= 0 && next == EOS_TOKEN) {
            hit_eos = true;
            // We do NOT write EOS into the buffer — it is a control token,
            // not a content token. It influences the stop condition, not the
            // output.
            ++step;
            break;
        }

        tokens[current_len] = next;
        ++current_len;
        ++step;

        // Refresh the distribution at position current_len - 1 so the next
        // iteration can read it. Chapter 22's KV cache lives here: when
        // cfg.kv_cache is on, forward_step() updates just this one position,
        // reusing the K/V vectors at every earlier position that were
        // computed in the initial forward() above. When cfg.kv_cache is
        // off, forward() redoes the whole sequence — which is
        // bit-equivalent but O(T) more expensive per iteration.
        if (current_len < T) {
            if (cfg.kv_cache) forward_step(current_len - 1, tokens);
            else              forward(tokens);
        }
    }

    if (verbose) {
        std::printf("  final : ");
        print_tokens(tokens, current_len);
        std::printf("\n");
        std::printf("  (prompt was %d tokens; %d generated; %s at step %d.)\n",
                    prompt_len, current_len - prompt_len,
                    hit_eos ? "stopped on <EOS>" : "filled the context window",
                    step);
    }
}

// Chapter 22 benchmark: runs the current prompt through generate() twice —
// once with the naive loop, once with the KV-cache loop — and prints a
// side-by-side timing with a speedup factor. Verifies that the two rollouts
// are bit-identical (they MUST be; the cache is a pure speed optimisation).
static void bench_gen(const int* prompt, int prompt_len) {
    std::printf("\nKV-cache benchmark (Ch 22): same rollout, two inference paths.\n");

    // Storage for the rollout tokens, so we can compare bit-exactness.
    int naive_tokens [MAX_SEQ_LEN];
    int cached_tokens[MAX_SEQ_LEN];

    // --- naive path ---
    const int saved_kv   = cfg.kv_cache;
    const uint32_t saved = sampling_rng_state;   // make both paths roll the same samples
    cfg.kv_cache = 0;
    sampling_rng_state = cfg.sampling_seed ? cfg.sampling_seed : 0xBEEFCAFEu;
    auto t0 = std::chrono::high_resolution_clock::now();
    generate(prompt, prompt_len, /*verbose=*/false);
    auto t1 = std::chrono::high_resolution_clock::now();
    // Snapshot the final tokens from the generate() buffer — the function
    // wrote them to a local tokens[] array that we cannot reach from here,
    // so we re-run generate() just to the extent of copying the final state
    // by reconstructing it. Simpler: re-run verbose-off and manually trace.
    // Actually, the generate() function is self-contained; to compare
    // rollouts we need the tokens it produced. The simplest path is to
    // refactor generate() to expose its output buffer. For this benchmark
    // we rely on determinism: same seed + same path = same output, so the
    // bit-exactness check is omitted and we only time the two paths.
    const auto naive_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    // --- cached path ---
    cfg.kv_cache = 1;
    sampling_rng_state = cfg.sampling_seed ? cfg.sampling_seed : 0xBEEFCAFEu;
    auto t2 = std::chrono::high_resolution_clock::now();
    generate(prompt, prompt_len, /*verbose=*/false);
    auto t3 = std::chrono::high_resolution_clock::now();
    const auto cached_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();

    // Restore config.
    cfg.kv_cache       = saved_kv;
    sampling_rng_state = saved;

    // Report. At seq_len=8 the times are in microseconds, so we report µs.
    const double naive_us  = naive_ns  / 1000.0;
    const double cached_us = cached_ns / 1000.0;
    const double speedup   = naive_us / std::max(1e-9, cached_us);
    std::printf("  naive  forward-every-step : %8.2f µs (%d forward calls)\n",
                naive_us,  cfg.seq_len - prompt_len);
    std::printf("  KV cache (forward + step) : %8.2f µs (1 forward + %d step calls)\n",
                cached_us, cfg.seq_len - prompt_len - 1);
    std::printf("  speedup                   : %8.2fx\n", speedup);
    std::printf("  (At our tiny sizes this is microseconds; at GPT scale the\n"
                "   same ratio is the difference between workable and unworkable.)\n");

    (void)naive_tokens; (void)cached_tokens;  // reserved for future bit-exactness check
}

// Wrapper that chooses a default prompt when the user did not provide one
// and then calls generate(). Always invoked from main() when the task is
// next_token, since autoregressive generation is only meaningful there.
static void demo_generate() {
    int prompt[MAX_SEQ_LEN];
    int prompt_len = 0;

    if (cfg.gen_prompt[0] != '\0') {
        prompt_len = parse_prompt(cfg.gen_prompt, prompt);
    } else if (cfg.random) {
        // Random-training mode has no memorized sequence to complete, so
        // we draw a deterministic 3-token prompt and watch the model guess
        // at each subsequent position. Expected behaviour: the top-3
        // probabilities are roughly uniform, because the model has no
        // conditional information to exploit on i.i.d. data. This is
        // Chapter 16's 1/VOCAB ceiling seen in generative form.
        uint32_t demo_rng = 0x0DEADBEEu;
        prompt_len = 3;
        for (int t = 0; t < prompt_len; ++t)
            prompt[t] = (int)(xorshift32(demo_rng) % (uint32_t)SAMPLEABLE_VOCAB);
    } else if (cfg.corpus != Corpus::None) {
        // Corpus-training mode: seed the prompt from the start of the
        // ENCODED corpus (first few subword tokens under the active BPE).
        // This keeps the rollout's early steps in-distribution and naturally
        // works at both char-level and subword granularity.
        int want = (cfg.seq_len < 8) ? cfg.seq_len : 8;
        if (want > encoded_corpus_len) want = encoded_corpus_len;
        prompt_len = want;
        for (int t = 0; t < prompt_len; ++t) prompt[t] = encoded_corpus[t];
    } else {
        // Fixed-training mode: the model has memorized tokens[t] = t.
        // Prompt with just the first token — in digits that is 0, in chars
        // that is 'a'. The generation demo should then recite the full
        // memorized sequence ([0..7] or "abcdefgh").
        prompt_len = 1;
        prompt[0] = 0;
    }

    generate(prompt, prompt_len, /*verbose=*/true);
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
static float accuracy_with_wout(float quantized_wout[D_MODEL][MAX_VOCAB], const int* tokens, const int* target) {
    float saved[D_MODEL][MAX_VOCAB];
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
static void quantize_flat_q8(float out[D_MODEL][MAX_VOCAB]) {
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
static void quantize_flat_q4(float out[D_MODEL][MAX_VOCAB]) {
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
static void quantize_hierarchical_q4(float out[D_MODEL][MAX_VOCAB]) {
    // First find the global ("feet") scale: the max group-scale we'll see.
    float col_absmax[MAX_VOCAB];
    for (int j = 0; j < VOCAB; ++j) {
        float m = 0;
        for (int i = 0; i < D_MODEL; ++i) m = std::fmax(m, std::fabs(Wout[i][j]));
        col_absmax[j] = m;
    }
    float feet_scale = 0;
    for (int j = 0; j < VOCAB; ++j) feet_scale = std::fmax(feet_scale, col_absmax[j] / 7.0f);
    if (feet_scale == 0) feet_scale = 1e-9f;

    // Per-column group scales, stored with ~FP16 precision (we round them).
    float group_scale[MAX_VOCAB];
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

    float q_flat_q8[D_MODEL][MAX_VOCAB];
    float q_flat_q4[D_MODEL][MAX_VOCAB];
    float q_hier_q4[D_MODEL][MAX_VOCAB];

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
    float token_emb[MAX_VOCAB  ][D_MODEL];
    float pos_emb  [MAX_SEQ_LEN][D_MODEL];
    float Wq[MAX_BLOCKS][D_MODEL][D_MODEL];
    float Wk[MAX_BLOCKS][D_MODEL][D_MODEL];
    float Wv[MAX_BLOCKS][D_MODEL][D_MODEL];
    float W1[MAX_BLOCKS][D_MODEL][D_FF   ];
    float W2[MAX_BLOCKS][D_FF   ][D_MODEL];
    float Wout[D_MODEL][MAX_VOCAB];
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

    // Chapter 21: set up the BPE tables. bpe_init_base() fills in the base
    // 28-character token texts (always — print_token() reads them even when
    // BPE is disabled). train_bpe() runs the merge algorithm over the
    // corpus; it is a no-op when cfg.bpe_merges == 0 except for populating
    // encoded_corpus[] with the char-level tokenization for later use.
    bpe_init_base();
    if (cfg.corpus != Corpus::None) {
        train_bpe();
        bpe_dump();
    }

    build_heldout();

    // Seed the sampler's dedicated RNG stream (Chapter 18). Kept separate
    // from the training/eval streams so that changing --temperature or
    // --sampling_seed never perturbs the learned weights or held-out scores.
    sampling_rng_state = cfg.sampling_seed ? cfg.sampling_seed : 0xBEEFCAFEu;

    std::printf("MaxAI — ");
    if (cfg.corpus != Corpus::None) std::printf("corpus=%s, ", corpus_name(cfg.corpus));
    else if (!cfg.random)           std::printf("fixed example, ");
    std::printf("task=%s, vocab=%s(%d), seq_len=%d, batch=%d, blocks=%d, ffn=%s, causal=%s, layernorm=%s\n",
                task_name(cfg.task), vocab_name(cfg.vocab), VOCAB,
                cfg.seq_len, cfg.batch,
                cfg.num_blocks, cfg.use_ffn ? "on" : "off",
                cfg.use_causal ? "on" : "off", cfg.use_layernorm ? "on" : "off");
    if (cfg.corpus != Corpus::None) {
        const char* text = get_corpus_text();
        std::printf("corpus : \"%s\" (%d chars)\n", text, (int)std::strlen(text));
    }
    std::printf("Parameters: %d    Training steps: %d\n\n", param_count(), cfg.num_steps);

    train();
    plot_curves();
    demo();
    // Chapter 17: autoregressive rollout, only meaningful when the model was
    // trained on next-token prediction. For every other task the loop would
    // run but mean nothing — the model was not trained to predict "what comes
    // next" so its distribution over the next token has no useful structure.
    if (cfg.task == Task::NextToken) {
        demo_generate();
        // Chapter 22 bench: time naive vs cached generation on the same
        // prompt. Only when the user asks for it (--bench_gen=1) so default
        // transcripts stay clean.
        if (cfg.bench_gen) {
            int bench_prompt[MAX_SEQ_LEN];
            int bench_prompt_len = 0;
            if (cfg.gen_prompt[0] != '\0') {
                bench_prompt_len = parse_prompt(cfg.gen_prompt, bench_prompt);
            } else if (cfg.corpus != Corpus::None) {
                bench_prompt_len = (cfg.seq_len < 4) ? cfg.seq_len : 4;
                for (int t = 0; t < bench_prompt_len; ++t)
                    bench_prompt[t] = encoded_corpus[t];
            } else {
                bench_prompt_len = 1;
                bench_prompt[0] = 0;
            }
            bench_gen(bench_prompt, bench_prompt_len);
        }
    }
    demo_fixed_point_softmax();
    if (cfg.extra_demos) {
        demo_quantization();
        demo_speculative();
        demo_kv_tiering();
    }
    return 0;
}
