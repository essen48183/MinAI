// ============================================================================
//  sgd_test.cpp — Stage 0 Simulator, SGD convergence test
// ----------------------------------------------------------------------------
//  Before we wrap MaxAI's full training loop around the Tile simulator, we
//  need to confirm that SGD can converge at all through a noisy Q8.8.8
//  tile with stochastic-rounded weight writes. This is the simplest
//  possible training test — no transformer, no softmax, no attention.
//  Just: can we teach a single tile to approximate a known target matrix
//  W_target by repeated SGD updates against a squared-error loss?
//
//  Setup:
//    - D_MODEL = 16 (one tile matches one matrix dimension from the book)
//    - Target: random W_target in [-0.5, 0.5], fixed per test
//    - Initial tile weights: small random values in [-0.1, 0.1]
//    - 5000 SGD steps of (x, y_target) pairs where y = W_target @ x
//    - MSE loss, analytical gradient, lr = 0.05
//    - Each weight update goes through Tile.write_delta(), which does
//      stochastic rounding across the three slices.
//
//  The test reports initial loss (averaged over first 10 steps) and final
//  loss (averaged over last 100 steps), plus the RMS distance between the
//  final tile weights and the target weights. A converged tile should
//  show final loss << initial loss, and param RMS error at or below the
//  noise floor of the tile.
//
//  Expected results (by noise setting):
//    Zero-noise       : converges cleanly, final loss << 0.001
//    Precision analog : converges, final loss slightly above zero
//    Standard analog  : converges to a higher floor (~noise level)
//    Hobbyist analog  : converges slowly, high residual loss
//
//  What passing this test proves: the training machinery (stochastic-
//  rounded writes through Q8.8.8 slices plus SGD) is functional. Does NOT
//  prove MaxAI trains — that is the next artifact.
// ============================================================================

#include "analog_sim.h"

#include <cstdio>
#include <cstdint>
#include <cmath>

using namespace analog_sim;

// ----------------------------------------------------------------------------
// Small RNG helpers — same xorshift32 pattern used throughout.
// ----------------------------------------------------------------------------
static uint32_t xs(uint32_t& s) {
    uint32_t x = s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return s = x;
}
static float u01(uint32_t& s) {
    return (xs(s) & 0x00FFFFFFu) / (float)0x01000000;
}

// ----------------------------------------------------------------------------
// Build a target weight matrix in a small range. Small values keep matmul
// outputs in a range the tile can represent without saturation.
// ----------------------------------------------------------------------------
static void build_target(float W[D_MODEL][D_MODEL], uint32_t seed) {
    uint32_t r = seed;
    for (int i = 0; i < D_MODEL; ++i)
        for (int j = 0; j < D_MODEL; ++j)
            W[i][j] = (2.0f * u01(r) - 1.0f) * 0.5f;
}

static void build_init(float W[D_MODEL][D_MODEL], uint32_t seed) {
    uint32_t r = seed;
    for (int i = 0; i < D_MODEL; ++i)
        for (int j = 0; j < D_MODEL; ++j)
            W[i][j] = (2.0f * u01(r) - 1.0f) * 0.1f;
}

// ----------------------------------------------------------------------------
// MSE loss and its analytical gradient with respect to the weight matrix.
//   L = (1/D) * sum_j (y_pred_j - y_true_j)^2
//   dL/dW[i][j] = (2/D) * (y_pred_j - y_true_j) * x[i]
// ----------------------------------------------------------------------------
static float mse(const float y_pred[D_MODEL], const float y_true[D_MODEL]) {
    float s = 0.0f;
    for (int j = 0; j < D_MODEL; ++j) {
        float d = y_pred[j] - y_true[j];
        s += d * d;
    }
    return s / D_MODEL;
}

static void grad(
    const float x[D_MODEL],
    const float y_pred[D_MODEL],
    const float y_true[D_MODEL],
    float g[D_MODEL][D_MODEL])
{
    for (int i = 0; i < D_MODEL; ++i) {
        float xi = x[i];
        for (int j = 0; j < D_MODEL; ++j) {
            g[i][j] = (2.0f / D_MODEL) * (y_pred[j] - y_true[j]) * xi;
        }
    }
}

// ----------------------------------------------------------------------------
// Run one training experiment at a given noise model. Returns summary stats.
// ----------------------------------------------------------------------------
struct Summary {
    float initial_loss;   // mean of first 10 steps
    float final_loss;     // mean of last 100 steps
    float param_rms_err;  // RMS of (final tile weights) - W_target
    long long writes;     // total Tile.write_delta calls
};

static Summary train_one(const NoiseModel& noise, int n_steps, float lr) {
    Tile tile(noise);

    float W_target[D_MODEL][D_MODEL];
    float W_init  [D_MODEL][D_MODEL];
    build_target(W_target, 0x13579BDFu);
    build_init  (W_init,   0x87654321u);

    tile.program(W_init);

    uint32_t rng = 0xABCDEF12u;

    float initial_acc = 0.0f;
    float final_acc   = 0.0f;
    int   final_n     = 0;

    for (int step = 0; step < n_steps; ++step) {
        // Draw a random input vector in [-1, 1].
        float x[D_MODEL];
        for (int i = 0; i < D_MODEL; ++i) x[i] = 2.0f * u01(rng) - 1.0f;

        // Ideal target: y_true = W_target @ x.
        float y_true[D_MODEL];
        for (int j = 0; j < D_MODEL; ++j) {
            float s = 0.0f;
            for (int i = 0; i < D_MODEL; ++i) s += W_target[i][j] * x[i];
            y_true[j] = s;
        }

        // Forward through the noisy tile.
        float y_pred[D_MODEL];
        tile.forward(x, y_pred);

        // Loss for logging.
        float L = mse(y_pred, y_true);
        if (step < 10)                 initial_acc += L;
        if (step >= n_steps - 100)   { final_acc   += L; ++final_n; }

        // Gradient of MSE w.r.t. W.
        float g[D_MODEL][D_MODEL];
        grad(x, y_pred, y_true, g);

        // Write every cell's update through stochastic-rounded write_delta.
        // At this scale the gradients are large enough that we don't need a
        // separate digital accumulator between steps; write_delta's own
        // per-slice stochastic rounding handles sub-tap deltas.
        for (int i = 0; i < D_MODEL; ++i) {
            for (int j = 0; j < D_MODEL; ++j) {
                float update = -lr * g[i][j];
                tile.write_delta(i, j, update);
            }
        }
    }

    // RMS distance between final tile weights and target weights.
    float err_sq = 0.0f;
    for (int i = 0; i < D_MODEL; ++i) {
        for (int j = 0; j < D_MODEL; ++j) {
            float d = tile.read(i, j) - W_target[i][j];
            err_sq += d * d;
        }
    }

    Summary s;
    s.initial_loss  = initial_acc / 10.0f;
    s.final_loss    = final_n > 0 ? final_acc / final_n : 0.0f;
    s.param_rms_err = std::sqrt(err_sq / (D_MODEL * D_MODEL));
    s.writes        = tile.total_writes();
    return s;
}

// ----------------------------------------------------------------------------
// Entry point — sweep four noise settings, report convergence.
// ----------------------------------------------------------------------------
int main() {
    std::printf("\n");
    std::printf("=============================================================\n");
    std::printf("  STAGE 0  -  SGD CONVERGENCE TEST\n");
    std::printf("  Can SGD converge a noisy Q8.8.8 tile to a target matrix?\n");
    std::printf("=============================================================\n\n");

    const int   n_steps = 5000;
    const float lr      = 0.05f;

    std::printf("  D_MODEL=%d, %d SGD steps, lr=%.3f, MSE loss\n", D_MODEL, n_steps, lr);
    std::printf("  Target: random W in [-0.5, 0.5]\n");
    std::printf("  Init  : random W in [-0.1, 0.1]\n\n");

    struct Case {
        const char* name;
        const char* note;
        NoiseModel  noise;
    };

    Case cases[4];

    cases[0].name = "Zero-noise";
    cases[0].note = "(Q8.8.8 quantization only)";
    cases[0].noise.sigma_per_mac = 0.0f;
    cases[0].noise.cal_residual  = 0.0f;

    cases[1].name = "Precision analog";
    cases[1].note = "(0.003% per-MAC)";
    cases[1].noise.sigma_per_mac = 0.00003f;
    cases[1].noise.cal_residual  = 0.000005f;

    cases[2].name = "Standard analog";
    cases[2].note = "(0.03% per-MAC)";
    cases[2].noise.sigma_per_mac = 0.0003f;
    cases[2].noise.cal_residual  = 0.00005f;

    cases[3].name = "Hobbyist analog";
    cases[3].note = "(0.3% per-MAC)";
    cases[3].noise.sigma_per_mac = 0.003f;
    cases[3].noise.cal_residual  = 0.0005f;

    std::printf("  %-22s  %-24s | %10s  %10s  %12s\n",
                "test", "", "init loss", "final loss", "param RMS");
    std::printf("  ----------------------+--------------------------+-----------------------------------------\n");
    for (int c = 0; c < 4; ++c) {
        Summary s = train_one(cases[c].noise, n_steps, lr);
        std::printf("  %-22s  %-24s | %10.5f  %10.5f  %12.6f\n",
                    cases[c].name, cases[c].note,
                    s.initial_loss, s.final_loss, s.param_rms_err);
    }

    std::printf("\n");
    std::printf("=============================================================\n");
    std::printf("  What to look for:\n");
    std::printf("    - final loss much smaller than initial loss (SGD worked)\n");
    std::printf("    - param RMS error at/near the noise floor of the tile\n");
    std::printf("    - the ordering across noise settings makes physical sense\n");
    std::printf("\n");
    std::printf("  If all four converge, SGD+stochastic-write machinery is\n");
    std::printf("  functional and Gate 1 (MaxAI training) is ready to write.\n");
    std::printf("=============================================================\n\n");
    return 0;
}
