// ============================================================================
//  analog_sim.cpp — Stage 0 Simulator, tile-level implementation
// ----------------------------------------------------------------------------
//  Models what happens physically in a Q8.8.8 bit-sliced analog tile, in
//  software. The book's comment style applies: long-form explanations
//  alongside the arithmetic, erring on the side of over-explaining why
//  each step is the way it is.
//
//  There are three pieces of physics to capture:
//
//    1. BIT-SLICE QUANTIZATION. A weight w in [-1, 1] is decomposed into
//       three 8-bit tap values s0, s1, s2. At the column summing junction
//       the reconstructed current is (s0 * 1.0) + (s1 * 1/256) + (s2 * 1/65536).
//       The quantization floor is ~2^-24 of full scale — better than float32
//       mantissa precision.
//
//    2. PER-CELL NOISE. Each cell contributes a small Gaussian error on
//       every read, representing thermal (Johnson-Nyquist) noise on the
//       resistive element plus any residual 1/f noise the autozero op-amp
//       didn't fully reject. Magnitude is parameterized.
//
//    3. SUMMING-NODE NOISE. A small extra Gaussian term at the column
//       represents calibration residual plus op-amp input noise.
//
//  For the single-tile test we leave drift at zero (short runs) and leave
//  stochastic-rounding details on the write side for the training-loop test.
// ============================================================================

#include "analog_sim.h"

#include <cmath>
#include <cstring>

namespace analog_sim {

// ----------------------------------------------------------------------------
// Deterministic RNG — same xorshift32 pattern the book uses everywhere, so a
// reader of minai.cpp recognizes it immediately.
// ----------------------------------------------------------------------------
static uint32_t xorshift32(uint32_t& s) {
    uint32_t x = s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return s = x;
}
static float rand_u01(uint32_t& s) {
    // Uniform in [0, 1). 24 bits of mantissa precision is plenty for the
    // statistical draws we need here.
    return (xorshift32(s) & 0x00FFFFFFu) / (float)0x01000000;
}
static float rand_gauss(uint32_t& s) {
    // Box-Muller. One gaussian sample per call. Not the cheapest sampler
    // in the world, but fine for a ~10k-trial test.
    float u1 = rand_u01(s);
    float u2 = rand_u01(s);
    if (u1 < 1e-7f) u1 = 1e-7f;            // guard against log(0)
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
}

// ----------------------------------------------------------------------------
// decompose — turn an ideal float weight into three 8-bit slice taps.
//
// For a weight w in [-1, 1]:
//   Slice 0 (MSB)    holds a "coarse" Q8 approximation of w.
//   Slice 1 (middle) holds the residual, scaled up by 256 to fit in its Q8.
//   Slice 2 (LSB)    holds the residual of that, scaled by 65536.
//
// Each tap is mapped from the native [-1, 1] slice value by:
//   tap = round((native + 1) * 127.5)    -> integer in [0, 255]
//   midscale (127.5) represents zero, tap=0 is -1, tap=255 is +1.
//
// The composite reconstruction has a quantization step of 1 / (127.5 * 65536)
// ≈ 1.2e-7, or ~23 effective bits ≈ float32 mantissa precision.
// ----------------------------------------------------------------------------
static void decompose(float w, uint8_t slices[SLICES]) {
    if (w >  1.0f) w =  1.0f;
    if (w < -1.0f) w = -1.0f;

    float remainder = w;
    for (int s = 0; s < SLICES; ++s) {
        float native = remainder / SLICE_WEIGHT[s];
        int   tap    = (int)std::lround((native + 1.0f) * 127.5f);
        if (tap < 0)   tap = 0;
        if (tap > 255) tap = 255;
        slices[s] = (uint8_t)tap;

        // Subtract what this slice actually represents, so the next slice
        // corrects the leftover.
        float reconstructed = ((tap - 127.5f) / 127.5f) * SLICE_WEIGHT[s];
        remainder -= reconstructed;
    }
}

// Ideal (noise-free) reconstruction of a weight from its three taps.
static float reconstruct_ideal(const uint8_t slices[SLICES]) {
    float w = 0.0f;
    for (int s = 0; s < SLICES; ++s) {
        w += ((slices[s] - 127.5f) / 127.5f) * SLICE_WEIGHT[s];
    }
    return w;
}

// ----------------------------------------------------------------------------
// Tile construction / programming / readback.
// ----------------------------------------------------------------------------
Tile::Tile(const NoiseModel& noise) : noise_(noise) {}

void Tile::program(const float W[D_MODEL][D_MODEL]) {
    for (int i = 0; i < D_MODEL; ++i) {
        for (int j = 0; j < D_MODEL; ++j) {
            uint8_t s[SLICES];
            decompose(W[i][j], s);
            for (int k = 0; k < SLICES; ++k) slice_[k][i][j] = s[k];
        }
    }
}

float Tile::read(int row, int col) const {
    uint8_t s[SLICES];
    for (int k = 0; k < SLICES; ++k) s[k] = slice_[k][row][col];
    return reconstruct_ideal(s);
}

// ----------------------------------------------------------------------------
// forward — the core noisy matmul.
//
// For each output column j:
//   1. Reconstruct each cell's weight from its three slices (noise-free
//      value of what was programmed).
//   2. Add per-cell Gaussian noise (thermal + residual 1/f). This is
//      equivalent to modeling a noisy current out of the cell that
//      scales with the input voltage.
//   3. Accumulate weight * input across the row (Kirchhoff summation).
//   4. Add a summing-node noise term (calibration residual + op-amp noise).
//
// Physically: the matmul itself — summation and multiplication — is exact
// (Kirchhoff + Ohm). The noise we model is the honest imprecision of the
// measurement channel. Drive it lower and effective bits go up.
// ----------------------------------------------------------------------------
void Tile::forward(const float x[D_MODEL], float y[D_MODEL]) const {
    for (int j = 0; j < D_MODEL; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < D_MODEL; ++i) {
            uint8_t s[SLICES];
            for (int k = 0; k < SLICES; ++k) s[k] = slice_[k][i][j];
            float w = reconstruct_ideal(s);

            // Per-cell Gaussian noise. Modeled as additive on the weight
            // value, which produces output noise scaled by the input
            // magnitude — the correct behaviour for thermal noise on a
            // resistive element driven by an input voltage.
            if (noise_.sigma_per_mac > 0) {
                w += noise_.sigma_per_mac * rand_gauss(rng_);
            }

            sum += w * x[i];
        }

        // Summing-node noise: calibration residual + op-amp input noise,
        // independent of how many cells fed the sum. For autozero op-amps
        // this is close to thermal-floor; without autozero it would also
        // include a 1/f component we're not modeling in this single-tile
        // short-run test.
        if (noise_.cal_residual > 0) {
            sum += noise_.cal_residual * rand_gauss(rng_);
        }

        y[j] = sum;
    }
}

// ----------------------------------------------------------------------------
// write_delta — single-cell SGD update with stochastic rounding across slices.
//
// Given a desired weight change `delta`, decompose it across the three
// slices in order of significance (MSB first) and, for each slice, write
// the integer tap change deterministically plus one extra tap
// stochastically based on the fractional part.
//
// This is Trick 2 from ARCHITECTURE.md in concrete form. The result is
// unbiased: expected change == delta, even when the integer-tap change
// is zero.
// ----------------------------------------------------------------------------
void Tile::write_delta(int row, int col, float delta) {
    for (int s = 0; s < SLICES; ++s) {
        float native_delta = delta / SLICE_WEIGHT[s];
        float tap_delta    = native_delta * 127.5f;     // in tap units

        int   int_part = (int)std::trunc(tap_delta);
        float frac     = tap_delta - int_part;

        int extra = 0;
        float r   = rand_u01(rng_);
        if (frac > 0.0f && r <  frac)  extra =  1;
        if (frac < 0.0f && r < -frac)  extra = -1;

        int new_tap = (int)slice_[s][row][col] + int_part + extra;
        if (new_tap < 0)   new_tap = 0;
        if (new_tap > 255) new_tap = 255;
        slice_[s][row][col] = (uint8_t)new_tap;

        // Subtract what we just committed from the remaining delta, so
        // subsequent (finer) slices compensate for this slice's residual.
        float committed_native = (int_part + extra) / 127.5f * SLICE_WEIGHT[s];
        delta -= committed_native;
    }
    ++writes_;
}

// ----------------------------------------------------------------------------
// measure_effective_bits — RMS-based precision metric.
//
//   bits = log2( RMS(signal) / RMS(error) )
//
// For a programmed tile with weights W and random inputs x:
//   * The "signal" is the ideal float32 matmul output Wx.
//   * The "error" is the difference between the tile's noisy output and
//     the ideal output.
//
// This is a standard SNR-to-bits conversion. It gives a single number
// that compresses the full precision story into something comparable
// across noise settings.
// ----------------------------------------------------------------------------
float measure_effective_bits(
    const Tile& tile,
    const float W[D_MODEL][D_MODEL],
    int   n_trials)
{
    double sum_sq_err    = 0.0;
    double sum_sq_signal = 0.0;
    uint32_t rng = 0xFEEDFACEu;

    for (int t = 0; t < n_trials; ++t) {
        float x[D_MODEL], y_ideal[D_MODEL], y_sim[D_MODEL];

        // Random input in [-1, 1].
        for (int i = 0; i < D_MODEL; ++i) x[i] = 2.0f * rand_u01(rng) - 1.0f;

        // Ideal float32 matmul.
        for (int j = 0; j < D_MODEL; ++j) {
            float s = 0.0f;
            for (int i = 0; i < D_MODEL; ++i) s += W[i][j] * x[i];
            y_ideal[j] = s;
        }

        // Simulated analog matmul through the noisy tile.
        tile.forward(x, y_sim);

        for (int j = 0; j < D_MODEL; ++j) {
            float err = y_sim[j] - y_ideal[j];
            sum_sq_err    += (double)err * err;
            sum_sq_signal += (double)y_ideal[j] * y_ideal[j];
        }
    }

    double rms_err    = std::sqrt(sum_sq_err    / (double)(n_trials * D_MODEL));
    double rms_signal = std::sqrt(sum_sq_signal / (double)(n_trials * D_MODEL));
    if (rms_err < 1e-30) rms_err = 1e-30;      // guard against log(0)
    double snr = rms_signal / rms_err;
    return (float)(std::log(snr) / std::log(2.0));
}

}  // namespace analog_sim
