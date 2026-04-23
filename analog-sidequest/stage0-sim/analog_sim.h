// ============================================================================
//  analog_sim.h — Stage 0 Simulator, tile-level interface
// ----------------------------------------------------------------------------
//  Models a single Q8.8.8 bit-sliced analog MAC tile in software with a
//  parameterized noise model. The interface is deliberately small: program
//  the tile with an ideal weight matrix, run a noisy forward pass, update
//  a single weight with stochastic rounding, read back a weight. That's
//  the whole tile abstraction.
//
//  Dimensions match MaxAI (D_MODEL = 16). Same pattern the book uses.
// ============================================================================

#pragma once

#include <cstdint>

namespace analog_sim {

// -----------------------------------------------------------------------------
// Dimensions — match the book's MaxAI default.
// -----------------------------------------------------------------------------
constexpr int D_MODEL = 16;

// -----------------------------------------------------------------------------
// Bit-slicing: three 8-bit slices per weight. Each slice is an independent
// 8-bit digipot whose current gets scaled by its binary weight at the column
// summing junction.
// -----------------------------------------------------------------------------
constexpr int   SLICES             = 3;
constexpr int   TAPS_PER_SLICE     = 256;              // 8-bit per slice
constexpr float SLICE_WEIGHT[SLICES] = { 1.0f, 1.0f/256.0f, 1.0f/65536.0f };

// -----------------------------------------------------------------------------
// NoiseModel — every noise term the architecture has to survive, in a single
// struct so tests can sweep each axis cleanly.
//
//   sigma_per_mac   : RMS weight-equivalent noise added to each cell on every
//                     read. Fraction of full weight scale (1.0 = whole range).
//                     0.003  = 0.3%, hobbyist-parts
//                     0.0003 = 0.03%, standard-analog with autozero
//                     0.00003 = 0.003%, precision-analog with matched parts
//                     0.0   = quantization ceiling only
//   sigma_write     : per-write tap jitter, in taps. 1.0 means one-tap RMS
//                     jitter on every write.
//   drift_ppm_per_s : slow drift of individual cells. Not used by the
//                     single-tile precision test; reserved for longer runs.
//   cal_residual    : residual column-sum error after calibration reference
//                     cells correct out drift and common-mode noise.
//                     Fraction of full output scale.
//   autozero        : if true, the summing op-amp is assumed autozero-grade
//                     (Analog Devices AD8629 class). Removes 1/f and DC
//                     drift at the summing node. Modeled by zeroing an
//                     additional slow noise term.
// -----------------------------------------------------------------------------
struct NoiseModel {
    float sigma_per_mac   = 0.0003f;  // default: standard-analog with autozero
    float sigma_write     = 0.0f;     // default: no write jitter
    float drift_ppm_per_s = 0.0f;     // default: no drift (short runs)
    float cal_residual    = 0.00005f; // default: 0.005% residual after cal
    bool  autozero        = true;
};

// -----------------------------------------------------------------------------
// Tile — one 16x16 Q8.8.8 bit-sliced analog MAC tile.
//
//   program()     : decompose an ideal float weight matrix into three 8-bit
//                   slices per cell and load the tile.
//   forward()     : compute y = W @ x through the noisy reconstruction path.
//   write_delta() : commit a single-cell SGD update, with stochastic rounding
//                   spreading the partial-tap change across slices.
//   read()        : reconstruct a cell's weight from its three slices (ideal,
//                   no noise — useful for comparing programmed vs. intended).
// -----------------------------------------------------------------------------
class Tile {
public:
    explicit Tile(const NoiseModel& noise = {});

    void  program(const float W[D_MODEL][D_MODEL]);
    void  forward(const float x[D_MODEL], float y[D_MODEL]) const;
    void  write_delta(int row, int col, float delta);
    float read(int row, int col) const;

    long long total_writes() const { return writes_; }

private:
    uint8_t  slice_[SLICES][D_MODEL][D_MODEL] = {};
    NoiseModel noise_;
    mutable uint32_t rng_ = 0xDEADBEEFu;
    long long writes_ = 0;
};

// -----------------------------------------------------------------------------
// measure_effective_bits — the core precision test.
//
//   Feeds `n_trials` random input vectors through the tile and an ideal
//   float32 reference matmul, then returns
//       bits = log2( RMS(signal) / RMS(error) ).
//
//   With zero noise the tile's error is bounded by bit-slice quantization,
//   giving roughly log2(2^24) = 24 bits. With realistic noise the RMS error
//   grows and the effective bits drops accordingly.
// -----------------------------------------------------------------------------
float measure_effective_bits(
    const Tile& tile,
    const float W[D_MODEL][D_MODEL],
    int   n_trials = 10000);

}  // namespace analog_sim
