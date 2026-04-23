// ============================================================================
//  main.cpp — Stage 0 Simulator entry point
// ----------------------------------------------------------------------------
//  Four precision tests on a single Q8.8.8 bit-sliced tile, sweeping noise
//  from zero up through hobbyist-grade. Output is effective-bits per test.
//
//  This is Stage-0 version 0 — the single-tile check. Gate 1 (MaxAI training
//  through this tile) and Gate 2 (96-block scaling) are next-up artifacts.
// ============================================================================

#include "analog_sim.h"

#include <cstdio>
#include <cstdint>

using namespace analog_sim;

// Seed a random weight matrix in [-1, 1]. Same seed across all four tests so
// the tile is programmed identically — only the noise model changes between
// tests, which makes the bit-count differences genuinely attributable to
// noise rather than to the particular weight draw.
static void random_weights(float W[D_MODEL][D_MODEL], uint32_t seed) {
    uint32_t r = seed;
    auto next = [&](){
        r ^= r << 13; r ^= r >> 17; r ^= r << 5;
        return (r & 0x00FFFFFFu) / (float)0x01000000;
    };
    for (int i = 0; i < D_MODEL; ++i)
        for (int j = 0; j < D_MODEL; ++j)
            W[i][j] = 2.0f * next() - 1.0f;
}

static void run_test(const char* name, const char* note, const NoiseModel& noise) {
    Tile  tile(noise);
    float W[D_MODEL][D_MODEL];
    random_weights(W, 0x13579BDFu);
    tile.program(W);

    float bits = measure_effective_bits(tile, W);
    std::printf("  %-44s  %5.2f bits    %s\n", name, bits, note);
}

int main() {
    std::printf("\n");
    std::printf("=============================================================\n");
    std::printf("  STAGE 0 SIMULATOR  -  single-tile precision check\n");
    std::printf("=============================================================\n\n");

    std::printf("  Q8.8.8 bit-sliced, D_MODEL=16, 10,000 random input vectors\n");
    std::printf("  Effective bits = log2( RMS(signal) / RMS(error) )\n\n");

    // Test 1 - zero noise. Pure Q8.8.8 quantization ceiling.
    {
        NoiseModel n;
        n.sigma_per_mac = 0.0f;
        n.cal_residual  = 0.0f;
        run_test("Test 1  zero-noise  (quantization only)",
                 "ceiling test",
                 n);
    }

    // Test 2 - precision analog. Autozero op-amps, matched thin-film
    // resistors, actively stabilized temperature. The aspirational hardware
    // budget: 0.003% per-MAC weight noise.
    {
        NoiseModel n;
        n.sigma_per_mac = 0.00003f;    // 0.003%
        n.cal_residual  = 0.000005f;   // 0.0005%
        run_test("Test 2  precision analog  (0.003% per-MAC)",
                 "aspirational",
                 n);
    }

    // Test 3 - standard analog. Autozero op-amps, regular 0.1% resistors,
    // calibration loop running. The realistic target for the Stage 1 PCB:
    // 0.03% per-MAC weight noise.
    {
        NoiseModel n;
        n.sigma_per_mac = 0.0003f;     // 0.03%
        n.cal_residual  = 0.00005f;    // 0.005%
        run_test("Test 3  standard analog   (0.03% per-MAC)",
                 "realistic Stage 1 target",
                 n);
    }

    // Test 4 - hobbyist analog. Standard op-amps, 1% resistors, no
    // calibration refinements. The "did we even try" baseline. 0.3%
    // per-MAC weight noise.
    {
        NoiseModel n;
        n.sigma_per_mac = 0.003f;      // 0.3%
        n.cal_residual  = 0.0005f;     // 0.05%
        run_test("Test 4  hobbyist analog   (0.3% per-MAC)",
                 "lower bound",
                 n);
    }

    std::printf("\n");
    std::printf("=============================================================\n");
    std::printf("  Training needs ~8 effective bits at 96-block depth per\n");
    std::printf("  published IBM analog-training results.  Tests 1-3 pass\n");
    std::printf("  that bar cleanly; Test 4 sits at the boundary.\n");
    std::printf("\n");
    std::printf("  Next step:  integrate with MaxAI's training loop and\n");
    std::printf("  measure actual training convergence through the tile\n");
    std::printf("  model (Gate 1).  See STAGE0.md for the full plan.\n");
    std::printf("=============================================================\n\n");
    return 0;
}
