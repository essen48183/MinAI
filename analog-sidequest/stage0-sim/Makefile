# Stage 0 Simulator — one-command build + run
#
# `make`        -> builds all stage-0 binaries
# `make run`    -> builds and runs both tests in sequence
# `make clean`  -> removes the binaries

CXX      = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

BINS   = stage0_sim stage0_sgd stage0_maxai
COMMON = analog_sim.cpp
HDRS   = analog_sim.h

all: $(BINS)

stage0_sim: main.cpp $(COMMON) $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(COMMON)

stage0_sgd: sgd_test.cpp $(COMMON) $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ sgd_test.cpp $(COMMON)

# Gate 1 — full MaxAI training through the analog sim. Standalone (carries
# its own analog hooks inline), so it does not link analog_sim.cpp.
stage0_maxai: maxai_analog.cpp
	$(CXX) $(CXXFLAGS) -o $@ maxai_analog.cpp

run: all
	@echo "---- stage0_sim  (precision test) ----"
	@./stage0_sim
	@echo "---- stage0_sgd  (SGD convergence test) ----"
	@./stage0_sgd
	@echo "---- stage0_maxai (Gate 1 default run at sigma=0.0003) ----"
	@./stage0_maxai --task=next_token --causal=1 --vocab=chars --corpus=seashells --bpe=16 \
	                --seq_len=16 --batch=16 --blocks=2 --layernorm=1 --steps=3000 \
	                --analog_noise=0.0003 --gen_prompt="she sells " 2>&1 | tail -40

clean:
	rm -f $(BINS)

.PHONY: all run clean
