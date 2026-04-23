# Stage 0 Simulator — one-command build + run
#
# `make`        -> builds all stage-0 binaries
# `make run`    -> builds and runs both tests in sequence
# `make clean`  -> removes the binaries

CXX      = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

BINS   = stage0_sim stage0_sgd
COMMON = analog_sim.cpp
HDRS   = analog_sim.h

all: $(BINS)

stage0_sim: main.cpp $(COMMON) $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ main.cpp $(COMMON)

stage0_sgd: sgd_test.cpp $(COMMON) $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ sgd_test.cpp $(COMMON)

run: all
	@echo "---- stage0_sim (precision test) ----"
	@./stage0_sim
	@echo "---- stage0_sgd (SGD convergence test) ----"
	@./stage0_sgd

clean:
	rm -f $(BINS)

.PHONY: all run clean
