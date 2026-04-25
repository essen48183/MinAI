# Stage 0.5 firmware — build on the Pi Zero 2W itself.
#
# `make`        -> compiles stage0_5
# `make run`    -> builds and runs all three experiments (sudo)
# `make exp1`   -> runs only Experiment 1 (slice-summer linearity)
# `make exp2`   -> runs only Experiment 2 (full Q8.8.8 dot product)
# `make exp3`   -> runs only Experiment 3 (stochastic-rounded write + readback)
# `make clean`  -> removes the binary

CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

stage0_5: stage0_5.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run: stage0_5
	sudo ./stage0_5 --all

exp1: stage0_5
	sudo ./stage0_5 --exp1

exp2: stage0_5
	sudo ./stage0_5 --exp2

exp3: stage0_5
	sudo ./stage0_5 --exp3

clean:
	rm -f stage0_5

.PHONY: run exp1 exp2 exp3 clean
