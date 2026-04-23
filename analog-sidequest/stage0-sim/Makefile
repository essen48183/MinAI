# Stage 0 Simulator — one-command build + run
#
# `make`        -> builds stage0_sim
# `make run`    -> builds and runs
# `make clean`  -> removes the binary

CXX      = clang++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

BIN  = stage0_sim
SRCS = main.cpp analog_sim.cpp
HDRS = analog_sim.h

$(BIN): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) -o $(BIN) $(SRCS)

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)

.PHONY: run clean
