# Minimalist AI: one file, no deps. Portable across Mac and Linux.
# On Windows, use CMake instead (see CMakeLists.txt and README.md).
#
# CXX defaults to the system C++ compiler ("c++"), which resolves to clang++
# on Mac and g++ on Linux. Override with e.g.  make CXX=g++-13
CXX      ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
TARGET   := minai
SRC      := minai.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
