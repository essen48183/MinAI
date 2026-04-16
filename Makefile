# Minimalist AI: one file, no deps.
# Apple Silicon native. Needs Xcode Command Line Tools (clang++).
CXX      := clang++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
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
