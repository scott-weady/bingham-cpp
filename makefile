
# Compiler and flags
CXX := g++
CXXFLAGS := -O3 -march=native -ftree-vectorize -fopenmp -Iinclude -std=c++23
LDFLAGS := -lfftw3_omp -lfftw3 -lm

# Directories
SRC_DIR := src
BUILD_DIR := build

# All source files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)

# Name of the final executable
TARGET := $(BUILD_DIR)/main

# Default target
all: $(TARGET)

# Compile all sources into one executable
$(TARGET): $(SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $@ $(LDFLAGS)

# Clean rule
clean:
	$(RM) -r $(BUILD_DIR)

.PHONY: all clean
