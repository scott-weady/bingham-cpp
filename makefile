
# params ?= params.h

# CXX = g++
# CXXFLAGS = -include $(params) -O3 -march=native -ftree-vectorize -fopenmp
# FFTFLAGS = -lfftw3_omp -lfftw3 -lm

# TARGETS = $(basename $(wildcard *.cpp))
# all : $(TARGETS)

# %:%.cpp *.h
# 	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@ $(FFTFLAGS)

# clean:
# 	-$(RM) $(TARGETS) *~

# .PHONY: all, clean

# Compiler and flags
CXX := g++
CXXFLAGS := -O3 -march=native -ftree-vectorize -fopenmp -Iinclude
LDFLAGS := -lfftw3_omp -lfftw3 -lm

# Optional: include a parameter header
# PARAMS ?= params.h
# CXXFLAGS += -include $(PARAMS)

# Directories
SRC_DIR := src
BUILD_DIR := build

# Find all source files and set corresponding output binaries
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
TARGETS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%, $(SRCS))

# Default target
all: $(TARGETS)

# Rule to compile each .cpp directly to a binary (no .o files)
$(BUILD_DIR)/%: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Clean rule
clean:
	$(RM) -r $(BUILD_DIR)

.PHONY: all clean
