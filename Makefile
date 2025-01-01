# Makefile for Parallel K-Means with OpenMP (CPU-Only)

# Compiler
CXX = g++-14

# Flags
CXXFLAGS = -O2 -fopenmp -std=c++11

# Source Files
SRC = main.cpp

# Output Executable
TARGET = kmeans_parallel

# Build Rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

# Clean Rule
clean:
	rm -f $(TARGET)
