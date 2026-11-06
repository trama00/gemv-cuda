# GEMV with CUDA — build
#
#   make                 release build (default DIM_SIZE=16384)
#   make DIM_SIZE=4096   smaller problem, for GPUs with little memory
#   make run             build and run
#   make clean

NVCC      ?= nvcc
DIM_SIZE  ?= 16384
THREADS   ?= 64
ARCH      ?= -arch=native

NVFLAGS   := -O3 -std=c++17 $(ARCH) -DDIM_SIZE=$(DIM_SIZE) -DTHREADS_PER_BLOCK=$(THREADS)
INCLUDES  := -Iinclude
TARGET    := gemv

SRC       := src/main.cu src/kernels.cu src/reference.cpp

.PHONY: all run clean
all: $(TARGET)

$(TARGET): $(SRC) include/gemv.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o
