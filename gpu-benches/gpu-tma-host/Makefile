NVCC := nvcc

SM ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//g')
NVCCFLAGS := -std=c++17 -O3 -gencode arch=compute_$(SM),code=sm_$(SM) --compiler-options="-O2 -pipe -Wall"
NAME := tma-host-prefetch

cuda-$(NAME): main.cu Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f cuda-$(NAME)
