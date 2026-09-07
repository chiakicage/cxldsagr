NVCC := nvcc

SM ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//g')
NVCCFLAGS := -std=c++17 -O3 -gencode arch=compute_$(SM),code=sm_$(SM) --compiler-options="-O2 -pipe -Wall"
LDFLAGS := -lnvidia-ml
NAME := pcie-host

cuda-$(NAME): main.cu Makefile
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

cpu-mem-bw: cpu_mem_bw.cpp Makefile
	g++ -std=c++17 -O3 -march=native -pthread -o $@ $<

cpu-mem-stream: cpu_mem_stream.cpp Makefile
	g++ -std=c++17 -O3 -march=native -pthread -o $@ $<

clean:
	rm -f cuda-$(NAME) cpu-mem-bw cpu-mem-stream
