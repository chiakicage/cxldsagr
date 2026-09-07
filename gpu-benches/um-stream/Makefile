NVCC := nvcc

# internal flags
SM      	?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//g')
NVCCFLAGS   := -std=c++11 -O3 -gencode arch=compute_$(SM),code=sm_$(SM) --compiler-options="-O2 -pipe -march=native -Wall -fopenmp" -Xcompiler -rdynamic --generate-line-info
CCFLAGS     := 
LDFLAGS     := -L/opt/cuda/lib64 -lcublas
NAME 		:= um-stream
PREFIX		:= .


$(PREFIX)/$(NAME): main.cu Makefile ../dtime.hpp ../MeasurementSeries.hpp ../gpu-error.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -f ./$(NAME)

