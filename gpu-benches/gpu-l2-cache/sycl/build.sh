clang++ -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda sycl-gpu-l2-cache.cpp -o  sycl-gpu-l2-cache -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_80
./sycl-gpu-l2-cache