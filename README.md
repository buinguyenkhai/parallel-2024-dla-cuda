# How to use

## Compiling

### C code
MinGW / GNU Compiler Collection
```
gcc dla_c.c -o dla_c -O3 -Wall -lm
```
Microsoft Visual C++ Compiler (MSVC)
```
cl dla_c.c /Fe:dla_c.exe /O2 /W4 /link
```
### CUDA code
NVIDIA's CUDA Compiler
```
nvcc dla_cuda.cu -o dla_cuda -O3 -arch=sm_XX
```
Replace XX with your [GPU's compute capability](https://developer.nvidia.com/cuda-gpus), for example RTX 3050 is 8.6 -> sm_86

## Running
```
./dla_c_or_cuda grid_size num_particles max_steps num_runs generate_csv
```
Example for saving csv files
```
./dla_c 200 1000 7000 10 generate_csv
./dla_cuda 200 1000 7000 10 generate_csv
```
Example for simulation only
```
./dla_c 200 1000 7000 10
./dla_cuda 200 1000 7000 10
```

## Visualizing
Change FILENAME and SIZE and run visualize.py