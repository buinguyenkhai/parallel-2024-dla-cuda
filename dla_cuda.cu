#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdbool.h>

// --- Runtime Configuration Variables ---
int GRID_SIZE_RUNTIME = 200;
int NUM_PARTICLES_RUNTIME = 1000;
int MAX_STEPS_RUNTIME = 7000;
int INITIAL_FROZEN_POINTS_RUNTIME = 4; 
size_t MAX_RLE_LINE_BUFFER_SIZE_RUNTIME;


#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error \"%s\" at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// --- Global Device Pointers ---
int* d_frozen_grid;
int* d_particle_grid;
int* d_contact_grid;
int* d_particles;
curandState* d_rand_states;
int* d_active_particle_count_atomic;
int* d_temp_particles;

// --- Host variable for active particles ---
int h_active_particles;

// --- Kernel Definitions ---
__global__ void init_frozen_grid_kernel(int* frozen_grid, int size_val) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < size_val && y < size_val) {
        frozen_grid[y * size_val + x] = 0;
    }
}

__global__ void init_particles_kernel(int* particles_arr, curandState* rand_states_arr, int num_particles_val, int size_val, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_particles_val) {
        curand_init(seed, i, 0, &rand_states_arr[i]);
        particles_arr[i * 2 + 0] = curand_uniform(&rand_states_arr[i]) * size_val; 
        particles_arr[i * 2 + 1] = curand_uniform(&rand_states_arr[i]) * size_val; 
        if (particles_arr[i * 2 + 0] >= size_val) particles_arr[i * 2 + 0] = size_val - 1;
        if (particles_arr[i * 2 + 1] >= size_val) particles_arr[i * 2 + 1] = size_val - 1;
    }
}

__global__ void clear_grid_kernel(int* grid, int total_size_val, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size_val) {
        grid[idx] = value;
    }
}

__global__ void random_walk_kernel(int* particles_arr, curandState* rand_states_arr, int current_active_particles, int size_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < current_active_particles) {
        curandState localState = rand_states_arr[i]; 
        int dx = (int)(floorf(curand_uniform(&localState) * 3.0f)) - 1;
        int dy = (int)(floorf(curand_uniform(&localState) * 3.0f)) - 1;
        int current_x = particles_arr[i * 2 + 0];
        int current_y = particles_arr[i * 2 + 1];
        current_x += dx; current_y += dy;
        if (current_x < 0) current_x = 0; if (current_x >= size_val) current_x = size_val - 1;
        if (current_y < 0) current_y = 0; if (current_y >= size_val) current_y = size_val - 1;
        particles_arr[i * 2 + 0] = current_x; particles_arr[i * 2 + 1] = current_y;
        rand_states_arr[i] = localState;
    }
}

__global__ void update_particle_grid_kernel(int* particles_arr, int* particle_grid_arr, int current_active_particles, int size_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < current_active_particles) {
        int x = particles_arr[i * 2 + 0]; int y = particles_arr[i * 2 + 1];
        particle_grid_arr[y * size_val + x] = 1; 
    }
}

__global__ void generate_contact_grid_kernel(int* frozen_grid_arr, int* contact_grid_arr, int size_val) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < size_val && y < size_val) {
        if (frozen_grid_arr[y * size_val + x] == 1) { 
            for (int dy_kernel = -1; dy_kernel <= 1; dy_kernel++) for (int dx_kernel = -1; dx_kernel <= 1; dx_kernel++) {
                int nx = x + dx_kernel; int ny = y + dy_kernel;
                if (nx >= 0 && nx < size_val && ny >= 0 && ny < size_val) {
                    atomicAdd(&contact_grid_arr[ny * size_val + nx], 1); 
                }
            }
        }
    }
}

__global__ void calculate_frozen_grid_kernel(int* frozen_grid_arr, int* particle_grid_arr, int* contact_grid_arr, int size_val) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < size_val && y < size_val) {
        if (contact_grid_arr[y * size_val + x] > 0 && particle_grid_arr[y * size_val + x] == 1) { 
            frozen_grid_arr[y * size_val + x] = 1; 
        }
    }
}

__global__ void remove_frozen_particles_kernel(
    int* current_particles_arr, int* new_particles_buffer_arr, int* frozen_grid_arr,
    int current_active_count, int* new_active_count_atomic_ptr, int size_val, int max_particles_allowed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < current_active_count) {
        int px = current_particles_arr[i * 2 + 0]; int py = current_particles_arr[i * 2 + 1];
        if (frozen_grid_arr[py * size_val + px] == 0) { 
            int write_idx = atomicAdd(new_active_count_atomic_ptr, 1); 
            if (write_idx < max_particles_allowed) { 
                new_particles_buffer_arr[write_idx * 2 + 0] = px;
                new_particles_buffer_arr[write_idx * 2 + 1] = py;
            }
        }
    }
}

void host_init_grids_specific_points(int size_val, int num_initial_points) {
    int val = 1;

    if (size_val > 0) { 
        if (num_initial_points == 1) {
            int mid_r = size_val / 2;
            int mid_c = size_val / 2;
            CUDA_CHECK(cudaMemcpy(&d_frozen_grid[mid_r * size_val + mid_c], &val, sizeof(int), cudaMemcpyHostToDevice));
        } else if (num_initial_points > 1) {
            for (int k = 0; k < num_initial_points; k++) {
                if (size_val == 0) break; 
                int r_idx = rand() % size_val; 
                int c_idx = rand() % size_val; 
                CUDA_CHECK(cudaMemcpy(&d_frozen_grid[r_idx * size_val + c_idx], &val, sizeof(int), cudaMemcpyHostToDevice));
            }
        }
    }
}

void record_state_rle(FILE* f, int step, int* h_frozen_grid_cpu, int* h_particle_grid_cpu) {
    char *line_buffer = (char*)malloc(MAX_RLE_LINE_BUFFER_SIZE_RUNTIME);
    if (!line_buffer) {
        fprintf(stderr, "CUDA RLE Error: Failed to allocate line_buffer.\n");
        return;
    }
    char* current_pos_in_line = line_buffer;
    size_t remaining_buffer_size = MAX_RLE_LINE_BUFFER_SIZE_RUNTIME;
    int written;

    written = snprintf(current_pos_in_line, remaining_buffer_size, "%d,", step);
     if (written < 0 || (size_t)written >= remaining_buffer_size) { 
        fprintf(stderr, "CUDA RLE Error: Buffer too small for step number.\n"); free(line_buffer); return; 
    }
    current_pos_in_line += written; remaining_buffer_size -= written;

    int prev = -1; int count = 0;
    for (int i = 0; i < GRID_SIZE_RUNTIME; i++) { 
        for (int j = 0; j < GRID_SIZE_RUNTIME; j++) {
            int val = 0;
            if (h_frozen_grid_cpu[i * GRID_SIZE_RUNTIME + j] == 1) val = 2;
            else if (h_particle_grid_cpu[i * GRID_SIZE_RUNTIME + j] == 1) val = 1;
            
            if (val == prev) { 
                count++; 
            } else {
                if (prev != -1) {
                    written = snprintf(current_pos_in_line, remaining_buffer_size, "%dx%d,", prev, count);
                    if (written < 0 || (size_t)written >= remaining_buffer_size) {
                        fprintf(stderr, "CUDA RLE Error: Buffer too small for segment.\n"); 
                        goto end_line_write_cuda_dynamic;
                    }
                    current_pos_in_line += written; remaining_buffer_size -= written;
                }
                prev = val; count = 1;
            }
        }
    }
end_line_write_cuda_dynamic:
    if (count > 0 && prev != -1) {
        written = snprintf(current_pos_in_line, remaining_buffer_size, "%dx%d\n", prev, count);
        if (written < 0 || (size_t)written >= remaining_buffer_size) {
             fprintf(stderr, "CUDA RLE Error: Buffer too small for last segment.\n");
        }
    } else if (prev == -1 && count == 0) { 
        written = snprintf(current_pos_in_line, remaining_buffer_size, "0x%ld\n", (long)GRID_SIZE_RUNTIME*GRID_SIZE_RUNTIME);
         if (written < 0 || (size_t)written >= remaining_buffer_size) {
              fprintf(stderr, "CUDA RLE Error: Buffer too small for empty grid segment.\n");
         }
    } else if (count > 0 && prev == -1) { 
         written = snprintf(current_pos_in_line, remaining_buffer_size, "0x%ld\n", (long)GRID_SIZE_RUNTIME*GRID_SIZE_RUNTIME); 
         if (written < 0 || (size_t)written >= remaining_buffer_size) {
              fprintf(stderr, "CUDA RLE Error: Buffer too small for fallback empty grid segment.\n");
         }
    }
    fputs(line_buffer, f); 
    free(line_buffer); 
}

void reinitialize_cuda_state(unsigned long long seed_param) { 
    h_active_particles = NUM_PARTICLES_RUNTIME; 

    dim3 threadsPerBlock2D(16, 16);
    dim3 numBlocks2D((GRID_SIZE_RUNTIME + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
                     (GRID_SIZE_RUNTIME + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);
    int threadsPerBlock1D = 256;

    init_frozen_grid_kernel<<<numBlocks2D, threadsPerBlock2D>>>(d_frozen_grid, GRID_SIZE_RUNTIME);
    CUDA_CHECK(cudaGetLastError()); 
    host_init_grids_specific_points(GRID_SIZE_RUNTIME, INITIAL_FROZEN_POINTS_RUNTIME); 

    int numBlocksParticles = (NUM_PARTICLES_RUNTIME + threadsPerBlock1D - 1) / threadsPerBlock1D;
    init_particles_kernel<<<numBlocksParticles, threadsPerBlock1D>>>(d_particles, d_rand_states, NUM_PARTICLES_RUNTIME, GRID_SIZE_RUNTIME, seed_param);
    CUDA_CHECK(cudaGetLastError());

    int totalGridCells = GRID_SIZE_RUNTIME * GRID_SIZE_RUNTIME;
    int numBlocksGridClear = (totalGridCells + threadsPerBlock1D - 1) / threadsPerBlock1D;
    clear_grid_kernel<<<numBlocksGridClear, threadsPerBlock1D>>>(d_particle_grid, totalGridCells, 0);
    CUDA_CHECK(cudaGetLastError());
    clear_grid_kernel<<<numBlocksGridClear, threadsPerBlock1D>>>(d_contact_grid, totalGridCells, 0);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemset(d_active_particle_count_atomic, 0, sizeof(int))); 
    CUDA_CHECK(cudaDeviceSynchronize()); 
}

void run_simulation_cuda(bool enable_rle_recording, const char* output_filename,
                         int* h_frozen_grid_cpu_buf, int* h_particle_grid_cpu_buf, int* steps_executed) {
    FILE* f_rle = NULL;
    *steps_executed = 0; 

    if (enable_rle_recording) {
        if (output_filename != NULL && remove(output_filename) == 0) {
        }
        f_rle = fopen(output_filename, "w");
        if (f_rle == NULL) {
            fprintf(stderr, "CUDA RLE run: Cannot open output file: %s\n", output_filename);
            return; 
        }
        printf("CUDA RLE run: Output will be written to %s\n", output_filename);
    }

    dim3 threadsPerBlock2D(16, 16);
    dim3 numBlocks2D((GRID_SIZE_RUNTIME + threadsPerBlock2D.x - 1) / threadsPerBlock2D.x,
                     (GRID_SIZE_RUNTIME + threadsPerBlock2D.y - 1) / threadsPerBlock2D.y);
    int threadsPerBlock1D = 256;
    int totalGridCells = GRID_SIZE_RUNTIME * GRID_SIZE_RUNTIME;
    int numBlocksGridClear = (totalGridCells + threadsPerBlock1D - 1) / threadsPerBlock1D;

    for (int step = 0; step < MAX_STEPS_RUNTIME; step++) {
        if (h_active_particles == 0) {
            if (enable_rle_recording && f_rle != NULL){ 
                 printf("CUDA RLE run: No active particles left at step %d. Stopping.\n", step);
            }
            break; 
        }

        int numBlocksActiveParticles = (h_active_particles + threadsPerBlock1D - 1) / threadsPerBlock1D;

        if (h_active_particles > 0) {
            random_walk_kernel<<<numBlocksActiveParticles, threadsPerBlock1D>>>(d_particles, d_rand_states, h_active_particles, GRID_SIZE_RUNTIME);
            CUDA_CHECK(cudaGetLastError()); 
        }

        clear_grid_kernel<<<numBlocksGridClear, threadsPerBlock1D>>>(d_particle_grid, totalGridCells, 0);
        CUDA_CHECK(cudaGetLastError());
        
        if (h_active_particles > 0) {
            update_particle_grid_kernel<<<numBlocksActiveParticles, threadsPerBlock1D>>>(d_particles, d_particle_grid, h_active_particles, GRID_SIZE_RUNTIME);
            CUDA_CHECK(cudaGetLastError());
        }

        clear_grid_kernel<<<numBlocksGridClear, threadsPerBlock1D>>>(d_contact_grid, totalGridCells, 0);
        CUDA_CHECK(cudaGetLastError());
        
        generate_contact_grid_kernel<<<numBlocks2D, threadsPerBlock2D>>>(d_frozen_grid, d_contact_grid, GRID_SIZE_RUNTIME);
        CUDA_CHECK(cudaGetLastError());
        
        calculate_frozen_grid_kernel<<<numBlocks2D, threadsPerBlock2D>>>(d_frozen_grid, d_particle_grid, d_contact_grid, GRID_SIZE_RUNTIME);
        CUDA_CHECK(cudaGetLastError());

        if (enable_rle_recording && f_rle != NULL) {
            CUDA_CHECK(cudaMemcpy(h_frozen_grid_cpu_buf, d_frozen_grid, (size_t)GRID_SIZE_RUNTIME * GRID_SIZE_RUNTIME * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_particle_grid_cpu_buf, d_particle_grid, (size_t)GRID_SIZE_RUNTIME * GRID_SIZE_RUNTIME * sizeof(int), cudaMemcpyDeviceToHost));
            record_state_rle(f_rle, step, h_frozen_grid_cpu_buf, h_particle_grid_cpu_buf);
        }

        CUDA_CHECK(cudaMemset(d_active_particle_count_atomic, 0, sizeof(int))); 
        if (h_active_particles > 0) {
            remove_frozen_particles_kernel<<<numBlocksActiveParticles, threadsPerBlock1D>>>(
                d_particles, d_temp_particles, d_frozen_grid,
                h_active_particles, d_active_particle_count_atomic, GRID_SIZE_RUNTIME, NUM_PARTICLES_RUNTIME);
            CUDA_CHECK(cudaGetLastError());
        }
        
        CUDA_CHECK(cudaMemcpy(&h_active_particles, d_active_particle_count_atomic, sizeof(int), cudaMemcpyDeviceToHost)); 
        if (h_active_particles > 0) {
            size_t bytes_to_copy = (size_t)h_active_particles * 2 * sizeof(int);
            if (bytes_to_copy <= (size_t)NUM_PARTICLES_RUNTIME * 2 * sizeof(int)) {
                 CUDA_CHECK(cudaMemcpy(d_particles, d_temp_particles, bytes_to_copy, cudaMemcpyDeviceToDevice));
            } else {
                fprintf(stderr, "Error: Attempting to copy too many particles after compaction. Active: %d, Max: %d\n", h_active_particles, NUM_PARTICLES_RUNTIME);
                h_active_particles = 0; 
            }
        }
        (*steps_executed)++; 
    }

    if (!enable_rle_recording) {
        CUDA_CHECK(cudaDeviceSynchronize()); 
    }

    if (enable_rle_recording && f_rle != NULL) {
        fclose(f_rle);
        printf("CUDA RLE run: Simulation data recording complete.\n");
    }
}

void print_cuda_usage(const char* prog_name) {
    fprintf(stderr, "Usage: %s <grid_size> <num_particles> <max_steps> <initial_frozen_points> <benchmark_runs> [generate_csv]\n", prog_name);
    fprintf(stderr, "  <grid_size>: Integer, size of one dimension of the square grid (e.g., 200).\n");
    fprintf(stderr, "  <num_particles>: Integer, initial number of moving particles (e.g., 1000).\n");
    fprintf(stderr, "  <max_steps>: Integer, maximum simulation steps (e.g., 7000).\n");
    fprintf(stderr, "  <initial_frozen_points>: Integer, number of initial frozen seed points (e.g., 0, 1, or more).\n");
    fprintf(stderr, "                           If 1, placed in the center. If >1, placed randomly.\n");
    fprintf(stderr, "  <benchmark_runs>: Integer, number of times to run the simulation for benchmarking (e.g., 5).\n");
    fprintf(stderr, "  [generate_csv]: Optional string 'generate_csv' to output RLE data to a file.\n");
    fprintf(stderr, "Example: %s 200 1000 7000 1 5 generate_csv (1 point in center)\n", prog_name);
    fprintf(stderr, "Example: %s 200 1000 7000 4 5 generate_csv (4 random points)\n", prog_name);
    fprintf(stderr, "Example (benchmark only): %s 200 1000 7000 4 5\n", prog_name);
}


int main(int argc, char *argv[]) { 
    unsigned long long seed = 42; 
    const char* cuda_output_filename = "dla_output_rle_cuda.csv";
    bool perform_rle_run = false;
    int num_benchmark_runs = 1;
    int cuda_rle_actual_steps = 0; 

    if (argc < 6 || argc > 7) { 
        print_cuda_usage(argv[0]);
        return 1;
    }

    GRID_SIZE_RUNTIME = atoi(argv[1]);
    NUM_PARTICLES_RUNTIME = atoi(argv[2]);
    MAX_STEPS_RUNTIME = atoi(argv[3]);
    INITIAL_FROZEN_POINTS_RUNTIME = atoi(argv[4]); 
    num_benchmark_runs = atoi(argv[5]);            

    if (GRID_SIZE_RUNTIME <= 0 || NUM_PARTICLES_RUNTIME <= 0 || MAX_STEPS_RUNTIME < 0 || 
        INITIAL_FROZEN_POINTS_RUNTIME < 0 || num_benchmark_runs <= 0) {
        fprintf(stderr, "Error: grid_size, num_particles, benchmark_runs must be positive.\n");
        fprintf(stderr, "       max_steps and initial_frozen_points must be non-negative.\n");
        print_cuda_usage(argv[0]);
        return 1;
    }
    MAX_RLE_LINE_BUFFER_SIZE_RUNTIME = (size_t)GRID_SIZE_RUNTIME * GRID_SIZE_RUNTIME * 8 + 100;


    if (argc == 7 && strcmp(argv[6], "generate_csv") == 0) { 
        perform_rle_run = true;
    }
    
    srand((unsigned int)seed); 

    printf("CUDA Simulation Configuration:\n");
    printf("  Grid Size: %d x %d\n", GRID_SIZE_RUNTIME, GRID_SIZE_RUNTIME);
    printf("  Number of Particles: %d\n", NUM_PARTICLES_RUNTIME);
    printf("  Max Steps: %d\n", MAX_STEPS_RUNTIME);
    printf("  Initial Frozen Points: %d (if 1, center; if >1, random)\n", INITIAL_FROZEN_POINTS_RUNTIME);
    printf("  Benchmark Runs: %d\n", num_benchmark_runs);
    printf("  Generate CSV: %s\n", perform_rle_run ? "Yes" : "No");
    printf("--------------------------------------------------\n");


    CUDA_CHECK(cudaMalloc((void**)&d_frozen_grid, (size_t)GRID_SIZE_RUNTIME * GRID_SIZE_RUNTIME * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_particle_grid, (size_t)GRID_SIZE_RUNTIME * GRID_SIZE_RUNTIME * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_contact_grid, (size_t)GRID_SIZE_RUNTIME * GRID_SIZE_RUNTIME * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_particles, (size_t)NUM_PARTICLES_RUNTIME * 2 * sizeof(int))); 
    CUDA_CHECK(cudaMalloc((void**)&d_rand_states, (size_t)NUM_PARTICLES_RUNTIME * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc((void**)&d_active_particle_count_atomic, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_temp_particles, (size_t)NUM_PARTICLES_RUNTIME * 2 * sizeof(int)));

    int* h_frozen_grid_cpu = (int*)malloc((size_t)GRID_SIZE_RUNTIME * GRID_SIZE_RUNTIME * sizeof(int));
    int* h_particle_grid_cpu = (int*)malloc((size_t)GRID_SIZE_RUNTIME * GRID_SIZE_RUNTIME * sizeof(int));
    if (!h_frozen_grid_cpu || !h_particle_grid_cpu) {
        fprintf(stderr, "Failed to allocate host memory for RLE buffers.\n"); 
        cudaFree(d_frozen_grid); cudaFree(d_particle_grid); cudaFree(d_contact_grid);
        cudaFree(d_particles); cudaFree(d_rand_states);
        cudaFree(d_active_particle_count_atomic); cudaFree(d_temp_particles);
        return 1;
    }

    float total_milliseconds = 0;
    long long total_benchmark_steps_cuda = 0; // Use long long for total steps
    printf("\nStarting CUDA Benchmark Phase (%d timed runs)\n", num_benchmark_runs);
    int cuda_benchmark_iteration_steps = 0;

    for (int i = 0; i < num_benchmark_runs; ++i) {
        srand((unsigned int)seed + i); 
        reinitialize_cuda_state(seed + i); 

        cudaEvent_t start_event, stop_event;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        
        CUDA_CHECK(cudaDeviceSynchronize()); 
        CUDA_CHECK(cudaEventRecord(start_event, 0));

        run_simulation_cuda(false, NULL, h_frozen_grid_cpu, h_particle_grid_cpu, &cuda_benchmark_iteration_steps);

        CUDA_CHECK(cudaEventRecord(stop_event, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event)); 
        float current_run_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&current_run_ms, start_event, stop_event));
        total_milliseconds += current_run_ms;
        total_benchmark_steps_cuda += cuda_benchmark_iteration_steps;
        
        double time_per_step_ms = 0;
        if (cuda_benchmark_iteration_steps > 0) {
            time_per_step_ms = (double)current_run_ms / cuda_benchmark_iteration_steps;
        }

        printf("CUDA Benchmark Run %d/%d: %d steps, %f ms total, %e ms/step (%e seconds/step).\n", 
               i + 1, num_benchmark_runs, cuda_benchmark_iteration_steps, current_run_ms, 
               time_per_step_ms, time_per_step_ms / 1000.0);

        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
    }
    if (num_benchmark_runs > 0) {
        float average_total_ms = total_milliseconds / num_benchmark_runs;
        printf("CUDA Average Total Benchmark Time (over %d runs): %f ms (%f seconds).\n", 
               num_benchmark_runs, average_total_ms, average_total_ms / 1000.0f);
        if (total_benchmark_steps_cuda > 0) {
            double average_time_per_step_ms = (double)total_milliseconds / total_benchmark_steps_cuda;
            printf("CUDA Average Benchmark Time Per Step (over %lld steps): %e ms/step (%e seconds/step).\n", 
                   total_benchmark_steps_cuda, average_time_per_step_ms, average_time_per_step_ms / 1000.0);
        } else {
            printf("CUDA Average Benchmark Time Per Step: N/A (0 steps executed).\n");
        }
    }
    printf("--------------------------------------------------\n");
    

    if (perform_rle_run) {
        printf("\nStarting CUDA RLE Output Generation Run\n");
        srand((unsigned int)seed); 
        reinitialize_cuda_state(seed); 
        run_simulation_cuda(true, cuda_output_filename, h_frozen_grid_cpu, h_particle_grid_cpu, &cuda_rle_actual_steps);
        printf("--------------------------------------------------\n");
    }
    
    free(h_frozen_grid_cpu);
    free(h_particle_grid_cpu);
    cudaFree(d_frozen_grid); cudaFree(d_particle_grid); cudaFree(d_contact_grid);
    cudaFree(d_particles); cudaFree(d_rand_states);
    cudaFree(d_active_particle_count_atomic); cudaFree(d_temp_particles);
    
    printf("\nCUDA simulation tasks complete.\n");
    if (perform_rle_run) {
        printf("The CUDA RLE simulation ran for %d steps.\n", cuda_rle_actual_steps);
    }
    return 0;
}
