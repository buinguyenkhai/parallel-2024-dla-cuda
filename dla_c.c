#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>

// --- Global Pointers for Dynamically Allocated Arrays ---
// These will be allocated based on runtime grid_size and num_particles_val
int* frozen_grid_g = NULL;    // 1D array representing 2D grid
int* particle_grid_g = NULL;  // 1D array
int* contact_grid_g = NULL;   // 1D array
int* particles_g = NULL;      // 2D array (num_particles_val x 2)
int active_particles_g;

// --- Runtime Configuration Variables ---
int GRID_SIZE_RUNTIME = 200;
int NUM_PARTICLES_RUNTIME = 1000;
int MAX_STEPS_RUNTIME = 7000;
size_t MAX_RLE_LINE_BUFFER_SIZE_RUNTIME;


// Helper macro for 2D to 1D indexing (assuming row-major order)
#define IDX(r, c, width) ((r) * (width) + (c))

// --- Memory Allocation and Deallocation ---
bool allocate_global_arrays(int r_grid_size, int r_num_particles) {
    // Deallocate first if already allocated (e.g., for multiple test runs in one program execution, though not current design)
    if (frozen_grid_g) {free(frozen_grid_g);}
    if (particle_grid_g) {free(particle_grid_g);}
    if (contact_grid_g) {free(contact_grid_g);}
    if (particles_g) {free(particles_g);}

    frozen_grid_g = (int*)malloc(r_grid_size * r_grid_size * sizeof(int));
    particle_grid_g = (int*)malloc(r_grid_size * r_grid_size * sizeof(int));
    contact_grid_g = (int*)malloc(r_grid_size * r_grid_size * sizeof(int));
    particles_g = (int*)malloc(r_num_particles * 2 * sizeof(int)); // For x and y coordinates

    if (!frozen_grid_g || !particle_grid_g || !contact_grid_g || !particles_g) {
        fprintf(stderr, "Error: Failed to allocate global arrays.\n");
        // Free any that were successfully allocated before failing
        if (frozen_grid_g) {free(frozen_grid_g); frozen_grid_g = NULL;}
        if (particle_grid_g) {free(particle_grid_g); particle_grid_g = NULL;}
        if (contact_grid_g) {free(contact_grid_g); contact_grid_g = NULL;}
        if (particles_g) {free(particles_g); particles_g = NULL;}
        return false;
    }
    return true;
}

void free_global_arrays() {
    if (frozen_grid_g) {free(frozen_grid_g); frozen_grid_g = NULL;}
    if (particle_grid_g) {free(particle_grid_g); particle_grid_g = NULL;}
    if (contact_grid_g) {free(contact_grid_g); contact_grid_g = NULL;}
    if (particles_g) {free(particles_g); particles_g = NULL;}
}


void init_grids() {
    for (int i = 0; i < GRID_SIZE_RUNTIME; i++)
        for (int j = 0; j < GRID_SIZE_RUNTIME; j++)
            frozen_grid_g[IDX(i, j, GRID_SIZE_RUNTIME)] = 0;

    int margin = 5;
    if (GRID_SIZE_RUNTIME > 2 * margin) { // Ensure margin is valid
        frozen_grid_g[IDX(margin, margin, GRID_SIZE_RUNTIME)] = 1;
        frozen_grid_g[IDX(margin, GRID_SIZE_RUNTIME - margin - 1, GRID_SIZE_RUNTIME)] = 1;
        frozen_grid_g[IDX(GRID_SIZE_RUNTIME - margin - 1, margin, GRID_SIZE_RUNTIME)] = 1;
        frozen_grid_g[IDX(GRID_SIZE_RUNTIME - margin - 1, GRID_SIZE_RUNTIME - margin - 1, GRID_SIZE_RUNTIME)] = 1;
    } else if (GRID_SIZE_RUNTIME > 0) { // Fallback for small grids
        frozen_grid_g[IDX(0,0,GRID_SIZE_RUNTIME)] = 1;
    }


    for (int i = 0; i < NUM_PARTICLES_RUNTIME; i++) {
        particles_g[i * 2 + 0] = rand() % GRID_SIZE_RUNTIME; // x
        particles_g[i * 2 + 1] = rand() % GRID_SIZE_RUNTIME; // y
    }
    active_particles_g = NUM_PARTICLES_RUNTIME;
}

void random_walk() {
    for (int i = 0; i < GRID_SIZE_RUNTIME; i++)
        for (int j = 0; j < GRID_SIZE_RUNTIME; j++)
            particle_grid_g[IDX(i, j, GRID_SIZE_RUNTIME)] = 0;

    for (int i = 0; i < active_particles_g; i++) {
        int dx = rand() % 3 - 1;
        int dy = rand() % 3 - 1;

        int current_x = particles_g[i * 2 + 0];
        int current_y = particles_g[i * 2 + 1];

        current_x += dx;
        current_y += dy;

        if (current_x < 0) current_x = 0;
        if (current_x >= GRID_SIZE_RUNTIME) current_x = GRID_SIZE_RUNTIME - 1;
        if (current_y < 0) current_y = 0;
        if (current_y >= GRID_SIZE_RUNTIME) current_y = GRID_SIZE_RUNTIME - 1;
        
        particles_g[i * 2 + 0] = current_x;
        particles_g[i * 2 + 1] = current_y;

        particle_grid_g[IDX(current_x, current_y, GRID_SIZE_RUNTIME)] = 1;
    }
}

void generate_contact_grid() {
    for (int i = 0; i < GRID_SIZE_RUNTIME; i++)
        for (int j = 0; j < GRID_SIZE_RUNTIME; j++)
            contact_grid_g[IDX(i, j, GRID_SIZE_RUNTIME)] = 0;

    for (int x = 0; x < GRID_SIZE_RUNTIME; x++) {
        for (int y = 0; y < GRID_SIZE_RUNTIME; y++) {
            if (frozen_grid_g[IDX(x, y, GRID_SIZE_RUNTIME)] == 1) {
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < GRID_SIZE_RUNTIME && ny >= 0 && ny < GRID_SIZE_RUNTIME) {
                            contact_grid_g[IDX(nx, ny, GRID_SIZE_RUNTIME)] += 1;
                        }
                    }
                }
            }
        }
    }
}

void calculate_frozen_grid() {
    for (int i = 0; i < GRID_SIZE_RUNTIME; i++) {
        for (int j = 0; j < GRID_SIZE_RUNTIME; j++) {
            if (contact_grid_g[IDX(i,j,GRID_SIZE_RUNTIME)] > 0 && particle_grid_g[IDX(i,j,GRID_SIZE_RUNTIME)] == 1) {
                frozen_grid_g[IDX(i,j,GRID_SIZE_RUNTIME)] = 1;
            }
        }
    }
}

void remove_frozen_particles() {
    // Temporary buffer for new particles. Could also be dynamically allocated if NUM_PARTICLES_RUNTIME is huge.
    // For now, assuming NUM_PARTICLES_RUNTIME allows stack allocation for this temp buffer.
    // If it's very large, this should also be heap allocated.
    int* new_particles_temp = (int*)malloc(NUM_PARTICLES_RUNTIME * 2 * sizeof(int));
    if (!new_particles_temp) {
        fprintf(stderr, "Error: Failed to allocate temporary particle buffer in remove_frozen_particles.\n");
        return; // Or handle error more gracefully
    }
    int new_count = 0;

    for (int i = 0; i < active_particles_g; i++) {
        int x = particles_g[i * 2 + 0];
        int y = particles_g[i * 2 + 1];
        if (frozen_grid_g[IDX(x,y,GRID_SIZE_RUNTIME)] == 0) { // If not frozen
            if (new_count < NUM_PARTICLES_RUNTIME) {
                 new_particles_temp[new_count * 2 + 0] = x;
                 new_particles_temp[new_count * 2 + 1] = y;
                 new_count++;
            }
        }
    }

    for (int i = 0; i < new_count; i++) {
        particles_g[i * 2 + 0] = new_particles_temp[i * 2 + 0];
        particles_g[i * 2 + 1] = new_particles_temp[i * 2 + 1];
    }
    active_particles_g = new_count;
    free(new_particles_temp);
}

void record_state_rle(FILE *f, int step) {
    // Allocate RLE buffer dynamically or ensure static buffer is large enough
    char *line_buffer = (char*)malloc(MAX_RLE_LINE_BUFFER_SIZE_RUNTIME);
    if (!line_buffer) {
        fprintf(stderr, "C RLE Error: Failed to allocate line_buffer.\n");
        return;
    }

    char *current_pos_in_line = line_buffer;
    size_t remaining_buffer_size = MAX_RLE_LINE_BUFFER_SIZE_RUNTIME;
    int written;

    written = snprintf(current_pos_in_line, remaining_buffer_size, "%d,", step);
    if (written < 0 || (size_t)written >= remaining_buffer_size) { 
        fprintf(stderr, "C RLE Error: Buffer too small for step number.\n"); free(line_buffer); return; 
    }
    current_pos_in_line += written;
    remaining_buffer_size -= written;

    int prev = -1; 
    int count = 0;

    for (int i = 0; i < GRID_SIZE_RUNTIME; i++) {
        for (int j = 0; j < GRID_SIZE_RUNTIME; j++) {
            int val = 0; 
            if (frozen_grid_g[IDX(i,j,GRID_SIZE_RUNTIME)] == 1) val = 2; 
            else if (particle_grid_g[IDX(i,j,GRID_SIZE_RUNTIME)] == 1) val = 1; 

            if (val == prev) {
                count++;
            } else {
                if (prev != -1) { 
                    written = snprintf(current_pos_in_line, remaining_buffer_size, "%dx%d,", prev, count);
                    if (written < 0 || (size_t)written >= remaining_buffer_size) {
                        fprintf(stderr, "C RLE Error: Buffer too small for segment.\n");
                        goto end_line_write_c_dynamic; 
                    }
                    current_pos_in_line += written;
                    remaining_buffer_size -= written;
                }
                prev = val;
                count = 1;
            }
        }
    }

end_line_write_c_dynamic:
    if (count > 0 && prev != -1) {
        written = snprintf(current_pos_in_line, remaining_buffer_size, "%dx%d\n", prev, count);
         if (written < 0 || (size_t)written >= remaining_buffer_size) {
             fprintf(stderr, "C RLE Error: Buffer too small for last segment.\n");
         }
    } else if (prev == -1 && count == 0) { 
        written = snprintf(current_pos_in_line, remaining_buffer_size, "0x%ld\n", (long)GRID_SIZE_RUNTIME * GRID_SIZE_RUNTIME);
         if (written < 0 || (size_t)written >= remaining_buffer_size) {
              fprintf(stderr, "C RLE Error: Buffer too small for empty grid segment.\n");
         }
    } else if (count > 0 && prev == -1) { 
         written = snprintf(current_pos_in_line, remaining_buffer_size, "0x%ld\n", (long)GRID_SIZE_RUNTIME * GRID_SIZE_RUNTIME);
         if (written < 0 || (size_t)written >= remaining_buffer_size) {
              fprintf(stderr, "C RLE Error: Buffer too small for fallback empty grid segment.\n");
         }
    }

    fputs(line_buffer, f);
    free(line_buffer); 
}

void run_simulation_c(bool enable_rle_recording, const char* output_filename) {
    FILE *f_rle = NULL;
    if (enable_rle_recording) {
        if (output_filename != NULL && remove(output_filename) == 0) {
             // Optional: printf("C RLE run: Deleted existing output file: %s\n", output_filename);
        }
        f_rle = fopen(output_filename, "w");
        if (f_rle == NULL) {
            fprintf(stderr, "C RLE run: Cannot open output file: %s\n", output_filename);
            return;
        }
        printf("C RLE run: Output will be written to %s\n", output_filename);
    }

    for (int step = 0; step < MAX_STEPS_RUNTIME; step++) {
        if (active_particles_g == 0) {
            if (enable_rle_recording){
                 printf("C RLE run: No active particles left at step %d. Stopping.\n", step);
            }
            break;
        }
        random_walk();
        generate_contact_grid();
        calculate_frozen_grid();
        if (enable_rle_recording && f_rle != NULL) {
            record_state_rle(f_rle, step);
        }
        remove_frozen_particles();
    }

    if (enable_rle_recording && f_rle != NULL) {
        fclose(f_rle);
        printf("C RLE run: Simulation complete.\n");
    }
}

void print_usage(const char* prog_name) {
    fprintf(stderr, "Usage: %s <grid_size> <num_particles> <max_steps> <benchmark_runs> [generate_csv]\n", prog_name);
    fprintf(stderr, "Example: %s 200 1000 7000 5 generate_csv\n", prog_name);
    fprintf(stderr, "Example (benchmark only): %s 200 1000 7000 5\n", prog_name);
}

int main(int argc, char *argv[]) {
    const char* c_output_filename = "dla_output_rle.csv";
    bool perform_rle_run = false;
    int num_benchmark_runs = 1; // Default to 1 benchmark run if not specified

    if (argc < 5 || argc > 6) {
        print_usage(argv[0]);
        return 1;
    }

    GRID_SIZE_RUNTIME = atoi(argv[1]);
    NUM_PARTICLES_RUNTIME = atoi(argv[2]);
    MAX_STEPS_RUNTIME = atoi(argv[3]);
    num_benchmark_runs = atoi(argv[4]);

    if (GRID_SIZE_RUNTIME <= 0 || NUM_PARTICLES_RUNTIME <= 0 || MAX_STEPS_RUNTIME < 0 || num_benchmark_runs <= 0) {
        fprintf(stderr, "Error: grid_size, num_particles, max_steps, and benchmark_runs must be positive integers.\n");
        print_usage(argv[0]);
        return 1;
    }
    
    MAX_RLE_LINE_BUFFER_SIZE_RUNTIME = (size_t)GRID_SIZE_RUNTIME * GRID_SIZE_RUNTIME * 8 + 100;


    if (argc == 6 && strcmp(argv[5], "generate_csv") == 0) {
        perform_rle_run = true;
    }

    printf("C Simulation Configuration:\n");
    printf("  Grid Size: %d x %d\n", GRID_SIZE_RUNTIME, GRID_SIZE_RUNTIME);
    printf("  Number of Particles: %d\n", NUM_PARTICLES_RUNTIME);
    printf("  Max Steps: %d\n", MAX_STEPS_RUNTIME);
    printf("  Benchmark Runs: %d\n", num_benchmark_runs);
    printf("  Generate CSV: %s\n", perform_rle_run ? "Yes" : "No");
    printf("--------------------------------------------------\n");

    if (!allocate_global_arrays(GRID_SIZE_RUNTIME, NUM_PARTICLES_RUNTIME)) {
        return 1; // Allocation failed
    }
    
    // --- Averaged Benchmark Run (C version) ---
    double total_elapsed_time_c_sec = 0;
    printf("\nStarting C Benchmark Phase (%d timed runs)\n", num_benchmark_runs);

    for (int i = 0; i < num_benchmark_runs; ++i) {
        srand(42); // Seed for C's rand() - consistent for each benchmark iteration
        init_grids(); 

        struct timespec start_time_c, end_time_c;
        clock_gettime(CLOCK_MONOTONIC, &start_time_c);

        run_simulation_c(false, NULL); // false = disable RLE

        clock_gettime(CLOCK_MONOTONIC, &end_time_c);
        double current_run_time_sec = (end_time_c.tv_sec - start_time_c.tv_sec) +
                                       (end_time_c.tv_nsec - start_time_c.tv_nsec) / 1e9;
        total_elapsed_time_c_sec += current_run_time_sec;
        printf("C Benchmark Run %d/%d took %f seconds.\n", i + 1, num_benchmark_runs, current_run_time_sec);
    }
    
    if (num_benchmark_runs > 0) {
        double average_time_c_sec = total_elapsed_time_c_sec / num_benchmark_runs;
        printf("C Average Computation Benchmark Time (over %d runs): %f seconds.\n", num_benchmark_runs, average_time_c_sec);
    }
    printf("--------------------------------------------------\n");

    // --- RLE Output Generation Run (C version) ---
    if (perform_rle_run) {
        printf("\nStarting C RLE Output Generation Run\n");
        srand(42);    // Re-seed to ensure the same simulation is visualized
        init_grids(); 
        run_simulation_c(true, c_output_filename);
        printf("--------------------------------------------------\n");
    }

    free_global_arrays(); // IMPORTANT: Free dynamically allocated memory
    printf("\nC simulation tasks complete.\n");
    return 0;
}
