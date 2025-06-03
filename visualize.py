import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

FILENAME = 'dla_output_rle_cuda.csv' # Or 'dla_output_rle.csv' for the C version
SIZE = 500 # This should match the GRID_SIZE_RUNTIME used in C/CUDA

def decode_rle(rle_line_full, grid_size_total, current_step_str="N/A"):
    """
    Decodes a single RLE (Run-Length Encoded) line into a 2D NumPy array.

    Args:
        rle_line_full (str): The full RLE string for a frame, including the step number.
                             Example: "0,0x100,1x5,2x3..."
        grid_size_total (int): The total number of cells in the grid (SIZE * SIZE).
        current_step_str (str): The step number as a string, for error reporting.

    Returns:
        numpy.ndarray: A 2D NumPy array representing the decoded grid.

    Raises:
        ValueError: If the RLE line is malformed or the decoded length doesn't match grid_size_total.
    """
    parts_with_step = rle_line_full.strip().split(',', 1) # Split only on the first comma
    if len(parts_with_step) < 2:
        raise ValueError(f"Malformed RLE line (missing comma after step): '{rle_line_full[:100]}...' for step {current_step_str}")

    rle_segments_str = parts_with_step[1]
    rle_segments = rle_segments_str.split(',')

    # Pre-allocate a NumPy array for speed. Using int8 as values are 0, 1, 2.
    decoded_flat_data = np.empty(grid_size_total, dtype=np.int8)
    current_idx = 0

    for segment in rle_segments:
        if not segment: # Handle potential empty segments
            continue
        try:
            value_str, count_str = segment.split('x')
            value = int(value_str)
            count = int(count_str)

            if value not in [0, 1, 2]:
                print(f"Warning (Step {current_step_str}): RLE segment value '{value}' out of expected range (0,1,2) in '{segment}'.")
            if count < 0:
                print(f"Warning (Step {current_step_str}): RLE segment count '{count}' is negative in '{segment}'. Skipping.")
                continue
            
            if current_idx + count > grid_size_total:
                raise ValueError(
                    f"CRITICAL ERROR (Step {current_step_str}): RLE segment count {count} for value {value} "
                    f"exceeds remaining grid size ({grid_size_total - current_idx} left at index {current_idx}). "
                    f"Segment: '{segment}'. Original line (first 100 chars): {rle_line_full[:100]}"
                )

            # Fill the pre-allocated NumPy array directly
            decoded_flat_data[current_idx : current_idx + count] = value
            current_idx += count

        except ValueError as e: # Catches errors from split or int conversion
            # Provide more context for the error
            print(f"Skipping malformed RLE segment: '{segment}' for step {current_step_str}. Error: {e}. Line (first 100 chars): {rle_line_full[:100]}...")
            continue 

    if current_idx != grid_size_total:
        error_msg = (f"CRITICAL ERROR (Step {current_step_str}): Decoded RLE length {current_idx} "
                     f"does not match grid size {grid_size_total}. "
                     f"Original line (first 100 chars): {rle_line_full[:100]}")
        raise ValueError(error_msg)

    return decoded_flat_data.reshape((SIZE, SIZE))

def load_rle_frames(filename, grid_dim_size):
    """
    Loads all RLE frames from the specified file.

    Args:
        filename (str): The path to the RLE CSV file.
        grid_dim_size (int): The dimension of one side of the square grid (e.g., SIZE).

    Returns:
        tuple: A tuple containing:
            - steps (list): A list of step numbers corresponding to each frame.
            - frames (list): A list of 2D NumPy arrays, where each array is a decoded grid frame.
    """
    frames = []
    steps = []
    grid_total_size = grid_dim_size * grid_dim_size
    
    try:
        with open(filename, 'r') as f:
            line_num = 0
            for rle_line_text in f:
                line_num += 1
                rle_line_text = rle_line_text.strip()
                if not rle_line_text:
                    continue
                
                try:
                    # Extract step number more robustly
                    step_part = rle_line_text.split(',', 1)[0]
                    step = int(step_part)
                except (ValueError, IndexError):
                    print(f"Skipping line {line_num} due to malformed step number: {rle_line_text[:50]}...")
                    continue

                try:
                    grid = decode_rle(rle_line_text, grid_total_size, step_part)
                    steps.append(step)
                    frames.append(grid)
                except ValueError as e:
                    print(f"Skipping frame at line {line_num} (Step {step_part}): {e}")
                except Exception as ex: # Catch any other unexpected errors during decoding
                    print(f"Unexpected error processing frame at line {line_num} (Step {step_part}): {ex}")

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        # Handle file not found: show message and exit
        plt.figure()
        plt.title(f"Error: File not found")
        plt.text(0.5, 0.5, f"Could not load simulation data from:\n{filename}\nPlease check the file path and name.", 
                 horizontalalignment='center', verticalalignment='center', color='red', fontsize=12)
        plt.axis('off')
        plt.show()
        exit()

    if not frames:
        print(f"No valid frames loaded from file: {filename}.")
        plt.figure()
        plt.title(f"No data loaded from {filename}")
        plt.text(0.5, 0.5, "Error: Could not load any valid simulation data.", 
                 horizontalalignment='center', verticalalignment='center', color='red', fontsize=12)
        plt.axis('off')
        plt.show()
        exit()
        
    return steps, frames

if __name__ == "__main__":
    print(f"Loading frames from {FILENAME} (Grid size: {SIZE}x{SIZE})")
    steps, frames = load_rle_frames(FILENAME, SIZE)
    
    if not frames: # Should be caught by load_rle_frames, but as a safeguard
        print("No frames to display. Exiting.")
        exit()
        
    print(f"Loaded {len(frames)} frames successfully.")

    fig, ax = plt.subplots(figsize=(8, 8)) # Adjust figure size if needed
    
    # Use the first frame to initialize the image
    im = ax.imshow(frames[0], cmap='viridis', vmin=0, vmax=2, interpolation='nearest') 
    
    # Add step text display
    step_text_obj = ax.text(0.02, 0.98, f"Step: {steps[0]}", color='white', fontsize=10,
                            transform=ax.transAxes, verticalalignment='top', 
                            bbox=dict(boxstyle='round,pad=0.3', fc='black', ec='none', alpha=0.6))
    
    title_text = f"DLA Simulation ({FILENAME})"
    ax.set_title(title_text)
    ax.axis('off') # Turn off axis numbers and ticks
    fig.tight_layout(pad=1.5) # Adjust padding

    def update(frame_index):
        if frame_index < len(frames):
            im.set_data(frames[frame_index])
            step_text_obj.set_text(f"Step: {steps[frame_index]}")
            return [im, step_text_obj]
        return [im] # Should not happen if frames arg to FuncAnimation is correct

    # Create animation
    # Interval can be adjusted for speed (milliseconds per frame)
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=0, blit=True, repeat=False)

    plt.show()
