import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

FILENAME = 'dla_output_rle.csv'
SIZE = 500

def decode_rle(rle_line_full, grid_size_total, current_step_str="N/A"):
    parts_with_step = rle_line_full.strip().split(',')
    if not parts_with_step:
        raise ValueError(f"Empty RLE line for step {current_step_str}")

    rle_segments = parts_with_step[1:] # Skip step number part

    decoded_values = []
    total_decoded_count = 0

    for segment in rle_segments:
        if not segment: # Handle potential empty segments from trailing commas, etc.
            continue
        try:
            value_str, count_str = segment.split('x')
            value = int(value_str)
            count = int(count_str)

            # Basic validation, can be expanded
            if value not in [0, 1, 2]:
                print(f"Warning (Step {current_step_str}): RLE segment value '{value}' out of expected range (0,1,2) in '{segment}'.")
            if count < 0:
                print(f"Warning (Step {current_step_str}): RLE segment count '{count}' is negative in '{segment}'. Skipping.")
                continue
            
            decoded_values.extend([value] * count)
            total_decoded_count += count

        except ValueError:
            # Catches errors from split (not two parts) or int conversion
            print(f"Skipping malformed RLE segment: '{segment}' for step {current_step_str}. Line (first 100 chars): {rle_line_full[:100]}...")
            continue # Skip this malformed segment

    if total_decoded_count != grid_size_total:
        # This error is critical for reshaping
        error_msg = (f"CRITICAL ERROR (Step {current_step_str}): Decoded RLE length {total_decoded_count} "
                     f"(from {len(decoded_values)} values) does not match grid size {grid_size_total}. "
                     f"Original line (first 100 chars): {rle_line_full[:100]}")
        raise ValueError(error_msg)

    return np.array(decoded_values).reshape((SIZE, SIZE)) # SIZE is the global for grid dimensions

def load_rle_frames(filename, grid_dim_size):
    frames = []
    steps = []
    grid_total_size = grid_dim_size * grid_dim_size
    with open(filename, 'r') as f:
        line_num = 0
        for rle_line_text in f:
            line_num+=1
            rle_line_text = rle_line_text.strip()
            if not rle_line_text:
                continue
            
            current_step_str = rle_line_text.split(',')[0]
            try:
                step = int(current_step_str)
                grid = decode_rle(rle_line_text, grid_total_size, current_step_str)
                steps.append(step)
                frames.append(grid)
            except ValueError as e:
                print(f"Skipping frame with eror {line_num} (Step {current_step_str}): {e}")
            except Exception as ex:
                print(f"Exception error with frame {line_num} (Step {current_step_str}): {ex}")


    if not frames:
        print(f"Can't load frames from file: {filename}.")
        # No frames loaded, exit or handle appropriately
        plt.figure()
        plt.title(f"No data loaded from {filename}")
        plt.text(0.5, 0.5, "Error: Could not load simulation data.", horizontalalignment='center', verticalalignment='center')
        plt.show()
        exit()
    return steps, frames

print(f"Loading frames from {FILENAME}")
steps, frames = load_rle_frames(FILENAME, SIZE)
print(f"Loaded {len(frames)} frames.")

fig, ax = plt.subplots()
# Ensure there's at least one frame before trying to display it
if not frames:
    print("No frames to display. Exiting.")
    exit()

im = ax.imshow(frames[0], cmap='viridis', vmin=0, vmax=2, interpolation='nearest') # 'nearest' for crisp pixels
step_text = ax.text(0.02, 0.95, f"Step: {steps[0]}", color='white', fontsize=10,
                    transform=ax.transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round,pad=0.3', fc='black', ec='none', alpha=0.7))
title_text = f"DLA Simulation ({FILENAME})"
title_obj = ax.set_title(title_text)
ax.axis('off')
fig.tight_layout()


def update(i):
    if i < len(frames):
        im.set_data(frames[i])
        step_text.set_text(f"Step: {steps[i]}")
        return [im, step_text] # Only return artists that changed
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1, blit=True, repeat=False)

plt.show()