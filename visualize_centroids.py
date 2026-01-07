import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

# --- Configuration ---
CSV_FILE_PATH = 'centroids_output.csv'
# ---------------------

def visualize_centroids(csv_path):
    """
    Loads centroid data from a CSV and creates an interactive plot
    with a slider to navigate through frames.

    Args:
        csv_path (str): The path to the centroids CSV file.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at '{csv_path}'")
        return

    # Load the data using pandas
    df = pd.read_csv(csv_path)

    # Determine fixed plot limits from the entire dataset to prevent resizing
    if not df.empty:
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        # Add a 10% buffer to the limits for better visualization
        x_buffer = (x_max - x_min) * 0.1 if (x_max - x_min) > 0 else 5
        y_buffer = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 5
        plot_xlim = (x_min - x_buffer, x_max + x_buffer)
        plot_ylim = (y_min - y_buffer, y_max + y_buffer)
    else:
        # Default limits if the CSV is empty
        plot_xlim = (-50, 50)
        plot_ylim = (-50, 50)

    # --- Color Mapping for Unique IDs ---
    # Get all unique track IDs from the entire dataset
    all_track_ids = sorted(df['track_id'].unique())
    # Create a map from track_id to a unique color
    # We use a colormap and cycle through it if there are more tracks than colors.
    cmap = plt.colormaps.get_cmap('tab20') # 'tab20' has 20 distinct colors
    color_map = {track_id: cmap(i % cmap.N) for i, track_id in enumerate(all_track_ids)}

    # --- Frame Handling ---
    # Create a continuous list of all frame IDs from min to max
    # to handle missing frames in the data.
    if df.empty:
        print("Warning: CSV file is empty. No data to plot.")
        all_frame_ids = []
    else:
        min_frame = int(df['frame_idx'].min())
        max_frame = int(df['frame_idx'].max())
        all_frame_ids = list(range(min_frame, max_frame + 1))

    if not all_frame_ids:
        return

    # Create the main plot and the slider axis
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.2)

    # Create an axis for the slider
    ax_slider = plt.axes([0.20, 0.05, 0.65, 0.03])

    # Create the slider widget
    frame_slider = Slider(
        ax=ax_slider,
        label='Frame ID',
        valmin=0,
        valmax=len(all_frame_ids) - 1,
        valinit=0,
        valfmt='%d' # Show slider value as an integer
    )

    def update(val):
        """Function to be called when the slider value changes."""
        frame_index = int(frame_slider.val)
        current_frame_id = all_frame_ids[frame_index]
        
        # Filter data for the current frame
        frame_data = df[df['frame_idx'] == current_frame_id]
        
        # Clear the previous plot
        ax.clear()
        
        # Plot each track with its unique color
        for track_id, group in frame_data.groupby('track_id'):
            color = color_map.get(track_id, 'black') # Default to black if ID is somehow not in map
            ax.scatter(group['x'], group['y'], color=color, label=f'Track {int(track_id)}')
            # Annotate the point for this track
            ax.text(group['x'].iloc[0] + 0.5, group['y'].iloc[0] + 0.5, f"T{int(track_id)}", fontsize=9)
            
        # Apply the fixed limits and other axis properties AFTER clearing
        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title(f"Object Centroids for Frame ID: {current_frame_id}")
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box') 
        # ax.legend() # Legend can get crowded, optionally enable it if needed
        fig.canvas.draw_idle()

    # Register the update function with the slider
    frame_slider.on_changed(update)

    # Initial plot
    update(0)

    plt.show()

if __name__ == '__main__':
    # Assuming the script is in the same directory as the CSV
    script_dir = os.path.dirname(__file__)
    csv_full_path = os.path.join(script_dir, CSV_FILE_PATH)
    visualize_centroids(csv_full_path)