import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import numpy as np
import argparse

def animate_vehicle_tracks(csv_path: str, ego_id: int):
    """
    Reads vehicle coordinate data from a CSV and animates the vehicle positions
    frame by frame.

    Args:
        csv_path (str): The full path to the vehicle_coordinates.csv file.
        ego_id (int): The vehicle ID to highlight in red.
    """
    # Load the dataset
    try:
        df = pd.read_csv(csv_path, index_col='frame_id')
    except FileNotFoundError:
        print(f"Error: The file was not found at {csv_path}")
        return

    # --- Animation Setup ---
    fig, ax = plt.subplots(figsize=(12, 12))

    # Determine plot limits from the entire dataset to keep the view static and centered
    max_abs_coord = max(df['x'].abs().max(), df['y'].abs().max())
    plot_limit = max_abs_coord + 20  # Add a 20-meter buffer
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)

    # Initialize a scatter plot object that will be updated in each frame
    other_vehicles_scatter = ax.scatter([], [], s=50, color='blue', label='Other Vehicles')
    (ego_point,) = ax.plot([], [], 'r^', markersize=12, label='Ego Vehicle')
    frame_text = ax.text(0.5, 1.01, '', ha='center', transform=ax.transAxes, fontsize=14)

    ax.set_title('Global Vehicle Tracks Animation')
    ax.set_xlabel('X Coordinate (meters)')
    ax.set_ylabel('Y Coordinate (meters)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    # Get the range of frames for the animation, ensuring we only use frames present in the data
    available_frames = sorted(df.index.unique())

    def update(frame_id):
        """The function that updates the plot for each animation frame."""
        # Select all vehicles in the current frame
        frame_data = df.loc[df.index == frame_id]

        # Find the ego vehicle in this frame
        ego_vehicle_data = frame_data[frame_data['vehicle_id'] == ego_id]

        # Get data for all other vehicles
        other_vehicles_data = frame_data[frame_data['vehicle_id'] != ego_id]

        # Update other vehicles
        other_coords = other_vehicles_data[['x', 'y']].values
        other_vehicles_scatter.set_offsets(other_coords)

        # Update ego vehicle
        if ego_vehicle_data.empty:
            ego_point.set_data([], [])
        else:
            ego_coords = ego_vehicle_data[['x', 'y']].values
            ego_point.set_data(ego_coords.T)

        frame_text.set_text(f'Frame ID: {frame_id}')

        return other_vehicles_scatter, ego_point, frame_text

    # Create and run the animation
    ani = animation.FuncAnimation(fig, update, frames=available_frames, blit=False, interval=50)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Animate vehicle tracks from a CSV file, focusing on an ego vehicle.")
    parser.add_argument('--csv-path', type=str, required=True, help='Full path to the vehicle_coordinates.csv file.')
    parser.add_argument('--ego-id', type=int, required=True, help='The ID of the ego vehicle to center the plot on.')
    
    args = parser.parse_args()

    animate_vehicle_tracks(csv_path=args.csv_path, ego_id=args.ego_id)