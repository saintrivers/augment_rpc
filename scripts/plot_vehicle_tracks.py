import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

from processing.groundtruth import GroundTruthReplay

def update_plot_for_frame(frame_id, gt_replay, other_vehicles_scatter, ego_point, frame_text):
    """
    Updates the scatter plots and text elements for a given animation frame.

    Args:
        frame_id (int): The current frame to render.
        gt_replay (GroundTruthReplay): The ground truth replay object.
        other_vehicles_scatter (matplotlib.collections.PathCollection): Scatter plot for other vehicles.
        ego_point (matplotlib.lines.Line2D): Plot object for the ego vehicle.
        frame_text (matplotlib.text.Text): Text object for displaying the frame ID.
    """
    # Use the replay object to get structured, ego-centric data for the frame
    gt_frame = gt_replay.get_frame_data(frame_id)

    # Update plot elements
    # To make the ego vehicle face "up", we plot:
    # - y_relative on the plot's x-axis (Left/Right)
    # - x_relative on the plot's y-axis (Forward/Backward)
    if not gt_frame.other_vehicles.empty:
        other_vehicles_scatter.set_offsets(gt_frame.other_vehicles[['y_relative', 'x_relative']].values)
    
    # The ego vehicle is always at the origin in this view
    ego_point.set_data([0], [0])
    frame_text.set_text(f'Frame ID: {frame_id}')

def animate_vehicle_tracks(csv_path: str, ego_id: int, view_radius: float = 80.0):
    """
    Reads vehicle coordinate data from a CSV and animates the vehicle positions
    frame by frame in an ego-centric view.

    Args:
        csv_path (str): The full path to the vehicle_coordinates.csv file.
        ego_id (int): The vehicle ID to highlight in red.
        view_radius (float): The plot range in meters.
    """
    # --- Data Handling is now decoupled and performs the transformation ---
    gt_replay = GroundTruthReplay(csv_path, ego_id)

    # --- Animation Setup ---
    fig, ax = plt.subplots(figsize=(12, 12))

    # Set fixed plot limits for the ego-centric view
    ax.set_xlim(-view_radius, view_radius)
    ax.set_ylim(-view_radius, view_radius)

    # Initialize a scatter plot object that will be updated in each frame
    other_vehicles_scatter = ax.scatter([], [], s=50, color='blue', label='Other Vehicles')
    (ego_point,) = ax.plot([], [], 'r^', markersize=12, label='Ego Vehicle')
    frame_text = ax.text(0.5, 1.01, '', ha='center', transform=ax.transAxes, fontsize=14)

    ax.set_title('Ego-Centric Vehicle Tracks Animation')
    ax.set_xlabel('Left/Right Distance (m)')
    ax.set_ylabel('Forward Distance (m)')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    # Get the range of frames for the animation, ensuring we only use frames present in the data
    available_frames = sorted(gt_replay.df.index.unique())

    def update(frame_id):
        """The function that updates the plot for each animation frame."""
        update_plot_for_frame(frame_id, gt_replay, other_vehicles_scatter, ego_point, frame_text)
        return other_vehicles_scatter, ego_point, frame_text

    # Create and run the animation
    ani = animation.FuncAnimation(fig, update, frames=available_frames, blit=False, interval=50)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Animate vehicle tracks from a CSV file, focusing on an ego vehicle.")
    parser.add_argument('--csv-path', type=str, required=True, help='Full path to the vehicle_coordinates.csv file.')
    parser.add_argument('--ego-id', type=int, required=True, help='The ID of the ego vehicle.')
    parser.add_argument('--radius', type=float, default=80.0, help='The view radius around the ego vehicle in meters.')
    
    args = parser.parse_args()

    animate_vehicle_tracks(csv_path=args.csv_path, ego_id=args.ego_id, view_radius=args.radius)