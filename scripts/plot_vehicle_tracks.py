import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

from processing.groundtruth import GroundTruthReplay

def update_plot_for_frame(frame_id, gt_replay, other_vehicles_scatter, ego_point, frame_text, view_mode):
    """
    Updates the scatter plots and text elements for a given animation frame.

    Args:
        frame_id (int): The current frame to render.
        gt_replay (GroundTruthReplay): The ground truth replay object.
        other_vehicles_scatter (matplotlib.collections.PathCollection): Scatter plot for other vehicles.
        ego_point (matplotlib.lines.Line2D): Plot object for the ego vehicle.
        frame_text (matplotlib.text.Text): Text object for displaying the frame ID.
        view_mode (str): The view mode ('ego-centric' or 'global').
    """
    # Use the replay object to get structured data for the frame
    gt_frame = gt_replay.get_frame_data(frame_id)

    if view_mode == 'ego-centric':
        # Update plot elements for ego-centric view
        # To make the ego vehicle face "up", we plot:
        # - y_relative on the plot's x-axis (Left/Right)
        # - x_relative on the plot's y-axis (Forward/Backward)
        if not gt_frame.other_vehicles.empty:
            other_vehicles_scatter.set_offsets(gt_frame.other_vehicles[['y_relative', 'x_relative']].values)
        
        # The ego vehicle is always at the origin in this view
        ego_point.set_data([0], [0])
    else: # global view
        other_vehicles_scatter.set_offsets(gt_frame.other_vehicles[['x', 'y']].values)
        if not gt_frame.ego_vehicle.empty:
            ego_point.set_data(gt_frame.ego_vehicle[['x', 'y']].values.T)
        else:
            ego_point.set_data([], [])

    frame_text.set_text(f'Frame ID: {frame_id}')

def animate_vehicle_tracks(csv_path: str, ego_id: int, view_mode: str, view_radius: float = 80.0):
    """
    Reads vehicle coordinate data from a CSV and animates the vehicle positions
    frame by frame in an ego-centric view.

    Args:
        csv_path (str): The full path to the vehicle_coordinates.csv file.
        ego_id (int): The ID of the ego vehicle.
        view_mode (str): The view mode, either 'ego-centric' or 'global'.
        view_radius (float): The plot range in meters.
    """
    # --- Data Handling is now decoupled and performs the transformation ---
    gt_replay = GroundTruthReplay(csv_path, ego_id)

    # --- Animation Setup ---
    fig, ax = plt.subplots(figsize=(12, 12))
    other_vehicles_scatter = ax.scatter([], [], s=50, color='blue', label='Other Vehicles')
    (ego_point,) = ax.plot([], [], 'r^', markersize=12, label='Ego Vehicle')
    frame_text = ax.text(0.5, 1.01, '', ha='center', transform=ax.transAxes, fontsize=14)
    
    if view_mode == 'ego-centric':
        ax.set_title('Ego-Centric Vehicle Tracks Animation')
        ax.set_xlabel('Left/Right Distance (m)')
        ax.set_ylabel('Forward Distance (m)')
        ax.set_xlim(-view_radius, view_radius)
        ax.set_ylim(-view_radius, view_radius)
    else: # global view
        ax.set_title('Global Vehicle Tracks Animation')
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        df = gt_replay.df
        max_abs_coord = max(df['x'].abs().max(), df['y'].abs().max())
        plot_limit = max_abs_coord + 20  # Add a 20-meter buffer
        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)
        
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    # Get the range of frames for the animation, ensuring we only use frames present in the data
    available_frames = range(gt_replay.sim_length_steps + 1)

    def update(frame_id):
        """The function that updates the plot for each animation frame."""
        update_plot_for_frame(frame_id, gt_replay, other_vehicles_scatter, ego_point, frame_text, view_mode)
        return other_vehicles_scatter, ego_point, frame_text

    # Create and run the animation
    ani = animation.FuncAnimation(fig, update, frames=available_frames, blit=False, interval=50)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Animate vehicle tracks from a CSV file, focusing on an ego vehicle.")
    parser.add_argument('--csv-path', type=str, required=True, help='Full path to the vehicle_coordinates.csv file.')
    parser.add_argument('--ego-id', type=int, required=True, help='The ID of the ego vehicle.')
    parser.add_argument('--view-mode', type=str, default='ego-centric', choices=['ego-centric', 'global'], help='The visualization perspective.')
    parser.add_argument('--radius', type=float, default=80.0, help='The view radius around the ego vehicle in meters.')
    
    args = parser.parse_args()

    animate_vehicle_tracks(csv_path=args.csv_path, ego_id=args.ego_id, view_mode=args.view_mode, view_radius=args.radius)