import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
import os

from processing.datareader import load_radar_data, load_ego_imu_data, load_radar_config
from processing.radarproc import RpcReplay


def world_to_ego(x, y, ego_x, ego_y, ego_yaw_deg):
    """
    Convert world coordinates (x, y) to ego-centric coordinates.
    """
    # Translate
    dx = x - ego_x
    dy = y - ego_y

    # Convert yaw to radians
    yaw = np.deg2rad(ego_yaw_deg)

    # Rotation: world → ego
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)

    x_ego = cos_yaw * dx - sin_yaw * dy
    y_ego = sin_yaw * dx + cos_yaw * dy

    return x_ego, y_ego

def create_ego_centric_view(df, ego_vehicle_id, range_m=50.0):
    """
    Returns ego-centric coordinates for all vehicles near the ego vehicle.
    """
    ego_df = df[df["vehicle_id"] == ego_vehicle_id]

    ego_frames = []

    for frame_id, ego_row in ego_df.groupby("frame_id"):
        ego_row = ego_row.iloc[0]

        # All vehicles in this frame
        frame_df = df[df["frame_id"] == frame_id].copy()

        # Transform coordinates
        x_ego, y_ego = world_to_ego(
            frame_df["x"].values,
            frame_df["y"].values,
            ego_row["x"],
            ego_row["y"],
            ego_row["yaw"]
        )

        frame_df["x_ego"] = x_ego
        frame_df["y_ego"] = y_ego
        frame_df["z_ego"] = frame_df["z"] - ego_row["z"]
        frame_df["yaw_ego"] = frame_df["yaw"] - ego_row["yaw"]

        # Distance filter (optional)
        frame_df["distance"] = np.sqrt(
            frame_df["x_ego"]**2 + frame_df["y_ego"]**2
        )

        frame_df = frame_df[
            frame_df["distance"] <= range_m
        ]

        frame_df["ego_vehicle_id"] = ego_vehicle_id
        ego_frames.append(frame_df)

    return pd.concat(ego_frames, ignore_index=True)

def prepare_experiment_data(datadir: str, ego_id: int) -> RpcReplay:
    """
    Loads rpc data structure designed for easy fetching and indexing of recorded simulation data.
    
    :param str datadir: absolute or relative path to the dataset root
    :param int ego_id: id of the ego vehicle set by the carla simulator
    :return rpc_replay: RpcReplay object
    """
    rpc, frame_ids = load_radar_data(datadir)
    imu_df = load_ego_imu_data(f"{datadir}/imu_data.csv", ego_id)
    # nearby_df, ego_traj, frame_ids = load_and_prepare_data(config.sim.datadir, config.sim.ego_id)
    sensor_transforms = load_radar_config(datadir)
    rpc_replay = RpcReplay(rpc, frame_ids, sensor_transforms, imu_df)
    return rpc_replay

def main():
    """
    Main function to load data, create an ego-centric view, and visualize it with an interactive slider.
    """
    parser = argparse.ArgumentParser(
        description="Visualize ego-centric ground truth and RPC for a given dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--datadir',
        type=str,
        required=True,
        help='Path to the data directory containing sensor data and CSVs.'
    )
    args = parser.parse_args()

    # Read ego_vehicle_id from metadata.json
    with open(os.path.join(args.datadir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    ego_vehicle_id = metadata["ego_vehicle_id"]

    df = pd.read_csv(f"{args.datadir}/vehicle_coordinates.csv")
    ego_df = create_ego_centric_view(df, ego_vehicle_id, range_m=80.0)
    rpc_replay = prepare_experiment_data(args.datadir, ego_vehicle_id)

    # --- Interactive Plotting with Slider ---

    # Get the list of unique frame IDs for the slider
    unique_frames = sorted(ego_df["frame_id"].unique())
    min_frame, max_frame = unique_frames[0], unique_frames[-1]

    # Create the figure and axes for the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)  # Make space for the slider

    vehicle_scatter = ax.scatter([], [], c="blue", label="Vehicles")
    rpc_scatter = ax.scatter([], [], s=1, cmap="viridis", label="RPC")
    ax.scatter(0, 0, c="red", s=100, label="Ego") # Ego is always at origin

    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
    ax.set_xlabel("Left (m)")
    ax.set_ylabel("Forward (m)")
    ax.legend()
    ax.axis("equal")
    ax.grid(True)
    ax.set_title("Ego-Centric View")
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)

    # Create the slider axis and the slider widget
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    frame_slider = plt.Slider(
        ax=ax_slider,
        label='Frame ID',
        valmin=min_frame,
        valmax=max_frame,
        valinit=min_frame,
        valstep=1  # Assuming frame IDs are integers
    )

    # Update function to be called when the slider value changes
    def update(val):
        frame_id = int(frame_slider.val)
        frame_data = ego_df[ego_df["frame_id"] == frame_id]
        vehicle_scatter.set_offsets(frame_data[["y_ego", "x_ego"]].values)
        
        frame = rpc_replay.get_frame_by_id(frame_id)
        rpc_scatter.set_offsets(np.column_stack((frame.y, frame.x)))
        ax.set_title(f"Ego-Centric View (Frame: {frame_id})")
        fig.canvas.draw_idle()
        

    # Register the update function with the slider
    frame_slider.on_changed(update)

    # Initial plot update
    update(min_frame)

    plt.show()

if __name__ == '__main__':
    main()
