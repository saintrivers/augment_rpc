import argparse
import json
import os
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from processing.datareader import load_radar_data, load_ego_imu_data
from processing.radarproc import collect_transformed_rpc


def init_overlay_plot(fig, view_radius, gt_size, gt_alpha):
    """Initializes a single 2D plot for overlaying all data."""
    ax = fig.add_subplot(111)
    # RPC points will be smaller and have some transparency
    rpc_scatter = ax.scatter([], [], c=[], cmap='plasma', s=5, vmin=-30, vmax=30, alpha=0.7, label='RPC Points')
    # Ground Truth neighbors will be larger and solid
    gt_scatter = ax.scatter([], [], c='blue', s=gt_size, alpha=gt_alpha, label='Neighbors (Ground Truth)')
    (ego_point,) = ax.plot([], [], 'r^', markersize=12, label='Ego')

    ax.set_title("Combined RPC and Ground Truth Overlay")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    # Using vehicle-centric coordinates (X-forward, Y-left)
    ax.set_xlabel('Left/Right Distance (m)')
    ax.set_ylabel('Forward Distance (m)')
    ax.set_xlim(view_radius, -view_radius)
    ax.set_ylim(-view_radius, view_radius)

    fig.colorbar(rpc_scatter, ax=ax, label='Velocity (m/s)', shrink=0.8)
    return ax, rpc_scatter, gt_scatter, ego_point


def plot_ground_truth(gt_scatter, frame_data, ego_x, ego_y, ego_yaw_deg):
    """Calculates and plots relative ground truth vehicle positions."""
    if not frame_data.empty:
        # We want [x, y] for a standard vehicle-forward view.
        relative_x = frame_data["x"].values - ego_x
        relative_y = frame_data["y"].values - ego_y
        yaw_rad = -np.deg2rad(ego_yaw_deg)
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)

        # Rotate coordinates to ego frame
        # rotated_x is forward, rotated_y is lateral (left is positive)
        rotated_x = relative_x * c - relative_y * s
        rotated_y = -(relative_x * s + relative_y * c)

        # Plot lateral on X-axis, forward on Y-axis
        gt_scatter.set_offsets(np.column_stack([rotated_y, rotated_x]))
    else:
        gt_scatter.set_offsets(np.empty((0, 2)))


def plot_rpc_data(rpc_scatter, rpc_data, rpc_idx, sensor_transforms, imu_record):
    """Calculates and plots transformed RPC points."""
    xs, ys, _, velocities = collect_transformed_rpc(rpc_data, rpc_idx, sensor_transforms, imu_record)
    
    if xs:
        # RPC points are in ego-relative frame (X-fwd, Y-right).
        # We need to plot lateral on X-axis (with left being positive) and forward on Y-axis.
        # We flip the sign of 'ys' to match the ground truth's "left-positive" convention.
        rpc_scatter.set_offsets(np.column_stack([-np.array(ys), xs]))
        rpc_scatter.set_array(np.array(velocities))
    else:
        rpc_scatter.set_offsets(np.empty((0, 2)))


def run_visualization(args):
    # --- 1. Load Data ---
    print("Loading data...")
    gt_all_df = pd.read_csv(os.path.join(args.datadir, 'vehicle_coordinates.csv'))
    ego_traj = gt_all_df[gt_all_df['vehicle_id'] == args.ego_id].set_index('frame_id')
    imu_df = load_ego_imu_data(f"{args.datadir}/imu_data.csv", args.ego_id)

    if ego_traj.empty:
        print(f"❌ Error: Ego vehicle with ID {args.ego_id} not found.")
        return

    # Load data sources based on the '--include' argument
    rpc_data, rpc_frame_ids, sensor_transforms = None, None, None
    if 'rpc' in args.include:
        try:
            with open(f'{args.datadir}/radar_config.json', 'r') as f:
                radar_config = json.load(f)
            sensor_transforms = {
                name: {'pos': np.array([c['transform']['x'], c['transform']['y'], c['transform']['z']]),
                       'yaw_deg': c['transform']['yaw_deg']}
                for name, c in radar_config['sensors'].items()
            }
            rpc_data, rpc_frame_ids = load_radar_data(args.datadir)
        except FileNotFoundError as e:
            print(f"Warning: Could not load RPC data: {e}")
            args.include.remove('rpc')

    # --- 2. Synchronize Frames ---
    gt_frames = set(gt_all_df['frame_id'].unique())
    common_frames = gt_frames
    if 'rpc' in args.include and rpc_frame_ids is not None:
        rpc_frames_set = set(rpc_frame_ids)
        common_frames = sorted(list(gt_frames.intersection(rpc_frames_set)))
    else:
        common_frames = sorted(list(gt_frames))

    # Always initialize the dictionary to prevent UnboundLocalError
    rpc_frame_id_to_idx = {frame_id: i for i, frame_id in enumerate(rpc_frame_ids)} if rpc_frame_ids else {}

    # --- 3. Setup Plot ---
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(f"Overlay View (Ego Vehicle {args.ego_id})", fontsize=16)
    ax, rpc_scatter, gt_scatter, ego_point = init_overlay_plot(fig, args.range, args.gt_size, args.gt_alpha)
    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)

    # --- 4. Snapshot Directory ---
    snapshot_dir = "plots/_snapshots"
    # This check is now inside the snapshot logic to avoid creating the dir unnecessarily
    if args.periodic_snapshots:
        Path(snapshot_dir).mkdir(exist_ok=True)

    # --- Data storage for accumulated snapshots ---
    accumulated_gt_points = []
    accumulated_rpc_points = []
    accumulated_rpc_velocities = []
    accumulated_frame_ids = []

    # --- 5. Animation Update Function ---
    def update(frame_id):
        try:
            ego_state = ego_traj.loc[frame_id]
            ego_x, ego_y, ego_yaw_deg = ego_state['x'], ego_state['y'], ego_state['yaw']
            imu_record = imu_df.loc[frame_id].to_dict()
        except KeyError:
            return rpc_scatter, gt_scatter, ego_point, frame_text

        # A. Update Ground Truth Plot
        if 'ground_truth' in args.include:
            gt_frame_data = gt_all_df[(gt_all_df['frame_id'] == frame_id) & (gt_all_df['vehicle_id'] != args.ego_id)]
            plot_ground_truth(gt_scatter, gt_frame_data, ego_x, ego_y, ego_yaw_deg)

        # B. Update RPC Plot
        if 'rpc' in args.include and rpc_data:
            rpc_idx = rpc_frame_id_to_idx.get(frame_id) # This is safe now
            plot_rpc_data(rpc_scatter, rpc_data, rpc_idx, sensor_transforms, imu_record)

        ego_point.set_data([0], [0])
        frame_text.set_text(f'Frame ID: {frame_id}')

        # C. Handle periodic snapshots
        if args.periodic_snapshots and common_frames.index(frame_id) % args.periodic_snapshots == 0:
            if args.snapshot_mode == 'individual':
                snapshot_path = os.path.join(snapshot_dir, f"snapshot_frame_{frame_id}.png")
                plt.savefig(snapshot_path, dpi=150)
                print(f"Saved snapshot: {snapshot_path}")
            elif args.snapshot_mode == 'accumulate':
                # Store the points from this snapshot frame
                if 'ground_truth' in args.include:
                    accumulated_gt_points.append(gt_scatter.get_offsets())
                if 'rpc' in args.include:
                    accumulated_rpc_points.append(rpc_scatter.get_offsets())
                    accumulated_rpc_velocities.append(rpc_scatter.get_array())
                accumulated_frame_ids.append(frame_id)
                
        return rpc_scatter, gt_scatter, ego_point, frame_text

    # --- 6. Create and Run/Save Animation ---
    # If we are only taking snapshots, we don't need to run the full animation GUI.
    # We can iterate manually, which is faster and doesn't require a display.
    if args.periodic_snapshots and not args.show_plot and not args.save_gif:
        if args.snapshot_mode == 'accumulate':
            print("Generating multi-plot snapshot figure...")
            # Determine grid size
            snapshot_indices = range(0, len(common_frames), args.periodic_snapshots)
            num_snapshots = len(snapshot_indices)

            if num_snapshots > 0:
                cols = int(np.ceil(np.sqrt(num_snapshots)))
                rows = int(np.ceil(num_snapshots / cols))
                grid_fig, grid_axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
                grid_axes = grid_axes.flatten()

                from tqdm import tqdm
                for i, frame_idx in enumerate(tqdm(snapshot_indices, desc="Generating subplots")):
                    frame_id = common_frames[frame_idx]
                    ax = grid_axes[i]

                    # Configure each subplot
                    ax.set_title(f"Frame ID: {frame_id}")
                    ax.set_xlim(args.range, -args.range)
                    ax.set_ylim(-args.range, args.range)
                    ax.set_aspect('equal')
                    ax.grid(True)
                    ax.plot([0], [0], 'r^', markersize=10) # Draw ego vehicle

                    # Get data for this specific frame
                    ego_state = ego_traj.loc[frame_id]
                    gt_frame_data = gt_all_df[(gt_all_df['frame_id'] == frame_id) & (gt_all_df['vehicle_id'] != args.ego_id)]

                    if 'ground_truth' in args.include:
                        temp_gt_scatter = ax.scatter([], [], c='blue', s=args.gt_size, alpha=args.gt_alpha, label='Ground Truth')
                        plot_ground_truth(temp_gt_scatter, gt_frame_data, ego_state['x'], ego_state['y'], ego_state['yaw'])
                    if 'rpc' in args.include:
                        rpc_idx = rpc_frame_id_to_idx.get(frame_id)
                        temp_rpc_scatter = ax.scatter([], [], c=[], cmap='plasma', s=5, vmin=-30, vmax=30, alpha=0.7, label='RPC')
                        plot_rpc_data(temp_rpc_scatter, rpc_data, rpc_idx, sensor_transforms, imu_df.loc[frame_id].to_dict())
                        
                    ax.legend(loc='upper right', fontsize='small')

                # Hide unused subplots
                for j in range(num_snapshots, len(grid_axes)):
                    grid_axes[j].set_visible(False)

                final_path = os.path.join(snapshot_dir, "multiplot_snapshot.png")
                # grid_fig.suptitle(f"Snapshots (Ego Vehicle)", fontsize=16)
                grid_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                grid_fig.savefig(final_path, dpi=200)
                print(f"Saved multi-plot snapshot to {final_path}")
        else: # individual snapshot mode
            print(f"Generating individual snapshots for {len(common_frames)} frames...")
            from tqdm import tqdm
            for frame_id in tqdm(common_frames, desc="Processing frames"):
                update(frame_id)
            plt.close(fig) # Close the figure after iterating
    else:
        # Otherwise, run the standard animation for display or GIF saving.
        ani = animation.FuncAnimation(fig, update, frames=common_frames, interval=100, blit=False)

        if args.save_gif:
            print(f"Saving animation to {args.save_gif}...")
            ani.save(args.save_gif, writer='pillow', fps=10)
            print(f"Animation saved to {args.save_gif}")

        if args.show_plot:
            print("Showing interactive plot...")
            plt.show()
            
    print("Done.")
    

def main():
    """
    Parses CLI arguments and runs the combined data visualization.
    """
    parser = argparse.ArgumentParser(
        description="CLI interface for the CARLA data visualizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--datadir',
        type=str,
        default='_output',
        help='Path to the data directory containing sensor data and CSVs.'
    )
    parser.add_argument(
        '--include',
        nargs='+',
        default=['rpc', 'ground_truth'],
        choices=['rpc', 'ground_truth'],
        help='Select which data sources to include in the plot.'
    )
    parser.add_argument(
        '--ego-id',
        type=int,
        required=True,
        help='The numerical ID of the ego vehicle.'
    )
    parser.add_argument(
        '--range',
        type=float,
        default=75.0,
        help="View radius around the ego vehicle in meters."
    )

    # --- Plotting Style Arguments ---
    style_group = parser.add_argument_group('Plotting Style Options')
    style_group.add_argument(
        '--gt-size',
        type=float,
        default=50,
        help='Marker size for ground truth points.'
    )
    style_group.add_argument(
        '--gt-alpha',
        type=float,
        default=1.0,
        help='Alpha (transparency) for ground truth points, from 0.0 to 1.0.'
    )
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--show-plot',
        action='store_true',
        help='If set, displays the plot window interactively.'
    )
    output_group.add_argument(
        '--save-gif',
        type=str,
        default=None,
        help='Saves the animation to a GIF file. Provide the desired output filename.'
    )
    output_group.add_argument(
        '--periodic-snapshots',
        type=int,
        metavar='N',
        default=None,
        help='Saves a snapshot of the plot every N frames.'
    )
    output_group.add_argument(
        '--snapshot-mode',
        type=str,
        choices=['individual', 'accumulate'],
        default='individual',
        help='Mode for periodic snapshots: "individual" for separate files, "accumulate" for a single combined plot.'
    )

    args = parser.parse_args()

    if not args.show_plot and not args.save_gif and not args.periodic_snapshots:
        parser.error("No output option selected. Please use --show-plot, --save-gif, or --periodic-snapshots.")

    run_visualization(args)
    
    
if __name__ == '__main__':
    main()