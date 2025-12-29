import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider

import argparse
from processing.association import HungarianMatcher
from processing.clustering import FrameCluster, RadarObject
from processing.datareader import load_metadata, load_radar_data, load_imu_data, load_radar_config, prepare_experiment_data
from processing.groundtruth import GroundTruthReplay
from processing.radarproc import RpcReplay
from pipepine.factory import RpcProcessFactory
from omegaconf import OmegaConf

def init_plot(fig, view_radius):
    """Initializes a single 2D plot for visualizing clustered RPC data."""
    ax = fig.add_subplot(111)
    # Scatter for clustered points (colored by cluster ID)
    cluster_scatter = ax.scatter([], [], c=[], cmap='jet', s=15, label='Clustered Points')
    # Scatter for noise points (gray)
    noise_scatter = ax.scatter([], [], c='gray', s=5, alpha=0.5, label='Noise')
    gt_scatter = ax.scatter([], [], marker='s', s=120, facecolors='none', edgecolors='yellow', linewidth=2, label='Ground Truth')
    (ego_point,) = ax.plot([0], [0], 'r^', markersize=12, label='Ego')

    ax.set_title("DBSCAN Clustering of Radar Point Cloud")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    # Using vehicle-centric coordinates (X-forward, Y-left)
    ax.set_xlabel('Left/Right Distance (m)')
    ax.set_ylabel('Forward Distance (m)')
    ax.set_xlim(view_radius, -view_radius)
    ax.set_ylim(-view_radius, view_radius)

    # Add a text element to display the frame ID
    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)
    
    # Add a text element for cluster stats
    stats_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=9)
    
    return ax, cluster_scatter, noise_scatter, gt_scatter, ego_point, frame_text, stats_text

def get_color_from_id(obj_id):
    """
    Returns a consistent color for a given integer ID.
    Uses 'tab20' which has 20 distinct colors. 
    Modulo 20 ensures we loop back if IDs get higher than 20.
    """
    cmap = plt.get_cmap('tab20')
    return cmap(obj_id % 20)


def hungarian_matching(
    params: dict[str, float], 
    processing_factory: RpcProcessFactory,
    matcher: HungarianMatcher,
    idx: int
    ) -> tuple[list[RadarObject], FrameCluster, list[int]]:
    """
    Performs Hungarian Matching between the current frame and the previous frame, assiging an ID to each RadarObject found and tracking the ID across time.
    
    :param idx: Zero-based index of the frame to process.
    :type idx: int
    :param dbscan_config: The `dbscan` configuration object from the YAML configuration file.
    :type dbscan_config: dict[str, float]
    :param rpc_replay: Description
    :type rpc_replay: RpcReplay
    :param processing_factory: Description
    :type processing_factory: RpcProcessFactory
    :param matcher: Description
    :type matcher: HungarianMatcher
    :return: Description
    :rtype: tuple[list[RadarObject], FrameCluster, list[int]]
    """
    
    # params = {
    #     "spatial_eps": dbscan_config.spatial_epsilon,
    #     "velocity_eps": dbscan_config.velocity_epsilon,
    #     "min_samples": dbscan_config.min_samples,
    #     "velocity_weight": dbscan_config.velocity_weight,
    #     "noise_velocity_threshold": dbscan_config.noise_velocity_threshold
    # }
    
    if idx < 1:
        """
        There is no previous frame to process. Return the processed clusters as it is.
        """
        return processing_factory.get_processed_frame(idx=idx, **params)
    
    # for idx in range(1, rpc_replay.sim_length_steps):
    moving_centroids_prev, _, _ = processing_factory.get_processed_frame(idx=idx - 1, **params)
    moving_centroids_curr, processed_frame, valid_labels  = processing_factory.get_processed_frame(idx=idx, **params)
    matched_output = matcher(moving_centroids_prev, moving_centroids_curr)

    for curr_idx, prev_idx in matched_output.items():
        # 1. Get the ID of the old object
        old_id = moving_centroids_prev[prev_idx].id

        # 2. Assign it to the new object
        moving_centroids_curr[curr_idx].id = old_id
            
    return moving_centroids_curr, processed_frame, valid_labels


def main():
    """Main function to run the hyperparameter visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize radar clustering with interactive hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument('--config', type=str, default="config/base.yml", help='Path to the YAML configuration file.')
    # args = parser.parse_args()
    args = "config/base.yml"
    ego_id, config = load_metadata(args)
    
    # --- Data Loading ---
    rpc_replay = prepare_experiment_data(config.sim.datadir, ego_id)
    gt_replay = GroundTruthReplay(os.path.join(config.sim.datadir, 'vehicle_coordinates.csv'), ego_id)

    # --- Clustering Factory & Hungarian Matcher ---
    processing_factory = RpcProcessFactory(rpc_replay)
    matcher = HungarianMatcher(max_distance=2.0)

    # --- Visualization ---
    fig = plt.figure(figsize=(12, 12))
    ax, cluster_scatter, noise_scatter, gt_scatter, ego_point, frame_text, stats_text = init_plot(fig, view_radius=80)

    # Adjust layout to make room for sliders
    fig.subplots_adjust(bottom=0.35)
    fig.suptitle('Radar Point Cloud Analysis', fontsize=16)

    # --- Interactive Sliders Setup ---
    # Define axes for sliders
    ax_frame = fig.add_axes([0.25, 0.25, 0.5, 0.03])
    ax_vel_weight = fig.add_axes([0.25, 0.20, 0.5, 0.03])
    ax_spatial_eps = fig.add_axes([0.25, 0.15, 0.5, 0.03])
    ax_velocity_eps = fig.add_axes([0.25, 0.10, 0.5, 0.03])
    ax_min_samples = fig.add_axes([0.25, 0.05, 0.5, 0.03])

    # Create slider widgets
    slider_vel_weight = Slider(
        ax=ax_vel_weight, label='Velocity Weight', valmin=0, valmax=20,
        valinit=config.dbscan.velocity_weight, valstep=0.1
    )
    slider_spatial_eps = Slider(
        ax=ax_spatial_eps, label='Spatial ε (m)', valmin=0.1, valmax=5,
        valinit=config.dbscan.spatial_epsilon, valstep=0.1
    )
    slider_velocity_eps = Slider(
        ax=ax_velocity_eps, label='Velocity ε (m/s)', valmin=0.1, valmax=5,
        valinit=config.dbscan.velocity_epsilon, valstep=0.1
    )
    slider_min_samples = Slider(
        ax=ax_min_samples, label='Min Samples', valmin=1, valmax=20,
        valinit=config.dbscan.min_samples, valstep=1
    )
    slider_frame = Slider(
        ax=ax_frame, label='Frame ID', valmin=0, valmax=rpc_replay.sim_length_steps,
        valinit=0, valstep=1
    )

    box_patches = []
    
    # --- Update Functions ---
    def update(val):
        """This function is called by any slider to reload data and redraw the plot."""
        idx = int(slider_frame.val)
        # rpc_index = frame_id

        # --- A. Data Loading and Processing ---
        # Only reload data if the frame has changed
        if idx != getattr(update, "last_frame_id", None):            
            # frame_rpc = rpc_replay[idx]
            # current_frame_data.update({'xs': frame_rpc.x, 'ys': frame_rpc.y, 'zs': frame_rpc.z, 'velocities': frame_rpc.velocities})
            update.last_frame_id = idx

        # --- B. Clustering and Analysis ---
        dbscan_config = {
            "spatial_eps": slider_spatial_eps.val,
            "velocity_eps": slider_velocity_eps.val,
            "min_samples": slider_min_samples.val,
            "velocity_weight": slider_vel_weight.val,
            "noise_velocity_threshold": config.dbscan.noise_velocity_threshold,
        }
        moving_centroids, processed_frame, valid_labels = hungarian_matching(dbscan_config, processing_factory, matcher, idx)

        for p in box_patches:
            p.remove()
        box_patches.clear()
        
        # 2. Draw new boxes
        # Assuming processed_frame.moving_centroids contains your list of RadarObjects
        if moving_centroids:
            pass # Bounding box drawing is removed

        # --- C. Ground Truth ---
        gt_frame = gt_replay.get_frame_data(idx)
        gt_scatter.set_offsets(
            gt_frame.other_vehicles[['y_relative', 'x_relative']].values
        )

        # --- D. Update Scatter Plots ---
        # Create a mask for points belonging to valid clusters
        valid_cluster_mask = np.isin(processed_frame.labels, valid_labels)

        # Noise points are now the original noise OR points from invalid clusters
        all_noise_mask = processed_frame.noise_mask | (~valid_cluster_mask & processed_frame.cluster_mask)

        noise_scatter.set_offsets(
            np.column_stack([
                -processed_frame.point_cloud[all_noise_mask, 1],
                processed_frame.point_cloud[all_noise_mask, 0]
            ])
        )

        # Plot only the points from valid clusters
        if np.any(valid_cluster_mask):
            cluster_scatter.set_offsets(
                np.column_stack([
                    -processed_frame.point_cloud[valid_cluster_mask, 1],
                    processed_frame.point_cloud[valid_cluster_mask, 0]
                ])
            )
            cluster_scatter.set_array(
                processed_frame.labels[valid_cluster_mask]
            )
        else:
            cluster_scatter.set_offsets(np.empty((0, 2)))

        frame_text.set_text(f'Frame ID: {idx}')
        stats_text.set_text(f'Valid Clusters: {len(valid_labels)}\nNoise Pts: {np.sum(all_noise_mask)}')
        fig.canvas.draw_idle()

    # Register the update function to be called on slider changes
    for s in [slider_frame, slider_vel_weight, slider_spatial_eps, slider_velocity_eps, slider_min_samples]:
        s.on_changed(update)

    # --- Initial Plot ---
    update(None)

    plt.show()


if __name__ == "__main__":
    main()
