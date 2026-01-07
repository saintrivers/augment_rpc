import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import argparse

from processing.visualization_data_provider import VisualizationDataProvider, MethodParams

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

def main():
    """Main function to run the hyperparameter visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize radar clustering with interactive hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default="config/base.yml", help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # --- Centralized Data Loading ---
    data_provider = VisualizationDataProvider(args.config)
    config = data_provider.config

    # --- Visualization ---
    fig = plt.figure(figsize=(12, 12))
    ax, cluster_scatter, noise_scatter, gt_scatter, ego_point, frame_text, stats_text = init_plot(fig, view_radius=80)

    # Adjust layout to make room for sliders
    fig.subplots_adjust(bottom=0.40)
    fig.suptitle('Radar Point Cloud Analysis', fontsize=16)

    # --- Interactive Sliders Setup ---
    # Define axes for sliders
    ax_frame = fig.add_axes([0.25, 0.30, 0.5, 0.03])
    ax_vel_weight = fig.add_axes([0.25, 0.25, 0.5, 0.03])
    ax_spatial_eps = fig.add_axes([0.25, 0.20, 0.5, 0.03])
    ax_velocity_eps = fig.add_axes([0.25, 0.15, 0.5, 0.03])
    ax_min_samples = fig.add_axes([0.25, 0.10, 0.5, 0.03])
    ax_noise_thresh = fig.add_axes([0.25, 0.05, 0.5, 0.03])

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
        ax=ax_frame, label='Frame ID', valmin=0, valmax=data_provider.rpc_replay.sim_length_steps,
        valinit=0, valstep=1
    )
    slider_noise_thresh = Slider(
        ax=ax_noise_thresh, label='Noise Vel Thresh', valmin=0.1, valmax=5,
        valinit=config.dbscan.noise_velocity_threshold, valstep=0.1
    )

    text_annotations = []
    
    # --- Update Functions ---
    def update(val):
        """This function is called by any slider to reload data and redraw the plot."""
        idx = int(slider_frame.val)

        # --- Get Processed Data ---
        params = MethodParams(
            vel_weight=slider_vel_weight.val,
            spatial_eps=slider_spatial_eps.val,
            velocity_eps=slider_velocity_eps.val,
            min_samples=slider_min_samples.val,
            noise_velocity_threshold=slider_noise_thresh.val
        )
        moving_centroids, processed_frame, valid_labels, gt_frame = data_provider.get_frame_data(
            idx, params
        )

        # Clear previous annotations
        for txt in text_annotations:
            txt.remove()
        text_annotations.clear()

        # --- C. Ground Truth ---
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
            # Create a mapping from the original DBSCAN label to the final tracked ID
            label_to_id_map = {label: obj.id for label, obj in zip(valid_labels, moving_centroids)}
            
            # Create an array of colors (based on track ID) for each point in the valid clusters
            point_ids = [label_to_id_map.get(l, -1) for l in processed_frame.labels[valid_cluster_mask]]

            cluster_scatter.set_offsets(
                np.column_stack([
                    -processed_frame.point_cloud[valid_cluster_mask, 1],
                    processed_frame.point_cloud[valid_cluster_mask, 0]
                ])
            )
            cluster_scatter.set_array(np.array(point_ids))

            # Add text annotations for each tracked object ID
            for centroid_obj in moving_centroids:
                # The plot's x-axis is the vehicle's -y, and the plot's y-axis is the vehicle's x.
                plot_x = -centroid_obj.centroid[1]
                plot_y = centroid_obj.centroid[0]
                
                txt = ax.text(plot_x, plot_y + 1, str(centroid_obj.id), color='black', fontsize=10, ha='center', va='bottom')
                text_annotations.append(txt)
        else:
            cluster_scatter.set_offsets(np.empty((0, 2)))
            
        frame_text.set_text(f'Frame ID: {idx}')
        stats_text.set_text(f'Valid Clusters: {len(valid_labels)}\nNoise Pts: {np.sum(all_noise_mask)}')
        fig.canvas.draw_idle()

    # Register the update function to be called on slider changes
    for s in [slider_frame, slider_vel_weight, slider_spatial_eps, slider_velocity_eps, slider_min_samples, slider_noise_thresh]:
        s.on_changed(update)

    # --- Initial Plot ---
    update(None)

    plt.show()


if __name__ == "__main__":
    main()
