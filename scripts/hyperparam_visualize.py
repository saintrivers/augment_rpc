import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from processing.clustering import RpcClustering
from processing.datareader import load_radar_data, load_imu_data, load_radar_config
from processing.radarproc import collect_transformed_rpc
from omegaconf import OmegaConf
from sklearn.cluster import DBSCAN
from processing.radarproc import RpcReplay


def cluster_points_dbscan(xs, ys, zs, velocities, epsilon, min_samples, velocity_weight):
    """
    Performs DBSCAN clustering on a 4D point cloud (x, y, z, velocity).

    Returns:
        DBSCAN: The fitted DBSCAN model instance.
    """
    # Create a 4D array for clustering: (x, y, z, weighted_velocity)
    point_cloud = np.column_stack((xs, ys, zs, np.multiply(velocities, velocity_weight)))

    model = DBSCAN(eps=epsilon, min_samples=min_samples)
    model.fit(point_cloud)
    return model


def init_plot(fig, view_radius):
    """Initializes a single 2D plot for visualizing clustered RPC data."""
    ax = fig.add_subplot(111)
    # Scatter for clustered points (colored by cluster ID)
    cluster_scatter = ax.scatter([], [], c=[], cmap='jet', s=15, label='Clustered Points')
    # Scatter for noise points (gray)
    noise_scatter = ax.scatter([], [], c='gray', s=5, alpha=0.5, label='Noise')
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
    
    return ax, cluster_scatter, noise_scatter, ego_point, frame_text, stats_text


def load_experiment_data(datadir: str, ego_id: int) -> RpcReplay:
    """
    Loads rpc data structure designed for easy fetching and indexing of recorded simulation data.
    
    :param str datadir: absolute or relative path to the dataset root
    :param int ego_id: id of the ego vehicle set by the carla simulator
    :return rpc_replay: RpcReplay object
    """
    rpc, frame_ids = load_radar_data(datadir)
    imu_df = load_imu_data(f"{datadir}/imu_data.csv", ego_id)
    # nearby_df, ego_traj, frame_ids = load_and_prepare_data(config.sim.datadir, config.sim.ego_id)
    sensor_transforms = load_radar_config(datadir)
    rpc_replay = RpcReplay(rpc, frame_ids, sensor_transforms, imu_df)
    return rpc_replay


def generate_clustering_mask(vel_weight, epsilon, min_samples, **current_frame_data):
    """
    Docstring for generate_clustering_mask

    """
    model = cluster_points_dbscan(
        epsilon=epsilon,
        min_samples=min_samples, 
        velocity_weight=vel_weight, 
        **current_frame_data
    )
    noise_label = -1
    point_cloud = np.column_stack((current_frame_data['xs'], current_frame_data['ys']))
    noise_mask = model.labels_ == noise_label
    cluster_mask = ~noise_mask


def main():
    """Main function to run the hyperparameter visualization."""
    conf_yaml = """
    sim:
        datadir: "/home/dayan/projects/carla/rpc_bsm_mapping/dataset/town3_4"
        ego_id: 51
    dbscan:
        velocity_weight: 5.0
        epsilon: 0.5
        min_samples: 2
        doppler_threshold: 0.5
    """
    config = OmegaConf.create(conf_yaml)

    # --- Data Loading ---
    rpc_replay = load_experiment_data(config.sim.datadir, config.sim.ego_id)

    # --- Visualization ---
    fig = plt.figure(figsize=(12, 12))
    ax, cluster_scatter, noise_scatter, ego_point, frame_text, stats_text = init_plot(fig, view_radius=80)

    # Adjust layout to make room for sliders
    fig.subplots_adjust(bottom=0.3)
    fig.suptitle('Radar Point Cloud Analysis', fontsize=16)

    # --- Interactive Sliders Setup ---
    # Define axes for sliders
    ax_frame = fig.add_axes([0.25, 0.2, 0.5, 0.03])
    ax_vel_weight = fig.add_axes([0.25, 0.15, 0.5, 0.03])
    ax_eps = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    ax_min_samples = fig.add_axes([0.25, 0.05, 0.5, 0.03])

    # Create slider widgets
    slider_vel_weight = Slider(
        ax=ax_vel_weight, label='Velocity Weight', valmin=0, valmax=20,
        valinit=config.dbscan.velocity_weight, valstep=0.1
    )
    slider_eps = Slider(
        ax=ax_eps, label='Epsilon (ε)', valmin=0.1, valmax=5,
        valinit=config.dbscan.epsilon, valstep=0.05
    )
    slider_min_samples = Slider(
        ax=ax_min_samples, label='Min Samples', valmin=1, valmax=20,
        valinit=config.dbscan.min_samples, valstep=1
    )
    slider_frame = Slider(
        ax=ax_frame, label='Frame ID', valmin=0, valmax=rpc_replay.sim_length_steps,
        valinit=0, valstep=1
    )

    # --- State for update function ---
    # These variables will hold the data for the currently selected frame
    current_frame_data = {
        'xs': [], 'ys': [], 'zs': [], 'velocities': []
    }

    # Create a mapping from frame_id to list index for quick lookups
    # frame_id_to_idx = {fid: i for i, fid in enumerate(frame_ids)}

    # --- Update Functions ---

    def update(val):
        """This function is called by any slider to reload data and redraw the plot."""
        idx = int(slider_frame.val)
        # rpc_index = frame_id

        # --- A. Data Loading and Processing ---
        # Only reload data if the frame has changed
        if idx != getattr(update, "last_frame_id", None):            
            frame_rpc = rpc_replay[idx]
            current_frame_data.update({'xs': frame_rpc.x, 'ys': frame_rpc.y, 'zs': frame_rpc.z, 'velocities': frame_rpc.velocities})
            update.last_frame_id = idx

        # --- B. Clustering ---
        clustering = RpcClustering(
            rpc_replay=rpc_replay,
            eps=slider_eps.val,
            min_samples=slider_min_samples.val,
            velocity_weight=slider_vel_weight.val
        )
        cluster = clustering[idx]

        # Plot noise points
        # 1 is y, 0 is x because I want the X (front facing) to point up
        noise_scatter.set_offsets(np.column_stack([-cluster.point_cloud[cluster.noise_mask, 1], cluster.point_cloud[cluster.noise_mask,0]])) 
        
        # Plot clustered points
        if np.any(cluster.cluster_mask):
            cluster_scatter.set_offsets(np.column_stack([-cluster.point_cloud[cluster.cluster_mask, 1], cluster.point_cloud[cluster.cluster_mask, 0]]))
            cluster_scatter.set_array(cluster.labels[cluster.cluster_mask])
        else:
            cluster_scatter.set_offsets(np.empty((0, 2)))

        frame_text.set_text(f'Frame ID: {idx}')
        num_clusters = len(set(cluster.labels)) - (1 if -1 in cluster.labels else 0)
        stats_text.set_text(f'Clusters: {num_clusters}\nNoise Pts: {np.sum(cluster.noise_mask)}')
        fig.canvas.draw_idle()

    # Register the update function to be called on slider changes
    for s in [slider_frame, slider_vel_weight, slider_eps, slider_min_samples]:
        s.on_changed(update)

    # --- Initial Plot ---
    update(None)

    plt.show()


if __name__ == "__main__":
    main()
